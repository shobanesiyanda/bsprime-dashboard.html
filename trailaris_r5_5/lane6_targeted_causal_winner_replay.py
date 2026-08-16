#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant,apply_rule
from lane5_pareto_management_stack import StackVariant,apply_stack
from lane4_market_state_entry_repair import enrich_market_state
from lane6c_winner_payoff_extension import extend_candidates

R54_END=337.4231242104393
B3_END=343.5447309255116
FRESH_FLOOR=7.261646557142409
DD_FLOOR=-4.637645574907678
FRESH_DD_FLOOR=-0.5196525640746019
LANE5_END=345.3896887826546
LANE5_LOSSES=228
LANE5_FLATS=421
LANE5_NFWR=0.673352435530086
EPS=1e-9
B3={'behavior':'close_a0.50_r0.20_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.20,'bars':1,'gate_metric':'reliability_win','gate_direction':'le','gate_threshold':0.2655132009803008}
LANE5_KEYS=['B3','B5','B8']

def num(x,d=np.nan):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def cond(z,f,d,th):
    if f not in z.columns:return pd.Series(False,index=z.index)
    s=pd.to_numeric(z[f],errors='coerce');return s.le(float(th)) if d=='le' else s.ge(float(th))

def rule_mask(z,r):
    if str(r['scope_type'])=='GLOBAL':m=pd.Series(True,index=z.index)
    elif str(r['scope_type'])=='ASSET_CLASS':m=z.asset_class.astype(str).eq(str(r['scope_value']))
    else:m=z.strategy_family.astype(str).eq(str(r['scope_value']))
    m &= cond(z,str(r['feature1']),str(r['dir1']),num(r['threshold1']))
    f2=str(r.get('feature2','') or '')
    if f2 and f2.lower()!='nan':m &= cond(z,f2,str(r['dir2']),num(r['threshold2']))
    return m.fillna(False)

def load_rule(path,rank):
    R=pd.read_csv(path)
    if 'clean_pass' in R.columns:R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])]
    R=R.reset_index(drop=True)
    if rank<0 or rank>=len(R):raise RuntimeError(f'rule rank {rank} unavailable; passing rows={len(R)}')
    return R.iloc[rank].to_dict(),len(R)

class CausalWinnerVariant:
    """Insert winner monetization after baseline weekly promotion/reliability scoring, before portfolio selection."""
    def __init__(self,mode,rule,base,promote,fcache,management,copies=1,quality_boost=.04,runner_frac=.25,target_r=5.,trail_gap=.8,max_extra=36):
        self.mode=str(mode).upper();self.rule=rule;self.base=base;self.promote=promote;self.fcache=fcache;self.management=management
        self.copies=int(copies);self.quality_boost=float(quality_boost);self.runner_frac=float(runner_frac);self.target_r=float(target_r);self.trail_gap=float(trail_gap);self.max_extra=int(max_extra)
        self.qualifying_promoted=0;self.extra_tranches_created=0;self.extensions_attempted=0;self.extension_delta_r=0.
    def promote_for_week(self,cands,week_start):
        # Exact R5.4 combined promotion first. Reliability uses only history exited before week_start.
        p,conf,stats=self.promote(cands,week_start,'combined')
        if p.empty:return p,conf,stats
        z=enrich_market_state(p,self.fcache)
        m=rule_mask(z,self.rule)
        z['winner_rule_qualifies']=m.to_numpy();self.qualifying_promoted+=int(m.sum())
        if self.mode=='PRIORITY':
            mm=m & (~z.get('is_addon',pd.Series(False,index=z.index)).astype(bool))
            q=pd.to_numeric(z.quality,errors='coerce').fillna(0)
            z.loc[mm,'quality']=np.minimum(.995,q.loc[mm]+self.quality_boost)
            # Baseline rank_score carries a .42 quality coefficient. Apply the equivalent priority delta
            # after causal eligibility is fixed, so priority cannot manufacture new promotion eligibility.
            z.loc[mm,'rank_score']=pd.to_numeric(z.loc[mm,'rank_score'],errors='coerce').fillna(0)+.42*self.quality_boost
        elif self.mode=='AMPLIFY':
            parents=z[~z.get('is_addon',pd.Series(False,index=z.index)).astype(bool)].copy()
            pqual=dict(zip(parents.campaign_id.astype(str),parents.winner_rule_qualifies.astype(bool)))
            pstr=dict(zip(parents.campaign_id.astype(str),parents.strategy.astype(str)))
            pfam=dict(zip(parents.campaign_id.astype(str),parents.strategy_family.astype(str)))
            addons=z[z.get('is_addon',pd.Series(False,index=z.index)).astype(bool)&z.strategy.astype(str).eq('WINNER_ONLY_ADDON')].copy()
            if len(addons) and 'parent_id' in addons.columns:
                addons=addons[addons.parent_id.astype(str).map(pqual).fillna(False)].copy()
            else:addons=addons.iloc[0:0].copy()
            parts=[z]
            for k in range(self.copies):
                if not len(addons):break
                q=addons.copy();q['campaign_id']=q.campaign_id.astype(str)+f'|WINAMP{k+1}'
                # This is still an add-on economically. Cap each extra tranche at the engine's 0.25% add-on risk.
                q['risk_fraction']=np.minimum(pd.to_numeric(q.risk_fraction,errors='coerce').fillna(.0025),.0025)
                q['quality']=np.minimum(.995,pd.to_numeric(q.quality,errors='coerce').fillna(0)+self.quality_boost)
                q['rank_score']=pd.to_numeric(q.rank_score,errors='coerce').fillna(0)+.42*self.quality_boost
                # Preserve the parent strategy/family for attribution while is_addon + parent_id keep add-on controls authoritative.
                q['strategy']=q.parent_id.astype(str).map(pstr).fillna(q.strategy.astype(str))
                q['strategy_family']=q.parent_id.astype(str).map(pfam).fillna(q.strategy_family.astype(str))
                parts.append(q);self.extra_tranches_created+=len(q)
            z=pd.concat(parts,ignore_index=True,sort=False)
        elif self.mode=='EXTEND':
            z['winner_extend_qualifies']=(m & (~z.get('is_addon',pd.Series(False,index=z.index)).astype(bool))).to_numpy()
        else:raise RuntimeError(f'unsupported mode {self.mode}')
        return z,conf,stats
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):
        if self.management=='LANE5':managed=apply_stack(cands,feature_cache,LANE5_KEYS)
        elif self.management=='B3':managed=apply_rule(cands,feature_cache,B3)
        else:managed=cands.copy()
        if self.mode=='EXTEND':
            managed,n,d=extend_candidates(managed,feature_cache,self.runner_frac,self.target_r,self.trail_gap,self.max_extra)
            self.extensions_attempted+=int(n);self.extension_delta_r+=float(d)
        return self.base.replay_r5(managed,specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--rule-rank',type=int,default=0);ap.add_argument('--mode',choices=['AMPLIFY','EXTEND','PRIORITY'],required=True);ap.add_argument('--copies',type=int,default=1);ap.add_argument('--quality-boost',type=float,default=.04);ap.add_argument('--runner-frac',type=float,default=.25);ap.add_argument('--target-r',type=float,default=5.);ap.add_argument('--trail-gap',type=float,default=.8);ap.add_argument('--max-extra-bars',type=int,default=36);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    rule,passing=load_rule(a.rules,a.rule_rank)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy');scope=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'))['scope']
    if len(data)!=34 or scope!={'approved_routes':34,'strategy_families':15,'route_strategy_cells':510}:raise RuntimeError(f'full-universe scope failure data={len(data)} scope={scope}')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_l6target_ctl');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.)
    if abs(float(ctl['end_equity'])-R54_END)>EPS:raise RuntimeError(f'R5.4 exact reproduction failed {ctl["end_equity"]}')
    b3,*_=eval_variant(ManagedVariant(B3,base,promote),data,cands,opps,fcache,specs,100.)
    if abs(float(b3['end_equity'])-B3_END)>1e-6:raise RuntimeError(f'Lane1B2 reproduction failed {b3["end_equity"]}')
    lane5,*_=eval_variant(StackVariant(LANE5_KEYS,base,promote),data,cands,opps,fcache,specs,100.)
    if abs(float(lane5['end_equity'])-LANE5_END)>1e-6 or int(lane5['losses'])!=LANE5_LOSSES or int(lane5['flat'])!=LANE5_FLATS:raise RuntimeError(f'certified Lane5 reproduction failed {lane5}')
    standalone_var=CausalWinnerVariant(a.mode,rule,base,promote,fcache,'B3',a.copies,a.quality_boost,a.runner_frac,a.target_r,a.trail_gap,a.max_extra_bars)
    standalone,SW,SEV,SDE,SPR=eval_variant(standalone_var,data,cands,opps,fcache,specs,100.)
    fusion_var=CausalWinnerVariant(a.mode,rule,base,promote,fcache,'LANE5',a.copies,a.quality_boost,a.runner_frac,a.target_r,a.trail_gap,a.max_extra_bars)
    fusion,FW,FEV,FDE,FPR=eval_variant(fusion_var,data,cands,opps,fcache,specs,100.)
    gates={'end_equity_above_lane5':float(fusion['end_equity'])>LANE5_END+EPS,'fresh_return_strictly_above':float(fusion['fresh_return_pct'])>FRESH_FLOOR+EPS,'max_weekly_dd_not_worse':float(fusion['max_weekly_dd_pct'])>=DD_FLOOR-EPS,'fresh_dd_not_worse':float(fusion['fresh_max_dd_pct'])>=FRESH_DD_FLOOR-EPS,'losses_at_or_below_228':int(fusion['losses'])<=LANE5_LOSSES,'flats_at_or_below_421':int(fusion['flat'])<=LANE5_FLATS,'nonflat_win_rate_at_least_lane5':float(fusion['nonflat_win_rate'])>=LANE5_NFWR-EPS,'assets_with_selected_at_least_33':int(fusion['assets_with_selected'])>=33,'full_universe_34x15x510':True}
    status={'state':'R5_5_LANE6_TARGETED_CAUSAL_WINNER_REPLAY_COMPLETE','evidence_class':'POST_RELIABILITY_PRE_SELECTION_WINNER_QUALIFICATION_PLUS_FULL_PORTFOLIO_REPLAY','mode':a.mode,'rule_file':str(a.rules),'rule_rank':a.rule_rank,'passing_rules_in_file':passing,'winner_rule':rule,'parameters':{'copies':a.copies,'quality_boost':a.quality_boost,'runner_frac':a.runner_frac,'target_r':a.target_r,'trail_gap':a.trail_gap,'max_extra_bars':a.max_extra_bars},'r5_4_control':ctl,'lane1b2_control':b3,'certified_lane5_control':lane5,'standalone_on_b3':standalone,'fusion_on_lane5':fusion,'standalone_delta_vs_b3':float(standalone['end_equity'])-B3_END,'fusion_delta_vs_lane5':float(fusion['end_equity'])-LANE5_END,'fresh_delta_vs_lane5':float(fusion['fresh_return_pct'])-FRESH_FLOOR,'runtime_counts':{'standalone_qualifying_promoted':standalone_var.qualifying_promoted,'standalone_extra_tranches_created':standalone_var.extra_tranches_created,'standalone_extensions_attempted':standalone_var.extensions_attempted,'standalone_extension_delta_r_pre_portfolio':standalone_var.extension_delta_r,'fusion_qualifying_promoted':fusion_var.qualifying_promoted,'fusion_extra_tranches_created':fusion_var.extra_tranches_created,'fusion_extensions_attempted':fusion_var.extensions_attempted,'fusion_extension_delta_r_pre_portfolio':fusion_var.extension_delta_r},'frontier_gates':gates,'frontier_pass':bool(all(gates.values())),'candidate_freeze':False,'scope':scope,'governance':'Winner rules were selected without Aug13-14 outcomes. At runtime they are evaluated only after the exact causal weekly promotion/reliability state exists and before portfolio selection. Amplification copies are created only after an already-promoted WINNER_ONLY_ADDON opportunity and remain add-ons capped at 0.25% risk. Priority changes ordering only after promotion eligibility is fixed. Extension activates only after a qualified unmodified base trade reaches the original 3R target. Full 34x15x510 chronology is recomputed. A frontier pass still requires forensic-completeness gates before freeze, then unseen forward and broker/server certification.'}
    (out/'R5_5_LANE6_TARGETED_STATUS.json').write_text(json.dumps(status,indent=2,default=str));SW.to_csv(out/'standalone_weekly.csv',index=False);FW.to_csv(out/'fusion_weekly.csv',index=False);print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
