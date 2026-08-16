#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane5_pareto_management_stack import apply_stack
from lane6b_winner_amplification import rule_mask
from lane6c_winner_payoff_extension import extend_candidates
from lane4_market_state_entry_repair import enrich_market_state

R54_END=337.4231242104393
L5_END=345.3896887826546
FRESH_FLOOR=7.261646557142409
DD_FLOOR=-4.637645574907678
FRESH_DD_FLOOR=-0.5196525640746019
LOSS_MAX=228
FLAT_MAX=421
NFWR_MIN=0.673352435530086
STACK=['B3','B5','B8']
EPS=1e-9

def boolcol(z,name):
    return z[name].astype(bool) if name in z.columns else pd.Series(False,index=z.index)

class FusionVariant:
    def __init__(self,base,promote,mode,params):self.base=base;self.promote=promote;self.mode=mode;self.p=params;self.extended=0;self.delta_r=0.
    def promote_for_week(self,cands,week_start):
        z=cands.copy()
        if self.mode=='AMPLIFY':
            parts=[z];elig=z[boolcol(z,'amp_parent_qualifies') & z.strategy.astype(str).eq('WINNER_ONLY_ADDON')].copy()
            for k in range(int(self.p['copies'])):
                if not len(elig):break
                q=elig.copy();q['strategy']=q['amp_parent_strategy'];q['strategy_family']=q['amp_parent_family'];q['quality']=np.minimum(.995,pd.to_numeric(q.quality,errors='coerce').fillna(0)+float(self.p['quality_boost']));q['campaign_id']=q.campaign_id.astype(str)+f'|L7AMP{k+1}';parts.append(q)
            z=pd.concat(parts,ignore_index=True,sort=False)
        elif self.mode=='PRIORITY':
            m=boolcol(z,'winner_priority_qualifies') & ~boolcol(z,'is_addon');q=pd.to_numeric(z.quality,errors='coerce').fillna(0);z.loc[m,'quality']=np.minimum(.995,q.loc[m]+float(self.p['quality_boost']))
        return self.promote(z,week_start,'combined')
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):
        z=apply_stack(cands,feature_cache,STACK)
        if self.mode=='EXTEND':
            z,n,d=extend_candidates(z,feature_cache,float(self.p['runner_frac']),float(self.p['target_r']),float(self.p.get('trail_gap',.8)),int(self.p.get('max_extra_bars',36)));self.extended=n;self.delta_r=d
        return self.base.replay_r5(z,specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--mode',choices=['AMPLIFY','EXTEND','PRIORITY'],required=True);ap.add_argument('--rule-rank',type=int,default=0);ap.add_argument('--copies',type=int,default=1);ap.add_argument('--quality-boost',type=float,default=.04);ap.add_argument('--runner-frac',type=float,default=.25);ap.add_argument('--target-r',type=float,default=4.0);ap.add_argument('--trail-gap',type=float,default=.8);ap.add_argument('--max-extra-bars',type=int,default=36);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    R=pd.read_csv(a.rules);R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True);rule=R.iloc[a.rule_rank].to_dict()
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_l7_ctl');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 exact reproduction failed')
    baseline,*_=eval_variant(FusionVariant(base,promote,'PRIORITY',{'quality_boost':0}),data,cands,opps,fcache,specs,100.0)
    if abs(float(baseline['end_equity'])-L5_END)>1e-6:raise RuntimeError(f'Lane5 fusion baseline failed {baseline["end_equity"]}')
    z=cands.copy();ze=enrich_market_state(z,fcache);qual=rule_mask(ze,rule).to_numpy()
    if a.mode=='PRIORITY':z['winner_priority_qualifies']=qual
    elif a.mode=='EXTEND':z['winner_extend_qualifies']=qual
    else:
        parents=z[~boolcol(z,'is_addon')].copy();pe=enrich_market_state(parents,fcache);pm=rule_mask(pe,rule);qm=dict(zip(pe.campaign_id.astype(str),pm.astype(bool)));ps=dict(zip(parents.campaign_id.astype(str),parents.strategy.astype(str)));pf=dict(zip(parents.campaign_id.astype(str),parents.strategy_family.astype(str)))
        z['amp_parent_qualifies']=z.parent_id.astype(str).map(qm).fillna(False) if 'parent_id' in z else False;z['amp_parent_strategy']=z.parent_id.astype(str).map(ps).fillna(z.strategy.astype(str)) if 'parent_id' in z else z.strategy.astype(str);z['amp_parent_family']=z.parent_id.astype(str).map(pf).fillna(z.strategy_family.astype(str)) if 'parent_id' in z else z.strategy_family.astype(str)
    p={'copies':a.copies,'quality_boost':a.quality_boost,'runner_frac':a.runner_frac,'target_r':a.target_r,'trail_gap':a.trail_gap,'max_extra_bars':a.max_extra_bars}
    var=FusionVariant(base,promote,a.mode,p);cand,VW,VEV,VDE,VPR=eval_variant(var,data,z,opps,fcache,specs,100.0)
    scope_ok=(len(data)==34 and scope=={'approved_routes':34,'strategy_families':15,'route_strategy_cells':510})
    gates={'end_equity_above_345_3896888':float(cand['end_equity'])>L5_END+EPS,'fresh_return_above_7_2616466':float(cand['fresh_return_pct'])>FRESH_FLOOR+EPS,'max_weekly_dd_not_worse':float(cand['max_weekly_dd_pct'])>=DD_FLOOR-EPS,'fresh_dd_not_worse':float(cand['fresh_max_dd_pct'])>=FRESH_DD_FLOOR-EPS,'losses_max_228':int(cand['losses'])<=LOSS_MAX,'flats_max_421':int(cand['flat'])<=FLAT_MAX,'nonflat_wr_at_least_67_3352pct':float(cand['nonflat_win_rate'])>=NFWR_MIN-EPS,'selected_assets_at_least_33':int(cand['assets_with_selected'])>=33,'full_universe_34x15x510':scope_ok}
    s={'state':'R5_5_LANE7_WINNER_MANAGEMENT_FUSION_REPLAY_COMPLETE','evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY','mode':a.mode,'parameters':p,'winner_rule_rank':a.rule_rank,'winner_rule':rule,'lane5_baseline':baseline,'candidate':cand,'delta_vs_lane5':float(cand['end_equity'])-L5_END,'extended_candidates':var.extended,'extension_delta_r_pre_portfolio':var.delta_r,'gates':gates,'frontier_pass':bool(all(gates.values())),'promotion_candidate':False,'scope':scope,'governance':'Lane 7 starts from the provisional Lane 5 B3+B5+B8 management leader and adds exactly one pre-Aug13-validated winner monetization mechanism. The complete portfolio is replayed chronologically under unchanged risk, margin, factor and breaker controls. No fusion can pass without strict fresh improvement and the locked 34x15x510 scope.'}
    (out/'result.json').write_text(json.dumps(s,indent=2,default=str));VW.to_csv(out/'weekly.csv',index=False);print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
