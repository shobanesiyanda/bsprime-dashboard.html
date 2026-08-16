#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant,apply_rule
from winner_causal_features import causal_decision_features

R54_END=337.4231242104393
INCUMBENT_END=343.5447309255116
INCUMBENT_FRESH=7.261646557142409
INCUMBENT_DD=-4.637645574907678
CHALLENGE_END=345.3896887826546
CHALLENGE_NFWR=0.673352435530086
CHALLENGE_LOSSES=228
CHALLENGE_FLATS=421
EPS=1e-9
B3={'behavior':'close_a0.50_r0.20_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.20,'bars':1,'gate_metric':'reliability_win','gate_direction':'le','gate_threshold':0.2655132009803008}

def num(x,d=np.nan):
    try:v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def rcond(z,f,d,th):
    if f not in z.columns:return pd.Series(False,index=z.index)
    s=pd.to_numeric(z[f],errors='coerce');return s.le(th) if d=='le' else s.ge(th)

def rule_mask(z,r):
    if str(r['scope_type'])=='GLOBAL':m=pd.Series(True,index=z.index)
    elif str(r['scope_type'])=='ASSET_CLASS':m=z.asset_class.eq(str(r['scope_value']))
    else:m=z.strategy_family.eq(str(r['scope_value']))
    m &= rcond(z,str(r['feature1']),str(r['dir1']),num(r['threshold1']))
    f2=str(r.get('feature2','') or '')
    if f2 and f2.lower()!='nan':m &= rcond(z,f2,str(r['dir2']),num(r['threshold2']))
    return m.fillna(False)

class WinnerAmplifyVariant:
    def __init__(self,base,promote,copies,boost,history):self.base=base;self.promote=promote;self.copies=int(copies);self.boost=float(boost);self.history=history
    def promote_for_week(self,cands,week_start):
        # The immutable selector learns only from the original economic history.
        # Extra units are created only after an original WINNER_ONLY_ADDON has already survived
        # the causal weekly selector, so copies cannot manufacture their own config/trailing evidence.
        p,conf,stats=self.promote(self.history,week_start,'combined')
        if p.empty:return p,conf,stats
        elig=p[p.get('amp_parent_qualifies',pd.Series(False,index=p.index)).astype(bool)&p.strategy.astype(str).eq('WINNER_ONLY_ADDON')].copy()
        if not len(elig) or self.copies<=0:return p,conf,stats
        parts=[p]
        for k in range(self.copies):
            q=elig.copy();q['quality']=np.minimum(.995,pd.to_numeric(q.quality,errors='coerce').fillna(0)+self.boost)
            # Base rank_score carries all causal selector evidence. Quality contributes .42 in the
            # immutable rank formula, so reflect only the configured priority delta here.
            q['rank_score']=pd.to_numeric(q.rank_score,errors='coerce').fillna(0)+.42*self.boost
            q['campaign_id']=q.campaign_id.astype(str)+f'|WINAMP{k+1}'
            q['amplification_copy']=k+1
            parts.append(q)
        return pd.concat(parts,ignore_index=True,sort=False),conf,stats
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):return self.base.replay_r5(apply_rule(cands,feature_cache,B3),specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--rule-rank',type=int,required=True);ap.add_argument('--copies',type=int,required=True);ap.add_argument('--quality-boost',type=float,required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    R=pd.read_csv(a.rules);R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True)
    if a.rule_rank>=len(R):
        s={'state':'R5_5_LANE6B_NOT_RUN_RULE_RANK_UNAVAILABLE','rule_rank':a.rule_rank,'passing_rules':len(R)};(out/'result.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));return
    rule=R.iloc[a.rule_rank].to_dict()
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy');scope=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'))['scope']
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_lane6b_ctl_{a.rule_rank}_{a.copies}');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 control reproduction failed')
    inc,*_=eval_variant(ManagedVariant(B3,base,promote),data,cands,opps,fcache,specs,100.0)
    if abs(float(inc['end_equity'])-INCUMBENT_END)>1e-6:raise RuntimeError('Lane1B2 incumbent reproduction failed')
    z=causal_decision_features(cands,fcache);parents=z[~z.get('is_addon',False).astype(bool)].copy();pm=rule_mask(parents,rule);qual=dict(zip(parents.campaign_id.astype(str),pm.astype(bool)))
    z['amp_parent_qualifies']=z.parent_id.astype(str).map(qual).fillna(False) if 'parent_id' in z else False
    var=WinnerAmplifyVariant(base,promote,a.copies,a.quality_boost,z);cand,VW,VEV,VDE,VPR=eval_variant(var,data,z,opps,fcache,specs,100.0)
    amp_selected=int(VEV.campaign_id.astype(str).str.contains('WINAMP',regex=False).sum()) if len(VEV) and 'campaign_id' in VEV else 0;scope_ok=(len(data)==34 and scope=={'approved_routes':34,'strategy_families':15,'route_strategy_cells':510})
    alpha={'end_equity_beats_1b2':float(cand['end_equity'])>INCUMBENT_END+EPS,'fresh_not_worse':float(cand['fresh_return_pct'])>=INCUMBENT_FRESH-EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=INCUMBENT_DD-EPS,'asset_coverage_not_worse':int(cand['assets_with_selected'])>=33,'full_universe_34x15x510':scope_ok}
    frontier={'end_equity_beats_lane5':float(cand['end_equity'])>CHALLENGE_END+EPS,'fresh_improved':float(cand['fresh_return_pct'])>INCUMBENT_FRESH+EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=INCUMBENT_DD-EPS,'fresh_dd_not_worse':float(cand['fresh_max_dd_pct'])>=-0.5196525640746019-EPS,'losses_at_or_below_lane5':int(cand['losses'])<=CHALLENGE_LOSSES,'flats_at_or_below_lane5':int(cand['flat'])<=CHALLENGE_FLATS,'nonflat_win_rate_at_least_lane5':float(cand['nonflat_win_rate'])>=CHALLENGE_NFWR-EPS,'asset_coverage_not_worse':int(cand['assets_with_selected'])>=33,'full_universe_34x15x510':scope_ok}
    s={'state':'R5_5_LANE6B_WINNER_AMPLIFICATION_REPLAY_COMPLETE','evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY','rule_rank':a.rule_rank,'winner_rule':rule,'copies':a.copies,'quality_boost':a.quality_boost,'qualifying_parent_candidates':int(pm.sum()),'amplified_addons_selected':amp_selected,'r5_4_control':ctl,'lane1b2_incumbent':inc,'candidate':cand,'delta_vs_1b2':float(cand['end_equity'])-INCUMBENT_END,'delta_vs_lane5_frontier':float(cand['end_equity'])-CHALLENGE_END,'alpha_gates':alpha,'alpha_candidate':bool(all(alpha.values())),'frontier_gates':frontier,'frontier_candidate':bool(all(frontier.values())),'promotion_candidate':False,'scope':scope,'governance':'Winner qualification uses weekly-causal decision features. The immutable selector learns only from the original candidate history. Extra 0.25%-risk add-on copies are created only after the original continuation add-on has already passed weekly promotion, then compete only under unchanged portfolio risk, margin, factor and breaker controls.'}
    (out/'result.json').write_text(json.dumps(s,indent=2,default=str));VW.to_csv(out/'weekly.csv',index=False);print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
