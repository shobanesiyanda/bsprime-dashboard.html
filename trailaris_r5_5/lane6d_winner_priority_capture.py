#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant,apply_rule
from lane4_market_state_entry_repair import enrich_market_state

R54_END=337.4231242104393
INC_END=343.5447309255116
INC_FRESH=7.261646557142409
INC_DD=-4.637645574907678
INC_NFWR=0.6671428571428571
EPS=1e-9
B3={'behavior':'close_a0.50_r0.20_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.20,'bars':1,'gate_metric':'reliability_win','gate_direction':'le','gate_threshold':0.2655132009803008}

def num(x,d=np.nan):
    try:v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def rcond(z,f,d,th):
    s=pd.to_numeric(z[f],errors='coerce');return s.le(th) if d=='le' else s.ge(th)

def rule_mask(z,r):
    if str(r['scope_type'])=='GLOBAL':m=pd.Series(True,index=z.index)
    elif str(r['scope_type'])=='ASSET_CLASS':m=z.asset_class.eq(str(r['scope_value']))
    else:m=z.strategy_family.eq(str(r['scope_value']))
    m &= rcond(z,str(r['feature1']),str(r['dir1']),num(r['threshold1']))
    f2=str(r.get('feature2','') or '')
    if f2 and f2.lower()!='nan':m &= rcond(z,f2,str(r['dir2']),num(r['threshold2']))
    return m.fillna(False)

class WinnerPriorityVariant:
    def __init__(self,base,promote,boost):self.base=base;self.promote=promote;self.boost=float(boost)
    def promote_for_week(self,cands,week_start):
        z=cands.copy();m=z.get('winner_priority_qualifies',pd.Series(False,index=z.index)).astype(bool)&(~z.get('is_addon',pd.Series(False,index=z.index)).astype(bool));q=pd.to_numeric(z.quality,errors='coerce').fillna(0);z.loc[m,'quality']=np.minimum(.995,q.loc[m]+self.boost);return self.promote(z,week_start,'combined')
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):return self.base.replay_r5(apply_rule(cands,feature_cache,B3),specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--rule-rank',type=int,required=True);ap.add_argument('--quality-boost',type=float,required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    R=pd.read_csv(a.rules);R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True)
    if a.rule_rank>=len(R):
        s={'state':'R5_5_LANE6D_NOT_RUN_RULE_RANK_UNAVAILABLE','rule_rank':a.rule_rank,'passing_rules':len(R)};(out/'result.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));return
    rule=R.iloc[a.rule_rank].to_dict()
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_lane6d_ctl_{a.rule_rank}_{a.quality_boost}');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 control reproduction failed')
    inc,*_=eval_variant(ManagedVariant(B3,base,promote),data,cands,opps,fcache,specs,100.0)
    if abs(float(inc['end_equity'])-INC_END)>1e-6:raise RuntimeError('Lane1B2 incumbent reproduction failed')
    z=cands.copy();ze=enrich_market_state(z,fcache);z['winner_priority_qualifies']=rule_mask(ze,rule).to_numpy();var=WinnerPriorityVariant(base,promote,a.quality_boost);cand,VW,VEV,VDE,VPR=eval_variant(var,data,z,opps,fcache,specs,100.0)
    qual_selected=int(VEV.get('winner_priority_qualifies',pd.Series(False,index=VEV.index)).astype(bool).sum()) if len(VEV) else 0
    approved=set(getattr(r4,'STRATEGY_FAMILIES',[]));actual=set(z.loc[~z.get('is_addon',False).astype(bool),'strategy_family'].astype(str).unique());scope_ok=len(data)==34 and len(approved)==15 and approved.issubset(actual)
    alpha={'end_equity_beats_1b2':float(cand['end_equity'])>INC_END+EPS,'fresh_not_worse':float(cand['fresh_return_pct'])>=INC_FRESH-EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=INC_DD-EPS,'asset_coverage_not_worse':int(cand['assets_with_selected'])>=33,'full_universe_34x15':scope_ok}
    frontier={**alpha,'fresh_improved':float(cand['fresh_return_pct'])>INC_FRESH+EPS,'losses_below_r54':int(cand['losses'])<=230,'flats_below_r54':int(cand['flat'])<=424,'nonflat_win_rate_at_least_1b2':float(cand['nonflat_win_rate'])>=INC_NFWR-EPS}
    s={'state':'R5_5_LANE6D_WINNER_PRIORITY_CAPTURE_COMPLETE','evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY','rule_rank':a.rule_rank,'winner_rule':rule,'quality_boost':a.quality_boost,'qualifying_candidates':int(z.winner_priority_qualifies.sum()),'selected_qualifying_executions':qual_selected,'r5_4_control':ctl,'lane1b2_incumbent':inc,'candidate':cand,'delta_vs_1b2':float(cand['end_equity'])-INC_END,'alpha_gates':alpha,'alpha_candidate':bool(all(alpha.values())),'frontier_gates':frontier,'frontier_candidate':bool(all(frontier.values())),'promotion_candidate':False,'promotion_blockers':['NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED'],'scope':{'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'governance':'The priority signal is selected only from first8 winner attribution and pre-Aug13 validation. It changes selector priority through quality only; it does not rewrite realized outcomes, expected-R labels, reliability history, initial risk sizing, risk ceilings, factor ceilings or portfolio breakers. Full 34x15 chronology determines whether additional winner-cohort capture actually improves the system.'}
    (out/'result.json').write_text(json.dumps(s,indent=2,default=str));VW.to_csv(out/'weekly.csv',index=False);print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
