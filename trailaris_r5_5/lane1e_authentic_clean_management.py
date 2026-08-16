#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

R54_END=337.4231242104393
EPS=1e-9

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--lane1d-status',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    src=json.load(open(a.lane1d_status));joint=int(src.get('joint_passing_rules',0))
    if joint<1:
        s={'state':'R5_5_LANE1E_NOT_RUN_NO_CLEAN_MANAGEMENT_RULE','joint_passing_rules':joint,'next_lane':'ENTRY_THESIS_AND_STRATEGY_REBUILD','governance':'No management rule is forced into authentic replay without discovery plus pre-fresh validation.'};(out/'R5_5_LANE1E_STATUS.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));return
    rule=src['best_joint_rule']
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    from lane1b_full_replay import ManagedVariant
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane1e_ctl')
    ctl,CW,CEV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>EPS or len(CEV)!=1102:raise RuntimeError('R5.4 control reproduction failed')
    cand,VW,VEV,VDE,VPR=eval_variant(ManagedVariant(rule,base,promote),data,cands,opps,fcache,specs,100.0)
    managed=int(VEV.get('management_rule',pd.Series(index=VEV.index,dtype=object)).notna().sum()) if len(VEV) else 0
    repair=(int(cand['losses'])<=int(ctl['losses']) and int(cand['flat'])<=int(ctl['flat']) and (int(cand['losses'])<int(ctl['losses']) or int(cand['flat'])<int(ctl['flat'])))
    gates={'end_equity_not_regressed':float(cand['end_equity'])>=float(ctl['end_equity'])-EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-EPS,'loss_flat_repair':repair,'nonflat_win_rate_not_worse':float(cand['nonflat_win_rate'])>=float(ctl['nonflat_win_rate'])-1e-12,'selected_asset_coverage_not_worse':int(cand['assets_with_selected'])>=int(ctl['assets_with_selected']),'full_universe_34x510':len(data)==34 and scope['approved_routes']==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510}
    pre_forward=all(gates.values())
    blockers=(['NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED'] if pre_forward else ['PRE_FORWARD_NON_REGRESSION_GATE_FAILED'])
    s={'state':'R5_5_LANE1E_AUTHENTIC_CLEAN_MANAGEMENT_COMPLETE','evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY','selected_clean_rule':rule,'r5_4_control':ctl,'candidate':cand,'managed_executions':managed,'delta_end_equity':float(cand['end_equity'])-float(ctl['end_equity']),'aug13_14_verification_delta_pct':float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct']),'gates':gates,'pre_forward_candidate':bool(pre_forward),'promotion_candidate':False,'promotion_blockers':blockers,'approved_scope':scope,'governance':'The selected rule was chosen without Aug13-14 outcomes. Aug13-14 is reported only as an already-inspected verification sample and cannot be used to tune the rule. A failed pre-forward non-regression gate is a hard rejection. A passing authentic replay is only a pre-forward R5.5 candidate until a new unseen forward period and certified route-specific broker degradation both pass.'}
    (out/'R5_5_LANE1E_STATUS.json').write_text(json.dumps(s,indent=2,default=str));VW.to_csv(out/'R5_5_LANE1E_WEEKLY.csv',index=False);print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
