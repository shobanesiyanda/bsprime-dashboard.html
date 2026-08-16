#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from lane1b_full_replay import ManagedVariant
from lane1b2_authentic_shortlist import choose_rules,as_bool,R54_END,EPS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--lane1a2-csv',required=True);ap.add_argument('--lane1a2-status',required=True);ap.add_argument('--rank-index',type=int,required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));approved=lock['scope']
    src=json.load(open(a.lane1a2_status));d=pd.read_csv(a.lane1a2_csv);rules=choose_rules(d,12,src.get('best_joint_rule'))
    if a.rank_index<0 or a.rank_index>=len(rules):raise RuntimeError(f'rank-index {a.rank_index} outside shortlist {len(rules)}')
    rule=rules.iloc[a.rank_index].to_dict()
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    if len(data)!=34:raise RuntimeError(f'route universe regressed {len(data)}/34')
    ctl_mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_parallel_ctl_{a.rank_index}')
    ctl,CW,CEV,CDE,CPR=eval_variant(ctl_mod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>EPS:raise RuntimeError('R5.4 control reproduction failed')
    cand,VW,VEV,VDE,VPR=eval_variant(ManagedVariant(rule,base,promote),data,cands,opps,fcache,specs,100.0)
    managed=int(VEV.get('management_rule',pd.Series(index=VEV.index,dtype=object)).notna().sum()) if len(VEV) else 0
    end_ok=float(cand['end_equity'])>=float(ctl['end_equity'])-EPS;fresh_delta=float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct']);fresh_ok=fresh_delta>EPS;dd_ok=float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-EPS
    repair=(int(cand['losses'])<=int(ctl['losses']) and int(cand['flat'])<=int(ctl['flat']) and (int(cand['losses'])<int(ctl['losses']) or int(cand['flat'])<int(ctl['flat'])))
    scope_ok=(approved['approved_routes']==34 and approved['strategy_families']==15 and approved['route_strategy_cells']==510 and len(data)==34)
    rec={'rank_index':a.rank_index,'rule':rule,'control':ctl,'candidate':cand,'managed_executions':managed,'delta_end_equity':float(cand['end_equity'])-float(ctl['end_equity']),'fresh_delta_pct':fresh_delta,'gates':{'end_equity_not_regressed':end_ok,'fresh_holdout_improved':fresh_ok,'drawdown_not_worse':dd_ok,'loss_flat_repair':repair,'full_universe_34x510':scope_ok},'promotion_candidate':bool(end_ok and fresh_ok and dd_ok and repair and scope_ok)}
    (out/f'rule_{a.rank_index:02d}.json').write_text(json.dumps(rec,indent=2,default=str));VW.to_csv(out/f'rule_{a.rank_index:02d}_weekly.csv',index=False);print(json.dumps(rec,indent=2,default=str))
if __name__=='__main__':main()
