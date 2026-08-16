#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from lane4b_authentic_market_state_gate import MarketStateVariant

R54_END=337.4231242104393
EPS=1e-9

def as_bool(s):return s if getattr(s,'dtype',None)==bool else s.astype(str).str.lower().isin(['true','1','yes'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--rules-csv',required=True);ap.add_argument('--status',required=True);ap.add_argument('--rank-index',type=int,required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    src=json.load(open(a.status));d=pd.read_csv(a.rules_csv);p=d[as_bool(d.clean_pass)].copy().sort_values(['score','valid_selection_frozen_delta_usd','disc_selection_frozen_delta_usd'],ascending=False).reset_index(drop=True)
    if src.get('partition',{}).get('fresh_outcomes_used_for_selection') is not False:raise RuntimeError('Lane4A holdout quarantine not proven')
    if not 0<=a.rank_index<len(p):raise RuntimeError(f'rank {a.rank_index} outside clean passers {len(p)}')
    rule=p.iloc[a.rank_index].to_dict();rule['rule_id']='|'.join([str(rule.get('target')),str(rule.get('scope_type')),str(rule.get('scope_value')),str(rule.get('feature1')),str(rule.get('dir1')),str(rule.get('threshold1')),str(rule.get('feature2','')),str(rule.get('dir2','')),str(rule.get('threshold2',''))])
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_lane4c_ctl_{a.rank_index}');ctl,CW,CEV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>EPS or len(CEV)!=1102:raise RuntimeError('R5.4 control reproduction failed')
    mod=MarketStateVariant(rule,promote,base,fcache);cand,VW,VEV,VDE,VPR=eval_variant(mod,data,cands,opps,fcache,specs,100.0)
    vd=float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct'])
    repair=(int(cand['losses'])<=int(ctl['losses']) and int(cand['flat'])<=int(ctl['flat']) and (int(cand['losses'])<int(ctl['losses']) or int(cand['flat'])<int(ctl['flat'])))
    scope_ok=len(data)==34 and scope['approved_routes']==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510
    gates={'end_equity_not_regressed':float(cand['end_equity'])>=float(ctl['end_equity'])-EPS,'aug13_14_verification_not_regressed':vd>=-EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-EPS,'loss_flat_repair':repair,'nonflat_win_rate_not_worse':float(cand['nonflat_win_rate'])>=float(ctl['nonflat_win_rate'])-1e-12,'selected_asset_coverage_not_worse':int(cand['assets_with_selected'])>=int(ctl['assets_with_selected']),'full_universe_34x510':scope_ok}
    rec={'rank_index':a.rank_index,'rule':rule,'control':ctl,'candidate':cand,'excluded_promotions':int(mod.excluded),'delta_end_equity':float(cand['end_equity'])-float(ctl['end_equity']),'aug13_14_verification_delta_pct':vd,'gates':gates,'pre_forward_candidate':bool(all(gates.values())),'promotion_candidate':False,'promotion_blockers':['NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED']}
    (out/f'rule_{a.rank_index:02d}.json').write_text(json.dumps(rec,indent=2,default=str));VW.to_csv(out/f'rule_{a.rank_index:02d}_weekly.csv',index=False);print(json.dumps(rec,indent=2,default=str))
if __name__=='__main__':main()
