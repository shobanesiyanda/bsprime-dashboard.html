#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant

R54_END=337.4231242104393
EPS=1e-9

def as_bool(s):return s if getattr(s,'dtype',None)==bool else s.astype(str).str.lower().isin(['true','1','yes'])

def shortlist(df,n=12):
    d=df[as_bool(df.joint_pass)].copy()
    nums=['disc_delta_usd','valid_delta_usd','disc_flat_to_profit','valid_flat_to_profit','disc_loss_to_nonloss','valid_loss_to_nonloss','disc_losses_improved','valid_losses_improved','disc_winners_harmed','valid_winners_harmed','applications_pre_fresh']
    for c in nums:d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
    d=d[(d.disc_flat_to_loss==0)&(d.valid_flat_to_loss==0)&(d.disc_winner_to_loss==0)&(d.valid_winner_to_loss==0)].copy()
    d['clean_priority']=8*d.valid_loss_to_nonloss+4*d.valid_losses_improved+3*d.valid_flat_to_profit+3*d.disc_loss_to_nonloss+1.5*d.disc_flat_to_profit+3*np.maximum(d.valid_delta_usd,0)+np.maximum(d.disc_delta_usd,0)-3*d.valid_winners_harmed-1.5*d.disc_winners_harmed-.01*d.applications_pre_fresh
    d=d.sort_values(['clean_priority','valid_delta_usd','disc_delta_usd'],ascending=False)
    d=d.drop_duplicates(['behavior','gate_metric','gate_direction'],keep='first')
    # Preserve behavioral diversity so full replay can choose between close and lock families.
    a=d[d.kind.astype(str).eq('close')].head(max(1,n//2));b=d[d.kind.astype(str).eq('lock')].head(max(1,n//3))
    p=pd.concat([a,b,d[~d.index.isin(a.index.union(b.index))]],axis=0).drop_duplicates().head(n)
    return p.reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--lane1d-csv',required=True);ap.add_argument('--lane1d-status',required=True);ap.add_argument('--rank-index',type=int,required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    status=json.load(open(a.lane1d_status));d=pd.read_csv(a.lane1d_csv);rules=shortlist(d,12)
    if status.get('split',{}).get('fresh_outcomes_used_for_selection') is not False:raise RuntimeError('Lane1D holdout quarantine not proven')
    if not 0<=a.rank_index<len(rules):raise RuntimeError(f'rank {a.rank_index} outside shortlist {len(rules)}')
    rule=rules.iloc[a.rank_index].to_dict()
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_lane1f_ctl_{a.rank_index}');ctl,CW,CEV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>EPS or len(CEV)!=1102:raise RuntimeError('R5.4 control reproduction failed')
    cand,VW,VEV,VDE,VPR=eval_variant(ManagedVariant(rule,base,promote),data,cands,opps,fcache,specs,100.0)
    managed=int(VEV.get('management_rule',pd.Series(index=VEV.index,dtype=object)).notna().sum()) if len(VEV) else 0
    verification_delta=float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct'])
    repair=(int(cand['losses'])<=int(ctl['losses']) and int(cand['flat'])<=int(ctl['flat']) and (int(cand['losses'])<int(ctl['losses']) or int(cand['flat'])<int(ctl['flat'])))
    gates={'end_equity_not_regressed':float(cand['end_equity'])>=float(ctl['end_equity'])-EPS,'aug13_14_verification_not_regressed':verification_delta>=-EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-EPS,'loss_flat_repair':repair,'nonflat_win_rate_not_worse':float(cand['nonflat_win_rate'])>=float(ctl['nonflat_win_rate'])-1e-12,'selected_asset_coverage_not_worse':int(cand['assets_with_selected'])>=int(ctl['assets_with_selected']),'full_universe_34x510':len(data)==34 and scope['approved_routes']==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510}
    rec={'rank_index':a.rank_index,'rule':rule,'control':ctl,'candidate':cand,'managed_executions':managed,'delta_end_equity':float(cand['end_equity'])-float(ctl['end_equity']),'aug13_14_verification_delta_pct':verification_delta,'gates':gates,'pre_forward_candidate':bool(all(gates.values())),'promotion_candidate':False,'promotion_blockers':['NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED']}
    (out/f'rule_{a.rank_index:02d}.json').write_text(json.dumps(rec,indent=2,default=str));VW.to_csv(out/f'rule_{a.rank_index:02d}_weekly.csv',index=False);print(json.dumps(rec,indent=2,default=str))
if __name__=='__main__':main()
