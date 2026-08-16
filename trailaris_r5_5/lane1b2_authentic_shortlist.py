#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant

R54_END=337.4231242104393
EPS=1e-9

def as_bool(s):
    if getattr(s,'dtype',None)==bool:return s
    return s.astype(str).str.lower().isin(['true','1','yes'])

def choose_rules(df,n,failed_rule=None):
    d=df.copy()
    d=d[as_bool(d['joint_pass'])].copy()
    for c in ['all_flat_to_loss','all_winner_to_loss','all_winners_harmed','all_loss_to_nonloss','all_losses_improved','all_flat_to_profit','hold_delta_usd','all_delta_usd','applications']:
        if c in d:d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
    d=d[(d.all_flat_to_loss==0)&(d.all_winner_to_loss==0)].copy()
    if failed_rule:
        m=(d.behavior.astype(str)==str(failed_rule.get('behavior')))&(d.gate_metric.astype(str)==str(failed_rule.get('gate_metric')))&(d.gate_direction.astype(str)==str(failed_rule.get('gate_direction')))&(np.isclose(pd.to_numeric(d.gate_threshold,errors='coerce'),float(failed_rule.get('gate_threshold')),equal_nan=False))
        d=d[~m]
    d['safe_tier']=(d.all_winners_harmed==0).astype(int)
    d['auth_priority']=(6*d.all_loss_to_nonloss+2*d.all_losses_improved+1.25*d.all_flat_to_profit+2*np.maximum(d.hold_delta_usd,0)+np.maximum(d.all_delta_usd,0)-2*d.all_winners_harmed-.015*d.applications)
    # Do not spend the authentic replay budget on near-identical threshold variants.
    d=d.sort_values(['safe_tier','auth_priority','hold_delta_usd','all_delta_usd'],ascending=[False,False,False,False])
    d=d.drop_duplicates(['behavior','gate_metric','gate_direction'],keep='first')
    safe=d[d.safe_tier==1]
    picked=safe.head(n)
    if len(picked)<n:picked=pd.concat([picked,d[~d.index.isin(picked.index)].head(n-len(picked))])
    return picked.head(n).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--lane1a2-csv',required=True);ap.add_argument('--lane1a2-status',required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--max-rules',type=int,default=12);a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'))
    approved=lock['scope']
    src=json.load(open(a.lane1a2_status));failed=src.get('best_joint_rule')
    d=pd.read_csv(a.lane1a2_csv);rules=choose_rules(d,a.max_rules,failed)
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    if len(data)!=int(approved['approved_routes']):raise RuntimeError(f"route universe regressed {len(data)}/{approved['approved_routes']}")
    control_mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane1b2_control')
    ctl,CW,CEV,CDE,CPR=eval_variant(control_mod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>EPS:raise RuntimeError('R5.4 control reproduction failed')
    rows=[];weekly={}
    for i,r in rules.iterrows():
        rule=r.to_dict();variant=ManagedVariant(rule,base,promote)
        cand,VW,VEV,VDE,VPR=eval_variant(variant,data,cands,opps,fcache,specs,100.0)
        managed=int(VEV.get('management_rule',pd.Series(index=VEV.index,dtype=object)).notna().sum()) if len(VEV) else 0
        end_ok=float(cand['end_equity'])>=float(ctl['end_equity'])-EPS
        fresh_delta=float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct'])
        fresh_improved=fresh_delta>EPS
        dd_ok=float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-EPS
        repair=(int(cand['losses'])<=int(ctl['losses']) and int(cand['flat'])<=int(ctl['flat']) and (int(cand['losses'])<int(ctl['losses']) or int(cand['flat'])<int(ctl['flat'])))
        scope_ok=(int(approved['approved_routes'])==34 and int(approved['strategy_families'])==15 and int(approved['route_strategy_cells'])==510 and len(data)==34)
        promote_ok=bool(end_ok and fresh_improved and dd_ok and repair and scope_ok)
        rec={
          'rule_index':int(i),'behavior':rule.get('behavior'),'kind':rule.get('kind'),'gate_metric':rule.get('gate_metric'),'gate_direction':rule.get('gate_direction'),'gate_threshold':rule.get('gate_threshold'),'discovery_applications':rule.get('applications'),'discovery_all_delta_usd':rule.get('all_delta_usd'),'discovery_hold_delta_usd':rule.get('hold_delta_usd'),'discovery_loss_to_nonloss':rule.get('all_loss_to_nonloss'),'discovery_flat_to_profit':rule.get('all_flat_to_profit'),'discovery_winners_harmed':rule.get('all_winners_harmed'),
          'managed_executions':managed,'end_equity':float(cand['end_equity']),'delta_end_equity':float(cand['end_equity'])-float(ctl['end_equity']),'compounded_return_pct':float(cand['compounded_return_pct']),'executed_trades':int(cand['executed_trades']),'wins':int(cand['wins']),'losses':int(cand['losses']),'flat':int(cand['flat']),'nonflat_win_rate':float(cand['nonflat_win_rate']),'max_weekly_dd_pct':float(cand['max_weekly_dd_pct']),'fresh_return_pct':float(cand['fresh_return_pct']),'fresh_delta_pct':fresh_delta,'assets_with_selected':int(cand['assets_with_selected']),'end_equity_not_regressed':end_ok,'fresh_holdout_improved':fresh_improved,'drawdown_not_worse':dd_ok,'loss_flat_repair':repair,'full_universe_34x510':scope_ok,'promotion_candidate':promote_ok
        }
        rows.append(rec);weekly[i]=VW.copy()
        print(json.dumps(rec,default=str))
    R=pd.DataFrame(rows).sort_values(['promotion_candidate','end_equity','fresh_delta_pct','losses','flat'],ascending=[False,False,False,True,True]).reset_index(drop=True)
    R.to_csv(out/'R5_5_LANE1B2_AUTHENTIC_SHORTLIST.csv',index=False)
    CW.to_csv(out/'R5_5_LANE1B2_CONTROL_WEEKLY.csv',index=False)
    if len(R):
        best=R.iloc[0].to_dict();orig=int(best['rule_index']);weekly[orig].to_csv(out/'R5_5_LANE1B2_BEST_WEEKLY.csv',index=False)
    else:best={}
    status={'state':'R5_5_LANE1B2_AUTHENTIC_SHORTLIST_COMPLETE','evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY','r5_4_control':ctl,'approved_scope':approved,'generated_candidate_activity':{'routes_with_data':len(data),'families_emitting_candidates':int(cands.strategy_family.nunique()),'note':'activity count is not the approved-universe definition'},'joint_rules_available':int(as_bool(d.joint_pass).sum()),'authentic_rules_tested':int(len(R)),'promotion_candidates':int(R.promotion_candidate.sum()) if len(R) else 0,'best_authentic_rule':best,'next_lane':'BROKER_EXECUTION_DEGRADATION' if len(R) and bool(best.get('promotion_candidate')) else 'STRUCTURE_REGIME_ENTRY_THESIS_REPAIR','governance':'R5.4 remains immutable. Approved scope comes from the R5.4 lock; emitted candidate families are reported separately and cannot redefine universe scope. Every shortlisted rule is evaluated through the full causal portfolio replay, not frozen-selection arithmetic.'}
    (out/'R5_5_LANE1B2_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
