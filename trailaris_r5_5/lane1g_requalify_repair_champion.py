#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import lane1_reliability_gated_repair as core
from lane1b_full_replay import ManagedVariant

R54_END=337.4231242104393
EPS=1e-9
FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
RULE={
 'behavior':'close_a0.50_r0.05_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.05,'bars':1,
 'gate_metric':'reliability_asset_strategy_win','gate_direction':'le','gate_threshold':0.3265411458233644
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane1g_ctl');ctl,CW,EV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>EPS or len(EV)!=1102:raise RuntimeError('R5.4 control reproduction failed')
    z=EV.copy();z['eval_week']=pd.to_datetime(z.eval_week,utc=True);z['decision_time']=pd.to_datetime(z.decision_time,utc=True);z['entry_time']=pd.to_datetime(z.entry_time,utc=True);z['exit_time']=pd.to_datetime(z.exit_time,utc=True)
    for c in ['net_pnl','net_r','risk_cash','cost_r','entry_price','stop_distance','direction',RULE['gate_metric']]:z[c]=pd.to_numeric(z[c],errors='coerce')
    weeks=sorted(z.eval_week.unique());cut=weeks[-3];disc=(z.eval_week<cut).to_numpy();valid=((z.eval_week>=cut)&(z.decision_time<FRESH0)).to_numpy();fresh=(z.decision_time>=FRESH0).to_numpy()
    paths={i:core.path_for(fcache.get(r.asset),r) for i,r in z.iterrows()};vals=[];hit=[]
    for i,r in z.iterrows():rr,h=core.close_rescue(r,paths[i],RULE['activate_mfe_r'],RULE['parameter_r'],RULE['bars']);vals.append(rr);hit.append(h)
    vals=np.asarray(vals,float);hit=np.asarray(hit,bool);m=pd.to_numeric(z[RULE['gate_metric']],errors='coerce').to_numpy(float);gate=hit & np.isfinite(m) & (m<=RULE['gate_threshold']+1e-12);base_r=z.net_r.to_numpy(float);cf=np.where(gate,vals,base_r)
    ds=core.seg_stats(z.loc[disc],cf[disc]);vs=core.seg_stats(z.loc[valid],cf[valid])
    clean_discovery=ds['delta_usd']>0 and ds['flat_to_loss']==0 and ds['winner_to_loss']==0 and (ds['losses_improved']>0 or ds['flat_to_profit']>0)
    clean_validation=vs['delta_usd']>=0 and vs['flat_to_loss']==0 and vs['winner_to_loss']==0 and (vs['losses_improved']>0 or vs['flat_to_profit']>0)
    clean_requalified=bool(clean_discovery and clean_validation)
    status={'state':'R5_5_LANE1G_REPAIR_CHAMPION_CLEAN_REQUALIFICATION_COMPLETE','evidence_class':'CLEAN_REQUALIFICATION_PLUS_END_TO_END_CAUSAL_FULL_REPLAY','rule':RULE,'partition':{'discovery_events':int(disc.sum()),'validation_pre_aug13_events':int(valid.sum()),'aug13_14_quarantined_events':int(fresh.sum()),'fresh_outcomes_used_for_requalification':False},'selection_frozen_requalification':{'discovery':ds,'validation_pre_aug13':vs,'clean_discovery_pass':clean_discovery,'clean_validation_pass':clean_validation,'clean_requalified':clean_requalified},'r5_4_control':ctl,'approved_scope':scope}
    if clean_requalified:
        cand,VW,VEV,VDE,VPR=eval_variant(ManagedVariant(RULE,base,promote),data,cands,opps,fcache,specs,100.0)
        managed=int(VEV.get('management_rule',pd.Series(index=VEV.index,dtype=object)).notna().sum()) if len(VEV) else 0
        verification_delta=float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct'])
        repair=(int(cand['losses'])<=int(ctl['losses']) and int(cand['flat'])<=int(ctl['flat']) and (int(cand['losses'])<int(ctl['losses']) or int(cand['flat'])<int(ctl['flat'])))
        scope_ok=len(data)==34 and scope['approved_routes']==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510
        gates={'end_equity_not_regressed':float(cand['end_equity'])>=float(ctl['end_equity'])-EPS,'aug13_14_verification_not_regressed':verification_delta>=-EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-EPS,'loss_flat_repair':repair,'nonflat_win_rate_not_worse':float(cand['nonflat_win_rate'])>=float(ctl['nonflat_win_rate'])-1e-12,'selected_asset_coverage_not_worse':int(cand['assets_with_selected'])>=int(ctl['assets_with_selected']),'full_universe_34x510':scope_ok}
        pre=all(gates.values());status.update(candidate=cand,managed_executions=managed,delta_end_equity=float(cand['end_equity'])-float(ctl['end_equity']),aug13_14_verification_delta_pct=verification_delta,gates=gates,pre_forward_candidate=bool(pre),promotion_candidate=False,promotion_blockers=(['NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED'] if pre else ['PRE_FORWARD_NON_REGRESSION_GATE_FAILED']))
        VW.to_csv(out/'R5_5_LANE1G_WEEKLY.csv',index=False)
    else:
        status.update(pre_forward_candidate=False,promotion_candidate=False,promotion_blockers=['CLEAN_PRE_AUG13_REQUALIFICATION_FAILED'])
    status['governance']='The rule is not re-selected using Aug13-14. Its fixed threshold and behavior are re-evaluated on first8 discovery plus later pre-Aug13 validation only. Full replay is allowed only if both clean segments pass. A full-replay pass is pre-forward evidence only; final promotion requires a new unseen forward sample and certified broker execution.'
    (out/'R5_5_LANE1G_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
