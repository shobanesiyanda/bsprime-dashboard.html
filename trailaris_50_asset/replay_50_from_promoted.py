#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe, extend_research_specs
from trailaris_100_asset.select_replay_100 import metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--universe-dir',type=Path,required=True);ap.add_argument('--promoted-root',type=Path,required=True);ap.add_argument('--specs',type=Path,required=True);ap.add_argument('--balance-control',type=Path,required=True);ap.add_argument('--approved16',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    approved=[x.strip() for x in a.approved16.read_text().splitlines() if x.strip()]
    cands=pd.read_csv(a.universe_dir/'ALL_CANDIDATES.csv.gz');cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    if int(cands.asset.nunique())!=50:raise RuntimeError(f'candidate asset integrity {int(cands.asset.nunique())}/50')
    idx=pd.read_csv(a.universe_dir/'FEATURE_INDEX.csv');fcache={}
    for _,r in idx.iterrows():
        x=pd.read_csv(a.universe_dir/str(r.feature_file));x['timestamp']=pd.to_datetime(x.timestamp,utc=True);fcache[str(r.asset)]=x.sort_values('timestamp').reset_index(drop=True)
    if len(fcache)!=50:raise RuntimeError(f'feature cache integrity {len(fcache)}/50')
    import trailaris_r4_full_universe_loop as r4
    base_specs=r4.load_specs(a.specs,'research-proxy');base_specs_frame=base_specs.reset_index() if 'asset' not in base_specs.columns else base_specs.copy()
    meta=idx[['asset','discovery_provider_name']].drop_duplicates().copy();install_universe(r4,meta);specs=extend_research_specs(base_specs_frame,meta).set_index('asset',drop=True)
    import R5_BASE_CAUSAL_ENGINE as base;base.r4.factor_vec=r4.factor_vec
    import quantitative_integrated_v2 as qv2
    files=sorted(a.promoted_root.rglob('QV2_50_PROMOTED_SHARD_*.csv.gz'));parts=[]
    for f in files:
        try:z=pd.read_csv(f)
        except pd.errors.EmptyDataError:continue
        if len(z):parts.append(z)
    P=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    if P.empty:raise RuntimeError('no promoted rows from parallel promotion stage')
    for col in ['eval_week','decision_time','exit_time']:P[col]=pd.to_datetime(P[col],utc=True)
    pairs=P[['eval_week_index','eval_week']].drop_duplicates();
    if len(pairs)!=26 or sorted(pairs.eval_week_index.astype(int).tolist())!=list(range(26)):raise RuntimeError('parallel promotion week/index integrity failed')
    weeks=[pd.Timestamp(x) for x in pairs.sort_values('eval_week_index').eval_week.tolist()];P=P.sort_values(['eval_week','decision_time','campaign_id']).reset_index(drop=True);P.to_csv(a.outdir/'QV2_50_PROMOTED.csv.gz',index=False,compression='gzip')
    equity=100.;Wrows=[];events=[];decisions=[];raw_end=max(pd.Timestamp(x.timestamp.iloc[-1]) for x in fcache.values())
    for wk in weeks:
        pr=P.loc[P.eval_week.eq(wk)].copy();ev,de=qv2.replay_r5(pr,specs,equity,fcache) if len(pr) else (pd.DataFrame(),pd.DataFrame())
        if len(ev):equity=float(ev.equity.iloc[-1]);events.append(ev.assign(eval_week=wk));decisions.append(de.assign(eval_week=wk))
        prev=Wrows[-1]['end_equity'] if Wrows else 100.;w=r4.weekly(ev,start=prev,feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),raw_end))
        if len(w):rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
        else:rec=dict(week_start=wk,start_equity=prev,end_equity=prev,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False);equity=prev
        Wrows.append(rec);print(json.dumps({'week':str(wk),'end_balance':equity,'weekly_return_pct':float(rec['weekly_return_pct'])}))
    W=pd.DataFrame(Wrows);EV=pd.concat(events,ignore_index=True) if events else pd.DataFrame();DE=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame();m=metrics(W,EV,DE,cands,P,fcache)
    if len(EV):
        m['expectancy_r_per_trade']=float(pd.to_numeric(EV.net_r,errors='coerce').mean());m['mean_net_pnl_usd_per_trade']=float(pd.to_numeric(EV.net_pnl,errors='coerce').mean())
    else:m['expectancy_r_per_trade']=m['mean_net_pnl_usd_per_trade']=0.0
    m['promoted_opportunities']=int(len(P));m['selected_opportunities']=int(len(EV));m['opportunity_capture_pct']=float(100*len(EV)/len(P)) if len(P) else 0.;m['weekly_return_min_pct']=float(W.weekly_return_pct.min());m['weekly_return_std_pct']=float(W.weekly_return_pct.std());m['pct_weeks_ge5']=float(100*(W.weekly_return_pct>=5).mean());m['pct_weeks_ge10']=float(100*(W.weekly_return_pct>=10).mean())
    control=json.loads(a.balance_control.read_text());c34=control['realized_closed_balance_replay'];historic=control['preserved_reference']
    delta34={'ending_balance_usd':m['end_balance']-float(c34['end_equity']),'compounded_return_percentage_points':m['compounded_return_pct']-float(c34['compounded_return_pct']),'profit_factor':m['profit_factor']-float(c34['profit_factor']),'max_weekly_drawdown_percentage_points':m['max_weekly_drawdown_pct']-float(c34['max_weekly_drawdown_pct']),'max_event_drawdown_percentage_points':m['max_event_equity_drawdown_pct']-float(c34['max_event_equity_drawdown_pct']),'executed_trades':m['executed_trades']-int(c34['executed_trades'])}
    status={'state':'QV2_50_ASSET_BALANCE_REPLAY_COMPLETE','compute_topology':'PARALLEL_INDEPENDENT_WEEKLY_PROMOTION__SEQUENTIAL_PORTFOLIO_REPLAY','experimental_delta':'ASSET_UNIVERSE_34_TO_50_ONLY','approved_additions':approved,'founder_approved_orcl_replacement':'NVDA.US','strategy_blob':'120390ab52e80f51bafedc42389ceb7420a410f2','closed_balance_compounding':True,'position_compounding':'EXISTING_STRATEGY_AND_EXPOSURE_CONTROLS','historic_34_reference':historic,'balance_compounded_34_control':c34,'candidate_50':m,'delta_50_minus_34_balance_control':delta34,'automatic_baseline_replacement':False,'decision_semantics_changed_by_parallelization':False,'evidence_class':'26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE'}
    W.to_csv(a.outdir/'QV2_50_WEEKLY.csv',index=False);EV.to_csv(a.outdir/'QV2_50_EVENTS.csv.gz',index=False,compression='gzip');DE.to_csv(a.outdir/'QV2_50_DECISIONS.csv.gz',index=False,compression='gzip');(a.outdir/'QV2_50_REPLAY_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
