#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe,extend_research_specs
from trailaris_50_asset.batch_consistent_drawdown import batch_consistent_event_drawdown,legacy_global_event_drawdown_pct
ORIGINAL34={'EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10'}
def pf(ev):
 if ev.empty:return 0.0
 x=pd.to_numeric(ev.net_pnl,errors='coerce').fillna(0);w=float(x[x>0].sum());l=float(-x[x<0].sum());return float(w/l) if l else (999.0 if w else 0.0)
def stats(ev):
 if ev.empty:return {'trades':0,'net_pnl_usd':0.0,'profit_factor':0.0,'nonflat_win_rate':0.0}
 x=pd.to_numeric(ev.net_pnl,errors='coerce').fillna(0);wins=int((x>0).sum());loss=int((x<0).sum());return {'trades':int(len(ev)),'net_pnl_usd':float(x.sum()),'profit_factor':pf(ev),'nonflat_win_rate':float(wins/(wins+loss)) if wins+loss else 0.0}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--universe-dir',type=Path,required=True);ap.add_argument('--promoted-root',type=Path,required=True);ap.add_argument('--assets',type=int,choices=[34,50],required=True);ap.add_argument('--specs',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
 cands=pd.read_csv(a.universe_dir/'ALL_CANDIDATES.csv.gz');cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
 if int(cands.asset.nunique())!=a.assets:raise RuntimeError(f'candidate asset integrity {cands.asset.nunique()}/{a.assets}')
 idx=pd.read_csv(a.universe_dir/'FEATURE_INDEX.csv');fcache={}
 for _,r in idx.iterrows():
  x=pd.read_csv(a.universe_dir/str(r.feature_file));x['timestamp']=pd.to_datetime(x.timestamp,utc=True);fcache[str(r.asset)]=x.sort_values('timestamp').reset_index(drop=True)
 if len(fcache)!=a.assets:raise RuntimeError(f'feature cache {len(fcache)}/{a.assets}')
 import trailaris_r4_full_universe_loop as r4
 meta=idx[['asset','discovery_provider_name']].drop_duplicates().copy();install_universe(r4,meta);base_specs=r4.load_specs(a.specs,'research-proxy');base_frame=base_specs.reset_index() if 'asset' not in base_specs.columns else base_specs.copy();specs=extend_research_specs(base_frame,meta).set_index('asset',drop=True)
 import R5_BASE_CAUSAL_ENGINE as base;base.r4.factor_vec=r4.factor_vec
 import quantitative_integrated_v2 as qv2
 files=sorted(a.promoted_root.rglob(f'QV2_{a.assets}_PROMOTED_SHARD_*.csv.gz'));parts=[]
 for f in files:
  try:z=pd.read_csv(f)
  except pd.errors.EmptyDataError:continue
  if len(z):parts.append(z)
 P=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
 if P.empty:raise RuntimeError('no promoted rows')
 for c in ('eval_week','decision_time','exit_time'):P[c]=pd.to_datetime(P[c],utc=True)
 pairs=P[['eval_week_index','eval_week']].drop_duplicates();
 if len(pairs)!=26 or sorted(pairs.eval_week_index.astype(int).tolist())!=list(range(26)):raise RuntimeError('promotion week integrity failed')
 weeks=[pd.Timestamp(x) for x in pairs.sort_values('eval_week_index').eval_week];P=P.sort_values(['eval_week','decision_time','campaign_id']).reset_index(drop=True);P.to_csv(a.outdir/f'QV2_{a.assets}_PROMOTED.csv.gz',index=False,compression='gzip')
 equity=100.0;Wrows=[];events=[];decisions=[];raw_end=max(pd.Timestamp(x.timestamp.iloc[-1]) for x in fcache.values())
 for wk in weeks:
  pr=P[P.eval_week.eq(wk)].copy();ev,de=qv2.replay_r5(pr,specs,equity,fcache) if len(pr) else (pd.DataFrame(),pd.DataFrame())
  if len(ev):equity=float(ev.equity.iloc[-1]);events.append(ev.assign(eval_week=wk));decisions.append(de.assign(eval_week=wk))
  prev=Wrows[-1]['end_equity'] if Wrows else 100.0;w=r4.weekly(ev,start=prev,feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),raw_end))
  if len(w):rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
  else:rec=dict(week_start=wk,start_equity=prev,end_equity=prev,weekly_return_pct=0.0,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.0,ge5=False,ge10=False);equity=prev
  Wrows.append(rec);print(json.dumps({'assets':a.assets,'week':str(wk),'end_balance':equity,'weekly_return_pct':float(rec['weekly_return_pct'])}))
 W=pd.DataFrame(Wrows);EV=pd.concat(events,ignore_index=True) if events else pd.DataFrame();DE=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame();W['week_start']=pd.to_datetime(W.week_start,utc=True)
 pnl=pd.to_numeric(EV.net_pnl,errors='coerce').fillna(0) if len(EV) else pd.Series(dtype=float);wins=int((pnl>0).sum());losses=int((pnl<0).sum());flats=int((pnl==0).sum());dd=batch_consistent_event_drawdown(W,EV,fcache,100.0)
 m={'currency':'USD','start_balance':100.0,'end_balance':float(W.end_equity.iloc[-1]),'compounded_return_pct':float((W.end_equity.iloc[-1]/100.0-1)*100),'evaluated_weeks':int(len(W)),'positive_weeks':int((W.weekly_return_pct>0).sum()),'negative_weeks':int((W.weekly_return_pct<0).sum()),'weeks_ge5':int((W.weekly_return_pct>=5).sum()),'weeks_ge10':int((W.weekly_return_pct>=10).sum()),'mean_week_pct':float(W.weekly_return_pct.mean()),'median_week_pct':float(W.weekly_return_pct.median()),'weekly_return_min_pct':float(W.weekly_return_pct.min()),'weekly_return_std_pct':float(W.weekly_return_pct.std()),'executed_trades':int(len(EV)),'wins':wins,'losses':losses,'flats':flats,'nonflat_win_rate':float(wins/(wins+losses)) if wins+losses else 0.0,'profit_factor':pf(EV),'max_weekly_drawdown_pct':float(pd.to_numeric(W.max_drawdown_pct,errors='coerce').min()),'batch_consistent_event_drawdown_pct':float(dd['max_event_equity_drawdown_pct']),'batch_consistent_worst_episode':dd['worst_episode'],'legacy_global_event_drawdown_pct':float(legacy_global_event_drawdown_pct(EV,100.0)),'promoted_opportunities':int(len(P)),'selected_opportunities':int(len(EV)),'opportunity_capture_pct':float(100*len(EV)/len(P)) if len(P) else 0.0,'candidate_assets':a.assets,'promoted_assets':int(P.asset.nunique()),'assets_with_selected':int(EV.asset.nunique()) if len(EV) else 0,'expectancy_r_per_trade':float(pd.to_numeric(EV.net_r,errors='coerce').mean()) if len(EV) else 0.0,'mean_net_pnl_usd_per_trade':float(pnl.mean()) if len(EV) else 0.0}
 core=EV[EV.asset.astype(str).isin(ORIGINAL34)].copy() if len(EV) else EV;add=EV[~EV.asset.astype(str).isin(ORIGINAL34)].copy() if len(EV) else EV
 status={'state':'QV2_INDEPENDENT_26W_REPLAY_COMPLETE','assets':a.assets,'raw_window':'2024-12-23_to_2025-07-21_exclusive','evaluated_window':'2025-01-20_to_2025-07-21_exclusive','strategy_blob':'120390ab52e80f51bafedc42389ceb7420a410f2','closed_balance_compounding':True,'quant_v2_modified':False,'asset_identity_modified':False,'candidate':m,'core34':stats(core),'expansion16':stats(add) if a.assets==50 else None,'drawdown_reporting':'BATCH_CONSISTENT_WITHIN_EACH_WEEK_LOCAL_REPLAY_BOOK; LEGACY_GLOBAL_METRIC_RETAINED_FOR_LINEAGE_ONLY','evidence_class':'INDEPENDENT_HISTORICAL_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE'}
 W.to_csv(a.outdir/f'QV2_{a.assets}_INDEPENDENT_WEEKLY.csv',index=False);EV.to_csv(a.outdir/f'QV2_{a.assets}_INDEPENDENT_EVENTS.csv.gz',index=False,compression='gzip');DE.to_csv(a.outdir/f'QV2_{a.assets}_INDEPENDENT_DECISIONS.csv.gz',index=False,compression='gzip');(a.outdir/f'QV2_{a.assets}_INDEPENDENT_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
