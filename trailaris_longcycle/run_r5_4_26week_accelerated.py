#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import trailaris_r4_full_universe_loop as r4
import variant_hierarchical_combined as mod

_G_CANDS=None

def _init(cands):
 global _G_CANDS
 _G_CANDS=cands

def _promote_week(wk):
 pr,conf,stats=mod.promote_for_week(_G_CANDS,pd.Timestamp(wk))
 return str(pd.Timestamp(wk)),pr,conf,stats

def rolling_windows(W,n=11):
 r=pd.to_numeric(W.weekly_return_pct,errors='coerce').fillna(0).to_numpy(float);rows=[]
 for i in range(len(W)-n+1):
  z=r[i:i+n];ret=(np.prod(1+z/100)-1)*100
  rows.append({'window':i+1,'start_week':str(W.week_start.iloc[i]),'end_week':str(W.week_start.iloc[i+n-1]),'weeks':n,'compounded_return_pct':float(ret),'start_equity_normalized':100.0,'end_equity_normalized':float(100*(1+ret/100)),'mean_week_pct':float(np.mean(z)),'median_week_pct':float(np.median(z)),'positive_weeks':int((z>0).sum()),'weeks_ge5':int((z>=5).sum()),'weeks_ge10':int((z>=10).sum()),'max_weekly_dd_pct':float(pd.to_numeric(W.max_drawdown_pct.iloc[i:i+n],errors='coerce').min())})
 return pd.DataFrame(rows)

def frame_equal_key(a,b):
 if len(a)!=len(b):return False
 if len(a)==0:return True
 cols=[c for c in ['campaign_id','decision_time','asset','strategy','promotion_tier','risk_fraction','rank_score'] if c in a.columns and c in b.columns]
 x=a[cols].copy().sort_values(cols[:2] if len(cols)>=2 else cols).reset_index(drop=True);y=b[cols].copy().sort_values(cols[:2] if len(cols)>=2 else cols).reset_index(drop=True)
 for c in cols:
  if pd.api.types.is_numeric_dtype(x[c]) or pd.api.types.is_numeric_dtype(y[c]):
   if not np.allclose(pd.to_numeric(x[c],errors='coerce').fillna(0),pd.to_numeric(y[c],errors='coerce').fillna(0),rtol=0,atol=1e-12):return False
  else:
   if not x[c].astype(str).equals(y[c].astype(str)):return False
 return True

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
 if len(data)!=34:raise RuntimeError(f'34-route gate failed: {len(data)}')
 t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max();first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC');weeks=[];wk=first
 while wk<=t1:weeks.append(wk);wk+=pd.Timedelta(days=7)
 if len(weeks)!=26:raise RuntimeError(f'expected 26 evaluated weeks, got {len(weeks)}')
 # Semantic guard: two representative weeks must be identical in the unmodified function before acceleration is accepted.
 guard_weeks=[weeks[min(3,len(weeks)-1)],weeks[-1]]
 guard_seq={str(w):mod.promote_for_week(cands,w)[0] for w in guard_weeks}
 ctx=mp.get_context('fork');workers=min(8,max(2,os.cpu_count() or 4));prom={};confs={};stats={}
 with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=_init,initargs=(cands,)) as ex:
  futs={ex.submit(_promote_week,w):w for w in weeks}
  for f in as_completed(futs):
   k,pr,cf,st=f.result();prom[k]=pr;confs[k]=cf;stats[k]=st
 for w in guard_weeks:
  k=str(w)
  if not frame_equal_key(guard_seq[k],prom[k]):raise RuntimeError(f'parallel semantic guard failed for {k}')
 equity=100.0;W=[];events=[];decisions=[];promoted_all=[]
 for wk in weeks:
  pr=prom[str(wk)]
  if len(pr):
   ev,de=mod.replay_r5(pr,specs,equity,fcache)
   if len(ev):equity=float(ev.equity.iloc[-1])
   events.append(ev.assign(eval_week=wk));decisions.append(de.assign(eval_week=wk));promoted_all.append(pr.assign(eval_week=wk))
  else:ev=pd.DataFrame();de=pd.DataFrame()
  w=r4.weekly(ev,start=(W[-1]['end_equity'] if W else 100.0),feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),t1))
  if len(w):rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
  else:
   st=W[-1]['end_equity'] if W else 100.;rec=dict(week_start=wk,start_equity=st,end_equity=st,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False);equity=st
  W.append(rec)
 W=pd.DataFrame(W);EV=pd.concat(events,ignore_index=True) if events else pd.DataFrame();DE=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame();PR=pd.concat(promoted_all,ignore_index=True) if promoted_all else pd.DataFrame();roll=rolling_windows(W,11)
 wins=int((EV.net_pnl>0).sum()) if len(EV) else 0;losses=int((EV.net_pnl<0).sum()) if len(EV) else 0;flat=int((EV.net_pnl==0).sum()) if len(EV) else 0;gw=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum()) if len(EV) else 0.;gl=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum()) if len(EV) else 0.;baseline=237.42312421043928
 eq=pd.concat([pd.Series([100.]),pd.to_numeric(EV.sort_values('timestamp').equity,errors='coerce').dropna()],ignore_index=True) if len(EV) else pd.Series([100.,float(W.end_equity.iloc[-1])]);dd=(eq/eq.cummax()-1)*100
 summary={'state':'R5_4_FROZEN_26_WEEK_ACCELERATED_EXACT_REPLAY_EXECUTED','parallel_semantic_guard':'PASS','architecture_changed':False,'routes_evaluated':34,'strategy_cells_required':510,'strategy_families_required':15,'evaluated_weeks':len(W),'start_equity':100.0,'end_equity':float(W.end_equity.iloc[-1]),'compounded_return_pct':float((W.end_equity.iloc[-1]/100-1)*100),'executed_trades':len(EV),'wins':wins,'losses':losses,'flat':flat,'nonflat_win_rate':float(wins/(wins+losses)) if wins+losses else 0.,'profit_factor':float(gw/gl) if gl else 999.,'mean_week_pct':float(W.weekly_return_pct.mean()),'median_week_pct':float(W.weekly_return_pct.median()),'positive_weeks':int((W.weekly_return_pct>0).sum()),'negative_weeks':int((W.weekly_return_pct<0).sum()),'max_weekly_drawdown_pct':float(W.max_drawdown_pct.min()),'max_event_equity_drawdown_pct':float(dd.min()),'rolling_11w_windows':len(roll),'rolling_11w_profitable_pct':float((roll.compounded_return_pct>0).mean()*100),'rolling_11w_min_return_pct':float(roll.compounded_return_pct.min()),'rolling_11w_median_return_pct':float(roll.compounded_return_pct.median()),'rolling_11w_mean_return_pct':float(roll.compounded_return_pct.mean()),'rolling_11w_max_return_pct':float(roll.compounded_return_pct.max()),'original_r5_4_11w_control_return_pct':baseline,'original_control_percentile_vs_longcycle_rolling_11w':float((roll.compounded_return_pct<=baseline+1e-12).mean()*100),'rolling_11w_windows_ge_original_control':int((roll.compounded_return_pct>=baseline-1e-12).sum()),'evidence_class':'EXTENDED_HISTORICAL_MARKET_RESEARCH_PROXY; REQUIRES ORIGINAL_FACTORY_LANE_CROSSCHECK'}
 W.to_csv(out/'R5_4_26W_WEEKLY.csv',index=False);roll.to_csv(out/'R5_4_ROLLING_11W.csv',index=False);EV.to_csv(out/'R5_4_26W_EVENTS.csv',index=False);DE.to_csv(out/'R5_4_26W_DECISIONS.csv',index=False);PR.to_csv(out/'R5_4_26W_PROMOTED.csv',index=False);(out/'R5_4_26W_SUMMARY.json').write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__':main()
