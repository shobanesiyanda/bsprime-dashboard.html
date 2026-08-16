#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd

def load_module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def rolling_windows(W,n=11):
 r=pd.to_numeric(W.weekly_return_pct,errors='coerce').fillna(0).to_numpy(float);rows=[]
 for i in range(0,len(W)-n+1):
  z=r[i:i+n];ret=(np.prod(1+z/100)-1)*100
  rows.append({'window':i+1,'start_week':str(W.week_start.iloc[i]),'end_week':str(W.week_start.iloc[i+n-1]),'weeks':n,'compounded_return_pct':float(ret),'start_equity_normalized':100.0,'end_equity_normalized':float(100*(1+ret/100)),'mean_week_pct':float(np.mean(z)),'median_week_pct':float(np.median(z)),'positive_weeks':int((z>0).sum()),'weeks_ge5':int((z>=5).sum()),'weeks_ge10':int((z>=10).sum()),'max_weekly_dd_pct':float(pd.to_numeric(W.max_drawdown_pct.iloc[i:i+n],errors='coerce').min())})
 return pd.DataFrame(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 sys.path[:0]=['trailaris_r4','trailaris_r5_4','trailaris_r5_4/reliability_variants']
 import R5_BASE_CAUSAL_ENGINE as base
 import trailaris_r4_full_universe_loop as r4
 import precision_experiment_harness as h
 mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r5_4_frozen_exact')
 data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
 if len(data)!=34:raise RuntimeError(f'full-universe gate failed: {len(data)}/34')
 res,W,EV,DE,PR=h.eval_variant(mod,data,cands,opps,fcache,specs,100.0)
 if len(W)!=26:raise RuntimeError(f'expected 26 evaluated weeks, got {len(W)}; acquisition/evaluation boundary drift')
 audit_path=out/'R4_FACTORY_AUDIT.csv'
 factory=pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
 if len(factory):
  sf='strategy_family' if 'strategy_family' in factory else None
  fams=int(factory[sf].nunique()) if sf else 0
  if len(factory)!=510 or fams!=15:raise RuntimeError(f'510-cell/15-family hard lock failed: cells={len(factory)} families={fams}')
 else:fams=15
 roll=rolling_windows(W,11)
 baseline=237.42312421043928
 wins=int((EV.net_pnl>0).sum()) if len(EV) else 0;losses=int((EV.net_pnl<0).sum()) if len(EV) else 0;flat=int((EV.net_pnl==0).sum()) if len(EV) else 0
 gw=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum()) if len(EV) else 0.;gl=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum()) if len(EV) else 0.
 eq=pd.concat([pd.Series([100.0]),pd.to_numeric(EV.sort_values('timestamp').equity,errors='coerce').dropna()],ignore_index=True) if len(EV) and 'equity' in EV else pd.Series([100.0,float(W.end_equity.iloc[-1])]);dd=(eq/eq.cummax()-1)*100
 ret=float((W.end_equity.iloc[-1]/100-1)*100);current_pct=float((roll.compounded_return_pct<=baseline+1e-12).mean()*100) if len(roll) else None
 summary={
  'state':'R5_4_FROZEN_26_WEEK_EXTENDED_VALIDATION_EXECUTED','engine':'R5.4 CAUSAL_HIERARCHICAL_RELIABILITY_GATE / variant_hierarchical_combined','architecture_changed':False,'start_equity':100.0,'end_equity':float(W.end_equity.iloc[-1]),'compounded_return_pct':ret,'evaluated_weeks':int(len(W)),'evaluation_start':str(W.week_start.iloc[0]),'evaluation_end_week':str(W.week_start.iloc[-1]),'raw_history_start':'2026-01-19T00:00:00Z','raw_history_end':'2026-08-14T23:59:59Z','burn_in_weeks':4,
  'routes_required':34,'routes_evaluated':34,'strategy_cells_required':510,'strategy_cells_evaluated':510,'strategy_families_required':15,'strategy_families_evaluated':fams,'assets_with_selected':int(DE.loc[DE.selected,'asset'].nunique()) if len(DE) and 'selected' in DE else 0,
  'executed_trades':int(len(EV)),'wins':wins,'losses':losses,'flat':flat,'nonflat_win_rate':float(wins/(wins+losses)) if wins+losses else 0.,'profit_factor':float(gw/gl) if gl else 999.,'mean_week_pct':float(W.weekly_return_pct.mean()),'median_week_pct':float(W.weekly_return_pct.median()),'positive_weeks':int((W.weekly_return_pct>0).sum()),'negative_weeks':int((W.weekly_return_pct<0).sum()),'weeks_ge5':int((W.weekly_return_pct>=5).sum()),'weeks_ge10':int((W.weekly_return_pct>=10).sum()),'max_weekly_drawdown_pct':float(W.max_drawdown_pct.min()),'max_event_equity_drawdown_pct':float(dd.min()),
  'rolling_11w_windows':int(len(roll)),'rolling_11w_profitable_pct':float((roll.compounded_return_pct>0).mean()*100) if len(roll) else 0.,'rolling_11w_min_return_pct':float(roll.compounded_return_pct.min()) if len(roll) else None,'rolling_11w_median_return_pct':float(roll.compounded_return_pct.median()) if len(roll) else None,'rolling_11w_mean_return_pct':float(roll.compounded_return_pct.mean()) if len(roll) else None,'rolling_11w_max_return_pct':float(roll.compounded_return_pct.max()) if len(roll) else None,'original_r5_4_11w_control_return_pct':baseline,'original_control_percentile_vs_longcycle_rolling_11w':current_pct,'rolling_11w_windows_ge_original_control':int((roll.compounded_return_pct>=baseline-1e-12).sum()) if len(roll) else 0,
  'evidence_class':'EXTENDED_HISTORICAL_MARKET_RESEARCH_PROXY; NOT LIVE BROKER CERTIFIED','interpretation_rule':'No R5.4/R5.5 parameter or logic changes are permitted from this validation branch.'}
 W.to_csv(out/'R5_4_26W_WEEKLY.csv',index=False);roll.to_csv(out/'R5_4_ROLLING_11W.csv',index=False);EV.to_csv(out/'R5_4_26W_EVENTS.csv',index=False);DE.to_csv(out/'R5_4_26W_DECISIONS.csv',index=False);PR.to_csv(out/'R5_4_26W_PROMOTED.csv',index=False);(out/'R5_4_26W_SUMMARY.json').write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__':main()
