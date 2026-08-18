#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ORIGINAL34={
'EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10'}
ADDITIONS={'COCOA.CMD','SUGAR.CMD','COFFEE.CMD','AAPL.US','SOYBEAN.CMD','NVDA.US','MSFT.US','USDCNH','V.US','AUDNZD','EURCHF','GBPNZD','FRA.IDX','TSLA.US','CHI.IDX','COPPER.CMD'}

def pf(g):
    w=float(g.loc[g.net_pnl>0,'net_pnl'].sum());l=float(-g.loc[g.net_pnl<0,'net_pnl'].sum())
    return float(w/l) if l else (999. if w else 0.)

def group_stats(g,key):
    rows=[]
    for k,z in g.groupby(key,dropna=False):
        nonflat=int((z.net_pnl!=0).sum());wins=int((z.net_pnl>0).sum())
        rows.append({key:str(k),'trades':int(len(z)),'net_pnl_usd':float(z.net_pnl.sum()),'profit_factor':pf(z),'nonflat_win_rate':float(wins/nonflat) if nonflat else 0.,'mean_net_r':float(pd.to_numeric(z.net_r,errors='coerce').mean())})
    return sorted(rows,key=lambda x:x['net_pnl_usd'],reverse=True)

def batch_consistent_event_dd(ev):
    if ev.empty:return 0.0,None
    x=ev.copy();x['timestamp']=pd.to_datetime(x.timestamp,utc=True);x['eval_week']=pd.to_datetime(x.eval_week,utc=True)
    rows=[]
    for _,g in x.groupby('eval_week',sort=True):
        # Engine closes every position due at t before new decisions at t. The last
        # event at a duplicated timestamp is the accounting-consistent post-batch state.
        h=g.sort_values('timestamp',kind='stable').groupby('timestamp',sort=True,as_index=False).tail(1)
        rows.append(h)
    y=pd.concat(rows,ignore_index=True)
    i=pd.to_numeric(y.drawdown,errors='coerce').idxmin();r=y.loc[i]
    return float(r.drawdown)*100.,{'eval_week':str(r.eval_week),'timestamp':str(r.timestamp),'asset':str(r.asset),'equity_usd':float(r.equity)}

def classify(a):
    if a in {'XAUUSD','XAGUSD','COPPER.CMD'}:return 'METAL'
    if a in {'BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD'}:return 'CRYPTO'
    if a in {'US100','US500','US30','GER40','UK100','JP225','FRA.IDX','CHI.IDX'}:return 'INDEX'
    if a in {'WTI','BRENT','NATGAS'}:return 'ENERGY'
    if a in {'COCOA.CMD','SUGAR.CMD','COFFEE.CMD','SOYBEAN.CMD'}:return 'COMMODITY'
    if a in {'V75','V50','V100','V25','V10'}:return 'SYNTHETIC'
    if a.endswith('.US'):return 'EQUITY'
    return 'FX'

def summarize_one(mode,d,control):
    st=json.loads((d/'QV2_50_REPLAY_STATUS.json').read_text());m=st['candidate_50']
    ev=pd.read_csv(d/'QV2_50_EVENTS.csv.gz');de=pd.read_csv(d/'QV2_50_DECISIONS.csv.gz');w=pd.read_csv(d/'QV2_50_WEEKLY.csv')
    ev['net_pnl']=pd.to_numeric(ev.net_pnl,errors='coerce');ev['net_r']=pd.to_numeric(ev.net_r,errors='coerce')
    ev['universe_role']=np.where(ev.asset.astype(str).isin(ADDITIONS),'EXPANSION16','CORE34');ev['asset_class']=ev.asset.astype(str).map(classify)
    corrected_dd,episode=batch_consistent_event_dd(ev);core=ev[ev.universe_role=='CORE34'];add=ev[ev.universe_role=='EXPANSION16']
    win=int((ev.net_pnl>0).sum());loss=int((ev.net_pnl<0).sum());flat=int((ev.net_pnl==0).sum());reasons=de.reason.astype(str).value_counts().to_dict();c34=control['realized_closed_balance_replay']
    return {
      'mode':mode,'quant_v2_modified':False,'promotion_estate_modified':False,'portfolio_overlay_only':mode!='CONTROL',
      'end_balance_usd':float(m['end_balance']),'return_pct':float(m['compounded_return_pct']),'positive_weeks':int(m['positive_weeks']),'weeks_ge5':int(m['weeks_ge5']),'weeks_ge10':int(m['weeks_ge10']),
      'trades':int(len(ev)),'wins':win,'losses':loss,'flats':flat,'profit_factor':pf(ev),'reported_event_dd_pct':float(m['max_event_equity_drawdown_pct']),'batch_consistent_event_dd_pct':corrected_dd,'batch_consistent_worst_episode':episode,'max_weekly_dd_pct':float(m['max_weekly_drawdown_pct']),
      'delta_vs_34':{'ending_balance_usd':float(m['end_balance'])-float(c34['end_equity']),'profit_factor':pf(ev)-float(c34['profit_factor']),'batch_consistent_event_dd_pct_points':corrected_dd-float(c34['max_event_equity_drawdown_pct']),'trades':int(len(ev))-int(c34['executed_trades'])},
      'core34':{'trades':int(len(core)),'net_pnl_usd':float(core.net_pnl.sum()),'profit_factor':pf(core),'share_net_event_pnl':float(core.net_pnl.sum()/ev.net_pnl.sum()) if ev.net_pnl.sum() else 0.},
      'expansion16':{'trades':int(len(add)),'net_pnl_usd':float(add.net_pnl.sum()),'profit_factor':pf(add),'share_net_event_pnl':float(add.net_pnl.sum()/ev.net_pnl.sum()) if ev.net_pnl.sum() else 0.,'selected_assets':int(add.asset.nunique())},
      'tier_attribution':group_stats(ev,'promotion_tier'),'family_attribution':group_stats(ev,'strategy_family'),'class_attribution':group_stats(ev,'asset_class'),'decision_reasons':{str(k):int(v) for k,v in reasons.items()},
      'weekly_return_min_pct':float(pd.to_numeric(w.weekly_return_pct,errors='coerce').min()),'weekly_return_std_pct':float(pd.to_numeric(w.weekly_return_pct,errors='coerce').std())
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--control',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();control=json.loads(a.control.read_text());modes=['CONTROL','FACTOR100','WEEK8','FACTOR100_WEEK8'];rows=[]
    for mode in modes:
        d=a.root/mode
        if (d/'QV2_50_REPLAY_STATUS.json').exists():rows.append(summarize_one(mode,d,control))
    by={r['mode']:r for r in rows};base=by.get('CONTROL')
    if base:
        for r in rows:r['delta_vs_control']={'end_balance_usd':r['end_balance_usd']-base['end_balance_usd'],'profit_factor':r['profit_factor']-base['profit_factor'],'batch_consistent_event_dd_pct_points':r['batch_consistent_event_dd_pct']-base['batch_consistent_event_dd_pct'],'trades':r['trades']-base['trades']}
    report={'state':'QV2_50_GEOMETRY_RESEARCH_COMPLETE','research_design':'PRE_SPECIFIED_PORTFOLIO_OVERLAYS_ON_EXACT_PRESERVED_50_PROMOTION_ESTATE','strategy_blob':'120390ab52e80f51bafedc42389ceb7420a410f2','quant_v2_modified':False,'asset_universe_modified':False,'automatic_baseline_replacement':False,'metric_correction':{'issue':'Global timestamp sorting of independently week-local event equity rows can interleave cross-week accounting states. Same-timestamp exits can also transiently mark another due-to-close position beyond its defined exit before the batch is finished.','corrected_measure':'Minimum engine drawdown after the final close event at each timestamp within each evaluated week. This changes attribution only; selection, sizing, P&L, weekly returns and ending balance are untouched.'},'variants':rows,'evidence_class':'EXPLORATORY_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; SAME WINDOW USED FOR VARIANT COMPARISON; FRESH VALIDATION REQUIRED BEFORE PROMOTION'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,default=str));flat=[]
    for r in rows:flat.append({k:r[k] for k in ['mode','end_balance_usd','return_pct','positive_weeks','trades','profit_factor','reported_event_dd_pct','batch_consistent_event_dd_pct','max_weekly_dd_pct','weekly_return_min_pct','weekly_return_std_pct']})
    pd.DataFrame(flat).to_csv(a.out.with_suffix('.csv'),index=False);print(json.dumps(report,indent=2,default=str))
if __name__=='__main__':main()
