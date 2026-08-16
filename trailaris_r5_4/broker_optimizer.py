#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROUTES=[
'XAUUSD','XAGUSD','EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF',
'BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
MIN_SAMPLE_N=1000
REQUIRED=[
'broker_id','route','available','sample_n','trade_lot','spread_cost_median_usd','spread_cost_p95_usd','commission_round_turn_usd',
'slippage_mean_usd','slippage_p95_usd','financing_long_usd','financing_short_usd','contract_size','tick_size','tick_value','min_lot','lot_step',
'margin_required_usd','fill_rate','reject_rate','latency_median_ms','latency_p95_ms','spec_hash','captured_from_target_server','broker_certified']

def _bool(v):
    if isinstance(v,bool): return v
    return str(v).strip().lower() in {'1','true','yes','y'}

def validate(df:pd.DataFrame):
    miss=[c for c in REQUIRED if c not in df.columns]
    if miss: raise ValueError(f'missing columns: {miss}')
    if not set(df.route).issubset(set(ROUTES)): raise ValueError('unknown route')
    return True

def eligible_row(r:pd.Series)->tuple[bool,str]:
    if not _bool(r.available): return False,'ROUTE_UNAVAILABLE'
    if int(r.sample_n)<MIN_SAMPLE_N: return False,'INSUFFICIENT_SAMPLE'
    if not _bool(r.captured_from_target_server): return False,'NOT_TARGET_SERVER_CAPTURED'
    if not _bool(r.broker_certified): return False,'NOT_BROKER_CERTIFIED'
    numeric=[c for c in REQUIRED if c not in {'broker_id','route','available','spec_hash','captured_from_target_server','broker_certified'}]
    for c in numeric:
        if pd.isna(r[c]) or not np.isfinite(float(r[c])): return False,f'MISSING_{c.upper()}'
    lot=float(r.trade_lot);minlot=float(r.min_lot);step=float(r.lot_step)
    if lot<minlot-1e-12:return False,'BASELINE_TRADE_BELOW_MIN_LOT'
    q=(lot-minlot)/step
    if abs(q-round(q))>1e-6:return False,'BASELINE_TRADE_NOT_ON_LOT_STEP'
    if float(r.fill_rate)<=0 or float(r.fill_rate)>1:return False,'INVALID_FILL_RATE'
    if float(r.reject_rate)<0 or float(r.reject_rate)>=1:return False,'INVALID_REJECT_RATE'
    if minlot<=0 or step<=0:return False,'INVALID_LOT_RULE'
    if not str(r.spec_hash).strip():return False,'MISSING_SPEC_HASH'
    return True,'ELIGIBLE'

def enrich(df:pd.DataFrame)->pd.DataFrame:
    x=df.copy()
    for c in REQUIRED:
        if c in {'broker_id','route','spec_hash'}:continue
        if c in {'available','captured_from_target_server','broker_certified'}:x[c]=x[c].map(_bool)
        else:x[c]=pd.to_numeric(x[c],errors='coerce')
    states=[eligible_row(r) for _,r in x.iterrows()]
    x['eligible']=[a for a,_ in states];x['eligibility_reason']=[b for _,b in states]
    x['financing_worst_usd']=x[['financing_long_usd','financing_short_usd']].abs().max(axis=1)
    x['all_in_cost_usd']=x['spread_cost_median_usd']+x['commission_round_turn_usd']+x['slippage_mean_usd'].clip(lower=0)+x['financing_worst_usd']
    x['tail_cost_usd']=x['spread_cost_p95_usd']+x['commission_round_turn_usd']+x['slippage_p95_usd'].clip(lower=0)+x['financing_worst_usd']
    return x

def _norm(s:pd.Series):
    lo=float(s.min());hi=float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi-lo<1e-12:return pd.Series(np.zeros(len(s)),index=s.index)
    return (s-lo)/(hi-lo)

def rank_route(g:pd.DataFrame)->pd.DataFrame:
    e=g[g.eligible].copy()
    if e.empty:
        out=g.copy();out['broker_score']=np.nan;out['rank']=np.nan;return out
    cost=_norm(e.all_in_cost_usd);tail=_norm(e.tail_cost_usd);reject=_norm(e.reject_rate);latency=_norm(e.latency_p95_ms);fill_penalty=_norm(1-e.fill_rate);margin=_norm(e.margin_required_usd)
    e['broker_score']=0.45*cost+0.20*tail+0.12*reject+0.08*fill_penalty+0.08*latency+0.07*margin
    e=e.sort_values(['broker_score','all_in_cost_usd','tail_cost_usd','broker_id']);e['rank']=np.arange(1,len(e)+1)
    ne=g[~g.eligible].copy();ne['broker_score']=np.nan;ne['rank']=np.nan
    return pd.concat([e,ne],ignore_index=True)

def optimize(df:pd.DataFrame):
    validate(df);x=enrich(df)
    ranked=pd.concat([rank_route(g) for _,g in x.groupby('route',sort=False)],ignore_index=True)
    winners=[]
    for route in ROUTES:
        g=ranked[(ranked.route==route)&(ranked.eligible)].sort_values('rank')
        winners.append({'route':route,'state':'CERTIFIED_WINNER' if len(g) else 'NO_CERTIFIED_BROKER','primary_broker':str(g.iloc[0].broker_id) if len(g) else '','backup_broker':str(g.iloc[1].broker_id) if len(g)>1 else '','primary_score':float(g.iloc[0].broker_score) if len(g) else None,'primary_all_in_cost_usd':float(g.iloc[0].all_in_cost_usd) if len(g) else None,'primary_tail_cost_usd':float(g.iloc[0].tail_cost_usd) if len(g) else None,'certified_candidates':int(len(g))})
    w=pd.DataFrame(winners)
    status={'routes_required':34,'routes_with_certified_winner':int((w.state=='CERTIFIED_WINNER').sum()),'broker_optimization_gate_pass':bool((w.state=='CERTIFIED_WINNER').all()),'selection_is_route_specific':True,'cost_unit':'USD_PER_ACTUAL_PROPOSED_ROUND_TURN','ib_revenue_used_in_score':False,'baseline_non_regression_required':True}
    return ranked,w,status

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--observations',required=True);ap.add_argument('--outdir',required=True)
    a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    ranked,w,status=optimize(pd.read_csv(a.observations));ranked.to_csv(out/'R5_4_BROKER_RANKINGS.csv',index=False);w.to_csv(out/'R5_4_ROUTE_BROKER_WINNERS.csv',index=False);(out/'R5_4_BROKER_OPTIMIZATION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))

if __name__=='__main__':main()
