#!/usr/bin/env python3
from __future__ import annotations
import argparse,io,json,urllib.request,zipfile
from pathlib import Path
import numpy as np,pandas as pd
MAP={'BTCUSD':'BTCUSDT','ETHUSD':'ETHUSDT','SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'}
BASE='https://data.binance.vision/data/futures/um'
START=pd.Timestamp('2026-05-04T00:00:00Z');END=pd.Timestamp('2026-08-14T23:59:59Z')

def getzip(url):
 req=urllib.request.Request(url,headers={'User-Agent':'Trailaris-R5.4-orderflow/1.0'})
 with urllib.request.urlopen(req,timeout=90) as r:b=r.read()
 z=zipfile.ZipFile(io.BytesIO(b));n=[x for x in z.namelist() if x.endswith('.csv')]
 if not n:raise ValueError('no csv '+url)
 return pd.read_csv(z.open(n[0]),header=None)

def unit(s):
 x=pd.to_numeric(s,errors='coerce').dropna();m=x.abs().median();return 'us' if m>1e14 else ('ms' if m>1e11 else 's')

def parse(d):
 if d.shape[1]<11:raise ValueError('unexpected Binance kline archive schema')
 ot=pd.to_numeric(d.iloc[:,0],errors='coerce');ct=pd.to_numeric(d.iloc[:,6],errors='coerce')
 out=pd.DataFrame({'open_time':pd.to_datetime(ot,unit=unit(ot),utc=True,errors='coerce'),'timestamp':pd.to_datetime(ct,unit=unit(ct),utc=True,errors='coerce'),'volume':pd.to_numeric(d.iloc[:,5],errors='coerce'),'trades':pd.to_numeric(d.iloc[:,8],errors='coerce'),'taker_buy_base':pd.to_numeric(d.iloc[:,9],errors='coerce')})
 return out.dropna().drop_duplicates('timestamp').sort_values('timestamp')

def month(symbol,y,m):
 ym=f'{y}-{m:02d}';return parse(getzip(f'{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{ym}.zip'))

def day(symbol,d):
 ds=d.strftime('%Y-%m-%d');return parse(getzip(f'{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{ds}.zip'))

def history(symbol):
 parts=[month(symbol,2026,m) for m in [5,6,7]]
 for d in pd.date_range('2026-08-01','2026-08-14',freq='D'):parts.append(day(symbol,d))
 z=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp');return z[(z.timestamp>=START)&(z.timestamp<=END)]

def causal_z(s,n=32,minp=12):
 p=s.shift(1);mu=p.rolling(n,min_periods=minp).mean();sd=p.rolling(n,min_periods=minp).std().replace(0,np.nan);return ((s-mu)/sd).clip(-5,5)

def build(asset,symbol):
 m1=history(symbol);source_bars=len(m1)
 # Aggregate closed 1m trade-flow evidence into non-overlapping 15m decision bars.
 # No future bar enters a decision: timestamp is the right edge of the closed interval.
 d=m1.set_index('timestamp').resample('15min',label='right',closed='right').agg(volume=('volume','sum'),taker_buy_base=('taker_buy_base','sum'),trades=('trades','sum')).dropna().reset_index()
 d=d[d.volume>0].copy();d['taker_sell_base']=(d.volume-d.taker_buy_base).clip(lower=0);d['flow_imbalance']=(d.taker_buy_base-d.taker_sell_base)/d.volume
 d['flow_z']=causal_z(d.flow_imbalance);trade_med=d.trades.shift(1).rolling(32,min_periods=12).median().replace(0,np.nan);d['activity']=np.log1p((d.trades/trade_med).clip(lower=0,upper=10));d['orderflow_score']=(d.flow_z*(.75+.25*np.tanh(d.activity))).clip(-4,4)
 d=d.dropna(subset=['orderflow_score']);scored_bars=len(d)
 # R4's factor family itself requires |score| >= 0.70; retain only those executable
 # evidence observations to avoid duplicating non-trigger rows in the estate.
 d=d[d.orderflow_score.abs()>=.70].copy();d['asset']=asset;d['carry_score']=np.nan;d['funding_score']=np.nan;d['event_score']=np.nan;d['evidence_state']='VERIFIED_BINANCE_USDM_TAKER_FLOW_15M_CAUSAL';d['source']='BINANCE_USDM_1M_KLINE_TAKER_BUY_BASE_VOLUME_ARCHIVE_AGG_15M'
 out=d[['timestamp','asset','orderflow_score','carry_score','funding_score','event_score','evidence_state','flow_imbalance','flow_z','trades','activity','source']]
 return out,{'asset':asset,'symbol':symbol,'source_m1_bars':source_bars,'scored_15m_bars':scored_bars,'factor_rows':len(out),'trigger_rows_abs_ge_070':len(out),'first':str(out.timestamp.min()),'last':str(out.timestamp.max()),'state':'PASS' if len(out) else 'NO_FACTOR_ROWS'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--audit',required=True);a=ap.parse_args();rs=[];au=[]
 for asset,symbol in MAP.items():d,s=build(asset,symbol);rs.append(d);au.append(s)
 f=pd.concat(rs,ignore_index=True).sort_values(['timestamp','asset']);Path(a.out).parent.mkdir(parents=True,exist_ok=True);f.to_csv(a.out,index=False);pd.DataFrame(au).to_csv(a.audit,index=False)
 s={'assets_required':5,'assets_pass':sum(x['state']=='PASS' for x in au),'source_m1_bars':sum(x['source_m1_bars'] for x in au),'scored_15m_bars':sum(x['scored_15m_bars'] for x in au),'factor_rows':len(f),'trigger_rows_abs_ge_070':len(f),'decision_interval':'15min_nonoverlapping_closed','causality':'15-minute bar is fully closed before timestamp; rolling normalization uses shifted prior 15-minute observations only','evidence_state':'VERIFIED_BINANCE_USDM_TAKER_FLOW_15M_CAUSAL'};Path(str(a.audit)+'.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));assert s['assets_pass']==5 and len(f)>0
if __name__=='__main__':main()
