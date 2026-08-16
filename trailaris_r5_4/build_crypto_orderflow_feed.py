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
 # Binance USD-M klines: open_time,o,h,l,c,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote,ignore
 if d.shape[1]<11:raise ValueError('unexpected Binance kline archive schema')
 ot=pd.to_numeric(d.iloc[:,0],errors='coerce');ct=pd.to_numeric(d.iloc[:,6],errors='coerce')
 out=pd.DataFrame({
  'open_time':pd.to_datetime(ot,unit=unit(ot),utc=True,errors='coerce'),
  'timestamp':pd.to_datetime(ct,unit=unit(ct),utc=True,errors='coerce'),
  'volume':pd.to_numeric(d.iloc[:,5],errors='coerce'),
  'trades':pd.to_numeric(d.iloc[:,8],errors='coerce'),
  'taker_buy_base':pd.to_numeric(d.iloc[:,9],errors='coerce')})
 out=out.dropna().drop_duplicates('timestamp').sort_values('timestamp');out=out[out.volume>0]
 out['taker_sell_base']=(out.volume-out.taker_buy_base).clip(lower=0)
 out['flow_imbalance']=(out.taker_buy_base-out.taker_sell_base)/out.volume
 return out

def month(symbol,y,m):
 ym=f'{y}-{m:02d}';return parse(getzip(f'{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{ym}.zip'))

def day(symbol,d):
 ds=d.strftime('%Y-%m-%d');return parse(getzip(f'{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{ds}.zip'))

def history(symbol):
 parts=[month(symbol,2026,m) for m in [5,6,7]]
 for d in pd.date_range('2026-08-01','2026-08-14',freq='D'):parts.append(day(symbol,d))
 z=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp')
 return z[(z.timestamp>=START)&(z.timestamp<=END)]

def causal_z(s,n=120,minp=40):
 # Score at each fully closed one-minute bar against strictly prior observations.
 p=s.shift(1);mu=p.rolling(n,min_periods=minp).mean();sd=p.rolling(n,min_periods=minp).std().replace(0,np.nan);return ((s-mu)/sd).clip(-5,5)

def build(asset,symbol):
 d=history(symbol)
 # Smooth only with current/past closed bars. Trade-count weighting discounts tiny bars.
 d['flow5']=d.flow_imbalance.rolling(5,min_periods=3).mean();d['flow_z']=causal_z(d.flow5)
 trade_med=d.trades.shift(1).rolling(120,min_periods=40).median().replace(0,np.nan)
 d['activity']=np.log1p((d.trades/trade_med).clip(lower=0,upper=10))
 d['orderflow_score']=(d.flow_z*(.75+.25*np.tanh(d.activity))).clip(-4,4)
 d=d.dropna(subset=['orderflow_score']);d['asset']=asset;d['carry_score']=np.nan;d['funding_score']=np.nan;d['event_score']=np.nan
 d['evidence_state']='VERIFIED_BINANCE_USDM_TAKER_FLOW_CAUSAL';d['source']='BINANCE_USDM_1M_KLINE_TAKER_BUY_BASE_VOLUME_ARCHIVE'
 out=d[['timestamp','asset','orderflow_score','carry_score','funding_score','event_score','evidence_state','flow_imbalance','flow5','flow_z','trades','activity','source']]
 return out,{'asset':asset,'symbol':symbol,'bars':len(history(symbol)),'factor_rows':len(out),'trigger_rows_abs_ge_070':int((out.orderflow_score.abs()>=.70).sum()),'first':str(out.timestamp.min()),'last':str(out.timestamp.max()),'state':'PASS' if len(out) else 'NO_FACTOR_ROWS'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--audit',required=True);a=ap.parse_args();rs=[];au=[]
 for asset,symbol in MAP.items():
  d,s=build(asset,symbol);rs.append(d);au.append(s)
 f=pd.concat(rs,ignore_index=True).sort_values(['timestamp','asset']);Path(a.out).parent.mkdir(parents=True,exist_ok=True);f.to_csv(a.out,index=False);pd.DataFrame(au).to_csv(a.audit,index=False)
 s={'assets_required':5,'assets_pass':sum(x['state']=='PASS' for x in au),'factor_rows':len(f),'trigger_rows_abs_ge_070':int((f.orderflow_score.abs()>=.70).sum()),'causality':'signal timestamp is closed-bar time; smoothing uses current/past closed bars; rolling normalization is shifted one bar','evidence_state':'VERIFIED_BINANCE_USDM_TAKER_FLOW_CAUSAL'};Path(str(a.audit)+'.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));assert s['assets_pass']==5 and s['trigger_rows_abs_ge_070']>0
if __name__=='__main__':main()
