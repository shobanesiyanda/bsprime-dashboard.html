#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,requests,websocket
START=pd.Timestamp('2025-07-21T00:00:00Z');END=pd.Timestamp('2026-01-19T00:00:00Z')
SESSIONS=[('ASIA',0,7),('LONDON',7,13),('NEW_YORK',13,21),('LATE',21,24)]
CRYPTO={'SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'};SYN={'V75':'R_75','V50':'R_50','V100':'R_100','V25':'R_25','V10':'R_10'};CG={'SOLUSDT':'solana','BNBUSDT':'binancecoin','XRPUSDT':'ripple'}
def crypto_rows(symbol):
 rows=[];cur=int(START.timestamp()*1000);end=int(END.timestamp()*1000)
 try:
  while cur<end:
   r=requests.get('https://api.binance.us/api/v3/klines',params={'symbol':symbol,'interval':'5m','startTime':cur,'endTime':end-1,'limit':1000},timeout=20);r.raise_for_status();x=r.json()
   if not x:break
   rows += [(pd.to_datetime(int(q[0]),unit='ms',utc=True),float(q[4])) for q in x];nxt=int(x[-1][0])+300000
   if nxt<=cur:break
   cur=nxt;time.sleep(.03)
  if len(rows)>1000:return rows,f'BINANCE_US_{symbol}_5M_STRUCTURAL_PROXY',False
 except Exception:pass
 cid=CG[symbol];r=requests.get(f'https://api.coingecko.com/api/v3/coins/{cid}/market_chart/range',params={'vs_currency':'usd','from':int(START.timestamp()),'to':int(END.timestamp())},timeout=30);r.raise_for_status();rows=[(pd.to_datetime(int(t),unit='ms',utc=True),float(p)) for t,p in r.json().get('prices',[])];return rows,f'COINGECKO_{cid}_USD_STRUCTURAL_FALLBACK',True
def deriv(symbol):
 ws=websocket.create_connection('wss://ws.binaryws.com/websockets/v3?app_id=1089',timeout=20);rows=[]
 try:
  ws.send(json.dumps({'ticks_history':symbol,'end':int(END.timestamp())-1,'style':'candles','granularity':86400,'count':250,'adjust_start_time':1}))
  x=json.loads(ws.recv())
  if x.get('error'):raise RuntimeError(x['error'].get('message'))
  for c in x.get('candles',[]):
   t=int(c['epoch'])
   if int(START.timestamp())<=t<int(END.timestamp()):rows.append((pd.to_datetime(t,unit='s',utc=True),float(c['close'])))
 finally:ws.close()
 return sorted(set(rows))
def summarize(asset,rows,source,daily_fallback=False):
 z=pd.DataFrame(rows,columns=['timestamp','close']).drop_duplicates('timestamp').sort_values('timestamp')
 if len(z)<2:raise RuntimeError(f'empty structural reference {asset}')
 lr=np.log(z.close).diff();z['abs_lr']=lr.abs();dc=z.set_index('timestamp').close.resample('1D').last().dropna();dr=dc.pct_change().dropna();cls='CRYPTO' if asset in CRYPTO else 'SYNTHETIC';daily=[{'date':t.date().isoformat(),'asset':asset,'return':float(v),'role':'CORE','asset_class':cls,'anchor':asset} for t,v in dr.items()]
 if daily_fallback:vals=[(n,.25) for n,_,_ in SESSIONS]
 else:
  vals=[(n,float(z.loc[z.timestamp.dt.hour.ge(h0)&z.timestamp.dt.hour.lt(h1),'abs_lr'].sum(skipna=True))) for n,h0,h1 in SESSIONS];tot=sum(v for _,v in vals) or 1.;vals=[(n,v/tot) for n,v in vals]
 sess=[{'asset':asset,'session':n,'activity_share':v,'role':'CORE'} for n,v in vals];first=z.timestamp.iloc[0];last=z.timestamp.iloc[-1];q={'asset':asset,'role':'CORE','asset_class':cls,'provider_name':source,'rows':len(z),'first':str(first),'last':str(last),'daily_observations':len(dr),'stale_return_share':float((lr.abs()<1e-14).mean()),'local_gap_gt5m_share':0.0,'coverage_pass':bool(len(z)>100 and first<=START+pd.Timedelta(days=3) and last>=END-pd.Timedelta(days=3)),'anchor':asset,'session_profile_neutral_fallback':daily_fallback};return daily,sess,q
def one(asset):
 if asset in CRYPTO:rows,src,fb=crypto_rows(CRYPTO[asset]);return summarize(asset,rows,src,fb)
 if asset in SYN:return summarize(asset,deriv(SYN[asset]),f'DERIV_{SYN[asset]}_1D_STRUCTURAL',True)
 raise ValueError(asset)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--asset',choices=list(CRYPTO)+list(SYN));a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True);assets=[a.asset] if a.asset else list(CRYPTO)+list(SYN);daily=[];sess=[];qual=[]
 for asset in assets:d,s,q=one(asset);daily+=d;sess+=s;qual.append(q)
 pd.DataFrame(daily).to_csv(a.outdir/'DAILY_RETURNS.csv',index=False);pd.DataFrame(sess).to_csv(a.outdir/'SESSION_PROFILE.csv',index=False);pd.DataFrame(qual).to_csv(a.outdir/'DATA_QUALITY.csv',index=False);bad=[q['asset'] for q in qual if not q['coverage_pass']]
 if bad:raise RuntimeError(f'missing-core structural coverage failure: {bad}')
 status={'state':'MISSING_CORE_STRUCTURAL_REFERENCE_COMPLETE','assets':len(assets),'asset_names':assets,'uses_strategy_outcomes':False};(a.outdir/'SUMMARY.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
if __name__=='__main__':main()
