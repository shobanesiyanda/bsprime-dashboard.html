#!/usr/bin/env python3
from __future__ import annotations
import argparse,io,json,time,urllib.request,zipfile
from pathlib import Path
import numpy as np,pandas as pd
START=pd.Timestamp('2026-01-19T00:00:00Z');END=pd.Timestamp('2026-08-14T23:59:59Z')
CRYPTO={'BTCUSD':'BTCUSDT','ETHUSD':'ETHUSDT','SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'}
ROUTES=['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
SYNTH={'V75','V50','V100','V25','V10'}
EVENTS=[
('2026-01-28T19:00:00Z','FOMC','https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'),
('2026-01-30T13:30:00Z','PPI','https://www.bls.gov/schedule/news_release/ppi.htm'),
('2026-02-11T13:30:00Z','NFP_EMPLOYMENT','https://www.bls.gov/schedule/news_release/empsit.htm'),
('2026-02-13T13:30:00Z','CPI','https://www.bls.gov/schedule/news_release/cpi.htm'),
('2026-02-27T13:30:00Z','PPI','https://www.bls.gov/schedule/news_release/ppi.htm'),
('2026-03-06T13:30:00Z','NFP_EMPLOYMENT','https://www.bls.gov/schedule/news_release/empsit.htm'),
('2026-03-11T12:30:00Z','CPI','https://www.bls.gov/schedule/news_release/cpi.htm'),
('2026-03-18T12:30:00Z','PPI','https://www.bls.gov/schedule/news_release/ppi.htm'),
('2026-03-18T18:00:00Z','FOMC','https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'),
('2026-04-03T12:30:00Z','NFP_EMPLOYMENT','https://www.bls.gov/schedule/news_release/empsit.htm'),
('2026-04-10T12:30:00Z','CPI','https://www.bls.gov/schedule/news_release/cpi.htm'),
('2026-04-14T12:30:00Z','PPI','https://www.bls.gov/schedule/news_release/ppi.htm'),
('2026-04-29T18:00:00Z','FOMC','https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'),
('2026-05-08T12:30:00Z','NFP_EMPLOYMENT','https://www.bls.gov/schedule/2026/home.htm'),
('2026-05-12T12:30:00Z','CPI','https://www.bls.gov/schedule/2026/home.htm'),
('2026-05-13T12:30:00Z','PPI','https://www.bls.gov/schedule/2026/home.htm'),
('2026-06-05T12:30:00Z','NFP_EMPLOYMENT','https://www.bls.gov/schedule/2026/home.htm'),
('2026-06-10T12:30:00Z','CPI','https://www.bls.gov/schedule/2026/home.htm'),
('2026-06-11T12:30:00Z','PPI','https://www.bls.gov/schedule/2026/home.htm'),
('2026-06-17T18:00:00Z','FOMC','https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'),
('2026-07-02T12:30:00Z','NFP_EMPLOYMENT','https://www.bls.gov/schedule/2026/home.htm'),
('2026-07-14T12:30:00Z','CPI','https://www.bls.gov/schedule/2026/home.htm'),
('2026-07-15T12:30:00Z','PPI','https://www.bls.gov/schedule/2026/home.htm'),
('2026-07-29T18:00:00Z','FOMC','https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'),
('2026-08-07T12:30:00Z','NFP_EMPLOYMENT','https://www.bls.gov/schedule/2026/home.htm')]

def fetch(url):
 err=None
 for i in range(4):
  try:return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Trailaris-R5.4-26W/1.0'}),timeout=90).read()
  except Exception as e:err=e;time.sleep(2*(i+1))
 raise err

def getzip(url,header=None):
 z=zipfile.ZipFile(io.BytesIO(fetch(url)));names=[n for n in z.namelist() if n.lower().endswith('.csv')]
 if not names:raise ValueError('zip has no csv '+url)
 return pd.read_csv(z.open(names[0]),header=header)
def unit(s):
 x=pd.to_numeric(s,errors='coerce').dropna();m=x.abs().median();return 'us' if m>1e14 else ('ms' if m>1e11 else 's')
def parse_kline(d):
 if d.shape[1]<11:raise ValueError('unexpected kline archive schema')
 ot=pd.to_numeric(d.iloc[:,0],errors='coerce');ct=pd.to_numeric(d.iloc[:,6],errors='coerce')
 return pd.DataFrame({'open_time':pd.to_datetime(ot,unit=unit(ot),utc=True,errors='coerce'),'timestamp':pd.to_datetime(ct,unit=unit(ct),utc=True,errors='coerce'),'volume':pd.to_numeric(d.iloc[:,5],errors='coerce'),'trades':pd.to_numeric(d.iloc[:,8],errors='coerce'),'taker_buy_base':pd.to_numeric(d.iloc[:,9],errors='coerce')}).dropna().drop_duplicates('timestamp').sort_values('timestamp')
def causal_z(s,n,minp):
 p=s.shift(1);mu=p.rolling(n,min_periods=minp).mean();sd=p.rolling(n,min_periods=minp).std().replace(0,np.nan);return ((s-mu)/sd).clip(-5,5)
def futures_kline_history(symbol,kind='klines'):
 base='https://data.binance.vision/data/futures/um';parts=[]
 for m in range(1,8):
  ym=f'2026-{m:02d}';parts.append(parse_kline(getzip(f'{base}/monthly/{kind}/{symbol}/1m/{symbol}-1m-{ym}.zip',header=None)))
 for d in pd.date_range('2026-08-01','2026-08-14',freq='D'):
  ds=d.strftime('%Y-%m-%d');parts.append(parse_kline(getzip(f'{base}/daily/{kind}/{symbol}/1m/{symbol}-1m-{ds}.zip',header=None)))
 z=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp');return z[(z.timestamp>=START)&(z.timestamp<=END)]
def parse_funding(raw):
 d=raw.copy();cols={str(c).strip().lower():c for c in d.columns}
 if 'calc_time' in cols:
  tc=cols['calc_time'];rc=cols.get('last_funding_rate') or cols.get('funding_rate')
  if rc is None:raise ValueError('funding rate column missing')
  t=pd.to_numeric(d[tc],errors='coerce');r=pd.to_numeric(d[rc],errors='coerce')
 else:
  if d.shape[1]<3:raise ValueError('unexpected funding layout')
  t=pd.to_numeric(d.iloc[:,0],errors='coerce');r=pd.to_numeric(d.iloc[:,-1],errors='coerce')
 return pd.DataFrame({'timestamp':pd.to_datetime(t,unit=unit(t),utc=True,errors='coerce'),'funding_rate':r}).dropna().drop_duplicates('timestamp').sort_values('timestamp')
def funding_history(symbol):
 base='https://data.binance.vision/data/futures/um';parts=[]
 for m in range(1,8):
  ym=f'2026-{m:02d}';url=f'{base}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip'
  try:d=parse_funding(getzip(url,header=0))
  except Exception:d=parse_funding(getzip(url,header=None))
  parts.append(d)
 z=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp');return z[(z.timestamp>=START)&(z.timestamp<=END)]
def premium_history(symbol):
 base='https://data.binance.vision/data/futures/um';parts=[]
 def pp(d):
  t=pd.to_numeric(d.iloc[:,0],errors='coerce');return pd.DataFrame({'timestamp':pd.to_datetime(t,unit=unit(t),utc=True,errors='coerce'),'premium':pd.to_numeric(d.iloc[:,4],errors='coerce')}).dropna().drop_duplicates('timestamp').sort_values('timestamp')
 for m in range(1,8):
  ym=f'2026-{m:02d}';parts.append(pp(getzip(f'{base}/monthly/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{ym}.zip',header=None)))
 for d in pd.date_range('2026-08-01','2026-08-14',freq='D'):
  ds=d.strftime('%Y-%m-%d');parts.append(pp(getzip(f'{base}/daily/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{ds}.zip',header=None)))
 z=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp');return z[(z.timestamp>=START)&(z.timestamp<=END)]
def build_crypto():
 rows=[];audit=[]
 for asset,symbol in CRYPTO.items():
  k=futures_kline_history(symbol);d=k.set_index('timestamp').resample('15min',label='right',closed='right').agg(volume=('volume','sum'),taker_buy_base=('taker_buy_base','sum'),trades=('trades','sum')).dropna().reset_index();d=d[d.volume>0].copy();d['sell']=(d.volume-d.taker_buy_base).clip(lower=0);d['imb']=(d.taker_buy_base-d.sell)/d.volume;d['z']=causal_z(d.imb,32,12);med=d.trades.shift(1).rolling(32,min_periods=12).median().replace(0,np.nan);d['activity']=np.log1p((d.trades/med).clip(0,10));d['orderflow_score']=(d.z*(.75+.25*np.tanh(d.activity))).clip(-4,4);d=d[d.orderflow_score.abs()>=.70].dropna(subset=['orderflow_score']);d['asset']=asset;d['carry_score']=np.nan;d['funding_score']=np.nan;d['event_score']=np.nan;d['evidence_state']='VERIFIED_BINANCE_USDM_TAKER_FLOW_15M_CAUSAL';rows.append(d[['timestamp','asset','orderflow_score','carry_score','funding_score','event_score','evidence_state']])
  f=funding_history(symbol);p=premium_history(symbol);m=pd.merge_asof(f.sort_values('timestamp'),p.sort_values('timestamp'),on='timestamp',direction='backward',tolerance=pd.Timedelta(minutes=2)).dropna();m['fz']=causal_z(m.funding_rate,20,8);m['pz']=causal_z(m.premium,30,12);m['funding_score']=-(.55*m.fz+.45*m.pz);m=m.dropna(subset=['funding_score']);m['funding_score']=m.funding_score.clip(-4,4);m['asset']=asset;m['orderflow_score']=np.nan;m['carry_score']=np.nan;m['event_score']=np.nan;m['evidence_state']='VERIFIED_BINANCE_PUBLIC_ARCHIVE_FUNDING_BASIS_CAUSAL';rows.append(m[['timestamp','asset','orderflow_score','carry_score','funding_score','event_score','evidence_state']]);audit.append({'asset':asset,'orderflow_rows':len(d),'funding_rows':len(m)})
 return rows,audit
def load_close(path):
 d=pd.read_csv(path,usecols=['timestamp','close']);d['timestamp']=pd.to_datetime(d.timestamp,utc=True,errors='coerce');d['close']=pd.to_numeric(d.close,errors='coerce');return d.dropna().drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp').close
def px(s,t,maxage):
 z=s.loc[:t]
 if z.empty:return None
 if t-z.index[-1]>pd.Timedelta(minutes=maxage):return None
 return float(z.iloc[-1])
def build_events(raw):
 rows=[];audit=[]
 for asset in ROUTES:
  if asset in SYNTH:audit.append({'asset':asset,'event_rows':0,'state':'NON_NATIVE_EVENT_FAMILY_GATED'});continue
  p=raw/f'{asset}_M1_normalized.csv'
  if not p.exists():audit.append({'asset':asset,'event_rows':0,'state':'M1_MISSING'});continue
  s=load_close(p);n=0
  for ets,fam,src in EVENTS:
   et=pd.Timestamp(ets);dt=et+pd.Timedelta(minutes=15);p0=px(s,et,10);p1=px(s,dt,5)
   if p0 is None or p1 is None or p0<=0:continue
   r=s.loc[et-pd.Timedelta(minutes=90):et].pct_change().dropna().tail(60)
   if len(r)<20:continue
   sig=float(r.std(ddof=1))*np.sqrt(15)
   if not np.isfinite(sig) or sig<=1e-12:continue
   score=float(np.clip(((p1/p0)-1)/sig,-4,4));rows.append({'timestamp':dt,'asset':asset,'orderflow_score':np.nan,'carry_score':np.nan,'funding_score':np.nan,'event_score':score,'evidence_state':'RELEASED_OFFICIAL_CALENDAR_CAUSAL_POST_EVENT','event_family':fam,'event_release_time':et,'event_source':src});n+=1
  audit.append({'asset':asset,'event_rows':n,'state':'PASS' if n else 'NO_EXECUTABLE_EVENT_WINDOW'})
 return pd.DataFrame(rows),audit
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--out',required=True);ap.add_argument('--audit',required=True);a=ap.parse_args();raw=Path(a.rawdir);parts,ca=build_crypto();ev,ea=build_events(raw);parts.append(ev[['timestamp','asset','orderflow_score','carry_score','funding_score','event_score','evidence_state']]);z=pd.concat(parts,ignore_index=True);z['timestamp']=pd.to_datetime(z.timestamp,utc=True);z=z[(z.timestamp>=START)&(z.timestamp<=END)].sort_values(['timestamp','asset','evidence_state']).drop_duplicates(['timestamp','asset','evidence_state']);Path(a.out).parent.mkdir(parents=True,exist_ok=True);z.to_csv(a.out,index=False);summary={'start':str(START),'end':str(END),'combined_rows':len(z),'event_rows':len(ev),'orderflow_rows':sum(x['orderflow_rows'] for x in ca),'funding_rows':sum(x['funding_rows'] for x in ca),'crypto_assets':ca,'event_routes':ea,'evidence_states':sorted(z.evidence_state.dropna().unique().tolist())};Path(a.audit).write_text(json.dumps(summary,indent=2,default=str));print(json.dumps({k:v for k,v in summary.items() if k not in {'crypto_assets','event_routes'}},indent=2));assert len(ROUTES)==34 and all(x['orderflow_rows']>0 and x['funding_rows']>0 for x in ca)
if __name__=='__main__':main()
