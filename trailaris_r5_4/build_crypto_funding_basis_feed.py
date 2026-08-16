#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, math, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

MAP={'BTCUSD':'BTCUSDT','ETHUSD':'ETHUSDT','SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'}
BASE='https://data.binance.vision/data/futures/um'
FAPI='https://fapi.binance.com'
START=pd.Timestamp('2026-05-04T00:00:00Z');END=pd.Timestamp('2026-08-14T23:59:59Z')

def get_json(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Trailaris-R5.4-evidence/1.0'})
    with urllib.request.urlopen(req,timeout=60) as r:return json.loads(r.read())

def funding_history(symbol):
    u=f"{FAPI}/fapi/v1/fundingRate?symbol={symbol}&startTime={int(START.timestamp()*1000)}&endTime={int(END.timestamp()*1000)}&limit=1000"
    j=get_json(u);d=pd.DataFrame(j)
    if d.empty:return d
    d['timestamp']=pd.to_datetime(pd.to_numeric(d.fundingTime),unit='ms',utc=True);d['funding_rate']=pd.to_numeric(d.fundingRate,errors='coerce')
    return d[['timestamp','funding_rate']].dropna().sort_values('timestamp')

def _read_zip(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Trailaris-R5.4-evidence/1.0'})
    with urllib.request.urlopen(req,timeout=90) as r:b=r.read()
    z=zipfile.ZipFile(io.BytesIO(b));names=[n for n in z.namelist() if n.lower().endswith('.csv')]
    if not names:raise ValueError('zip has no csv: '+url)
    return pd.read_csv(z.open(names[0]),header=None)

def premium_month(symbol,year,month):
    ym=f'{year}-{month:02d}';url=f'{BASE}/monthly/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{ym}.zip'
    d=_read_zip(url);return parse_kline(d)

def premium_day(symbol,day):
    ds=day.strftime('%Y-%m-%d');url=f'{BASE}/daily/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{ds}.zip'
    d=_read_zip(url);return parse_kline(d)

def parse_kline(d):
    # Binance public kline archives use standard kline order. We only require open time + close.
    if d.shape[1]<5:raise ValueError('unexpected premium kline columns')
    t=pd.to_numeric(d.iloc[:,0],errors='coerce');unit='us' if t.dropna().median()>1e14 else ('ms' if t.dropna().median()>1e11 else 's')
    out=pd.DataFrame({'timestamp':pd.to_datetime(t,unit=unit,utc=True,errors='coerce'),'premium':pd.to_numeric(d.iloc[:,4],errors='coerce')})
    return out.dropna().drop_duplicates('timestamp').sort_values('timestamp')

def premium_history(symbol):
    parts=[]
    for y,m in [(2026,5),(2026,6),(2026,7)]:
        try:parts.append(premium_month(symbol,y,m))
        except Exception as e:print('MONTH_MISS',symbol,y,m,repr(e))
    for day in pd.date_range('2026-08-01','2026-08-14',freq='D'):
        try:parts.append(premium_day(symbol,day))
        except Exception as e:print('DAY_MISS',symbol,day.date(),repr(e))
    if not parts:return pd.DataFrame(columns=['timestamp','premium'])
    d=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp')
    return d[(d.timestamp>=START)&(d.timestamp<=END)]

def causal_z(s,window,minp):
    # shift(1): each observation is standardized only against evidence strictly before it.
    past=s.shift(1);mu=past.rolling(window,min_periods=minp).mean();sd=past.rolling(window,min_periods=minp).std().replace(0,np.nan)
    return ((s-mu)/sd).clip(-5,5)

def build_asset(asset,symbol):
    f=funding_history(symbol);p=premium_history(symbol)
    if f.empty or p.empty:return pd.DataFrame(),{'asset':asset,'funding_rows':len(f),'premium_rows':len(p),'factor_rows':0,'state':'EVIDENCE_MISSING'}
    # Premium regime at each funding settlement time: use last premium observation at or before funding time.
    p=p.sort_values('timestamp');f=f.sort_values('timestamp')
    m=pd.merge_asof(f,p,on='timestamp',direction='backward',tolerance=pd.Timedelta(minutes=2)).dropna()
    # Funding z uses 20 prior settlements; premium z uses previous 30 settlement-aligned observations.
    m['funding_z']=causal_z(m.funding_rate,20,8);m['premium_z']=causal_z(m.premium,30,12)
    # Convergence thesis: exceptionally positive funding/premium is short-biased; exceptionally negative is long-biased.
    m['funding_score']=-(.55*m.funding_z+.45*m.premium_z)
    m=m.dropna(subset=['funding_score']);m['funding_score']=m.funding_score.clip(-4,4)
    m['asset']=asset;m['orderflow_score']=np.nan;m['carry_score']=np.nan;m['event_score']=np.nan
    m['evidence_state']='VERIFIED_BINANCE_PUBLIC_FUTURES_FUNDING_BASIS_CAUSAL'
    m['source_funding']='BINANCE_USDM_FUNDING_RATE_HISTORY';m['source_basis']='BINANCE_USDM_PREMIUM_INDEX_KLINES'
    out=m[['timestamp','asset','orderflow_score','carry_score','funding_score','event_score','evidence_state','funding_rate','premium','funding_z','premium_z','source_funding','source_basis']]
    st={'asset':asset,'symbol':symbol,'funding_rows':len(f),'premium_rows':len(p),'factor_rows':len(out),'trigger_rows_abs_ge_070':int((out.funding_score.abs()>=.70).sum()),'first':str(out.timestamp.min()) if len(out) else None,'last':str(out.timestamp.max()) if len(out) else None,'state':'PASS' if len(out) else 'NO_FACTOR_ROWS'}
    return out,st

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--audit',required=True);a=ap.parse_args();rows=[];audit=[]
    for asset,symbol in MAP.items():
        d,s=build_asset(asset,symbol);audit.append(s)
        if len(d):rows.append(d)
    feed=pd.concat(rows,ignore_index=True).sort_values(['timestamp','asset']) if rows else pd.DataFrame()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);feed.to_csv(a.out,index=False);pd.DataFrame(audit).to_csv(a.audit,index=False)
    summary={'assets_required':5,'assets_pass':int(sum(x['state']=='PASS' for x in audit)),'factor_rows':len(feed),'trigger_rows_abs_ge_070':int((feed.funding_score.abs()>=.70).sum()) if len(feed) else 0,'causality':'funding settlement and premium observed at/before timestamp; z-score baselines use shifted prior observations only','evidence_state':'VERIFIED_BINANCE_PUBLIC_FUTURES_FUNDING_BASIS_CAUSAL'}
    Path(str(a.audit)+'.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));assert summary['assets_pass']==5

if __name__=='__main__':main()
