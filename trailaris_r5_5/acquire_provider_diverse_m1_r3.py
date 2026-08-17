#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

START=pd.Timestamp('2026-01-19T00:00:00Z'); END=pd.Timestamp('2026-08-14T23:59:59Z')
COLS=['timestamp','open','high','low','close','bid','ask','spread','volume','source','asset']
HISTDATA={
'EURUSD':'EURUSD','GBPUSD':'GBPUSD','USDJPY':'USDJPY','AUDUSD':'AUDUSD','USDCAD':'USDCAD','USDCHF':'USDCHF','NZDUSD':'NZDUSD','EURJPY':'EURJPY','GBPJPY':'GBPJPY','EURGBP':'EURGBP','AUDJPY':'AUDJPY','CADJPY':'CADJPY','GBPCHF':'GBPCHF','XAUUSD':'XAUUSD','XAGUSD':'XAGUSD','US100':'NSXUSD','US500':'SPXUSD','GER40':'GRXEUR','UK100':'UKXGBP','JP225':'JPXJPY','BRENT':'BCOUSD'}
COINBASE={'BTCUSD':'BTC-USD','ETHUSD':'ETH-USD','SOLUSD':'SOL-USD','BNBUSD':'BNB-USD','XRPUSD':'XRP-USD'}
PRIMARY_ONLY=['US30','NATGAS','WTI','V75','V50','V100','V25','V10']
HIST_PREFIX='https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/'
HIST_POST='https://www.histdata.com/get.php'; CB='https://api.exchange.coinbase.com'

def write_norm(rawdir,asset,df,source):
    d=df.copy(); d['timestamp']=pd.to_datetime(d.timestamp,utc=True,errors='coerce'); d=d.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp',keep='last'); d=d[(d.timestamp>=START)&(d.timestamp<=END)].copy()
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['open','high','low','close']); d['bid']=d.close; d['ask']=''; d['spread']=''; d['source']=source; d['asset']=asset; d=d[COLS]
    if len(d)<1000: raise RuntimeError(f'{asset}: rows={len(d)}')
    first,last=d.timestamp.min(),d.timestamp.max()
    if first>START+pd.Timedelta(days=1): raise RuntimeError(f'{asset}: starts {first}')
    if last<END-pd.Timedelta(days=2): raise RuntimeError(f'{asset}: ends {last}')
    d.to_csv(Path(rawdir)/f'{asset}_M1_normalized.csv',index=False,date_format='%Y-%m-%dT%H:%M:%S.%fZ')
    return {'asset':asset,'source':source,'rows':len(d),'first':str(first),'last':str(last),'median_gap_seconds':float(d.timestamp.diff().dropna().dt.total_seconds().median()),'status':'PASS'}

def hist_month(pair,month,s):
    referer=f'{HIST_PREFIX}{pair.lower()}/2026/{month}'
    for attempt in range(4):
        try:
            r1=s.get(referer,timeout=45,allow_redirects=True); r1.raise_for_status(); node=BeautifulSoup(r1.content,'html.parser').find('input',{'id':'tk'})
            if not node or not node.attrs.get('value'): raise RuntimeError('token missing')
            data={'tk':node.attrs['value'],'date':'2026','datemonth':f'2026{month:02d}','platform':'ASCII','timeframe':'M1','fxpair':pair.upper()}
            r=s.post(HIST_POST,data=data,headers={'Referer':referer,'Origin':'https://www.histdata.com','Content-Type':'application/x-www-form-urlencoded','User-Agent':'Mozilla/5.0'},timeout=90); r.raise_for_status()
            if not r.content.startswith(b'PK'): raise RuntimeError(f'non-ZIP bytes={len(r.content)}')
            return r.content
        except Exception as e:
            if attempt==3: raise RuntimeError(f'{pair} 2026-{month:02d}: {e}')
            time.sleep(1.0+attempt)

def hist_one(asset,pair,rawdir):
    frames=[]
    with requests.Session() as s:
        for m in range(1,9):
            with zipfile.ZipFile(io.BytesIO(hist_month(pair,m,s))) as z:
                for n in [x for x in z.namelist() if x.lower().endswith('.csv')]:
                    txt=z.read(n).decode('utf-8',errors='ignore')
                    if not txt.strip(): continue
                    x=pd.read_csv(io.StringIO(txt),sep=';',header=None,names=['dt','open','high','low','close','volume'],engine='python'); local=pd.to_datetime(x.dt.astype(str).str.strip(),format='%Y%m%d %H%M%S',errors='coerce'); x['timestamp']=local.dt.tz_localize('Etc/GMT+5',nonexistent='NaT',ambiguous='NaT').dt.tz_convert('UTC'); frames.append(x[['timestamp','open','high','low','close','volume']])
            time.sleep(.10)
    if not frames: raise RuntimeError(f'{asset}: no frames')
    return write_norm(rawdir,asset,pd.concat(frames,ignore_index=True),'HISTDATA_GENERIC_ASCII_M1_BID_INDEPENDENT_PROVIDER')

def cb_one(asset,product,rawdir):
    s=requests.Session(); s.headers.update({'User-Agent':'TrailAris-independent-validation/1.1'})
    p=s.get(f'{CB}/products/{product}',timeout=30)
    if p.status_code!=200: raise RuntimeError(f'{asset}: product {product} HTTP {p.status_code}')
    rows={}; t0=START.floor('min')
    while t0<=END:
        t1=min(t0+pd.Timedelta(minutes=299),END.floor('min')); params={'granularity':60,'start':t0.isoformat(),'end':t1.isoformat()}; err=None
        for attempt in range(7):
            try:
                r=s.get(f'{CB}/products/{product}/candles',params=params,timeout=30)
                if r.status_code==429: time.sleep(1.0+attempt*.5); continue
                r.raise_for_status(); payload=r.json()
                if not isinstance(payload,list): raise RuntimeError(str(payload)[:200])
                for b in payload:
                    if isinstance(b,list) and len(b)>=6:
                        ts=int(b[0]); dt=pd.Timestamp(ts,unit='s',tz='UTC')
                        if START<=dt<=END: rows[ts]=[dt,float(b[3]),float(b[2]),float(b[1]),float(b[4]),float(b[5])]
                err=None; break
            except Exception as e: err=e; time.sleep(.5*(attempt+1))
        if err is not None: raise RuntimeError(f'{asset}: request {t0}: {err}')
        t0=t1+pd.Timedelta(minutes=1); time.sleep(.35)
    d=pd.DataFrame([rows[k] for k in sorted(rows)],columns=['timestamp','open','high','low','close','volume']); return write_norm(rawdir,asset,d,'COINBASE_EXCHANGE_OFFICIAL_M1_INDEPENDENT_PROVIDER')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rawdir',required=True); ap.add_argument('--evidence',required=True); a=ap.parse_args(); raw=Path(a.rawdir); ev=Path(a.evidence); raw.mkdir(parents=True,exist_ok=True); ev.mkdir(parents=True,exist_ok=True)
    rec=[]; errors=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut={ex.submit(hist_one,x,p,raw):(x,'HISTDATA') for x,p in HISTDATA.items()}
        for f in as_completed(fut):
            x,prov=fut[f]
            try: r=f.result(); rec.append(r); print(json.dumps(r),flush=True)
            except Exception as e: q={'asset':x,'provider':prov,'error':str(e)}; errors.append(q); print(json.dumps(q),flush=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        fut={ex.submit(cb_one,x,p,raw):(x,'COINBASE_EXCHANGE') for x,p in COINBASE.items()}
        for f in as_completed(fut):
            x,prov=fut[f]
            try: r=f.result(); rec.append(r); print(json.dumps(r),flush=True)
            except Exception as e: q={'asset':x,'provider':prov,'error':str(e)}; errors.append(q); print(json.dumps(q),flush=True)
    status={'state':'PROVIDER_DIVERSE_ACQUISITION_COMPLETE' if not errors else 'PROVIDER_DIVERSE_ACQUISITION_FAIL_CLOSED','window':{'start':str(START),'end':str(END)},'independent_routes_targeted':26,'independent_routes_passed':len(rec),'independent_routes_failed':len(errors),'primary_only_routes':PRIMARY_ONLY,'full_34_route_independent_provider_claim':False,'records':sorted(rec,key=lambda x:x['asset']),'errors':errors,'promotion_allowed':False}
    (ev/'PROVIDER_DIVERSE_ACQUISITION_STATUS.json').write_text(json.dumps(status,indent=2)); pd.DataFrame(rec).to_csv(ev/'PROVIDER_DIVERSE_COVERAGE.csv',index=False)
    if errors: raise SystemExit(2)
if __name__=='__main__': main()
