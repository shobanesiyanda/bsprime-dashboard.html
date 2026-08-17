#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

START = pd.Timestamp('2026-01-19T00:00:00Z')
END = pd.Timestamp('2026-08-14T23:59:59Z')
COLUMNS = ['timestamp','open','high','low','close','bid','ask','spread','volume','source','asset']

# HistData symbols documented by HistData for M1 Generic ASCII feeds.
HISTDATA = {
    'EURUSD':'EURUSD','GBPUSD':'GBPUSD','USDJPY':'USDJPY','AUDUSD':'AUDUSD','USDCAD':'USDCAD','USDCHF':'USDCHF',
    'NZDUSD':'NZDUSD','EURJPY':'EURJPY','GBPJPY':'GBPJPY','EURGBP':'EURGBP','AUDJPY':'AUDJPY','CADJPY':'CADJPY',
    'GBPCHF':'GBPCHF','XAUUSD':'XAUUSD','XAGUSD':'XAGUSD','US100':'NSXUSD','US500':'SPXUSD','GER40':'GRXEUR',
    'UK100':'UKXGBP','JP225':'JPXJPY','WTI':'WTIUSD','BRENT':'BCOUSD',
}

# Coinbase Exchange public candles. Each product is probed; unsupported/incomplete products fail closed.
COINBASE = {
    'BTCUSD':'BTC-USD','ETHUSD':'ETH-USD','SOLUSD':'SOL-USD','BNBUSD':'BNB-USD','XRPUSD':'XRP-USD',
}

PRIMARY_ONLY = ['US30','NATGAS','V75','V50','V100','V25','V10']

HIST_PREFIX = 'https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/'
HIST_POST = 'https://www.histdata.com/get.php'
CB = 'https://api.exchange.coinbase.com'


def write_normalized(rawdir: Path, asset: str, df: pd.DataFrame, source: str) -> dict:
    if df.empty:
        raise RuntimeError(f'{asset}: no independent rows')
    d = df.copy()
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True, errors='coerce')
    d = d.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp', keep='last')
    d = d[(d.timestamp >= START) & (d.timestamp <= END)].copy()
    for c in ['open','high','low','close','volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['open','high','low','close'])
    d['bid'] = d['close']
    d['ask'] = ''
    d['spread'] = ''
    d['source'] = source
    d['asset'] = asset
    d = d[COLUMNS]
    if len(d) < 1000:
        raise RuntimeError(f'{asset}: independent history too small: {len(d)}')
    first, last = d.timestamp.min(), d.timestamp.max()
    if first > START + pd.Timedelta(days=1):
        raise RuntimeError(f'{asset}: independent history starts too late: {first}')
    if last < END - pd.Timedelta(days=2):
        raise RuntimeError(f'{asset}: independent history ends too early: {last}')
    out = rawdir / f'{asset}_M1_normalized.csv'
    d.to_csv(out, index=False, date_format='%Y-%m-%dT%H:%M:%S.%fZ')
    med = d.timestamp.diff().dropna().dt.total_seconds().median()
    return {'asset':asset,'source':source,'rows':int(len(d)),'first':str(first),'last':str(last),'median_gap_seconds':None if pd.isna(med) else float(med),'status':'PASS'}


def hist_download_month(pair: str, year: int, month: int, session: requests.Session) -> bytes:
    referer = f'{HIST_PREFIX}{pair.lower()}/{year}/{month}'
    r1 = session.get(referer, timeout=45, allow_redirects=True)
    r1.raise_for_status()
    soup = BeautifulSoup(r1.content, 'html.parser')
    node = soup.find('input', {'id':'tk'})
    if not node or not node.attrs.get('value'):
        raise RuntimeError(f'HistData token missing for {pair} {year}-{month:02d}')
    token = node.attrs['value']
    data = {'tk':token,'date':str(year),'datemonth':f'{year}{month:02d}','platform':'ASCII','timeframe':'M1','fxpair':pair.upper()}
    headers = {'Referer':referer,'Origin':'https://www.histdata.com','Content-Type':'application/x-www-form-urlencoded','User-Agent':'Mozilla/5.0'}
    r = session.post(HIST_POST, data=data, headers=headers, timeout=90)
    r.raise_for_status()
    if not r.content.startswith(b'PK'):
        raise RuntimeError(f'HistData non-ZIP response for {pair} {year}-{month:02d}, bytes={len(r.content)}')
    return r.content


def acquire_hist(asset: str, pair: str, rawdir: Path) -> dict:
    frames=[]
    with requests.Session() as s:
        for month in range(1,9):
            blob = hist_download_month(pair, 2026, month, s)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                names=[n for n in z.namelist() if n.lower().endswith('.csv')]
                if not names:
                    raise RuntimeError(f'{asset}: HistData ZIP has no CSV for month {month}')
                for name in names:
                    txt=z.read(name).decode('utf-8',errors='ignore')
                    if not txt.strip():
                        continue
                    x=pd.read_csv(io.StringIO(txt),sep=';',header=None,names=['dt','open','high','low','close','volume'],engine='python')
                    # HistData Generic ASCII timestamps are EST without daylight-saving adjustment: UTC-05:00 fixed.
                    local=pd.to_datetime(x.dt.astype(str).str.strip(),format='%Y%m%d %H%M%S',errors='coerce')
                    x['timestamp']=local.dt.tz_localize('Etc/GMT+5',nonexistent='NaT',ambiguous='NaT').dt.tz_convert('UTC')
                    frames.append(x[['timestamp','open','high','low','close','volume']])
            time.sleep(0.15)
    if not frames:
        raise RuntimeError(f'{asset}: no HistData frames')
    return write_normalized(rawdir,asset,pd.concat(frames,ignore_index=True),'HISTDATA_GENERIC_ASCII_M1_BID_INDEPENDENT_PROVIDER')


def cb_one(asset: str, product: str, rawdir: Path) -> dict:
    s=requests.Session(); s.headers.update({'User-Agent':'TrailAris-independent-validation/1.0'})
    probe=s.get(f'{CB}/products/{product}',timeout=30)
    if probe.status_code != 200:
        raise RuntimeError(f'{asset}: Coinbase product unavailable {product}: HTTP {probe.status_code}')
    rows={}
    t0=START.floor('min')
    # Coinbase documents a maximum of 300 candles/request; use 299-minute windows.
    while t0 <= END:
        t1=min(t0+pd.Timedelta(minutes=299),END.floor('min'))
        params={'granularity':60,'start':t0.isoformat(),'end':t1.isoformat()}
        last_err=None
        for attempt in range(5):
            try:
                r=s.get(f'{CB}/products/{product}/candles',params=params,timeout=30)
                if r.status_code==429:
                    time.sleep(1.0+attempt); continue
                r.raise_for_status(); payload=r.json()
                if not isinstance(payload,list):
                    raise RuntimeError(str(payload)[:300])
                for b in payload:
                    if not isinstance(b,list) or len(b)<6: continue
                    ts=int(b[0]); dt=pd.Timestamp(ts,unit='s',tz='UTC')
                    if START <= dt <= END:
                        # Coinbase schema: time, low, high, open, close, volume
                        rows[ts]=[dt,float(b[3]),float(b[2]),float(b[1]),float(b[4]),float(b[5])]
                last_err=None; break
            except Exception as e:
                last_err=e; time.sleep(0.5*(attempt+1))
        if last_err is not None:
            raise RuntimeError(f'{asset}: Coinbase candle request failed at {t0}: {last_err}')
        t0=t1+pd.Timedelta(minutes=1)
        time.sleep(0.55)
    d=pd.DataFrame([rows[k] for k in sorted(rows)],columns=['timestamp','open','high','low','close','volume'])
    return write_normalized(rawdir,asset,d,'COINBASE_EXCHANGE_OFFICIAL_M1_INDEPENDENT_PROVIDER')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rawdir',required=True)
    ap.add_argument('--evidence',required=True)
    args=ap.parse_args()
    rawdir=Path(args.rawdir); rawdir.mkdir(parents=True,exist_ok=True)
    evidence=Path(args.evidence); evidence.mkdir(parents=True,exist_ok=True)

    records=[]; errors=[]
    # HistData requests are deliberately conservative to avoid hammering the source.
    for asset,pair in HISTDATA.items():
        try:
            rec=acquire_hist(asset,pair,rawdir); records.append(rec); print(json.dumps(rec))
        except Exception as e:
            errors.append({'asset':asset,'provider':'HISTDATA','error':str(e)}); print(json.dumps(errors[-1]))

    # Crypto provider requests can run concurrently by asset while preserving per-asset throttling.
    with ThreadPoolExecutor(max_workers=3) as ex:
        fut={ex.submit(cb_one,a,p,rawdir):(a,p) for a,p in COINBASE.items()}
        for f in as_completed(fut):
            a,p=fut[f]
            try:
                rec=f.result(); records.append(rec); print(json.dumps(rec))
            except Exception as e:
                errors.append({'asset':a,'provider':'COINBASE_EXCHANGE','product':p,'error':str(e)}); print(json.dumps(errors[-1]))

    status={
        'state':'PROVIDER_DIVERSE_ACQUISITION_COMPLETE' if not errors else 'PROVIDER_DIVERSE_ACQUISITION_FAIL_CLOSED',
        'window':{'start':str(START),'end':str(END)},
        'independent_routes_targeted':len(HISTDATA)+len(COINBASE),
        'independent_routes_passed':len(records),
        'independent_routes_failed':len(errors),
        'primary_only_routes':PRIMARY_ONLY,
        'full_34_route_independent_provider_claim':False,
        'reason_full_independence_unavailable':'US30/NATGAS are not covered by this independent feed set; V10/V25/V50/V75/V100 are proprietary Deriv synthetic markets and require target-server/broker extraction rather than a nonexistent independent market source.',
        'records':sorted(records,key=lambda x:x['asset']),
        'errors':errors,
        'promotion_allowed':False,
    }
    (evidence/'PROVIDER_DIVERSE_ACQUISITION_STATUS.json').write_text(json.dumps(status,indent=2))
    pd.DataFrame(records).to_csv(evidence/'PROVIDER_DIVERSE_COVERAGE.csv',index=False)
    if errors:
        raise SystemExit(2)

if __name__=='__main__':
    main()
