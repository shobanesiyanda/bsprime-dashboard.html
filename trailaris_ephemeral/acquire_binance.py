#!/usr/bin/env python3
import csv, hashlib, io, json, os, urllib.request, urllib.error, zipfile
from datetime import date, timedelta
from pathlib import Path

OUT=Path('trailaris_ephemeral/raw'); OUT.mkdir(parents=True,exist_ok=True)
BASE='https://data.binance.vision/data/spot'
ASSETS={'BTCUSD':'BTCUSDT','ETHUSD':'ETHUSDT','SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'}
START=date(2022,1,1); END=date(2026,8,12)
coverage={}

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'TrailarisResearch/0.15'})
    with urllib.request.urlopen(req,timeout=40) as r:return r.read()

def decode_zip(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist()
        if not names:return []
        text=z.read(names[0]).decode('utf-8')
    return list(csv.reader(io.StringIO(text)))

def ts_ms(x):
    v=int(float(x))
    if v>10**14: v//=1000 # Binance spot archives use microseconds from 2025 onward
    return v

def monthly_iter(start,end):
    y,m=start.year,start.month
    while (y,m)<=(end.year,end.month):
        yield y,m
        m+=1
        if m==13:y+=1;m=1

def acquire(asset,symbol):
    dest=OUT/f'{asset}.csv'; rows=0; first=last=None; errors=[]
    with dest.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['timestamp','open','high','low','close','volume'])
        # complete months through July 2026
        for y,m in monthly_iter(START,date(2026,7,31)):
            stem=f'{symbol}-5m-{y:04d}-{m:02d}.zip'
            url=f'{BASE}/monthly/klines/{symbol}/5m/{stem}'
            try:
                raw=fetch(url); chk=fetch(url+'.CHECKSUM').decode().split()[0]
                if hashlib.sha256(raw).hexdigest().lower()!=chk.lower(): raise RuntimeError('checksum mismatch')
                for r in decode_zip(raw):
                    if len(r)<6:continue
                    t=ts_ms(r[0]); d=date.fromtimestamp(t/1000)
                    if d<START or d>END:continue
                    w.writerow([t,r[1],r[2],r[3],r[4],r[5]]);rows+=1;first=t if first is None else first;last=t
            except Exception as e: errors.append(f'{stem}:{e!r}')
        # daily files for Aug 1-12, 2026
        d=date(2026,8,1)
        while d<=END:
            ds=d.isoformat(); stem=f'{symbol}-5m-{ds}.zip';url=f'{BASE}/daily/klines/{symbol}/5m/{stem}'
            try:
                raw=fetch(url);chk=fetch(url+'.CHECKSUM').decode().split()[0]
                if hashlib.sha256(raw).hexdigest().lower()!=chk.lower():raise RuntimeError('checksum mismatch')
                for r in decode_zip(raw):
                    if len(r)<6:continue
                    t=ts_ms(r[0]);w.writerow([t,r[1],r[2],r[3],r[4],r[5]]);rows+=1;first=t if first is None else first;last=t
            except Exception as e: errors.append(f'{stem}:{e!r}')
            d+=timedelta(days=1)
    coverage[asset]={'source':'BINANCE_OFFICIAL_SPOT','symbol':symbol,'rows':rows,'first':first,'last':last,'status':'OK' if rows>100 else 'FAIL','errors':errors[:10],'error_count':len(errors)}
    print(asset,rows,'errors',len(errors))

for a,s in ASSETS.items(): acquire(a,s)
Path('trailaris_ephemeral/coverage_binance.json').write_text(json.dumps(coverage,indent=2))
