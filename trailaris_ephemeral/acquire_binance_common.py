#!/usr/bin/env python3
import csv,hashlib,io,json,urllib.request,zipfile
from datetime import date,timedelta
from pathlib import Path
OUT=Path('trailaris_ephemeral/raw_common');OUT.mkdir(parents=True,exist_ok=True)
BASE='https://data.binance.vision/data/spot';ASSETS={'BTCUSD':'BTCUSDT','ETHUSD':'ETHUSDT','SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'}
START=date(2025,8,15);END=date(2026,8,12);cov={}
def fetch(u):
 r=urllib.request.Request(u,headers={'User-Agent':'TrailarisResearch/0.15'});return urllib.request.urlopen(r,timeout=40).read()
def rows(raw):
 with zipfile.ZipFile(io.BytesIO(raw)) as z:return list(csv.reader(io.StringIO(z.read(z.namelist()[0]).decode())))
def ms(x):
 v=int(float(x));return v//1000 if v>10**14 else v
def addzip(w,url):
 raw=fetch(url);chk=fetch(url+'.CHECKSUM').decode().split()[0]
 if hashlib.sha256(raw).hexdigest().lower()!=chk.lower():raise RuntimeError('checksum mismatch')
 n=0;first=last=None
 for r in rows(raw):
  if len(r)<6:continue
  t=ms(r[0]);d=date.fromtimestamp(t/1000)
  if START<=d<=END:w.writerow([t,r[1],r[2],r[3],r[4],r[5]]);n+=1;first=t if first is None else first;last=t
 return n,first,last
for a,s in ASSETS.items():
 p=OUT/f'{a}.csv';n=0;first=last=None;errs=[]
 with p.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['timestamp','open','high','low','close','volume'])
  y,m=2025,8
  while (y,m)<=(2026,7):
   stem=f'{s}-5m-{y:04d}-{m:02d}.zip';url=f'{BASE}/monthly/klines/{s}/5m/{stem}'
   try:
    k,fi,la=addzip(w,url);n+=k;first=fi if first is None and fi is not None else first;last=la if la is not None else last
   except Exception as e:errs.append(f'{stem}:{e!r}')
   m+=1
   if m==13:y+=1;m=1
  d=date(2026,8,1)
  while d<=END:
   stem=f'{s}-5m-{d.isoformat()}.zip';url=f'{BASE}/daily/klines/{s}/5m/{stem}'
   try:
    k,fi,la=addzip(w,url);n+=k;first=fi if first is None and fi is not None else first;last=la if la is not None else last
   except Exception as e:errs.append(f'{stem}:{e!r}')
   d+=timedelta(days=1)
 cov[a]={'source':'BINANCE_OFFICIAL_SPOT','rows':n,'first':first,'last':last,'status':'OK' if n>100 else 'FAIL','error_count':len(errs),'errors':errs[:5]};print(a,n,len(errs))
Path('trailaris_ephemeral/coverage_common_binance.json').write_text(json.dumps(cov,indent=2))
