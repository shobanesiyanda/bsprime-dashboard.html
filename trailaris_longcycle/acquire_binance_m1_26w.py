#!/usr/bin/env python3
import csv,hashlib,io,json,urllib.request,zipfile
from datetime import date,timedelta,datetime,timezone
from pathlib import Path
OUT=Path('trailaris_longcycle/raw_m1');OUT.mkdir(parents=True,exist_ok=True)
BASE='https://data.binance.vision/data/spot';ASSETS={'BTCUSD':'BTCUSDT','ETHUSD':'ETHUSDT','SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'}
START=date(2026,1,19);END=date(2026,8,14);cov={}
def fetch(u):return urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'TrailarisResearch/R5.4-26W'}),timeout=90).read()
def ziprows(raw):
 with zipfile.ZipFile(io.BytesIO(raw)) as z:return list(csv.reader(io.StringIO(z.read(z.namelist()[0]).decode())))
def ms(x):
 v=int(float(x));return v//1000 if v>10**14 else v
def iso(t):return datetime.fromtimestamp(t/1000,tz=timezone.utc).isoformat().replace('+00:00','Z')
def addzip(w,url,asset):
 raw=fetch(url);chk=fetch(url+'.CHECKSUM').decode().split()[0]
 if hashlib.sha256(raw).hexdigest().lower()!=chk.lower():raise RuntimeError('checksum mismatch')
 n=0;first=last=None
 for r in ziprows(raw):
  if len(r)<6:continue
  t=ms(r[0]);d=datetime.fromtimestamp(t/1000,tz=timezone.utc).date()
  if START<=d<=END:w.writerow([iso(t),r[1],r[2],r[3],r[4],'','','',r[5],'BINANCE_OFFICIAL_SPOT_M1',asset]);n+=1;first=t if first is None else first;last=t
 return n,first,last
for asset,sym in ASSETS.items():
 p=OUT/f'{asset}_M1_normalized.csv';n=0;first=last=None;errs=[]
 with p.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['timestamp','open','high','low','close','bid','ask','spread','volume','source','asset'])
  for m in range(1,8):
   stem=f'{sym}-1m-2026-{m:02d}.zip';url=f'{BASE}/monthly/klines/{sym}/1m/{stem}'
   try:k,fi,la=addzip(w,url,asset);n+=k;first=fi if first is None and fi else first;last=la or last
   except Exception as e:errs.append(f'{stem}:{e!r}')
  d=date(2026,8,1)
  while d<=END:
   stem=f'{sym}-1m-{d.isoformat()}.zip';url=f'{BASE}/daily/klines/{sym}/1m/{stem}'
   try:k,fi,la=addzip(w,url,asset);n+=k;first=fi if first is None and fi else first;last=la or last
   except Exception as e:errs.append(f'{stem}:{e!r}')
   d+=timedelta(days=1)
 cov[asset]={'source':'BINANCE_OFFICIAL_SPOT_M1','rows':n,'first':first,'last':last,'status':'OK' if n>100 else 'FAIL','errors':errs[:20]};print(asset,n,len(errs))
Path('trailaris_longcycle/coverage_m1_binance.json').write_text(json.dumps(cov,indent=2))
if any(x['status']!='OK' for x in cov.values()):raise SystemExit(2)
