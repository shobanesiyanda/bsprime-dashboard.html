#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
import pandas as pd
ASSETS={'BTCUSD':'BTCUSDT','ETHUSD':'ETHUSDT','SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'}
BASE='https://data.binance.vision/data/spot/monthly/klines'
def fetch(u):return urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'TrailarisResearch/QV2-50-Independent'}),timeout=120).read()
def unit_ms(v):
 v=int(float(v));return v//1000 if v>10**14 else v
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',default='2024-12-23T00:00:00Z');ap.add_argument('--end',default='2025-07-21T00:00:00Z');ap.add_argument('--outdir',type=Path,default=Path('raw_crypto'));a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
 start=pd.Timestamp(a.start);end=pd.Timestamp(a.end);months=pd.period_range(start=start.tz_localize(None).to_period('M'),end=(end-pd.Timedelta(nanoseconds=1)).tz_localize(None).to_period('M'),freq='M');evidence=[]
 for asset,sym in ASSETS.items():
  rows=[];seen=set();errors=[]
  for m in months:
   ym=str(m);stem=f'{sym}-1m-{ym}.zip';url=f'{BASE}/{sym}/1m/{stem}'
   try:
    raw=fetch(url);chk=fetch(url+'.CHECKSUM').decode().split()[0]
    if hashlib.sha256(raw).hexdigest().lower()!=chk.lower():raise RuntimeError('checksum mismatch '+stem)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:rr=csv.reader(io.StringIO(z.read(z.namelist()[0]).decode()))
    for r in rr:
     if len(r)<6:continue
     try:t=unit_ms(r[0])
     except:continue
     ts=pd.Timestamp(t,unit='ms',tz='UTC')
     if ts<start or ts>=end or t in seen:continue
     seen.add(t);o,h,l,c=map(float,r[1:5])
     if not(h>=max(o,c) and l<=min(o,c) and h>=l):continue
     rows.append([ts.isoformat().replace('+00:00','Z'),o,h,l,c,'','','',r[5],'BINANCE_OFFICIAL_SPOT_M1',asset])
   except Exception as e:errors.append(f'{stem}:{e!r}')
  rows.sort(key=lambda x:x[0]);first=pd.Timestamp(rows[0][0]) if rows else None;last=pd.Timestamp(rows[-1][0]) if rows else None;coverage=bool(rows and first<=start+pd.Timedelta(days=1) and last>=end-pd.Timedelta(days=4));ok=len(rows)>=1000 and coverage and not errors
  p=a.outdir/f'{asset}_M1_normalized.csv'
  if ok:
   with p.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['timestamp','open','high','low','close','bid','ask','spread','volume','source','asset']);w.writerows(rows)
  evidence.append({'instrument_id':sym.lower(),'canonical_instrument':asset,'provider_name':'BINANCE_OFFICIAL_SPOT_M1','rows':len(rows),'first':str(first) if first is not None else '','last':str(last) if last is not None else '','full_window_coverage':coverage,'status':'PASS' if ok else 'FAIL','file':str(p) if ok else '','errors':errors[:10]})
  print(asset,len(rows),'PASS' if ok else 'FAIL',errors[:1])
 out={'start':str(start),'end_exclusive':str(end),'assets_scheduled':5,'assets_pass':sum(x['status']=='PASS' for x in evidence),'evidence':evidence};(a.outdir/'ACQUISITION_EVIDENCE_CRYPTO.json').write_text(json.dumps(out,indent=2));print(json.dumps({'assets_scheduled':5,'assets_pass':out['assets_pass']},indent=2));raise SystemExit(0 if out['assets_pass']==5 else 2)
if __name__=='__main__':main()
