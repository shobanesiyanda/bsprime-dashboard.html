#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,datetime as dt,json,lzma,struct,time,urllib.error,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

# Dukascopy minute-candle BI5 layout: seconds, open, close, low, high, volume.
REC=struct.Struct('>5If')
PRICE_DIVISOR=1000.0

def day_url(d:dt.date)->str:
    return f'https://datafeed.dukascopy.com/datafeed/MSFTUSUSD/{d.year}/{d.month-1:02d}/{d.day:02d}/BID_candles_min_1.bi5'

def fetch_day(d:dt.date,start:dt.datetime,end:dt.datetime):
    url=day_url(d);last=''
    for attempt in range(1,8):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Trailaris-QV2-Independent-Validation/1.0'})
            with urllib.request.urlopen(req,timeout=45) as r: blob=r.read();code=int(getattr(r,'status',200))
            raw=lzma.decompress(blob,format=lzma.FORMAT_AUTO)
            if len(raw)%REC.size: return {'date':d.isoformat(),'url':url,'status':'LAYOUT_FAIL','compressed_bytes':len(blob),'decompressed_bytes':len(raw),'rows':[],'error':'record alignment'}
            rows=[];invalid=0;prev_s=-60
            for off in range(0,len(raw),REC.size):
                sec,o,c,l,h,vol=REC.unpack_from(raw,off)
                if sec>=86400 or sec%60 or sec<prev_s: invalid+=1;continue
                prev_s=sec
                O,C,L,H=(o/PRICE_DIVISOR,c/PRICE_DIVISOR,l/PRICE_DIVISOR,h/PRICE_DIVISOR)
                if not (H>=max(O,C) and L<=min(O,C) and H>=L and min(O,H,L,C)>0): invalid+=1;continue
                if float(vol)<=0: continue
                t=dt.datetime.combine(d,dt.time(0,0),tzinfo=dt.timezone.utc)+dt.timedelta(seconds=int(sec))
                if start<=t<end: rows.append([t.isoformat().replace('+00:00','Z'),O,H,L,C,C,'','',float(vol),'DUKASCOPY_OFFICIAL_RAW_M1_BID_CANDLES','MSFT.US'])
            return {'date':d.isoformat(),'url':url,'status':'ACTIVE' if rows else 'ZERO_VOLUME','http_code':code,'compressed_bytes':len(blob),'decompressed_bytes':len(raw),'records':len(raw)//REC.size,'active_rows':len(rows),'invalid_records':invalid,'rows':rows,'first':rows[0][0] if rows else '', 'last':rows[-1][0] if rows else ''}
        except urllib.error.HTTPError as e:
            if e.code==404: return {'date':d.isoformat(),'url':url,'status':'NO_DATA_404','http_code':404,'rows':[]}
            last=f'HTTP {e.code}: {e}'
        except Exception as e: last=str(e)
        time.sleep(min(20,attempt*2))
    return {'date':d.isoformat(),'url':url,'status':'FETCH_FAIL','rows':[],'error':last}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',required=True);ap.add_argument('--end',required=True);ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    start=dt.datetime.fromisoformat(a.start.replace('Z','+00:00'));end=dt.datetime.fromisoformat(a.end.replace('Z','+00:00'))
    if not start<end:raise RuntimeError('invalid start/end')
    a.outdir.mkdir(parents=True,exist_ok=True)
    d=start.date();last=(end-dt.timedelta(microseconds=1)).date();days=[]
    while d<=last:
        if d.weekday()<5: days.append(d)
        d+=dt.timedelta(days=1)
    results=[]
    with ThreadPoolExecutor(max_workers=max(1,min(a.workers,16))) as ex:
        fut={ex.submit(fetch_day,d,start,end):d for d in days}
        for f in as_completed(fut):
            r=f.result();results.append(r);print(json.dumps({k:r.get(k) for k in ('date','status','active_rows','invalid_records','error') if k in r}))
    results.sort(key=lambda r:r['date'])
    rows=[];invalid=0;fetch_failures=0;layout_failures=0;active_days=0;no_data_days=0
    for r in results:
        rows.extend(r.pop('rows',[]));invalid+=int(r.get('invalid_records',0))
        if r['status']=='ACTIVE':active_days+=1
        elif r['status']=='FETCH_FAIL':fetch_failures+=1
        elif r['status']=='LAYOUT_FAIL':layout_failures+=1
        elif r['status'] in ('NO_DATA_404','ZERO_VOLUME'):no_data_days+=1
    rows.sort(key=lambda r:r[0]);ded=[];seen={};dupes=0;conflicts=0
    for r in rows:
        t=r[0];sig=tuple(r[1:5])
        if t in seen:
            dupes+=1
            if seen[t]!=sig:conflicts+=1
            continue
        seen[t]=sig;ded.append(r)
    rows=ded
    first=rows[0][0] if rows else '';last_ts=rows[-1][0] if rows else ''
    first_dt=dt.datetime.fromisoformat(first.replace('Z','+00:00')) if first else None;last_dt=dt.datetime.fromisoformat(last_ts.replace('Z','+00:00')) if last_ts else None
    coverage=bool(first_dt and last_dt and first_dt<=start+dt.timedelta(days=3) and last_dt>=end-dt.timedelta(days=4))
    ok=len(rows)>=45000 and coverage and invalid==0 and fetch_failures==0 and layout_failures==0 and conflicts==0
    outcsv=a.outdir/'MSFT.US_M1_normalized.csv'
    if ok:
        with outcsv.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['timestamp','open','high','low','close','bid','ask','spread','volume','source','asset']);w.writerows(rows)
    evidence={'state':'MSFT_US_INDEPENDENT_RAW_M1_RECOVERY','instrument_id':'msftususd','canonical_instrument':'MSFT.US','provider_name':'MSFT.US/USD','provider_raw_symbol':'MSFTUSUSD','source':'DUKASCOPY_OFFICIAL_RAW_M1_BID_CANDLES','start':start.isoformat(),'end_exclusive':end.isoformat(),'price_divisor':PRICE_DIVISOR,'record_layout':'>IIIIIf = seconds,open,close,low,high,volume','rows':len(rows),'active_days':active_days,'no_data_or_zero_volume_weekdays':no_data_days,'fetch_failures':fetch_failures,'invalid_records':invalid,'bad_layout_days':layout_failures,'duplicate_timestamps_deduplicated':dupes,'conflicting_duplicates':conflicts,'first':first,'last':last_ts,'full_window_coverage':coverage,'status':'PASS' if ok else 'FAIL','file':str(outcsv) if ok else '','normalization':'OFFICIAL_RAW_DAILY_M1_BID_CANDLES__LZMA__24_BYTE_BIG_ENDIAN__CORRECT_O_C_L_H_FIELD_ORDER__ZERO_VOLUME_CARRY_FORWARD_BARS_EXCLUDED__NO_INTERPOLATION__NO_SYNTHETIC_FILL','days':results}
    (a.outdir/'MSFT_RAW_RECOVERY_EVIDENCE.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({k:evidence[k] for k in ['state','rows','active_days','no_data_or_zero_volume_weekdays','fetch_failures','invalid_records','bad_layout_days','first','last','full_window_coverage','status']},indent=2))
    if not ok:raise SystemExit(2)

if __name__=='__main__':main()
