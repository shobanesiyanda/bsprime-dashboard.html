#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,datetime as dt,json,lzma,struct,subprocess,tempfile
from pathlib import Path

REC=struct.Struct('>5If')  # seconds-from-day-start, O,H,L,C integer prices, volume
PRICE_DIVISOR=1000.0       # MSFT.US/USD raw Dukascopy scale, empirically cross-checked against package output

def day_url(d:dt.date)->str:
    # Dukascopy raw path months are zero-indexed.
    return f'https://datafeed.dukascopy.com/datafeed/MSFTUSUSD/{d.year}/{d.month-1:02d}/{d.day:02d}/BID_candles_min_1.bi5'

def fetch(url:str,tmp:Path)->tuple[str,int,str]:
    p=subprocess.run(['curl','-L','--retry','4','--retry-all-errors','--retry-delay','2','--connect-timeout','20','--max-time','90','-sS','-w','%{http_code}','-o',str(tmp),url],capture_output=True,text=True)
    code=(p.stdout or '')[-3:]
    size=tmp.stat().st_size if tmp.exists() else 0
    return code,size,(p.stderr or '').strip()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',required=True);ap.add_argument('--end',required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args()
    start=dt.datetime.fromisoformat(a.start.replace('Z','+00:00'));end=dt.datetime.fromisoformat(a.end.replace('Z','+00:00'))
    if not start<end:raise RuntimeError('invalid start/end')
    a.outdir.mkdir(parents=True,exist_ok=True);rows=[];days=[];invalid=0;bad_layout=0;active_days=0;zero_volume_days=0;http_failures=0
    d=start.date();last_date=(end-dt.timedelta(microseconds=1)).date()
    while d<=last_date:
        url=day_url(d)
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'day.bi5';code,size,err=fetch(url,f)
            rec={'date':d.isoformat(),'url':url,'http_code':code,'compressed_bytes':size,'status':'NO_DATA','active_rows':0}
            if code=='200' and size>0:
                try:raw=lzma.decompress(f.read_bytes())
                except Exception as e:
                    rec['status']='DECOMPRESS_FAIL';rec['error']=str(e);days.append(rec);bad_layout+=1;d+=dt.timedelta(days=1);continue
                if len(raw)%REC.size:
                    rec['status']='LAYOUT_FAIL';rec['decompressed_bytes']=len(raw);days.append(rec);bad_layout+=1;d+=dt.timedelta(days=1);continue
                day_rows=[]
                prev_s=-60
                for off in range(0,len(raw),REC.size):
                    sec,o,h,l,c,vol=REC.unpack(raw[off:off+REC.size])
                    if sec<0 or sec>=86400 or sec%60 or sec<prev_s:
                        invalid+=1;continue
                    prev_s=sec
                    O,H,L,C=(o/PRICE_DIVISOR,h/PRICE_DIVISOR,l/PRICE_DIVISOR,c/PRICE_DIVISOR)
                    if not (H>=max(O,C) and L<=min(O,C) and H>=L and O>0 and H>0 and L>0 and C>0):
                        invalid+=1;continue
                    # Raw daily files contain zero-volume carry-forward bars outside the stock-CFD session.
                    # Admit only provider-observed active bars; no interpolation or synthetic filling.
                    if float(vol)<=0:continue
                    t=dt.datetime.combine(d,dt.time(0,0),tzinfo=dt.timezone.utc)+dt.timedelta(seconds=int(sec))
                    if not (start<=t<end):continue
                    day_rows.append([t.isoformat().replace('+00:00','Z'),O,H,L,C,C,'','',float(vol),'DUKASCOPY_OFFICIAL_RAW_M1_BID_CANDLES','MSFT.US'])
                rec['decompressed_bytes']=len(raw);rec['records']=len(raw)//REC.size;rec['active_rows']=len(day_rows)
                if day_rows:
                    rec['status']='ACTIVE';rec['first']=day_rows[0][0];rec['last']=day_rows[-1][0];active_days+=1;rows.extend(day_rows)
                else:
                    rec['status']='ZERO_VOLUME';zero_volume_days+=1
            else:
                http_failures+=1;rec['error']=err
            days.append(rec)
        d+=dt.timedelta(days=1)
    rows.sort(key=lambda r:r[0]);seen=set();ded=[];dupes=0;conflicts=0
    for r in rows:
        t=r[0];sig=tuple(r[1:5])
        if t in seen:
            dupes+=1;continue
        seen.add(t);ded.append(r)
    rows=ded
    first=rows[0][0] if rows else '';last=rows[-1][0] if rows else ''
    first_dt=dt.datetime.fromisoformat(first.replace('Z','+00:00')) if first else None;last_dt=dt.datetime.fromisoformat(last.replace('Z','+00:00')) if last else None
    coverage=bool(first_dt and last_dt and first_dt<=start+dt.timedelta(days=3) and last_dt>=end-dt.timedelta(days=4))
    # US stock-CFD minute session provides roughly 390 active bars/trading day. The threshold is deliberately
    # conservative and only guards against a sparse/partial estate; it does not manufacture missing bars.
    ok=len(rows)>=45000 and coverage and invalid==0 and bad_layout==0 and conflicts==0
    outcsv=a.outdir/'MSFT.US_M1_normalized.csv'
    if ok:
        with outcsv.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['timestamp','open','high','low','close','bid','ask','spread','volume','source','asset']);w.writerows(rows)
    evidence={'state':'MSFT_US_INDEPENDENT_RAW_M1_RECOVERY','instrument_id':'msftususd','canonical_instrument':'MSFT.US','provider_name':'MSFT.US/USD','provider_raw_symbol':'MSFTUSUSD','source':'DUKASCOPY_OFFICIAL_RAW_M1_BID_CANDLES','start':start.isoformat(),'end_exclusive':end.isoformat(),'price_divisor':PRICE_DIVISOR,'rows':len(rows),'active_days':active_days,'zero_volume_days':zero_volume_days,'http_failures':http_failures,'invalid_records':invalid,'bad_layout_days':bad_layout,'duplicate_timestamps_deduplicated':dupes,'conflicting_duplicates':conflicts,'first':first,'last':last,'full_window_coverage':coverage,'status':'PASS' if ok else 'FAIL','file':str(outcsv) if ok else '','normalization':'OFFICIAL_RAW_DAILY_M1_BID_CANDLES__LZMA__24_BYTE_BIG_ENDIAN_CANDLE_RECORDS__ZERO_VOLUME_CARRY_FORWARD_BARS_EXCLUDED__NO_INTERPOLATION__NO_SYNTHETIC_FILL','days':days}
    (a.outdir/'MSFT_RAW_RECOVERY_EVIDENCE.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({k:evidence[k] for k in ['state','rows','active_days','zero_volume_days','http_failures','invalid_records','bad_layout_days','first','last','full_window_coverage','status']},indent=2))
    if not ok:raise SystemExit(2)

if __name__=='__main__':main()
