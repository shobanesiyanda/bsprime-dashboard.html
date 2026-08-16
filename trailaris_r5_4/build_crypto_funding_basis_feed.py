#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

MAP={'BTCUSD':'BTCUSDT','ETHUSD':'ETHUSDT','SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'}
BASE='https://data.binance.vision/data/futures/um'
START=pd.Timestamp('2026-05-04T00:00:00Z');END=pd.Timestamp('2026-08-14T23:59:59Z')
# Binance public fundingRate is a monthly archive. At 16 Aug 2026 the completed
# May-July archives are the causal funding evidence boundary; never synthesize
# August settlements that are not yet in the official archive.
FUNDING_MONTHS=[(2026,5),(2026,6),(2026,7)]

def _zip_bytes(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Trailaris-R5.4-evidence/1.0'})
    with urllib.request.urlopen(req,timeout=90) as r:return r.read()

def _read_zip(url,header=None):
    z=zipfile.ZipFile(io.BytesIO(_zip_bytes(url)));names=[n for n in z.namelist() if n.lower().endswith('.csv')]
    if not names:raise ValueError('zip has no csv: '+url)
    return pd.read_csv(z.open(names[0]),header=header)

def _ts_unit(s):
    x=pd.to_numeric(s,errors='coerce').dropna()
    if x.empty:return 'ms'
    med=x.abs().median();return 'us' if med>1e14 else ('ms' if med>1e11 else 's')

def parse_funding_archive(raw):
    # Current Binance funding archives are headered; retain a positional fallback
    # so the parser remains fail-closed if archive formatting differs.
    d=raw.copy()
    cols={str(c).strip().lower():c for c in d.columns}
    if 'calc_time' in cols:
        tc=cols['calc_time'];rc=cols.get('last_funding_rate') or cols.get('funding_rate')
        if rc is None:raise ValueError('funding archive missing rate column')
        t=pd.to_numeric(d[tc],errors='coerce');r=pd.to_numeric(d[rc],errors='coerce')
    else:
        # Re-read positional layout is expected to be calc_time, interval_hours, rate.
        if d.shape[1]<3:raise ValueError('unexpected funding archive layout')
        t=pd.to_numeric(d.iloc[:,0],errors='coerce');r=pd.to_numeric(d.iloc[:,-1],errors='coerce')
    out=pd.DataFrame({'timestamp':pd.to_datetime(t,unit=_ts_unit(t),utc=True,errors='coerce'),'funding_rate':r})
    return out.dropna().drop_duplicates('timestamp').sort_values('timestamp')

def funding_month(symbol,y,m):
    ym=f'{y}-{m:02d}';url=f'{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip'
    try:return parse_funding_archive(_read_zip(url,header=0))
    except Exception:
        return parse_funding_archive(_read_zip(url,header=None))

def funding_history(symbol):
    parts=[]
    for y,m in FUNDING_MONTHS:
        parts.append(funding_month(symbol,y,m))
    d=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp')
    return d[(d.timestamp>=START)&(d.timestamp<=END)]

def premium_month(symbol,y,m):
    ym=f'{y}-{m:02d}';url=f'{BASE}/monthly/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{ym}.zip'
    return parse_kline(_read_zip(url,header=None))

def premium_day(symbol,day):
    ds=day.strftime('%Y-%m-%d');url=f'{BASE}/daily/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{ds}.zip'
    return parse_kline(_read_zip(url,header=None))

def parse_kline(d):
    if d.shape[1]<5:raise ValueError('unexpected premium kline columns')
    t=pd.to_numeric(d.iloc[:,0],errors='coerce')
    out=pd.DataFrame({'timestamp':pd.to_datetime(t,unit=_ts_unit(t),utc=True,errors='coerce'),'premium':pd.to_numeric(d.iloc[:,4],errors='coerce')})
    return out.dropna().drop_duplicates('timestamp').sort_values('timestamp')

def premium_history(symbol):
    parts=[]
    for y,m in FUNDING_MONTHS:parts.append(premium_month(symbol,y,m))
    # August basis is available daily, but without an archived August funding settlement
    # it is retained only for provenance and does not create funding-basis signals.
    for day in pd.date_range('2026-08-01','2026-08-14',freq='D'):
        try:parts.append(premium_day(symbol,day))
        except Exception as e:print('PREMIUM_DAY_MISS',symbol,day.date(),repr(e))
    d=pd.concat(parts,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp')
    return d[(d.timestamp>=START)&(d.timestamp<=END)]

def causal_z(s,window,minp):
    past=s.shift(1);mu=past.rolling(window,min_periods=minp).mean();sd=past.rolling(window,min_periods=minp).std().replace(0,np.nan)
    return ((s-mu)/sd).clip(-5,5)

def build_asset(asset,symbol):
    f=funding_history(symbol);p=premium_history(symbol)
    if f.empty or p.empty:return pd.DataFrame(),{'asset':asset,'funding_rows':len(f),'premium_rows':len(p),'factor_rows':0,'state':'EVIDENCE_MISSING'}
    m=pd.merge_asof(f.sort_values('timestamp'),p.sort_values('timestamp'),on='timestamp',direction='backward',tolerance=pd.Timedelta(minutes=2)).dropna()
    m['funding_z']=causal_z(m.funding_rate,20,8);m['premium_z']=causal_z(m.premium,30,12)
    # Pre-specified convergence hypothesis; this is a research signal, not assumed alpha.
    m['funding_score']=-(.55*m.funding_z+.45*m.premium_z)
    m=m.dropna(subset=['funding_score']);m['funding_score']=m.funding_score.clip(-4,4)
    m['asset']=asset;m['orderflow_score']=np.nan;m['carry_score']=np.nan;m['event_score']=np.nan
    m['evidence_state']='VERIFIED_BINANCE_PUBLIC_ARCHIVE_FUNDING_BASIS_CAUSAL'
    m['source_funding']='BINANCE_USDM_MONTHLY_FUNDING_RATE_ARCHIVE';m['source_basis']='BINANCE_USDM_PREMIUM_INDEX_KLINES_ARCHIVE'
    out=m[['timestamp','asset','orderflow_score','carry_score','funding_score','event_score','evidence_state','funding_rate','premium','funding_z','premium_z','source_funding','source_basis']]
    st={'asset':asset,'symbol':symbol,'funding_rows':len(f),'premium_rows':len(p),'factor_rows':len(out),'trigger_rows_abs_ge_070':int((out.funding_score.abs()>=.70).sum()),'first':str(out.timestamp.min()) if len(out) else None,'last':str(out.timestamp.max()) if len(out) else None,'funding_archive_boundary':str(f.timestamp.max()) if len(f) else None,'state':'PASS' if len(out) else 'NO_FACTOR_ROWS'}
    return out,st

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--audit',required=True);a=ap.parse_args();rows=[];audit=[]
    for asset,symbol in MAP.items():
        d,s=build_asset(asset,symbol);audit.append(s)
        if len(d):rows.append(d)
    feed=pd.concat(rows,ignore_index=True).sort_values(['timestamp','asset']) if rows else pd.DataFrame()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);feed.to_csv(a.out,index=False);pd.DataFrame(audit).to_csv(a.audit,index=False)
    summary={'assets_required':5,'assets_pass':int(sum(x['state']=='PASS' for x in audit)),'factor_rows':len(feed),'trigger_rows_abs_ge_070':int((feed.funding_score.abs()>=.70).sum()) if len(feed) else 0,'funding_archive_window':'2026-05-04 through latest completed official monthly archive (expected 2026-07-31 at run date)','august_policy':'NO_SYNTHETIC_FUNDING; no August funding-basis trigger without official archived settlement evidence','causality':'funding settlement and premium observed at/before timestamp; z-score baselines use shifted prior observations only','evidence_state':'VERIFIED_BINANCE_PUBLIC_ARCHIVE_FUNDING_BASIS_CAUSAL'}
    Path(str(a.audit)+'.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));assert summary['assets_pass']==5

if __name__=='__main__':main()
