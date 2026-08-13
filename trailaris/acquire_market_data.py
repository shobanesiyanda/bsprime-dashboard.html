#!/usr/bin/env python3
import asyncio, io, json, subprocess, zipfile
from pathlib import Path
import pandas as pd
import requests

START='2026-05-04'; END='2026-08-13'
OUT=Path('trailaris/raw'); OUT.mkdir(parents=True,exist_ok=True)
DUKA={
'EURUSD':'eurusd','GBPUSD':'gbpusd','USDJPY':'usdjpy','AUDUSD':'audusd','USDCAD':'usdcad','USDCHF':'usdchf','NZDUSD':'nzdusd',
'EURJPY':'eurjpy','GBPJPY':'gbpjpy','EURGBP':'eurgbp','AUDJPY':'audjpy','CADJPY':'cadjpy','GBPCHF':'gbpchf','XAUUSD':'xauusd','XAGUSD':'xagusd',
'US100':'usatechidxusd','US500':'usa500idxusd','US30':'usa30idxusd','GER40':'deuidxeur','UK100':'gbridxgbp','JP225':'jpnidxjpy',
'WTI':'lightcmdusd','BRENT':'brentcmdusd','NATGAS':'gascmdusd'}
CRYPTO={'BTCUSD':'BTCUSDT','ETHUSD':'ETHUSDT','SOLUSD':'SOLUSDT','BNBUSD':'BNBUSDT','XRPUSD':'XRPUSDT'}
SYNTH={'V75':'R_75','V50':'R_50','V100':'R_100','V25':'R_25','V10':'R_10'}


def normalize(df):
    df=df.copy(); df.columns=[str(c).strip().lower() for c in df.columns]
    tc=next((c for c in ('timestamp','time','datetime','date') if c in df.columns),None)
    if not tc: raise ValueError('timestamp column missing '+str(list(df.columns)))
    ts=df[tc]
    if pd.api.types.is_numeric_dtype(ts):
        x=float(pd.to_numeric(ts,errors='coerce').dropna().iloc[0]); unit='ms' if x>1e11 else 's'
        df['timestamp']=pd.to_datetime(ts,unit=unit,utc=True)
    else: df['timestamp']=pd.to_datetime(ts,utc=True,errors='coerce')
    for c in ('open','high','low','close','volume'):
        if c not in df.columns: raise ValueError('missing '+c)
        df[c]=pd.to_numeric(df[c],errors='coerce')
    return df[['timestamp','open','high','low','close','volume']].dropna(subset=['timestamp','open','high','low','close']).sort_values('timestamp').drop_duplicates('timestamp')


def dukascopy(asset,inst):
    work=OUT/('_'+asset); work.mkdir(exist_ok=True)
    cp=subprocess.run(['npx','--yes','dukascopy-node','-i',inst,'-from',START,'-to',END,'-t','m1','-f','csv'],cwd=work,capture_output=True,text=True,timeout=900)
    if cp.returncode: raise RuntimeError((cp.stderr+' '+cp.stdout)[-1500:])
    files=list(work.glob('*.csv'))
    if not files: raise RuntimeError('no csv output')
    d=normalize(pd.read_csv(max(files,key=lambda p:p.stat().st_size)))
    d.to_csv(OUT/(asset+'.csv.gz'),index=False,compression='gzip')
    return len(d)


def binance(asset,symbol):
    rows=[]; d=pd.Timestamp(START,tz='UTC'); e=pd.Timestamp(END,tz='UTC')
    while d<e:
        stem=f'{symbol}-1m-{d:%Y-%m-%d}.zip'; url=f'https://data.binance.vision/data/spot/daily/klines/{symbol}/1m/{stem}'
        r=requests.get(url,timeout=40)
        if r.status_code==200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z: raw=z.read(z.namelist()[0])
            x=pd.read_csv(io.BytesIO(raw),header=None,names=['open_time','open','high','low','close','volume','close_time','qv','trades','tb','tq','ignore'])
            first=float(x.open_time.iloc[0]); unit='us' if first>1e14 else ('ms' if first>1e11 else 's')
            x['timestamp']=pd.to_datetime(x.open_time,unit=unit,utc=True)
            rows.append(x[['timestamp','open','high','low','close','volume']])
        d+=pd.Timedelta(days=1)
    if not rows: raise RuntimeError('no Binance daily archives')
    x=pd.concat(rows,ignore_index=True)
    for c in ('open','high','low','close','volume'): x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.dropna().sort_values('timestamp').drop_duplicates('timestamp')
    x.to_csv(OUT/(asset+'.csv.gz'),index=False,compression='gzip')
    return len(x)


async def deriv_chunk(symbol,start,end):
    import websockets
    async with websockets.connect('wss://ws.derivws.com/websockets/v3?app_id=1089',ping_interval=20,max_size=20_000_000) as ws:
        req={'ticks_history':symbol,'start':int(start.timestamp()),'end':int(end.timestamp()),'style':'candles','granularity':60,'adjust_start_time':1}
        await ws.send(json.dumps(req)); msg=json.loads(await asyncio.wait_for(ws.recv(),timeout=45))
        if 'error' in msg: raise RuntimeError(str(msg['error']))
        return msg.get('candles',[])


def deriv(asset,symbol):
    rows=[]; s=pd.Timestamp(START,tz='UTC'); finish=pd.Timestamp(END,tz='UTC')
    while s<finish:
        e=min(s+pd.Timedelta(days=2),finish-pd.Timedelta(seconds=1)); rows.extend(asyncio.run(deriv_chunk(symbol,s,e))); s=e+pd.Timedelta(seconds=1)
    if not rows: raise RuntimeError('no Deriv candles')
    x=pd.DataFrame(rows); x['timestamp']=pd.to_datetime(x.epoch,unit='s',utc=True); x['volume']=0.0
    for c in ('open','high','low','close'): x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x[['timestamp','open','high','low','close','volume']].dropna().sort_values('timestamp').drop_duplicates('timestamp')
    x.to_csv(OUT/(asset+'.csv.gz'),index=False,compression='gzip'); return len(x)


rows=[]
for asset,src in list(DUKA.items())+list(CRYPTO.items())+list(SYNTH.items()):
    try:
        print('ACQUIRE',asset,flush=True)
        n=dukascopy(asset,src) if asset in DUKA else (binance(asset,src) if asset in CRYPTO else deriv(asset,src))
        rows.append([asset,'OK',n,''])
    except Exception as exc:
        print('FAILED',asset,exc,flush=True); rows.append([asset,'FAILED',0,str(exc)[:1000]])

cov=pd.DataFrame(rows,columns=['asset','state','rows','error'])
cov.to_csv('trailaris/coverage.csv',index=False)
Path('trailaris/coverage.json').write_text(json.dumps({'required':34,'ok':int((cov.state=='OK').sum()),'failed':cov.loc[cov.state!='OK','asset'].tolist()},indent=2))
print(cov.to_string(index=False))
