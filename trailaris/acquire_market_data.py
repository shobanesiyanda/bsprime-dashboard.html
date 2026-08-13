#!/usr/bin/env python3
from pathlib import Path
import json, subprocess
import pandas as pd

out=Path('trailaris/raw'); out.mkdir(parents=True,exist_ok=True)
work=out/'_natgas'; work.mkdir(exist_ok=True)
cmd=['npx','--yes','dukascopy-node','-i','gascmdusd','-from','2026-05-04','-to','2026-08-13','-t','m5','-f','csv','-dir',str(work.resolve())]
cp=subprocess.run(cmd,capture_output=True,text=True,timeout=900)
rows=[]
if cp.returncode==0:
    for p in sorted([q for q in work.rglob('*') if q.is_file() and q.stat().st_size>100],key=lambda q:q.stat().st_size,reverse=True):
        try:
            x=pd.read_csv(p); x.columns=[str(c).strip().lower() for c in x.columns]
            tc=next((c for c in ('timestamp','time','datetime','date') if c in x.columns),None)
            if tc is None: continue
            ts=x[tc]
            if pd.api.types.is_numeric_dtype(ts):
                z=float(pd.to_numeric(ts,errors='coerce').dropna().iloc[0]); unit='us' if z>1e14 else ('ms' if z>1e11 else 's')
                x['timestamp']=pd.to_datetime(ts,unit=unit,utc=True)
            else: x['timestamp']=pd.to_datetime(ts,utc=True,errors='coerce')
            for c in ('open','high','low','close'): x[c]=pd.to_numeric(x[c],errors='coerce')
            if 'volume' not in x.columns: x['volume']=0.0
            x['volume']=pd.to_numeric(x['volume'],errors='coerce').fillna(0)
            x=x[['timestamp','open','high','low','close','volume']].dropna(subset=['timestamp','open','high','low','close']).sort_values('timestamp').drop_duplicates('timestamp')
            if len(x)>100:
                x.to_csv(out/'NATGAS.csv.gz',index=False,compression='gzip'); rows=[['NATGAS','OK',len(x),'']]; break
        except Exception:
            pass
if not rows: rows=[['NATGAS','FAILED',0,(cp.stderr+' '+cp.stdout)[-1200:]]]
c=pd.DataFrame(rows,columns=['asset','state','rows','error']); c.to_csv('trailaris/coverage.csv',index=False)
Path('trailaris/coverage.json').write_text(json.dumps({'required':34,'selected':['NATGAS'],'ok':int((c.state=='OK').sum()),'failed':c.loc[c.state!='OK','asset'].tolist()},indent=2))
print(c.to_string(index=False))
if c.state.iloc[0]!='OK': raise SystemExit(2)
