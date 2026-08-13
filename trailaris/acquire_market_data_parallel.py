#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, pandas as pd

# Load only definitions from the acquisition module; do not execute its serial driver.
text=Path('trailaris/acquire_market_data.py').read_text()
prefix=text.split('\nrows=[]\n',1)[0]
ns={'__name__':'trailaris_acquisition_definitions'}
exec(compile(prefix,'trailaris/acquire_market_data.py','exec'),ns)
DUKA,CRYPTO,SYNTH=ns['DUKA'],ns['CRYPTO'],ns['SYNTH']

def one(asset,src):
    try:
        print('ACQUIRE',asset,flush=True)
        if asset in DUKA: n=ns['dukascopy'](asset,src)
        elif asset in CRYPTO: n=ns['binance'](asset,src)
        else: n=ns['deriv'](asset,src)
        return [asset,'OK',n,'']
    except Exception as exc:
        print('FAILED',asset,exc,flush=True)
        return [asset,'FAILED',0,str(exc)[:1000]]

items=list(DUKA.items())+list(CRYPTO.items())+list(SYNTH.items())
rows=[]
with ThreadPoolExecutor(max_workers=8) as pool:
    futures=[pool.submit(one,a,s) for a,s in items]
    for f in as_completed(futures): rows.append(f.result())
rows.sort(key=lambda x:[a for a,_ in items].index(x[0]))
cov=pd.DataFrame(rows,columns=['asset','state','rows','error'])
cov.to_csv('trailaris/coverage.csv',index=False)
Path('trailaris/coverage.json').write_text(json.dumps({'required':34,'ok':int((cov.state=='OK').sum()),'failed':cov.loc[cov.state!='OK','asset'].tolist()},indent=2))
print(cov.to_string(index=False))
