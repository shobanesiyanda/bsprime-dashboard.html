#!/usr/bin/env python3
import json, os, subprocess
from pathlib import Path
import pandas as pd

# Load the approved acquisition definitions from the immutable source commit.
subprocess.run(['git','fetch','--quiet','--depth=30','origin','trailaris-data-runtime-20260813'],check=False)
text=subprocess.check_output(['git','show','bccbe92864d8f1ac8eaf7f19151c4faa7b47272f:trailaris/acquire_market_data.py'],text=True)
text=text.replace("'-t','m1'","'-t','m5'")
text=text.replace("{symbol}-1m-","{symbol}-5m-")
text=text.replace("/{symbol}/1m/{stem}","/{symbol}/5m/{stem}")
text=text.replace("'granularity':60","'granularity':300")
prefix=text.split('\nrows=[]\n',1)[0]
ns={'__name__':'trailaris_acquisition_definitions'}
exec(compile(prefix,'trailaris/acquire_market_data_immutable.py','exec'),ns)
DUKA,CRYPTO,SYNTH=ns['DUKA'],ns['CRYPTO'],ns['SYNTH']
items=list(DUKA.items())+list(CRYPTO.items())+list(SYNTH.items())
shards=6
run_number=int(os.environ.get('GITHUB_RUN_NUMBER','0'))
shard=run_number % shards
selected=[(a,s) for i,(a,s) in enumerate(items) if i % shards == shard]
print('TRAILARIS_SHARD',shard,'OF',shards,'ASSETS',[a for a,_ in selected],flush=True)
rows=[]
for asset,src in selected:
    try:
        if asset in DUKA: n=ns['dukascopy'](asset,src)
        elif asset in CRYPTO: n=ns['binance'](asset,src)
        else: n=ns['deriv'](asset,src)
        rows.append([asset,'OK',n,''])
        print('OK',asset,n,flush=True)
    except Exception as exc:
        rows.append([asset,'FAILED',0,str(exc)[:1000]])
        print('FAILED',asset,exc,flush=True)
cov=pd.DataFrame(rows,columns=['asset','state','rows','error'])
cov.to_csv('trailaris/coverage.csv',index=False)
Path('trailaris/coverage.json').write_text(json.dumps({'required':34,'shard':shard,'shards':shards,'selected':[a for a,_ in selected],'ok':int((cov.state=='OK').sum()),'failed':cov.loc[cov.state!='OK','asset'].tolist()},indent=2))
print(cov.to_string(index=False))
