#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, os, subprocess
from pathlib import Path
import pandas as pd

subprocess.run(['git','fetch','--quiet','--depth=30','origin','trailaris-data-runtime-20260813'],check=False)
text=subprocess.check_output(['git','show','bccbe92864d8f1ac8eaf7f19151c4faa7b47272f:trailaris/acquire_market_data.py'],text=True)
text=text.replace("'-t','m1'","'-t','m5'")
text=text.replace("{symbol}-1m-","{symbol}-5m-")
text=text.replace("/{symbol}/1m/{stem}","/{symbol}/5m/{stem}")
text=text.replace("'granularity':60","'granularity':300")
prefix=text.split('\nrows=[]\n',1)[0]
ns={'__name__':'trailaris_acquisition_definitions'}
exec(compile(prefix,'trailaris/acquire_market_data_immutable.py','exec'),ns)
DUKA,CRYPTO,SYNTH=ns['DUKA'],ns['CRYPTO'],ns['SYNTH']; OUT=ns['OUT']

def duka_fixed(asset,inst):
    work=OUT/('_fixed_'+asset); work.mkdir(exist_ok=True)
    cmd=['npx','--yes','dukascopy-node','-i',inst,'-from',ns['START'],'-to',ns['END'],'-t','m5','-f','csv','-dir',str(work.resolve())]
    cp=subprocess.run(cmd,capture_output=True,text=True,timeout=900)
    if cp.returncode: raise RuntimeError((cp.stderr+' '+cp.stdout)[-1800:])
    files=[p for p in work.rglob('*') if p.is_file() and p.stat().st_size>100 and p.suffix.lower() not in ('.bi5','.json')]
    if not files: raise RuntimeError('no materialized CSV-like output; '+cp.stdout[-1200:])
    errors=[]
    for p in sorted(files,key=lambda q:q.stat().st_size,reverse=True):
        try:
            d=ns['normalize'](pd.read_csv(p))
            if len(d)>10:
                d.to_csv(OUT/(asset+'.csv.gz'),index=False,compression='gzip')
                return len(d)
        except Exception as e: errors.append(f'{p.name}:{e}')
    raise RuntimeError('materialized files unreadable; '+' | '.join(errors[:4]))

def acquire_one(asset,src):
    try:
        if asset in DUKA: n=duka_fixed(asset,src)
        elif asset in CRYPTO: n=ns['binance'](asset,src)
        else: n=ns['deriv'](asset,src)
        print('OK',asset,n,flush=True); return [asset,'OK',n,'']
    except Exception as exc:
        print('FAILED',asset,exc,flush=True); return [asset,'FAILED',0,str(exc)[:1400]]

items=list(DUKA.items())+list(CRYPTO.items())+list(SYNTH.items())
shards=6
run_number=int(os.environ.get('GITHUB_RUN_NUMBER','0'))
shard=run_number % shards
selected=[(a,s) for i,(a,s) in enumerate(items) if i % shards == shard]
print('TRAILARIS_SHARD',shard,'OF',shards,'ASSETS',[a for a,_ in selected],flush=True)
rows=[]
with ThreadPoolExecutor(max_workers=len(selected)) as pool:
    fut=[pool.submit(acquire_one,a,s) for a,s in selected]
    for f in as_completed(fut): rows.append(f.result())
order=[a for a,_ in selected]; rows.sort(key=lambda x:order.index(x[0]))
cov=pd.DataFrame(rows,columns=['asset','state','rows','error'])
cov.to_csv('trailaris/coverage.csv',index=False)
Path('trailaris/coverage.json').write_text(json.dumps({'required':34,'shard':shard,'shards':shards,'selected':order,'ok':int((cov.state=='OK').sum()),'failed':cov.loc[cov.state!='OK','asset'].tolist()},indent=2))
print(cov.to_string(index=False))
