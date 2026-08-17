#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def safe(asset:str)->str:return hashlib.sha1(asset.encode()).hexdigest()[:16]

def main():
    p=argparse.ArgumentParser();p.add_argument('--rawdir',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    import trailaris_r4_full_universe_loop as r4
    ev=json.loads((a.rawdir/'ACQUISITION_EVIDENCE.json').read_text())
    accepted=[];rejected=[]
    for e in ev.get('evidence',[]):
        if e.get('status')!='PASS' or not e.get('file'):
            rejected.append({**e,'feature_state':'ACQUISITION_REJECT'});continue
        f=Path(e['file']);asset=e['canonical_instrument']
        try:
            _,_,xf,_=r4.build_asset_components((asset,str(f)))
            need=[c for c in ['timestamp','high','low','close'] if c in xf.columns]
            if set(['timestamp','high','low','close'])-set(need):raise RuntimeError(f'missing feature columns {need}')
            z=xf[need].copy();z['timestamp']=pd.to_datetime(z.timestamp,utc=True);z=z.sort_values('timestamp').drop_duplicates('timestamp')
            z.to_csv(a.outdir/f'FEATURE_{safe(asset)}.csv.gz',index=False,compression='gzip')
            accepted.append({**e,'feature_state':'PASS','feature_rows':len(z),'feature_file':f'FEATURE_{safe(asset)}.csv.gz'})
        except Exception as ex:
            rejected.append({**e,'feature_state':'R4_FEATURE_REJECT','feature_error':f'{type(ex).__name__}:{str(ex)[:500]}'})
        finally:
            try:f.unlink()
            except Exception:pass
    pd.DataFrame(accepted).to_csv(a.outdir/'FEATURE_ACCEPTED.csv',index=False);pd.DataFrame(rejected).to_csv(a.outdir/'FEATURE_REJECTED.csv',index=False)
    out={'state':'FEATURE_CACHE_SHARD_COMPLETE','scheduled':ev.get('assets_scheduled',0),'feature_assets':len(accepted),'rejected':len(rejected),'quant_v2_modified':False};(a.outdir/'FEATURE_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
