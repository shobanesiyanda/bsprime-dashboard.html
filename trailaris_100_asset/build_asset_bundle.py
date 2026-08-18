#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe, anchor_for


def safe(asset: str) -> str:
    return hashlib.sha1(asset.encode('utf-8')).hexdigest()[:16]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rawdir',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    ev=json.loads((a.rawdir/'ACQUISITION_EVIDENCE.json').read_text())
    passed=[e for e in ev.get('evidence',[]) if e.get('status')=='PASS' and e.get('file')]
    failed=[e for e in ev.get('evidence',[]) if e.get('status')!='PASS' or not e.get('file')]
    if failed:
        raise RuntimeError(f"fixed-universe acquisition failure: {len(failed)} assets; first={failed[:3]}")
    meta=pd.DataFrame([{'asset':str(e['canonical_instrument']),'discovery_provider_name':str(e.get('provider_name',''))} for e in passed])
    import trailaris_r4_full_universe_loop as r4
    mapping=install_universe(r4,meta)
    cand=[];feature_index=[]
    for e in passed:
        asset=str(e['canonical_instrument']);provider=str(e.get('provider_name',''));f=Path(e['file'])
        a0,c,xf,o=r4.build_asset_components((asset,str(f)))
        if str(a0)!=asset: raise RuntimeError(f'asset identity mismatch {a0} != {asset}')
        if xf is None or len(xf)==0: raise RuntimeError(f'empty feature frame {asset}')
        z=xf.copy();z['timestamp']=pd.to_datetime(z['timestamp'],utc=True,errors='coerce')
        if z['timestamp'].isna().any(): raise RuntimeError(f'invalid feature timestamps {asset}')
        z=z.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        fn=f'FEATURE_{safe(asset)}.csv.gz';z.to_csv(a.outdir/fn,index=False,compression='gzip')
        feature_index.append({'asset':asset,'discovery_provider_name':provider,'instrument_id':str(e.get('instrument_id','')),'feature_file':fn,'feature_rows':len(z),'anchor':anchor_for(asset,provider)})
        if c is not None and len(c):
            q=c.copy();q['discovery_provider_id']=str(e.get('instrument_id',''));q['discovery_provider_name']=provider;q['universe_adapter_anchor']=mapping.get(asset,anchor_for(asset,provider));cand.append(q)
    C=pd.concat(cand,ignore_index=True) if cand else pd.DataFrame()
    C.to_csv(a.outdir/'CANDIDATES.csv.gz',index=False,compression='gzip')
    pd.DataFrame(feature_index).to_csv(a.outdir/'FEATURE_INDEX.csv',index=False)
    out={'state':'ASSET_BUNDLE_COMPLETE','scheduled':int(ev.get('assets_scheduled',len(passed))),'assets_pass':len(passed),'feature_assets':len(feature_index),'candidate_assets':int(C.asset.nunique()) if len(C) else 0,'candidate_rows':len(C),'quant_v2_modified':False}
    (a.outdir/'BUNDLE_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
