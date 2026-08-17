#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from qv2_asset_universe_adapter import install_asset

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    import trailaris_r4_full_universe_loop as r4
    ev=json.loads((a.rawdir/'ACQUISITION_EVIDENCE.json').read_text())
    cand=[];accepted=[];rejected=[]
    for e in ev.get('evidence',[]):
        if e.get('status')!='PASS' or not e.get('file'):
            rejected.append({**e,'candidate_state':'ACQUISITION_REJECT'});continue
        asset=str(e['canonical_instrument']); provider=str(e.get('provider_name','')); f=Path(e['file'])
        try:
            anchor=install_asset(r4,asset,provider)
            a0,c,xf,o=r4.build_asset_components((asset,str(f)))
            if c is not None and len(c):
                z=c.copy();z['discovery_provider_id']=e['instrument_id'];z['discovery_provider_name']=provider;z['universe_adapter_anchor']=anchor;cand.append(z)
            accepted.append({**e,'candidate_state':'PASS','universe_adapter_anchor':anchor,'candidate_rows':0 if c is None else len(c),'opportunity_rows':0 if o is None else len(o)})
        except Exception as ex:
            rejected.append({**e,'candidate_state':'R4_EXTENDED_UNIVERSE_REJECT','candidate_error':f'{type(ex).__name__}:{str(ex)[:500]}'})
        finally:
            try:f.unlink()
            except Exception:pass
    if cand: pd.concat(cand,ignore_index=True).to_csv(a.outdir/'CANDIDATES.csv.gz',index=False,compression='gzip')
    else: pd.DataFrame().to_csv(a.outdir/'CANDIDATES.csv.gz',index=False,compression='gzip')
    pd.DataFrame(accepted).to_csv(a.outdir/'ACCEPTED_ASSETS.csv',index=False);pd.DataFrame(rejected).to_csv(a.outdir/'REJECTED_ASSETS.csv',index=False)
    out={'state':'EXACT_QV2_EXPANDED_CANDIDATE_SHARD_COMPLETE','acquisition_scheduled':ev.get('assets_scheduled',0),'acquisition_pass':ev.get('assets_pass',0),'r4_extended_accepted_assets':len(accepted),'rejected_assets':len(rejected),'candidate_rows':sum(len(x) for x in cand),'quant_v2_modified':False,'normalized_r_used':False}
    (a.outdir/'CANDIDATE_SHARD_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
