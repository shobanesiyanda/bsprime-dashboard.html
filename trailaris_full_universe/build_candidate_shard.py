#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--rawdir',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)

    import trailaris_r4_full_universe_loop as r4

    evidence=json.loads((args.rawdir/'ACQUISITION_EVIDENCE.json').read_text())
    cand=[]; accepted=[]; rejected=[]
    for e in evidence.get('evidence',[]):
        if e.get('status')!='PASS' or not e.get('file'):
            rejected.append({**e,'candidate_state':'ACQUISITION_REJECT'})
            continue
        asset=e['canonical_instrument']; f=Path(e['file'])
        try:
            # final frozen factor contract check immediately before candidate generation
            r4.factor_vec(asset,1); r4.factor_vec(asset,-1)
            a,c,xf,o=r4.build_asset_components((asset,str(f)))
            if c is not None and len(c):
                z=c.copy();z['discovery_provider_id']=e['instrument_id'];z['discovery_provider_name']=e.get('provider_name','');cand.append(z)
            accepted.append({**e,'candidate_state':'PASS','candidate_rows':0 if c is None else len(c),'opportunity_rows':0 if o is None else len(o)})
        except Exception as ex:
            rejected.append({**e,'candidate_state':'R4_CONTRACT_REJECT','candidate_error':f'{type(ex).__name__}:{str(ex)[:500]}'})
        finally:
            try:f.unlink()
            except Exception:pass

    if cand:
        pd.concat(cand,ignore_index=True).to_csv(args.outdir/'CANDIDATES.csv.gz',index=False,compression='gzip')
    else:
        pd.DataFrame().to_csv(args.outdir/'CANDIDATES.csv.gz',index=False,compression='gzip')
    pd.DataFrame(accepted).to_csv(args.outdir/'ACCEPTED_ASSETS.csv',index=False)
    pd.DataFrame(rejected).to_csv(args.outdir/'REJECTED_ASSETS.csv',index=False)
    out={'state':'CANDIDATE_SHARD_COMPLETE','acquisition_scheduled':evidence.get('assets_scheduled',0),'acquisition_pass':evidence.get('assets_pass',0),'r4_accepted_assets':len(accepted),'r4_rejected_assets':len(rejected),'candidate_rows':sum(len(x) for x in cand),'quant_v2_modified':False}
    (args.outdir/'CANDIDATE_SHARD_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
