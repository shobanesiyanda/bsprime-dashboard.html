#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    files=sorted(a.candidate_root.rglob('CANDIDATES.csv.gz'))
    parts=[]
    for f in files:
        try:
            z=pd.read_csv(f)
            if len(z):parts.append(z)
        except pd.errors.EmptyDataError:pass
    if not parts:raise RuntimeError('no frozen-contract-compatible candidate rows produced')
    cands=pd.concat(parts,ignore_index=True)
    cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    cands=cands.sort_values(['decision_time','campaign_id']).reset_index(drop=True)
    cands.to_csv(a.outdir/'ALL_CANDIDATES.csv.gz',index=False,compression='gzip')

    import quantitative_integrated_v2 as qv2
    t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max()
    first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC')
    weeks=[];wk=first
    while wk<=t1:weeks.append(wk);wk+=pd.Timedelta(days=7)
    if len(weeks)!=26:raise RuntimeError(f'expected 26 evaluated weeks, got {len(weeks)}')

    promoted=[]
    for wk in weeks:
        pr,_,_=qv2.promote_for_week(cands,wk)
        if len(pr):promoted.append(pr.assign(eval_week=wk))
    P=pd.concat(promoted,ignore_index=True) if promoted else pd.DataFrame()
    P.to_csv(a.outdir/'PROMOTED_SELECTION.csv.gz',index=False,compression='gzip')
    if len(P):
        selected=P[['asset','discovery_provider_id','discovery_provider_name']].drop_duplicates().sort_values(['asset','discovery_provider_id'])
    else:selected=pd.DataFrame(columns=['asset','discovery_provider_id','discovery_provider_name'])
    with (a.outdir/'SELECTED_ASSETS.tsv').open('w',encoding='utf-8') as f:
        f.write('instrument_id\tcanonical_instrument\tprovider_name\n')
        for _,r in selected.iterrows():f.write(f"{r.discovery_provider_id}\t{r.asset}\t{r.discovery_provider_name}\n")
    out={'state':'FROZEN_QUANT_V2_FULL_UNIVERSE_SELECTION_COMPLETE','candidate_files':len(files),'candidate_rows':len(cands),'candidate_assets':int(cands.asset.nunique()),'evaluated_weeks':len(weeks),'promoted_rows':len(P),'promoted_assets':int(P.asset.nunique()) if len(P) else 0,'quant_v2_modified':False}
    (a.outdir/'SELECTION_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
