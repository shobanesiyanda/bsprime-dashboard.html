#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--universe-dir',type=Path,required=True)
    ap.add_argument('--shard',type=int,required=True)
    ap.add_argument('--shards',type=int,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    cands=pd.read_csv(a.universe_dir/'ALL_CANDIDATES.csv.gz')
    cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    idx=pd.read_csv(a.universe_dir/'FEATURE_INDEX.csv')
    import trailaris_r4_full_universe_loop as r4
    meta=idx[['asset','discovery_provider_name']].drop_duplicates().copy();install_universe(r4,meta)
    import R5_BASE_CAUSAL_ENGINE as base;base.r4.factor_vec=r4.factor_vec
    import quantitative_integrated_v2 as qv2
    t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max();first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC')
    weeks=[];wk=first
    while wk<=t1:weeks.append(wk);wk+=pd.Timedelta(days=7)
    if len(weeks)!=26:raise RuntimeError(f'expected 26 evaluated weeks, got {len(weeks)}')
    assigned=[(i,w) for i,w in enumerate(weeks) if i%a.shards==a.shard]
    promoted=[];counts=[]
    for i,wk in assigned:
        p,_,_=qv2.promote_for_week(cands,wk)
        if len(p):
            p=p.assign(eval_week=wk,eval_week_index=i);promoted.append(p)
        counts.append({'eval_week_index':i,'eval_week':str(wk),'promoted_rows':int(len(p)),'promoted_assets':int(p.asset.nunique()) if len(p) else 0})
        print(json.dumps(counts[-1]))
    P=pd.concat(promoted,ignore_index=True) if promoted else pd.DataFrame()
    P.to_csv(a.outdir/f'QV2_100_PROMOTED_SHARD_{a.shard}.csv.gz',index=False,compression='gzip')
    status={'state':'QV2_100_WEEKLY_PROMOTION_SHARD_COMPLETE','shard':a.shard,'shards':a.shards,'candidate_rows':len(cands),'candidate_assets':int(cands.asset.nunique()),'assigned_week_count':len(assigned),'weeks':counts,'promoted_rows':len(P),'strategy_blob':'120390ab52e80f51bafedc42389ceb7420a410f2','decision_semantics_changed':False}
    (a.outdir/f'QV2_100_PROMOTION_SHARD_{a.shard}_STATUS.json').write_text(json.dumps(status,indent=2,default=str))
    print(json.dumps(status,indent=2,default=str))

if __name__=='__main__':main()
