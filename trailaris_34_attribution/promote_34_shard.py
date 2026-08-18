#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--universe-dir',type=Path,required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    cands=pd.read_csv(a.universe_dir/'ALL_CANDIDATES.csv.gz');cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    if int(cands.asset.nunique())!=34:raise RuntimeError('expected exact 34 candidate assets')
    idx=pd.read_csv(a.universe_dir/'FEATURE_INDEX.csv');meta=idx[['asset','discovery_provider_name']].drop_duplicates().copy()
    import trailaris_r4_full_universe_loop as r4;install_universe(r4,meta)
    import R5_BASE_CAUSAL_ENGINE as base;base.r4.factor_vec=r4.factor_vec
    import longcycle_bounded_memory_candidate as bounded
    import longcycle_integrated_negative_refinement as neg
    import quantitative_integrated_v2 as qv2
    t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max();first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC')
    weeks=[];wk=first
    while wk<=t1:weeks.append(wk);wk+=pd.Timedelta(days=7)
    if len(weeks)!=26:raise RuntimeError(f'expected 26 weeks got {len(weeks)}')
    assigned=[(i,w) for i,w in enumerate(weeks) if i%a.shards==a.shard];promoted=[];rejections=[];counts=[]
    for i,wk in assigned:
        target=cands[(cands.decision_time>=wk)&(cands.decision_time<wk+pd.Timedelta(days=7))]
        p0,_,_=bounded.promote_for_week(cands,wk)
        p1,r1=neg.entry_gate(p0)
        p2,_,_=qv2.promote_for_week(cands,wk)
        if len(p2):p2=p2.assign(eval_week=wk,eval_week_index=i);promoted.append(p2)
        if len(r1):rejections.append(r1.assign(eval_week=wk,eval_week_index=i,gate_stage='R5_5_NEGATIVE_EDGE'))
        if len(p1):
            k2=set(p2.campaign_id.astype(str)) if len(p2) else set();qr=p1[~p1.campaign_id.astype(str).isin(k2)].copy()
            if len(qr):rejections.append(qr.assign(eval_week=wk,eval_week_index=i,gate_stage='QUANT_V2_STRICT_POSTERIOR'))
        rec={'eval_week_index':i,'eval_week':str(wk),'target_candidate_rows':int(len(target)),'bounded_promoted':int(len(p0)),'after_r5_5_negative_gate':int(len(p1)),'quant_v2_promoted':int(len(p2)),'r5_5_removed':int(len(r1)),'quant_v2_removed':int(len(p1)-len(p2))}
        counts.append(rec);print(json.dumps(rec))
    P=pd.concat(promoted,ignore_index=True) if promoted else pd.DataFrame();R=pd.concat(rejections,ignore_index=True) if rejections else pd.DataFrame()
    P.to_csv(a.outdir/f'QV2_34_PROMOTED_SHARD_{a.shard}.csv.gz',index=False,compression='gzip');R.to_csv(a.outdir/f'QV2_34_GATE_REJECTIONS_SHARD_{a.shard}.csv.gz',index=False,compression='gzip')
    status={'state':'QV2_34_GATE_CASCADE_SHARD_COMPLETE','shard':a.shard,'shards':a.shards,'candidate_rows':int(len(cands)),'assigned_week_count':len(assigned),'weeks':counts,'promoted_rows':int(len(P)),'gate_rejection_rows':int(len(R)),'strategy_blob':'120390ab52e80f51bafedc42389ceb7420a410f2'}
    (a.outdir/f'QV2_34_PROMOTION_SHARD_{a.shard}_STATUS.json').write_text(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
