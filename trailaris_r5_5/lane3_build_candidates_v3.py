#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane3_build_candidates_v2 import prep,emit,origin_for_displacement,mean_reversion_candidates,num,EPS

def supply_demand_candidates(asset,x,r4):
    strict=[];balanced=[];seen_s=set();seen_b=set()
    atr=pd.to_numeric(x.atr,errors='coerce').replace(0,np.nan)
    disp=((x.close-x.open).abs()/atr).fillna(0)
    up=((x.close>x.prior12_hi)&(disp>=.70)).fillna(False)
    dn=((x.close<x.prior12_lo)&(disp>=.70)).fillna(False)
    origins=[(int(j),'demand') for j in np.flatnonzero(up.to_numpy())]+[(int(j),'supply') for j in np.flatnonzero(dn.to_numpy())]
    origins.sort()
    for created,side in origins:
        k=origin_for_displacement(x,created,side)
        if k is None:continue
        o=x.iloc[k];d=float(disp.iloc[created])
        if side=='demand':zl=float(o.low);zh=float(max(o.open,o.close));dr=1
        else:zl=float(min(o.open,o.close));zh=float(o.high);dr=-1
        touches=0
        for i in range(created+1,min(len(x)-2,created+73)):
            r=x.iloc[i];prev=x.iloc[i-1];age=i-created
            if (side=='demand' and r.close<zl) or (side=='supply' and r.close>zh):break
            overlap=(r.low<=zh and r.high>=zl)
            if not overlap:continue
            trend=int(num(r.get('h1_trend'),0));strength=max(0,num(r.get('h1_strength'),0));body=num(r.get('body_frac'),0)
            if side=='demand':
                liq=r.low<min(float(prev.low),num(r.get('prior6_lo'),float(prev.low)));reclaim=r.close>zh;bos=r.close>prev.high;aligned=trend==1
            else:
                liq=r.high>max(float(prev.high),num(r.get('prior6_hi'),float(prev.high)));reclaim=r.close<zl;bos=r.close<prev.low;aligned=trend==-1
            if aligned and liq and reclaim and bos:
                key=(pd.Timestamp(r.timestamp).floor('30min'),dr)
                if d>=.70 and touches<=1 and body>=.42 and key not in seen_b:
                    q=.66+.06*min(d,2)+.04*min(strength,2)+(.05 if touches==0 else 0)-.001*age
                    ctx=f"R55_BALANCED_{side.upper()}_ZONE;origin={x.timestamp.iloc[k]};created={x.timestamp.iloc[created]};disp={d:.2f};age={age};touches={touches};LIQ_SWEEP;RECLAIM;BOS"
                    c=emit(r4,asset,'SUPPLY_DEMAND_SWEEP_BOS','SUPPLY_DEMAND',x,i,dr,q,ctx,42,'PROTECTED_RUNNER')
                    if c:balanced.append(c);seen_b.add(key)
                if d>=.90 and age<=48 and touches==0 and body>=.52 and key not in seen_s:
                    q=.69+.06*min(d,2)+.04*min(strength,2)+.05-.001*age
                    ctx=f"R55_STRICT_{side.upper()}_ZONE;origin={x.timestamp.iloc[k]};created={x.timestamp.iloc[created]};disp={d:.2f};age={age};touches=0;LIQ_SWEEP;RECLAIM;BOS"
                    c=emit(r4,asset,'SUPPLY_DEMAND_SWEEP_BOS','SUPPLY_DEMAND',x,i,dr,q,ctx,42,'PROTECTED_RUNNER')
                    if c:strict.append(c);seen_s.add(key)
            touches+=1
            if touches>=2:break
    return strict,balanced

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    import trailaris_r4_asset_adapter as r4a
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir))
    if len(data)!=34:raise RuntimeError(f'route universe regressed {len(data)}/34')
    buckets={'sd_strict':[],'sd_balanced':[],'mr_strict':[],'mr_balanced':[]};stats=[]
    for asset,x0 in fcache.items():
        x=prep(x0);ss=[];sb=[];ms=[];mb=[]
        if r4a.family_applicable(asset,'SUPPLY_DEMAND'):ss,sb=supply_demand_candidates(asset,x,r4)
        if r4a.family_applicable(asset,'MEAN_REVERSION'):ms,mb=mean_reversion_candidates(asset,x,r4)
        buckets['sd_strict']+=ss;buckets['sd_balanced']+=sb;buckets['mr_strict']+=ms;buckets['mr_balanced']+=mb
        stats.append({'asset':asset,'sd_strict':len(ss),'sd_balanced':len(sb),'mr_strict':len(ms),'mr_balanced':len(mb)})
        print(asset,len(ss),len(sb),len(ms),len(mb),flush=True)
    for name,rows in buckets.items():pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    pd.DataFrame(stats).to_csv(out/'by_asset.csv',index=False)
    status={'state':'R5_5_LANE3_CANDIDATE_BUILD_V3_COMPLETE','revision':'PAYOFF_GATED_ZONE_LIFECYCLE_V3','routes':34,'counts':{k:len(v) for k,v in buckets.items()},'governance':'Supply/demand provenance is scanned per causally completed displacement/BOS origin through at most its next 72 completed M5 bars, terminating on invalidation or after the second touch. This is logically equivalent to active-zone tracking for the allowed strict/balanced touch limits but avoids repeated global zone scans. Mean reversion remains payoff-gated: exhaustion plus value re-entry, weak trend/volatility, extreme-side EMA20 state, and >=0.75R balanced or >=1.00R strict decision-time room back to value.'}
    (out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__':main()
