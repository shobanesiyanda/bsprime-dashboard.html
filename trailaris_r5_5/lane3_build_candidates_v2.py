#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import deque
import numpy as np,pandas as pd

EPS=1e-12

def num(x,d=np.nan):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def prep(x):
    z=x.copy().reset_index(drop=True);z['timestamp']=pd.to_datetime(z.timestamp,utc=True)
    rng=(z.high-z.low).replace(0,np.nan)
    if 'range' not in z:z['range']=rng
    if 'body' not in z:z['body']=(z.close-z.open).abs()
    if 'body_frac' not in z:z['body_frac']=(z.body/z['range']).fillna(0)
    if 'atr' not in z:
        prev=z.close.shift();tr=pd.concat([(z.high-z.low).abs(),(z.high-prev).abs(),(z.low-prev).abs()],axis=1).max(axis=1);z['atr']=tr.rolling(14,min_periods=5).mean().bfill()
    if 'ema20' not in z:z['ema20']=z.close.ewm(span=20,adjust=False).mean()
    if 'z30' not in z:z['z30']=(z.close-z.close.rolling(30,min_periods=20).mean())/z.close.rolling(30,min_periods=20).std().replace(0,np.nan)
    if 'compression' not in z:
        med=z.atr.rolling(50,min_periods=20).median();z['compression']=z.atr/med.replace(0,np.nan)
    z['prior12_hi']=z.high.shift(1).rolling(12,min_periods=6).max();z['prior12_lo']=z.low.shift(1).rolling(12,min_periods=6).min()
    z['prior6_hi']=z.high.shift(1).rolling(6,min_periods=3).max();z['prior6_lo']=z.low.shift(1).rolling(6,min_periods=3).min()
    return z

def emit(r4,asset,strategy,family,x,i,dr,q,ctx,max_bars,management):
    c=r4.candidate(asset,strategy,x,i,dr,float(np.clip(q,0,1)),ctx,management=management,max_bars=max_bars)
    if not c:return None
    c['strategy']=strategy;c['strategy_family']=family;c['quality']=float(np.clip(q,0,1));c['context']=ctx
    c['campaign_id']=f"{asset}|{strategy}|{pd.Timestamp(c['decision_time']).isoformat()}"
    return c

def origin_for_displacement(x,j,side):
    for k in range(j-1,max(11,j-5),-1):
        r=x.iloc[k]
        if side=='demand' and r.close<=r.open:return k
        if side=='supply' and r.close>=r.open:return k
    return None

def supply_demand_candidates(asset,x,r4):
    strict=[];balanced=[];seen_s=set();seen_b=set();zones=deque()
    for i in range(16,len(x)-2):
        r=x.iloc[i];prev=x.iloc[i-1];atr=max(num(r.atr,0),EPS)
        keep=deque()
        for z in zones:
            age=i-z['created']
            if age>72:continue
            invalid=(r.close<z['low']) if z['side']=='demand' else (r.close>z['high'])
            if invalid:continue
            overlap=(r.low<=z['high'] and r.high>=z['low'])
            if overlap:
                trend=int(num(r.get('h1_trend'),0));strength=max(0,num(r.get('h1_strength'),0));body=num(r.body_frac,0)
                if z['side']=='demand' and trend==1:
                    liq=r.low<min(float(prev.low),num(r.prior6_lo,float(prev.low)));reclaim=r.close>z['high'];bos=r.close>prev.high
                    if liq and reclaim and bos:
                        key=(pd.Timestamp(r.timestamp).floor('30min'),1)
                        if z['disp']>=.70 and z['touches']<=1 and body>=.42 and key not in seen_b:
                            q=.66+.06*min(z['disp'],2)+.04*min(strength,2)+(.05 if z['touches']==0 else 0)-.001*age
                            ctx=f"R55_BALANCED_DEMAND_ZONE;origin={x.timestamp.iloc[z['origin']]};created={x.timestamp.iloc[z['created']]};disp={z['disp']:.2f};age={age};touches={z['touches']};LIQ_SWEEP;RECLAIM;BOS"
                            c=emit(r4,asset,'SUPPLY_DEMAND_SWEEP_BOS','SUPPLY_DEMAND',x,i,1,q,ctx,42,'PROTECTED_RUNNER')
                            if c:balanced.append(c);seen_b.add(key)
                        if z['disp']>=.90 and age<=48 and z['touches']==0 and body>=.52 and key not in seen_s:
                            q=.69+.06*min(z['disp'],2)+.04*min(strength,2)+.05-.001*age
                            ctx=f"R55_STRICT_DEMAND_ZONE;origin={x.timestamp.iloc[z['origin']]};created={x.timestamp.iloc[z['created']]};disp={z['disp']:.2f};age={age};touches=0;LIQ_SWEEP;RECLAIM;BOS"
                            c=emit(r4,asset,'SUPPLY_DEMAND_SWEEP_BOS','SUPPLY_DEMAND',x,i,1,q,ctx,42,'PROTECTED_RUNNER')
                            if c:strict.append(c);seen_s.add(key)
                elif z['side']=='supply' and trend==-1:
                    liq=r.high>max(float(prev.high),num(r.prior6_hi,float(prev.high)));reclaim=r.close<z['low'];bos=r.close<prev.low
                    if liq and reclaim and bos:
                        key=(pd.Timestamp(r.timestamp).floor('30min'),-1)
                        if z['disp']>=.70 and z['touches']<=1 and body>=.42 and key not in seen_b:
                            q=.66+.06*min(z['disp'],2)+.04*min(strength,2)+(.05 if z['touches']==0 else 0)-.001*age
                            ctx=f"R55_BALANCED_SUPPLY_ZONE;origin={x.timestamp.iloc[z['origin']]};created={x.timestamp.iloc[z['created']]};disp={z['disp']:.2f};age={age};touches={z['touches']};LIQ_SWEEP;RECLAIM;BOS"
                            c=emit(r4,asset,'SUPPLY_DEMAND_SWEEP_BOS','SUPPLY_DEMAND',x,i,-1,q,ctx,42,'PROTECTED_RUNNER')
                            if c:balanced.append(c);seen_b.add(key)
                        if z['disp']>=.90 and age<=48 and z['touches']==0 and body>=.52 and key not in seen_s:
                            q=.69+.06*min(z['disp'],2)+.04*min(strength,2)+.05-.001*age
                            ctx=f"R55_STRICT_SUPPLY_ZONE;origin={x.timestamp.iloc[z['origin']]};created={x.timestamp.iloc[z['created']]};disp={z['disp']:.2f};age={age};touches=0;LIQ_SWEEP;RECLAIM;BOS"
                            c=emit(r4,asset,'SUPPLY_DEMAND_SWEEP_BOS','SUPPLY_DEMAND',x,i,-1,q,ctx,42,'PROTECTED_RUNNER')
                            if c:strict.append(c);seen_s.add(key)
                z=dict(z);z['touches']+=1
            keep.append(z)
        zones=keep
        disp=abs(float(r.close-r.open))/atr
        if np.isfinite(r.prior12_hi) and r.close>r.prior12_hi and disp>=.70:
            k=origin_for_displacement(x,i,'demand')
            if k is not None:
                o=x.iloc[k];zones.append({'side':'demand','origin':k,'created':i,'low':float(o.low),'high':float(max(o.open,o.close)),'disp':float(disp),'touches':0})
        if np.isfinite(r.prior12_lo) and r.close<r.prior12_lo and disp>=.70:
            k=origin_for_displacement(x,i,'supply')
            if k is not None:
                o=x.iloc[k];zones.append({'side':'supply','origin':k,'created':i,'low':float(min(o.open,o.close)),'high':float(o.high),'disp':float(disp),'touches':0})
        if len(zones)>96:zones=deque(list(zones)[-96:])
    return strict,balanced

def payoff_to_value_r(c,value,dr):
    if not c:return np.nan
    ent=num(c.get('entry_price'));sd=abs(num(c.get('stop_distance')))
    if not np.isfinite(ent) or not np.isfinite(sd) or sd<=EPS or not np.isfinite(value):return np.nan
    return float((value-ent)*int(dr)/sd)

def mean_reversion_candidates(asset,x,r4):
    strict=[];balanced=[];seen_s=set();seen_b=set()
    z=x.z30.astype(float);pz=z.shift(1);strength=x.get('h1_strength',pd.Series(0.,index=x.index)).abs().astype(float);comp=x.compression.astype(float)
    prior11_hi=x.high.shift(2).rolling(11,min_periods=5).max();prior11_lo=x.low.shift(2).rolling(11,min_periods=5).min()
    rng=(x.high.shift(1)-x.low.shift(1)).replace(0,np.nan)
    upper_wick=(x.high.shift(1)-pd.concat([x.open.shift(1),x.close.shift(1)],axis=1).max(axis=1))/rng
    lower_wick=(pd.concat([x.open.shift(1),x.close.shift(1)],axis=1).min(axis=1)-x.low.shift(1))/rng
    high_base=(pz>=2.0)&(z<pz)&(z<2.0)&(x.close<x.open)&(x.close<x.close.shift(1))&(x.high.shift(1)>prior11_hi)&(upper_wick>=.20)&(strength<=.50)&(comp<=1.45)&(x.close>x.ema20)
    low_base=(pz<=-2.0)&(z>pz)&(z>-2.0)&(x.close>x.open)&(x.close>x.close.shift(1))&(x.low.shift(1)<prior11_lo)&(lower_wick>=.20)&(strength<=.50)&(comp<=1.45)&(x.close<x.ema20)
    idx=np.flatnonzero((high_base|low_base).fillna(False).to_numpy())
    for i in idx:
        if i<61 or i>=len(x)-2:continue
        dr=-1 if bool(high_base.iloc[i]) else 1;wick=float(upper_wick.iloc[i] if dr==-1 else lower_wick.iloc[i]);peak=float(pz.iloc[i]);cv=float(comp.iloc[i]);value=float(x.ema20.iloc[i]);key=(pd.Timestamp(x.timestamp.iloc[i]).floor('30min'),dr)
        if key not in seen_b:
            q=.62+.06*min(abs(peak)-2,2)+.08*min(wick,1)+.05*max(0,1-cv)
            c=emit(r4,asset,'MEAN_REVERSION_EXTREME','MEAN_REVERSION',x,i,dr,q,'R55_BALANCED_EXHAUST_PENDING_PAYOFF',30,'PARTIAL_RUNNER')
            payoff=payoff_to_value_r(c,value,dr)
            if c and np.isfinite(payoff) and payoff>=.75:
                c['quality']=float(np.clip(q+.03*min(payoff,2),0,1));c['context']=f"R55_BALANCED_EXHAUST;peak_z={peak:.2f};reentry_z={float(z.iloc[i]):.2f};wick={wick:.2f};compression={cv:.2f};value_payoff_r={payoff:.2f};VALUE_REENTRY;PAYOFF_GATE"
                balanced.append(c);seen_b.add(key)
        strict_ok=(abs(peak)>=2.25 and strength.iloc[i]<=.35 and cv<=1.25 and wick>=.30)
        if strict_ok and key not in seen_s:
            q=.66+.06*min(abs(peak)-2,2)+.08*min(wick,1)+.05*max(0,1-cv)
            c=emit(r4,asset,'MEAN_REVERSION_EXTREME','MEAN_REVERSION',x,i,dr,q,'R55_STRICT_EXHAUST_PENDING_PAYOFF',30,'PARTIAL_RUNNER')
            payoff=payoff_to_value_r(c,value,dr)
            if c and np.isfinite(payoff) and payoff>=1.00:
                c['quality']=float(np.clip(q+.04*min(payoff,2),0,1));c['context']=f"R55_STRICT_EXHAUST;peak_z={peak:.2f};reentry_z={float(z.iloc[i]):.2f};wick={wick:.2f};compression={cv:.2f};value_payoff_r={payoff:.2f};VALUE_REENTRY;PAYOFF_GATE"
                strict.append(c);seen_s.add(key)
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
    status={'state':'R5_5_LANE3_CANDIDATE_BUILD_V2_COMPLETE','revision':'PAYOFF_GATED_V2_1','routes':34,'counts':{k:len(v) for k,v in buckets.items()},'governance':'Each rebuilt candidate is generated causally from bars completed before its decision bar. Supply/demand zones are created only after displacement/BOS completes, then tracked forward for age, invalidation and touches. Mean reversion requires completed exhaustion, re-entry while still on the extreme side of EMA20 value, weak-trend/volatility controls, and decision-time reward-to-value of at least 0.75R balanced or 1.00R strict relative to the candidate stop distance. Candidate files are built once and reused unchanged across the replay matrix.'}
    (out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__':main()
