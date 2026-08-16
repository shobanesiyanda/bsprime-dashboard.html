#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

R54_END=337.4231242104393
EPS=1e-12
TARGETS={'SUPPLY_DEMAND_SWEEP_BOS','MEAN_REVERSION_EXTREME'}
VARIANTS=[
 {'id':'BOTH_STRICT','sd':'strict','mr':'strict'},
 {'id':'SD_STRICT_MR_BALANCED','sd':'strict','mr':'balanced'},
 {'id':'SD_BALANCED_MR_STRICT','sd':'balanced','mr':'strict'},
 {'id':'BOTH_BALANCED','sd':'balanced','mr':'balanced'},
 {'id':'SD_STRICT_ONLY','sd':'strict','mr':'off'},
 {'id':'MR_STRICT_ONLY','sd':'off','mr':'strict'},
]

def n(x,d=np.nan):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def prep(x):
    z=x.copy();z['timestamp']=pd.to_datetime(z.timestamp,utc=True)
    if 'range' not in z:z['range']=(z.high-z.low).replace(0,np.nan)
    if 'body' not in z:z['body']=(z.close-z.open).abs()
    if 'body_frac' not in z:z['body_frac']=(z.body/z['range']).fillna(0)
    if 'atr' not in z:
        prev=z.close.shift();tr=pd.concat([(z.high-z.low).abs(),(z.high-prev).abs(),(z.low-prev).abs()],axis=1).max(axis=1);z['atr']=tr.rolling(14,min_periods=5).mean().bfill()
    if 'ema20' not in z:z['ema20']=z.close.ewm(span=20,adjust=False).mean()
    if 'z30' not in z:z['z30']=(z.close-z.close.rolling(30,min_periods=20).mean())/z.close.rolling(30,min_periods=20).std().replace(0,np.nan)
    if 'compression' not in z:
        med=z.atr.rolling(50,min_periods=20).median();z['compression']=z.atr/med.replace(0,np.nan)
    if 'rng12_hi' not in z:z['rng12_hi']=z.high.shift(1).rolling(12,min_periods=6).max()
    if 'rng12_lo' not in z:z['rng12_lo']=z.low.shift(1).rolling(12,min_periods=6).min()
    return z.reset_index(drop=True)

def emit_candidate(r4,asset,strategy,family,x,i,dr,q,context,max_bars=36,management='PROTECTED_RUNNER'):
    c=r4.candidate(asset,strategy,x,i,dr,q,context,management=management,max_bars=max_bars)
    if not c:return None
    c['strategy']=strategy;c['strategy_family']=family;c['quality']=float(np.clip(q,0,1));c['context']=context
    c['campaign_id']=f"{asset}|{strategy}|{pd.Timestamp(c['decision_time']).isoformat()}"
    return c

def prior_zones(x,i,side,mode):
    # Causal provenance: zone origin + completed displacement/BOS must predate decision bar i.
    cfg={'strict':{'lookback':72,'max_age':48,'min_disp':.90,'max_touches':0},'balanced':{'lookback':96,'max_age':72,'min_disp':.70,'max_touches':1}}[mode]
    out=[];start=max(14,i-cfg['lookback'])
    for k in range(start,max(start,i-4)):
        origin=x.iloc[k];atr=max(n(origin.atr,0),1e-12)
        if side=='demand' and not (origin.close<=origin.open):continue
        if side=='supply' and not (origin.close>=origin.open):continue
        hist=x.iloc[max(0,k-12):k]
        if len(hist)<6:continue
        prior_hi=float(hist.high.max());prior_lo=float(hist.low.min())
        completed=None;disp=0.
        for j in range(k+1,min(i,k+5)):
            b=x.iloc[j];d=abs(float(b.close-b.open))/max(n(b.atr,atr),1e-12)
            if side=='demand' and b.close>prior_hi and d>=cfg['min_disp']:
                completed=j;disp=d;break
            if side=='supply' and b.close<prior_lo and d>=cfg['min_disp']:
                completed=j;disp=d;break
        if completed is None:continue
        age=i-completed
        if age>cfg['max_age']:continue
        if side=='demand':zl=float(origin.low);zh=float(max(origin.open,origin.close))
        else:zl=float(min(origin.open,origin.close));zh=float(origin.high)
        post=x.iloc[completed+1:i]
        invalid=(post.close<zl).any() if side=='demand' else (post.close>zh).any()
        if invalid:continue
        touches=0
        if len(post):
            if side=='demand':touches=int(((post.low<=zh)&(post.high>=zl)).sum())
            else:touches=int(((post.high>=zl)&(post.low<=zh)).sum())
        if touches>cfg['max_touches']:continue
        out.append({'k':k,'completed':completed,'low':zl,'high':zh,'age':age,'touches':touches,'disp':disp})
    return sorted(out,key=lambda z:(z['age'],z['touches'],-z['disp']))

def gen_supply(asset,x,r4,mode):
    if mode=='off':return []
    rows=[];seen=set();strict=mode=='strict'
    for i in range(60,len(x)-2):
        r=x.iloc[i];prev=x.iloc[i-1];atr=max(n(r.atr,0),1e-12);trend=int(n(r.get('h1_trend'),0));strength=max(0,n(r.get('h1_strength'),0))
        if trend==1:
            zs=prior_zones(x,i,'demand',mode)
            if zs:
                z=zs[0];inside=(r.low<=z['high'] and r.high>=z['low']);reclaim=r.close>z['high'];liq=r.low<min(float(prev.low),float(x.low.iloc[max(0,i-6):i].min()));bos=r.close>prev.high;body=r.body_frac>=(.52 if strict else .42)
                if inside and reclaim and liq and bos and body:
                    key=(pd.Timestamp(r.timestamp).floor('30min'),1)
                    if key not in seen:
                        q=.66+.06*min(z['disp'],2)+.04*min(strength,2)+(.05 if z['touches']==0 else 0)-.001*z['age'];ctx=f"R55_DEMAND_ZONE;origin={x.timestamp.iloc[z['k']]};disp={z['disp']:.2f};age={z['age']};touches={z['touches']};LIQ_SWEEP;RECLAIM;BOS"
                        c=emit_candidate(r4,asset,'SUPPLY_DEMAND_SWEEP_BOS','SUPPLY_DEMAND',x,i,1,q,ctx,max_bars=42)
                        if c:rows.append(c);seen.add(key)
        if trend==-1:
            zs=prior_zones(x,i,'supply',mode)
            if zs:
                z=zs[0];inside=(r.high>=z['low'] and r.low<=z['high']);reclaim=r.close<z['low'];liq=r.high>max(float(prev.high),float(x.high.iloc[max(0,i-6):i].max()));bos=r.close<prev.low;body=r.body_frac>=(.52 if strict else .42)
                if inside and reclaim and liq and bos and body:
                    key=(pd.Timestamp(r.timestamp).floor('30min'),-1)
                    if key not in seen:
                        q=.66+.06*min(z['disp'],2)+.04*min(strength,2)+(.05 if z['touches']==0 else 0)-.001*z['age'];ctx=f"R55_SUPPLY_ZONE;origin={x.timestamp.iloc[z['k']]};disp={z['disp']:.2f};age={z['age']};touches={z['touches']};LIQ_SWEEP;RECLAIM;BOS"
                        c=emit_candidate(r4,asset,'SUPPLY_DEMAND_SWEEP_BOS','SUPPLY_DEMAND',x,i,-1,q,ctx,max_bars=42)
                        if c:rows.append(c);seen.add(key)
    return rows

def gen_mean(asset,x,r4,mode):
    if mode=='off':return []
    rows=[];seen=set();strict=mode=='strict'
    zthr=2.25 if strict else 2.0;max_strength=.35 if strict else .50;max_vol=1.25 if strict else 1.45
    for i in range(61,len(x)-2):
        r=x.iloc[i];p=x.iloc[i-1];z=n(r.z30);pz=n(p.z30);strength=abs(n(r.get('h1_strength'),0));comp=n(r.compression,1)
        if not np.isfinite(z) or not np.isfinite(pz) or strength>max_strength or comp>max_vol:continue
        # Exhaustion must already have occurred and decision bar must re-enter value; no blind fade of a still-expanding extreme.
        if pz>=zthr and z<pz and z<zthr and r.close<r.open and r.close<p.close:
            swept=p.high>max(float(x.high.iloc[max(0,i-12):i-1].max()),float(p.open))
            wick=(p.high-max(p.open,p.close))/max(float(p.high-p.low),1e-12)
            value_reentry=r.close<=p.close and r.close>r.ema20 if strict else r.close<=p.close
            if swept and wick>=(.30 if strict else .20) and value_reentry:
                key=(pd.Timestamp(r.timestamp).floor('30min'),-1)
                if key not in seen:
                    q=.62+.06*min(abs(pz)-2,2)+.08*min(wick,1)+.05*max(0,1-comp);ctx=f"R55_EXHAUST_HIGH;peak_z={pz:.2f};reentry_z={z:.2f};wick={wick:.2f};compression={comp:.2f};VALUE_REENTRY"
                    c=emit_candidate(r4,asset,'MEAN_REVERSION_EXTREME','MEAN_REVERSION',x,i,-1,q,ctx,max_bars=30,management='PARTIAL_RUNNER')
                    if c:rows.append(c);seen.add(key)
        if pz<=-zthr and z>pz and z>-zthr and r.close>r.open and r.close>p.close:
            swept=p.low<min(float(x.low.iloc[max(0,i-12):i-1].min()),float(p.open))
            wick=(min(p.open,p.close)-p.low)/max(float(p.high-p.low),1e-12)
            value_reentry=r.close>=p.close and r.close<r.ema20 if strict else r.close>=p.close
            if swept and wick>=(.30 if strict else .20) and value_reentry:
                key=(pd.Timestamp(r.timestamp).floor('30min'),1)
                if key not in seen:
                    q=.62+.06*min(abs(pz)-2,2)+.08*min(wick,1)+.05*max(0,1-comp);ctx=f"R55_EXHAUST_LOW;peak_z={pz:.2f};reentry_z={z:.2f};wick={wick:.2f};compression={comp:.2f};VALUE_REENTRY"
                    c=emit_candidate(r4,asset,'MEAN_REVERSION_EXTREME','MEAN_REVERSION',x,i,1,q,ctx,max_bars=30,management='PARTIAL_RUNNER')
                    if c:rows.append(c);seen.add(key)
    return rows

def rebuild_all(fcache,r4,r4a,cfg):
    rows=[];by=[]
    for asset,x0 in fcache.items():
        x=prep(x0)
        sd=[];mr=[]
        if r4a.family_applicable(asset,'SUPPLY_DEMAND'):sd=gen_supply(asset,x,r4,cfg['sd'])
        if r4a.family_applicable(asset,'MEAN_REVERSION'):mr=gen_mean(asset,x,r4,cfg['mr'])
        rows.extend(sd);rows.extend(mr);by.append({'asset':asset,'sd_candidates':len(sd),'mr_candidates':len(mr)})
    return pd.DataFrame(rows),pd.DataFrame(by)

class Variant:
    def __init__(self,base,promote):self.base=base;self.promote=promote
    def promote_for_week(self,cands,week_start):return self.promote(cands,week_start,'combined')
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):return self.base.replay_r5(cands,specs,start,feature_cache)

def raw_stats(cands):
    if cands.empty:return []
    out=[]
    for s,g in cands.groupby('strategy'):
        out.append({'strategy':s,'candidates':len(g),'mean_net_r':float(g.net_r.mean()),'median_net_r':float(g.net_r.median()),'win_rate':float((g.net_r>0).mean()),'flat_rate':float((g.net_r==0).mean())})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--variant-index',type=int,required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    cfg=VARIANTS[a.variant_index]
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    import trailaris_r4_asset_adapter as r4a
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_lane3_ctl_{a.variant_index}')
    ctl,CW,CEV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 control reproduction failed')
    rebuilt,byasset=rebuild_all(fcache,r4,r4a,cfg)
    base_other=cands[~cands.strategy.isin(TARGETS)].copy()
    if len(rebuilt):
        rebuilt['decision_time']=pd.to_datetime(rebuilt.decision_time,utc=True);rebuilt['exit_time']=pd.to_datetime(rebuilt.exit_time,utc=True)
        c2=pd.concat([base_other,rebuilt],ignore_index=True,sort=False)
    else:c2=base_other
    c2=c2.sort_values('decision_time').reset_index(drop=True)
    cand,VW,VEV,VDE,VPR=eval_variant(Variant(base,promote),data,c2,opps,fcache,specs,100.0)
    fresh_delta=float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct']);end_delta=float(cand['end_equity'])-float(ctl['end_equity'])
    gates={'end_equity_not_regressed':end_delta>=-1e-9,'fresh_holdout_not_regressed':fresh_delta>=-1e-9,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-1e-9,'losses_not_worse':int(cand['losses'])<=int(ctl['losses']),'flats_not_worse':int(cand['flat'])<=int(ctl['flat']),'full_universe_34x510':len(data)==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510}
    status={'state':'R5_5_LANE3_STRATEGY_REBUILD_FULL_REPLAY_COMPLETE','evidence_class':'END_TO_END_CAUSAL_REBUILT_STRATEGY_REPLAY','variant_index':a.variant_index,'config':cfg,'r5_4_control':ctl,'candidate':cand,'delta_end_equity':end_delta,'fresh_delta_pct':fresh_delta,'rebuilt_raw_candidates':int(len(rebuilt)),'rebuilt_raw_stats':raw_stats(rebuilt),'by_asset_nonzero':byasset[(byasset.sd_candidates+byasset.mr_candidates)>0].to_dict('records'),'gates':gates,'candidate_pass':all(gates.values()),'approved_scope':scope,'governance':'R5.4 is untouched. In the R5.5 successor candidate only, the two negative playbooks are rebuilt rather than deleted: supply/demand requires causal zone provenance, displacement/BOS, freshness/touch quality, liquidity sweep and reclaim; mean reversion requires exhaustion, value re-entry, weak trend and volatility regime. Reliability promotion and the full portfolio replay remain unchanged.'}
    (out/f'variant_{a.variant_index:02d}.json').write_text(json.dumps(status,indent=2,default=str));VW.to_csv(out/f'variant_{a.variant_index:02d}_weekly.csv',index=False);byasset.to_csv(out/f'variant_{a.variant_index:02d}_by_asset.csv',index=False);print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
