#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, hashlib, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import trailaris_r4_asset_adapter as r4a

# Trailaris R4 asset-native full-opportunity research runner.
# Evidence firewall: SMOKE proves mechanics only. EMPIRICAL requires 34 authentic paths/specs.
ROUTES=[
"EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD",
"EURJPY","GBPJPY","EURGBP","AUDJPY","CADJPY","GBPCHF",
"XAUUSD","XAGUSD","BTCUSD","ETHUSD","SOLUSD","BNBUSD","XRPUSD",
"US100","US500","US30","GER40","UK100","JP225","WTI","BRENT","NATGAS",
"V75","V50","V100","V25","V10"]
CONTINUOUS={"BTCUSD","ETHUSD","SOLUSD","BNBUSD","XRPUSD","V75","V50","V100","V25","V10"}
FX=set(ROUTES[:13]); METALS={"XAUUSD","XAGUSD"}; CRYPTO={"BTCUSD","ETHUSD","SOLUSD","BNBUSD","XRPUSD"}
INDICES={"US100","US500","US30","GER40","UK100","JP225"}; ENERGY={"WTI","BRENT","NATGAS"}; SYNTH={"V75","V50","V100","V25","V10"}
TARGET_START=pd.Timestamp("2026-05-04T00:00:00Z"); TARGET_END=pd.Timestamp("2026-08-12T23:59:59Z")

STRATEGY_FAMILIES=[
"SUPPLY_DEMAND","LIQUIDITY_STRUCTURE","TREND_FOLLOWING","MOMENTUM_PULLBACK","BREAKOUT_RETEST",
"MEAN_REVERSION","VOLATILITY_REGIME","SESSION_TIME","ORDER_FLOW_MICROSTRUCTURE","RELATIVE_VALUE_PAIRS",
"INTERMARKET","CARRY_VALUE","CRYPTO_FUNDING_BASIS","EVENT_RESPONSE","ENSEMBLE_REGIME"]

PLAYBOOK_TO_FAMILY={
"DAY_OPEN_DRIVE_CONTINUATION":"SESSION_TIME","DAY_OPEN_DRIVE_FADE":"SESSION_TIME",
"SESSION_OPEN_RANGE_BREAKOUT":"SESSION_TIME","SESSION_OPEN_SWEEP_RECLAIM":"LIQUIDITY_STRUCTURE",
"HTF_TREND_BREAKOUT":"TREND_FOLLOWING","HTF_PULLBACK_CONTINUATION":"MOMENTUM_PULLBACK",
"COMPRESSION_EXPANSION":"VOLATILITY_REGIME","BREAKOUT_RETEST":"BREAKOUT_RETEST","FAILED_BREAK_REVERSAL":"LIQUIDITY_STRUCTURE",
"MEAN_REVERSION_EXTREME":"MEAN_REVERSION","SUPPLY_DEMAND_SWEEP_BOS":"SUPPLY_DEMAND",
"CLOSE_SESSION_CONTINUATION":"SESSION_TIME","CLOSE_SESSION_FADE":"MEAN_REVERSION",
"INTRADAY_SWING_CONTINUATION":"TREND_FOLLOWING","VOLATILITY_REVERSAL":"VOLATILITY_REGIME",
"RELATIVE_VALUE_ROTATION":"RELATIVE_VALUE_PAIRS","INTERMARKET_LEAD_LAG":"INTERMARKET","ORDER_FLOW_FIXTURE":"ORDER_FLOW_MICROSTRUCTURE","CARRY_VALUE_FIXTURE":"CARRY_VALUE","FUNDING_BASIS_FIXTURE":"CRYPTO_FUNDING_BASIS","EVENT_RESPONSE_FIXTURE":"EVENT_RESPONSE","ENSEMBLE_CONFLUENCE":"ENSEMBLE_REGIME","WINNER_ONLY_ADDON":"CAMPAIGN_MANAGEMENT"}


FACTORS=["USD","EUR","GBP","JPY","CHF","AUD","NZD","CAD","METAL","CRYPTO","ENERGY","EQUITY","SYNTH"]

# ---------------- data/spec helpers ----------------
def read_route(path:Path)->pd.DataFrame:
    d=pd.read_csv(path)
    tsraw=d["timestamp"]
    if pd.api.types.is_numeric_dtype(tsraw):
        vals=pd.to_numeric(tsraw,errors="coerce")
        med=vals.dropna().abs().median() if vals.notna().any() else 0
        unit="ms" if med>1e11 else "s"
        d["timestamp"]=pd.to_datetime(vals,unit=unit,utc=True,errors="coerce")
    else:
        d["timestamp"]=pd.to_datetime(tsraw,utc=True,errors="coerce")
    for c in ["open","high","low","close","spread","bid","ask","volume"]:
        if c in d.columns:d[c]=pd.to_numeric(d[c],errors="coerce")
    if "spread" not in d:d["spread"]=0.0
    d["spread"]=d["spread"].fillna(0.0)
    return d.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def validate_34(rawdir:Path,mode:str)->Tuple[Dict[str,pd.DataFrame],pd.DataFrame]:
    data={}; rows=[]
    for a in ROUTES:
        p=rawdir/f"{a}_M1_normalized.csv"
        if not p.exists(): rows.append({"asset":a,"state":"MISSING","rows":0,"reason":"path absent"}); continue
        try:d=read_route(p)
        except Exception as e:rows.append({"asset":a,"state":"INVALID","rows":0,"reason":str(e)});continue
        reason=[]
        src=str(d.get("source",pd.Series([""])).iloc[0]) if len(d) else ""
        med=d.timestamp.diff().dropna().dt.total_seconds().median()
        if mode in {"empirical","research-proxy"}:
            if "SMOKE" in src.upper() or "FIXTURE" in src.upper():reason.append("fixture rejected")
            if d.timestamp.min()>TARGET_START+pd.Timedelta(days=1) or d.timestamp.max()<TARGET_END-pd.Timedelta(days=1):reason.append("target window incomplete")
            if pd.notna(med) and med>75:reason.append("not M1")
        inv=((d.high<d.low)|(d.high<d[["open","close"]].max(axis=1))|(d.low>d[["open","close"]].min(axis=1))).sum()
        if inv:reason.append(f"invalid_ohlc={int(inv)}")
        st="PASS" if not reason else "INVALID"
        rows.append({"asset":a,"state":st,"rows":len(d),"first":str(d.timestamp.min()),"last":str(d.timestamp.max()),"median_sec":float(med) if pd.notna(med) else None,"source":src,"reason":";".join(reason)})
        if st=="PASS":data[a]=d
    return data,pd.DataFrame(rows)

SPEC_COLS=["asset","contract_size","tick_size","tick_value","volume_min","volume_step","volume_max","margin_per_lot"]
def load_specs(path:Path,mode:str)->pd.DataFrame:
    s=pd.read_csv(path)
    if not set(SPEC_COLS).issubset(s.columns):raise ValueError("contract spec columns missing")
    if set(s.asset)!=set(ROUTES):raise ValueError("contract spec route set !=34")
    if mode=="empirical" and "evidence_state" in s.columns:
        bad=s[s.evidence_state.astype(str).str.contains("FIXTURE|PROXY|DATA_REQUIRED|UNKNOWN",case=False,regex=True)]
        if len(bad):raise ValueError("non-authoritative contract specs: "+",".join(bad.asset))
    for c in SPEC_COLS[1:]:s[c]=pd.to_numeric(s[c],errors="raise")
    return s.set_index("asset")

def fixture_specs(path:Path):
    rows=[]
    for a in ROUTES:
        if a in FX: tick=.00001 if "JPY" not in a else .001;tv=1;vmin=.001;step=.001;margin=.30;cs=100000
        elif a in METALS:tick=.01;tv=1;vmin=.01;step=.01;margin=5;cs=100
        elif a in CRYPTO:tick=.01;tv=.01;vmin=.01;step=.01;margin=2;cs=1
        elif a in INDICES:tick=.1;tv=.1;vmin=.01;step=.01;margin=2;cs=1
        elif a in ENERGY:tick=.01;tv=1;vmin=.01;step=.01;margin=4;cs=100
        else:tick=.01;tv=.01;vmin=.001;step=.001;margin=.5;cs=1
        rows.append(dict(asset=a,contract_size=cs,tick_size=tick,tick_value=tv,volume_min=vmin,volume_step=step,volume_max=100,margin_per_lot=margin,evidence_state="SMOKE_FIXTURE_ONLY"))
    pd.DataFrame(rows).to_csv(path,index=False)

def round_lot(v,row):
    if v<row.volume_min:return 0.0
    return float(min(row.volume_max, row.volume_min+math.floor((v-row.volume_min+1e-12)/row.volume_step)*row.volume_step))

def size_trade(asset,equity,risk_frac,stop_dist,specs):
    r=specs.loc[asset]; cash_per_lot=(stop_dist/r.tick_size)*r.tick_value
    if cash_per_lot<=0:return 0,0,0
    lot=round_lot((equity*risk_frac)/cash_per_lot,r); risk=lot*cash_per_lot; margin=lot*r.margin_per_lot
    return lot,risk,margin

# ---------------- feature engineering ----------------
def atr(d,n=14):
    prev=d.close.shift();tr=pd.concat([(d.high-d.low).abs(),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(n,min_periods=max(3,n//3)).mean().bfill()

def resample(d,rule):
    x=d.set_index("timestamp")
    y=x.resample(rule,label="right",closed="right").agg({"open":"first","high":"max","low":"min","close":"last","spread":"mean","volume":"sum" if "volume" in x else "size"}).dropna(subset=["open","high","low","close"]).reset_index()
    return y

def add_features(m5):
    x=m5.copy();x["atr"]=atr(x,14);x["ema20"]=x.close.ewm(span=20,adjust=False).mean();x["ema50"]=x.close.ewm(span=50,adjust=False).mean()
    x["rng20_hi"]=x.high.shift(1).rolling(20,min_periods=10).max();x["rng20_lo"]=x.low.shift(1).rolling(20,min_periods=10).min()
    x["rng12_hi"]=x.high.shift(1).rolling(12,min_periods=6).max();x["rng12_lo"]=x.low.shift(1).rolling(12,min_periods=6).min()
    x["body"]=(x.close-x.open).abs();x["range"]=(x.high-x.low).replace(0,np.nan);x["body_frac"]=(x.body/x.range).fillna(0)
    x["ret1"]=x.close.pct_change();x["z30"]=(x.close-x.close.rolling(30,min_periods=20).mean())/x.close.rolling(30,min_periods=20).std().replace(0,np.nan)
    x["atr_med50"]=x.atr.rolling(50,min_periods=20).median();x["compression"]=x.atr/(x.atr_med50.replace(0,np.nan))
    x["hour"]=x.timestamp.dt.hour+x.timestamp.dt.minute/60
    return x

def htf_state(d):
    h1=resample(d,"1h");h1["atr"]=atr(h1,14);h1["ema20"]=h1.close.ewm(span=20,adjust=False).mean();h1["ema50"]=h1.close.ewm(span=50,adjust=False).mean()
    h1["trend"]=np.where(h1.ema20>h1.ema50,1,np.where(h1.ema20<h1.ema50,-1,0));h1["strength"]=(h1.ema20-h1.ema50).abs()/h1.atr.replace(0,np.nan)
    return h1[["timestamp","trend","strength","atr","ema20","ema50"]]

# ---------------- sessions ----------------
def sessions_for(asset):
    if asset in FX or asset in METALS:return [("TOKYO",0.0),("LONDON",7.0),("NEW_YORK",13.5)]
    if asset in {"US100","US500","US30"}:return [("US_CASH",13.5)]
    if asset in {"GER40","UK100"}:return [("EU_CASH",7.0)]
    if asset=="JP225":return [("TOKYO_CASH",0.0)]
    if asset in ENERGY:return [("LONDON",7.0),("NEW_YORK",13.5)]
    if asset in CRYPTO:return [("UTC_ASIA",0.0),("UTC_EUROPE",8.0),("UTC_US",14.0)]
    return [("BLOCK_A",0.0),("BLOCK_B",8.0),("BLOCK_C",16.0)]

def close_hour(asset):
    if asset in {"US100","US500","US30"}:return 20.0
    if asset in {"GER40","UK100"}:return 15.5
    if asset=="JP225":return 6.0
    if asset in FX or asset in METALS:return 21.0
    if asset in ENERGY:return 20.0
    return 23.5

# ---------------- outcome simulator ----------------
def simulate_trade(x,i,direction,stop_mult=1.15,target_r=3.0,max_bars=48,management="PROTECTED_RUNNER",time_exit=None):
    if i>=len(x)-2:return None
    ent_i=i+1; ent=float(x.open.iloc[ent_i]); a=float(x.atr.iloc[i]);
    if not np.isfinite(a) or a<=0:return None
    sd=max(a*stop_mult,float(x.spread.iloc[i])*4,1e-12); stop=ent-direction*sd; target=ent+direction*target_r*sd
    remaining=1.0; realized_r=0.0; floor_r=-1.0; mfe=0.;mae=0.;giveback=0.;addon_signal=None;reason="TIME"
    exit_time=x.timestamp.iloc[min(len(x)-1,ent_i+max_bars)]; exit_px=float(x.close.iloc[min(len(x)-1,ent_i+max_bars)])
    last_extreme=ent
    for j in range(ent_i,min(len(x),ent_i+max_bars+1)):
        b=x.iloc[j]
        fav=(float(b.high)-ent) if direction==1 else (ent-float(b.low)); adv=(ent-float(b.low)) if direction==1 else (float(b.high)-ent)
        mfe=max(mfe,fav/sd);mae=max(mae,adv/sd)
        # partial/protection sequence, causally activated once touched
        if management=="PROTECTED_RUNNER":
            if mfe>=.75 and floor_r<0: floor_r=0.0
            if mfe>=1.5 and floor_r<.5: floor_r=.5
            if mfe>=2.0: floor_r=max(floor_r,mfe-.8)
        elif management=="PARTIAL_RUNNER":
            if mfe>=.75 and remaining>0.81: realized_r+=.20*.75;remaining-=.20;floor_r=max(floor_r,0.0)
            if mfe>=1.5 and remaining>0.61: realized_r+=.20*1.5;remaining-=.20;floor_r=max(floor_r,.5)
            if mfe>=2.0: floor_r=max(floor_r,mfe-.9)
        # winner-only add-on trigger after protected base and fresh continuation break
        if addon_signal is None and mfe>=.80 and floor_r>=0 and j>=ent_i+4:
            prev=x.iloc[max(ent_i,j-5):j]
            if direction==1 and float(b.close)>float(prev.high.max()) and b.body_frac>=.45:addon_signal=j
            if direction==-1 and float(b.close)<float(prev.low.min()) and b.body_frac>=.45:addon_signal=j
        # conservative intrabar ordering: protective stop before target
        eff_stop=ent+direction*floor_r*sd if floor_r>-1 else stop
        stopped=(float(b.low)<=eff_stop if direction==1 else float(b.high)>=eff_stop)
        targeted=(float(b.high)>=target if direction==1 else float(b.low)<=target)
        if stopped:
            rr=(eff_stop-ent)*direction/sd;realized_r+=remaining*rr;exit_px=eff_stop;exit_time=b.timestamp;reason="PROTECTED_STOP" if floor_r>=0 else "STOP";remaining=0;break
        if targeted:
            realized_r+=remaining*target_r;exit_px=target;exit_time=b.timestamp;reason="TARGET";remaining=0;break
        if time_exit is not None and b.timestamp>=time_exit:
            rr=(float(b.close)-ent)*direction/sd;realized_r+=remaining*rr;exit_px=float(b.close);exit_time=b.timestamp;reason="TIME_RULE";remaining=0;break
    if remaining>0:
        rr=(exit_px-ent)*direction/sd;realized_r+=remaining*rr
    cost_r=max(0,float(x.spread.iloc[i])/sd);net_r=realized_r-cost_r
    giveback=max(0,mfe-max(0,realized_r))
    return dict(entry_time=x.timestamp.iloc[ent_i],decision_time=x.timestamp.iloc[i],exit_time=exit_time,direction=int(direction),entry_price=ent,exit_price=float(exit_px),stop_distance=sd,gross_r=float(realized_r),cost_r=float(cost_r),net_r=float(net_r),mfe_r=float(mfe),mae_r=float(mae),giveback_r=float(giveback),exit_reason=reason,addon_trigger_index=addon_signal)

# ---------------- signal generation ----------------
def candidate(asset,strategy,x,i,direction,quality,context,management="PROTECTED_RUNNER",max_bars=48,time_exit=None,parent_id=""):
    o=simulate_trade(x,i,direction,max_bars=max_bars,management=management,time_exit=time_exit)
    if not o:return None
    o.update(asset=asset,strategy=strategy,strategy_family=PLAYBOOK_TO_FAMILY.get(strategy,"UNMAPPED"),quality=float(np.clip(quality,0,1)),context=context,management=management,parent_id=parent_id,is_addon=False)
    o["campaign_id"]=f"{asset}|{strategy}|{pd.Timestamp(o['decision_time']).isoformat()}"
    return o

def generate_candidates(asset,d):
    m5=add_features(resample(d,"5min")); h1=htf_state(d)
    x=pd.merge_asof(m5.sort_values("timestamp"),h1.rename(columns={"trend":"h1_trend","strength":"h1_strength","atr":"h1_atr","ema20":"h1_ema20","ema50":"h1_ema50"}).sort_values("timestamp"),on="timestamp",direction="backward")
    x["date"]=x.timestamp.dt.floor("D")
    rows=[]; seen=set()
    def emit(strategy,i,dir_,q,ctx,management="PROTECTED_RUNNER",max_bars=48,time_exit=None):
        key=(strategy,x.date.iloc[i],int(i//6)) # cooldown bucket ~30m
        if key in seen:return
        c=candidate(asset,strategy,x,i,dir_,q,ctx,management,max_bars,time_exit)
        if c:rows.append(c);seen.add(key)
    # Per-bar structural families. Allows multiple opportunities/day across families.
    for i in range(55,len(x)-2):
        r=x.iloc[i]; prev=x.iloc[i-1]
        if not np.isfinite(r.atr) or r.atr<=0:continue
        trend=int(r.h1_trend) if pd.notna(r.h1_trend) else 0; strength=float(r.h1_strength) if pd.notna(r.h1_strength) else 0
        # HTF trend breakout
        if trend==1 and r.close>r.rng20_hi and r.body_frac>.45:emit("HTF_TREND_BREAKOUT",i,1,.55+min(.35,strength/4),"H1_UP+M5_20BAR_BREAK")
        if trend==-1 and r.close<r.rng20_lo and r.body_frac>.45:emit("HTF_TREND_BREAKOUT",i,-1,.55+min(.35,strength/4),"H1_DOWN+M5_20BAR_BREAK")
        # Pullback continuation
        if trend==1 and prev.low<=prev.ema20 and r.close>r.ema20 and r.close>prev.high and r.body_frac>.45:emit("HTF_PULLBACK_CONTINUATION",i,1,.62,"EMA20_RECLAIM_UP")
        if trend==-1 and prev.high>=prev.ema20 and r.close<r.ema20 and r.close<prev.low and r.body_frac>.45:emit("HTF_PULLBACK_CONTINUATION",i,-1,.62,"EMA20_RECLAIM_DOWN")
        # Compression expansion
        if r.compression<.78 and i+1<len(x):
            pass
        if prev.compression<.8 and r.close>r.rng12_hi and r.body_frac>.5:emit("COMPRESSION_EXPANSION",i,1,.66,"ATR_COMPRESSION_BREAK_UP")
        if prev.compression<.8 and r.close<r.rng12_lo and r.body_frac>.5:emit("COMPRESSION_EXPANSION",i,-1,.66,"ATR_COMPRESSION_BREAK_DOWN")
        # completed break + retest family (separate from first-break momentum).
        if i>=2:
            p2=x.iloc[i-2];p1=x.iloc[i-1]
            level_hi=p2.rng20_hi;level_lo=p2.rng20_lo
            if pd.notna(level_hi) and p1.close>level_hi and r.low<=level_hi and r.close>level_hi and r.body_frac>.30:
                emit("BREAKOUT_RETEST",i,1,.68,"20BAR_BREAK_RETEST_UP",max_bars=36)
            if pd.notna(level_lo) and p1.close<level_lo and r.high>=level_lo and r.close<level_lo and r.body_frac>.30:
                emit("BREAKOUT_RETEST",i,-1,.68,"20BAR_BREAK_RETEST_DOWN",max_bars=36)
        # failed break
        if r.high>r.rng20_hi and r.close<r.rng20_hi and r.body_frac>.25:emit("FAILED_BREAK_REVERSAL",i,-1,.60,"UPSIDE_SWEEP_RECLAIM")
        if r.low<r.rng20_lo and r.close>r.rng20_lo and r.body_frac>.25:emit("FAILED_BREAK_REVERSAL",i,1,.60,"DOWNSIDE_SWEEP_RECLAIM")
        # mean reversion only when H1 trend weak
        if strength<.45 and pd.notna(r.z30):
            if r.z30>2.0:emit("MEAN_REVERSION_EXTREME",i,-1,.55+min(.25,(r.z30-2)/4),"Z30_HIGH_WEAK_HTF",management="PARTIAL_RUNNER",max_bars=30)
            elif r.z30<-2.0:emit("MEAN_REVERSION_EXTREME",i,1,.55+min(.25,(-r.z30-2)/4),"Z30_LOW_WEAK_HTF",management="PARTIAL_RUNNER",max_bars=30)
        # approximate S/D sweep + BOS: sweep 12-range and displacement in HTF direction
        if trend==1 and r.low<r.rng12_lo and r.close>prev.high and r.body_frac>.55:emit("SUPPLY_DEMAND_SWEEP_BOS",i,1,.72,"HTF_DEMAND_SWEEP+BOS")
        if trend==-1 and r.high>r.rng12_hi and r.close<prev.low and r.body_frac>.55:emit("SUPPLY_DEMAND_SWEEP_BOS",i,-1,.72,"HTF_SUPPLY_SWEEP+BOS")
        # intraday swing continuation
        mom=(r.close-x.close.iloc[max(0,i-6)])/max(r.atr,1e-12)
        if trend==1 and mom>.8 and r.close>prev.high and r.body_frac>.5:emit("INTRADAY_SWING_CONTINUATION",i,1,.58,"6BAR_MOMENTUM_UP",max_bars=24)
        if trend==-1 and mom<-.8 and r.close<prev.low and r.body_frac>.5:emit("INTRADAY_SWING_CONTINUATION",i,-1,.58,"6BAR_MOMENTUM_DOWN",max_bars=24)
        # volatility reversal
        imp=(r.close-r.open)/max(r.atr,1e-12)
        if imp>1.4 and i+1<len(x) and x.close.iloc[i+1]<r.open:emit("VOLATILITY_REVERSAL",i+1,-1,.57,"IMPULSE_EXHAUST_UP",max_bars=18)
        if imp<-1.4 and i+1<len(x) and x.close.iloc[i+1]>r.open:emit("VOLATILITY_REVERSAL",i+1,1,.57,"IMPULSE_EXHAUST_DOWN",max_bars=18)
    # Session/day open families
    for day,g in x.groupby("date"):
        if len(g)<20:continue
        idxs=g.index.to_list(); first=idxs[0]
        # Opening-of-day drive/rejection. This captures opening dislocation even when
        # the feed is continuous and therefore has no literal bar-to-bar gap.
        prev_rows=x[x.timestamp<g.timestamp.iloc[0]]
        if len(prev_rows) and len(idxs)>=4:
            prev_close=float(prev_rows.close.iloc[-1]); confirm_idx=idxs[2]; first_idx=idxs[0]
            drive=(float(x.close.loc[confirm_idx])-prev_close)/max(float(g.atr.iloc[0]),1e-12)
            if abs(drive)>.25:
                dr=1 if drive>0 else -1
                follow=(float(x.close.loc[idxs[3]])-float(x.close.loc[confirm_idx]))*dr
                if follow>=0:emit("DAY_OPEN_DRIVE_CONTINUATION",idxs[3],dr,min(.82,.55+abs(drive)/4),f"OPEN_DRIVE_ATR={drive:.2f}",max_bars=24)
                else:emit("DAY_OPEN_DRIVE_FADE",idxs[3],-dr,min(.82,.55+abs(drive)/4),f"OPEN_REJECT_ATR={drive:.2f}",max_bars=24)
        # session ORB/sweep
        for sname,start in r4a.session_starts(asset,day):
            opening=x[(x.timestamp>=start)&(x.timestamp<start+pd.Timedelta(minutes=30))]
            after=x[(x.timestamp>=start+pd.Timedelta(minutes=30))&(x.timestamp<start+pd.Timedelta(hours=3))]
            if len(opening)<3 or len(after)<2:continue
            hi=float(opening.high.max());lo=float(opening.low.min());rg=max(hi-lo,1e-12)
            for ii in after.index:
                rr=x.loc[ii]
                if rr.close>hi and rr.body_frac>.45:
                    emit("SESSION_OPEN_RANGE_BREAKOUT",ii,1,.65,f"{sname}_ORB_UP",max_bars=30);break
                if rr.close<lo and rr.body_frac>.45:
                    emit("SESSION_OPEN_RANGE_BREAKOUT",ii,-1,.65,f"{sname}_ORB_DOWN",max_bars=30);break
            for ii in after.index:
                rr=x.loc[ii]
                if rr.high>hi+.10*rg and rr.close<hi:
                    emit("SESSION_OPEN_SWEEP_RECLAIM",ii,-1,.67,f"{sname}_SWEEP_HIGH",max_bars=24);break
                if rr.low<lo-.10*rg and rr.close>lo:
                    emit("SESSION_OPEN_SWEEP_RECLAIM",ii,1,.67,f"{sname}_SWEEP_LOW",max_bars=24);break
        # session/day close families: late impulse -> next available open/time exit
        close_ts=r4a.close_timestamp(asset,day); late=g[(g.timestamp>=close_ts-pd.Timedelta(minutes=30))&(g.timestamp<=close_ts+pd.Timedelta(minutes=6))]
        if len(late)>=3:
            li=late.index[-1]; lr=x.loc[li]; recent=g[g.index<=li].tail(12)
            move=(float(lr.close)-float(recent.open.iloc[0]))/max(float(lr.atr),1e-12)
            future=x[x.timestamp>lr.timestamp]
            if len(future):
                texit=future.timestamp.iloc[min(len(future)-1,12)]
                if abs(move)>.8:
                    dr=1 if move>0 else -1
                    emit("CLOSE_SESSION_CONTINUATION",li,dr,min(.8,.55+abs(move)/5),f"LATE_MOVE_ATR={move:.2f}",max_bars=18,time_exit=texit)
                if abs(move)>1.4 and (float(lr.close)-float(lr.ema20))*np.sign(move)>0:
                    emit("CLOSE_SESSION_FADE",li,-(1 if move>0 else -1),.58,f"LATE_EXTENSION_ATR={move:.2f}",management="PARTIAL_RUNNER",max_bars=18,time_exit=texit)
    c=pd.DataFrame(rows)
    if c.empty:return c,x
    # Build winner-only add-on child candidates from base outcomes. Child is causal at trigger time and only admitted if parent selected/open.
    addons=[]
    for _,r in c.iterrows():
        if pd.isna(r.addon_trigger_index):continue
        j=int(r.addon_trigger_index)
        if j>=len(x)-2:continue
        aout=simulate_trade(x,j,int(r.direction),stop_mult=.9,target_r=2.0,max_bars=24,management="PROTECTED_RUNNER")
        if not aout:continue
        aout.update(asset=asset,strategy="WINNER_ONLY_ADDON",strategy_family="CAMPAIGN_MANAGEMENT",quality=min(.9,float(r.quality)+.05),context=f"PARENT={r.campaign_id}",management="PROTECTED_RUNNER",parent_id=r.campaign_id,is_addon=True)
        aout["campaign_id"]=f"{asset}|WINNER_ONLY_ADDON|{pd.Timestamp(aout['decision_time']).isoformat()}|{hash(r.campaign_id)%100000}"
        addons.append(aout)
    if addons:c=pd.concat([c,pd.DataFrame(addons)],ignore_index=True)
    return c.sort_values("decision_time").reset_index(drop=True),x


# ---------------- cross-asset/factor families ----------------
def nearest_feature_index(x,ts):
    arr=x.timestamp.searchsorted(pd.Timestamp(ts))
    if arr>=len(x)-2:arr=len(x)-3
    return max(0,int(arr))

def generate_relative_value_candidates(feature_cache):
    rows=[];groups=[list(FX),list(METALS),list(CRYPTO),list(INDICES),list(ENERGY),list(SYNTH)]
    for assets in groups:
        assets=[a for a in assets if a in feature_cache]
        if len(assets)<2:continue
        series={a:feature_cache[a].set_index('timestamp').close.resample('1h').last().pct_change() for a in assets}
        wide=pd.concat(series,axis=1).dropna(how='all')
        for ts,row in wide.iterrows():
            z=(row-row.mean())/(row.std() if pd.notna(row.std()) and row.std()>0 else np.nan)
            if z.isna().all():continue
            for a in [z.idxmax(),z.idxmin()]:
                zv=float(z[a])
                if abs(zv)<1.0:continue
                x=feature_cache[a];i=nearest_feature_index(x,ts);dr=1 if zv>0 else -1
                c=candidate(a,'RELATIVE_VALUE_ROTATION',x,i,dr,min(.82,.58+abs(zv)/8),f'CLASS_RELATIVE_Z={zv:.2f}',max_bars=24)
                if c:rows.append(c)
    return pd.DataFrame(rows)

def generate_intermarket_candidates(feature_cache):
    # Explicit economic/market relationships; signal requires leader impulse plus target lag, not bare correlation.
    rel=[('USDCAD','WTI',-1),('CADJPY','WTI',1),('XAGUSD','XAUUSD',1),('ETHUSD','BTCUSD',1),('SOLUSD','BTCUSD',1),('BNBUSD','BTCUSD',1),('XRPUSD','BTCUSD',1),('US100','US500',1),('US30','US500',1),('GER40','US500',1),('UK100','US500',1),('JP225','US500',1)]
    rows=[]
    for target,leader,sgn in rel:
        if target not in feature_cache or leader not in feature_cache:continue
        xt=feature_cache[target].set_index('timestamp');xl=feature_cache[leader].set_index('timestamp')
        rt=xt.close.resample('1h').last().pct_change();rl=xl.close.resample('1h').last().pct_change()
        q=pd.concat({'target':rt,'leader':rl},axis=1).dropna()
        lstd=q.leader.rolling(24,min_periods=8).std()
        for k,(ts,r) in enumerate(q.iterrows()):
            sd=float(lstd.loc[ts]) if ts in lstd.index and pd.notna(lstd.loc[ts]) else 0
            if sd<=0 or abs(float(r.leader))<1.25*sd:continue
            desired=(1 if r.leader>0 else -1)*sgn
            # target must still lag the leader impulse in normalized magnitude
            if abs(float(r.target))>=.75*abs(float(r.leader)):continue
            x=feature_cache[target];i=nearest_feature_index(x,ts)
            c=candidate(target,'INTERMARKET_LEAD_LAG',x,i,desired,.66,f'LEADER={leader};LEADER_RET={r.leader:.5f};TARGET_RET={r.target:.5f}',max_bars=24)
            if c:rows.append(c)
    return pd.DataFrame(rows)

def load_factor_feed(path:Path):
    if not path.exists():return pd.DataFrame()
    d=pd.read_csv(path);d['timestamp']=pd.to_datetime(d.timestamp,utc=True,errors='coerce')
    return d.dropna(subset=['timestamp','asset']).sort_values('timestamp')

def generate_factor_candidates(feature_cache,factors,mode):
    if factors.empty:return pd.DataFrame()
    rows=[]
    for _,r in factors.iterrows():
        a=str(r.asset)
        if a not in feature_cache:continue
        x=feature_cache[a];i=nearest_feature_index(x,r.timestamp)
        entries=[
            ('orderflow_score','ORDER_FLOW_FIXTURE','ORDER_FLOW_MICROSTRUCTURE'),
            ('carry_score','CARRY_VALUE_FIXTURE','CARRY_VALUE'),
            ('event_score','EVENT_RESPONSE_FIXTURE','EVENT_RESPONSE')]
        if a in CRYPTO:entries.append(('funding_score','FUNDING_BASIS_FIXTURE','CRYPTO_FUNDING_BASIS'))
        for col,play,fam in entries:
            if col not in factors.columns or pd.isna(r.get(col)):continue
            score=float(r[col])
            if abs(score)<.70:continue
            # Empirical factor families may only come from externally supplied evidence rows.
            ev=str(r.get('evidence_state',''))
            if mode in {'empirical','research-proxy'} and not any(k in ev.upper() for k in ['AUTHORITATIVE','VERIFIED','RELEASED','BROKER']):continue
            c=candidate(a,play,x,i,1 if score>0 else -1,min(.88,.60+abs(score)/5),f'{col}={score:.2f};EVIDENCE={ev}',max_bars=36)
            if c:
                c['strategy_family']=fam;rows.append(c)
    return pd.DataFrame(rows)

def generate_ensemble_candidates(cands,feature_cache):
    if cands.empty:return pd.DataFrame()
    x=cands[~cands.strategy_family.isin(['CAMPAIGN_MANAGEMENT','ENSEMBLE_REGIME'])].copy()
    x['bucket']=pd.to_datetime(x.decision_time,utc=True).dt.floor('15min')
    rows=[]
    for (a,b,dr),g in x.groupby(['asset','bucket','direction']):
        fams=set(g.strategy_family)
        if len(fams)<2:continue
        best=g.sort_values('quality',ascending=False).iloc[0];xf=feature_cache[a];i=nearest_feature_index(xf,b)
        q=min(.92,float(best.quality)+.04*min(4,len(fams)-1))
        c=candidate(a,'ENSEMBLE_CONFLUENCE',xf,i,int(dr),q,'FAMILIES='+','.join(sorted(fams)),max_bars=36)
        if c:rows.append(c)
    return pd.DataFrame(rows)

def smoke_factor_feed(feature_cache,path:Path):
    rows=[]
    for j,a in enumerate(ROUTES):
        x=feature_cache[a]
        if x.empty:continue
        days=sorted(x.timestamp.dt.floor('D').unique())
        for k,day in enumerate(days):
            # deterministic fixture factors; mechanics only, never performance evidence.
            ts=pd.Timestamp(day)+pd.Timedelta(hours=12,minutes=30)
            vals=np.sin(np.array([j+1,k+2,j+k+3,j*2+k+4],dtype=float))
            rows.append(dict(timestamp=ts,asset=a,orderflow_score=float(vals[0]),carry_score=float(vals[1]),funding_score=float(vals[2]) if a in CRYPTO else 0.0,event_score=float(vals[3]),evidence_state='SMOKE_FIXTURE_ONLY'))
    pd.DataFrame(rows).to_csv(path,index=False)
    return pd.DataFrame(rows)

def strategy_cell_audit(cands,factor_feed_present,mode):
    rows=[]
    triggered=set(zip(cands.asset,cands.strategy_family)) if len(cands) else set()
    for a in ROUTES:
        for f in STRATEGY_FAMILIES:
            state='EXECUTED_TRIGGERED' if (a,f) in triggered else 'EXECUTED_NO_TRIGGER'
            reason=''
            if f=='ORDER_FLOW_MICROSTRUCTURE' and not factor_feed_present:state='DATA_GATED';reason='BID_ASK_TICKS_OR_DEPTH_REQUIRED'
            elif f=='CARRY_VALUE' and not factor_feed_present:state='DATA_GATED';reason='RATES_SWAP_FORWARD_CURVES_REQUIRED'
            elif f=='CRYPTO_FUNDING_BASIS' and a not in CRYPTO:state='CONDITIONAL_NOT_APPLICABLE';reason='NON_CRYPTO_ROUTE'
            elif f=='CRYPTO_FUNDING_BASIS' and not factor_feed_present:state='DATA_GATED';reason='FUNDING_BASIS_OI_REQUIRED'
            elif f=='EVENT_RESPONSE' and not factor_feed_present:state='DATA_GATED';reason='TIMESTAMPED_RELEASE_FEED_REQUIRED'
            rows.append(dict(asset=a,strategy_family=f,state=state,reason=reason))
    return pd.DataFrame(rows)

# R4 asset-native strategy audit override
def strategy_cell_audit(cands,factor_feed_present,mode):
    rows=[]
    triggered=set(zip(cands.asset,cands.strategy_family)) if len(cands) else set()
    for a in ROUTES:
        for f in STRATEGY_FAMILIES:
            app=r4a.family_applicable(a,f)
            if not app:
                state='NON_NATIVE_FAMILY_GATED';reason='ASSET_CLASS_NOT_NATIVE_FOR_FAMILY'
            else:
                state='EXECUTED_TRIGGERED' if (a,f) in triggered else 'EXECUTED_NO_TRIGGER';reason=''
                req=r4a.evidence_requirement(a,f)
                if mode in {'empirical','research-proxy'} and f in {'ORDER_FLOW_MICROSTRUCTURE','CARRY_VALUE','CRYPTO_FUNDING_BASIS','EVENT_RESPONSE'} and not factor_feed_present:
                    state='DATA_GATED';reason=req
            rows.append(dict(asset=a,strategy_family=f,applicable=app,state=state,reason=reason,required_evidence=r4a.evidence_requirement(a,f)))
    return pd.DataFrame(rows)

# ---------------- opportunity benchmark ----------------
def decompose_path(g):
    z=g.sort_values("timestamp").reset_index(drop=True).copy();z["atr"]=atr(z,14);close=z.close.to_numpy();ts=z.timestamp.to_list();spr=z.spread.fillna(0).to_numpy();av=z.atr.to_numpy()
    if len(z)<10:return []
    direction=0;anchor=close[0];ext=close[0];ci=0;rows=[]
    for i in range(1,len(z)):
        th=max(.55*max(av[i],1e-12),4*max(spr[i],0))
        if direction==0:
            if close[i]-anchor>=th:direction=1;ci=i;anchor=close[i];ext=close[i]
            elif anchor-close[i]>=th:direction=-1;ci=i;anchor=close[i];ext=close[i]
        elif direction==1:
            ext=max(ext,close[i])
            if ext-close[i]>=th:
                gross=max(0,ext-anchor);ex=max(0,gross-4*spr[ci]);rows.append((1,ts[ci],ts[i],gross,ex));direction=-1;ci=i;anchor=close[i];ext=close[i]
        else:
            ext=min(ext,close[i])
            if close[i]-ext>=th:
                gross=max(0,anchor-ext);ex=max(0,gross-4*spr[ci]);rows.append((-1,ts[ci],ts[i],gross,ex));direction=1;ci=i;anchor=close[i];ext=close[i]
    return rows

def opportunity_ledger(asset,d):
    """Daily opportunity benchmark.
    `executable_swing_pct` is the causal decomposition benchmark. Opening/closing
    fields are gross sub-window diagnostics so missed day-open/day-close movement
    is visible rather than hidden inside a single close-to-close number.
    """
    rows=[]
    for day,g in d.groupby(d.timestamp.dt.floor("D")):
        g=g.sort_values("timestamp")
        if len(g)<30:continue
        first=float(g.open.iloc[0]);last=float(g.close.iloc[-1]);den=max(abs(first),1e-12)
        directional=abs(last-first)
        segs=decompose_path(g[["timestamp","open","high","low","close","spread"]])
        swing=sum(s[4] for s in segs)
        # first/last 60 minutes of the asset's observed trading day
        o60=g.iloc[:min(60,len(g))]; c60=g.iloc[max(0,len(g)-60):]
        open60=abs(float(o60.close.iloc[-1])-float(o60.open.iloc[0]))/max(abs(float(o60.open.iloc[0])),1e-12)*100
        close60=abs(float(c60.close.iloc[-1])-float(c60.open.iloc[0]))/max(abs(float(c60.open.iloc[0])),1e-12)*100
        # gross range available around each applicable session/cash open, 90m window.
        sess_range=0.0
        for _,start in r4a.session_starts(asset,day):
            sg=g[(g.timestamp>=start)&(g.timestamp<start+pd.Timedelta(hours=2))]
            if len(sg)>=6:
                sess_range+=(float(sg.high.max())-float(sg.low.min()))/max(abs(float(sg.open.iloc[0])),1e-12)*100
        rows.append(dict(asset=asset,day=day,
                         directional_open_close_move=directional,
                         directional_open_close_pct=100*directional/den,
                         open_60m_move_pct=open60,close_60m_move_pct=close60,
                         gross_session_open_range_pct=sess_range,
                         executable_swing_move=swing,executable_swing_pct=100*swing/den,
                         segments=len(segs)))
    return pd.DataFrame(rows)

# ---------------- PortfolioOS chronological event engine ----------------
def week_key(ts):
    t=pd.Timestamp(ts);return (t-pd.Timedelta(days=t.weekday())).floor("D")

def factor_vec(asset,dir_):
    v=np.zeros(len(FACTORS));ix={f:i for i,f in enumerate(FACTORS)}
    if asset in FX:
        b,q=asset[:3],asset[3:];
        if b in ix:v[ix[b]]+=.55*dir_
        if q in ix:v[ix[q]]-=.55*dir_
    elif asset in METALS:v[ix["METAL"]]+=.55*dir_;v[ix["USD"]]-=.15*dir_
    elif asset in CRYPTO:v[ix["CRYPTO"]]+=.55*dir_
    elif asset in ENERGY:v[ix["ENERGY"]]+=.55*dir_
    elif asset in INDICES:v[ix["EQUITY"]]+=.55*dir_
    else:v[ix["SYNTH"]]+=.55*dir_
    return v

def replay(cands,specs,start=100.0,feature_cache=None):
    """Chronological shared-equity replay with persistent concurrent positions.
    Uses a min-heap for exits so thousands of asynchronous campaigns remain tractable.
    """
    import heapq
    if cands.empty:return pd.DataFrame(),pd.DataFrame()
    c=cands.copy();c["decision_time"]=pd.to_datetime(c.decision_time,utc=True);c["exit_time"]=pd.to_datetime(c.exit_time,utc=True)
    c=c.sort_values(["decision_time","quality"],ascending=[True,False]).reset_index(drop=True)
    balance=float(start);peak=balance;week=None;week_start=balance;objective=False;floor=False
    openpos={}; exit_heap=[]; events=[];decisions=[];seq=0
    factors=np.zeros(len(FACTORS));open_risk_cash=0.0;open_margin_cash=0.0
    config_week_r={}; asset_week_r={}
    def mark_state(p,t):
        # Information available only up to decision time; no future bars are read.
        if not feature_cache or p["asset"] not in feature_cache:return None
        x=feature_cache[p["asset"]];ts=x.timestamp
        lo=int(ts.searchsorted(pd.Timestamp(p["entry_time"]),side="left"));hi=int(ts.searchsorted(pd.Timestamp(t),side="right"))
        if hi<=lo:return None
        z=x.iloc[lo:hi];mark=float(z.close.iloc[-1]);ent=float(p["entry_price"]);sd=max(float(p["stop_distance"]),1e-12);dr=int(p["direction"])
        cur=(mark-ent)*dr/sd-float(p.get("cost_r",0.0))
        fav=((float(z.high.max())-ent)/sd if dr==1 else (ent-float(z.low.min()))/sd)
        floor=-1.0
        if fav>=.75:floor=0.0
        if fav>=1.5:floor=.5
        if fav>=2.0:floor=max(floor,fav-.8)
        return {"mark":mark,"current_r":cur,"mfe_to_date":fav,"floor_r":floor}
    def mtm_equity(t):
        eq=float(balance)
        for p0 in openpos.values():
            st=mark_state(p0,t)
            if st is not None:eq+=float(p0["risk_cash"])*float(st["current_r"])
        return eq
    def recycle_for(newrow,t,new_utility,reason):
        nonlocal balance,peak,factors,open_risk_cash,open_margin_cash
        if reason not in {"OPEN_RISK_BUDGET","MARGIN_BUDGET","FACTOR_DUPLICATION"}:return False,None
        choices=[]
        for pid,p0 in openpos.items():
            if bool(p0.get("is_addon",False)):continue
            st=mark_state(p0,t)
            if st is None or st["floor_r"]<0:continue
            u0=float(p0.get("entry_utility",p0.get("quality",0.0)))
            # recycle only when the challenger is materially better and incumbent is protected.
            if new_utility<=u0+.10:continue
            newfac=factors-factor_vec(p0["asset"],int(p0["direction"]))+factor_vec(newrow.asset,int(newrow.direction))
            if np.abs(newfac).max()>1.25:continue
            choices.append((u0,pid,p0,st,newfac))
        if not choices:return False,None
        _,pid,p0,st,newfac=min(choices,key=lambda q:q[0])
        openpos.pop(pid,None)
        rr=max(float(st["floor_r"]),float(st["current_r"])) if st["current_r"]<st["floor_r"] else float(st["current_r"])
        # Since floor protection is a stop, current liquidation cannot assume a better fill than the mark.
        rr=float(st["current_r"])
        before_eq=mtm_equity(t);pnl=float(p0["risk_cash"])*rr;balance+=pnl
        factors=newfac-factor_vec(newrow.asset,int(newrow.direction)) # incumbent removed; challenger added only if selected below
        open_risk_cash=max(0.0,open_risk_cash-float(p0["risk_cash"]));open_margin_cash=max(0.0,open_margin_cash-float(p0["margin"]))
        eq=mtm_equity(t);peak=max(peak,eq);dd=eq/peak-1
        ep={**p0,"timestamp":pd.Timestamp(t),"exit_time":pd.Timestamp(t),"exit_price":float(st["mark"]),"net_r":rr,"gross_r":rr+float(p0.get("cost_r",0.0)),"net_pnl":pnl,"equity_before":before_eq,"equity":eq,"balance_after":balance,"drawdown":dd,"exit_reason":"DYNAMIC_CAPITAL_RECYCLE"}
        events.append(ep)
        wkx=week_key(t);ck=(wkx,p0["asset"],p0["strategy"]);ak=(wkx,p0["asset"]);config_week_r[ck]=config_week_r.get(ck,0.0)+rr;asset_week_r[ak]=asset_week_r.get(ak,0.0)+rr
        return True,pid
    def close_until(t):
        nonlocal balance,peak,factors,open_risk_cash,open_margin_cash
        while exit_heap and exit_heap[0][0]<=t:
            _,_,pid=heapq.heappop(exit_heap)
            p=openpos.pop(pid,None)
            if p is None:continue
            pnl=p["risk_cash"]*p["net_r"];before_eq=mtm_equity(p["exit_time"]);balance+=pnl
            factors-=factor_vec(p["asset"],p["direction"]);open_risk_cash=max(0.0,open_risk_cash-p["risk_cash"]);open_margin_cash=max(0.0,open_margin_cash-p["margin"])
            eq=mtm_equity(p["exit_time"]);peak=max(peak,eq);dd=eq/peak-1
            events.append({**p,"timestamp":p["exit_time"],"net_pnl":pnl,"equity_before":before_eq,"equity":eq,"balance_after":balance,"drawdown":dd})
            wkx=week_key(p["exit_time"]); ck=(wkx,p["asset"],p["strategy"]); ak=(wkx,p["asset"])
            config_week_r[ck]=config_week_r.get(ck,0.0)+float(p["net_r"]); asset_week_r[ak]=asset_week_r.get(ak,0.0)+float(p["net_r"])
    for t,due in c.groupby("decision_time",sort=True):
        wk=week_key(t)
        if week is None:
            week=wk;week_start=float(start);objective=False;floor=False
        else:
            while t>=week+pd.Timedelta(days=7):
                boundary=week+pd.Timedelta(days=7);close_until(boundary);week=boundary;week_start=mtm_equity(boundary);objective=False;floor=False
        close_until(t)
        equity_now=mtm_equity(t)
        growth=equity_now/week_start-1 if week_start else 0
        if growth>=.05:objective=True
        if objective and growth<=.04:floor=True
        risk_scale=0 if floor else (.25 if growth>=.10 else 1.0)
        # Existing Trailaris DD de-compounding doctrine: reduce new risk after weekly loss,
        # without ever raising the 1% aggregate ceiling. A 5% weekly loss pauses new
        # foundations until the next weekly reset.
        if growth<=-.05:risk_scale=0.0
        elif growth<=-.035:risk_scale*=.25
        elif growth<=-.02:risk_scale*=.50
        elif growth<=-.01:risk_scale*=.75
        risk_budget=min(.01,max(0,growth-.04)) if objective else .01
        for _,r in due.sort_values("quality",ascending=False).iterrows():
            rf=(.0025 if bool(r.is_addon) else .005)*risk_scale
            lot,risk,margin=size_trade(r.asset,equity_now,rf,float(r.stop_distance),specs);reason="SELECTED"
            if bool(r.is_addon):
                if not r.parent_id or r.parent_id not in openpos:reason="ADDON_PARENT_NOT_OPEN"
                elif float(openpos[r.parent_id]["mfe_r"])<.75:reason="ADDON_PARENT_NOT_PROVEN"
            projected=factors+factor_vec(r.asset,int(r.direction))
            ck=(wk,r.asset,r.strategy);ak=(wk,r.asset)
            if reason=="SELECTED" and (not bool(r.is_addon)) and any(p["asset"]==r.asset for p in openpos.values()):reason="SAME_ASSET_CAMPAIGN_ALREADY_OPEN"
            if reason=="SELECTED" and config_week_r.get(ck,0.0)<=-2.0:reason="CONFIG_WEEKLY_LOSS_BREAKER"
            if reason=="SELECTED" and asset_week_r.get(ak,0.0)<=-3.0:reason="ASSET_WEEKLY_LOSS_BREAKER"
            if reason=="SELECTED" and lot<=0:reason="LOT_BELOW_MIN"
            if reason=="SELECTED" and (open_risk_cash+risk)/max(equity_now,1e-12)>risk_budget+1e-12:reason="OPEN_RISK_BUDGET"
            if reason=="SELECTED" and (open_margin_cash+margin)/max(equity_now,1e-12)>.60:reason="MARGIN_BUDGET"
            if reason=="SELECTED" and np.abs(projected).max()>1.25:reason="FACTOR_DUPLICATION"
            utility=float(r.quality)-1.2*float(r.cost_r)-.25*(margin/max(equity_now,1e-12))
            if reason=="SELECTED" and utility<.12:reason="UTILITY_FLOOR"
            recycled=False;recycled_pid=None
            if reason in {"OPEN_RISK_BUDGET","MARGIN_BUDGET","FACTOR_DUPLICATION"} and utility>=.12:
                recycled,recycled_pid=recycle_for(r,t,utility,reason)
                if recycled:
                    # Re-size from the new shared equity and re-check the unchanged hard ceilings.
                    equity_now=mtm_equity(t);lot,risk,margin=size_trade(r.asset,equity_now,rf,float(r.stop_distance),specs);projected=factors+factor_vec(r.asset,int(r.direction))
                    if lot>0 and (open_risk_cash+risk)/max(equity_now,1e-12)<=risk_budget+1e-12 and (open_margin_cash+margin)/max(equity_now,1e-12)<=.60 and np.abs(projected).max()<=1.25:reason="SELECTED"
                    else:reason="RECYCLE_DID_NOT_FREE_ENOUGH_CAPACITY"
            decisions.append(dict(decision_time=t,campaign_id=r.campaign_id,asset=r.asset,strategy=r.strategy,strategy_family=r.strategy_family,is_addon=bool(r.is_addon),selected=reason=="SELECTED",reason=reason,quality=float(r.quality),utility=utility,recycled_incumbent_id=recycled_pid or "",balance=balance,equity=equity_now,week_growth=growth,risk_budget=risk_budget,lot=lot,risk_cash=risk,margin=margin,open_risk_pct_before=100*open_risk_cash/max(equity_now,1e-12),projected_open_risk_pct=100*(open_risk_cash+risk)/max(equity_now,1e-12),open_margin_pct_before=100*open_margin_cash/max(equity_now,1e-12),projected_open_margin_pct=100*(open_margin_cash+margin)/max(equity_now,1e-12)))
            if reason=="SELECTED":
                p=r.to_dict();p.update(lot=lot,risk_cash=risk,margin=margin,equity_at_entry=equity_now,balance_at_entry=balance,entry_utility=utility)
                openpos[r.campaign_id]=p;factors=projected;open_risk_cash+=risk;open_margin_cash+=margin;seq+=1
                heapq.heappush(exit_heap,(pd.Timestamp(r.exit_time),seq,r.campaign_id))
    close_until(pd.Timestamp.max.tz_localize("UTC"))
    ev=pd.DataFrame(events)
    if len(ev):ev=ev.sort_values("timestamp").reset_index(drop=True)
    return ev,pd.DataFrame(decisions)

# ---------------- reporting / attribution ----------------
def _event_mark_r(row,t,feature_cache):
    if not feature_cache or row.asset not in feature_cache:return 0.0
    x=feature_cache[row.asset];ts=x.timestamp;hi=int(ts.searchsorted(pd.Timestamp(t),side="right"))
    if hi<=0:return 0.0
    mark=float(x.close.iloc[min(hi-1,len(x)-1)]);ent=float(row.entry_price);sd=max(float(row.stop_distance),1e-12);dr=int(row.direction)
    return (mark-ent)*dr/sd-float(row.cost_r)

def _equity_at(events,t,start,feature_cache):
    if events.empty:return float(start)
    e=events;et=pd.to_datetime(e.exit_time,utc=True);en=pd.to_datetime(e.entry_time,utc=True)
    eq=float(start)+float(e.loc[et<=t,"net_pnl"].sum())
    op=e[(en<=t)&(et>t)]
    for _,r in op.iterrows():eq+=float(r.risk_cash)*_event_mark_r(r,t,feature_cache)
    return eq

def weekly(events,start=100.0,feature_cache=None,start_ts=None,end_ts=None):
    if events.empty:return pd.DataFrame()
    e=events.copy();e["entry_time"]=pd.to_datetime(e.entry_time,utc=True);e["exit_time"]=pd.to_datetime(e.exit_time,utc=True)
    s=pd.Timestamp(start_ts) if start_ts is not None else e.entry_time.min();z=pd.Timestamp(end_ts) if end_ts is not None else e.exit_time.max()
    first=week_key(s);last=week_key(z);rows=[]
    for ws in pd.date_range(first,last,freq="7D",tz="UTC"):
        we=ws+pd.Timedelta(days=7);a=max(ws,s);b=min(we,z+pd.Timedelta(microseconds=1))
        start_eq=_equity_at(e,a,start,feature_cache);end_eq=_equity_at(e,b,start,feature_cache)
        # MTM drawdown sampled every six hours plus actual exits.
        pts=list(pd.date_range(a,b,freq="6h",tz="UTC"));pts += e.loc[(e.exit_time>=a)&(e.exit_time<=b),"exit_time"].tolist();pts=sorted(set(pts+[a,b]))
        vals=pd.Series([_equity_at(e,t,start,feature_cache) for t in pts],dtype=float);dd=float((vals/vals.cummax()-1).min()) if len(vals) else 0.0
        ret=end_eq/start_eq-1 if start_eq else 0.0
        entered=int(((e.entry_time>=a)&(e.entry_time<b)).sum());closed=int(((e.exit_time>=a)&(e.exit_time<b)).sum())
        rows.append(dict(week_start=ws.date().isoformat(),start_equity=start_eq,end_equity=end_eq,weekly_return_pct=100*ret,campaigns_entered=entered,campaigns_closed=closed,max_drawdown_pct=100*dd,ge5=ret>=.05,ge10=ret>=.10))
    return pd.DataFrame(rows)

def monthly(w):
    if w.empty:return pd.DataFrame()
    x=w.copy();x["month"]=x.week_start.str[:7];rows=[]
    for m,g in x.groupby("month"):
        rows.append(dict(month=m,weeks=len(g),start_equity=float(g.start_equity.iloc[0]),end_equity=float(g.end_equity.iloc[-1]),monthly_return_pct=100*(float(g.end_equity.iloc[-1])/float(g.start_equity.iloc[0])-1),weeks_ge5=int(g.ge5.sum()),weeks_ge10=int(g.ge10.sum())))
    return pd.DataFrame(rows)

def opportunity_period(data,start_ts,end_ts):
    frames=[]
    for a,d in data.items():
        z=d[(d.timestamp>start_ts)&(d.timestamp<=end_ts)].copy()
        if len(z):
            o=opportunity_ledger(a,z)
            if len(o):frames.append(o)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def _miss_reasons(asset,events,raw_cands,promoted,decisions,opp_pct,cap_pct):
    ev=events[events.asset==asset] if not events.empty else pd.DataFrame()
    raw=raw_cands[raw_cands.asset==asset] if not raw_cands.empty else pd.DataFrame()
    pro=promoted[promoted.asset==asset] if not promoted.empty else pd.DataFrame()
    de=decisions[decisions.asset==asset] if not decisions.empty else pd.DataFrame()
    reasons=[]
    if len(raw)==0:reasons.append("NO_STRATEGY_SIGNAL")
    elif len(pro)==0:reasons.append("CONFIG_NOT_VALIDATION_PROMOTED")
    elif len(pro)<.25*len(raw):reasons.append("PRECISION_FILTER_REMOVED_MOST_SIGNALS")
    if len(de) and (~de.selected).any():
        reasons.extend(de.loc[~de.selected,"reason"].value_counts().head(3).index.tolist())
    if len(ev) and (ev.net_r<=0).mean()>.5:reasons.append("FALSE_BREAK_OR_WRONG_DIRECTION")
    if len(ev) and ev.giveback_r.mean()>.75:reasons.append("WINNER_GIVEBACK")
    if opp_pct>0 and cap_pct/opp_pct<.25:reasons.append("LOW_OPPORTUNITY_CAPTURE")
    return "|".join(dict.fromkeys(reasons))

def capture_attribution(events,opps,raw_cands,promoted,decisions):
    rows=[]
    for asset in ROUTES:
        o=opps[opps.asset==asset];ev=events[events.asset==asset] if not events.empty else pd.DataFrame();raw=raw_cands[raw_cands.asset==asset];pro=promoted[promoted.asset==asset];de=decisions[decisions.asset==asset] if not decisions.empty else pd.DataFrame()
        opp_pct=float(o.executable_swing_pct.sum()) if len(o) else 0.0
        cap_pct=0.0
        if len(ev):
            cap_pct=float(((((ev.exit_price-ev.entry_price)*ev.direction).clip(lower=0))/ev.entry_price.abs().clip(lower=1e-12)*100).sum())
        ratio=min(1.0,cap_pct/opp_pct) if opp_pct>0 else 0.0
        rows.append(dict(asset=asset,
            directional_open_close_pct=float(o.directional_open_close_pct.sum()) if len(o) else 0.0,
            open_60m_move_pct=float(o.open_60m_move_pct.sum()) if len(o) else 0.0,
            gross_session_open_range_pct=float(o.gross_session_open_range_pct.sum()) if len(o) else 0.0,
            close_60m_move_pct=float(o.close_60m_move_pct.sum()) if len(o) else 0.0,
            executable_swing_opportunity_pct=opp_pct,captured_positive_move_pct=cap_pct,
            capture_ratio=ratio,missed_move_pct=max(0.0,opp_pct-cap_pct),
            raw_candidate_signals=len(raw),precision_promoted_signals=len(pro),selected_campaigns=len(ev),
            miss_reasons=_miss_reasons(asset,events,raw_cands,promoted,decisions,opp_pct,cap_pct)))
    return pd.DataFrame(rows)

def weekly_asset_capture(events,opps,raw_cands,promoted,decisions,start_ts,end_ts):
    rows=[]
    weeks=pd.date_range(week_key(start_ts),week_key(end_ts),freq="7D",tz="UTC")
    O=opps.copy();O["week_start"]=pd.to_datetime(O.day,utc=True).map(week_key) if len(O) else []
    def addwk(df,col):
        x=df.copy()
        if len(x):x["week_start"]=pd.to_datetime(x[col],utc=True).map(week_key)
        return x
    EV=addwk(events,"timestamp");RAW=addwk(raw_cands,"decision_time");PRO=addwk(promoted,"decision_time");DE=addwk(decisions,"decision_time")
    for ws in weeks:
        for asset in ROUTES:
            o=O[(O.asset==asset)&(O.week_start==ws)] if len(O) else pd.DataFrame();ev=EV[(EV.asset==asset)&(EV.week_start==ws)] if len(EV) else pd.DataFrame();raw=RAW[(RAW.asset==asset)&(RAW.week_start==ws)] if len(RAW) else pd.DataFrame();pro=PRO[(PRO.asset==asset)&(PRO.week_start==ws)] if len(PRO) else pd.DataFrame();de=DE[(DE.asset==asset)&(DE.week_start==ws)] if len(DE) else pd.DataFrame()
            opp_pct=float(o.executable_swing_pct.sum()) if len(o) else 0.0
            cap_pct=float(((((ev.exit_price-ev.entry_price)*ev.direction).clip(lower=0))/ev.entry_price.abs().clip(lower=1e-12)*100).sum()) if len(ev) else 0.0
            rows.append(dict(week_start=ws.date().isoformat(),asset=asset,
                directional_open_close_pct=float(o.directional_open_close_pct.sum()) if len(o) else 0.0,
                open_60m_move_pct=float(o.open_60m_move_pct.sum()) if len(o) else 0.0,
                gross_session_open_range_pct=float(o.gross_session_open_range_pct.sum()) if len(o) else 0.0,
                close_60m_move_pct=float(o.close_60m_move_pct.sum()) if len(o) else 0.0,
                executable_swing_opportunity_pct=opp_pct,captured_positive_move_pct=cap_pct,
                capture_ratio=min(1.0,cap_pct/opp_pct) if opp_pct>0 else 0.0,missed_move_pct=max(0.0,opp_pct-cap_pct),
                raw_candidate_signals=len(raw),precision_promoted_signals=len(pro),selected_campaigns=len(ev),
                net_pnl=float(ev.net_pnl.sum()) if len(ev) else 0.0,
                miss_reasons=_miss_reasons(asset,ev,raw,pro,de,opp_pct,cap_pct)))
    return pd.DataFrame(rows)

def strategy_stats(cands,mask):
    x=cands[mask].copy()
    if x.empty:return pd.DataFrame(columns=["asset","strategy","n","mean_r","median_r","win_rate","mean_cost","mean_quality"])
    return x.groupby(["asset","strategy"]).agg(n=("net_r","size"),mean_r=("net_r","mean"),median_r=("net_r","median"),win_rate=("net_r",lambda s:(s>0).mean()),mean_cost=("cost_r","mean"),mean_quality=("quality","mean")).reset_index()

def _precision_apply(df,cfg):
    if df.empty:return df
    t=pd.to_datetime(df.decision_time,utc=True);m=(df.quality>=cfg['min_quality'])&(df.cost_r<=cfg['max_cost_r'])
    if int(cfg['direction'])!=0:m &= (df.direction==int(cfg['direction']))
    h0,h1=int(cfg['hour_start']),int(cfg['hour_end']);m &= (t.dt.hour>=h0)&(t.dt.hour<h1)
    return df[m]

def _search_precision_configs(train):
    rows=[];qgrid=np.array([.55,.62,.68,.74]);cgrid=np.array([.20,.35,.50]);dirs=[0,1,-1];hours=[(0,24),(0,7),(7,12),(12,18),(18,24)]
    for (a,stg),g in train.groupby(['asset','strategy'],sort=False):
        qa=g.quality.to_numpy(float);ca=g.cost_r.to_numpy(float);da=g.direction.to_numpy(int);ra=g.net_r.to_numpy(float);ha=pd.to_datetime(g.decision_time,utc=True).dt.hour.to_numpy(int)
        best=None
        for q in qgrid:
            mq=qa>=q
            if mq.sum()<4:continue
            for mc in cgrid:
                mb=mq&(ca<=mc)
                if mb.sum()<4:continue
                for dr in dirs:
                    md=mb if dr==0 else mb&(da==dr)
                    if md.sum()<4:continue
                    for h0,h1 in hours:
                        m=md&(ha>=h0)&(ha<h1);n=int(m.sum())
                        if n<4:continue
                        rv=ra[m];cv=ca[m];mean=float(rv.mean());med=float(np.median(rv));sd=float(rv.std());win=float((rv>0).mean());cost=float(cv.mean())
                        if mean<=0 or med<=-.30 or win<.30:continue
                        robust=mean-.25*sd/max(n**.5,1)+.05*win-.05*cost
                        if best is None or robust>best['robust_score']:
                            best=dict(asset=a,strategy=stg,min_quality=float(q),max_cost_r=float(mc),direction=int(dr),hour_start=int(h0),hour_end=int(h1),train_n=n,train_mean_r=mean,train_median_r=med,train_win_rate=win,train_mean_cost=cost,robust_score=robust)
        if best is not None:rows.append(best)
    return pd.DataFrame(rows)

def apply_repair(cands,train_end,val_end):
    """Precision repair with immutable train -> validation -> untouched sequence.
    Configuration search is asset+strategy specific and uses only information
    observable at decision time (quality, cost, direction, hour regime).
    """
    tr=cands.decision_time<=train_end;va=(cands.decision_time>train_end)&(cands.decision_time<=val_end);te=cands.decision_time>val_end
    train=cands[tr].copy();val=cands[va].copy();test=cands[te].copy();st=strategy_stats(cands,tr)
    proposed=_search_precision_configs(train)
    val_rows=[];confirmed_rows=[];cfgmap={}
    for _,cfg in proposed.iterrows():
        key=(cfg.asset,cfg.strategy);z=_precision_apply(val[(val.asset==cfg.asset)&(val.strategy==cfg.strategy)],cfg)
        n=len(z);mean=float(z.net_r.mean()) if n else float('nan');med=float(z.net_r.median()) if n else float('nan');win=float((z.net_r>0).mean()) if n else 0.0;cost=float(z.cost_r.mean()) if n else float('nan')
        rec={**cfg.to_dict(),'validation_n':n,'validation_mean_r':mean,'validation_median_r':med,'validation_win_rate':win,'validation_mean_cost':cost}
        val_rows.append(rec)
        if n>=2 and mean>0 and med>-.25 and win>=.35:
            rec['validation_confirmed']=True;confirmed_rows.append(rec);cfgmap[key]={k:rec[k] for k in ['min_quality','max_cost_r','direction','hour_start','hour_end']}
    vstats=pd.DataFrame(val_rows);confirmed=pd.DataFrame(confirmed_rows)
    val_parts=[];test_parts=[]
    for key,cfg in cfgmap.items():
        a,s=key;val_parts.append(_precision_apply(val[(val.asset==a)&(val.strategy==s)],cfg));test_parts.append(_precision_apply(test[(test.asset==a)&(test.strategy==s)],cfg))
    val_rep=pd.concat(val_parts,ignore_index=True) if val_parts else pd.DataFrame(columns=cands.columns)
    final=pd.concat(test_parts,ignore_index=True) if test_parts else pd.DataFrame(columns=cands.columns)
    return val_rep,final,st,proposed,vstats,confirmed,cfgmap

def diagnosis(w,cap):
    if w.empty:return {"state":"NO_TRADES","issues":["NO_SELECTED_CAMPAIGNS"]}
    issues=[]
    if w.weekly_return_pct.mean()<5:issues.append("MEAN_WEEK_BELOW_5")
    if w.ge5.mean()<.5:issues.append("INSUFFICIENT_5PCT_WEEK_DENSITY")
    if w.max_drawdown_pct.min()<-5:issues.append("DRAWDOWN_ABOVE_5")
    low=cap[cap.capture_ratio<.25]
    if len(low)>17:issues.append("UNIVERSE_WIDE_LOW_CAPTURE")
    return {"state":"TARGET_MET" if not issues else "REPAIR_REQUIRED","mean_week_pct":float(w.weekly_return_pct.mean()),"median_week_pct":float(w.weekly_return_pct.median()),"weeks":len(w),"weeks_ge5":int(w.ge5.sum()),"weeks_ge10":int(w.ge10.sum()),"max_weekly_dd_pct":float(w.max_drawdown_pct.min()),"issues":issues}

# ---------------- smoke data with multi-regime structure ----------------
def generate_smoke(rawdir:Path,weeks=2):
    rawdir.mkdir(parents=True,exist_ok=True);start=pd.Timestamp("2026-05-04T00:00:00Z");end=start+pd.Timedelta(days=7*weeks)-pd.Timedelta(minutes=1)
    ts_all=pd.date_range(start,end,freq="1min",tz="UTC")
    for j,a in enumerate(ROUTES):
        ts=ts_all if a in CONTINUOUS else ts_all[ts_all.weekday<5]
        rng=np.random.default_rng(20260816+j);n=len(ts);base=1.1 if a in FX else (2000 if a=="XAUUSD" else 25 if a in CRYPTO else 1000)
        minute=np.arange(n); dayphase=np.sin(2*np.pi*(minute%1440)/1440+j*.2); session=np.sin(2*np.pi*(minute%480)/480+j*.4)
        vol=.00016 if a in FX else .00045
        shock=np.zeros(n); shock[(minute%1440==450)|(minute%1440==840)]=rng.normal(0,vol*8,((minute%1440==450)|(minute%1440==840)).sum())
        # explicit day-boundary discontinuities exercise open-gap continuation/fade mechanics.
        day_boundary=(minute%1440==0); shock[day_boundary]+=rng.normal(0,vol*14,day_boundary.sum())
        ret=rng.normal(0,vol,n)+vol*.10*dayphase+vol*.16*session+shock
        logp=np.log(base)+np.cumsum(ret);close=np.exp(logp);open_=np.r_[close[0],close[:-1]];span=np.abs(close-open_)+close*vol*.6*(.5+rng.random(n));high=np.maximum(open_,close)+span*.35;low=np.minimum(open_,close)-span*.35
        spread=np.maximum(close*(.00004 if a in FX else .00012),1e-9)
        pd.DataFrame(dict(timestamp=ts,open=open_,high=high,low=low,close=close,spread=spread,volume=1.0,source="SMOKE_SYNTHETIC_FIXTURE",asset=a)).to_csv(rawdir/f"{a}_M1_normalized.csv",index=False)

# ---------------- parallel asset reconstruction ----------------
def build_asset_components(args):
    asset, path = args
    d=read_route(Path(path))
    c,xf=generate_candidates(asset,d)
    o=opportunity_ledger(asset,d)
    return asset,c,xf,o

# ---------------- orchestration ----------------
def run(rawdir:Path,specfile:Path,out:Path,mode:str,start=100.0):
    out.mkdir(parents=True,exist_ok=True);data,v=validate_34(rawdir,mode);v.to_csv(out/"01_data_validation.csv",index=False)
    status={"mode":mode.upper(),"routes_required":34,"routes_pass":len(data),"starting_equity":start,"strategy_families_defined":len(STRATEGY_FAMILIES),"evidence_class":("SMOKE_FIXTURE_ONLY" if mode=="smoke" else ("HISTORICAL_MARKET_PROXY_NOT_BROKER_CERTIFIED" if mode=="research-proxy" else "TARGET_BROKER_EMPIRICAL_NOT_YET_EA_CERTIFIED"))}
    if len(data)!=34:status["state"]="BLOCKED_34_ROUTE_DATA_GATE";(out/"STATUS.json").write_text(json.dumps(status,indent=2));return status
    try:specs=load_specs(specfile,mode)
    except Exception as e:status.update(state="BLOCKED_SPEC_GATE",spec_error=str(e));(out/"STATUS.json").write_text(json.dumps(status,indent=2));return status
    allc=[];allo=[];coverage=[];feature_cache={}
    # Full 34-route reconstruction runs in parallel. Scope is unchanged; this only
    # removes pandas/Python serial overhead from the 510-cell research estate.
    jobs=[(a,str(rawdir/f"{a}_M1_normalized.csv")) for a in ROUTES]
    workers=min(8,max(2,(os.cpu_count() or 4)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(build_asset_components,j):j[0] for j in jobs}
        bundles={}
        for fut in as_completed(futs):
            a,c,xf,o=fut.result();bundles[a]=(c,xf,o)
    for a in ROUTES:
        c,xf,o=bundles[a];feature_cache[a]=xf
        if len(c):allc.append(c)
        if len(o):allo.append(o)
    base=pd.concat(allc,ignore_index=True) if allc else pd.DataFrame()
    rv=generate_relative_value_candidates(feature_cache);im=generate_intermarket_candidates(feature_cache)
    factor_path=rawdir/"Trailaris_R3_FactorFeed.csv"
    if mode=="smoke" and not factor_path.exists():smoke_factor_feed(feature_cache,factor_path)
    ff=load_factor_feed(factor_path);fc=generate_factor_candidates(feature_cache,ff,mode)
    pre=[x for x in [base,rv,im,fc] if x is not None and len(x)]
    cands=pd.concat(pre,ignore_index=True) if pre else pd.DataFrame()
    ens=generate_ensemble_candidates(cands,feature_cache)
    if len(ens):cands=pd.concat([cands,ens],ignore_index=True)
    # R4: enforce asset-native family applicability. Cells remain in the 510-cell factory,
    # but non-native families are hard-gated rather than silently treated as transferable.
    if len(cands):
        native_mask=cands.apply(lambda r: (r.strategy_family=="CAMPAIGN_MANAGEMENT") or r4a.family_applicable(str(r.asset),str(r.strategy_family)),axis=1)
        cands=cands[native_mask].reset_index(drop=True)
    opps=pd.concat(allo,ignore_index=True) if allo else pd.DataFrame()
    for a in ROUTES:
        ca=cands[cands.asset==a] if len(cands) else pd.DataFrame();fam=set(ca.strategy_family) if len(ca) else set();coverage.append(dict(asset=a,candidate_count=len(ca),playbooks_triggered=len(set(ca.strategy)) if len(ca) else 0,families_triggered=len(fam & set(STRATEGY_FAMILIES)),families="|".join(sorted(fam)),playbooks="|".join(sorted(set(ca.strategy))) if len(ca) else ""))
    cov=pd.DataFrame(coverage);audit=strategy_cell_audit(cands,not ff.empty,mode)
    cands.to_csv(out/"02_all_candidate_campaigns.csv",index=False);opps.to_csv(out/"03_opportunity_ledger.csv",index=False);cov.to_csv(out/"04_asset_strategy_coverage.csv",index=False);audit.to_csv(out/"04b_strategy_factory_510_audit.csv",index=False)
    # immutable chronological 60/20/20 split by time
    t0=cands.decision_time.min();t1=cands.decision_time.max();train_end=t0+(t1-t0)*.60;val_end=t0+(t1-t0)*.80
    train=cands[cands.decision_time<=train_end];val=cands[(cands.decision_time>train_end)&(cands.decision_time<=val_end)];test=cands[cands.decision_time>val_end]
    # baseline validation and final test are never used to tune prior to their turn
    ev0,dec0=replay(val,specs,start,feature_cache);w0=weekly(ev0,start,feature_cache,train_end,val_end);opp_val=opportunity_period(data,train_end,val_end);cap0=capture_attribution(ev0,opp_val,val,val,dec0);dg0=diagnosis(w0,cap0)
    val_rep,test_rep,st,proposed,vstats,confirmed,qmap=apply_repair(cands,train_end,val_end)
    st.to_csv(out/"05_train_strategy_stats.csv",index=False);proposed.to_csv(out/"06_train_proposed_configurations.csv",index=False);vstats.to_csv(out/"06b_validation_configuration_stats.csv",index=False);confirmed.to_csv(out/"06c_validation_confirmed_configurations.csv",index=False)
    (out/"07_precision_thresholds.json").write_text(json.dumps({f"{a}|{ss}":cfg for (a,ss),cfg in qmap.items()},indent=2,default=float))
    # No asset is removed. Assets without a validation-confirmed configuration are
    # explicitly queued for the next variant-generation repair pass.
    rq=[]
    confirmed_assets=set(confirmed.asset) if len(confirmed) else set()
    for asset in ROUTES:
        rq.append({'asset':asset,'state':'VALIDATED_CONFIG_PRESENT' if asset in confirmed_assets else 'RESEARCH_NEXT_VARIANT_REQUIRED','confirmed_configurations':int((confirmed.asset==asset).sum()) if len(confirmed) else 0,'candidate_families_seen':int(cands[cands.asset==asset].strategy_family.nunique())})
    pd.DataFrame(rq).to_csv(out/"06d_34_asset_repair_queue.csv",index=False)
    ev1,dec1=replay(val_rep,specs,start,feature_cache);w1=weekly(ev1,start,feature_cache,train_end,val_end);cap1=capture_attribution(ev1,opp_val,val,val_rep,dec1);dg1=diagnosis(w1,cap1)
    ev2,dec2=replay(test_rep,specs,start,feature_cache);w2=weekly(ev2,start,feature_cache,val_end,t1);m2=monthly(w2);opp_test=opportunity_period(data,val_end,t1);cap2=capture_attribution(ev2,opp_test,test,test_rep,dec2);dg2=diagnosis(w2,cap2)
    wcap=weekly_asset_capture(ev2,opp_test,test,test_rep,dec2,val_end,t1)
    # write results
    for name,df in [("08_validation_baseline_events",ev0),("09_validation_baseline_weekly",w0),("10_validation_repaired_events",ev1),("11_validation_repaired_weekly",w1),("12_untouched_test_events",ev2),("13_untouched_test_decisions",dec2),("14_untouched_test_weekly",w2),("15_untouched_test_monthly",m2),("16_untouched_capture_attribution",cap2),("16b_weekly_34_asset_capture",wcap)]:df.to_csv(out/(name+".csv"),index=False)
    (out/"17_diagnosis.json").write_text(json.dumps({"validation_baseline":dg0,"validation_repaired":dg1,"untouched_test":dg2},indent=2))
    tmp=cands.copy();tmp["day"]=pd.to_datetime(tmp.decision_time,utc=True).dt.floor("D");dens=tmp.groupby(["asset","day"]).size()
    opening_mask=cands.strategy.str.contains("DAY_OPEN|SESSION_OPEN",regex=True);closing_mask=cands.strategy.str.contains("CLOSE_SESSION",regex=True)
    selected_max_risk=float(dec2.loc[dec2.selected,"projected_open_risk_pct"].max()) if len(dec2) and dec2.selected.any() else 0.0
    recycled_count=int((ev2.exit_reason=="DYNAMIC_CAPITAL_RECYCLE").sum()) if len(ev2) and "exit_reason" in ev2 else 0
    status.update(state="FULL_34_ASSET_OPPORTUNITY_LOOP_EXECUTED",candidate_campaigns=len(cands),assets_with_candidates=int((cov.candidate_count>0).sum()),strategy_families_triggered_total=len(set(cands.strategy_family)&set(STRATEGY_FAMILIES)),strategy_cells_total=510,strategy_cells_triggered=int((audit.state=="EXECUTED_TRIGGERED").sum()),strategy_cells_data_gated=int((audit.state=="DATA_GATED").sum()),opening_candidates=int(opening_mask.sum()),closing_candidates=int(closing_mask.sum()),median_candidates_per_asset_day=float(dens.median()) if len(dens) else 0.0,max_candidates_per_asset_day=int(dens.max()) if len(dens) else 0,multi_candidate_asset_day_pct=float(100*(dens>1).mean()) if len(dens) else 0.0,max_selected_open_risk_pct=selected_max_risk,dynamic_capital_recycles=recycled_count,train_end=str(train_end),validation_end=str(val_end),validation_baseline=dg0,validation_repaired=dg1,untouched_test=dg2,ending_equity=float(ev2.equity.iloc[-1]) if len(ev2) else start)
    (out/"STATUS.json").write_text(json.dumps(status,indent=2,default=str));return status

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("smoke");s.add_argument("--outdir",default="/mnt/data/trailaris_r3_smoke");s.add_argument("--weeks",type=int,default=2)
    e=sub.add_parser("empirical");e.add_argument("--rawdir",default="/mnt/data/trailaris_raw");e.add_argument("--specs",default="/mnt/data/Trailaris_34Route_ContractSpecs.csv");e.add_argument("--outdir",default="/mnt/data/trailaris_r4_empirical")
    rp=sub.add_parser("research-proxy");rp.add_argument("--rawdir",required=True);rp.add_argument("--specs",required=True);rp.add_argument("--outdir",required=True)
    x=sub.add_parser("existing-smoke");x.add_argument("--rawdir",required=True);x.add_argument("--specs",required=True);x.add_argument("--outdir",required=True)
    a=ap.parse_args()
    if a.cmd=="smoke":
        out=Path(a.outdir);raw=out/"raw";spec=out/"fixture_specs.csv";generate_smoke(raw,a.weeks);fixture_specs(spec);print(json.dumps(run(raw,spec,out,"smoke",100.0),indent=2,default=str))
    elif a.cmd=="existing-smoke":
        print(json.dumps(run(Path(a.rawdir),Path(a.specs),Path(a.outdir),"smoke",100.0),indent=2,default=str))
    elif a.cmd=="research-proxy":
        print(json.dumps(run(Path(a.rawdir),Path(a.specs),Path(a.outdir),"research-proxy",100.0),indent=2,default=str))
    else:print(json.dumps(run(Path(a.rawdir),Path(a.specs),Path(a.outdir),"empirical",100.0),indent=2,default=str))
if __name__=="__main__":main()
