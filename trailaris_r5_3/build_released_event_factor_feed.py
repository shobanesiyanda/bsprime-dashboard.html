#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROUTES=[
"EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD",
"EURJPY","GBPJPY","EURGBP","AUDJPY","CADJPY","GBPCHF",
"XAUUSD","XAGUSD","BTCUSD","ETHUSD","SOLUSD","BNBUSD","XRPUSD",
"US100","US500","US30","GER40","UK100","JP225","WTI","BRENT","NATGAS",
"V75","V50","V100","V25","V10"]
SYNTH={"V75","V50","V100","V25","V10"}

# Released USD-core events already assembled in Trailaris v0.14 from official
# BLS/Federal Reserve calendars. Times are UTC. Only events inside the R5
# May-04..Aug-14 research window are reproduced here. This is a causal
# post-release response lane: the score is timestamped 15 minutes AFTER release
# and uses no price data later than that decision time.
EVENTS=[
 ("2026-05-08T12:30:00Z","NFP_EMPLOYMENT","https://www.bls.gov/schedule/2026/home.htm"),
 ("2026-05-12T12:30:00Z","CPI","https://www.bls.gov/schedule/2026/home.htm"),
 ("2026-05-13T12:30:00Z","PPI","https://www.bls.gov/schedule/2026/home.htm"),
 ("2026-06-05T12:30:00Z","NFP_EMPLOYMENT","https://www.bls.gov/schedule/2026/home.htm"),
 ("2026-06-10T12:30:00Z","CPI","https://www.bls.gov/schedule/2026/home.htm"),
 ("2026-06-11T12:30:00Z","PPI","https://www.bls.gov/schedule/2026/home.htm"),
 ("2026-06-17T18:00:00Z","FOMC","https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
 ("2026-07-02T12:30:00Z","NFP_EMPLOYMENT","https://www.bls.gov/schedule/2026/home.htm"),
 ("2026-07-14T12:30:00Z","CPI","https://www.bls.gov/schedule/2026/home.htm"),
 ("2026-07-15T12:30:00Z","PPI","https://www.bls.gov/schedule/2026/home.htm"),
 ("2026-07-29T18:00:00Z","FOMC","https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
 ("2026-08-07T12:30:00Z","NFP_EMPLOYMENT","https://www.bls.gov/schedule/2026/home.htm"),
]

def load_close(path:Path):
    d=pd.read_csv(path,usecols=lambda c:c in {"timestamp","close"})
    d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True,errors="coerce")
    d["close"]=pd.to_numeric(d["close"],errors="coerce")
    d=d.dropna().drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return d.close

def px_at_or_before(s:pd.Series,t:pd.Timestamp,max_age_min=10):
    z=s.loc[:t]
    if z.empty:return None
    ts=z.index[-1]
    if (t-ts)>pd.Timedelta(minutes=max_age_min):return None
    return float(z.iloc[-1])

def event_score(s:pd.Series,event_time:pd.Timestamp,delay=15):
    decision=event_time+pd.Timedelta(minutes=delay)
    p0=px_at_or_before(s,event_time,10); p1=px_at_or_before(s,decision,5)
    if p0 is None or p1 is None or p0<=0:return None
    pre=s.loc[event_time-pd.Timedelta(minutes=90):event_time]
    r=pre.pct_change().dropna().tail(60)
    if len(r)<20:return None
    sigma=float(r.std(ddof=1))*np.sqrt(delay)
    if not np.isfinite(sigma) or sigma<=1e-12:return None
    z=((p1/p0)-1.0)/sigma
    return decision,float(np.clip(z,-4.0,4.0)),p0,p1,sigma

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--rawdir",required=True);ap.add_argument("--out",required=True);ap.add_argument("--audit",required=True)
    a=ap.parse_args();raw=Path(a.rawdir);out=Path(a.out);auditp=Path(a.audit);out.parent.mkdir(parents=True,exist_ok=True)
    rows=[];audit=[]
    for asset in ROUTES:
        p=raw/f"{asset}_M1_normalized.csv"
        if not p.exists():
            audit.append({"asset":asset,"state":"M1_MISSING","event_rows":0});continue
        if asset in SYNTH:
            # EVENT_RESPONSE is non-native for the synthetic routes in the 510-cell factory.
            audit.append({"asset":asset,"state":"NON_NATIVE_EVENT_FAMILY_GATED","event_rows":0});continue
        s=load_close(p); n=0
        for ets,fam,src in EVENTS:
            et=pd.Timestamp(ets)
            q=event_score(s,et)
            if q is None:continue
            decision,score,p0,p1,sigma=q
            # Preserve all observed responses in the evidence file; the R4 factor
            # engine applies its own |score| >= 0.70 entry threshold.
            rows.append({
              "timestamp":decision.isoformat(),"asset":asset,
              "orderflow_score":"","carry_score":"","funding_score":"",
              "event_score":score,
              "evidence_state":"RELEASED_OFFICIAL_CALENDAR_CAUSAL_POST_EVENT",
              "event_family":fam,"event_release_time":et.isoformat(),
              "event_source":src,"response_delay_minutes":15,
              "event_px":p0,"decision_px":p1,"pre_event_sigma_15m":sigma
            });n+=1
        audit.append({"asset":asset,"state":"RELEASED_EVENT_EVIDENCE_BUILT" if n else "NO_EXECUTABLE_EVENT_WINDOW","event_rows":n})
    df=pd.DataFrame(rows).sort_values(["timestamp","asset"]) if rows else pd.DataFrame(columns=["timestamp","asset","orderflow_score","carry_score","funding_score","event_score","evidence_state"])
    df.to_csv(out,index=False)
    ad=pd.DataFrame(audit);ad.to_csv(auditp,index=False)
    summary={
      "routes_required":34,"routes_audited":len(ad),"synthetic_event_non_native":int((ad.state=="NON_NATIVE_EVENT_FAMILY_GATED").sum()),
      "routes_with_released_event_evidence":int((ad.event_rows>0).sum()),"factor_rows":len(df),
      "evidence_class":"RELEASED_OFFICIAL_CALENDAR_CAUSAL_POST_EVENT",
      "causality":"decision timestamp = release + 15m; score uses only data through decision timestamp",
      "promotion_limit":"research factor evidence only; does not certify target-broker execution"
    }
    Path(str(auditp)+".json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    assert len(ad)==34

if __name__=="__main__":main()
