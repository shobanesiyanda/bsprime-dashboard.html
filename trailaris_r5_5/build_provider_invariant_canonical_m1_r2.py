#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROUTES=['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
PRIMARY_ONLY={'US30','NATGAS','WTI','V75','V50','V100','V25','V10'}
DUAL=set(ROUTES)-PRIMARY_ONLY
COLS=['timestamp','open','high','low','close','bid','ask','spread','volume','source','asset']
WINDOW=10080
MIN_PERIODS=1440
Q=0.999
EPS=1e-12


def load(path: Path) -> pd.DataFrame:
    d=pd.read_csv(path)
    d['timestamp']=pd.to_datetime(d['timestamp'],utc=True,errors='coerce')
    for c in ['open','high','low','close','volume']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.dropna(subset=['timestamp','open','high','low','close']).sort_values('timestamp').drop_duplicates('timestamp',keep='last').reset_index(drop=True)


def comps(d: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x=d[['timestamp','open','high','low','close','volume']].copy()
    prev=x['close'].shift(1)
    x[f'{prefix}_gap']=np.log(x['open']/prev)
    x[f'{prefix}_c']=np.log(x['close']/x['open'])
    x[f'{prefix}_h']=np.log(x['high']/x['open'])
    x[f'{prefix}_l']=np.log(x['low']/x['open'])
    return x.rename(columns={c:f'{prefix}_{c}' for c in ['open','high','low','close','volume']})


def causal_clip(p: pd.Series, q: pd.Series, mode: str):
    diff=(p-q).abs()
    threshold=diff.shift(1).rolling(WINDOW,min_periods=MIN_PERIODS).quantile(Q)
    available=q.notna() & threshold.notna()
    if mode=='high': more_extreme=p>q
    elif mode=='low': more_extreme=p<q
    else: more_extreme=p.abs()>q.abs()
    flag=available & more_extreme & (diff>threshold)
    out=p.copy()
    delta=(p-q).clip(lower=-threshold,upper=threshold)
    out.loc[flag]=(q+delta).loc[flag]
    return out,flag,threshold


def build_dual(asset: str,p: pd.DataFrame,q: pd.DataFrame):
    a=comps(p,'p'); b=comps(q,'q')
    z=a.merge(b,on='timestamp',how='left',sort=False)
    cc,fc,tc=causal_clip(z.p_c,z.q_c,'return')
    hh,fh,th=causal_clip(z.p_h,z.q_h,'high')
    ll,fl,tl=causal_clip(z.p_l,z.q_l,'low')
    gg,fg,tg=causal_clip(z.p_gap,z.q_gap,'return')
    gg=gg.fillna(0.0); cc=cc.fillna(0.0); hh=hh.fillna(0.0); ll=ll.fillna(0.0)
    anchor=float(z.p_open.iloc[0])
    steps=(gg+cc).to_numpy(dtype=float,copy=True); steps[0]=float(cc.iloc[0])
    logclose=np.log(anchor)+np.cumsum(steps)
    logopen=logclose-cc.to_numpy(dtype=float,copy=True)
    op=np.exp(logopen); cl=np.exp(logclose)
    hi=op*np.exp(hh.to_numpy(dtype=float,copy=True)); lo=op*np.exp(ll.to_numpy(dtype=float,copy=True))
    hi=np.maximum.reduce([hi,op,cl]); lo=np.minimum.reduce([lo,op,cl])
    out=pd.DataFrame({'timestamp':z.timestamp,'open':op,'high':hi,'low':lo,'close':cl,'bid':cl,'ask':'','spread':'','volume':z.p_volume.fillna(0.0),'source':'CANONICAL_REFERENCE_FEED_CAUSAL_TAIL_CLIP_R2','asset':asset})[COLS]
    overlap=int(z.q_close.notna().sum())
    audit={'asset':asset,'mode':'REFERENCE_FEED_CAUSAL_TAIL_CLIP_R2','rows':len(out),'primary_rows':len(p),'independent_rows':len(q),'overlap_rows':overlap,'close_clip_count':int(fc.sum()),'high_clip_count':int(fh.sum()),'low_clip_count':int(fl.sum()),'gap_clip_count':int(fg.sum()),'clip_rule':f'trailing {WINDOW}-bar q={Q}, min_periods={MIN_PERIODS}, lagged/causal; only more-extreme primary discrepancies clipped'}
    return out,audit


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--primary',required=True); ap.add_argument('--independent',required=True); ap.add_argument('--out',required=True); ap.add_argument('--audit',required=True); args=ap.parse_args()
    primary=Path(args.primary); independent=Path(args.independent); outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    audits=[]
    for asset in ROUTES:
        p=load(primary/f'{asset}_M1_normalized.csv')
        if asset in DUAL:
            q=load(independent/f'{asset}_M1_normalized.csv'); out,au=build_dual(asset,p,q)
        else:
            out=p.copy(); out['bid']=out.close; out['ask']=''; out['spread']=''; out['source']='CANONICAL_PRIMARY_ONLY_R2'; out['asset']=asset; out=out[COLS]
            au={'asset':asset,'mode':'PRIMARY_ONLY_R2','rows':len(out),'primary_rows':len(p),'independent_rows':0,'overlap_rows':0,'close_clip_count':0,'high_clip_count':0,'low_clip_count':0,'gap_clip_count':0}
        assert len(out)>=1000,(asset,len(out)); out.to_csv(outdir/f'{asset}_M1_normalized.csv',index=False,date_format='%Y-%m-%dT%H:%M:%S.%fZ'); audits.append(au)
    summary={'state':'CANONICAL_PROVIDER_INVARIANCE_R2_COMPLETE','architecture':'QUALITY_RANKED_REFERENCE_FEED_WITH_CAUSAL_INDEPENDENT_TAIL_CORRECTION','routes':34,'dual_provider_routes':len(DUAL),'primary_only_routes':len(PRIMARY_ONLY),'reference_grid':'primary official acquisition timestamps only','correction_quantile':Q,'correction_window_bars':WINDOW,'correction_min_periods':MIN_PERIODS,'parameter_tuning_used':False,'champion_outcomes_used_in_construction':False,'quant_v2_modified':False,'total_close_clips':sum(x['close_clip_count'] for x in audits),'total_high_clips':sum(x['high_clip_count'] for x in audits),'total_low_clips':sum(x['low_clip_count'] for x in audits),'total_gap_clips':sum(x['gap_clip_count'] for x in audits),'routes_detail':audits}
    Path(args.audit).write_text(json.dumps(summary,indent=2)); print(json.dumps({k:v for k,v in summary.items() if k!='routes_detail'},indent=2))
if __name__=='__main__': main()
