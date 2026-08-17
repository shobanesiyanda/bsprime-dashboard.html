#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROUTES=['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
DUAL=set(ROUTES)-{'US30','NATGAS','WTI','V75','V50','V100','V25','V10'}
COLS=['timestamp','open','high','low','close','bid','ask','spread','volume','source','asset']


def load(path: Path) -> pd.DataFrame:
    d=pd.read_csv(path)
    d['timestamp']=pd.to_datetime(d['timestamp'],utc=True,errors='coerce')
    for c in ['open','high','low','close','volume']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['timestamp','open','high','low','close']).sort_values('timestamp').drop_duplicates('timestamp',keep='last')
    return d


def med(values):
    v=[float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def components(d: pd.DataFrame, tag: str) -> pd.DataFrame:
    x=d[['timestamp','open','high','low','close','volume']].copy().sort_values('timestamp')
    prev=x['close'].shift(1)
    gap=x['timestamp'].diff().dt.total_seconds()
    x[f'{tag}_gap']=np.log(x['open']/prev)
    x[f'{tag}_bar_c']=np.log(x['close']/x['open'])
    x[f'{tag}_bar_h']=np.log(x['high']/x['open'])
    x[f'{tag}_bar_l']=np.log(x['low']/x['open'])
    x[f'{tag}_prev_gap_s']=gap
    return x.rename(columns={c:f'{tag}_{c}' for c in ['open','high','low','close','volume']})


def build_dual(asset: str, p: pd.DataFrame, q: pd.DataFrame):
    a=components(p,'p'); b=components(q,'q')
    z=pd.merge(a,b,on='timestamp',how='outer').sort_values('timestamp').reset_index(drop=True)
    rows=[]; prev_can=None; disagreement=[]
    for _,r in z.iterrows():
        # A reconnect return after an isolated provider gap is not mixed into a one-minute
        # consensus bar. When both providers have a market/session gap, both are allowed.
        pg=r.get('p_prev_gap_s'); qg=r.get('q_prev_gap_s')
        p_ok_gap=pd.notna(r.get('p_gap')) and (pd.isna(pg) or pg<=90)
        q_ok_gap=pd.notna(r.get('q_gap')) and (pd.isna(qg) or qg<=90)
        if pd.notna(pg) and pd.notna(qg) and pg>90 and qg>90:
            p_ok_gap=pd.notna(r.get('p_gap')); q_ok_gap=pd.notna(r.get('q_gap'))
        gaps=[r.get('p_gap') if p_ok_gap else None,r.get('q_gap') if q_ok_gap else None]
        bc=med([r.get('p_bar_c'),r.get('q_bar_c')]); bh=med([r.get('p_bar_h'),r.get('q_bar_h')]); bl=med([r.get('p_bar_l'),r.get('q_bar_l')])
        if bc is None or bh is None or bl is None: continue
        if prev_can is None:
            anchor=med([r.get('p_open'),r.get('q_open')])
            if anchor is None or anchor<=0: continue
            op=anchor
        else:
            g=med(gaps); op=prev_can*math.exp(g if g is not None else 0.0)
        cl=op*math.exp(bc); hi=op*math.exp(bh); lo=op*math.exp(bl)
        hi=max(hi,op,cl); lo=min(lo,op,cl)
        # Volume is not provider-comparable for HistData (bar volume is zero). Use the
        # primary feed's native volume as an auxiliary field; price formation is dual-feed.
        vol=r.get('p_volume'); vol=float(vol) if pd.notna(vol) else 0.0
        rows.append([r.timestamp,op,hi,lo,cl,cl,'','',vol,'CANONICAL_DUAL_PROVIDER_RETURN_MEDIAN_R1',asset])
        prev_can=cl
        if pd.notna(r.get('p_bar_c')) and pd.notna(r.get('q_bar_c')):
            disagreement.append(abs(float(r.p_bar_c)-float(r.q_bar_c)))
    out=pd.DataFrame(rows,columns=COLS)
    audit={'asset':asset,'mode':'DUAL_PROVIDER_RETURN_MEDIAN_R1','rows':len(out),'primary_rows':len(p),'independent_rows':len(q),'overlap_rows':int(pd.merge(p[['timestamp']],q[['timestamp']],on='timestamp').shape[0]),'median_abs_intrabar_logreturn_disagreement':float(np.median(disagreement)) if disagreement else None,'p95_abs_intrabar_logreturn_disagreement':float(np.quantile(disagreement,.95)) if disagreement else None}
    return out,audit


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--primary',required=True); ap.add_argument('--independent',required=True); ap.add_argument('--out',required=True); ap.add_argument('--audit',required=True); a=ap.parse_args()
    primary=Path(a.primary); independent=Path(a.independent); outdir=Path(a.out); outdir.mkdir(parents=True,exist_ok=True)
    audits=[]
    for asset in ROUTES:
        pp=primary/f'{asset}_M1_normalized.csv'; assert pp.exists(),f'primary missing {asset}'
        p=load(pp)
        if asset in DUAL:
            qp=independent/f'{asset}_M1_normalized.csv'; assert qp.exists(),f'independent missing {asset}'
            q=load(qp); out,au=build_dual(asset,p,q)
        else:
            out=p.copy(); out['bid']=out['close']; out['ask']=''; out['spread']=''; out['source']='CANONICAL_PRIMARY_ONLY_R1'; out['asset']=asset; out=out[COLS]
            au={'asset':asset,'mode':'PRIMARY_ONLY_R1','rows':len(out),'primary_rows':len(p),'independent_rows':0,'overlap_rows':0,'median_abs_intrabar_logreturn_disagreement':None,'p95_abs_intrabar_logreturn_disagreement':None}
        assert len(out)>=1000,(asset,len(out)); out.to_csv(outdir/f'{asset}_M1_normalized.csv',index=False,date_format='%Y-%m-%dT%H:%M:%S.%fZ'); audits.append(au)
    s={'state':'CANONICAL_PROVIDER_INVARIANCE_R1_COMPLETE','routes':34,'dual_provider_routes':len(DUAL),'primary_only_routes':34-len(DUAL),'price_rule':'equal-weight median of provider log gap and intrabar OHLC moves; with two feeds median equals midpoint','volume_rule':'primary auxiliary volume retained because HistData bar volume is not provider-comparable','parameter_tuning_used':False,'champion_outcomes_used_in_construction':False,'quant_v2_modified':False,'routes_detail':audits}
    Path(a.audit).write_text(json.dumps(s,indent=2)); print(json.dumps({k:v for k,v in s.items() if k!='routes_detail'},indent=2))
if __name__=='__main__': main()
