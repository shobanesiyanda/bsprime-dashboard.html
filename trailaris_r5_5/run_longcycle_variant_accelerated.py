#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, multiprocessing as mp, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

CONTROL = {
    'end_equity':1136.4556497760627,'compounded_return_pct':1036.4556497760627,'executed_trades':2152,'wins':856,'losses':505,'flats':791,
    'nonflat_win_rate':0.6289493019838354,'profit_factor':2.791840637059962,'max_weekly_drawdown_pct':-2.9475910077805922,'max_event_equity_drawdown_pct':-3.1202037867421484,
    'positive_weeks':26,'rolling_11w_min_return_pct':151.2997031258902,'rolling_11w_median_return_pct':175.621313340644,'rolling_11w_mean_return_pct':176.56265438902557,'rolling_11w_max_return_pct':212.75588274483837,
}
_G_CANDS=None;_G_MOD=None

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def init_worker(cands,module_path):
    global _G_CANDS,_G_MOD;_G_CANDS=cands;_G_MOD=load(Path(module_path),'variant_worker')

def promote_worker(wk):
    pr,cf,st=_G_MOD.promote_for_week(_G_CANDS,pd.Timestamp(wk));return str(pd.Timestamp(wk)),pr,cf,st

def equal_key(a,b):
    if len(a)!=len(b):return False
    if len(a)==0:return True
    cols=[c for c in ['campaign_id','decision_time','asset','strategy','promotion_tier','risk_fraction','rank_score','reliability_score'] if c in a.columns and c in b.columns]
    x=a[cols].copy().sort_values(cols[:2] if len(cols)>=2 else cols).reset_index(drop=True);y=b[cols].copy().sort_values(cols[:2] if len(cols)>=2 else cols).reset_index(drop=True)
    for c in cols:
        if pd.api.types.is_numeric_dtype(x[c]) or pd.api.types.is_numeric_dtype(y[c]):
            if not np.allclose(pd.to_numeric(x[c],errors='coerce').fillna(0),pd.to_numeric(y[c],errors='coerce').fillna(0),rtol=0,atol=1e-12):return False
        elif not x[c].astype(str).equals(y[c].astype(str)):return False
    return True

def rolling(W,n=11):
    r=pd.to_numeric(W.weekly_return_pct,errors='coerce').fillna(0).to_numpy(float);rows=[]
    for i in range(len(W)-n+1):
        z=r[i:i+n];ret=(np.prod(1+z/100)-1)*100
        rows.append({'window':i+1,'start_week':str(W.week_start.iloc[i]),'end_week':str(W.week_start.iloc[i+n-1]),'compounded_return_pct':float(ret),'mean_week_pct':float(np.mean(z)),'median_week_pct':float(np.median(z)),'positive_weeks':int((z>0).sum()),'weeks_ge5':int((z>=5).sum()),'weeks_ge10':int((z>=10).sum()),'max_weekly_dd_pct':float(pd.to_numeric(W.max_drawdown_pct.iloc[i:i+n],errors='coerce').min())})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--variant',required=True);ap.add_argument('--name',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);sys.path[:0]=['trailaris_r4','trailaris_r5_4','trailaris_r5_4/reliability_variants','trailaris_r5_5']
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    mod=load(Path(a.variant),a.name+'_main')
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    if len(data)!=34:raise RuntimeError(f'full-universe data gate failed {len(data)}/34')
    audit=Path('trailaris_longcycle/results/R4_FACTORY_AUDIT.csv');fa=pd.read_csv(audit);fams=int(fa.strategy_family.nunique())
    if len(fa)!=510 or fams!=15:raise RuntimeError(f'factory lock failed cells={len(fa)} families={fams}')
    t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max();first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC');weeks=[];wk=first
    while wk<=t1:weeks.append(wk);wk+=pd.Timedelta(days=7)
    if len(weeks)!=26:raise RuntimeError(f'expected 26 evaluated weeks, got {len(weeks)}')
    guards=[weeks[3],weeks[len(weeks)//2],weeks[-1]];seq={str(w):mod.promote_for_week(cands,w)[0] for w in guards};prom={}
    ctx=mp.get_context('fork');workers=min(8,max(2,os.cpu_count() or 4));module_path=str(Path(a.variant).resolve())
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_worker,initargs=(cands,module_path)) as ex:
        futs={ex.submit(promote_worker,w):w for w in weeks}
        for f in as_completed(futs):k,pr,cf,st=f.result();prom[k]=pr
    for w in guards:
        if not equal_key(seq[str(w)],prom[str(w)]):raise RuntimeError(f'parallel semantic guard failed {w}')
    equity=100.;W=[];events=[];decisions=[];promoted=[]
    for wk in weeks:
        pr=prom[str(wk)]
        if len(pr):
            ev,de=mod.replay_r5(pr,specs,equity,fcache)
            if len(ev):equity=float(ev.equity.iloc[-1])
            events.append(ev.assign(eval_week=wk));decisions.append(de.assign(eval_week=wk));promoted.append(pr.assign(eval_week=wk))
        else:ev=pd.DataFrame();de=pd.DataFrame()
        w=r4.weekly(ev,start=(W[-1]['end_equity'] if W else 100.),feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),t1))
        if len(w):rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
        else:
            st=W[-1]['end_equity'] if W else 100.;rec=dict(week_start=wk,start_equity=st,end_equity=st,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False);equity=st
        W.append(rec)
    W=pd.DataFrame(W);EV=pd.concat(events,ignore_index=True) if events else pd.DataFrame();DE=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame();PR=pd.concat(promoted,ignore_index=True) if promoted else pd.DataFrame();R=rolling(W)
    wins=int((EV.net_pnl>0).sum());losses=int((EV.net_pnl<0).sum());flats=int((EV.net_pnl==0).sum());gw=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum());gl=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum())
    eq=pd.concat([pd.Series([100.]),pd.to_numeric(EV.sort_values('timestamp').equity,errors='coerce').dropna()],ignore_index=True);dd=(eq/eq.cummax()-1)*100
    m={'end_equity':float(W.end_equity.iloc[-1]),'compounded_return_pct':float((W.end_equity.iloc[-1]/100-1)*100),'evaluated_weeks':len(W),'positive_weeks':int((W.weekly_return_pct>0).sum()),'negative_weeks':int((W.weekly_return_pct<0).sum()),'mean_week_pct':float(W.weekly_return_pct.mean()),'median_week_pct':float(W.weekly_return_pct.median()),'weeks_ge5':int((W.weekly_return_pct>=5).sum()),'weeks_ge10':int((W.weekly_return_pct>=10).sum()),'executed_trades':len(EV),'wins':wins,'losses':losses,'flats':flats,'loss_rate':float(losses/len(EV)),'flat_rate':float(flats/len(EV)),'nonflat_win_rate':float(wins/(wins+losses)),'profit_factor':float(gw/gl) if gl else 999.,'max_weekly_drawdown_pct':float(W.max_drawdown_pct.min()),'max_event_equity_drawdown_pct':float(dd.min()),'assets_with_selected':int(DE.loc[DE.selected,'asset'].nunique()),'rolling_11w_windows':len(R),'rolling_11w_min_return_pct':float(R.compounded_return_pct.min()),'rolling_11w_median_return_pct':float(R.compounded_return_pct.median()),'rolling_11w_upper_quartile_return_pct':float(R.compounded_return_pct.quantile(.75)),'rolling_11w_mean_return_pct':float(R.compounded_return_pct.mean()),'rolling_11w_max_return_pct':float(R.compounded_return_pct.max())}
    keys=[k for k in CONTROL if k in m];delta={k:m[k]-CONTROL[k] for k in keys}
    extra={}
    for col in ['recovery_reallocated','recovery_qualified']:
        if col in DE.columns:extra[col+'_selected_count']=int((DE.selected & DE[col].fillna(False).astype(bool)).sum())
    status={'state':'R5_5_26W_CAUSAL_VARIANT_REPLAY_COMPLETE','variant':a.name,'parallel_semantic_guard':'PASS','scope':{'routes':34,'strategy_families':15,'route_strategy_cells':510},'frozen_control':CONTROL,'candidate':m,'delta_candidate_minus_control':delta,'variant_counters':extra,'automatic_baseline_promotion':False,'evidence_class':'FULL_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE'}
    stem=a.name.upper();W.to_csv(out/f'{stem}_WEEKLY.csv',index=False);R.to_csv(out/f'{stem}_ROLLING_11W.csv',index=False);EV.to_csv(out/f'{stem}_EVENTS.csv',index=False);DE.to_csv(out/f'{stem}_DECISIONS.csv',index=False);PR.to_csv(out/f'{stem}_PROMOTED.csv',index=False);(out/f'{stem}_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
