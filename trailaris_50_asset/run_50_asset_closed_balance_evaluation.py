#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

CORE=[
'EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF',
'XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS',
'V75','V50','V100','V25','V10']


def _worker(job):
    asset,path,records=job
    import pandas as pd
    import trailaris_r4_full_universe_loop as r4
    from qv2_asset_universe_adapter import install_universe
    install_universe(r4,pd.DataFrame(records))
    return r4.build_asset_components((asset,path))


def build_generalised_universe(rawdir:Path,assets:list[str],meta:pd.DataFrame,r4):
    from qv2_asset_universe_adapter import install_universe
    install_universe(r4,meta)
    records=meta.to_dict('records')
    jobs=[]
    for asset in assets:
        f=rawdir/f'{asset}_M1_normalized.csv'
        if not f.exists(): raise RuntimeError(f'missing M1 input: {asset}')
        jobs.append((asset,str(f),records))
    bundles={};workers=min(8,max(2,os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(_worker,j):j[0] for j in jobs}
        for fut in as_completed(futs):
            asset,c,xf,o=fut.result();bundles[asset]=(c,xf,o)
    allc=[];allo=[];fcache={}
    for asset in assets:
        c,xf,o=bundles[asset]
        if xf is None or len(xf)==0: raise RuntimeError(f'empty feature cache: {asset}')
        xf=xf.copy();xf['timestamp']=pd.to_datetime(xf['timestamp'],utc=True,errors='coerce');xf=xf.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        fcache[asset]=xf
        if c is not None and len(c): allc.append(c)
        if o is not None and len(o): allo.append(o)
    base=pd.concat(allc,ignore_index=True)
    extras=[r4.generate_relative_value_candidates(fcache),r4.generate_intermarket_candidates(fcache)]
    ff=r4.load_factor_feed(rawdir/'Trailaris_R3_FactorFeed.csv')
    extras.append(r4.generate_factor_candidates(fcache,ff,'research-proxy'))
    cands=pd.concat([x for x in [base,*extras] if x is not None and len(x)],ignore_index=True)
    ens=r4.generate_ensemble_candidates(cands,fcache)
    if ens is not None and len(ens): cands=pd.concat([cands,ens],ignore_index=True)
    import trailaris_r4_asset_adapter as r4a
    m=cands.apply(lambda z:(str(z.strategy_family)=='CAMPAIGN_MANAGEMENT') or r4a.family_applicable(str(z.asset),str(z.strategy_family)),axis=1)
    cands=cands[m].reset_index(drop=True)
    cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    return cands,fcache


def weeks_from(cands):
    t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max();first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC')
    weeks=[];wk=first
    while wk<=t1: weeks.append(wk);wk+=pd.Timedelta(days=7)
    if len(weeks)!=26: raise RuntimeError(f'expected 26 evaluated weeks, got {len(weeks)}')
    return weeks,t1


def promote(cands,weeks,qv2):
    out=[]
    for wk in weeks:
        p,_,_=qv2.promote_for_week(cands,wk)
        if len(p): out.append(p.assign(eval_week=wk))
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()


def rolling(W,n=11):
    r=pd.to_numeric(W.weekly_return_pct,errors='coerce').fillna(0).to_numpy(float);rows=[]
    for i in range(len(W)-n+1):
        z=r[i:i+n];rows.append({'window':i+1,'start_week':str(W.week_start.iloc[i]),'end_week':str(W.week_start.iloc[i+n-1]),'compounded_return_pct':float((np.prod(1+z/100)-1)*100),'positive_weeks':int((z>0).sum()),'weeks_ge5':int((z>=5).sum()),'weeks_ge10':int((z>=10).sum()),'max_weekly_dd_pct':float(pd.to_numeric(W.max_drawdown_pct.iloc[i:i+n],errors='coerce').min())})
    return pd.DataFrame(rows)


def replay(P,weeks,t1,specs,fcache,qv2,r4):
    equity=100.;Wrows=[];events=[];decisions=[]
    for wk in weeks:
        pr=P.loc[P.eval_week.eq(wk)].drop(columns=['eval_week'],errors='ignore').copy() if len(P) else pd.DataFrame()
        ev,de=qv2.replay_r5(pr,specs,equity,fcache) if len(pr) else (pd.DataFrame(),pd.DataFrame())
        if len(ev): equity=float(ev.equity.iloc[-1])
        if len(ev): events.append(ev.assign(eval_week=wk))
        if len(de): decisions.append(de.assign(eval_week=wk))
        prev=Wrows[-1]['end_equity'] if Wrows else 100.
        w=r4.weekly(ev,start=prev,feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),t1))
        if len(w): rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
        else: rec=dict(week_start=wk,start_equity=prev,end_equity=prev,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False);equity=prev
        Wrows.append(rec)
    W=pd.DataFrame(Wrows);EV=pd.concat(events,ignore_index=True) if events else pd.DataFrame();DE=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame();R=rolling(W)
    wins=int((EV.net_pnl>0).sum()) if len(EV) else 0;losses=int((EV.net_pnl<0).sum()) if len(EV) else 0;flats=int((EV.net_pnl==0).sum()) if len(EV) else 0
    gw=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum()) if len(EV) else 0.;gl=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum()) if len(EV) else 0.
    eq=pd.concat([pd.Series([100.]),pd.to_numeric(EV.sort_values('timestamp').equity,errors='coerce').dropna()],ignore_index=True) if len(EV) else pd.Series([100.]);dd=(eq/eq.cummax()-1)*100
    selected=pd.Series(dtype=bool)
    if len(DE) and 'selected' in DE:
        selected=DE['selected'].astype(str).str.lower().isin(['true','1','yes'])
    metrics={'currency':'USD','start_equity':100.0,'end_equity':float(W.end_equity.iloc[-1]),'compounded_return_pct':float((W.end_equity.iloc[-1]/100.-1)*100.),'evaluated_weeks':len(W),'positive_weeks':int((W.weekly_return_pct>0).sum()),'negative_weeks':int((W.weekly_return_pct<0).sum()),'weeks_ge5':int((W.weekly_return_pct>=5).sum()),'weeks_ge10':int((W.weekly_return_pct>=10).sum()),'mean_week_pct':float(W.weekly_return_pct.mean()),'median_week_pct':float(W.weekly_return_pct.median()),'executed_trades':len(EV),'wins':wins,'losses':losses,'flats':flats,'nonflat_win_rate':float(wins/(wins+losses)) if wins+losses else 0.,'profit_factor':float(gw/gl) if gl else (999. if gw else 0.),'expectancy_r':float(pd.to_numeric(EV.net_r,errors='coerce').mean()) if len(EV) and 'net_r' in EV else 0.,'max_weekly_drawdown_pct':float(pd.to_numeric(W.max_drawdown_pct,errors='coerce').min()),'max_event_equity_drawdown_pct':float(dd.min()),'promoted_rows':len(P),'promoted_assets':int(P.asset.nunique()) if len(P) else 0,'decision_rows':len(DE),'selected_decision_rows':int(selected.sum()) if len(selected) else 0,'opportunity_capture_ratio':float(selected.mean()) if len(selected) else 0.,'rolling_11w_min_return_pct':float(R.compounded_return_pct.min()) if len(R) else 0.,'rolling_11w_median_return_pct':float(R.compounded_return_pct.median()) if len(R) else 0.,'rolling_11w_max_return_pct':float(R.compounded_return_pct.max()) if len(R) else 0.}
    contrib=pd.DataFrame()
    if len(EV):
        contrib=EV.groupby('asset').agg(trades=('net_pnl','size'),net_pnl=('net_pnl','sum'),wins=('net_pnl',lambda s:int((s>0).sum())),losses=('net_pnl',lambda s:int((s<0).sum()))).reset_index();den=float(contrib.net_pnl.abs().sum()) or 1.;contrib['absolute_pnl_share']=contrib.net_pnl.abs()/den;contrib=contrib.sort_values('net_pnl',ascending=False)
        metrics['top_asset_absolute_pnl_concentration']=float(contrib.absolute_pnl_share.max())
    else: metrics['top_asset_absolute_pnl_concentration']=0.
    return metrics,W,EV,DE,R,contrib


def same_control(x,y):
    ints=['positive_weeks','negative_weeks','executed_trades','wins','losses','flats']
    nums=['end_equity','compounded_return_pct','profit_factor','max_weekly_drawdown_pct','max_event_equity_drawdown_pct']
    diffs={}
    for k in ints: diffs[k]=int(x.get(k,0))-int(y.get(k,0))
    for k in nums: diffs[k]=float(x.get(k,0))-float(y.get(k,0))
    ok=all(v==0 for k,v in diffs.items() if k in ints) and all(abs(diffs[k])<=1e-8 for k in nums)
    return ok,diffs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--specs',type=Path,required=True);ap.add_argument('--control-status',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    approved=[x.strip() for x in Path('trailaris_50_asset/QV2_50_APPROVED_EVALUATION_16.txt').read_text().splitlines() if x.strip()]
    if len(approved)!=16 or len(set(approved))!=16: raise RuntimeError('approved 16 integrity failure')
    m=pd.read_csv(a.manifest,sep='\t');m=m[m.canonical_instrument.astype(str).isin(approved)].copy();
    if set(m.canonical_instrument.astype(str))!=set(approved): raise RuntimeError('approved provider manifest mismatch')
    meta34=pd.DataFrame({'asset':CORE,'discovery_provider_name':['']*len(CORE)});meta16=m.rename(columns={'canonical_instrument':'asset','provider_name':'discovery_provider_name'})[['asset','discovery_provider_name']];meta50=pd.concat([meta34,meta16],ignore_index=True)
    import trailaris_r4_full_universe_loop as r4
    import R5_BASE_CAUSAL_ENGINE as base
    import quantitative_integrated_v2 as qv2
    from qv2_asset_universe_adapter import extend_research_specs
    base.r4.factor_vec=r4.factor_vec
    base_specs=r4.load_specs(a.specs,'research-proxy')

    c34,f34=build_generalised_universe(a.rawdir,CORE,meta34,r4);w34,t134=weeks_from(c34);p34=promote(c34,w34,qv2);m34,W34,E34,D34,R34,C34=replay(p34,w34,t134,base_specs,f34,qv2,r4)
    control=json.loads(a.control_status.read_text())['candidate'];ok,diffs=same_control(m34,control)
    if not ok:
        status={'state':'BASELINE_INTEGRITY_MISMATCH__50_EVALUATION_NOT_ACCEPTED','control':control,'reconstructed_34':m34,'delta_reconstructed_minus_control':diffs,'approved_16':approved}
        (a.outdir/'QV2_50_EVALUATION_STATUS.json').write_text(json.dumps(status,indent=2,default=str));W34.to_csv(a.outdir/'QV2_34_RECONSTRUCTED_WEEKLY.csv',index=False);raise RuntimeError(json.dumps(status,default=str))

    c50,f50=build_generalised_universe(a.rawdir,CORE+approved,meta50,r4);w50,t150=weeks_from(c50)
    if w50!=w34: raise RuntimeError('34/50 evaluation week mismatch')
    p50=promote(c50,w50,qv2);specs50=extend_research_specs(base_specs,meta50);m50,W50,E50,D50,R50,C50=replay(p50,w50,t150,specs50,f50,qv2,r4)
    delta={k:(m50[k]-control[k]) for k in m50 if k in control and isinstance(m50[k],(int,float,np.integer,np.floating)) and isinstance(control[k],(int,float,np.integer,np.floating))}
    gates={'all_26_weeks_positive':m50['positive_weeks']==26,'ending_balance_improved':m50['end_equity']>control['end_equity'],'profit_factor_not_worse':m50['profit_factor']>=control['profit_factor'],'weekly_drawdown_not_worse':m50['max_weekly_drawdown_pct']>=control['max_weekly_drawdown_pct'],'event_drawdown_not_worse':m50['max_event_equity_drawdown_pct']>=control['max_event_equity_drawdown_pct']}
    status={'state':'QV2_50_CLOSED_BALANCE_EVALUATION_COMPLETE','evidence_class':'CONTROLLED_50_ASSET_RESEARCH_PROXY__NOT_BROKER_CERTIFIED','strategy_modified':False,'fresh_position_sizing':'CURRENT_REALIZED_CLOSED_BALANCE','floating_pnl_enlarges_fresh_sizing_base':False,'approved_additions':approved,'asset_count':50,'baseline_integrity_reproduction':'PASS','baseline_34':control,'candidate_50':m50,'delta_50_minus_34':delta,'comparison_gates':gates,'successor_gate_pass':bool(all(gates.values()))}
    (a.outdir/'QV2_50_EVALUATION_STATUS.json').write_text(json.dumps(status,indent=2,default=str));W34.to_csv(a.outdir/'QV2_34_RECONSTRUCTED_WEEKLY.csv',index=False);W50.to_csv(a.outdir/'QV2_50_WEEKLY.csv',index=False);E50.to_csv(a.outdir/'QV2_50_EVENTS.csv.gz',index=False,compression='gzip');D50.to_csv(a.outdir/'QV2_50_DECISIONS.csv.gz',index=False,compression='gzip');R50.to_csv(a.outdir/'QV2_50_ROLLING_11W.csv',index=False);C50.to_csv(a.outdir/'QV2_50_ASSET_CONTRIBUTION.csv',index=False);p50.to_csv(a.outdir/'QV2_50_PROMOTED.csv.gz',index=False,compression='gzip');print(json.dumps(status,indent=2,default=str))

if __name__=='__main__': main()
