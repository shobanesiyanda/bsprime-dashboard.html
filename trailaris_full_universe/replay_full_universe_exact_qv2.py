#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from qv2_asset_universe_adapter import install_universe, extend_research_specs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selection-dir',type=Path,required=True);ap.add_argument('--feature-root',type=Path,required=True);ap.add_argument('--specs',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    cands=pd.read_csv(a.selection_dir/'ALL_CANDIDATES.csv.gz');cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    try:P=pd.read_csv(a.selection_dir/'PROMOTED_SELECTION.csv.gz')
    except pd.errors.EmptyDataError:P=pd.DataFrame()
    if len(P):
        P['decision_time']=pd.to_datetime(P.decision_time,utc=True);P['exit_time']=pd.to_datetime(P.exit_time,utc=True);P['eval_week']=pd.to_datetime(P.eval_week,utc=True)
    fcache={}
    for af in sorted(a.feature_root.rglob('FEATURE_ACCEPTED.csv')):
        try:z=pd.read_csv(af)
        except pd.errors.EmptyDataError:continue
        for _,r in z.iterrows():
            fp=af.parent/str(r.feature_file)
            if not fp.exists():continue
            x=pd.read_csv(fp);x['timestamp']=pd.to_datetime(x.timestamp,utc=True);fcache[str(r.canonical_instrument)]=x.sort_values('timestamp').reset_index(drop=True)
    promoted_assets=set(P.asset.astype(str)) if len(P) else set();missing=sorted(promoted_assets-set(fcache))
    if missing:raise RuntimeError(f'missing selected feature caches: {missing[:30]} total={len(missing)}')
    import trailaris_r4_full_universe_loop as r4
    meta=(P[['asset','discovery_provider_name']].drop_duplicates() if len(P) else cands[['asset','discovery_provider_name']].drop_duplicates())
    mapping=install_universe(r4,meta)
    import R5_BASE_CAUSAL_ENGINE as base
    base.r4.factor_vec=r4.factor_vec
    import quantitative_integrated_v2 as qv2
    base_specs=r4.load_specs(a.specs,'research-proxy');specs=extend_research_specs(base_specs,meta)
    t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max();first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC')
    weeks=[];wk=first
    while wk<=t1:weeks.append(wk);wk+=pd.Timedelta(days=7)
    if len(weeks)!=26:raise RuntimeError(f'expected 26 evaluated weeks, got {len(weeks)}')
    equity=100.0;W=[];events=[];decisions=[]
    for wk in weeks:
        pr=P.loc[P.eval_week.eq(wk)].copy() if len(P) else pd.DataFrame()
        if len(pr):
            ev,de=qv2.replay_r5(pr,specs,equity,fcache)
            if len(ev):equity=float(ev.equity.iloc[-1])
            events.append(ev.assign(eval_week=wk));decisions.append(de.assign(eval_week=wk))
        else:ev=pd.DataFrame()
        prev=W[-1]['end_equity'] if W else 100.0
        w=r4.weekly(ev,start=prev,feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),t1))
        if len(w):rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
        else:rec=dict(week_start=wk,start_equity=prev,end_equity=prev,weekly_return_pct=0.0,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.0,ge5=False,ge10=False);equity=prev
        W.append(rec)
    W=pd.DataFrame(W);EV=pd.concat(events,ignore_index=True) if events else pd.DataFrame();DE=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame()
    if len(EV):
        wins=int((EV.net_pnl>0).sum());losses=int((EV.net_pnl<0).sum());flats=int((EV.net_pnl==0).sum());gw=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum());gl=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum());eq=pd.concat([pd.Series([100.0]),pd.to_numeric(EV.sort_values('timestamp').equity,errors='coerce').dropna()],ignore_index=True);dd=(eq/eq.cummax()-1)*100
    else:wins=losses=flats=0;gw=gl=0.0;dd=pd.Series([0.0])
    nonflat=wins+losses
    metrics={'currency':'USD','start_equity':100.0,'end_equity':float(W.end_equity.iloc[-1]),'compounded_return_pct':float((W.end_equity.iloc[-1]/100.0-1)*100),'evaluated_weeks':len(W),'positive_weeks':int((W.weekly_return_pct>0).sum()),'negative_weeks':int((W.weekly_return_pct<0).sum()),'weeks_ge5':int((W.weekly_return_pct>=5).sum()),'weeks_ge10':int((W.weekly_return_pct>=10).sum()),'mean_week_pct':float(W.weekly_return_pct.mean()),'median_week_pct':float(W.weekly_return_pct.median()),'executed_trades':len(EV),'wins':wins,'losses':losses,'flats':flats,'nonflat_win_rate':float(wins/nonflat) if nonflat else 0.0,'profit_factor':float(gw/gl) if gl else (999.0 if gw else 0.0),'max_weekly_drawdown_pct':float(W.max_drawdown_pct.min()),'max_event_equity_drawdown_pct':float(dd.min()),'candidate_assets':int(cands.asset.nunique()),'promoted_assets':len(promoted_assets),'feature_assets':len(fcache),'assets_with_selected':int(DE.loc[DE.selected.astype(bool),'asset'].nunique()) if len(DE) and 'selected' in DE else 0}
    status={'state':'FULL_UNIVERSE_EXACT_FROZEN_QV2_REPLAY_COMPLETE','evidence_class':'INDEPENDENT_EXPANDED_UNIVERSE_RESEARCH_PROXY; FROZEN_QV2_EXECUTION_SEMANTICS; NOT_BROKER_CERTIFIED','frozen_quant_v2_blob':'120390ab52e80f51bafedc42389ceb7420a410f2','quant_v2_modified':False,'normalized_r_used':False,'broker_research_proxy_sizing_used':True,'canonical_identity_precondition_required':True,'candidate':metrics}
    W.to_csv(a.outdir/'FULL_UNIVERSE_EXACT_QV2_WEEKLY.csv',index=False);EV.to_csv(a.outdir/'FULL_UNIVERSE_EXACT_QV2_EVENTS.csv.gz',index=False,compression='gzip');DE.to_csv(a.outdir/'FULL_UNIVERSE_EXACT_QV2_DECISIONS.csv.gz',index=False,compression='gzip');pd.DataFrame([{'asset':k,'anchor':v} for k,v in sorted(mapping.items())]).to_csv(a.outdir/'FULL_UNIVERSE_ADAPTER_MAP.csv',index=False);(a.outdir/'FULL_UNIVERSE_EXACT_QV2_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
