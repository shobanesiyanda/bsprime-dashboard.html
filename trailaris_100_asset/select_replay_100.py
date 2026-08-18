#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe, extend_research_specs


def metrics(W,EV,DE,cands,P,fcache):
    wins=int((EV.net_pnl>0).sum()) if len(EV) else 0;losses=int((EV.net_pnl<0).sum()) if len(EV) else 0;flats=int((EV.net_pnl==0).sum()) if len(EV) else 0
    gw=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum()) if len(EV) else 0.;gl=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum()) if len(EV) else 0.
    if len(EV):
        eq=pd.concat([pd.Series([100.]),pd.to_numeric(EV.sort_values('timestamp').equity,errors='coerce').dropna()],ignore_index=True);dd=(eq/eq.cummax()-1)*100
    else:dd=pd.Series([0.])
    nonflat=wins+losses
    return {'currency':'USD','start_balance':100.0,'end_balance':float(W.end_equity.iloc[-1]),'compounded_return_pct':float((W.end_equity.iloc[-1]/100.-1)*100),'evaluated_weeks':int(len(W)),'positive_weeks':int((W.weekly_return_pct>0).sum()),'negative_weeks':int((W.weekly_return_pct<0).sum()),'weeks_ge5':int((W.weekly_return_pct>=5).sum()),'weeks_ge10':int((W.weekly_return_pct>=10).sum()),'mean_week_pct':float(W.weekly_return_pct.mean()),'median_week_pct':float(W.weekly_return_pct.median()),'executed_trades':int(len(EV)),'wins':wins,'losses':losses,'flats':flats,'nonflat_win_rate':float(wins/nonflat) if nonflat else 0.,'profit_factor':float(gw/gl) if gl else (999. if gw else 0.),'max_weekly_drawdown_pct':float(pd.to_numeric(W.max_drawdown_pct,errors='coerce').min()),'max_event_equity_drawdown_pct':float(dd.min()),'candidate_assets':int(cands.asset.nunique()),'promoted_assets':int(P.asset.nunique()) if len(P) else 0,'feature_assets':len(fcache),'assets_with_selected':int(DE.loc[DE.selected.astype(bool),'asset'].nunique()) if len(DE) and 'selected' in DE else 0}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--universe-dir',type=Path,required=True);ap.add_argument('--specs',type=Path,required=True);ap.add_argument('--balance-control',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    cands=pd.read_csv(a.universe_dir/'ALL_CANDIDATES.csv.gz');cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    idx=pd.read_csv(a.universe_dir/'FEATURE_INDEX.csv');fcache={}
    for _,r in idx.iterrows():
        x=pd.read_csv(a.universe_dir/str(r.feature_file));x['timestamp']=pd.to_datetime(x.timestamp,utc=True);fcache[str(r.asset)]=x.sort_values('timestamp').reset_index(drop=True)
    if len(fcache)!=100:raise RuntimeError(f'feature cache integrity {len(fcache)}/100')
    import trailaris_r4_full_universe_loop as r4
    meta=idx[['asset','discovery_provider_name']].drop_duplicates().copy();install_universe(r4,meta)
    import R5_BASE_CAUSAL_ENGINE as base;base.r4.factor_vec=r4.factor_vec
    import quantitative_integrated_v2 as qv2
    t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max();first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC')
    weeks=[];wk=first
    while wk<=t1:weeks.append(wk);wk+=pd.Timedelta(days=7)
    if len(weeks)!=26:raise RuntimeError(f'expected 26 evaluated weeks, got {len(weeks)}')
    promoted=[]
    for wk in weeks:
        p,_,_=qv2.promote_for_week(cands,wk)
        if len(p):promoted.append(p.assign(eval_week=wk))
    P=pd.concat(promoted,ignore_index=True) if promoted else pd.DataFrame()
    P.to_csv(a.outdir/'QV2_100_PROMOTED.csv.gz',index=False,compression='gzip')
    base_specs=r4.load_specs(a.specs,'research-proxy');specs=extend_research_specs(base_specs,meta)
    equity=100.;Wrows=[];events=[];decisions=[]
    raw_end=max(pd.Timestamp(x.timestamp.iloc[-1]) for x in fcache.values())
    for wk in weeks:
        pr=P.loc[P.eval_week.eq(wk)].copy() if len(P) else pd.DataFrame()
        if len(pr):
            ev,de=qv2.replay_r5(pr,specs,equity,fcache)
            if len(ev):equity=float(ev.equity.iloc[-1])
            events.append(ev.assign(eval_week=wk));decisions.append(de.assign(eval_week=wk))
        else:ev=pd.DataFrame();de=pd.DataFrame()
        prev=Wrows[-1]['end_equity'] if Wrows else 100.
        w=r4.weekly(ev,start=prev,feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),raw_end))
        if len(w):rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
        else:rec=dict(week_start=wk,start_equity=prev,end_equity=prev,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False);equity=prev
        Wrows.append(rec)
    W=pd.DataFrame(Wrows);EV=pd.concat(events,ignore_index=True) if events else pd.DataFrame();DE=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame()
    m=metrics(W,EV,DE,cands,P,fcache)
    control=json.loads(a.balance_control.read_text());c34=control['realized_closed_balance_replay'];historic=control['preserved_reference']
    delta34={
      'ending_balance_usd':m['end_balance']-float(c34['end_equity']),
      'compounded_return_percentage_points':m['compounded_return_pct']-float(c34['compounded_return_pct']),
      'profit_factor':m['profit_factor']-float(c34['profit_factor']),
      'max_weekly_drawdown_percentage_points':m['max_weekly_drawdown_pct']-float(c34['max_weekly_drawdown_pct']),
      'max_event_drawdown_percentage_points':m['max_event_equity_drawdown_pct']-float(c34['max_event_equity_drawdown_pct']),
      'executed_trades':m['executed_trades']-int(c34['executed_trades'])
    }
    status={'state':'QV2_100_ASSET_BALANCE_REPLAY_COMPLETE','experimental_delta':'ASSET_UNIVERSE_34_TO_100_ONLY','strategy_blob':'120390ab52e80f51bafedc42389ceb7420a410f2','closed_balance_compounding':True,'position_compounding':'EXISTING_STRATEGY_AND_EXPOSURE_CONTROLS','historic_34_reference':historic,'balance_compounded_34_control':c34,'candidate_100':m,'delta_100_minus_34_balance_control':delta34,'automatic_baseline_replacement':False,'evidence_class':'26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE'}
    W.to_csv(a.outdir/'QV2_100_WEEKLY.csv',index=False);EV.to_csv(a.outdir/'QV2_100_EVENTS.csv.gz',index=False,compression='gzip');DE.to_csv(a.outdir/'QV2_100_DECISIONS.csv.gz',index=False,compression='gzip');(a.outdir/'QV2_100_REPLAY_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))

if __name__=='__main__':main()
