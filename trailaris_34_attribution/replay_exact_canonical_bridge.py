#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe

MIN_ROUTE_VETO_N=24
MIN_STRATEGY_VETO_N=48
MAX_VETO_PWIN=.40
MAX_VETO_CONSERVATIVE_EV_R=-.12
MIN_VETO_TAIL_LOSS_R=.75
FACTOR_HALF_LIFE_HOURS=12.0
EPS=1e-12

def factor_support_audit(z,base):
    if z.empty:return z
    out=z.copy().sort_values(['decision_time','campaign_id']).reset_index(drop=True);state=np.zeros(len(base.r4.FACTORS));last=None;support=np.zeros(len(out))
    for t,idx in out.groupby('decision_time',sort=True).groups.items():
        t=pd.Timestamp(t)
        if last is not None:state*=math.exp(-math.log(2.)*max(0.,(t-last).total_seconds()/3600.)/FACTOR_HALF_LIFE_HOURS)
        ns=float(np.linalg.norm(state));pending=[]
        for i in list(idx):
            r=out.loc[i];fv=np.asarray(base.r4.factor_vec(str(r.asset),int(r.direction)),dtype=float);nf=float(np.linalg.norm(fv));support[i]=np.clip(float(np.dot(fv,state)/(nf*ns)) if nf>EPS and ns>EPS else 0.,-1.,1.);pending.append((fv,float(r.get('quality',.5)),float(r.get('quant_confidence',0.))))
        for fv,q,c in pending:state+=fv*max(0.,q)*(.5+.5*c)
        last=t
    out['quant_factor_support']=support;return out

def metrics(EV,W):
    wins=int((EV.net_pnl>0).sum());losses=int((EV.net_pnl<0).sum());flats=int((EV.net_pnl==0).sum());gw=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum());gl=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum());eq=pd.concat([pd.Series([100.]),pd.to_numeric(EV.sort_values('timestamp').equity,errors='coerce')],ignore_index=True);dd=(eq/eq.cummax()-1)*100
    return {'end_equity':float(W.end_equity.iloc[-1]),'compounded_return_pct':float((W.end_equity.iloc[-1]/100.-1)*100),'positive_weeks':int((W.weekly_return_pct>0).sum()),'weeks_ge5':int((W.weekly_return_pct>=5).sum()),'weeks_ge10':int((W.weekly_return_pct>=10).sum()),'executed_trades':int(len(EV)),'wins':wins,'losses':losses,'flats':flats,'nonflat_win_rate':float(wins/(wins+losses)),'profit_factor':float(gw/gl),'max_weekly_drawdown_pct':float(pd.to_numeric(W.max_drawdown_pct).min()),'max_event_equity_drawdown_pct':float(dd.min())}

def replay(PR,specs,fcache,raw_end,v2,r4):
    equity=100.;Wrows=[];EE=[];DD=[]
    for wk in sorted(pd.Timestamp(x) for x in PR.eval_week.unique()):
        src=PR[PR.eval_week.eq(wk)].drop(columns=['eval_week'],errors='ignore').copy();ev,de=v2.replay_r5(src,specs,equity,fcache)
        if len(ev):equity=float(ev.equity.iloc[-1]);EE.append(ev.assign(eval_week=wk));DD.append(de.assign(eval_week=wk))
        prev=Wrows[-1]['end_equity'] if Wrows else 100.;w=r4.weekly(ev,start=prev,feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),raw_end));rec=w.iloc[0].to_dict() if len(w) else dict(week_start=wk,start_equity=prev,end_equity=prev,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False);equity=float(rec['end_equity']);Wrows.append(rec)
    return pd.concat(EE,ignore_index=True),pd.concat(DD,ignore_index=True),pd.DataFrame(Wrows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assembled',type=Path,required=True);ap.add_argument('--canonical',type=Path,required=True);ap.add_argument('--specs',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--kept-file',type=Path);ap.add_argument('--mode',choices=['derive','replay-kept'],required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    idx=pd.read_csv(a.assembled/'FEATURE_INDEX.csv');meta=idx[['asset','discovery_provider_name']].drop_duplicates();fcache={}
    for _,r in idx.iterrows():x=pd.read_csv(a.assembled/str(r.feature_file));x['timestamp']=pd.to_datetime(x.timestamp,utc=True);fcache[str(r.asset)]=x.sort_values('timestamp').reset_index(drop=True)
    import trailaris_r4_full_universe_loop as r4;install_universe(r4,meta)
    import R5_BASE_CAUSAL_ENGINE as base;base.r4.factor_vec=r4.factor_vec
    import longcycle_integrated_negative_refinement as v2
    import quantitative_integrated_v1 as q1
    specs=r4.load_specs(a.specs,'research-proxy');raw_end=max(pd.Timestamp(x.timestamp.iloc[-1]) for x in fcache.values())
    if a.mode=='derive':
        PR0=pd.read_csv(a.canonical/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_PROMOTED.csv');cands=pd.read_csv(a.assembled/'ALL_CANDIDATES.csv.gz')
        for d in (PR0,cands):
            for c in ('entry_time','decision_time','exit_time','eval_week'):
                if c in d:d[c]=pd.to_datetime(d[c],utc=True,errors='coerce')
        kept=[];veto=[]
        for wk in sorted(pd.Timestamp(x) for x in PR0.eval_week.unique()):
            src=PR0[PR0.eval_week.eq(wk)].copy();models=q1._build_models(cands,wk);rows=[]
            for _,r in src.iterrows():
                x=r.copy()
                for k,v in q1._posterior_for_row(r,models).items():x[k]=v
                rows.append(x)
            z=pd.DataFrame(rows);z['quant_v2_original_rank_score']=pd.to_numeric(z.rank_score,errors='coerce').fillna(0.);z['quant_v2_original_risk_fraction']=pd.to_numeric(z.risk_fraction,errors='coerce').fillna(.0025)
            bad=((pd.to_numeric(z.quant_route_n,errors='coerce').fillna(0)>=MIN_ROUTE_VETO_N)&(pd.to_numeric(z.quant_strategy_n,errors='coerce').fillna(0)>=MIN_STRATEGY_VETO_N)&(pd.to_numeric(z.quant_pwin,errors='coerce').fillna(.5)<MAX_VETO_PWIN)&(pd.to_numeric(z.quant_conservative_ev_r,errors='coerce').fillna(0)<MAX_VETO_CONSERVATIVE_EV_R)&(pd.to_numeric(z.quant_tail_loss_r,errors='coerce').fillna(0)>=MIN_VETO_TAIL_LOSS_R))
            z['quant_v2_veto_pass']=~bad;z['quant_v2_veto_reason']=np.where(bad,'MATERIAL_ADVERSE_POSTERIOR','PASS');veto.append(z[bad].copy());q=z[~bad].copy();q=factor_support_audit(q,base);q['rank_score']=q.quant_v2_original_rank_score;q['risk_fraction']=q.quant_v2_original_risk_fraction;q['quant_risk_scale']=1.;kept.append(q)
        PR=pd.concat(kept,ignore_index=True);V=pd.concat(veto,ignore_index=True);PR.to_csv(a.outdir/'QV2_34_EXACT_BRIDGE_KEPT.csv.gz',index=False,compression='gzip');V.to_csv(a.outdir/'QV2_34_EXACT_BRIDGE_VETO.csv.gz',index=False,compression='gzip')
    else:
        PR=pd.read_csv(a.kept_file)
        for c in ('entry_time','decision_time','exit_time','eval_week'):PR[c]=pd.to_datetime(PR[c],utc=True,errors='coerce')
        V=pd.DataFrame()
    EV,DE,W=replay(PR,specs,fcache,raw_end,v2,r4);m=metrics(EV,W);EV.to_csv(a.outdir/'QV2_34_EXACT_BRIDGE_EVENTS.csv.gz',index=False,compression='gzip');DE.to_csv(a.outdir/'QV2_34_EXACT_BRIDGE_DECISIONS.csv.gz',index=False,compression='gzip');W.to_csv(a.outdir/'QV2_34_EXACT_BRIDGE_WEEKLY.csv',index=False);(a.outdir/'QV2_34_EXACT_BRIDGE_STATUS.json').write_text(json.dumps({'mode':a.mode,'quant_kept':int(len(PR)),'quant_vetoed':int(len(V)),'metrics':m},indent=2));print(json.dumps({'mode':a.mode,'quant_kept':len(PR),'quant_vetoed':len(V),'metrics':m},indent=2))
if __name__=='__main__':main()
