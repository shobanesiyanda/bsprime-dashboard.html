#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import trailaris_r4_full_universe_loop as r4

BASELINE_END=254.4465598357797
BASELINE_FRESH=7.826996335713843
BASELINE_DD=-7.083721208160998
BASELINE_LOSSES=374

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def eval_variant(mod,data,cands,opps,fcache,specs,start=100.0):
    t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max();first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC')
    weeks=[];events=[];decisions=[];promoted_all=[];equity=start;wk=first
    while wk<=t1:
        pr,conf,stats=mod.promote_for_week(cands,wk)
        if len(pr):
            ev,de=mod.replay_r5(pr,specs,equity,fcache)
            if len(ev):equity=float(ev.equity.iloc[-1])
            events.append(ev.assign(eval_week=wk));decisions.append(de.assign(eval_week=wk));promoted_all.append(pr.assign(eval_week=wk))
        else:ev=pd.DataFrame();de=pd.DataFrame()
        w=r4.weekly(ev,start=(weeks[-1]['end_equity'] if weeks else start),feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),t1))
        if len(w):rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
        else:
            st=weeks[-1]['end_equity'] if weeks else start;rec=dict(week_start=wk,start_equity=st,end_equity=st,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False);equity=st
        weeks.append(rec);wk+=pd.Timedelta(days=7)
    W=pd.DataFrame(weeks);EV=pd.concat(events,ignore_index=True) if events else pd.DataFrame();DE=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame();PR=pd.concat(promoted_all,ignore_index=True) if promoted_all else pd.DataFrame()
    fresh0=pd.Timestamp('2026-08-13T00:00:00Z');aug10=pd.Timestamp('2026-08-10T00:00:00Z');week_pr,_,_=mod.promote_for_week(cands,aug10);fresh_pr=week_pr[(week_pr.decision_time>=fresh0)&(week_pr.decision_time<=t1)].copy()
    fresh_ev,fresh_de=mod.replay_r5(fresh_pr,specs,100.0,fcache) if len(fresh_pr) else (pd.DataFrame(),pd.DataFrame())
    fresh_w=r4.weekly(fresh_ev,start=100.0,feature_cache=fcache,start_ts=fresh0,end_ts=t1) if len(fresh_ev) else pd.DataFrame()
    wins=int((EV.net_pnl>0).sum()) if len(EV) else 0;losses=int((EV.net_pnl<0).sum()) if len(EV) else 0;flat=int((EV.net_pnl==0).sum()) if len(EV) else 0
    end=float(W.end_equity.iloc[-1]) if len(W) else start;fresh=float(fresh_w.weekly_return_pct.iloc[-1]) if len(fresh_w) else 0.;dd=float(W.max_drawdown_pct.min()) if len(W) else 0.
    result=dict(end_equity=end,compounded_return_pct=100*(end/start-1),weeks=len(W),mean_week_pct=float(W.weekly_return_pct.mean()),median_week_pct=float(W.weekly_return_pct.median()),weeks_ge5=int(W.ge5.sum()),weeks_ge10=int(W.ge10.sum()),max_weekly_dd_pct=dd,executed_trades=int(len(EV)),wins=wins,losses=losses,flat=flat,nonflat_win_rate=float(wins/(wins+losses)) if wins+losses else 0.,fresh_return_pct=fresh,fresh_max_dd_pct=float(fresh_w.max_drawdown_pct.iloc[-1]) if len(fresh_w) else 0.,assets_with_selected=int(DE.loc[DE.selected,'asset'].nunique()) if len(DE) and 'selected' in DE else 0)
    result['beats_or_matches_return_baseline']=end>=BASELINE_END-1e-9
    result['fresh_not_regressed']=fresh>=BASELINE_FRESH-1e-9
    result['losses_improved']=losses<BASELINE_LOSSES
    result['drawdown_improved']=dd>BASELINE_DD
    result['promotion_eligible']=bool(result['beats_or_matches_return_baseline'] and result['fresh_not_regressed'] and (result['losses_improved'] or result['drawdown_improved']))
    return result,W,EV,DE,PR

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--variant-dir',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    variants=[]
    for p in sorted(Path(a.variant_dir).glob('variant_*.py')):
        name=p.stem;mod=load_module(p,name);res,W,EV,DE,PR=eval_variant(mod,data,cands,opps,fcache,specs);res['variant']=name;variants.append(res)
        W.to_csv(out/f'{name}_weekly.csv',index=False)
    df=pd.DataFrame(variants).sort_values(['promotion_eligible','end_equity','losses'],ascending=[False,False,True]);df.to_csv(out/'R5_4_PRECISION_EXPERIMENTS.csv',index=False)
    best=df.iloc[0].to_dict() if len(df) else {}
    status={'baseline_end_equity':BASELINE_END,'baseline_fresh_return_pct':BASELINE_FRESH,'baseline_losses':BASELINE_LOSSES,'variants_tested':len(df),'promotion_eligible_variants':int(df.promotion_eligible.sum()) if len(df) else 0,'best_variant':best}
    (out/'R5_4_PRECISION_EXPERIMENT_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))

if __name__=='__main__':main()
