#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

BASELINE_ID='R5_5_INTEGRATED_NEGATIVE_REFINEMENT_V2_26W_LOCKED_20260816'
EXPECTED={
    'end_equity':1366.8938578865148,
    'compounded_return_pct':1266.8938578865148,
    'positive_weeks':26,
    'executed_trades':2136,
    'wins':1221,
    'losses':478,
    'flats':437,
    'profit_factor':2.9195127192871873,
    'max_weekly_drawdown_pct':-3.7440606836682933,
    'max_event_equity_drawdown_pct':-2.7023988320913572,
}
MIN_ROUTE_VETO_N=24
MIN_STRATEGY_VETO_N=48
MAX_VETO_PWIN=0.40
MAX_VETO_CONSERVATIVE_EV_R=-0.12
MIN_VETO_TAIL_LOSS_R=0.75
FACTOR_HALF_LIFE_HOURS=12.0
EPS=1e-12


def _build_component(job):
    import trailaris_r4_full_universe_loop as r4
    return r4.build_asset_components(job)


def load_feature_cache(rawdir: Path, routes: list[str]):
    jobs=[(asset,str(rawdir/f'{asset}_M1_normalized.csv')) for asset in routes]
    cache={}; workers=min(8,max(2,os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(_build_component,j):j[0] for j in jobs}
        for fut in as_completed(futs):
            asset,_cands,xf,_opps=fut.result()
            if xf is None or len(xf)==0: raise RuntimeError(f'empty exact R4 feature frame: {asset}')
            xf=xf.copy(); xf['timestamp']=pd.to_datetime(xf['timestamp'],utc=True,errors='coerce')
            xf=xf.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            if {'timestamp','high','low','close'}-set(xf.columns): raise RuntimeError(f'feature frame incomplete: {asset}')
            cache[str(asset)]=xf
    if set(cache)!=set(routes): raise RuntimeError(f'exact R4 feature cache gate failed: {len(cache)}/34')
    raw_end=max(pd.Timestamp(x.timestamp.iloc[-1]) for x in cache.values())
    return cache,raw_end


def rolling(W,n=11):
    r=pd.to_numeric(W.weekly_return_pct,errors='coerce').fillna(0).to_numpy(float); rows=[]
    for i in range(len(W)-n+1):
        z=r[i:i+n]; ret=(np.prod(1+z/100)-1)*100
        rows.append({'window':i+1,'start_week':str(W.week_start.iloc[i]),'end_week':str(W.week_start.iloc[i+n-1]),
                     'compounded_return_pct':float(ret),'mean_week_pct':float(np.mean(z)),'median_week_pct':float(np.median(z)),
                     'positive_weeks':int((z>0).sum()),'weeks_ge5':int((z>=5).sum()),'weeks_ge10':int((z>=10).sum()),
                     'max_weekly_dd_pct':float(pd.to_numeric(W.max_drawdown_pct.iloc[i:i+n],errors='coerce').min())})
    return pd.DataFrame(rows)


def metrics(EV,DE,W,R):
    wins=int((EV.net_pnl>0).sum()); losses=int((EV.net_pnl<0).sum()); flats=int((EV.net_pnl==0).sum())
    gw=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum()); gl=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum())
    eq=pd.concat([pd.Series([100.]),pd.to_numeric(EV.sort_values('timestamp').equity,errors='coerce').dropna()],ignore_index=True)
    dd=(eq/eq.cummax()-1)*100
    return {'end_equity':float(W.end_equity.iloc[-1]),'compounded_return_pct':float((W.end_equity.iloc[-1]/100-1)*100),
            'evaluated_weeks':int(len(W)),'positive_weeks':int((W.weekly_return_pct>0).sum()),'negative_weeks':int((W.weekly_return_pct<0).sum()),
            'mean_week_pct':float(W.weekly_return_pct.mean()),'median_week_pct':float(W.weekly_return_pct.median()),
            'weeks_ge5':int((W.weekly_return_pct>=5).sum()),'weeks_ge10':int((W.weekly_return_pct>=10).sum()),
            'executed_trades':int(len(EV)),'wins':wins,'losses':losses,'flats':flats,
            'loss_rate':float(losses/len(EV)) if len(EV) else 0.,'flat_rate':float(flats/len(EV)) if len(EV) else 0.,
            'nonflat_win_rate':float(wins/(wins+losses)) if wins+losses else 0.,'profit_factor':float(gw/gl) if gl else 999.,
            'max_weekly_drawdown_pct':float(pd.to_numeric(W.max_drawdown_pct,errors='coerce').min()),
            'max_event_equity_drawdown_pct':float(dd.min()),
            'assets_with_selected':int(DE.loc[DE.selected,'asset'].nunique()) if len(DE) and DE.selected.any() else 0,
            'rolling_11w_windows':int(len(R)),'rolling_11w_min_return_pct':float(R.compounded_return_pct.min()),
            'rolling_11w_median_return_pct':float(R.compounded_return_pct.median()),
            'rolling_11w_upper_quartile_return_pct':float(R.compounded_return_pct.quantile(.75)),
            'rolling_11w_mean_return_pct':float(R.compounded_return_pct.mean()),'rolling_11w_max_return_pct':float(R.compounded_return_pct.max())}


def replay_estate(PR,specs,fcache,raw_end,mod,r4):
    equity=100.; Wrows=[]; events=[]; decisions=[]
    weeks=sorted(pd.Timestamp(x) for x in PR.eval_week.unique())
    for wk in weeks:
        src=PR[PR.eval_week==wk].drop(columns=['eval_week'],errors='ignore').copy()
        ev,de=mod.replay_r5(src,specs,equity,fcache) if len(src) else (pd.DataFrame(),pd.DataFrame())
        if len(ev): equity=float(ev.equity.iloc[-1])
        events.append(ev.assign(eval_week=wk)); decisions.append(de.assign(eval_week=wk))
        prev=Wrows[-1]['end_equity'] if Wrows else 100.
        w=r4.weekly(ev,start=prev,feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),raw_end))
        if len(w): rec=w.iloc[0].to_dict(); equity=float(rec['end_equity'])
        else: rec=dict(week_start=wk,start_equity=prev,end_equity=prev,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False); equity=prev
        Wrows.append(rec)
    W=pd.DataFrame(Wrows); EV=pd.concat(events,ignore_index=True); DE=pd.concat(decisions,ignore_index=True); R=rolling(W)
    return metrics(EV,DE,W,R),W,EV,DE,R


def assert_identity(m):
    exact_int=['positive_weeks','executed_trades','wins','losses','flats']
    for k in exact_int:
        if int(m[k])!=int(EXPECTED[k]): raise RuntimeError(f'CANONICAL IDENTITY FAIL {k}: {m[k]} != {EXPECTED[k]}')
    tol={'end_equity':1e-8,'compounded_return_pct':1e-8,'profit_factor':1e-10,'max_weekly_drawdown_pct':1e-10,'max_event_equity_drawdown_pct':1e-10}
    for k,t in tol.items():
        if abs(float(m[k])-float(EXPECTED[k]))>t: raise RuntimeError(f'CANONICAL IDENTITY FAIL {k}: {m[k]} != {EXPECTED[k]}')


def factor_support_audit(z,base):
    if z.empty: return z
    out=z.copy().sort_values(['decision_time','campaign_id']).reset_index(drop=True)
    state=np.zeros(len(base.r4.FACTORS),dtype=float); last_t=None; support=np.zeros(len(out),dtype=float)
    for t,idx in out.groupby('decision_time',sort=True).groups.items():
        t=pd.Timestamp(t)
        if last_t is not None:
            dt=max(0.,(t-last_t).total_seconds()/3600.); state*=math.exp(-math.log(2.)*dt/FACTOR_HALF_LIFE_HOURS)
        ns=float(np.linalg.norm(state)); pending=[]
        for i in list(idx):
            r=out.loc[i]; fv=np.asarray(base.r4.factor_vec(str(r.asset),int(r.direction)),dtype=float); nf=float(np.linalg.norm(fv))
            support[i]=np.clip(float(np.dot(fv,state)/(nf*ns)) if nf>EPS and ns>EPS else 0.,-1.,1.)
            pending.append((fv,float(r.get('quality',.5)),float(r.get('quant_confidence',0.))))
        for fv,q,conf in pending: state+=fv*max(0.,q)*(.5+.5*conf)
        last_t=t
    out['quant_factor_support']=support
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rawdir',required=True); ap.add_argument('--specs',required=True); ap.add_argument('--factory-audit',required=True); ap.add_argument('--canonical-dir',required=True); ap.add_argument('--outdir',required=True); a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); canon_dir=Path(a.canonical_dir)
    sys.path[:0]=['trailaris_r4','trailaris_r5_4','trailaris_r5_4/reliability_variants','trailaris_r5_5']
    import trailaris_r4_full_universe_loop as r4
    import R5_BASE_CAUSAL_ENGINE as base
    import longcycle_integrated_negative_refinement as v2
    import quantitative_integrated_v1 as q1

    fa=pd.read_csv(a.factory_audit)
    if len(fa)!=510 or int(fa.strategy_family.nunique())!=15: raise RuntimeError('factory 510x15 lock failed')
    cstatus=json.loads((canon_dir/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_STATUS.json').read_text())
    if cstatus['source_promotion_estate']!='EXACT_R5_5_MEMORY112_PROMOTED_3417' or cstatus['integrated_counters']['promoted_before_gate']!=3417 or cstatus['integrated_counters']['entry_gate_rejected']!=228 or cstatus['integrated_counters']['promoted_after_gate']!=3189: raise RuntimeError('canonical estate counters invalid')
    if abs(cstatus['candidate']['end_equity']-EXPECTED['end_equity'])>1e-9: raise RuntimeError('canonical artifact baseline mismatch')

    PR0=pd.read_csv(canon_dir/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_PROMOTED.csv')
    RJ0=pd.read_csv(canon_dir/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_GATE_REJECTED.csv')
    for d in (PR0,RJ0):
        for c in ('entry_time','decision_time','exit_time','eval_week'): d[c]=pd.to_datetime(d[c],utc=True,errors='coerce')
    if len(PR0)!=3189 or len(RJ0)!=228 or PR0.eval_week.nunique()!=26: raise RuntimeError('canonical post-gate estate files invalid')

    routes=list(r4.ROUTES); specs=r4.load_specs(Path(a.specs),'research-proxy')
    fcache,raw_end=load_feature_cache(Path(a.rawdir),routes)

    identity,W0,EV0,DE0,R0=replay_estate(PR0,specs,fcache,raw_end,v2,r4)
    assert_identity(identity)
    print('CANONICAL_IDENTITY_GATE PASS',json.dumps(identity,sort_keys=True))

    # Build the identical full causal candidate universe only for historical quant evidence.
    data,_v,cands,_opps,_qcache=base.build_universe(Path(a.rawdir))
    if len(data)!=34: raise RuntimeError(f'quant evidence universe gate failed {len(data)}/34')

    kept_all=[]; veto_all=[]
    for wk in sorted(pd.Timestamp(x) for x in PR0.eval_week.unique()):
        src=PR0[PR0.eval_week==wk].copy(); models=q1._build_models(cands,wk); rows=[]
        for _,r in src.iterrows():
            x=r.copy()
            for k,v in q1._posterior_for_row(r,models).items(): x[k]=v
            rows.append(x)
        z=pd.DataFrame(rows)
        z['quant_v2_original_rank_score']=pd.to_numeric(z['rank_score'],errors='coerce').fillna(0.)
        z['quant_v2_original_risk_fraction']=pd.to_numeric(z['risk_fraction'],errors='coerce').fillna(.0025)
        bad=((pd.to_numeric(z.quant_route_n,errors='coerce').fillna(0)>=MIN_ROUTE_VETO_N)&
             (pd.to_numeric(z.quant_strategy_n,errors='coerce').fillna(0)>=MIN_STRATEGY_VETO_N)&
             (pd.to_numeric(z.quant_pwin,errors='coerce').fillna(.5)<MAX_VETO_PWIN)&
             (pd.to_numeric(z.quant_conservative_ev_r,errors='coerce').fillna(0)<MAX_VETO_CONSERVATIVE_EV_R)&
             (pd.to_numeric(z.quant_tail_loss_r,errors='coerce').fillna(0)>=MIN_VETO_TAIL_LOSS_R))
        z['quant_v2_veto_pass']=~bad; z['quant_v2_veto_reason']=np.where(bad,'MATERIAL_ADVERSE_POSTERIOR','PASS')
        veto=z.loc[bad].copy(); kept=z.loc[~bad].copy()
        if len(kept):
            kept=factor_support_audit(kept,base); kept['rank_score']=kept['quant_v2_original_rank_score']; kept['risk_fraction']=kept['quant_v2_original_risk_fraction']; kept['quant_risk_scale']=1.0; kept['quant_v2_architecture']='STRICT_NEGATIVE_VETO_ONLY__CANONICAL_V2_ESTATE'
            kept_all.append(kept)
        if len(veto): veto_all.append(veto)
    PR=pd.concat(kept_all,ignore_index=True) if kept_all else PR0.iloc[0:0].copy(); VETO=pd.concat(veto_all,ignore_index=True) if veto_all else PR0.iloc[0:0].copy()
    cand,W,EV,DE,R=replay_estate(PR,specs,fcache,raw_end,v2,r4)

    gates={
      'positive_weeks_26':cand['positive_weeks']==26,
      'end_equity_ge_v2':cand['end_equity']>=EXPECTED['end_equity'],
      'profit_factor_ge_v2':cand['profit_factor']>=EXPECTED['profit_factor'],
      'max_weekly_drawdown_no_worse':cand['max_weekly_drawdown_pct']>=EXPECTED['max_weekly_drawdown_pct'],
      'max_event_drawdown_no_worse':cand['max_event_equity_drawdown_pct']>=EXPECTED['max_event_equity_drawdown_pct'],
    }
    delta={k:float(cand[k]-identity[k]) for k in cand if k in identity and isinstance(cand[k],(int,float,np.integer,np.floating)) and isinstance(identity[k],(int,float,np.integer,np.floating))}
    audit={
      'state':'R5_5_QUANTITATIVE_V2_EXACT_CANONICAL_ESTATE_REPLAY_COMPLETE',
      'baseline_id':BASELINE_ID,'canonical_artifact_run_id':31973576216,'canonical_artifact_id':9270611209,
      'canonical_identity_gate':'PASS','canonical_identity':identity,'candidate':cand,'delta_candidate_minus_canonical_v2':delta,
      'canonical_estate':{'memory112_before_v2_gate':3417,'v2_gate_rejected':228,'canonical_v2_post_gate':3189,'quant_v2_vetoed':int(len(VETO)),'quant_v2_kept':int(len(PR))},
      'quant_summary':{
        'mean_posterior_win_probability':float(pd.to_numeric(PR.quant_pwin,errors='coerce').mean()) if len(PR) else None,
        'mean_regularised_expected_r':float(pd.to_numeric(PR.quant_ev_r,errors='coerce').mean()) if len(PR) else None,
        'mean_conservative_expected_r':float(pd.to_numeric(PR.quant_conservative_ev_r,errors='coerce').mean()) if len(PR) else None,
        'mean_quant_confidence':float(pd.to_numeric(PR.quant_confidence,errors='coerce').mean()) if len(PR) else None,
        'mean_factor_support':float(pd.to_numeric(PR.quant_factor_support,errors='coerce').mean()) if len(PR) and 'quant_factor_support' in PR else None,
      },
      'promotion_gates':gates,'all_numeric_promotion_gates_pass':bool(all(gates.values())),'automatic_promotion':False,
      'causality':'QUANT MODEL USES ONLY CANDIDATES EXITED BEFORE EVALUATED WEEK; TARGET-WEEK OUTCOMES ARE NOT DECISION INPUTS; CANONICAL V2 RANK/RISK/EXECUTION/MANAGEMENT PRESERVED.',
      'evidence_class':'FULL_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE'
    }
    (out/'R5_5_QUANTITATIVE_V2_EXACT_ESTATE_STATUS.json').write_text(json.dumps(audit,indent=2,default=str))
    W0.to_csv(out/'R5_5_CANONICAL_IDENTITY_WEEKLY.csv',index=False); W.to_csv(out/'R5_5_QUANTITATIVE_V2_EXACT_ESTATE_WEEKLY.csv',index=False); R.to_csv(out/'R5_5_QUANTITATIVE_V2_EXACT_ESTATE_ROLLING_11W.csv',index=False)
    PR.to_csv(out/'R5_5_QUANTITATIVE_V2_EXACT_ESTATE_PROMOTED.csv',index=False); VETO.to_csv(out/'R5_5_QUANTITATIVE_V2_EXACT_ESTATE_VETOED.csv',index=False); EV.to_csv(out/'R5_5_QUANTITATIVE_V2_EXACT_ESTATE_EVENTS.csv',index=False); DE.to_csv(out/'R5_5_QUANTITATIVE_V2_EXACT_ESTATE_DECISIONS.csv',index=False)
    print(json.dumps(audit,indent=2,default=str))

if __name__=='__main__': main()
