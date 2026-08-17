from __future__ import annotations
import math
import numpy as np
import pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import quantitative_integrated_v1 as q1
import longcycle_integrated_negative_refinement as v2

# Trailaris Quantitative Integrated V2
# -----------------------------------
# Ex-ante architecture frozen before the replay. This candidate deliberately
# preserves the proven R5.5 V2 ranking, sizing, execution and management stack.
# Mathematics may only veto a candidate when causal historical evidence is
# materially adverse. Quantitative scores and factor consensus are audit-only.
# No universal de-risking, Kelly rescaling, positive risk boost, or broad rank
# overlay is permitted in this version.

LOOKBACK_DAYS = q1.LOOKBACK_DAYS
MIN_ROUTE_VETO_N = 24
MIN_STRATEGY_VETO_N = 48
MAX_VETO_PWIN = 0.40
MAX_VETO_CONSERVATIVE_EV_R = -0.12
MIN_VETO_TAIL_LOSS_R = 0.75
FACTOR_HALF_LIFE_HOURS = q1.FACTOR_HALF_LIFE_HOURS
EPS = 1e-12


def _annotate(promoted: pd.DataFrame, cands: pd.DataFrame, week_start: pd.Timestamp) -> pd.DataFrame:
    if promoted.empty:
        return promoted.copy()
    models=q1._build_models(cands,week_start)
    rows=[]
    for _,r in promoted.iterrows():
        x=r.copy()
        for k,v in q1._posterior_for_row(r,models).items():
            x[k]=v
        rows.append(x)
    return pd.DataFrame(rows)


def _factor_support_audit(z: pd.DataFrame) -> pd.DataFrame:
    if z.empty:
        return z
    out=z.copy().sort_values(['decision_time','campaign_id']).reset_index(drop=True)
    state=np.zeros(len(base.r4.FACTORS),dtype=float)
    last_t=None
    support=np.zeros(len(out),dtype=float)
    for t,idx in out.groupby('decision_time',sort=True).groups.items():
        t=pd.Timestamp(t)
        if last_t is not None:
            dt=max(0.0,(t-last_t).total_seconds()/3600.0)
            state*=math.exp(-math.log(2.0)*dt/FACTOR_HALF_LIFE_HOURS)
        norm_state=float(np.linalg.norm(state))
        pending=[]
        for i in list(idx):
            r=out.loc[i]
            fv=np.asarray(base.r4.factor_vec(str(r.asset),int(r.direction)),dtype=float)
            nf=float(np.linalg.norm(fv))
            align=float(np.dot(fv,state)/(nf*norm_state)) if nf>EPS and norm_state>EPS else 0.0
            support[i]=np.clip(align,-1.0,1.0)
            pending.append((fv,float(r.get('quality',0.5)),float(r.get('quant_confidence',0.0))))
        for fv,q,conf in pending:
            state+=fv*max(0.0,q)*(0.5+0.5*conf)
        last_t=t
    out['quant_factor_support']=support
    return out


def promote_for_week(cands, week_start):
    # Exact frozen R5.5 V2 candidate set is the starting point.
    p,conf,stats=v2.promote_for_week(cands,week_start)
    if p.empty:
        return p,conf,stats

    z=_annotate(p,cands,pd.Timestamp(week_start))
    z['quant_v2_original_rank_score']=pd.to_numeric(z['rank_score'],errors='coerce').fillna(0.0)
    z['quant_v2_original_risk_fraction']=pd.to_numeric(z['risk_fraction'],errors='coerce').fillna(.0025)

    # Strict conjunctive veto. Sparse or ambiguous evidence is never allowed to
    # suppress a V2 opportunity. All four adverse conditions must hold.
    bad=(
        (pd.to_numeric(z['quant_route_n'],errors='coerce').fillna(0)>=MIN_ROUTE_VETO_N)
        & (pd.to_numeric(z['quant_strategy_n'],errors='coerce').fillna(0)>=MIN_STRATEGY_VETO_N)
        & (pd.to_numeric(z['quant_pwin'],errors='coerce').fillna(.5)<MAX_VETO_PWIN)
        & (pd.to_numeric(z['quant_conservative_ev_r'],errors='coerce').fillna(0)<MAX_VETO_CONSERVATIVE_EV_R)
        & (pd.to_numeric(z['quant_tail_loss_r'],errors='coerce').fillna(0)>=MIN_VETO_TAIL_LOSS_R)
    )
    z['quant_v2_veto_pass']=~bad
    z['quant_v2_veto_reason']=np.where(bad,'MATERIAL_ADVERSE_POSTERIOR','PASS')
    z=z.loc[~bad].copy()
    if z.empty:
        return z,conf,stats

    # Factor consensus is recorded for attribution only. Rank and risk are
    # restored exactly to V2 values and therefore cannot throttle winners.
    z=_factor_support_audit(z)
    z['rank_score']=z['quant_v2_original_rank_score']
    z['risk_fraction']=z['quant_v2_original_risk_fraction']
    z['quant_risk_scale']=1.0
    z['quant_base_risk_fraction']=z['quant_v2_original_risk_fraction']
    z['quant_v2_architecture']='STRICT_NEGATIVE_VETO_ONLY__V2_RANK_AND_RISK_PRESERVED'
    return z,conf,stats


def replay_r5(cands, specs, start=100.0, feature_cache=None):
    # Preserve exact V2 management and causal execution.
    return v2.replay_r5(cands,specs,start,feature_cache)
