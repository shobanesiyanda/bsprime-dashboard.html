from __future__ import annotations
import os
import numpy as np
import pandas as pd
import quantitative_integrated_v1 as q1
import longcycle_integrated_negative_refinement as v2

# Quantitative V1 component ablation harness.
# IMPORTANT: this module is diagnostic only. Its results MUST NOT be used to tune
# or select Quantitative V2 parameters on the same 26-week target window.
MODE = os.environ.get('TRAILARIS_QUANT_ABLATION', 'AUDIT_ONLY').strip().upper()
ALLOWED = {'AUDIT_ONLY','NO_SIZING','NO_GATING','NO_FACTOR','NO_RANK'}
if MODE not in ALLOWED:
    raise RuntimeError(f'unknown TRAILARIS_QUANT_ABLATION={MODE}')


def _annotate(promoted: pd.DataFrame, cands: pd.DataFrame, week_start: pd.Timestamp) -> pd.DataFrame:
    if promoted.empty:
        return promoted.copy()
    models = q1._build_models(cands, week_start)
    rows=[]
    for _, r in promoted.iterrows():
        x=r.copy()
        for k,v in q1._posterior_for_row(r, models).items():
            x[k]=v
        rows.append(x)
    z=pd.DataFrame(rows)
    route_bad=(
        (z.quant_route_n >= q1.MIN_ROUTE_GATE_N)
        & (z.quant_conservative_ev_r < -0.06)
        & (z.quant_pwin < 0.44)
    )
    strat_bad=(
        (z.quant_strategy_n >= q1.MIN_STRATEGY_GATE_N)
        & (z.quant_conservative_ev_r < -0.09)
        & (z.quant_pwin < 0.43)
    )
    z['quant_gate_pass']=~(route_bad|strat_bad)
    z['quant_gate_reason']=np.where(route_bad,'NEGATIVE_ROUTE_POSTERIOR',np.where(strat_bad,'NEGATIVE_STRATEGY_POSTERIOR','PASS'))
    z['quant_original_rank_score']=pd.to_numeric(z['rank_score'],errors='coerce').fillna(0.0)
    z['quant_original_risk_fraction']=pd.to_numeric(z['risk_fraction'],errors='coerce').fillna(.0025)
    return z


def _apply_rank(z: pd.DataFrame) -> pd.DataFrame:
    if z.empty:
        return z
    z=z.copy()
    z['rank_score']=(
        pd.to_numeric(z['rank_score'],errors='coerce').fillna(0.0)
        + .20*pd.to_numeric(z['quant_score'],errors='coerce').fillna(0.0)
        + .04*np.tanh(pd.to_numeric(z['quant_ev_r'],errors='coerce').fillna(0.0))
        - .03*pd.to_numeric(z['cost_r'],errors='coerce').fillna(0.0)
    )
    return z


def _apply_sizing(z: pd.DataFrame) -> pd.DataFrame:
    if z.empty:
        return z
    z=z.copy()
    raw_k=np.clip(pd.to_numeric(z['quant_fractional_kelly'],errors='coerce').fillna(0.0),0.0,1.0)
    conf=np.clip(pd.to_numeric(z['quant_confidence'],errors='coerce').fillna(0.0),0.0,1.0)
    ev=pd.to_numeric(z['quant_conservative_ev_r'],errors='coerce').fillna(0.0)
    risk_scale=np.clip(0.55+0.65*raw_k*conf+0.10*(ev>.20).astype(float),0.45,1.15)
    base_risk=pd.to_numeric(z['risk_fraction'],errors='coerce').fillna(.0025)
    z['quant_base_risk_fraction']=base_risk
    z['quant_risk_scale']=risk_scale
    z['risk_fraction']=np.minimum(q1.MAX_PER_TRADE_RISK,base_risk*risk_scale)
    return z


def promote_for_week(cands, week_start):
    p, conf, stats=v2.promote_for_week(cands, week_start)
    if p.empty:
        return p, conf, stats
    z=_annotate(p,cands,pd.Timestamp(week_start))

    if MODE=='AUDIT_ONLY':
        z['quant_ablation_mode']=MODE
        z['quant_risk_scale']=1.0
        z['quant_base_risk_fraction']=z['quant_original_risk_fraction']
        return z, conf, stats

    if MODE!='NO_GATING':
        z=z[z.quant_gate_pass].copy()
    if z.empty:
        return z, conf, stats

    if MODE!='NO_RANK':
        z=_apply_rank(z)

    if MODE!='NO_SIZING':
        z=_apply_sizing(z)
    else:
        z['quant_base_risk_fraction']=z['quant_original_risk_fraction']
        z['quant_risk_scale']=1.0
        z['risk_fraction']=z['quant_original_risk_fraction']

    if MODE!='NO_FACTOR':
        z=q1._causal_factor_consensus(z)
    else:
        z['quant_factor_support']=0.0

    z['quant_ablation_mode']=MODE
    return z, conf, stats


def replay_r5(cands, specs, start=100.0, feature_cache=None):
    return v2.replay_r5(cands, specs, start, feature_cache)
