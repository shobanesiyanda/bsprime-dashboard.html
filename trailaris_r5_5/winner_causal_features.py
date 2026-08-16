from __future__ import annotations
import numpy as np,pandas as pd
from lane4_market_state_entry_repair import enrich_market_state

RELIABILITY_COLUMNS=[
 'reliability_score','reliability_expected_r','reliability_win','reliability_n',
 'reliability_strategy_n','reliability_hour_mean_r','reliability_strategy_mean_r',
 'reliability_strategy_win','reliability_asset_strategy_mean_r','reliability_asset_strategy_win'
]

def evaluation_week_start(series: pd.Series) -> pd.Series:
    t=pd.to_datetime(series,utc=True)
    return t.dt.floor('D')-pd.to_timedelta(t.dt.weekday,unit='D')

def causal_decision_features(cands: pd.DataFrame, feature_cache=None) -> pd.DataFrame:
    """Attach market-state + reliability features using only information available at each decision week.

    Reliability is rebuilt exactly as the production/replay selector sees it: only rows with exit_time
    strictly before the Monday evaluation timestamp enter history. Aug13-14 therefore receives the Aug10
    reliability state; no later outcome can affect its own qualification.
    """
    from reliability_common import annotate
    z=enrich_market_state(cands.copy(),feature_cache)
    if z.empty:return z
    z['decision_time']=pd.to_datetime(z.decision_time,utc=True)
    z['exit_time']=pd.to_datetime(z.exit_time,utc=True)
    z['_winner_eval_week']=evaluation_week_start(z.decision_time)
    for col in RELIABILITY_COLUMNS:
        if col not in z.columns:z[col]=np.nan
    for wk,idx in z.groupby('_winner_eval_week',sort=True).groups.items():
        sub=annotate(z.loc[idx].copy(),z,pd.Timestamp(wk))
        for col in RELIABILITY_COLUMNS:
            if col in sub.columns:z.loc[idx,col]=sub[col].to_numpy()
    return z.drop(columns=['_winner_eval_week'])

def combined_promote_clean_history(candidate_set: pd.DataFrame, history_cands: pd.DataFrame, week_start):
    """R5.4 combined promotion with candidate competition from candidate_set but reliability history from
    the unmodified economic history. Required for amplification copies so synthetic candidate duplication
    cannot manufacture its own historical reliability evidence.
    """
    import R5_BASE_CAUSAL_ENGINE as base
    from reliability_common import annotate
    p,conf,stats=base.promote_for_week(candidate_set,week_start)
    if p.empty:return p,conf,stats
    z=annotate(p,history_cands,week_start)
    z['rank_score']=z.rank_score+.18*z.reliability_score
    bad=((z.reliability_strategy_n>=12)&(z.reliability_expected_r<-.04))|((z.reliability_n>=10)&(z.reliability_expected_r<-.08))
    bad |= ((z.reliability_hour_mean_r<-.06)&(z.reliability_expected_r<0)&(z.reliability_strategy_n>=12))
    return z[~bad].copy(),conf,stats
