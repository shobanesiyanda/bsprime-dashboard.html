from __future__ import annotations
import numpy as np
import pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
from reliability_common import annotate

# R5.5 long-cycle causal repair 1:
# configuration-generation memory is bounded to two reliability windows.
# Reliability/evidence memory itself remains the frozen R5.4 56-day hierarchy.
PROPOSAL_MEMORY_DAYS = 112
VALIDATION_DAYS = 14


def _base_promote_bounded(cands, week_start):
    val0 = week_start - pd.Timedelta(days=VALIDATION_DAYS)
    old0 = val0 - pd.Timedelta(days=PROPOSAL_MEMORY_DAYS)
    old = cands[(cands.exit_time < val0) & (cands.exit_time >= old0)].copy()
    recent = cands[(cands.exit_time >= val0) & (cands.exit_time < week_start)].copy()
    target = cands[(cands.decision_time >= week_start) & (cands.decision_time < week_start + pd.Timedelta(days=7))].copy()
    if target.empty:
        return target, pd.DataFrame(), pd.DataFrame()

    prop = base.search_configs(old) if len(old) else pd.DataFrame()
    confirmed = []
    if len(prop):
        for _, cfg in prop.iterrows():
            z = base.cfg_apply(recent[(recent.asset == cfg.asset) & (recent.strategy == cfg.strategy)], cfg)
            n = len(z)
            mean = float(z.net_r.mean()) if n else np.nan
            med = float(z.net_r.median()) if n else np.nan
            win = float((z.net_r > 0).mean()) if n else 0.0
            # Exact frozen R5.4 validation contract; only proposal-memory source changes.
            if n >= 2 and mean > 0 and med > -.25 and win >= .40:
                q = cfg.to_dict()
                q.update(validation_n=n, validation_mean_r=mean, validation_median_r=med,
                         validation_win_rate=win, tier='A', proposal_memory_days=PROPOSAL_MEMORY_DAYS)
                confirmed.append(q)
    conf = pd.DataFrame(confirmed)

    parts = []
    if len(conf):
        for _, cfg in conf.iterrows():
            z = base.cfg_apply(target[(target.asset == cfg.asset) & (target.strategy == cfg.strategy)], cfg).copy()
            if len(z):
                z['promotion_tier'] = 'A'
                z['risk_fraction'] = .005
                z['proposal_memory_days'] = PROPOSAL_MEMORY_DAYS
                parts.append(z)
    promoted = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=target.columns.tolist()+['promotion_tier','risk_fraction','proposal_memory_days'])

    # Preserve frozen R5.4 Tier-B causal trailing-expectancy fallback unchanged.
    stats = base.trailing_stats(cands, week_start)
    statmap = {(r.asset, r.strategy): r for _, r in stats.iterrows()} if len(stats) else {}
    akeys = set(zip(promoted.asset, promoted.strategy)) if len(promoted) else set()
    b = []
    for _, r in target.iterrows():
        k = (r.asset, r.strategy)
        if k in akeys:
            continue
        st = statmap.get(k)
        if st is None or int(st.n) < 8:
            continue
        sd = 0.0 if pd.isna(st.sd) else float(st.sd)
        lower = float(st.mean_r) - .45 * sd / max(np.sqrt(float(st.n)), 1)
        if float(st.mean_r) > .10 and float(st.median_r) > -.15 and float(st.win) >= .45 and lower > -.05 and float(r.quality) >= .68 and float(r.cost_r) <= .35:
            x = r.copy()
            x['promotion_tier'] = 'B'
            x['risk_fraction'] = .0025
            x['proposal_memory_days'] = PROPOSAL_MEMORY_DAYS
            b.append(x)
    if b:
        promoted = pd.concat([promoted, pd.DataFrame(b)], ignore_index=True)
    if promoted.empty:
        return promoted, conf, stats

    # Preserve frozen base expected-utility ranking before hierarchical reliability overlay.
    er=[]; wr=[]; stab=[]; gb=[]
    for _, r in promoted.iterrows():
        st = statmap.get((r.asset, r.strategy)); mean=0.; win=.5; sd=1.; give=0.
        if st is not None:
            mean=float(st.mean_r); win=float(st.win); sd=0. if pd.isna(st.sd) else float(st.sd); give=float(st.mean_giveback)
        er.append(mean); wr.append(win); stab.append(mean/max(sd,.25)); gb.append(give)
    promoted['expected_r']=er; promoted['trailing_win_rate']=wr; promoted['stability']=stab; promoted['trailing_giveback']=gb
    promoted['rank_score']=(.42*promoted.quality+.28*np.tanh(promoted.expected_r)+.16*promoted.trailing_win_rate+.10*np.tanh(promoted.stability)-.22*promoted.cost_r-.04*np.minimum(promoted.trailing_giveback,3))
    promoted['tier_ord']=promoted.promotion_tier.map({'A':0,'B':1}).fillna(9)
    promoted=promoted.sort_values(['campaign_id','tier_ord','rank_score'],ascending=[True,True,False]).drop_duplicates('campaign_id').drop(columns='tier_ord')
    return promoted, conf, stats


def promote_for_week(cands, week_start):
    p, conf, stats = _base_promote_bounded(cands, week_start)
    if p.empty:
        return p, conf, stats
    z = annotate(p, cands, week_start)
    # Exact frozen R5.4 combined reliability rank + gate.
    z['rank_score'] = z.rank_score + .18*z.reliability_score
    bad=((z.reliability_strategy_n>=12)&(z.reliability_expected_r<-.04))|((z.reliability_n>=10)&(z.reliability_expected_r<-.08))
    bad |= ((z.reliability_hour_mean_r<-.06)&(z.reliability_expected_r<0)&(z.reliability_strategy_n>=12))
    z=z[~bad].copy()
    return z, conf, stats


replay_r5 = base.replay_r5
