from __future__ import annotations
import math
import numpy as np
import pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import longcycle_integrated_negative_refinement as v2

# Trailaris Quantitative Intelligence & Mathematical Decision Layer V1
# ------------------------------------------------------------------
# This is a causal overlay on the locked R5.5 Integrated Negative Refinement V2.
# It never reads the current/future candidate outcome when deciding whether to
# promote, rank, or size that candidate. Every statistical input is built only
# from campaigns whose exit_time is strictly before the evaluated week.
#
# Quantitative components in this first independent 26-week replay:
#   1. hierarchical Bayesian win-probability shrinkage;
#   2. regularised conditional expected-R estimation;
#   3. uncertainty/downside-tail penalty;
#   4. bounded fractional-Kelly risk allocation inside existing hard risk caps;
#   5. causal cross-currency/cross-factor consensus ranking;
#   6. V2 execution/management retained so the incremental effect is identifiable.
#
# No target-week outcomes, current-trade MFE/MAE, future bars, or final result are
# used in the decision layer.

LOOKBACK_DAYS = 112
PRIOR_STRENGTH_GLOBAL = 18.0
SHRINK_FAMILY = 16.0
SHRINK_STRATEGY = 12.0
SHRINK_ROUTE = 8.0
SHRINK_HOUR = 16.0
MIN_ROUTE_GATE_N = 12
MIN_STRATEGY_GATE_N = 24
MAX_PER_TRADE_RISK = 0.005
FACTOR_HALF_LIFE_HOURS = 12.0
EPS = 1e-12


def _clean(g: pd.DataFrame) -> np.ndarray:
    if g is None or len(g) == 0:
        return np.asarray([], dtype=float)
    return pd.to_numeric(g['net_r'], errors='coerce').dropna().to_numpy(float)


def _raw_stats(g: pd.DataFrame) -> dict:
    r = _clean(g)
    n = int(len(r))
    if n == 0:
        return dict(n=0, mean=0.0, sd=1.0, pwin=0.5, avg_win=0.75,
                    avg_loss=0.75, tail_loss=0.75, flat=0.0)
    wins = r[r > EPS]
    losses = -r[r < -EPS]
    nonflat = len(wins) + len(losses)
    pwin = float(len(wins) / nonflat) if nonflat else 0.5
    avg_win = float(wins.mean()) if len(wins) else 0.75
    avg_loss = float(losses.mean()) if len(losses) else 0.75
    neg = np.sort(losses)
    if len(neg):
        k = max(1, int(math.ceil(0.20 * len(neg))))
        tail_loss = float(neg[-k:].mean())
    else:
        tail_loss = avg_loss
    return dict(
        n=n,
        mean=float(r.mean()),
        sd=float(r.std(ddof=1)) if n > 1 else 1.0,
        pwin=pwin,
        avg_win=max(avg_win, EPS),
        avg_loss=max(avg_loss, EPS),
        tail_loss=max(tail_loss, EPS),
        flat=float((np.abs(r) <= EPS).mean()),
    )


def _blend(child: dict, prior: dict, k: float) -> dict:
    n = float(child['n'])
    w = n / (n + k) if n + k > 0 else 0.0
    out = {'n': int(n)}
    for key in ['mean', 'pwin', 'avg_win', 'avg_loss', 'tail_loss', 'flat']:
        out[key] = float(w * child[key] + (1.0 - w) * prior[key])
    # pooled uncertainty approximation; deliberately conservative for sparse cells
    out['sd'] = float(max(EPS, math.sqrt(w * child['sd']**2 + (1.0 - w) * prior['sd']**2)))
    return out


def _global_prior(h: pd.DataFrame) -> dict:
    raw = _raw_stats(h)
    # Symmetric weak prior on non-flat win probability and zero-R expectancy.
    n = float(raw['n'])
    p = (n * raw['pwin'] + PRIOR_STRENGTH_GLOBAL * 0.5) / (n + PRIOR_STRENGTH_GLOBAL)
    mean = n * raw['mean'] / (n + PRIOR_STRENGTH_GLOBAL)
    out = raw.copy()
    out['pwin'] = float(p)
    out['mean'] = float(mean)
    return out


def _build_models(cands: pd.DataFrame, week_start: pd.Timestamp) -> dict:
    h = cands[(cands.exit_time < week_start) &
              (cands.exit_time >= week_start - pd.Timedelta(days=LOOKBACK_DAYS))].copy()
    if len(h) == 0:
        gp = _raw_stats(h)
        return dict(global_=gp, family={}, strategy={}, route={}, hour={})
    h['decision_time'] = pd.to_datetime(h['decision_time'], utc=True)
    h['hour_band'] = (h['decision_time'].dt.hour // 4).astype(int)
    gp = _global_prior(h)
    fam = {}
    for k, g in h.groupby('strategy_family', sort=False):
        fam[k] = _blend(_raw_stats(g), gp, SHRINK_FAMILY)
    strat = {}
    for k, g in h.groupby('strategy', sort=False):
        prior = fam.get(str(g.strategy_family.iloc[0]), gp)
        strat[k] = _blend(_raw_stats(g), prior, SHRINK_STRATEGY)
    route = {}
    for k, g in h.groupby(['asset', 'strategy'], sort=False):
        route[k] = _blend(_raw_stats(g), strat.get(k[1], gp), SHRINK_ROUTE)
    hour = {}
    for k, g in h.groupby('hour_band', sort=False):
        hour[int(k)] = _blend(_raw_stats(g), gp, SHRINK_HOUR)
    return dict(global_=gp, family=fam, strategy=strat, route=route, hour=hour)


def _posterior_for_row(r: pd.Series, models: dict) -> dict:
    gp = models['global_']
    fam = models['family'].get(r.strategy_family, gp)
    ss = models['strategy'].get(r.strategy, fam)
    rr = models['route'].get((r.asset, r.strategy), ss)
    hb = int(pd.Timestamp(r.decision_time).hour // 4)
    hh = models['hour'].get(hb, gp)

    # Route/strategy evidence has highest authority, then family/session context.
    p = .48 * rr['pwin'] + .27 * ss['pwin'] + .15 * fam['pwin'] + .10 * hh['pwin']
    mu = .50 * rr['mean'] + .25 * ss['mean'] + .15 * fam['mean'] + .10 * hh['mean']
    aw = .50 * rr['avg_win'] + .25 * ss['avg_win'] + .15 * fam['avg_win'] + .10 * hh['avg_win']
    al = .50 * rr['avg_loss'] + .25 * ss['avg_loss'] + .15 * fam['avg_loss'] + .10 * hh['avg_loss']
    tail = .50 * rr['tail_loss'] + .25 * ss['tail_loss'] + .15 * fam['tail_loss'] + .10 * hh['tail_loss']
    sd = max(EPS, .50 * rr['sd'] + .25 * ss['sd'] + .15 * fam['sd'] + .10 * hh['sd'])
    n_eff = max(1.0, .55 * rr['n'] + .30 * ss['n'] + .15 * fam['n'])

    # Normal-approximate lower bound on expected R plus explicit tail penalty.
    # This is ranking/gating evidence, not a claim of exact confidence coverage.
    se = sd / math.sqrt(n_eff + PRIOR_STRENGTH_GLOBAL)
    conservative_ev = float(mu - 0.55 * se - 0.08 * tail)

    # General binary-outcome Kelly fraction in loss-unit terms. We use only a
    # small bounded fraction of it and never relax the portfolio's 1% open-risk cap.
    b = max(EPS, aw / max(al, EPS))
    kelly = max(0.0, min(1.0, (b * p - (1.0 - p)) / b))
    confidence = 1.0 - math.exp(-n_eff / 24.0)
    score = (
        .46 * math.tanh(conservative_ev / .45)
        + .24 * np.clip((p - .50) * 2.0, -1.0, 1.0)
        + .15 * confidence
        - .10 * math.tanh(tail / 2.0)
        - .05 * float(r.get('cost_r', 0.0))
    )
    return dict(
        quant_pwin=float(p), quant_ev_r=float(mu), quant_conservative_ev_r=conservative_ev,
        quant_avg_win_r=float(aw), quant_avg_loss_r=float(al), quant_tail_loss_r=float(tail),
        quant_effective_n=float(n_eff), quant_confidence=float(confidence),
        quant_fractional_kelly=float(kelly), quant_score=float(score),
        quant_route_n=int(rr['n']), quant_strategy_n=int(ss['n'])
    )


def _causal_factor_consensus(z: pd.DataFrame) -> pd.DataFrame:
    if z.empty:
        return z
    out = z.copy().sort_values(['decision_time', 'campaign_id']).reset_index(drop=True)
    state = np.zeros(len(base.r4.FACTORS), dtype=float)
    last_t = None
    support = np.zeros(len(out), dtype=float)
    for t, idx in out.groupby('decision_time', sort=True).groups.items():
        t = pd.Timestamp(t)
        if last_t is not None:
            dt = max(0.0, (t - last_t).total_seconds() / 3600.0)
            state *= math.exp(-math.log(2.0) * dt / FACTOR_HALF_LIFE_HOURS)
        norm_state = float(np.linalg.norm(state))
        pending = []
        for i in list(idx):
            r = out.loc[i]
            fv = np.asarray(base.r4.factor_vec(str(r.asset), int(r.direction)), dtype=float)
            nf = float(np.linalg.norm(fv))
            align = float(np.dot(fv, state) / (nf * norm_state)) if nf > EPS and norm_state > EPS else 0.0
            support[i] = np.clip(align, -1.0, 1.0)
            pending.append((fv, float(r.get('quality', 0.5)), float(r.get('quant_confidence', 0.0))))
        # Same-timestamp candidates cannot influence one another; update only after scoring all.
        for fv, q, conf in pending:
            state += fv * max(0.0, q) * (0.5 + 0.5 * conf)
        last_t = t
    out['quant_factor_support'] = support
    out['rank_score'] = pd.to_numeric(out['rank_score'], errors='coerce').fillna(0.0) + .07 * out['quant_factor_support']
    return out


def _quantify(promoted: pd.DataFrame, cands: pd.DataFrame, week_start: pd.Timestamp) -> pd.DataFrame:
    if promoted.empty:
        return promoted.copy()
    models = _build_models(cands, week_start)
    rows = []
    for _, r in promoted.iterrows():
        x = r.copy()
        for k, v in _posterior_for_row(r, models).items():
            x[k] = v
        rows.append(x)
    z = pd.DataFrame(rows)

    # Suppression requires material causal evidence; sparse cells are ranked, not vetoed.
    route_bad = (
        (z.quant_route_n >= MIN_ROUTE_GATE_N)
        & (z.quant_conservative_ev_r < -0.06)
        & (z.quant_pwin < 0.44)
    )
    strat_bad = (
        (z.quant_strategy_n >= MIN_STRATEGY_GATE_N)
        & (z.quant_conservative_ev_r < -0.09)
        & (z.quant_pwin < 0.43)
    )
    z['quant_gate_pass'] = ~(route_bad | strat_bad)
    z['quant_gate_reason'] = np.where(route_bad, 'NEGATIVE_ROUTE_POSTERIOR',
                              np.where(strat_bad, 'NEGATIVE_STRATEGY_POSTERIOR', 'PASS'))
    z = z[z.quant_gate_pass].copy()
    if z.empty:
        return z

    # Quantitative rank overlay. Cost is already represented in historical net-R;
    # the explicit cost term only penalises unusually expensive current candidates.
    z['rank_score'] = (
        pd.to_numeric(z['rank_score'], errors='coerce').fillna(0.0)
        + .20 * z['quant_score']
        + .04 * np.tanh(z['quant_ev_r'])
        - .03 * pd.to_numeric(z['cost_r'], errors='coerce').fillna(0.0)
    )

    # Bounded fractional-Kelly sizing. Strong Tier-B evidence can graduate toward
    # Tier-A risk, while uncertain/weak opportunities surrender capacity. No trade
    # exceeds the frozen 0.5% per-trade risk fraction and the base replay still
    # enforces <=1% aggregate open risk.
    raw_k = np.clip(pd.to_numeric(z['quant_fractional_kelly'], errors='coerce').fillna(0.0), 0.0, 1.0)
    conf = np.clip(pd.to_numeric(z['quant_confidence'], errors='coerce').fillna(0.0), 0.0, 1.0)
    ev = pd.to_numeric(z['quant_conservative_ev_r'], errors='coerce').fillna(0.0)
    risk_scale = np.clip(0.55 + 0.65 * raw_k * conf + 0.10 * (ev > .20).astype(float), 0.45, 1.15)
    base_risk = pd.to_numeric(z['risk_fraction'], errors='coerce').fillna(.0025)
    z['quant_base_risk_fraction'] = base_risk
    z['quant_risk_scale'] = risk_scale
    z['risk_fraction'] = np.minimum(MAX_PER_TRADE_RISK, base_risk * risk_scale)

    z = _causal_factor_consensus(z)
    return z


def promote_for_week(cands, week_start):
    # Start strictly from the locked V2 candidate set; quant maths is an overlay,
    # not a replacement strategy universe and not a scope-narrowing shortcut.
    p, conf, stats = v2.promote_for_week(cands, week_start)
    if p.empty:
        return p, conf, stats
    q = _quantify(p, cands, pd.Timestamp(week_start))
    return q, conf, stats


def replay_r5(cands, specs, start=100.0, feature_cache=None):
    # Retain V2 causal management/execution so this replay identifies the incremental
    # contribution of mathematical selection, ranking, factor context and sizing.
    return v2.replay_r5(cands, specs, start, feature_cache)
