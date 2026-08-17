from __future__ import annotations

import math
import numpy as np
import pandas as pd

import longcycle_integrated_negative_refinement as v2

# Trailaris successor research overlay.
# The frozen R5.5 V2 architecture remains untouched. This module adds only
# causal mathematical/logical decision overlays on top of V2 promotion,
# ranking and risk allocation.

LONG_LOOKBACK_DAYS = 112
RECENT_LOOKBACK_DAYS = 14
CORR_LOOKBACK_DAYS = 56

BAYES_ALPHA_WIN = 1.0
BAYES_ALPHA_LOSS = 1.0
BAYES_ALPHA_FLAT = 1.0

MATH_MIN_EVIDENCE_N = 8
HARD_GATE_MIN_N = 12
HARD_GATE_MAX_POSTERIOR_EV_R = -0.05
HARD_GATE_MAX_LOWER_EV_R = -0.15

FULL_RISK_MIN_POSTERIOR_EV_R = 0.10
FULL_RISK_MIN_LOWER_EV_R = 0.00
MID_RISK_MIN_POSTERIOR_EV_R = 0.00
MID_RISK_MIN_LOWER_EV_R = -0.10
SPARSE_EVIDENCE_RISK_SCALE = 0.75
MID_EVIDENCE_RISK_SCALE = 0.75
WEAK_EVIDENCE_RISK_SCALE = 0.50

VOL_STRESS_RATIO = 1.50
CORR_STRESS_LEVEL = 0.75
STRESS_RISK_SCALE = 0.75


def _safe_num(x, default=0.0):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)


def _coerce_times(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    for c in ("decision_time", "exit_time"):
        if c in z.columns:
            z[c] = pd.to_datetime(z[c], utc=True, errors="coerce")
    return z


def _posterior_summary(g: pd.DataFrame) -> dict:
    rv = pd.to_numeric(g.get("net_r", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(float)
    n = int(len(rv))
    if n == 0:
        return {
            "math_n": 0,
            "math_p_win": 1.0 / 3.0,
            "math_p_loss": 1.0 / 3.0,
            "math_p_flat": 1.0 / 3.0,
            "math_avg_win_r": 0.0,
            "math_avg_loss_r": 1.0,
            "math_posterior_ev_r": -1.0 / 3.0,
            "math_lower_ev_r": -1.0 / 3.0,
            "math_cvar10_r": -1.0,
            "math_long_mean_r": 0.0,
            "math_sd_r": 1.0,
        }

    wins = int((rv > 0).sum())
    losses = int((rv < 0).sum())
    flats = int((rv == 0).sum())

    aw = BAYES_ALPHA_WIN + wins
    al = BAYES_ALPHA_LOSS + losses
    af = BAYES_ALPHA_FLAT + flats
    den = aw + al + af
    p_win, p_loss, p_flat = aw / den, al / den, af / den

    pos = rv[rv > 0]
    neg = rv[rv < 0]
    avg_win = float(pos.mean()) if len(pos) else 0.0
    avg_loss = float(-neg.mean()) if len(neg) else 1.0
    posterior_ev = float(p_win * avg_win - p_loss * avg_loss)

    sd = float(np.std(rv, ddof=1)) if n >= 2 else 1.0
    if not np.isfinite(sd):
        sd = 1.0
    se = sd / max(math.sqrt(n), 1.0)
    lower_ev = float(posterior_ev - se)

    if n >= 2:
        q10 = float(np.quantile(rv, 0.10))
        tail = rv[rv <= q10]
        cvar10 = float(tail.mean()) if len(tail) else float(rv.min())
    else:
        cvar10 = float(rv[0])

    return {
        "math_n": n,
        "math_p_win": float(p_win),
        "math_p_loss": float(p_loss),
        "math_p_flat": float(p_flat),
        "math_avg_win_r": avg_win,
        "math_avg_loss_r": avg_loss,
        "math_posterior_ev_r": posterior_ev,
        "math_lower_ev_r": lower_ev,
        "math_cvar10_r": cvar10,
        "math_long_mean_r": float(rv.mean()),
        "math_sd_r": sd,
    }


def _math_maps(cands: pd.DataFrame, week_start: pd.Timestamp):
    c = _coerce_times(cands)
    long0 = week_start - pd.Timedelta(days=LONG_LOOKBACK_DAYS)
    recent0 = week_start - pd.Timedelta(days=RECENT_LOOKBACK_DAYS)

    hist = c[(c.exit_time < week_start) & (c.exit_time >= long0)].copy()
    recent = hist[hist.exit_time >= recent0].copy()

    long_map = {}
    if len(hist):
        for key, g in hist.groupby(["asset", "strategy"], sort=False):
            long_map[(str(key[0]), str(key[1]))] = _posterior_summary(g)

    recent_map = {}
    if len(recent):
        for key, g in recent.groupby(["asset", "strategy"], sort=False):
            rr = pd.to_numeric(g.net_r, errors="coerce").dropna()
            recent_map[(str(key[0]), str(key[1]))] = {
                "math_recent_n": int(len(rr)),
                "math_recent_mean_r": float(rr.mean()) if len(rr) else 0.0,
            }
    return long_map, recent_map


def _risk_scale(n: int, posterior_ev: float, lower_ev: float) -> float:
    if n < MATH_MIN_EVIDENCE_N:
        return SPARSE_EVIDENCE_RISK_SCALE
    if posterior_ev >= FULL_RISK_MIN_POSTERIOR_EV_R and lower_ev >= FULL_RISK_MIN_LOWER_EV_R:
        return 1.0
    if posterior_ev >= MID_RISK_MIN_POSTERIOR_EV_R and lower_ev >= MID_RISK_MIN_LOWER_EV_R:
        return MID_EVIDENCE_RISK_SCALE
    return WEAK_EVIDENCE_RISK_SCALE


def promote_for_week(cands, week_start):
    # First reproduce the exact R5.5 V2 proposal/reliability/negative-edge gates.
    promoted, conf, stats = v2.promote_for_week(cands, week_start)
    if promoted.empty:
        return promoted, conf, stats

    wk = pd.Timestamp(week_start)
    if wk.tzinfo is None:
        wk = wk.tz_localize("UTC")
    else:
        wk = wk.tz_convert("UTC")

    long_map, recent_map = _math_maps(cands, wk)
    rows = []
    for _, r in promoted.iterrows():
        out = r.copy()
        key = (str(r.asset), str(r.strategy))
        s = long_map.get(key, _posterior_summary(pd.DataFrame()))
        rs = recent_map.get(key, {"math_recent_n": 0, "math_recent_mean_r": s["math_long_mean_r"]})

        recent_mean = _safe_num(rs["math_recent_mean_r"], s["math_long_mean_r"])
        regime_delta = recent_mean - _safe_num(s["math_long_mean_r"])
        regime_score = float(np.tanh(regime_delta / 0.50))

        n = int(s["math_n"])
        posterior_ev = _safe_num(s["math_posterior_ev_r"])
        lower_ev = _safe_num(s["math_lower_ev_r"])

        hard_reject = (
            n >= HARD_GATE_MIN_N
            and posterior_ev <= HARD_GATE_MAX_POSTERIOR_EV_R
            and lower_ev <= HARD_GATE_MAX_LOWER_EV_R
        )

        scale = _risk_scale(n, posterior_ev, lower_ev)

        # Kelly is a bounded diagnostic, not permission to lever above V2 risk.
        p_nonflat = _safe_num(s["math_p_win"]) + _safe_num(s["math_p_loss"])
        if p_nonflat > 0 and _safe_num(s["math_avg_win_r"]) > 0:
            p = _safe_num(s["math_p_win"]) / p_nonflat
            q = _safe_num(s["math_p_loss"]) / p_nonflat
            b = _safe_num(s["math_avg_win_r"]) / max(_safe_num(s["math_avg_loss_r"], 1.0), 1e-12)
            kelly = max(0.0, p - q / max(b, 1e-12))
        else:
            kelly = 0.0

        tail_penalty = max(0.0, -_safe_num(s["math_cvar10_r"]) - 0.75)
        overlay = (
            0.18 * np.tanh(posterior_ev / 0.50)
            + 0.08 * np.tanh(lower_ev / 0.50)
            + 0.05 * regime_score
            + 0.03 * np.tanh(kelly)
            - 0.05 * np.tanh(tail_penalty)
        )

        for k, v in s.items():
            out[k] = v
        out["math_recent_n"] = int(rs["math_recent_n"])
        out["math_recent_mean_r"] = recent_mean
        out["math_regime_delta_r"] = float(regime_delta)
        out["math_regime_score"] = regime_score
        out["math_kelly_diagnostic"] = float(kelly)
        out["math_tail_penalty"] = float(tail_penalty)
        out["math_gate_pass"] = bool(not hard_reject)
        out["math_pre_market_risk_scale"] = float(scale)
        out["risk_fraction"] = float(r.get("risk_fraction", 0.0)) * float(scale)
        out["rank_score"] = float(r.get("rank_score", 0.0)) + float(overlay)
        rows.append(out)

    z = pd.DataFrame(rows)
    if z.empty:
        return z, conf, stats
    z = z[z.math_gate_pass.fillna(False).astype(bool)].copy()
    return z, conf, stats


def _hourly_return_series(x: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    if x is None or len(x) == 0:
        return pd.Series(dtype=float)
    z = x.copy()
    z["timestamp"] = pd.to_datetime(z["timestamp"], utc=True, errors="coerce")
    z["close"] = pd.to_numeric(z["close"], errors="coerce")
    z = z.dropna(subset=["timestamp", "close"])
    z = z[(z.timestamp >= start) & (z.timestamp < end)]
    if len(z) < 20:
        return pd.Series(dtype=float)
    h = z.set_index("timestamp")["close"].resample("1h").last().dropna()
    return h.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def _market_overlay(cands: pd.DataFrame, feature_cache):
    if cands.empty or not feature_cache:
        return cands.copy()

    z = cands.copy()
    z["decision_time"] = pd.to_datetime(z["decision_time"], utc=True, errors="coerce")
    first = z.decision_time.min()
    if pd.isna(first):
        return z

    week_start = first.normalize() - pd.Timedelta(days=int(first.weekday()))
    corr0 = week_start - pd.Timedelta(days=CORR_LOOKBACK_DAYS)
    recent0 = week_start - pd.Timedelta(days=RECENT_LOOKBACK_DAYS)

    assets = sorted(set(z.asset.astype(str)))
    rmap = {}
    for a in assets:
        x = feature_cache.get(a)
        rmap[a] = _hourly_return_series(x, corr0, week_start)

    joined = pd.concat({a: s for a, s in rmap.items() if len(s)}, axis=1) if any(len(s) for s in rmap.values()) else pd.DataFrame()
    corr = joined.corr(min_periods=24) if len(joined) else pd.DataFrame()

    asset_diag = {}
    for a in assets:
        s = rmap.get(a, pd.Series(dtype=float))
        recent = s[s.index >= recent0] if len(s) else pd.Series(dtype=float)
        long_sd = float(s.std()) if len(s) >= 2 else 0.0
        recent_sd = float(recent.std()) if len(recent) >= 2 else 0.0
        vol_ratio = recent_sd / long_sd if long_sd > 0 else 1.0
        if not np.isfinite(vol_ratio):
            vol_ratio = 1.0

        if len(recent) >= 2:
            cum = float((1.0 + recent).prod() - 1.0)
            denom = max(recent_sd * math.sqrt(len(recent)), 1e-12)
            trend_t = float(np.clip(cum / denom, -4.0, 4.0))
        else:
            trend_t = 0.0

        crowd = 0.0
        if a in corr.columns:
            vals = pd.to_numeric(corr[a].drop(labels=[a], errors="ignore"), errors="coerce").dropna()
            vals = vals[vals > 0].sort_values(ascending=False).head(3)
            if len(vals):
                crowd = float(vals.mean())

        asset_diag[a] = (float(vol_ratio), float(trend_t), float(crowd))

    rows = []
    for _, r in z.iterrows():
        out = r.copy()
        vol_ratio, trend_t, crowd = asset_diag.get(str(r.asset), (1.0, 0.0, 0.0))
        raw_dir = _safe_num(r.get("direction", 0))
        direction = 1.0 if raw_dir > 0 else (-1.0 if raw_dir < 0 else 0.0)
        alignment = float(direction * trend_t)

        stress_scale = 1.0
        if vol_ratio >= VOL_STRESS_RATIO:
            stress_scale *= STRESS_RISK_SCALE
        if crowd >= CORR_STRESS_LEVEL:
            stress_scale *= STRESS_RISK_SCALE

        market_overlay = (
            0.04 * np.tanh(alignment)
            - 0.06 * max(0.0, crowd)
            - 0.04 * max(0.0, vol_ratio - 1.0)
        )

        out["math_vol_ratio_14d_56d"] = vol_ratio
        out["math_directional_trend_t"] = trend_t
        out["math_directional_alignment"] = alignment
        out["math_corr_crowding"] = crowd
        out["math_market_risk_scale"] = float(stress_scale)
        out["risk_fraction"] = float(r.get("risk_fraction", 0.0)) * float(stress_scale)
        out["rank_score"] = float(r.get("rank_score", 0.0)) + float(market_overlay)
        rows.append(out)
    return pd.DataFrame(rows)


def replay_r5(cands, specs, start=100.0, feature_cache=None):
    # Market-state and cross-asset math are applied with price history ending
    # before the replay week. V2 management remains unchanged downstream.
    adjusted = _market_overlay(cands, feature_cache)
    return v2.replay_r5(adjusted, specs, start, feature_cache)
