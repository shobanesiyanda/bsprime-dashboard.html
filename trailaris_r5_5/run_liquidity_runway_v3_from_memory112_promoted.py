#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

V2_BASELINE = {
    "baseline_id": "R5_5_INTEGRATED_NEGATIVE_REFINEMENT_V2_26W_LOCKED_20260816",
    "end_equity": 1366.8938578865148,
    "compounded_return_pct": 1266.8938578865148,
    "executed_trades": 2136,
    "wins": 1221,
    "losses": 478,
    "flats": 437,
    "nonflat_win_rate": 0.7186580341377281,
    "profit_factor": 2.9195127192871873,
    "max_weekly_drawdown_pct": -3.7440606836682933,
    "max_event_equity_drawdown_pct": -2.7023988320913572,
    "positive_weeks": 26,
    "rolling_11w_min_return_pct": 175.6827305105836,
    "rolling_11w_median_return_pct": 198.27176646902836,
    "rolling_11w_mean_return_pct": 200.3837728607354,
    "rolling_11w_max_return_pct": 223.82144949395718,
}

CHECKPOINTS = (0.75, 1.00, 1.25)
LOOKBACK_DAYS = 112
K_MIN = 24
K_MAX = 80
SHRINK_N = 24.0

NUMERIC_ENTRY_FEATURES = [
    "quality", "expected_r", "reliability_score",
    "reliability_asset_strategy_mean_r", "rank_score", "cost_r",
    "px_volatility_r", "px_impulse_r", "px_efficiency",
    "px_range_ratio", "px_friction_ratio", "hour_sin", "hour_cos",
]
NUMERIC_CONT_FEATURES = NUMERIC_ENTRY_FEATURES + ["elapsed_minutes"]


def rolling(W: pd.DataFrame, n: int = 11) -> pd.DataFrame:
    r = pd.to_numeric(W.weekly_return_pct, errors="coerce").fillna(0).to_numpy(float)
    rows = []
    for i in range(len(W) - n + 1):
        z = r[i:i+n]
        rows.append({
            "window": i + 1,
            "start_week": str(W.week_start.iloc[i]),
            "end_week": str(W.week_start.iloc[i+n-1]),
            "compounded_return_pct": float((np.prod(1 + z/100) - 1) * 100),
            "mean_week_pct": float(np.mean(z)),
            "median_week_pct": float(np.median(z)),
            "positive_weeks": int((z > 0).sum()),
            "weeks_ge5": int((z >= 5).sum()),
            "weeks_ge10": int((z >= 10).sum()),
            "max_weekly_dd_pct": float(pd.to_numeric(W.max_drawdown_pct.iloc[i:i+n], errors="coerce").min()),
        })
    return pd.DataFrame(rows)


def _build_component(job):
    import trailaris_r4_full_universe_loop as r4
    return r4.build_asset_components(job)


def load_feature_cache(rawdir: Path, routes: list[str]):
    jobs = [(a, str(rawdir / f"{a}_M1_normalized.csv")) for a in routes]
    for a, p in jobs:
        if not Path(p).exists():
            raise RuntimeError(f"missing preserved M1 path: {a}")
    cache = {}
    workers = min(8, max(2, os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_build_component, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            asset, _cands, xf, _opps = fut.result()
            if xf is None or len(xf) == 0:
                raise RuntimeError(f"empty exact R4 feature frame: {asset}")
            xf = xf.copy()
            xf["timestamp"] = pd.to_datetime(xf["timestamp"], utc=True, errors="coerce")
            xf = xf.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            need = {"timestamp", "high", "low", "close"}
            missing = need - set(xf.columns)
            if missing:
                raise RuntimeError(f"{asset} exact R4 feature frame missing {sorted(missing)}")
            cache[str(asset)] = xf
    if set(cache) != set(routes):
        raise RuntimeError(f"exact R4 feature cache gate failed: {len(cache)}/34")
    raw_end = max(pd.Timestamp(x.timestamp.iloc[-1]) for x in cache.values())
    return cache, raw_end


def _safe(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _pre_window(x: pd.DataFrame, asof: pd.Timestamp, minutes: int = 60):
    ts = x["timestamp"]
    hi = int(ts.searchsorted(pd.Timestamp(asof), side="left"))
    if hi <= 2:
        return x.iloc[0:0]
    lo_t = pd.Timestamp(asof) - pd.Timedelta(minutes=minutes)
    lo = int(ts.searchsorted(lo_t, side="left"))
    return x.iloc[lo:hi]


def snapshot_features(r, fcache, asof=None):
    asof = pd.Timestamp(asof if asof is not None else r.decision_time)
    x = fcache.get(str(r.asset))
    sd = max(_safe(r.get("stop_distance", 0.0)), 1e-12)
    dr = int(_safe(r.get("direction", 1), 1))
    z = _pre_window(x, asof, 60) if x is not None else pd.DataFrame()
    if len(z) >= 4:
        close = pd.to_numeric(z["close"], errors="coerce").to_numpy(float)
        high = pd.to_numeric(z["high"], errors="coerce").to_numpy(float)
        low = pd.to_numeric(z["low"], errors="coerce").to_numpy(float)
        good = np.isfinite(close) & np.isfinite(high) & np.isfinite(low)
        close, high, low = close[good], high[good], low[good]
    else:
        close = high = low = np.array([], dtype=float)
    if len(close) >= 4:
        vol_r = float((np.nanmax(high) - np.nanmin(low)) / sd)
        j = max(0, len(close) - 15)
        impulse_r = float(dr * (close[-1] - close[j]) / sd)
        dif = np.diff(close)
        efficiency = float(abs(close[-1] - close[0]) / (np.nansum(np.abs(dif)) + 1e-12))
        z15h, z15l = high[-min(15, len(high)):], low[-min(15, len(low)):]
        recent_range = float(np.nanmax(z15h) - np.nanmin(z15l))
        if len(high) > 15:
            prevh, prevl = high[:-15], low[:-15]
            prev_range = float(np.nanmax(prevh) - np.nanmin(prevl))
        else:
            prev_range = recent_range
        range_ratio = recent_range / max(prev_range, 1e-12)
    else:
        vol_r, impulse_r, efficiency, range_ratio = 0.0, 0.0, 0.0, 1.0
    cost_r = _safe(r.get("cost_r", 0.0))
    friction = cost_r / max(vol_r, 0.10)
    hour = asof.hour + asof.minute / 60.0
    return {
        "quality": _safe(r.get("quality", 0.0)),
        "expected_r": _safe(r.get("expected_r", 0.0)),
        "reliability_score": _safe(r.get("reliability_score", 0.0)),
        "reliability_asset_strategy_mean_r": _safe(r.get("reliability_asset_strategy_mean_r", 0.0)),
        "rank_score": _safe(r.get("rank_score", 0.0)),
        "cost_r": cost_r,
        "px_volatility_r": vol_r,
        "px_impulse_r": impulse_r,
        "px_efficiency": efficiency,
        "px_range_ratio": range_ratio,
        "px_friction_ratio": friction,
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
    }


def first_reach_time(r, fcache, checkpoint):
    x = fcache.get(str(r.asset))
    if x is None or len(x) == 0:
        return pd.NaT
    ts = x["timestamp"]
    entry, exit_ = pd.Timestamp(r.entry_time), pd.Timestamp(r.exit_time)
    lo_i = int(ts.searchsorted(entry, side="left"))
    hi_i = int(ts.searchsorted(exit_, side="right"))
    if hi_i <= lo_i:
        return pd.NaT
    z = x.iloc[lo_i:hi_i]
    ent = _safe(r.entry_price)
    sd = max(_safe(r.stop_distance), 1e-12)
    dr = int(_safe(r.direction, 1))
    fav = ((pd.to_numeric(z["high"], errors="coerce").to_numpy(float) - ent) / sd
           if dr == 1 else
           (ent - pd.to_numeric(z["low"], errors="coerce").to_numpy(float)) / sd)
    idx = np.flatnonzero(fav >= float(checkpoint))
    return pd.Timestamp(z.iloc[int(idx[0])]["timestamp"]) if len(idx) else pd.NaT


def enrich_estate(PR0, fcache):
    rows = []
    for _, r in PR0.iterrows():
        d = r.to_dict()
        d.update(snapshot_features(r, fcache, r.decision_time))
        mfe = max(0.0, _safe(r.get("mfe_r", 0.0)))
        d["label_dev075"] = float(mfe >= 0.75)
        d["label_run150"] = float(mfe >= 1.50)
        d["label_mfe_clip"] = float(min(mfe, 20.0))
        d["label_net_r"] = _safe(r.get("net_r", 0.0))
        for cp in CHECKPOINTS:
            t = first_reach_time(r, fcache, cp)
            key = str(cp).replace(".", "_")
            d[f"cp_{key}_time"] = t
            d[f"cp_{key}_reached"] = bool(pd.notna(t))
            if pd.notna(t):
                sf = snapshot_features(r, fcache, t)
                for k, v in sf.items():
                    d[f"cp_{key}_{k}"] = v
                d[f"cp_{key}_elapsed_minutes"] = max(0.0, (pd.Timestamp(t) - pd.Timestamp(r.entry_time)).total_seconds() / 60.0)
            else:
                for k in NUMERIC_ENTRY_FEATURES:
                    d[f"cp_{key}_{k}"] = np.nan
                d[f"cp_{key}_elapsed_minutes"] = np.nan
        rows.append(d)
    return pd.DataFrame(rows)


def _robust_matrix(hist, row, feature_cols):
    X = hist[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    q = np.array([_safe(row.get(c, np.nan), np.nan) for c in feature_cols], dtype=float)
    med = np.nanmedian(X, axis=0)
    q1 = np.nanquantile(X, .25, axis=0)
    q3 = np.nanquantile(X, .75, axis=0)
    scale = np.where((q3 - q1) > 1e-9, q3 - q1, np.nanstd(X, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    X = np.where(np.isfinite(X), X, med)
    q = np.where(np.isfinite(q), q, med)
    return X, q, med, scale


def knn_predict(hist, row, feature_cols, targets):
    if len(hist) < 12:
        out = {}
        for t in targets:
            s = pd.to_numeric(hist[t], errors="coerce").dropna() if t in hist else pd.Series(dtype=float)
            out[t] = float(s.mean()) if len(s) else 0.0
        out["neighbors"] = int(len(hist))
        out["confidence"] = float(min(1.0, len(hist) / 50))
        return out
    X, q, med, scale = _robust_matrix(hist, row, feature_cols)
    d = np.sqrt(np.nanmean(((X - q) / scale) ** 2, axis=1))
    same_as = ((hist["asset"].astype(str).to_numpy() == str(row.get("asset", ""))) &
               (hist["strategy"].astype(str).to_numpy() == str(row.get("strategy", ""))))
    same_family = hist["strategy_family"].astype(str).to_numpy() == str(row.get("strategy_family", ""))
    d = d + np.where(same_as, 0.0, np.where(same_family, 0.20, 0.45))
    k = int(min(K_MAX, max(K_MIN, round(2 * np.sqrt(len(hist))))))
    ix = np.argsort(d)[:k]
    w = 1.0 / (0.20 + d[ix])
    out = {}
    for t in targets:
        y = pd.to_numeric(hist.iloc[ix][t], errors="coerce").to_numpy(float)
        ok = np.isfinite(y)
        if not ok.any():
            local = 0.0
            n = 0
        else:
            local = float(np.average(y[ok], weights=w[ok]))
            n = int(ok.sum())
        g = pd.to_numeric(hist[t], errors="coerce").dropna()
        prior = float(g.mean()) if len(g) else local
        out[t] = float((n * local + SHRINK_N * prior) / (n + SHRINK_N))
    out["neighbors"] = k
    out["confidence"] = float(min(1.0, k / 60.0) * min(1.0, len(hist) / 120.0))
    return out


def percentile_score(s, v, higher_better=True):
    a = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if len(a) < 10 or not np.isfinite(v):
        return .5
    p = float((a <= v).mean())
    return p if higher_better else 1 - p


def liquidity_proxy_score(hist, row, prefix=""):
    def c(name):
        return prefix + name
    vol = percentile_score(hist[c("px_volatility_r")], _safe(row.get(c("px_volatility_r"), 0)), True)
    eff = percentile_score(hist[c("px_efficiency")], _safe(row.get(c("px_efficiency"), 0)), True)
    rr = percentile_score(hist[c("px_range_ratio")], _safe(row.get(c("px_range_ratio"), 1)), True)
    fr = percentile_score(hist[c("px_friction_ratio")], _safe(row.get(c("px_friction_ratio"), 1)), False)
    imp = np.clip((_safe(row.get(c("px_impulse_r"), 0)) + 0.40) / 1.20, 0, 1)
    return float(np.clip(.30 * vol + .20 * eff + .15 * rr + .20 * fr + .15 * imp, 0, 1))


def liq_state(score):
    if score >= .72:
        return "ABUNDANT_SUPPORTIVE"
    if score >= .58:
        return "SUPPORTIVE"
    if score >= .44:
        return "ADEQUATE"
    if score >= .32:
        return "THIN"
    if score >= .20:
        return "DETERIORATING"
    return "HOSTILE"


def entry_assessment(hist, r):
    if len(hist) < 60:
        return {
            "p_reach_0_75r_before_stop": 0.50,
            "p_reach_1_50r_from_entry": 0.50,
            "expected_executable_excursion_r": 1.50,
            "predicted_net_r": 0.0,
            "runway_confidence": 0.0,
            "liquidity_sufficiency_score": 0.50,
            "liquidity_sufficiency_state": "V2_BASELINE_FALLBACK",
            "campaign_runway_class": "V2_BASELINE_FALLBACK",
            "runway_risk_multiplier": 1.0,
        }
    pred = knn_predict(hist, r, NUMERIC_ENTRY_FEATURES, ["label_dev075", "label_run150", "label_mfe_clip", "label_net_r"])
    liq = liquidity_proxy_score(hist, r)
    p075 = float(np.clip(pred["label_dev075"], 0, 1))
    p150 = float(np.clip(pred["label_run150"], 0, 1))
    eee = max(0.0, float(pred["label_mfe_clip"]))
    exp_net = float(pred["label_net_r"])
    if p075 < .45 and p150 < .30 and eee < .90 and liq < .38:
        cls = "REJECT_INSUFFICIENT_RUNWAY"
        mult = 0.0
    elif p075 >= .56 and (p150 >= .44 or eee >= 1.50) and liq >= .38:
        cls = "RUNNER"
        mult = 1.0
    else:
        cls = "EXTRACTION"
        mult = .70 if p075 >= .50 else .50
    return {
        "p_reach_0_75r_before_stop": p075,
        "p_reach_1_50r_from_entry": p150,
        "expected_executable_excursion_r": eee,
        "predicted_net_r": exp_net,
        "runway_confidence": pred["confidence"],
        "liquidity_sufficiency_score": liq,
        "liquidity_sufficiency_state": liq_state(liq),
        "campaign_runway_class": cls,
        "runway_risk_multiplier": mult,
    }


def cp_row_from_enriched(r, cp):
    key = str(cp).replace(".", "_")
    d = r.to_dict() if isinstance(r, pd.Series) else dict(r)
    for k in NUMERIC_ENTRY_FEATURES:
        d[k] = d.get(f"cp_{key}_{k}", np.nan)
    d["elapsed_minutes"] = d.get(f"cp_{key}_elapsed_minutes", np.nan)
    return d


def continuation_assessment(hist_enriched, r, cp):
    key = str(cp).replace(".", "_")
    h = hist_enriched[hist_enriched[f"cp_{key}_reached"] == True].copy()
    if len(h) < 30:
        return {
            "p_continue_1_50r": 0.50,
            "expected_remaining_excursion_r": max(0.0, 1.50 - float(cp)),
            "continuation_confidence": 0.0,
            "liquidity_score": 0.50,
            "liquidity_state": "V2_BASELINE_FALLBACK",
            "insufficient_history": True,
        }
    hh = pd.DataFrame(index=h.index)
    for k in NUMERIC_ENTRY_FEATURES:
        hh[k] = pd.to_numeric(h[f"cp_{key}_{k}"], errors="coerce")
    hh["elapsed_minutes"] = pd.to_numeric(h[f"cp_{key}_elapsed_minutes"], errors="coerce")
    for c in ("asset", "strategy", "strategy_family", "label_run150", "label_mfe_clip"):
        hh[c] = h[c].values
    row = cp_row_from_enriched(r, cp)
    hh["label_remaining_excursion"] = np.maximum(0.0, pd.to_numeric(hh["label_mfe_clip"], errors="coerce") - float(cp))
    pred = knn_predict(hh, row, NUMERIC_CONT_FEATURES, ["label_run150", "label_remaining_excursion"])
    liq = liquidity_proxy_score(hh, row)
    return {
        "p_continue_1_50r": float(np.clip(pred["label_run150"], 0, 1)),
        "expected_remaining_excursion_r": max(0.0, float(pred["label_remaining_excursion"])),
        "continuation_confidence": pred["confidence"],
        "liquidity_score": liq,
        "liquidity_state": liq_state(liq),
        "insufficient_history": False,
    }


def target_bank_fraction(runway_class, cp, cont, baseline_weak_state=False):
    if bool(cont.get("insufficient_history", False)):
        target = .25 if (baseline_weak_state and float(cp) == .75) else 0.0
        return float(target), float("nan")
    p = cont["p_continue_1_50r"]
    liq = cont["liquidity_score"]
    rem = cont["expected_remaining_excursion_r"]
    rem_support = float(np.clip(rem / max(1.50 - cp, 0.25), 0, 1))
    score = .60 * p + .25 * liq + .15 * rem_support
    if score >= .72:
        target = 0.00
    elif score >= .58:
        target = 0.10
    elif score >= .45:
        target = 0.25
    elif score >= .30:
        target = 0.40
    else:
        target = 0.60
    if runway_class == "EXTRACTION" and cp == 0.75 and score < .72:
        target = max(target, .25)
    return float(target), float(score)


def apply_adaptive_banking(r, hist, fcache):
    old_gross = _safe(r.get("gross_r", _safe(r.get("net_r", 0)) + _safe(r.get("cost_r", 0))))
    total_bank = 0.0
    bank_value = 0.0
    trace = []
    for cp in CHECKPOINTS:
        key = str(cp).replace(".", "_")
        if not bool(r.get(f"cp_{key}_reached", False)):
            continue
        cont = continuation_assessment(hist, r, cp)
        baseline_weak = (_safe(r.get("rank_score", 999.0), 999.0) <= .42 and
                         _safe(r.get("reliability_score", 999.0), 999.0) <= 0.0)
        target, score = target_bank_fraction(str(r.get("campaign_runway_class", "EXTRACTION")), cp, cont, baseline_weak)
        target = max(total_bank, target)
        inc = max(0.0, target - total_bank)
        if inc > 0:
            bank_value += inc * cp
            total_bank += inc
        trace.append({"checkpoint_r": cp, "continuation_score": score, "target_bank_fraction": target,
                      "incremental_bank_fraction": inc, **cont})
    out = r.copy()
    if total_bank <= 0:
        out["r5_5_v3_adaptive_bank_fraction"] = 0.0
        out["r5_5_v3_runner_fraction"] = 1.0
        out["r5_5_v3_management_trace"] = json.dumps(trace, separators=(",", ":"))
        return out
    new_gross = bank_value + (1 - total_bank) * old_gross
    cost = _safe(r.get("cost_r", 0))
    out["gross_r"] = float(new_gross)
    out["net_r"] = float(new_gross - cost)
    if pd.notna(r.get("mfe_r", np.nan)):
        out["giveback_r"] = float(max(0.0, _safe(r.get("mfe_r", 0)) - out["net_r"]))
    out["r5_5_v3_adaptive_bank_fraction"] = float(total_bank)
    out["r5_5_v3_runner_fraction"] = float(1 - total_bank)
    out["r5_5_v3_management_trace"] = json.dumps(trace, separators=(",", ":"))
    out["r5_5_v3_management_repaired"] = True
    out["r5_5_v3_original_exit_reason"] = str(r.get("exit_reason", ""))
    out["exit_reason"] = "R5_5_V3_ADAPTIVE_BANK__" + str(r.get("exit_reason", ""))
    return out


def metrics(EV, DE, W, roll):
    wins = int((EV.net_pnl > 0).sum())
    losses = int((EV.net_pnl < 0).sum())
    flats = int((EV.net_pnl == 0).sum())
    gw = float(EV.loc[EV.net_pnl > 0, "net_pnl"].sum())
    gl = float(-EV.loc[EV.net_pnl < 0, "net_pnl"].sum())
    eq = pd.concat([pd.Series([100.0]), pd.to_numeric(EV.sort_values("timestamp").equity, errors="coerce").dropna()], ignore_index=True)
    dd = (eq / eq.cummax() - 1) * 100
    return {
        "end_equity": float(W.end_equity.iloc[-1]),
        "compounded_return_pct": float((W.end_equity.iloc[-1] / 100 - 1) * 100),
        "evaluated_weeks": int(len(W)),
        "positive_weeks": int((W.weekly_return_pct > 0).sum()),
        "negative_weeks": int((W.weekly_return_pct < 0).sum()),
        "mean_week_pct": float(W.weekly_return_pct.mean()),
        "median_week_pct": float(W.weekly_return_pct.median()),
        "weeks_ge5": int((W.weekly_return_pct >= 5).sum()),
        "weeks_ge10": int((W.weekly_return_pct >= 10).sum()),
        "executed_trades": int(len(EV)),
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "loss_rate": float(losses / len(EV)) if len(EV) else 0.0,
        "flat_rate": float(flats / len(EV)) if len(EV) else 0.0,
        "nonflat_win_rate": float(wins / (wins + losses)) if wins + losses else 0.0,
        "profit_factor": float(gw / gl) if gl else 999.0,
        "max_weekly_drawdown_pct": float(pd.to_numeric(W.max_drawdown_pct, errors="coerce").min()),
        "max_event_equity_drawdown_pct": float(dd.min()) if len(dd) else 0.0,
        "assets_with_selected": int(DE.loc[DE.selected, "asset"].nunique()) if len(DE) and DE.selected.any() else 0,
        "rolling_11w_windows": int(len(roll)),
        "rolling_11w_min_return_pct": float(roll.compounded_return_pct.min()),
        "rolling_11w_median_return_pct": float(roll.compounded_return_pct.median()),
        "rolling_11w_mean_return_pct": float(roll.compounded_return_pct.mean()),
        "rolling_11w_max_return_pct": float(roll.compounded_return_pct.max()),
    }


def delta(c, b):
    keys = ["end_equity", "compounded_return_pct", "executed_trades", "wins", "losses", "flats",
            "nonflat_win_rate", "profit_factor", "max_weekly_drawdown_pct", "max_event_equity_drawdown_pct",
            "positive_weeks", "rolling_11w_min_return_pct", "rolling_11w_median_return_pct",
            "rolling_11w_mean_return_pct", "rolling_11w_max_return_pct"]
    return {k: float(c[k] - b[k]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rawdir", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--promoted", required=True)
    ap.add_argument("--factory-audit", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = ["trailaris_r4", "trailaris_r5_4", "trailaris_r5_4/reliability_variants", "trailaris_r5_5"]
    import trailaris_r4_full_universe_loop as r4
    import R5_BASE_CAUSAL_ENGINE as base
    import longcycle_integrated_negative_refinement as v2

    fa = pd.read_csv(a.factory_audit)
    if len(fa) != 510 or int(fa.strategy_family.nunique()) != 15:
        raise RuntimeError("factory hard lock failed")
    PR0 = pd.read_csv(a.promoted)
    for c in ("entry_time", "decision_time", "exit_time", "eval_week"):
        PR0[c] = pd.to_datetime(PR0[c], utc=True, errors="coerce")
    PR0 = PR0.dropna(subset=["entry_time", "decision_time", "exit_time", "eval_week"]).copy()
    weeks = sorted(pd.Timestamp(x) for x in PR0.eval_week.unique())
    if len(PR0) != 3417 or len(weeks) != 26 or int(PR0.asset.nunique()) != 34:
        raise RuntimeError("memory112 estate hard lock failed")
    required = {"mfe_r", "net_r", "stop_distance", "entry_price", "direction", "quality", "expected_r",
                "reliability_score", "reliability_asset_strategy_mean_r", "rank_score", "cost_r",
                "strategy_family", "strategy"}
    missing = required - set(PR0.columns)
    if missing:
        raise RuntimeError(f"promotion estate missing required causal/training fields: {sorted(missing)}")
    routes = list(r4.ROUTES)
    specs = r4.load_specs(Path(a.specs), "research-proxy")
    fcache, raw_end = load_feature_cache(Path(a.rawdir), routes)
    E = enrich_estate(PR0, fcache)

    equity = 100.0
    Wrows, events, decisions, kept_rows, rejected_rows, assessment_rows = [], [], [], [], [], []
    for wk in weeks:
        hist = E[(E.exit_time < wk) & (E.exit_time >= wk - pd.Timedelta(days=LOOKBACK_DAYS))].copy()
        src = E[E.eval_week == wk].copy()
        v2_kept, v2_rejected = v2.entry_gate(src)
        if len(v2_rejected):
            rr = v2_rejected.copy()
            rr["r5_5_v3_gate_reason"] = "V2_NEGATIVE_EDGE_GATE"
            rejected_rows.append(rr.assign(eval_week=wk))
        assessed = []
        for _, r in v2_kept.iterrows():
            a1 = entry_assessment(hist, r)
            q = r.copy()
            for k, v in a1.items():
                q[k] = v
            assessment_rows.append({**{k: q.get(k) for k in ["campaign_id", "asset", "strategy", "strategy_family", "eval_week"]}, **a1})
            if a1["campaign_runway_class"] == "REJECT_INSUFFICIENT_RUNWAY":
                q["r5_5_v3_gate_reason"] = "RUNWAY_REJECT"
                rejected_rows.append(pd.DataFrame([q]))
                continue
            q["risk_fraction"] = _safe(q.get("risk_fraction", .005)) * a1["runway_risk_multiplier"]
            q = apply_adaptive_banking(q, hist, fcache)
            assessed.append(q)
        kept = pd.DataFrame(assessed)
        if len(kept):
            replay_cols = kept.drop(columns=["eval_week"], errors="ignore")
            ev, de = base.replay_r5(replay_cols, specs, equity, fcache)
            if len(ev):
                equity = float(ev.equity.iloc[-1])
            events.append(ev.assign(eval_week=wk))
            decisions.append(de.assign(eval_week=wk))
            kept_rows.append(kept.assign(eval_week=wk))
        else:
            ev = pd.DataFrame()
            de = pd.DataFrame()
        prev = Wrows[-1]["end_equity"] if Wrows else 100.0
        end_ts = min(wk + pd.Timedelta(days=7), raw_end)
        w = r4.weekly(ev, start=prev, feature_cache=fcache, start_ts=wk, end_ts=end_ts)
        if len(w):
            rec = w.iloc[0].to_dict()
            equity = float(rec["end_equity"])
        else:
            rec = dict(week_start=wk, start_equity=prev, end_equity=prev, weekly_return_pct=0.0,
                       campaigns_entered=0, campaigns_closed=0, max_drawdown_pct=0.0, ge5=False, ge10=False)
            equity = float(prev)
        rec["source_promoted"] = int(len(src))
        rec["v3_promoted_after_all_gates"] = int(len(kept))
        Wrows.append(rec)

    W = pd.DataFrame(Wrows)
    EV = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    DE = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    PR = pd.concat(kept_rows, ignore_index=True) if kept_rows else pd.DataFrame()
    RJ = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame()
    AS = pd.DataFrame(assessment_rows)
    roll = rolling(W)
    cand = metrics(EV, DE, W, roll)
    mg = EV.get("r5_5_v3_management_repaired", pd.Series(False, index=EV.index)).fillna(False).astype(bool) if len(EV) else pd.Series(dtype=bool)
    cls = PR.get("campaign_runway_class", pd.Series(dtype=str)).value_counts().to_dict() if len(PR) else {}
    liq = PR.get("liquidity_sufficiency_state", pd.Series(dtype=str)).value_counts().to_dict() if len(PR) else {}
    status = {
        "state": "R5_5_LIQUIDITY_RUNWAY_V3_26W_REPLAY_COMPLETE",
        "baseline": V2_BASELINE,
        "scope": {"routes": 34, "strategy_families": 15, "route_strategy_cells": 510, "weeks": 26},
        "source_promotion_estate": "EXACT_R5_5_MEMORY112_PROMOTED_3417",
        "feature_cache_source": "EXACT_R4_BUILD_ASSET_COMPONENTS_FROM_PRESERVED_34_ROUTE_M1_CACHE",
        "candidate": cand,
        "delta_candidate_minus_locked_v2": delta(cand, V2_BASELINE),
        "architecture": {
            "entry_continuation_separated": True,
            "runway_model": "CAUSAL_112D_REGULARIZED_KNN_FROM_PRIOR_EXITED_PROMOTIONS_WITH_LOCKED_V2_FALLBACK_WHEN_HISTORY_INSUFFICIENT",
            "direct_liquidity": "NOT_AVAILABLE_IN_HISTORICAL_ESTATE",
            "liquidity_model": "CAUSAL_PRICE_VOLATILITY_IMPULSE_EFFICIENCY_RANGE_FRICTION_PROXY",
            "developed_position_checkpoints_r": list(CHECKPOINTS),
            "adaptive_banking": "0_TO_60_PERCENT_CUMULATIVE_BY_CONTINUATION_STATE",
            "runner": "UNRESTRICTED_REMAINDER; NO_GLOBAL_3R_CEILING",
            "capacity_credit_after_partial": False,
            "v2_negative_edge_gate_preserved": True,
        },
        "counters": {
            "entry_assessments": int(len(AS)),
            "promoted_after_all_gates": int(len(PR)),
            "rejected_total": int(len(RJ)),
            "runway_classes": {str(k): int(v) for k, v in cls.items()},
            "liquidity_states": {str(k): int(v) for k, v in liq.items()},
            "adaptively_banked_selected_trades": int(mg.sum()) if len(mg) else 0,
        },
        "causality": "Current entry features use strictly pre-decision bars. Weekly runway models train only on promotions exited before the evaluation week. Historical MFE is used only as a training label. Developed-position re-underwriting occurs only after the contemporaneous path first reaches each checkpoint; no future MFE/MAE enters current eligibility. Partial banking does not shorten the frozen exit time or create extra capacity credit.",
        "automatic_baseline_promotion": False,
        "evidence_class": "FULL_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; LIQUIDITY_IS_PROXY_NOT_DIRECT_BOOK; BROKER_PRODUCTION_CERTIFICATION_SEPARATE",
    }
    prefix = "R5_5_LIQUIDITY_RUNWAY_V3"
    W.to_csv(out / f"{prefix}_WEEKLY.csv", index=False)
    roll.to_csv(out / f"{prefix}_ROLLING_11W.csv", index=False)
    EV.to_csv(out / f"{prefix}_EVENTS.csv", index=False)
    DE.to_csv(out / f"{prefix}_DECISIONS.csv", index=False)
    PR.to_csv(out / f"{prefix}_PROMOTED.csv", index=False)
    RJ.to_csv(out / f"{prefix}_REJECTED.csv", index=False)
    AS.to_csv(out / f"{prefix}_ENTRY_ASSESSMENTS.csv", index=False)
    (out / f"{prefix}_STATUS.json").write_text(json.dumps(status, indent=2, default=str))
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()
