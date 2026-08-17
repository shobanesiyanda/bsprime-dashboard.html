#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE_KEYS = [
    "end_equity",
    "compounded_return_pct",
    "positive_weeks",
    "negative_weeks",
    "mean_week_pct",
    "median_week_pct",
    "weeks_ge5",
    "weeks_ge10",
    "executed_trades",
    "wins",
    "losses",
    "flats",
    "loss_rate",
    "flat_rate",
    "nonflat_win_rate",
    "profit_factor",
    "max_weekly_drawdown_pct",
    "max_event_equity_drawdown_pct",
    "assets_with_selected",
    "rolling_11w_min_return_pct",
    "rolling_11w_median_return_pct",
    "rolling_11w_mean_return_pct",
    "rolling_11w_max_return_pct",
]


def num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--v2-status", required=True)
    ap.add_argument("--math-status", required=True)
    ap.add_argument("--v2-promoted", required=True)
    ap.add_argument("--math-promoted", required=True)
    ap.add_argument("--math-decisions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    baseline = json.loads(Path(a.baseline).read_text())
    perf = baseline["performance"]
    v2 = json.loads(Path(a.v2_status).read_text())["candidate"]
    mathc = json.loads(Path(a.math_status).read_text())["candidate"]

    # Reproduction gate: the execution path must reproduce the frozen V2 control
    # before the mathematical overlay is accepted as a valid comparison.
    mismatches = {}
    for k, b in perf.items():
        if k not in v2 or not num(b) or not num(v2[k]):
            continue
        tol = 1e-9 if isinstance(b, float) else 0
        if abs(float(v2[k]) - float(b)) > tol:
            mismatches[k] = {"canonical": b, "reproduced": v2[k], "delta": float(v2[k]) - float(b)}
    if mismatches:
        raise RuntimeError("R5.5 V2 reproduction gate failed: " + json.dumps(mismatches, sort_keys=True))

    delta = {}
    for k in CORE_KEYS:
        if k in mathc and k in perf and num(mathc[k]) and num(perf[k]):
            delta[k] = float(mathc[k]) - float(perf[k])

    v2p = pd.read_csv(a.v2_promoted)
    mp = pd.read_csv(a.math_promoted)
    md = pd.read_csv(a.math_decisions)

    diag = {
        "v2_promoted_rows": int(len(v2p)),
        "math_promoted_rows": int(len(mp)),
        "math_gate_removed_rows_vs_v2": int(max(0, len(v2p) - len(mp))),
    }

    if "math_pre_market_risk_scale" in mp.columns:
        s = pd.to_numeric(mp["math_pre_market_risk_scale"], errors="coerce").dropna()
        diag["pre_market_risk_scale_distribution"] = {str(k): int(v) for k, v in s.value_counts().sort_index().items()}
    if "math_posterior_ev_r" in mp.columns:
        s = pd.to_numeric(mp["math_posterior_ev_r"], errors="coerce").dropna()
        if len(s):
            diag["posterior_ev_r"] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p10": float(s.quantile(0.10)),
                "p90": float(s.quantile(0.90)),
            }
    if "math_lower_ev_r" in mp.columns:
        s = pd.to_numeric(mp["math_lower_ev_r"], errors="coerce").dropna()
        if len(s):
            diag["lower_ev_r"] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p10": float(s.quantile(0.10)),
                "p90": float(s.quantile(0.90)),
            }

    if len(md) and "selected" in md.columns:
        sel = md["selected"].astype(str).str.lower().isin(["true", "1"])
        diag["selected_decisions"] = int(sel.sum())
        for col in ("math_market_risk_scale", "math_vol_ratio_14d_56d", "math_corr_crowding", "math_directional_alignment"):
            if col in md.columns:
                s = pd.to_numeric(md.loc[sel, col], errors="coerce").dropna()
                if len(s):
                    diag[col] = {"mean": float(s.mean()), "median": float(s.median()), "max": float(s.max()), "min": float(s.min())}

    result = {
        "state": "R5_5_V2_MATH_LOGICAL_26W_REPLAY_COMPLETE",
        "source_baseline_id": baseline["baseline_id"],
        "v2_reproduction_gate": "PASS",
        "scope": baseline["scope"],
        "frozen_r5_5_v2": perf,
        "math_logical_candidate": mathc,
        "delta_math_minus_r5_5_v2": delta,
        "weekly_probability_observed": {
            "p_week_ge_5pct": float(mathc["weeks_ge5"] / 26.0),
            "p_week_ge_10pct": float(mathc["weeks_ge10"] / 26.0),
            "p_positive_week": float(mathc["positive_weeks"] / 26.0),
        },
        "mathematical_diagnostics": diag,
        "automatic_promotion": False,
        "evidence_class": "FULL_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE",
    }
    Path(a.out).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
