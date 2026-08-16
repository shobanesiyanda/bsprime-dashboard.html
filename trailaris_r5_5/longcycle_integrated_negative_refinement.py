from __future__ import annotations
import numpy as np
import pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import longcycle_bounded_memory_candidate as bounded

# Integrated R5.5 negative-side refinement V2.
# Entry rules use decision-time fields only. Trade management uses the exact
# contemporaneous feature path; final MFE/MAE are never eligibility inputs.
R54_GATE_MAX_QUALITY = 0.76
R54_GATE_MAX_ASSET_STRATEGY_MEAN_R = 0.05

MEM112_GATE_MAX_QUALITY = 0.65
MEM112_GATE_MAX_RELIABILITY_SCORE = 0.15

FLAT_TRIGGER_R = 0.75
FLAT_PARTIAL_FRACTION = 0.25
FLAT_MAX_RANK_SCORE = 0.42
FLAT_MAX_RELIABILITY_SCORE = 0.0


def entry_gate(promoted):
    if promoted.empty:
        return promoted.copy(), promoted.copy()
    q = pd.to_numeric(promoted['quality'], errors='coerce')
    ar = pd.to_numeric(promoted['reliability_asset_strategy_mean_r'], errors='coerce')
    rel = pd.to_numeric(promoted['reliability_score'], errors='coerce')
    tier = promoted['promotion_tier'].astype(str)

    bad_r54 = (q <= R54_GATE_MAX_QUALITY) & (ar <= R54_GATE_MAX_ASSET_STRATEGY_MEAN_R)
    bad_mem112 = (tier == 'A') & (q <= MEM112_GATE_MAX_QUALITY) & (rel <= MEM112_GATE_MAX_RELIABILITY_SCORE)
    bad = bad_r54 | bad_mem112

    kept = promoted.loc[~bad].copy()
    rejected = promoted.loc[bad].copy()
    kept['r5_5_negative_edge_gate_pass'] = True
    rejected['r5_5_negative_edge_gate_pass'] = False
    if len(rejected):
        rr54 = bad_r54.loc[rejected.index]
        rmem = bad_mem112.loc[rejected.index]
        rejected['r5_5_negative_edge_gate_reason'] = np.where(
            rr54 & rmem, 'R54_AND_MEMORY112_NEGATIVE_COHORT',
            np.where(rr54, 'R54_NEGATIVE_COHORT', 'MEMORY112_LOW_QUALITY_RELIABILITY_NEGATIVE_COHORT')
        )
    return kept, rejected


def promote_for_week(cands, week_start):
    p, conf, stats = bounded.promote_for_week(cands, week_start)
    p, _ = entry_gate(p)
    return p, conf, stats


def _path_slice(r, feature_cache):
    x = feature_cache.get(str(r.asset)) if feature_cache else None
    if x is None or len(x) == 0:
        return None
    ts = pd.to_datetime(x['timestamp'], utc=True)
    entry = pd.Timestamp(r.entry_time)
    exit_ = pd.Timestamp(r.exit_time)
    lo = int(ts.searchsorted(entry, side='left'))
    hi = int(ts.searchsorted(exit_, side='right'))
    if hi <= lo:
        return None
    z = x.iloc[lo:hi].copy().reset_index(drop=True)
    z['timestamp'] = pd.to_datetime(z['timestamp'], utc=True)
    return z


def _favorable_path(z, r):
    ent = float(r.entry_price)
    sd = max(float(r.stop_distance), 1e-12)
    dr = int(r.direction)
    hi = pd.to_numeric(z['high'], errors='coerce').to_numpy(float)
    lo = pd.to_numeric(z['low'], errors='coerce').to_numpy(float)
    return (hi - ent) / sd if dr == 1 else (ent - lo) / sd


def _apply_management_row(r, feature_cache):
    if str(r.get('management', '')) != 'PROTECTED_RUNNER':
        return r
    weak_state = (
        float(r.get('rank_score', 999.0)) <= FLAT_MAX_RANK_SCORE
        and float(r.get('reliability_score', 999.0)) <= FLAT_MAX_RELIABILITY_SCORE
    )
    if not weak_state:
        return r

    z = _path_slice(r, feature_cache)
    if z is None or len(z) == 0:
        return r
    fav = _favorable_path(z, r)
    trigger_idx = np.flatnonzero(np.asarray(fav) >= FLAT_TRIGGER_R)
    if len(trigger_idx) == 0:
        return r

    # Realize 25% of original size at +0.75R, leave 75% on the exact frozen
    # exit path. This monetizes the structural 0R dead-zone without tightening
    # the runner stop, shortening holding time, or deleting the right tail.
    # Conservatively, no extra risk capacity is credited after the partial.
    out = r.copy()
    old_gross = float(r.get('gross_r', r.get('net_r', 0.0) + r.get('cost_r', 0.0)))
    new_gross = FLAT_PARTIAL_FRACTION * FLAT_TRIGGER_R + (1.0 - FLAT_PARTIAL_FRACTION) * old_gross
    cost = float(r.get('cost_r', 0.0))
    out['gross_r'] = float(new_gross)
    out['net_r'] = float(new_gross - cost)
    out['giveback_r'] = float(max(0.0, float(r.get('mfe_r', np.nan)) - out['net_r'])) if pd.notna(r.get('mfe_r', np.nan)) else r.get('giveback_r', np.nan)
    out['r5_5_partial_trigger_time'] = pd.Timestamp(z.loc[int(trigger_idx[0]), 'timestamp'])
    out['r5_5_partial_fraction'] = FLAT_PARTIAL_FRACTION
    out['r5_5_partial_trigger_r'] = FLAT_TRIGGER_R
    out['r5_5_management_repaired'] = True
    out['r5_5_original_exit_reason'] = str(r.get('exit_reason', ''))
    out['exit_reason'] = 'R5_5_WEAK_STATE_PARTIAL_25_AT_075R__' + str(r.get('exit_reason', ''))
    return out


def apply_management(cands, feature_cache):
    if cands.empty:
        return cands.copy()
    rows = [_apply_management_row(r, feature_cache) for _, r in cands.iterrows()]
    out = pd.DataFrame(rows)
    if 'r5_5_management_repaired' not in out.columns:
        out['r5_5_management_repaired'] = False
    out['r5_5_management_repaired'] = out['r5_5_management_repaired'].fillna(False).astype(bool)
    return out


def replay_r5(cands, specs, start=100.0, feature_cache=None):
    managed = apply_management(cands, feature_cache)
    return base.replay_r5(managed, specs, start, feature_cache)
