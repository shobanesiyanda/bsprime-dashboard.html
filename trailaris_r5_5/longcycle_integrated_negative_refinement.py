from __future__ import annotations
import numpy as np
import pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import longcycle_bounded_memory_candidate as bounded

# Integrated R5.5 negative-side refinement.
# All production decisions use only information available at decision/management time.
# Post-event MFE/MAE were used only to diagnose the structural management defects.
LOSS_GATE_MAX_QUALITY = 0.76
LOSS_GATE_MAX_ASSET_STRATEGY_MEAN_R = 0.05

DEVELOPMENT_TRIGGER_R = 0.50
DEVELOPMENT_FLOOR_R = -0.50

FLAT_TRIGGER_R = 0.75
FLAT_FLOOR_R = 0.15
FLAT_MAX_RANK_SCORE = 0.42
FLAT_MAX_RELIABILITY_SCORE = 0.0


def promote_for_week(cands, week_start):
    p, conf, stats = bounded.promote_for_week(cands, week_start)
    if p.empty:
        return p, conf, stats
    # Exact 26-week forensic population showed this broad causal state was
    # negative expectancy (PF < 1) and unstable across weeks. This is an
    # entry-time gate, not an outcome/MFE gate.
    bad = (
        (pd.to_numeric(p['quality'], errors='coerce') <= LOSS_GATE_MAX_QUALITY)
        & (pd.to_numeric(p['reliability_asset_strategy_mean_r'], errors='coerce')
           <= LOSS_GATE_MAX_ASSET_STRATEGY_MEAN_R)
    )
    p = p.loc[~bad].copy()
    p['r5_5_negative_edge_gate_pass'] = True
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


def _r_paths(z, r):
    ent = float(r.entry_price)
    sd = max(float(r.stop_distance), 1e-12)
    dr = int(r.direction)
    hi = pd.to_numeric(z['high'], errors='coerce').to_numpy(float)
    lo = pd.to_numeric(z['low'], errors='coerce').to_numpy(float)
    cl = pd.to_numeric(z['close'], errors='coerce').to_numpy(float)
    if dr == 1:
        fav = (hi - ent) / sd
        adv = (lo - ent) / sd
        close_r = (cl - ent) / sd
    else:
        fav = (ent - lo) / sd
        adv = (ent - hi) / sd
        close_r = (ent - cl) / sd
    return fav, adv, close_r


def _first_index_ge(a, threshold):
    idx = np.flatnonzero(np.asarray(a) >= float(threshold))
    return int(idx[0]) if len(idx) else None


def _first_index_le_after(a, threshold, start):
    if start is None:
        return None
    arr = np.asarray(a)
    idx = np.flatnonzero(arr[int(start):] <= float(threshold))
    return int(start + idx[0]) if len(idx) else None


def _apply_management_row(r, feature_cache):
    # Only the protected-runner population produced the structural flat state.
    if str(r.get('management', '')) != 'PROTECTED_RUNNER':
        return r

    z = _path_slice(r, feature_cache)
    if z is None or len(z) < 2:
        return r
    fav, adv, close_r = _r_paths(z, r)

    weak_flat_state = (
        float(r.get('rank_score', 999.0)) <= FLAT_MAX_RANK_SCORE
        and float(r.get('reliability_score', 999.0)) <= FLAT_MAX_RELIABILITY_SCORE
    )

    # Soft protection after genuine development: once +0.50R has printed,
    # a weak-state campaign is no longer permitted to decay all the way to -1R.
    # Protection becomes active from the NEXT M1 bar to avoid same-bar
    # high/low ordering assumptions.
    dev_i = _first_index_ge(fav, DEVELOPMENT_TRIGGER_R) if weak_flat_state else None
    dev_stop_i = _first_index_le_after(
        adv, DEVELOPMENT_FLOOR_R, None if dev_i is None else dev_i + 1
    )

    # Structural flat repair: the frozen engine protects at +0.75R but leaves
    # the floor at exactly 0 until +1.50R. In low-rank/nonpositive-reliability
    # states, ratchet the protected floor to +0.15R from the next M1 bar.
    flat_i = _first_index_ge(fav, FLAT_TRIGGER_R) if weak_flat_state else None
    flat_stop_i = _first_index_le_after(
        adv, FLAT_FLOOR_R, None if flat_i is None else flat_i + 1
    )

    chosen_i = None
    chosen_r = None
    chosen_reason = None

    # If +0.75R occurs before a soft -0.50R stop, the stronger +0.15R floor
    # supersedes the development floor.
    if flat_i is not None and flat_stop_i is not None:
        if dev_stop_i is None or flat_i < dev_stop_i:
            chosen_i = flat_stop_i
            chosen_r = FLAT_FLOOR_R
            chosen_reason = 'R5_5_WEAK_STATE_PROTECTED_PLUS_015R'
    if chosen_i is None and dev_stop_i is not None:
        chosen_i = dev_stop_i
        chosen_r = DEVELOPMENT_FLOOR_R
        chosen_reason = 'R5_5_DEVELOPED_LOSS_FLOOR_MINUS_050R'

    if chosen_i is None:
        return r

    # Never move an exit later than the frozen candidate exit.
    t = pd.Timestamp(z.loc[chosen_i, 'timestamp'])
    if t >= pd.Timestamp(r.exit_time):
        return r

    out = r.copy()
    ent = float(r.entry_price); sd = max(float(r.stop_distance), 1e-12); dr = int(r.direction)
    out['exit_time'] = t
    out['exit_price'] = ent + dr * float(chosen_r) * sd
    out['gross_r'] = float(chosen_r)
    out['net_r'] = float(chosen_r) - float(r.get('cost_r', 0.0))
    out['mfe_r'] = float(np.nanmax(fav[:chosen_i+1]))
    out['mae_r'] = float(max(0.0, -np.nanmin(adv[:chosen_i+1])))
    out['giveback_r'] = float(max(0.0, out['mfe_r'] - out['net_r']))
    out['exit_reason'] = chosen_reason
    out['r5_5_management_repaired'] = True
    return out


def _apply_management(cands, feature_cache):
    if cands.empty:
        return cands
    rows = [_apply_management_row(r, feature_cache) for _, r in cands.iterrows()]
    out = pd.DataFrame(rows)
    if 'r5_5_management_repaired' not in out.columns:
        out['r5_5_management_repaired'] = False
    out['r5_5_management_repaired'] = out['r5_5_management_repaired'].fillna(False).astype(bool)
    return out


def replay_r5(cands, specs, start=100.0, feature_cache=None):
    managed = _apply_management(cands, feature_cache)
    return base.replay_r5(managed, specs, start, feature_cache)
