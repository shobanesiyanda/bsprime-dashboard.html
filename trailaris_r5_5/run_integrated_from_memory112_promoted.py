#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

CONTROL = {
    'end_equity': 1136.4556497760627,
    'compounded_return_pct': 1036.4556497760627,
    'executed_trades': 2152,
    'wins': 856,
    'losses': 505,
    'flats': 791,
    'nonflat_win_rate': 0.6289493019838354,
    'profit_factor': 2.791840637059962,
    'max_weekly_drawdown_pct': -2.9475910077805922,
    'max_event_equity_drawdown_pct': -3.1202037867421484,
    'positive_weeks': 26,
    'rolling_11w_min_return_pct': 151.2997031258902,
    'rolling_11w_median_return_pct': 175.621313340644,
    'rolling_11w_mean_return_pct': 176.56265438902557,
    'rolling_11w_max_return_pct': 212.75588274483837,
}

MEMORY112 = {
    'end_equity': 1272.6945454530814,
    'compounded_return_pct': 1172.6945454530814,
    'executed_trades': 2206,
    'wins': 879,
    'losses': 505,
    'flats': 822,
    'nonflat_win_rate': 0.6351156069364162,
    'profit_factor': 2.837039669341938,
    'max_weekly_drawdown_pct': -2.669798710057736,
    'max_event_equity_drawdown_pct': -2.669798710057736,
    'positive_weeks': 26,
    'rolling_11w_min_return_pct': 157.34514876285473,
    'rolling_11w_median_return_pct': 209.09129682460488,
    'rolling_11w_mean_return_pct': 203.3551308151562,
    'rolling_11w_max_return_pct': 222.81395939432213,
}


def rolling(W: pd.DataFrame, n: int = 11) -> pd.DataFrame:
    r = pd.to_numeric(W.weekly_return_pct, errors='coerce').fillna(0).to_numpy(float)
    rows = []
    for i in range(len(W) - n + 1):
        z = r[i:i+n]
        ret = (np.prod(1 + z/100) - 1) * 100
        rows.append({
            'window': i + 1,
            'start_week': str(W.week_start.iloc[i]),
            'end_week': str(W.week_start.iloc[i+n-1]),
            'compounded_return_pct': float(ret),
            'mean_week_pct': float(np.mean(z)),
            'median_week_pct': float(np.median(z)),
            'positive_weeks': int((z > 0).sum()),
            'weeks_ge5': int((z >= 5).sum()),
            'weeks_ge10': int((z >= 10).sum()),
            'max_weekly_dd_pct': float(pd.to_numeric(W.max_drawdown_pct.iloc[i:i+n], errors='coerce').min()),
        })
    return pd.DataFrame(rows)


def _build_component(job):
    import trailaris_r4_full_universe_loop as r4
    return r4.build_asset_components(job)


def load_feature_cache(rawdir: Path, routes: list[str]) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    # Exact frozen engine semantics: build the same per-asset feature frame used
    # by R5_BASE_CAUSAL_ENGINE.build_universe(), but do not regenerate candidates
    # or factors because the exact memory112 promotion estate is already frozen.
    jobs = [(asset, str(rawdir / f'{asset}_M1_normalized.csv')) for asset in routes]
    for asset, p in jobs:
        if not Path(p).exists():
            raise RuntimeError(f'missing preserved M1 path: {asset}')
    cache = {}
    workers = min(8, max(2, os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_build_component, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            asset, _cands, xf, _opps = fut.result()
            if xf is None or len(xf) == 0:
                raise RuntimeError(f'empty exact R4 feature frame: {asset}')
            xf = xf.copy()
            xf['timestamp'] = pd.to_datetime(xf['timestamp'], utc=True, errors='coerce')
            xf = xf.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            need = {'timestamp','high','low','close'}
            missing = need - set(xf.columns)
            if missing:
                raise RuntimeError(f'{asset} exact R4 feature frame missing {sorted(missing)}')
            cache[str(asset)] = xf
    if set(cache) != set(routes):
        raise RuntimeError(f'exact R4 feature cache gate failed: {len(cache)}/34')
    raw_end = max(pd.Timestamp(x.timestamp.iloc[-1]) for x in cache.values())
    return cache, raw_end


def metrics(EV: pd.DataFrame, DE: pd.DataFrame, W: pd.DataFrame, roll: pd.DataFrame) -> dict:
    wins = int((EV.net_pnl > 0).sum())
    losses = int((EV.net_pnl < 0).sum())
    flats = int((EV.net_pnl == 0).sum())
    gw = float(EV.loc[EV.net_pnl > 0, 'net_pnl'].sum())
    gl = float(-EV.loc[EV.net_pnl < 0, 'net_pnl'].sum())
    eq = pd.concat([pd.Series([100.0]), pd.to_numeric(EV.sort_values('timestamp').equity, errors='coerce').dropna()], ignore_index=True)
    dd = (eq / eq.cummax() - 1) * 100
    return {
        'end_equity': float(W.end_equity.iloc[-1]),
        'compounded_return_pct': float((W.end_equity.iloc[-1] / 100 - 1) * 100),
        'evaluated_weeks': int(len(W)),
        'positive_weeks': int((W.weekly_return_pct > 0).sum()),
        'negative_weeks': int((W.weekly_return_pct < 0).sum()),
        'mean_week_pct': float(W.weekly_return_pct.mean()),
        'median_week_pct': float(W.weekly_return_pct.median()),
        'weeks_ge5': int((W.weekly_return_pct >= 5).sum()),
        'weeks_ge10': int((W.weekly_return_pct >= 10).sum()),
        'executed_trades': int(len(EV)),
        'wins': wins,
        'losses': losses,
        'flats': flats,
        'loss_rate': float(losses / len(EV)) if len(EV) else 0.0,
        'flat_rate': float(flats / len(EV)) if len(EV) else 0.0,
        'nonflat_win_rate': float(wins / (wins + losses)) if wins + losses else 0.0,
        'profit_factor': float(gw / gl) if gl else 999.0,
        'max_weekly_drawdown_pct': float(pd.to_numeric(W.max_drawdown_pct, errors='coerce').min()),
        'max_event_equity_drawdown_pct': float(dd.min()) if len(dd) else 0.0,
        'assets_with_selected': int(DE.loc[DE.selected, 'asset'].nunique()) if len(DE) and DE.selected.any() else 0,
        'rolling_11w_windows': int(len(roll)),
        'rolling_11w_min_return_pct': float(roll.compounded_return_pct.min()),
        'rolling_11w_median_return_pct': float(roll.compounded_return_pct.median()),
        'rolling_11w_upper_quartile_return_pct': float(roll.compounded_return_pct.quantile(.75)),
        'rolling_11w_mean_return_pct': float(roll.compounded_return_pct.mean()),
        'rolling_11w_max_return_pct': float(roll.compounded_return_pct.max()),
    }


def delta(candidate: dict, benchmark: dict) -> dict:
    keys = [
        'end_equity','compounded_return_pct','executed_trades','wins','losses','flats',
        'nonflat_win_rate','profit_factor','max_weekly_drawdown_pct','max_event_equity_drawdown_pct',
        'positive_weeks','rolling_11w_min_return_pct','rolling_11w_median_return_pct',
        'rolling_11w_mean_return_pct','rolling_11w_max_return_pct'
    ]
    return {k: float(candidate[k] - benchmark[k]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rawdir', required=True)
    ap.add_argument('--specs', required=True)
    ap.add_argument('--promoted', required=True)
    ap.add_argument('--factory-audit', required=True)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = ['trailaris_r4','trailaris_r5_4','trailaris_r5_4/reliability_variants','trailaris_r5_5']
    import trailaris_r4_full_universe_loop as r4
    import longcycle_integrated_negative_refinement as mod

    fa = pd.read_csv(a.factory_audit)
    if len(fa) != 510 or int(fa.strategy_family.nunique()) != 15:
        raise RuntimeError(f'factory hard lock failed: cells={len(fa)} families={fa.strategy_family.nunique()}')

    PR0 = pd.read_csv(a.promoted)
    for c in ('entry_time','decision_time','exit_time','eval_week'):
        PR0[c] = pd.to_datetime(PR0[c], utc=True, errors='coerce')
    PR0 = PR0.dropna(subset=['entry_time','decision_time','exit_time','eval_week']).copy()
    weeks = sorted(pd.Timestamp(x) for x in PR0.eval_week.unique())
    if len(PR0) != 3417:
        raise RuntimeError(f'exact memory112 promotion estate changed: {len(PR0)} != 3417')
    if len(weeks) != 26:
        raise RuntimeError(f'exact memory112 week estate changed: {len(weeks)} != 26')
    if int(PR0.asset.nunique()) != 34:
        raise RuntimeError(f'exact memory112 route estate changed: {PR0.asset.nunique()} != 34')
    routes = list(r4.ROUTES)
    if len(routes) != 34 or set(PR0.asset.unique()) != set(routes):
        raise RuntimeError('memory112 promoted routes do not match frozen 34-route registry')

    specs = r4.load_specs(Path(a.specs), 'research-proxy')
    fcache, raw_end = load_feature_cache(Path(a.rawdir), routes)

    equity = 100.0
    Wrows, events, decisions, promoted_kept, rejected_gate = [], [], [], [], []
    for wk in weeks:
        src = PR0[PR0.eval_week == wk].copy()
        kept, rejected = mod.entry_gate(src)
        kept = kept.drop(columns=['eval_week'], errors='ignore')
        rejected = rejected.drop(columns=['eval_week'], errors='ignore')
        if len(kept):
            ev, de = mod.replay_r5(kept, specs, equity, fcache)
            if len(ev):
                equity = float(ev.equity.iloc[-1])
            events.append(ev.assign(eval_week=wk))
            decisions.append(de.assign(eval_week=wk))
            promoted_kept.append(kept.assign(eval_week=wk))
        else:
            ev = pd.DataFrame(); de = pd.DataFrame()
        if len(rejected):
            rejected_gate.append(rejected.assign(eval_week=wk))

        prev = Wrows[-1]['end_equity'] if Wrows else 100.0
        end_ts = min(wk + pd.Timedelta(days=7), raw_end)
        w = r4.weekly(ev, start=prev, feature_cache=fcache, start_ts=wk, end_ts=end_ts)
        if len(w):
            rec = w.iloc[0].to_dict(); equity = float(rec['end_equity'])
        else:
            rec = dict(week_start=wk,start_equity=prev,end_equity=prev,weekly_return_pct=0.0,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.0,ge5=False,ge10=False)
            equity = float(prev)
        rec['memory112_promoted_before_gate'] = int(len(src))
        rec['integrated_gate_rejected'] = int(len(rejected))
        rec['integrated_promoted_after_gate'] = int(len(kept))
        Wrows.append(rec)

    W = pd.DataFrame(Wrows)
    EV = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    DE = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    PR = pd.concat(promoted_kept, ignore_index=True) if promoted_kept else pd.DataFrame()
    RJ = pd.concat(rejected_gate, ignore_index=True) if rejected_gate else pd.DataFrame()
    roll = rolling(W)
    cand = metrics(EV, DE, W, roll)

    mgmt = EV.get('r5_5_management_repaired', pd.Series(False, index=EV.index)).fillna(False).astype(bool) if len(EV) else pd.Series(dtype=bool)
    repair_counts = EV.loc[mgmt, 'exit_reason'].value_counts().to_dict() if len(EV) and mgmt.any() else {}
    gate_counts = RJ.get('r5_5_negative_edge_gate_reason', pd.Series(dtype=str)).value_counts().to_dict() if len(RJ) else {}
    status = {
        'state': 'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_V2_26W_REPLAY_COMPLETE',
        'source_promotion_estate': 'EXACT_R5_5_MEMORY112_PROMOTED_3417',
        'source_promotion_rows': int(len(PR0)),
        'source_promotion_weeks': int(len(weeks)),
        'feature_cache_source': 'EXACT_R4_BUILD_ASSET_COMPONENTS_FROM_PRESERVED_34_ROUTE_M1_CACHE',
        'scope': {'routes':34,'strategy_families':15,'route_strategy_cells':510,'weeks':26},
        'frozen_control': CONTROL,
        'memory112_component_leader': MEMORY112,
        'candidate': cand,
        'delta_candidate_minus_frozen_r5_4': delta(cand, CONTROL),
        'delta_candidate_minus_memory112': delta(cand, MEMORY112),
        'integrated_counters': {
            'promoted_before_gate': int(len(PR0)),
            'entry_gate_rejected': int(len(RJ)),
            'entry_gate_reasons': {str(k): int(v) for k,v in gate_counts.items()},
            'promoted_after_gate': int(len(PR)),
            'management_repaired_selected_trades': int(mgmt.sum()) if len(mgmt) else 0,
            'management_repair_exit_reasons': {str(k): int(v) for k,v in repair_counts.items()},
        },
        'causality': 'Entry exclusion uses decision-time fields only. Flat partial realization uses entry-time rank/reliability and the exact contemporaneous R4 feature path to detect first +0.75R reach; final MFE/MAE are not eligibility inputs. Exit timing remains frozen and no extra capacity is credited after the partial.',
        'automatic_promotion': False,
        'evidence_class': 'FULL_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE',
    }

    W.to_csv(out/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_WEEKLY.csv', index=False)
    roll.to_csv(out/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_ROLLING_11W.csv', index=False)
    EV.to_csv(out/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_EVENTS.csv', index=False)
    DE.to_csv(out/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_DECISIONS.csv', index=False)
    PR.to_csv(out/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_PROMOTED.csv', index=False)
    RJ.to_csv(out/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_GATE_REJECTED.csv', index=False)
    (out/'R5_5_INTEGRATED_NEGATIVE_REFINEMENT_STATUS.json').write_text(json.dumps(status, indent=2, default=str))
    print(json.dumps(status, indent=2, default=str))

if __name__ == '__main__':
    main()
