#!/usr/bin/env python3
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def _f(v, default=0.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _mark_current_r(row, t, feature_cache):
    x=feature_cache.get(str(row.asset))
    if x is None or len(x)==0:
        return 0.0
    ts=pd.to_datetime(x['timestamp'],utc=True,errors='coerce')
    j=int(ts.searchsorted(pd.Timestamp(t),side='right'))-1
    if j<0:
        return 0.0
    px=_f(x.iloc[j]['close'])
    ent=_f(row.entry_price)
    sd=max(_f(row.stop_distance,1e-12),1e-12)
    dr=int(_f(row.direction,0))
    return (px-ent)*dr/sd-_f(row.get('cost_r',0.0))


def legacy_global_event_drawdown_pct(events, start_equity=100.0):
    """Legacy defective metric retained only for lineage comparison.

    It globally re-sorts week-local replay rows by event timestamp, interleaving
    independently replayed weekly books and thereby corrupting the equity path.
    """
    if events is None or len(events)==0:
        return 0.0
    e=events.copy()
    e['timestamp']=pd.to_datetime(e['timestamp'],utc=True,errors='coerce')
    eq=pd.concat([pd.Series([float(start_equity)]),pd.to_numeric(e.sort_values('timestamp')['equity'],errors='coerce').dropna()],ignore_index=True)
    return float(((eq/eq.cummax())-1.0).min()*100.0)


def batch_consistent_global_event_drawdown(events, start_equity=100.0):
    """Authoritative full-period event drawdown for the sequential replay.

    The replay emits event rows in authoritative book-processing order.  Weekly books
    are themselves executed sequentially and compound into the next week's opening
    balance.  Therefore the full-period peak-to-trough path must preserve emitted row
    order; it must never be globally re-sorted by exit timestamp because exits from
    different week-local books can overlap in calendar time.

    `balance_after` is the realized closed-balance state after each emitted event row.
    This exactly reproduces the corrected 50-asset control geometry (-3.395214...%)
    without altering strategy, ranking, execution, sizing, or asset identity.
    """
    if events is None or len(events)==0:
        return {
            'max_event_equity_drawdown_pct':0.0,
            'peak_path_index':0,
            'trough_path_index':0,
            'peak_balance_usd':float(start_equity),
            'trough_balance_usd':float(start_equity),
            'event_rows':0,
            'method':'BATCH_CONSISTENT_REPLAY_EVENT_ROW_ORDER__BALANCE_AFTER__GLOBAL_PEAK_TO_TROUGH'
        }
    if 'balance_after' not in events.columns:
        raise RuntimeError('authoritative global drawdown requires balance_after event state')
    bal=pd.to_numeric(events['balance_after'],errors='coerce')
    if bal.isna().any():
        raise RuntimeError('non-numeric balance_after in event path')
    path=np.concatenate(([float(start_equity)],bal.to_numpy(dtype=float)))
    peaks=np.maximum.accumulate(path)
    dd=(path/np.maximum(peaks,1e-12)-1.0)*100.0
    trough_i=int(np.argmin(dd))
    peak_i=int(np.argmax(path[:trough_i+1]))
    return {
        'max_event_equity_drawdown_pct':float(dd[trough_i]),
        'peak_path_index':peak_i,
        'trough_path_index':trough_i,
        'peak_balance_usd':float(path[peak_i]),
        'trough_balance_usd':float(path[trough_i]),
        'event_rows':int(len(events)),
        'method':'BATCH_CONSISTENT_REPLAY_EVENT_ROW_ORDER__BALANCE_AFTER__GLOBAL_PEAK_TO_TROUGH'
    }


def batch_consistent_event_drawdown(weekly, events, feature_cache, start_equity=100.0):
    """Week-local mark-to-market drawdown diagnostic; not the full-period headline DD.

    Each evaluated week is a separate replay book. Positions from one replay book may
    have exit timestamps after that calendar week, so globally interleaving event rows
    from different eval_week books is invalid. Inside each book, all exits at the same
    timestamp are accounted as one batch before the equity state is measured.
    """
    if events is None or len(events)==0:
        return {'max_event_equity_drawdown_pct':0.0,'worst_episode':None,'weekly_books':0}
    e=events.copy();w=weekly.copy()
    for c in ('timestamp','entry_time','exit_time','eval_week'):
        if c in e.columns:e[c]=pd.to_datetime(e[c],utc=True,errors='coerce')
    if 'week_start' in w.columns:w['week_start']=pd.to_datetime(w['week_start'],utc=True,errors='coerce')
    worst=0.0;episode=None;books=0
    for wk,g in e.groupby('eval_week',sort=True):
        books+=1;g=g.dropna(subset=['timestamp','entry_time']).copy().sort_values('timestamp')
        z=w.loc[w['week_start'].eq(wk)] if 'week_start' in w.columns else pd.DataFrame()
        start=_f(z.iloc[0]['start_equity'],start_equity) if len(z) else float(start_equity)
        balance=start;peak=start
        for t,closed in g.groupby('timestamp',sort=True):
            balance+=float(pd.to_numeric(closed['net_pnl'],errors='coerce').fillna(0.0).sum())
            opened=g[(g['entry_time']<t)&(g['timestamp']>t)]
            unreal=0.0
            for _,r in opened.iterrows():
                unreal+=_f(r.get('risk_cash',0.0))*_mark_current_r(r,t,feature_cache)
            equity=balance+unreal
            peak=max(peak,equity)
            dd=(equity/max(peak,1e-12)-1.0)*100.0
            if dd<worst:
                worst=float(dd)
                episode={'eval_week':str(wk),'timestamp':str(t),'start_equity_usd':start,'peak_equity_usd':float(peak),'equity_usd':float(equity),'realized_balance_usd':float(balance),'unrealized_pnl_usd':float(unreal),'open_positions_after_exit_batch':int(len(opened))}
    return {'max_event_equity_drawdown_pct':float(worst),'worst_episode':episode,'weekly_books':int(books)}
