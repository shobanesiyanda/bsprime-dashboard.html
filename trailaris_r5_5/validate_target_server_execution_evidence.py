#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    'route','broker_id','candidate_scope_eligible','availability_state','available',
    'captured_from_target_server','broker_certified','certification_state','broker_symbol',
    'account_type','server','capture_start','capture_end','sample_n','trade_lot',
    'spread_cost_median_usd','spread_cost_p95_usd','commission_round_turn_usd',
    'slippage_mean_usd','slippage_p95_usd','financing_long_usd','financing_short_usd',
    'contract_size','tick_size','tick_value','min_lot','lot_step','max_lot',
    'margin_required_usd','leverage','stop_level','freeze_level','execution_mode',
    'fill_rate','reject_rate','latency_median_ms','latency_p95_ms','session_rules',
    'weekend_rules','spec_hash'
]
NUMERIC_REQUIRED = [
    'sample_n','trade_lot','spread_cost_median_usd','spread_cost_p95_usd',
    'commission_round_turn_usd','slippage_mean_usd','slippage_p95_usd',
    'financing_long_usd','financing_short_usd','contract_size','tick_size','tick_value',
    'min_lot','lot_step','max_lot','margin_required_usd','leverage','stop_level',
    'freeze_level','fill_rate','reject_rate','latency_median_ms','latency_p95_ms'
]
TEXT_REQUIRED = [
    'broker_symbol','account_type','server','capture_start','capture_end','execution_mode',
    'session_rules','weekend_rules','spec_hash'
]
BOOL_TRUE = ['available','captured_from_target_server','broker_certified']


def as_bool(x):
    if isinstance(x, bool): return x
    return str(x).strip().lower() in {'true','1','yes','y'}


def nonempty(x):
    return pd.notna(x) and str(x).strip() not in {'','nan','None','null'}


def finite_nonnegative(x):
    try:
        v=float(x)
        return np.isfinite(v) and v >= 0
    except Exception:
        return False


def validate_cell(row: pd.Series) -> list[str]:
    problems=[]
    for c in BOOL_TRUE:
        if not as_bool(row.get(c)): problems.append(f'{c}=false')
    if str(row.get('availability_state','')).strip().upper() in {'','UNVERIFIED_TARGET_SERVER','UNKNOWN'}:
        problems.append('availability_state_unverified')
    for c in TEXT_REQUIRED:
        if not nonempty(row.get(c)): problems.append(f'{c}=missing')
    for c in NUMERIC_REQUIRED:
        if not finite_nonnegative(row.get(c)): problems.append(f'{c}=missing_or_invalid')
    try:
        if float(row.get('sample_n',0)) <= 0: problems.append('sample_n_not_positive')
    except Exception: pass
    try:
        fr=float(row.get('fill_rate')); rr=float(row.get('reject_rate'))
        if not (0 <= fr <= 1): problems.append('fill_rate_out_of_range')
        if not (0 <= rr <= 1): problems.append('reject_rate_out_of_range')
    except Exception: pass
    try:
        if float(row.get('spread_cost_p95_usd')) < float(row.get('spread_cost_median_usd')):
            problems.append('spread_p95_below_median')
        if float(row.get('slippage_p95_usd')) < float(row.get('slippage_mean_usd')):
            problems.append('slippage_p95_below_mean')
        if float(row.get('latency_p95_ms')) < float(row.get('latency_median_ms')):
            problems.append('latency_p95_below_median')
    except Exception: pass
    # The spec hash must be an actual immutable hash, not a label.
    h=str(row.get('spec_hash','')).strip().lower()
    if nonempty(h) and not (len(h)==64 and all(ch in '0123456789abcdef' for ch in h)):
        problems.append('spec_hash_not_sha256')
    return problems


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--require-pass',action='store_true')
    args=ap.parse_args()
    inp=Path(args.input); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(inp)
    missing=[c for c in REQUIRED_COLUMNS if c not in d.columns]
    if missing: raise SystemExit(f'missing required columns: {missing}')
    eligible=d[d['candidate_scope_eligible'].map(as_bool)].copy()
    rows=[]
    for i,r in eligible.iterrows():
        problems=validate_cell(r)
        rows.append({'row_index':int(i),'route':r['route'],'broker_id':r['broker_id'],'pass':not problems,'problems':'|'.join(problems)})
    audit=pd.DataFrame(rows)
    if len(audit): audit.to_csv(out/'TARGET_SERVER_CELL_AUDIT.csv',index=False)
    passed=audit[audit['pass']] if len(audit) else pd.DataFrame(columns=['route','broker_id'])
    routes=sorted(map(str,d['route'].dropna().unique()))
    passed_routes=sorted(map(str,passed['route'].dropna().unique())) if len(passed) else []
    missing_routes=sorted(set(routes)-set(passed_routes))
    state='TARGET_SERVER_EXECUTION_EVIDENCE_PASS' if len(routes)==34 and not missing_routes else 'FAIL_CLOSED_UNTIL_TARGET_SERVER_EVIDENCE'
    status={
        'state':state,
        'input':str(inp),
        'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
        'routes_total':len(routes),
        'eligible_cells':int(len(eligible)),
        'cells_passing_complete_empirical_evidence_contract':int(len(passed)),
        'routes_with_at_least_one_passing_broker_cell':len(passed_routes),
        'routes_without_passing_broker_cell':missing_routes,
        'advertised_spread_is_certification':False,
        'target_server_capture_required':True,
        'quant_v2_rules_changed':False,
        'production_promotion_allowed': state=='TARGET_SERVER_EXECUTION_EVIDENCE_PASS',
        'note':'This validator checks completeness and internal consistency of supplied target-server evidence. It does not manufacture broker evidence or infer it from marketing pages.'
    }
    (out/'TARGET_SERVER_EXECUTION_EVIDENCE_STATUS.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))
    if args.require_pass and state!='TARGET_SERVER_EXECUTION_EVIDENCE_PASS':
        raise SystemExit(2)

if __name__=='__main__': main()
