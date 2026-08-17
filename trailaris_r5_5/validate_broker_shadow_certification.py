#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEDGERS = {
    'signals':'signal_ledger.csv',
    'orders':'order_request_ledger.csv',
    'acks':'broker_acknowledgement_ledger.csv',
    'fills':'fill_and_reject_ledger.csv',
    'costs':'slippage_and_latency_ledger.csv',
    'management':'management_and_exit_ledger.csv',
    'pnl':'pnl_reconciliation.csv',
}
COMMON = ['event_id','route']
REQUIRED = {
    'signals':['event_id','route','signal_time','direction','candidate_rules_commit'],
    'orders':['event_id','route','order_request_time','requested_volume','requested_order_type'],
    'acks':['event_id','route','broker_id','server','ack_time','broker_order_id','ack_state'],
    'fills':['event_id','route','fill_state','fill_time','requested_volume','filled_volume','fill_price','reject_reason'],
    'costs':['event_id','route','spread_cost_usd','commission_usd','slippage_usd','latency_ms'],
    'management':['event_id','route','management_start_time','exit_time','exit_reason','exit_price'],
    'pnl':['event_id','route','model_net_pnl_usd','broker_net_pnl_usd','reconciliation_delta_usd'],
}
FROZEN='194031e05594e6921b5410e3eb328a5fb53fde94'


def nonempty(x): return pd.notna(x) and str(x).strip() not in {'','nan','None','null'}
def numeric(x):
    try: return np.isfinite(float(x))
    except Exception: return False


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dir',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--require-pass',action='store_true'); args=ap.parse_args()
    root=Path(args.dir); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    loaded={}; problems=[]; hashes={}
    for key,name in LEDGERS.items():
        p=root/name
        if not p.exists():
            problems.append(f'missing_ledger:{name}'); continue
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest(); d=pd.read_csv(p); loaded[key]=d
        miss=[c for c in REQUIRED[key] if c not in d.columns]
        if miss: problems.append(f'{name}:missing_columns:{"|".join(miss)}')
        if 'event_id' in d.columns and d.event_id.astype(str).duplicated().any(): problems.append(f'{name}:duplicate_event_id')
    if len(loaded)==len(LEDGERS):
        base_ids=set(loaded['signals'].event_id.astype(str))
        if not base_ids: problems.append('signal_ledger_empty')
        for key,d in loaded.items():
            ids=set(d.event_id.astype(str));
            if ids != base_ids: problems.append(f'{LEDGERS[key]}:event_id_set_mismatch')
        sig=loaded['signals']
        if 'candidate_rules_commit' in sig.columns and not sig.candidate_rules_commit.astype(str).eq(FROZEN).all(): problems.append('candidate_rules_commit_drift')
        ack=loaded['acks']
        for c in ['broker_id','server','broker_order_id','ack_state']:
            if c in ack.columns and not ack[c].map(nonempty).all(): problems.append(f'acks:{c}_missing')
        fills=loaded['fills']
        if {'fill_state','filled_volume','requested_volume'}.issubset(fills.columns):
            fs=fills.fill_state.astype(str).str.upper(); fv=pd.to_numeric(fills.filled_volume,errors='coerce'); rv=pd.to_numeric(fills.requested_volume,errors='coerce')
            if ((fs=='FILLED') & ((fv<=0)|fv.isna())).any(): problems.append('filled_event_without_positive_volume')
            if ((fs=='REJECTED') & fills.get('reject_reason',pd.Series(index=fills.index,dtype=object)).map(nonempty).eq(False)).any(): problems.append('rejected_event_without_reason')
            if (fv.fillna(0) > rv.fillna(0)+1e-12).any(): problems.append('filled_volume_exceeds_requested')
        costs=loaded['costs']
        for c in ['spread_cost_usd','commission_usd','slippage_usd','latency_ms']:
            if c in costs.columns and not costs[c].map(numeric).all(): problems.append(f'costs:{c}_non_numeric')
        pnl=loaded['pnl']
        if {'model_net_pnl_usd','broker_net_pnl_usd','reconciliation_delta_usd'}.issubset(pnl.columns):
            m=pd.to_numeric(pnl.model_net_pnl_usd,errors='coerce'); b=pd.to_numeric(pnl.broker_net_pnl_usd,errors='coerce'); r=pd.to_numeric(pnl.reconciliation_delta_usd,errors='coerce')
            if ((b-m-r).abs()>1e-8).any(): problems.append('pnl_reconciliation_identity_failed')
    state='BROKER_SHADOW_CERTIFICATION_EVIDENCE_PASS' if not problems else 'FAIL_CLOSED_UNTIL_COMPLETE_SHADOW_EVIDENCE'
    status={
        'state':state,'candidate_rules_commit':FROZEN,'ledger_hashes':hashes,'problems':problems,
        'required_chain':'signal -> order request -> broker acknowledgement -> fill/reject -> costs/latency -> management/exit -> P&L reconciliation',
        'real_target_server_evidence_required':True,'demo_or_non_risk_shadow_is_acceptable_for_preproduction_certification':True,
        'quant_v2_rules_changed':False,'production_promotion_allowed':state=='BROKER_SHADOW_CERTIFICATION_EVIDENCE_PASS'
    }
    (out/'BROKER_SHADOW_CERTIFICATION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
    if args.require_pass and problems: raise SystemExit(2)

if __name__=='__main__': main()
