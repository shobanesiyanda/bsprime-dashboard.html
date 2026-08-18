#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from trailaris_50_asset.batch_consistent_drawdown import batch_consistent_global_event_drawdown


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--status',type=Path,required=True)
    ap.add_argument('--events',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=None)
    a=ap.parse_args()
    out=a.out or a.status
    status=json.loads(a.status.read_text())
    ev=pd.read_csv(a.events)
    g=batch_consistent_global_event_drawdown(ev,float(status.get('candidate',{}).get('start_balance',100.0)))
    c=status.setdefault('candidate',{})
    previous=c.get('batch_consistent_event_drawdown_pct')
    if previous is not None:
        c['week_local_mark_to_market_drawdown_pct']=float(previous)
    c['batch_consistent_global_event_drawdown_pct']=float(g['max_event_equity_drawdown_pct'])
    c['max_event_equity_drawdown_pct']=float(g['max_event_equity_drawdown_pct'])
    c['batch_consistent_global_event_drawdown_evidence']=g
    status['drawdown_reporting']='AUTHORITATIVE FULL-PERIOD DD = BATCH-CONSISTENT REPLAY EVENT ROW ORDER / balance_after / GLOBAL PEAK-TO-TROUGH. WEEK-LOCAL MARK-TO-MARKET AND LEGACY TIMESTAMP-RESORTED VALUES ARE DIAGNOSTICS ONLY.'
    status['drawdown_reporting_corrected']=True
    status['quant_v2_modified_by_drawdown_fix']=False
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(status,indent=2,default=str))
    print(json.dumps({'state':'DRAWDOWN_STATUS_CORRECTED','status':str(out),'global_event_drawdown_pct':g['max_event_equity_drawdown_pct'],'event_rows':g['event_rows'],'method':g['method']},indent=2))

if __name__=='__main__':main()
