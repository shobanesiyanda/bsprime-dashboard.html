#!/usr/bin/env python3
from pathlib import Path
import json,pandas as pd
ROUTES=['XAUUSD','XAGUSD','EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
BROKERS=['EXNESS_RAW_SPREAD_MT5','IC_MARKETS_RAW_SPREAD_MT5','PEPPERSTONE_RAZOR_MT5','FP_MARKETS_RAW_MT5','HFM_ZERO_MT5','TICKMILL_RAW_MT5','DERIV_MT5_SYNTHETIC']
SYN={'V75','V50','V100','V25','V10'}
FIELDS=['broker_symbol','account_type','server','capture_start','capture_end','sample_n','trade_lot','spread_cost_median_usd','spread_cost_p95_usd','commission_round_turn_usd','slippage_mean_usd','slippage_p95_usd','financing_long_usd','financing_short_usd','contract_size','tick_size','tick_value','min_lot','lot_step','max_lot','margin_required_usd','leverage','stop_level','freeze_level','execution_mode','fill_rate','reject_rate','latency_median_ms','latency_p95_ms','session_rules','weekend_rules','spec_hash']
rows=[]
for a in ROUTES:
  for b in BROKERS:
    native=(b=='DERIV_MT5_SYNTHETIC' and a in SYN) or (b!='DERIV_MT5_SYNTHETIC' and a not in SYN)
    r={'route':a,'broker_id':b,'scope_state':'IN_SCOPE_238_CELL_ESTATE','candidate_native':native,'available':False,'captured_from_target_server':False,'broker_certified':False,'certification_state':'TARGET_SERVER_EVIDENCE_REQUIRED' if native else 'NON_NATIVE_BROKER_ROUTE_GATED'}
    for c in FIELDS:r[c]=''
    rows.append(r)
df=pd.DataFrame(rows);Path('trailaris_r5_4').mkdir(exist_ok=True);df.to_csv('trailaris_r5_4/R5_4_BROKER_ROUTE_CERTIFICATION_238.csv',index=False)
s={'routes':len(ROUTES),'brokers':len(BROKERS),'cells':len(df),'native_candidate_cells':int(df.candidate_native.sum()),'gated_non_native_cells':int((~df.candidate_native).sum()),'certified_cells':0,'selection_state':'FAIL_CLOSED_UNTIL_TARGET_SERVER_EVIDENCE'}
Path('trailaris_r5_4/R5_4_BROKER_ROUTE_CERTIFICATION_STATUS.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));assert len(df)==238 and df.route.nunique()==34 and df.broker_id.nunique()==7
