#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--status34',type=Path,required=True);ap.add_argument('--status50',type=Path,required=True);ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 s34=json.loads(a.status34.read_text());s50=json.loads(a.status50.read_text());sp=json.loads(a.spec.read_text());m34=s34['candidate'];m50=s50['candidate'];e=s50['expansion16'];g=sp['serious_production_candidate_gate']
 d34=float(m34['batch_consistent_global_event_drawdown_pct']);d50=float(m50['batch_consistent_global_event_drawdown_pct'])
 checks={
  'end_balance_50_gt_34':m50['end_balance']>m34['end_balance'],
  'executed_trades_50_gt_34':m50['executed_trades']>m34['executed_trades'],
  'profit_factor_50_min':m50['profit_factor']>=g['profit_factor_50_min'],
  'profit_factor_50_ratio_to_34_min':m50['profit_factor']/max(m34['profit_factor'],1e-12)>=g['profit_factor_50_ratio_to_34_min'],
  'global_event_drawdown_50_floor_pct':d50>=g['global_event_drawdown_50_floor_pct'],
  'global_event_drawdown_relative':d50>=d34-g['global_event_drawdown_50_not_worse_than_34_by_more_than_percentage_points'],
  'positive_weeks_50_min':m50['positive_weeks']>=g['positive_weeks_50_min'],
  'positive_weeks_relative':m50['positive_weeks']>=m34['positive_weeks']-g['positive_weeks_50_not_lower_than_34_by_more_than'],
  'expansion16_net_pnl_positive':e['net_pnl_usd']>g['expansion16_net_pnl_usd_gt'],
  'expansion16_profit_factor_min':e['profit_factor']>=g['expansion16_profit_factor_min'],
  'executed_trades_50_min':m50['executed_trades']>=g['executed_trades_50_min']
 }
 supported=all(checks.values())
 out={'state':'QV2_50_INDEPENDENT_VALIDATION_GATE_EVALUATED','decision':'PRODUCTION_UNIVERSE_CANDIDATE_SUPPORTED' if supported else 'PRODUCTION_UNIVERSE_CANDIDATE_NOT_YET_SUPPORTED','all_predeclared_checks_pass':supported,'checks':checks,'fresh_34':m34,'fresh_50':m50,'fresh_expansion16':e,'delta_50_minus_34':{'end_balance_usd':m50['end_balance']-m34['end_balance'],'return_percentage_points':m50['compounded_return_pct']-m34['compounded_return_pct'],'executed_trades':m50['executed_trades']-m34['executed_trades'],'profit_factor':m50['profit_factor']-m34['profit_factor'],'nonflat_win_rate_percentage_points':100*(m50['nonflat_win_rate']-m34['nonflat_win_rate']),'global_event_drawdown_percentage_points':d50-d34,'mean_week_percentage_points':m50['mean_week_pct']-m34['mean_week_pct']},'drawdown_authority':'BATCH_CONSISTENT_REPLAY_EVENT_ROW_ORDER__BALANCE_AFTER__GLOBAL_PEAK_TO_TROUGH','strategy_modified':False,'universe_50_modified':False,'automatic_production_promotion':False,'next_gate':'BROKER_AND_ACCOUNT_TYPE_CERTIFICATION' if supported else 'RETAIN_34_AS_RISK_QUALITY_CONTROL_AND_INVESTIGATE_FRESH_DIFFERENCE_WITHOUT_RETROACTIVE_TUNING','evidence_class':'INDEPENDENT_HISTORICAL_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,default=str));print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
