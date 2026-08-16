#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

R54_END=337.4231242104393
EPS=1e-12

def num(x,d=np.nan):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def mask_rule(df,rule):
    if df.empty:return pd.Series(False,index=df.index)
    def one(field,direction,threshold):
        if field=='decision_hour':s=pd.to_datetime(df.decision_time,utc=True).dt.hour.astype(float)
        elif field in df:s=pd.to_numeric(df[field],errors='coerce')
        else:return pd.Series(False,index=df.index)
        return s.le(float(threshold)) if direction=='le' else s.ge(float(threshold))
    m=one(str(rule['field1']),str(rule['dir1']),num(rule['threshold1']))
    f2=str(rule.get('field2','') or '')
    if f2 and f2.lower()!='nan':m=m & one(f2,str(rule['dir2']),num(rule['threshold2']))
    return m.fillna(False)

class EntryGateVariant:
    def __init__(self,rule,promote,base):self.rule=rule;self.promote=promote;self.base=base
    def promote_for_week(self,cands,week_start):
        pr,conf,stats=self.promote(cands,week_start,'combined')
        if len(pr):
            m=mask_rule(pr,self.rule)
            pr=pr.loc[~m].copy();pr['r5_5_entry_gate']=self.rule.get('rule_id','LANE2B')
        return pr,conf,stats
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):return self.base.replay_r5(cands,specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--lane2a-status',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    src=json.load(open(a.lane2a_status));passing=int(src.get('passing_rules',0))
    if passing<1:
        status={'state':'R5_5_LANE2B_NOT_RUN_NO_PASSING_ENTRY_GATE','passing_rules':passing,'next_lane':'ASSET_STRATEGY_REGIME_THESIS_REPAIR','promotion_candidate':False,'promotion_blockers':['NO_CLEAN_ENTRY_GATE','NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED'],'governance':'No selection-frozen entry gate is forced into authentic replay without first8 discovery plus later pre-Aug13 validation. Aug13-14 is excluded from selection.'}
        (out/'R5_5_LANE2B_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));return
    if src.get('revision')!='FORENSIC_COHORT_RECONCILED_FRESH_QUARANTINE_V3':raise RuntimeError('Lane2B requires clean V3 Lane2A status')
    rule=dict(src['best_rule']);rule['rule_id']='|'.join([str(rule.get('target')),str(rule.get('rule_type')),str(rule.get('field1')),str(rule.get('dir1')),str(rule.get('threshold1')),str(rule.get('field2','')),str(rule.get('dir2','')),str(rule.get('threshold2',''))])
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane2b_ctl')
    ctl,CW,CEV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9 or len(CEV)!=1102:raise RuntimeError('R5.4 control reproduction failed')
    cand,VW,VEV,VDE,VPR=eval_variant(EntryGateVariant(rule,promote,base),data,cands,opps,fcache,specs,100.0)
    end_ok=float(cand['end_equity'])>=float(ctl['end_equity'])-1e-9
    verification_delta=float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct']);verification_ok=verification_delta>=-1e-9
    dd_ok=float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-1e-9
    losses_ok=int(cand['losses'])<int(ctl['losses']);flats_ok=int(cand['flat'])<=int(ctl['flat'])
    winrate_ok=float(cand['nonflat_win_rate'])>=float(ctl['nonflat_win_rate'])-1e-12
    routes_ok=(scope['approved_routes']==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510 and len(data)==34)
    asset_ok=int(cand['assets_with_selected'])>=int(ctl['assets_with_selected'])
    gates={'end_equity_not_regressed':end_ok,'aug13_14_verification_not_regressed':verification_ok,'drawdown_not_worse':dd_ok,'losses_reduced':losses_ok,'flats_not_worse':flats_ok,'nonflat_win_rate_not_worse':winrate_ok,'selected_asset_coverage_not_worse':asset_ok,'full_universe_34x510':routes_ok}
    pre_forward=all(gates.values())
    status={'state':'R5_5_LANE2B_AUTHENTIC_ENTRY_GATE_COMPLETE','evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY','selected_lane2a_rule':rule,'r5_4_control':ctl,'candidate':cand,'delta_end_equity':float(cand['end_equity'])-float(ctl['end_equity']),'aug13_14_verification_delta_pct':verification_delta,'gates':gates,'pre_forward_candidate':bool(pre_forward),'promotion_candidate':False,'promotion_blockers':['NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED'],'approved_scope':scope,'governance':'The entry gate was selected using first8 discovery and later pre-Aug13 validation only. Aug13-14 is read here only as an already-inspected verification sample and cannot promote R5.5. The entry gate acts only on promoted candidates using decision-time fields; R5.4 reliability selection and replay logic otherwise remain unchanged. Final promotion requires a new unseen forward holdout and target-server-certified broker degradation.'}
    (out/'R5_5_LANE2B_STATUS.json').write_text(json.dumps(status,indent=2,default=str));VW.to_csv(out/'R5_5_LANE2B_WEEKLY.csv',index=False);print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
