#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane4_market_state_entry_repair import enrich_market_state,scope_mask

R54_END=337.4231242104393
EPS=1e-9

def num(x,d=np.nan):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def rule_mask(z,r):
    m=scope_mask(z,str(r['scope_type']),str(r['scope_value']))
    s=pd.to_numeric(z[str(r['feature1'])],errors='coerce');m &= s.le(num(r['threshold1'])) if str(r['dir1'])=='le' else s.ge(num(r['threshold1']))
    f2=str(r.get('feature2','') or '')
    if f2 and f2.lower()!='nan':
        s2=pd.to_numeric(z[f2],errors='coerce');m &= s2.le(num(r['threshold2'])) if str(r['dir2'])=='le' else s2.ge(num(r['threshold2']))
    return m.fillna(False)

class MarketStateVariant:
    def __init__(self,rule,promote,base,fcache):self.rule=rule;self.promote=promote;self.base=base;self.fcache=fcache;self.excluded=0
    def promote_for_week(self,cands,week_start):
        pr,conf,stats=self.promote(cands,week_start,'combined')
        if len(pr):
            z=enrich_market_state(pr,self.fcache);m=rule_mask(z,self.rule);self.excluded+=int(m.sum());pr=z.loc[~m].copy();pr['r5_5_market_state_gate']=self.rule.get('rule_id','LANE4B')
        return pr,conf,stats
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):return self.base.replay_r5(cands,specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--lane4a-status',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    src=json.load(open(a.lane4a_status));passing=int(src.get('passing_rules',0))
    if passing<1:
        s={'state':'R5_5_LANE4B_NOT_RUN_NO_CLEAN_MARKET_STATE_GATE','passing_rules':passing,'pre_forward_candidate':False,'promotion_candidate':False,'promotion_blockers':['NO_CLEAN_MARKET_STATE_ENTRY_GATE','NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED']};(out/'R5_5_LANE4B_STATUS.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));return
    rule=dict(src['best_rule']);rule['rule_id']='|'.join([str(rule.get('target')),str(rule.get('scope_type')),str(rule.get('scope_value')),str(rule.get('feature1')),str(rule.get('dir1')),str(rule.get('threshold1')),str(rule.get('feature2','')),str(rule.get('dir2','')),str(rule.get('threshold2',''))])
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    baseline_families=set(cands.strategy_family.astype(str).dropna().unique())
    if len(data)!=34:raise RuntimeError(f'actual asset universe regressed {len(data)}/34')
    if len(baseline_families)!=15:raise RuntimeError(f'actual strategy-family universe regressed {len(baseline_families)}/15')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane4b_ctl');ctl,CW,CEV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>EPS or len(CEV)!=1102:raise RuntimeError('R5.4 control reproduction failed')
    mod=MarketStateVariant(rule,promote,base,fcache);cand,VW,VEV,VDE,VPR=eval_variant(mod,data,cands,opps,fcache,specs,100.0)
    verification_delta=float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct'])
    repair=(int(cand['losses'])<=int(ctl['losses']) and int(cand['flat'])<=int(ctl['flat']) and (int(cand['losses'])<int(ctl['losses']) or int(cand['flat'])<int(ctl['flat'])))
    gates={'end_equity_not_regressed':float(cand['end_equity'])>=float(ctl['end_equity'])-EPS,'aug13_14_verification_not_regressed':verification_delta>=-EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-EPS,'loss_flat_repair':repair,'nonflat_win_rate_not_worse':float(cand['nonflat_win_rate'])>=float(ctl['nonflat_win_rate'])-1e-12,'selected_asset_coverage_not_worse':int(cand['assets_with_selected'])>=int(ctl['assets_with_selected']),'actual_strategy_family_universe_preserved':len(baseline_families)==15,'full_universe_34x510':len(data)==34 and scope['approved_routes']==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510}
    pre=all(gates.values())
    blockers=['NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED'] if pre else ['PRE_FORWARD_NON_REGRESSION_GATE_FAILED']
    s={'state':'R5_5_LANE4B_AUTHENTIC_MARKET_STATE_GATE_COMPLETE','evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY','selected_rule':rule,'r5_4_control':ctl,'candidate':cand,'excluded_promotions':int(mod.excluded),'delta_end_equity':float(cand['end_equity'])-float(ctl['end_equity']),'aug13_14_verification_delta_pct':verification_delta,'actual_family_count':len(baseline_families),'gates':gates,'pre_forward_candidate':bool(pre),'promotion_candidate':False,'promotion_blockers':blockers,'approved_scope':scope,'governance':'The market-state gate was selected only on first8 discovery plus later pre-Aug13 validation and uses only completed decision-bar state. The complete actual 34-asset and 15-family research universe is mandatory. Aug13-14 is already-inspected verification only. R5.4 selection/replay logic otherwise remains unchanged. Final promotion requires a new unseen forward holdout and target-server-certified broker degradation.'}
    (out/'R5_5_LANE4B_STATUS.json').write_text(json.dumps(s,indent=2,default=str));VW.to_csv(out/'R5_5_LANE4B_WEEKLY.csv',index=False);print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
