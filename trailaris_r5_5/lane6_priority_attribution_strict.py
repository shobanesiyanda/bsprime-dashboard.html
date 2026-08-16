#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant
from lane4_market_state_entry_repair import enrich_market_state
from lane6_path_specific_attribution import FIELDS,test1,test2

R54_END=337.4231242104393
INC_END=343.5447309255116
FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
EPS=1e-12
B3={'behavior':'close_a0.50_r0.20_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.20,'bars':1,'gate_metric':'reliability_win','gate_direction':'le','gate_threshold':0.2655132009803008}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_l6prio_ctl');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 reproduction failed')
    inc,W,EV,DE,PR=eval_variant(ManagedVariant(B3,base,promote),data,cands,opps,fcache,specs,100.0)
    if abs(float(inc['end_equity'])-INC_END)>1e-6:raise RuntimeError('Lane1B2 reproduction failed')
    z=enrich_market_state(EV,fcache);z['decision_time']=pd.to_datetime(z.decision_time,utc=True);z['eval_week']=pd.to_datetime(z.eval_week,utc=True)
    weeks=sorted(z.eval_week.unique());cut=weeks[-3];early=z[z.eval_week<cut].copy();valid=z[(z.eval_week>=cut)&(z.decision_time<FRESH0)].copy();fresh=z[z.decision_time>=FRESH0].copy()
    disc_pnl=pd.to_numeric(early.net_pnl,errors='coerce').fillna(0);disc_nr=pd.to_numeric(early.net_r,errors='coerce').fillna(0);disc_winner_nr=disc_nr[disc_pnl>EPS]
    floor=float(disc_winner_nr.median()) if len(disc_winner_nr) else .5
    for q in [z,early,valid,fresh]:
        pnl=pd.to_numeric(q.net_pnl,errors='coerce').fillna(0);nr=pd.to_numeric(q.net_r,errors='coerce').fillna(0);q['priority_label']=(pnl>EPS)&(nr>=floor);q['priority_utility']=np.where(q.priority_label,nr,0.)
    fields=[f for f in FIELDS if f in z.columns];one=test1(early,valid,'priority_label','priority_utility',fields);two=test2(early,valid,'priority_label','priority_utility',one);R=pd.concat([x for x in [one,two] if len(x)],ignore_index=True,sort=False) if len(one) or len(two) else pd.DataFrame()
    if len(R):
        R['score']=8*R.valid_target_lift+4*R.disc_target_lift+3*np.log1p(R.valid_n)+np.log1p(R.disc_n)+4*np.maximum(R.valid_mean_utility,0)+2*np.maximum(R.disc_mean_utility,0);R=R.sort_values(['clean_pass','score','valid_target_lift','valid_mean_utility'],ascending=[False,False,False,False]).reset_index(drop=True);R.to_csv(out/'R5_5_LANE6_PRIORITY_CAPTURE_RULES_STRICT.csv',index=False)
    best=R[R.clean_pass].iloc[0].to_dict() if len(R) and R.clean_pass.any() else (R.iloc[0].to_dict() if len(R) else {})
    s={'state':'R5_5_LANE6_PRIORITY_ATTRIBUTION_DISCOVERY_FROZEN_COMPLETE','evidence_class':'DISCOVERY_DEFINED_TARGET_PLUS_PRE_AUG13_VALIDATION','scope':{'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'priority_net_r_floor':floor,'priority_floor_source':'DISCOVERY_WINNERS_ONLY','discovery_target_rate':float(early.priority_label.mean()),'validation_target_rate':float(valid.priority_label.mean()),'fresh_target_rate_diagnostic_only':float(fresh.priority_label.mean()),'rules_tested':int(len(R)),'passing_rules':int(R.clean_pass.sum()) if len(R) else 0,'best_pre_aug13_rule':best,'fresh_outcomes_used_for_selection':False,'governance':'The economic priority threshold is frozen from discovery winners only. Validation outcomes do not define the target or thresholds. Aug13-14 is diagnostic only. Any nominated priority rule must return to full 34x15x510 authentic replay.'}
    (out/'R5_5_LANE6_PRIORITY_STRICT_STATUS.json').write_text(json.dumps(s,indent=2,default=str));print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
