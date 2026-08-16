#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from lane1b_full_replay import ManagedVariant
from lane5_pareto_management_stack import StackVariant
from lane6_targeted_causal_winner_replay import (
    CausalWinnerVariant,load_rule,B3,LANE5_KEYS,R54_END,B3_END,LANE5_END,
    LANE5_LOSSES,LANE5_FLATS,LANE5_NFWR,FRESH_FLOOR,DD_FLOOR,FRESH_DD_FLOOR,EPS
)
from research_universe_cache import load_or_build

PATHS=[
 ('AMPLIFY','R5_5_LANE6_AMPLIFICATION_RULES.csv',dict(copies=1,quality_boost=.04,runner_frac=.25,target_r=5.,trail_gap=.8,max_extra=36)),
 ('EXTEND','R5_5_LANE6_PAYOFF_EXTENSION_RULES.csv',dict(copies=1,quality_boost=.04,runner_frac=.25,target_r=5.,trail_gap=.8,max_extra=36)),
 ('PRIORITY','R5_5_LANE6_PRIORITY_CAPTURE_RULES.csv',dict(copies=1,quality_boost=.04,runner_frac=.25,target_r=5.,trail_gap=.8,max_extra=36)),
]

def gates(x,scope_ok):
    return {
      'end_equity_above_lane5':float(x['end_equity'])>LANE5_END+EPS,
      'fresh_return_strictly_above':float(x['fresh_return_pct'])>FRESH_FLOOR+EPS,
      'max_weekly_dd_not_worse':float(x['max_weekly_dd_pct'])>=DD_FLOOR-EPS,
      'fresh_dd_not_worse':float(x['fresh_max_dd_pct'])>=FRESH_DD_FLOOR-EPS,
      'losses_at_or_below_228':int(x['losses'])<=LANE5_LOSSES,
      'flats_at_or_below_421':int(x['flat'])<=LANE5_FLATS,
      'nonflat_win_rate_at_least_lane5':float(x['nonflat_win_rate'])>=LANE5_NFWR-EPS,
      'assets_with_selected_at_least_33':int(x['assets_with_selected'])>=33,
      'full_universe_34x15x510':bool(scope_ok)
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--rule-dir',required=True);ap.add_argument('--snapshot',default='');ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);rd=Path(a.rule_dir)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    snap=Path(a.snapshot) if a.snapshot else None
    data,v,cands,opps,fcache=load_or_build(base,Path(a.rawdir),snap);specs=r4.load_specs(Path(a.specs),'research-proxy');scope=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'))['scope']
    scope_ok=(len(data)==34 and int(scope.get('approved_routes',0))==34 and int(scope.get('strategy_families',0))==15 and int(scope.get('route_strategy_cells',0))==510 and bool(scope.get('full_universe_required',True)))
    if not scope_ok:raise RuntimeError(f'full-universe scope failure data={len(data)} scope={scope}')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_l6pathmatrix_ctl');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.)
    if abs(float(ctl['end_equity'])-R54_END)>EPS:raise RuntimeError(f'R5.4 reproduction failed {ctl}')
    b3,*_=eval_variant(ManagedVariant(B3,base,promote),data,cands,opps,fcache,specs,100.)
    if abs(float(b3['end_equity'])-B3_END)>1e-6:raise RuntimeError(f'B3 reproduction failed {b3}')
    lane5,*_=eval_variant(StackVariant(LANE5_KEYS,base,promote),data,cands,opps,fcache,specs,100.)
    if abs(float(lane5['end_equity'])-LANE5_END)>1e-6 or int(lane5['losses'])!=LANE5_LOSSES or int(lane5['flat'])!=LANE5_FLATS:raise RuntimeError(f'Lane5 reproduction failed {lane5}')
    rows=[];details={}
    for mode,fn,params in PATHS:
        rule,passing=load_rule(rd/fn,0)
        standalone_var=CausalWinnerVariant(mode,rule,base,promote,fcache,'B3',**params)
        standalone,*_=eval_variant(standalone_var,data,cands,opps,fcache,specs,100.)
        fusion_var=CausalWinnerVariant(mode,rule,base,promote,fcache,'LANE5',**params)
        fusion,*_=eval_variant(fusion_var,data,cands,opps,fcache,specs,100.)
        g=gates(fusion,scope_ok)
        row={'mode':mode,'rules_file':fn,'passing_rules':passing,'standalone_end_equity':standalone['end_equity'],'standalone_fresh_return_pct':standalone['fresh_return_pct'],'fusion_end_equity':fusion['end_equity'],'fusion_delta_vs_lane5':float(fusion['end_equity'])-LANE5_END,'fusion_fresh_return_pct':fusion['fresh_return_pct'],'fusion_fresh_delta':float(fusion['fresh_return_pct'])-FRESH_FLOOR,'wins':fusion['wins'],'losses':fusion['losses'],'flat':fusion['flat'],'nonflat_win_rate':fusion['nonflat_win_rate'],'max_weekly_dd_pct':fusion['max_weekly_dd_pct'],'fresh_max_dd_pct':fusion['fresh_max_dd_pct'],'assets_with_selected':fusion['assets_with_selected'],'frontier_pass':bool(all(g.values()))}
        rows.append(row);details[mode]={'winner_rule':rule,'parameters':params,'standalone':standalone,'fusion':fusion,'frontier_gates':g,'runtime_counts':{'standalone_qualifying_promoted':standalone_var.qualifying_promoted,'standalone_extra_tranches_created':standalone_var.extra_tranches_created,'standalone_extensions_attempted':standalone_var.extensions_attempted,'fusion_qualifying_promoted':fusion_var.qualifying_promoted,'fusion_extra_tranches_created':fusion_var.extra_tranches_created,'fusion_extensions_attempted':fusion_var.extensions_attempted}}
    df=pd.DataFrame(rows).sort_values(['frontier_pass','fusion_fresh_return_pct','fusion_end_equity'],ascending=[False,False,False]);df.to_csv(out/'R5_5_LANE6_PATH_TARGETED_MATRIX.csv',index=False)
    leader=df.iloc[0].to_dict() if len(df) else {}
    status={'state':'R5_5_LANE6_PATH_TARGETED_MATRIX_COMPLETE','evidence_class':'SINGLE_BUILD_POST_RELIABILITY_PRE_SELECTION_PATH_SPECIFIC_FULL_PORTFOLIO_REPLAYS','r5_4_control':ctl,'lane1b2_control':b3,'certified_lane5_control':lane5,'paths':details,'leader':leader,'frontier_candidates':int(df.frontier_pass.sum()) if len(df) else 0,'scope':scope,'universe_snapshot_used':bool(snap and snap.exists()),'next_action':('RUN_PARAMETER_ROBUSTNESS_AROUND_FRONTIER_LEADER' if len(df) and bool(df.frontier_pass.any()) else 'USE_LANE6E_FRESH_STATE_DIAGNOSIS_FOR_CAUSAL_REPAIR'),'governance':'Each path uses its own pre-Aug13 path-specific rule, not a generic winner rule. Winner qualification occurs only after exact causal weekly promotion/reliability scoring and before portfolio selection. The universe is built once and shared across all path tests; the reusable snapshot caches only build_universe outputs and changes compute efficiency only, never market logic. The first pass uses one conservative mechanism setting per path. Parameter expansion is allowed only around a path that shows causal value.'}
    (out/'R5_5_LANE6_PATH_TARGETED_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
