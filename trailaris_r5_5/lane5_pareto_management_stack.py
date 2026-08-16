#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import apply_rule

EPS=1e-9
R54_END=337.4231242104393
INCUMBENT_END=343.5447309255116
INCUMBENT_FRESH=7.261646557142409
INCUMBENT_NFWR=0.6671428571428571
R54_DD=-4.637645574907678
R54_FRESH_DD=-0.5196525640746019

RULES={
 'B3':{'behavior':'close_a0.50_r0.20_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.20,'bars':1,'gate_metric':'reliability_win','gate_direction':'le','gate_threshold':0.2655132009803008},
 'B4':{'behavior':'close_a0.50_r0.05_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.05,'bars':1,'gate_metric':'reliability_asset_strategy_win','gate_direction':'le','gate_threshold':0.3265411458233644},
 'B5':{'behavior':'close_a0.50_r0.05_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.05,'bars':1,'gate_metric':'trailing_win_rate','gate_direction':'le','gate_threshold':0.3254593175853018},
 'B8':{'behavior':'close_a0.50_r0.10_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.10,'bars':1,'gate_metric':'trailing_win_rate','gate_direction':'le','gate_threshold':0.272264631043257},
}
STACKS=[
 ('B3_ONLY',['B3']),
 ('B3_THEN_B4',['B3','B4']),
 ('B3_THEN_B5',['B3','B5']),
 ('B3_THEN_B8',['B3','B8']),
 ('B3_B4_B8',['B3','B4','B8']),
 ('B3_B5_B8',['B3','B5','B8']),
 ('B3_B4_B5',['B3','B4','B5']),
 ('B3_B5_B4',['B3','B5','B4']),
 ('B4_THEN_B3',['B4','B3']),
 ('B5_THEN_B3',['B5','B3']),
 ('B8_THEN_B3',['B8','B3']),
]

def apply_stack(cands,feature_cache,keys):
    z=cands.copy()
    if 'management_rule' not in z.columns:z['management_rule']=pd.Series([None]*len(z),index=z.index,dtype=object)
    for key in keys:
        eligible=z.index[z['management_rule'].isna()]
        if not len(eligible):break
        sub=apply_rule(z.loc[eligible].copy(),feature_cache,RULES[key])
        if 'management_rule' not in sub.columns:continue
        changed=sub.index[sub['management_rule'].notna()]
        if not len(changed):continue
        for col in sub.columns:
            if col not in z.columns:z[col]=pd.NA
        z.loc[changed,sub.columns]=sub.loc[changed,sub.columns]
    return z

class StackVariant:
    def __init__(self,keys,base,promote):self.keys=keys;self.base=base;self.promote=promote
    def promote_for_week(self,cands,week_start):return self.promote(cands,week_start,'combined')
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):return self.base.replay_r5(apply_stack(cands,feature_cache,self.keys),specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--variant-index',type=int,required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args()
    if a.variant_index<0 or a.variant_index>=len(STACKS):raise SystemExit('variant-index outside stack matrix')
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);name,keys=STACKS[a.variant_index]
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    families=sorted(cands.strategy_family.dropna().astype(str).unique().tolist());routes=len(data);cells=routes*len(families)
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_l5_ctl_{a.variant_index}')
    ctl,CW,CEV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>EPS:raise RuntimeError('R5.4 control reproduction failed')
    inc,IW,IEV,IDE,IPR=eval_variant(StackVariant(['B3'],base,promote),data,cands,opps,fcache,specs,100.0)
    if abs(float(inc['end_equity'])-INCUMBENT_END)>1e-6:raise RuntimeError(f'Lane1B2 incumbent reproduction failed: {inc["end_equity"]}')
    cand,VW,VEV,VDE,VPR=eval_variant(StackVariant(keys,base,promote),data,cands,opps,fcache,specs,100.0)
    managed=int(VEV.get('management_rule',pd.Series(index=VEV.index,dtype=object)).notna().sum()) if len(VEV) else 0
    gates={
      'beats_lane1b2_end_equity':float(cand['end_equity'])>INCUMBENT_END+EPS,
      'fresh_return_beats_incumbent':float(cand['fresh_return_pct'])>INCUMBENT_FRESH+EPS,
      'max_weekly_dd_not_worse_than_r54':float(cand['max_weekly_dd_pct'])>=R54_DD-EPS,
      'fresh_dd_not_worse_than_r54':float(cand['fresh_max_dd_pct'])>=R54_FRESH_DD-EPS,
      'losses_below_r54':int(cand['losses'])<=230,
      'flats_below_r54':int(cand['flat'])<=424,
      'nonflat_win_rate_at_least_lane1b2':float(cand['nonflat_win_rate'])>=INCUMBENT_NFWR-EPS,
      'selected_asset_coverage_not_worse_than_r54':int(cand['assets_with_selected'])>=33,
      'actual_full_universe_34x15x510':routes==34 and len(families)==15 and cells==510 and scope['approved_routes']==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510,
    }
    status={
      'state':'R5_5_LANE5_PARETO_MANAGEMENT_STACK_FULL_REPLAY_COMPLETE',
      'evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY',
      'variant_index':a.variant_index,'config_id':name,'stack':keys,'rules':[RULES[k] for k in keys],
      'r5_4_control':ctl,'lane1b2_incumbent_reproduced':inc,'candidate':cand,
      'delta_vs_r54_end_equity':float(cand['end_equity'])-float(ctl['end_equity']),
      'delta_vs_lane1b2_end_equity':float(cand['end_equity'])-INCUMBENT_END,
      'managed_executions':managed,'actual_input_families':families,
      'scope':{'routes':routes,'strategy_families':len(families),'route_strategy_cells':cells},
      'gates':gates,'frontier_pass':bool(all(gates.values())),
      'promotion_candidate':False,
      'promotion_blockers':['NEW_UNSEEN_FORWARD_HOLDOUT_REQUIRED','TARGET_SERVER_BROKER_EXECUTION_CERTIFICATION_REQUIRED'],
      'governance':'Lane 1B-2 rule 3 is reproduced as the incumbent return floor. Secondary repair rules are applied only to executions not already managed by higher-priority rules. Every stack uses the unchanged full 34-route, actual 15-family input universe and locked portfolio replay. A stack must beat the Lane 1B-2 return leader while also beating the R5.4 loss/flat quality floor before it can become a pre-forward challenger.'
    }
    (out/f'variant_{a.variant_index:02d}.json').write_text(json.dumps(status,indent=2,default=str));VW.to_csv(out/f'variant_{a.variant_index:02d}_weekly.csv',index=False);print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
