#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

R54_END=337.4231242104393
TARGETS={'SUPPLY_DEMAND_SWEEP_BOS','MEAN_REVERSION_EXTREME'}
VARIANTS=[
 {'id':'BOTH_STRICT','sd':'strict','mr':'strict'},
 {'id':'SD_STRICT_MR_BALANCED','sd':'strict','mr':'balanced'},
 {'id':'SD_BALANCED_MR_STRICT','sd':'balanced','mr':'strict'},
 {'id':'BOTH_BALANCED','sd':'balanced','mr':'balanced'},
 {'id':'SD_STRICT_ONLY','sd':'strict','mr':'off'},
 {'id':'MR_STRICT_ONLY','sd':'off','mr':'strict'},
]

def load_csv(path):
    p=Path(path)
    if not p.exists() or p.stat().st_size==0:return pd.DataFrame()
    try:z=pd.read_csv(p)
    except pd.errors.EmptyDataError:return pd.DataFrame()
    if len(z):
        for c in ['decision_time','entry_time','exit_time']:
            if c in z:z[c]=pd.to_datetime(z[c],utc=True)
        if 'is_addon' in z:z['is_addon']=z.is_addon.astype(str).str.lower().isin(['true','1','yes'])
    return z

def stats(z):
    out=[]
    if z.empty:return out
    for s,g in z.groupby('strategy'):
        out.append({'strategy':s,'candidates':len(g),'mean_net_r':float(pd.to_numeric(g.net_r,errors='coerce').mean()),'median_net_r':float(pd.to_numeric(g.net_r,errors='coerce').median()),'win_rate':float((pd.to_numeric(g.net_r,errors='coerce')>0).mean())})
    return out

class Variant:
    def __init__(self,base,promote):self.base=base;self.promote=promote
    def promote_for_week(self,cands,week_start):return self.promote(cands,week_start,'combined')
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):return self.base.replay_r5(cands,specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--candidate-dir',required=True);ap.add_argument('--variant-index',type=int,required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    cfg=VARIANTS[a.variant_index];cd=Path(a.candidate_dir)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    baseline_families=set(cands.strategy_family.astype(str).dropna().unique())
    if len(data)!=34:raise RuntimeError(f'actual asset universe regressed {len(data)}/34')
    if len(baseline_families)!=15:raise RuntimeError(f'actual baseline strategy-family universe regressed {len(baseline_families)}/15')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_lane3v2_ctl_{a.variant_index}')
    ctl,CW,CEV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9 or int(ctl['executed_trades'])!=1102:raise RuntimeError('R5.4 control reproduction failed')
    parts=[]
    if cfg['sd']!='off':parts.append(load_csv(cd/f"sd_{cfg['sd']}.csv"))
    if cfg['mr']!='off':parts.append(load_csv(cd/f"mr_{cfg['mr']}.csv"))
    rebuilt=pd.concat([x for x in parts if len(x)],ignore_index=True,sort=False) if any(len(x) for x in parts) else pd.DataFrame()
    other=cands[~cands.strategy.isin(TARGETS)].copy()
    c2=pd.concat([other,rebuilt],ignore_index=True,sort=False) if len(rebuilt) else other
    c2['decision_time']=pd.to_datetime(c2.decision_time,utc=True);c2['exit_time']=pd.to_datetime(c2.exit_time,utc=True);c2=c2.sort_values('decision_time').reset_index(drop=True)
    candidate_families=set(c2.strategy_family.astype(str).dropna().unique())
    family_universe_preserved=(candidate_families==baseline_families and len(candidate_families)==15)
    cand,VW,VEV,VDE,VPR=eval_variant(Variant(base,promote),data,c2,opps,fcache,specs,100.0)
    end_delta=float(cand['end_equity'])-float(ctl['end_equity']);fresh_delta=float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct'])
    gates={'end_equity_not_regressed':end_delta>=-1e-9,'fresh_holdout_not_regressed':fresh_delta>=-1e-9,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-1e-9,'losses_not_worse':int(cand['losses'])<=int(ctl['losses']),'flats_not_worse':int(cand['flat'])<=int(ctl['flat']),'actual_strategy_family_universe_preserved':family_universe_preserved,'full_universe_34x510':len(data)==34 and scope['approved_routes']==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510}
    status={'state':'R5_5_LANE3_STRATEGY_REBUILD_V2_FULL_REPLAY_COMPLETE','evidence_class':'END_TO_END_CAUSAL_REBUILT_STRATEGY_REPLAY','variant_index':a.variant_index,'config':cfg,'r5_4_control':ctl,'candidate':cand,'delta_end_equity':end_delta,'fresh_delta_pct':fresh_delta,'rebuilt_candidates':int(len(rebuilt)),'rebuilt_raw_stats':stats(rebuilt),'actual_family_count':len(candidate_families),'actual_family_universe_preserved':family_universe_preserved,'gates':gates,'candidate_pass':all(gates.values()),'approved_scope':scope,'governance':'The two negative playbooks are rebuilt only inside the R5.5 candidate. Every full-system candidate must preserve the actual 34-asset research universe and the complete 15-family strategy universe; family-off variants are diagnostics only and fail the promotion gate. Precomputed rebuilt candidates are immutable inputs to each replay. R5.4 files, reliability promotion, portfolio constraints and replay logic remain unchanged.'}
    (out/f'variant_{a.variant_index:02d}.json').write_text(json.dumps(status,indent=2,default=str));VW.to_csv(out/f'variant_{a.variant_index:02d}_weekly.csv',index=False);print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
