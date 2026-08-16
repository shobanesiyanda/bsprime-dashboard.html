#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import trailaris_r4_full_universe_loop as r4
import variant_hierarchical_gate as gate
from precision_experiment_harness import eval_variant
from mean_reversion_management_experiments import VARIANTS,transform,payoff_stats
from non_regression_gate import evaluate,BASE

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy');rows=[]
    for name,cfg in VARIANTS.items():
        cz=transform(cands,fcache,cfg);res,W,EV,DE,PR=eval_variant(gate,data,cz,opps,fcache,specs);res['variant']='hierarchical_gate_plus_'+name;res['mre']=payoff_stats(EV)
        grosswin=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum()) if len(EV) else 0.;grossloss=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum()) if len(EV) else 0.
        strict=evaluate({'end_equity':res['end_equity'],'mean_week_pct':res['mean_week_pct'],'median_week_pct':res['median_week_pct'],'weeks_ge5':res['weeks_ge5'],'weeks_ge10':res['weeks_ge10'],'max_weekly_dd_pct':res['max_weekly_dd_pct'],'fresh_return_pct':res['fresh_return_pct'],'fresh_max_dd_pct':res['fresh_max_dd_pct'],'assets_with_selected':res['assets_with_selected'],'losses':res['losses'],'nonflat_win_rate':res['nonflat_win_rate'],'profit_factor':grosswin/grossloss if grossloss else 999.})
        res['profit_factor']=grosswin/grossloss if grossloss else 999.;res['strict_nonregression']=strict['behavioral_non_regression_pass'];res['promotion_allowed']=strict['promotion_allowed'];rows.append(res);W.to_csv(out/f'{res["variant"]}_weekly.csv',index=False)
    flat=[]
    for r in rows:
        q={k:v for k,v in r.items() if k!='mre'};q.update({f'mre_{k}':v for k,v in r['mre'].items()});flat.append(q)
    df=pd.DataFrame(flat).sort_values(['promotion_allowed','end_equity','fresh_return_pct'],ascending=[False,False,False]);df.to_csv(out/'R5_4_HYBRID_EXPERIMENTS.csv',index=False);s={'variants_tested':len(df),'strict_promotion_candidates':int(df.promotion_allowed.sum()),'best':df.iloc[0].to_dict()};(out/'R5_4_HYBRID_STATUS.json').write_text(json.dumps(s,indent=2,default=str));print(json.dumps(s,indent=2,default=str))
    g=df[df.variant=='hierarchical_gate_plus_baseline'].iloc[0];assert abs(float(g.end_equity)-337.4231242104393)<=1e-9
if __name__=='__main__':main()
