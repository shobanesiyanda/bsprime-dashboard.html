#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import trailaris_r4_full_universe_loop as r4
from precision_experiment_harness import eval_variant
from non_regression_gate import evaluate,BASE
import variant_baseline
import variant_hierarchical_gate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--base-specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    raw=Path(a.rawdir);spec=pd.read_csv(a.base_specs);xau=pd.read_csv(raw/'XAUUSD_M1_normalized.csv',usecols=['close']);max_px=float(pd.to_numeric(xau.close,errors='coerce').max())
    # HFM Cent research-only XAU profile from current official published contract terms:
    # 1 lot = 1 ounce; tick 0.01; tick value USD 0.01/lot; min/step 0.01;
    # published margin requirement 0.05%. Fixed replay margin uses the maximum observed
    # XAU price in the historical window, making the proxy conservative for this test.
    m=spec.asset.eq('XAUUSD');assert m.sum()==1
    spec.loc[m,'contract_size']=1.;spec.loc[m,'tick_size']=.01;spec.loc[m,'tick_value']=.01;spec.loc[m,'volume_min']=.01;spec.loc[m,'volume_step']=.01;spec.loc[m,'margin_per_lot']=max_px*.0005
    spec.loc[m,'evidence_state']='RESEARCH_PROXY_HFM_CENT_XAU_PUBLISHED_TERMS_NOT_TARGET_SERVER_CERTIFIED'
    sp=out/'HFM_CENT_XAU_RESEARCH_SPECS.csv';spec.to_csv(sp,index=False)
    data,v,cands,opps,fcache=base.build_universe(raw);specs=r4.load_specs(sp,'research-proxy')
    rows=[]
    for name,mod in [('baseline_logic_hfm_cent_xau',variant_baseline),('hierarchical_gate_hfm_cent_xau',variant_hierarchical_gate)]:
        res,W,EV,DE,PR=eval_variant(mod,data,cands,opps,fcache,specs);grosswin=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum()) if len(EV) else 0.;grossloss=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum()) if len(EV) else 0.;pf=grosswin/grossloss if grossloss else 999.
        q=evaluate({'end_equity':res['end_equity'],'mean_week_pct':res['mean_week_pct'],'median_week_pct':res['median_week_pct'],'weeks_ge5':res['weeks_ge5'],'weeks_ge10':res['weeks_ge10'],'max_weekly_dd_pct':res['max_weekly_dd_pct'],'fresh_return_pct':res['fresh_return_pct'],'fresh_max_dd_pct':res['fresh_max_dd_pct'],'assets_with_selected':res['assets_with_selected'],'losses':res['losses'],'nonflat_win_rate':res['nonflat_win_rate'],'profit_factor':pf})
        xsel=DE[(DE.asset=='XAUUSD')&(DE.selected)] if len(DE) else pd.DataFrame();xev=EV[EV.asset=='XAUUSD'] if len(EV) else pd.DataFrame()
        res.update(variant=name,profit_factor=pf,xau_selected=int(len(xsel)),xau_executed=int(len(xev)),xau_net_pnl=float(xev.net_pnl.sum()) if len(xev) else 0.,xau_lots=float(xev.lot.sum()) if len(xev) else 0.,strict_nonregression=q['behavioral_non_regression_pass'],promotion_allowed=q['promotion_allowed']);rows.append(res);W.to_csv(out/f'{name}_weekly.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(out/'R5_4_HFM_CENT_XAU_EXPERIMENTS.csv',index=False)
    s={'evidence_class':'RESEARCH_PROXY_HFM_CENT_XAU_PUBLISHED_TERMS_NOT_TARGET_SERVER_CERTIFIED','xau_contract_size_oz':1.0,'xau_tick_size':.01,'xau_tick_value_usd_per_lot':.01,'xau_min_lot':.01,'xau_margin_rate_proxy':.0005,'max_xau_price_used_for_margin':max_px,'variants':df.to_dict('records'),'any_34_asset_execution':bool((df.assets_with_selected>=34).any()),'any_strict_promotion':bool(df.promotion_allowed.any())};(out/'R5_4_HFM_CENT_XAU_STATUS.json').write_text(json.dumps(s,indent=2,default=str));print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
