#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import lane1_reliability_gated_repair as core

LOCKED_END=337.4231242104393
LOCKED_TRADES=1102
EPS=1e-12
FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy');mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane1d_control')
    res,W,EV,DE,PR=eval_variant(mod,data,cands,opps,fcache,specs,100.0)
    if abs(float(res['end_equity'])-LOCKED_END)>1e-9 or len(EV)!=LOCKED_TRADES:raise RuntimeError('exact R5.4 control reproduction failed')
    EV=EV.copy();EV['eval_week']=pd.to_datetime(EV.eval_week,utc=True);EV['decision_time']=pd.to_datetime(EV.decision_time,utc=True);EV['entry_time']=pd.to_datetime(EV.entry_time,utc=True);EV['exit_time']=pd.to_datetime(EV.exit_time,utc=True)
    for c in ['net_pnl','net_r','risk_cash','cost_r','entry_price','stop_distance','direction']:
        EV[c]=pd.to_numeric(EV[c],errors='coerce').fillna(0.0)
    weeks=sorted(EV.eval_week.unique());cut=weeks[-3]
    disc=(EV.eval_week<cut).to_numpy();valid=((EV.eval_week>=cut)&(EV.decision_time<FRESH0)).to_numpy();fresh=(EV.decision_time>=FRESH0).to_numpy()
    if disc.sum()==0 or valid.sum()==0 or fresh.sum()==0:raise RuntimeError(f'bad split disc={disc.sum()} valid={valid.sum()} fresh={fresh.sum()}')
    paths={i:core.path_for(fcache.get(r.asset),r) for i,r in EV.iterrows()}
    behaviors=[]
    for act in [.50,.75]:
      for ret in [.05,.10,.15,.20]:
        for bars in [1,2]:behaviors.append((f'close_a{act:.2f}_r{ret:.2f}_{bars}b','close',act,ret,bars))
    for act in [.50,.75,1.00]:
      for fl in [.02,.05,.10,.15,.20]:behaviors.append((f'lock_a{act:.2f}_f{fl:.2f}','lock',act,fl,1))
    metrics=[c for c in ['reliability_score','reliability_expected_r','reliability_win','reliability_strategy_mean_r','reliability_strategy_win','reliability_asset_strategy_mean_r','reliability_asset_strategy_win','expected_r','trailing_win_rate','stability','trailing_giveback','quality','rank_score'] if c in EV.columns]
    base_r=EV.net_r.to_numpy(float);rows=[]
    for name,kind,act,x,bars in behaviors:
        vals=[];hit=[]
        for i,r in EV.iterrows():
            rr,h=(core.close_rescue(r,paths[i],act,x,bars) if kind=='close' else core.profit_lock(r,paths[i],act,x));vals.append(rr);hit.append(h)
        vals=np.asarray(vals,float);hit=np.asarray(hit,bool)
        for metric in ['ALL',*metrics]:
            gs=[('all',None)] if metric=='ALL' else []
            if metric!='ALL':
                q=pd.to_numeric(EV.loc[disc,metric],errors='coerce').dropna()
                for qt in [.20,.30,.40,.50,.60,.70,.80] if len(q) else []:
                    th=float(q.quantile(qt));gs.extend([('le',th),('ge',th)])
            for direction,th in gs:
                if metric=='ALL':gate=np.ones(len(EV),dtype=bool)
                else:
                    m=pd.to_numeric(EV[metric],errors='coerce').to_numpy(float);gate=(m<=th) if direction=='le' else (m>=th)
                    gate=np.nan_to_num(gate,nan=False).astype(bool)
                gate &= hit;cf=np.where(gate,vals,base_r)
                ds=core.seg_stats(EV.loc[disc],cf[disc]);vs=core.seg_stats(EV.loc[valid],cf[valid])
                discover_ok=ds['delta_usd']>EPS and ds['flat_to_loss']==0 and ds['winner_to_loss']==0 and (ds['losses_improved']>0 or ds['flat_to_profit']>0)
                validate_ok=vs['delta_usd']>=-EPS and vs['flat_to_loss']==0 and vs['winner_to_loss']==0 and (vs['losses_improved']>0 or vs['flat_to_profit']>0)
                rows.append({'behavior':name,'kind':kind,'activate_mfe_r':act,'parameter_r':x,'bars':bars,'gate_metric':metric,'gate_direction':direction,'gate_threshold':th,'applications_pre_fresh':int(gate[~fresh].sum()),'discovery_pass':discover_ok,'validation_pass':validate_ok,**{f'disc_{k}':v for k,v in ds.items()},**{f'valid_{k}':v for k,v in vs.items()}})
    D=pd.DataFrame(rows);D['joint_pass']=D.discovery_pass&D.validation_pass
    D['repair_score']=D.valid_loss_to_nonloss*6+D.valid_losses_improved*2+D.valid_flat_to_profit*1.5+D.disc_loss_to_nonloss*2+D.disc_flat_to_profit-.5*D.disc_winners_harmed-.75*D.valid_winners_harmed
    D=D.sort_values(['joint_pass','repair_score','valid_delta_usd','disc_delta_usd'],ascending=[False,False,False,False]).reset_index(drop=True);D.to_csv(out/'R5_5_LANE1D_CLEAN_MANAGEMENT_RULES.csv',index=False)
    best=D[D.joint_pass].iloc[0].to_dict() if D.joint_pass.any() else (D.iloc[0].to_dict() if len(D) else {})
    status={'state':'R5_5_LANE1D_HOLDOUT_CLEAN_MANAGEMENT_DISCOVERY_COMPLETE','evidence_class':'SELECTION_FROZEN_PRE_FRESH_DISCOVERY_VALIDATION','r5_4_control':res,'scope':{'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'split':{'discovery_events':int(disc.sum()),'validation_pre_fresh_events':int(valid.sum()),'fresh_quarantine_events':int(fresh.sum()),'fresh_quarantine_start':str(FRESH0),'fresh_outcomes_used_for_selection':False},'behaviors':len(behaviors),'gate_metrics':metrics,'rules_tested':int(len(D)),'joint_passing_rules':int(D.joint_pass.sum()),'best_joint_rule':best,'governance':'No Aug13-14 outcome, delta or transition is used to select or rank management rules. Thresholds come from discovery only; later pre-fresh decisions provide validation. The winning rule must be frozen before a full authentic replay reads the quarantined sample. Because Aug13-14 was already inspected in earlier R5.5 development, final promotion still requires a new unseen forward holdout.'}
    (out/'R5_5_LANE1D_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
