#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

LOCKED_END=337.4231242104393
LOCKED_TRADES=1102
EPS=1e-12

def num(x,d=0.0):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def path_for(frame,row):
    if frame is None or len(frame)==0:return None
    t=frame['timestamp'];a=pd.Timestamp(row.entry_time);b=pd.Timestamp(row.exit_time)
    lo=int(t.searchsorted(a,side='left'));hi=int(t.searchsorted(b,side='left'))
    return frame.iloc[lo:hi].copy()

def close_rescue(row,frame,activate=.5,retreat=.1,bars=1):
    """Causal completed-bar rescue. MFE activation may occur on the current bar; exit is only at its completed close."""
    base=num(row.net_r);cost=max(0.0,num(row.cost_r));ent=num(row.entry_price);sd=max(abs(num(row.stop_distance)),1e-12);dr=int(num(row.direction,1))
    if frame is None or len(frame)==0:return base,False
    mfe=0.;run=0
    for _,b in frame.iterrows():
        hi=num(b.get('high'));lo=num(b.get('low'));cl=num(b.get('close'));op=num(b.get('open'),cl)
        fav=(hi-ent)/sd if dr==1 else (ent-lo)/sd;mfe=max(mfe,fav)
        adverse=cl<op if dr==1 else cl>op;run=run+1 if adverse else 0
        cr=(cl-ent)*dr/sd-cost
        # Retreat threshold is NET R, so a rescue cannot be called a profit if costs make it negative.
        if mfe+EPS>=activate and cr<=retreat+EPS and cr>0 and run>=bars:return cr,True
    return base,False

def profit_lock(row,frame,activate=.5,floor_net=.05):
    """Activate a protected stop only after a completed bar has reached activation MFE; later bars can hit it.
    Gap-through execution uses the adverse bar open, so this is conservative rather than guaranteed-profit hindsight.
    """
    base=num(row.net_r);cost=max(0.0,num(row.cost_r));ent=num(row.entry_price);sd=max(abs(num(row.stop_distance)),1e-12);dr=int(num(row.direction,1))
    if frame is None or len(frame)==0:return base,False
    active=False;gross_floor=float(floor_net)+cost;stop=ent+dr*gross_floor*sd
    for _,b in frame.iterrows():
        hi=num(b.get('high'));lo=num(b.get('low'));op=num(b.get('open'));cl=num(b.get('close'))
        if active:
            if dr==1 and lo<=stop+EPS:
                px=op if op<stop else stop;return (px-ent)/sd-cost,True
            if dr==-1 and hi>=stop-EPS:
                px=op if op>stop else stop;return (ent-px)/sd-cost,True
        fav=(hi-ent)/sd if dr==1 else (ent-lo)/sd
        if fav+EPS>=activate:active=True
    return base,False

def seg_stats(z,cf):
    x=z.copy();x['cf_r']=cf;x['cf_pnl']=x.risk_cash*x.cf_r;x['delta']=x.cf_pnl-x.net_pnl
    bw=x.net_pnl>0;bl=x.net_pnl<0;bf=x.net_pnl.abs()<=1e-12;cw=x.cf_pnl>EPS;cl=x.cf_pnl<-EPS;cf0=x.cf_pnl.abs()<=EPS
    return dict(n=len(x),delta_usd=float(x.delta.sum()),wins=int(cw.sum()),losses=int(cl.sum()),flats=int(cf0.sum()),flat_to_profit=int((bf&cw).sum()),flat_to_loss=int((bf&cl).sum()),losses_improved=int((bl&(x.cf_pnl>x.net_pnl+EPS)).sum()),loss_to_nonloss=int((bl&~cl).sum()),winners_harmed=int((bw&(x.cf_pnl<x.net_pnl-EPS)).sum()),winner_to_loss=int((bw&cl).sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy');mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_gate_control')
    res,W,EV,DE,PR=eval_variant(mod,data,cands,opps,fcache,specs,100.0)
    assert abs(float(res['end_equity'])-LOCKED_END)<=1e-9 and len(EV)==LOCKED_TRADES
    EV=EV.copy();EV['eval_week']=pd.to_datetime(EV.eval_week,utc=True);EV['entry_time']=pd.to_datetime(EV.entry_time,utc=True);EV['exit_time']=pd.to_datetime(EV.exit_time,utc=True)
    for c in ['net_pnl','net_r','risk_cash','cost_r','entry_price','stop_distance','direction']:
        EV[c]=pd.to_numeric(EV[c],errors='coerce').fillna(0.0)
    weeks=sorted(EV.eval_week.unique());cut=weeks[-3];early=EV.eval_week<cut;late=~early
    paths={i:path_for(fcache.get(r.asset),r) for i,r in EV.iterrows()}
    behaviors=[]
    for act in [.50,.75]:
      for ret in [.05,.10,.15,.20]:
        for bars in [1,2]:behaviors.append((f'close_a{act:.2f}_r{ret:.2f}_{bars}b','close',act,ret,bars))
    for act in [.50,.75,1.00]:
      for fl in [.02,.05,.10,.15,.20]:behaviors.append((f'lock_a{act:.2f}_f{fl:.2f}','lock',act,fl,1))
    # Pre-entry causal fields only. These were known at the decision time in locked R5.4.
    metrics=[c for c in ['reliability_score','reliability_expected_r','reliability_win','reliability_strategy_mean_r','reliability_strategy_win','reliability_asset_strategy_mean_r','reliability_asset_strategy_win','expected_r','trailing_win_rate','stability','trailing_giveback','quality','rank_score'] if c in EV.columns]
    base_r=EV.net_r.to_numpy(float);rows=[];beh_cache={}
    for name,kind,act,x,bars in behaviors:
        vals=[];hit=[]
        for i,r in EV.iterrows():
            rr,h=(close_rescue(r,paths[i],act,x,bars) if kind=='close' else profit_lock(r,paths[i],act,x));vals.append(rr);hit.append(h)
        vals=np.array(vals,float);hit=np.array(hit,bool);beh_cache[name]=(vals,hit,kind,act,x,bars)
    # Ungated controls plus simple causal one-dimensional gates. Thresholds come ONLY from first-8-week quantiles.
    for name,(vals,hit,kind,act,x,bars) in beh_cache.items():
        for metric in ['ALL',*metrics]:
            specs_gate=[('all',None)] if metric=='ALL' else []
            if metric!='ALL':
                q=pd.to_numeric(EV.loc[early,metric],errors='coerce').dropna()
                if len(q):
                    for qt in [.20,.30,.40,.50,.60,.70,.80]:
                        th=float(q.quantile(qt));specs_gate.extend([('le',th),('ge',th)])
            for direction,th in specs_gate:
                gate=np.ones(len(EV),dtype=bool) if metric=='ALL' else (pd.to_numeric(EV[metric],errors='coerce').fillna(np.nan).to_numpy()<=th if direction=='le' else pd.to_numeric(EV[metric],errors='coerce').fillna(np.nan).to_numpy()>=th)
                gate=np.asarray(gate,dtype=bool)&hit
                cf=np.where(gate,vals,base_r)
                es=seg_stats(EV.loc[early],cf[early.to_numpy()]);ls=seg_stats(EV.loc[late],cf[late.to_numpy()]);fs=seg_stats(EV,cf)
                discover_ok=es['delta_usd']>EPS and es['flat_to_loss']==0 and es['winner_to_loss']==0 and (es['losses_improved']>0 or es['flat_to_profit']>0)
                validate_ok=ls['delta_usd']>=-EPS and ls['flat_to_loss']==0 and ls['winner_to_loss']==0
                rows.append({'behavior':name,'kind':kind,'activate_mfe_r':act,'parameter_r':x,'bars':bars,'gate_metric':metric,'gate_direction':direction,'gate_threshold':th,'applications':int(gate.sum()),'discovery_pass':discover_ok,'validation_pass':validate_ok,**{f'disc_{k}':v for k,v in es.items()},**{f'hold_{k}':v for k,v in ls.items()},**{f'all_{k}':v for k,v in fs.items()}})
    D=pd.DataFrame(rows)
    # Rank target repair first, then economics, but only after discovery+holdout safety gates.
    D['joint_pass']=D.discovery_pass&D.validation_pass
    D['repair_score']=D.all_loss_to_nonloss*4+D.all_losses_improved*1.5+D.all_flat_to_profit-D.all_winners_harmed*.5
    D=D.sort_values(['joint_pass','repair_score','all_delta_usd','hold_delta_usd'],ascending=[False,False,False,False]);D.to_csv(out/'R5_5_LANE1A2_GATED_REPAIRS.csv',index=False)
    top=D.iloc[0].to_dict();status={'state':'R5_5_LANE1A2_RELIABILITY_GATED_DISCOVERY_COMPLETE','r5_4_control':res,'split':{'discovery_weeks':8,'holdout_weeks':3,'cut':str(cut)},'scope':{'routes':34,'strategy_cells':510,'strategy_families':15,'locked_executions':len(EV)},'behaviors':len(behaviors),'gate_metrics':metrics,'rules_tested':len(D),'joint_passing_rules':int(D.joint_pass.sum()),'best_joint_rule':top,'governance':'Gate thresholds are estimated from first-eight-week pre-entry causal fields only; final three weeks are untouched validation. This is still selection-frozen discovery and must enter end-to-end replay before promotion.'};(out/'R5_5_LANE1A2_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
