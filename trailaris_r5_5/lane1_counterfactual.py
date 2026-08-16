#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

LOCKED_END=337.4231242104393
LOCKED_TRADES=1102
EPS=1e-12

def _num(x,default=0.0):
    try:
        v=float(x)
        return v if np.isfinite(v) else default
    except Exception:return default

def _path(frame,row):
    ts=frame['timestamp']
    ent=pd.Timestamp(row['entry_time']); ex=pd.Timestamp(row['exit_time'])
    lo=int(ts.searchsorted(ent,side='left')); hi=int(ts.searchsorted(ex,side='left'))
    # Strictly exclude the baseline exit bar. An early R5.5 close may only use a fully
    # completed bar that existed before the immutable R5.4 exit event.
    return frame.iloc[lo:hi].copy()

def counterfactual_r(row,frame,partial_mfe=None,partial_fraction=0.0,rescue_mfe=None,rescue_retreat=None,rescue_adverse_bars=1):
    base_r=_num(row.get('net_r')); cost=max(0.0,_num(row.get('cost_r'))); gross_base=base_r+cost
    if frame is None or len(frame)==0:return base_r,False,False
    ent=_num(row.get('entry_price')); sd=max(abs(_num(row.get('stop_distance'))),1e-12); dr=int(_num(row.get('direction'),1))
    remaining=1.0;realized=0.0;partial_done=False;mfe=0.0;adverse_run=0
    for _,b in frame.iterrows():
        hi=_num(b.get('high'));lo=_num(b.get('low'));cl=_num(b.get('close'));op=_num(b.get('open'),cl)
        fav=((hi-ent)/sd if dr==1 else (ent-lo)/sd)
        mfe=max(mfe,fav)
        if partial_mfe is not None and not partial_done and mfe+EPS>=partial_mfe:
            f=min(max(float(partial_fraction),0.0),0.95)
            realized += f*float(partial_mfe); remaining-=f; partial_done=True
        close_r=(cl-ent)*dr/sd
        adverse=(cl<op if dr==1 else cl>op)
        adverse_run=adverse_run+1 if adverse else 0
        if rescue_mfe is not None and rescue_retreat is not None and mfe+EPS>=float(rescue_mfe) and close_r<=float(rescue_retreat)+EPS and adverse_run>=int(rescue_adverse_bars):
            # Exit at a completed bar close; subtract the original full-position cost burden once.
            return realized+remaining*close_r-cost,True,partial_done
    return realized+remaining*gross_base-cost,False,partial_done

def metrics(ev,cf_r,name,params):
    z=ev.copy();z['cf_r']=cf_r
    z['cf_pnl']=z['risk_cash']*z['cf_r'];z['delta_pnl']=z['cf_pnl']-z['net_pnl']
    bflat=z.net_pnl.abs()<=1e-12;bloss=z.net_pnl<0;bwin=z.net_pnl>0
    cflat=z.cf_pnl.abs()<=1e-12;closs=z.cf_pnl<0;cwin=z.cf_pnl>0
    weeks=sorted(pd.to_datetime(z.eval_week,utc=True).dropna().unique()) if 'eval_week' in z else []
    fresh=z[pd.to_datetime(z.eval_week,utc=True)>=weeks[-3]] if len(weeks)>=3 else z.iloc[0:0]
    order=z.sort_values('timestamp').copy();eq=100.0+order.cf_pnl.cumsum();peak=eq.cummax();dd=(eq/peak-1.0) if len(eq) else pd.Series(dtype=float)
    return {
      'variant':name,**params,'evidence_class':'SELECTION_FROZEN_CAUSAL_MANAGEMENT_DISCOVERY_ONLY',
      'trades':int(len(z)),'selection_frozen_end_equity':float(100.0+z.cf_pnl.sum()),'delta_vs_locked_usd':float(z.delta_pnl.sum()),
      'wins':int(cwin.sum()),'losses':int(closs.sum()),'flats':int(cflat.sum()),
      'flat_to_profit':int((bflat&cwin).sum()),'flat_to_loss':int((bflat&closs).sum()),
      'losses_improved':int((bloss&(z.cf_pnl>z.net_pnl+1e-12)).sum()),'loss_to_nonloss':int((bloss&~closs).sum()),
      'winner_pnl_delta_usd':float(z.loc[bwin,'delta_pnl'].sum()),'winners_harmed':int((bwin&(z.cf_pnl<z.net_pnl-1e-12)).sum()),'winner_to_loss':int((bwin&closs).sum()),
      'fresh_delta_usd':float(fresh.delta_pnl.sum()) if len(fresh) else 0.0,
      'selection_frozen_max_dd_pct':float(dd.min()*100) if len(dd) else 0.0,
      'discovery_guard_pass':bool(int((bflat&closs).sum())==0 and int((bwin&closs).sum())==0 and float(z.delta_pnl.sum())>0 and (float(fresh.delta_pnl.sum())>=-1e-12 if len(fresh) else True))
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_locked_control')
    res,W,EV,DE,PR=eval_variant(mod,data,cands,opps,fcache,specs,100.0)
    if abs(float(res['end_equity'])-LOCKED_END)>1e-9 or len(EV)!=LOCKED_TRADES:raise RuntimeError(f'R5.4 control drift: {res.get("end_equity")} / {len(EV)}')
    EV=EV.copy();EV['timestamp']=pd.to_datetime(EV.timestamp,utc=True);EV['entry_time']=pd.to_datetime(EV.entry_time,utc=True);EV['exit_time']=pd.to_datetime(EV.exit_time,utc=True)
    for c in ['net_pnl','net_r','risk_cash','cost_r','entry_price','stop_distance','direction','mfe_r','mae_r']:
        EV[c]=pd.to_numeric(EV[c],errors='coerce').fillna(0.0)
    paths={}
    for i,r in EV.iterrows():paths[i]=_path(fcache.get(r.asset),r) if r.asset in fcache else None
    variants=[('control',None,0.0,None,None,1)]
    for th in [0.50,0.75,1.00,1.25,1.50]:
      for f in [0.05,0.10,0.15,0.20,0.25]:variants.append((f'partial_{int(f*100)}_at_{th:.2f}R',th,f,None,None,1))
    for rm in [0.50,0.75]:
      for retreat in [0.20,0.10,0.00,-0.10]:
        for bars in [1,2]:variants.append((f'rescue_{rm:.2f}R_retreat_{retreat:+.2f}R_{bars}bar',None,0.0,rm,retreat,bars))
    # Combined grid is deliberately bounded: only practical partial/rescue pairs, not brute-force hindsight search.
    for th,f in [(0.75,0.10),(0.75,0.15),(1.00,0.10),(1.00,0.15)]:
      for rm,retreat,bars in [(0.50,0.10,1),(0.50,0.00,2),(0.75,0.10,1),(0.75,0.00,2)]:
        variants.append((f'combo_p{int(f*100)}_{th:.2f}_r{rm:.2f}_{retreat:+.2f}_{bars}b',th,f,rm,retreat,bars))
    rows=[];detail=[]
    for name,pth,pf,rm,ret,bars in variants:
      vals=[];early=0;partials=0
      for i,r in EV.iterrows():
        rr,e,p=counterfactual_r(r,paths.get(i),pth,pf,rm,ret,bars);vals.append(rr);early+=int(e);partials+=int(p)
      m=metrics(EV,pd.Series(vals,index=EV.index),name,{'partial_mfe_r':pth,'partial_fraction':pf,'rescue_mfe_r':rm,'rescue_retreat_r':ret,'rescue_adverse_bars':bars,'early_exits':early,'partial_fills':partials});rows.append(m)
    d=pd.DataFrame(rows).sort_values(['discovery_guard_pass','selection_frozen_end_equity'],ascending=[False,False]);d.to_csv(out/'R5_5_LANE1A_COUNTERFACTUALS.csv',index=False)
    control=d[d.variant=='control'].iloc[0].to_dict();best=d[d.variant!='control'].iloc[0].to_dict()
    status={'state':'R5_5_LANE1A_DISCOVERY_COMPLETE','r5_4_control':res,'scope':{'routes':34,'strategy_cells':510,'strategy_families':15,'locked_executions':len(EV)},'method':'All rules are applied to every locked execution using only completed bars before its immutable R5.4 exit. Partial fills are threshold-triggered for all outcomes; rescue exits are causal completed-bar exits for all outcomes. Selection and baseline risk cash remain frozen, so this lane ranks management hypotheses but is not promotion evidence.','variants_tested':len(d),'guard_passing_variants':int(d.discovery_guard_pass.sum()),'best_discovery_variant':best,'next_gate':'Rebuild the best rule(s) inside the full candidate/reliability/replay engine so changed exits, compounding, open risk, portfolio competition and fresh holdout are recomputed end-to-end.'}
    (out/'R5_5_LANE1A_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))

if __name__=='__main__':main()
