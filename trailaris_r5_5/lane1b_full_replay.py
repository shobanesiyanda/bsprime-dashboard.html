#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd

R54_END=337.4231242104393
EPS=1e-12

def num(x,d=0.0):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def gate_ok(row,rule):
    metric=str(rule.get('gate_metric','ALL'))
    if metric=='ALL': return True
    if metric not in row.index: return False
    v=num(row.get(metric),np.nan);th=num(rule.get('gate_threshold'),np.nan)
    if not np.isfinite(v) or not np.isfinite(th): return False
    direction=str(rule.get('gate_direction','all'))
    if direction=='le': return v<=th+EPS
    if direction=='ge': return v>=th-EPS
    return True

def trade_path(frame,row):
    if frame is None or len(frame)==0:return None
    t=pd.to_datetime(frame['timestamp'],utc=True)
    a=pd.Timestamp(row.get('entry_time',row.decision_time));b=pd.Timestamp(row.exit_time)
    lo=int(t.searchsorted(a,side='left'));hi=int(t.searchsorted(b,side='left'))
    if hi<=lo:return None
    z=frame.iloc[lo:hi].copy();z['timestamp']=t.iloc[lo:hi].to_numpy()
    return z

def rescue_exit(row,frame,activate,retreat,bars):
    cost=max(0.,num(row.get('cost_r')));ent=num(row.get('entry_price'));sd=max(abs(num(row.get('stop_distance'))),1e-12);dr=int(num(row.get('direction'),1))
    z=trade_path(frame,row)
    if z is None:return None
    mfe=0.;run=0
    for _,b in z.iterrows():
        hi=num(b.get('high'));lo=num(b.get('low'));cl=num(b.get('close'));op=num(b.get('open'),cl)
        fav=(hi-ent)/sd if dr==1 else (ent-lo)/sd;mfe=max(mfe,fav)
        adverse=cl<op if dr==1 else cl>op;run=run+1 if adverse else 0
        rr=(cl-ent)*dr/sd-cost
        if mfe+EPS>=activate and rr<=retreat+EPS and rr>0 and run>=bars:
            xt=min(pd.Timestamp(row.exit_time),pd.Timestamp(b.timestamp)+pd.Timedelta(minutes=1))
            return rr,xt,cl
    return None

def lock_exit(row,frame,activate,floor_net):
    cost=max(0.,num(row.get('cost_r')));ent=num(row.get('entry_price'));sd=max(abs(num(row.get('stop_distance'))),1e-12);dr=int(num(row.get('direction'),1))
    z=trade_path(frame,row)
    if z is None:return None
    active=False;gross=float(floor_net)+cost;stop=ent+dr*gross*sd
    for _,b in z.iterrows():
        hi=num(b.get('high'));lo=num(b.get('low'));op=num(b.get('open'));ts=pd.Timestamp(b.timestamp)
        if active:
            if dr==1 and lo<=stop+EPS:
                px=op if op<stop else stop;rr=(px-ent)/sd-cost
                return rr,min(pd.Timestamp(row.exit_time),ts+pd.Timedelta(minutes=1)),px
            if dr==-1 and hi>=stop-EPS:
                px=op if op>stop else stop;rr=(ent-px)/sd-cost
                return rr,min(pd.Timestamp(row.exit_time),ts+pd.Timedelta(minutes=1)),px
        fav=(hi-ent)/sd if dr==1 else (ent-lo)/sd
        if fav+EPS>=activate: active=True
    return None

def apply_rule(cands,feature_cache,rule):
    if cands.empty:return cands
    z=cands.copy();z['decision_time']=pd.to_datetime(z.decision_time,utc=True);z['exit_time']=pd.to_datetime(z.exit_time,utc=True)
    modified=[]
    kind=str(rule['kind']);act=num(rule['activate_mfe_r']);param=num(rule['parameter_r']);bars=int(num(rule.get('bars'),1))
    rid=f"{rule['behavior']}|{rule.get('gate_metric','ALL')}|{rule.get('gate_direction','all')}|{rule.get('gate_threshold')}"
    for i,row in z.iterrows():
        if not gate_ok(row,rule):continue
        frame=feature_cache.get(str(row.asset)) if feature_cache else None
        hit=rescue_exit(row,frame,act,param,bars) if kind=='close' else lock_exit(row,frame,act,param)
        if hit is None:continue
        rr,xt,px=hit
        if xt>=pd.Timestamp(row.exit_time)-pd.Timedelta(microseconds=1):continue
        z.at[i,'net_r']=float(rr);z.at[i,'exit_time']=xt
        if 'exit_price' in z.columns:z.at[i,'exit_price']=float(px)
        z.at[i,'management_rule']=rid;z.at[i,'management_original_exit_time']=pd.Timestamp(row.exit_time);z.at[i,'management_original_net_r']=num(row.net_r)
        modified.append(i)
    return z

class ManagedVariant:
    def __init__(self,rule,base,promote):self.rule=rule;self.base=base;self.promote=promote
    def promote_for_week(self,cands,week_start):return self.promote(cands,week_start,'combined')
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):
        return self.base.replay_r5(apply_rule(cands,feature_cache,self.rule),specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--lane1a2-status',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    src=json.load(open(a.lane1a2_status));joint=int(src.get('joint_passing_rules',0))
    if joint<1:
        status={'state':'R5_5_LANE1B_NOT_RUN_NO_JOINT_LANE1A2_RULE','joint_passing_rules':joint,'next_lane':'STRUCTURE_REGIME_ENTRY_THESIS_REPAIR','governance':'No rule is forced into full replay when discovery plus untouched holdout gates fail.'}
        (out/'R5_5_LANE1B_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));return
    rule=src['best_joint_rule']
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    control_mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane1b_control')
    ctl,CW,CEV,CDE,CPR=eval_variant(control_mod,data,cands,opps,fcache,specs,100.0)
    assert abs(float(ctl['end_equity'])-R54_END)<=1e-9
    variant=ManagedVariant(rule,base,promote)
    cand,VW,VEV,VDE,VPR=eval_variant(variant,data,cands,opps,fcache,specs,100.0)
    routes=len(data);families=int(cands.strategy_family.nunique());cells=routes*families
    no_return_regression=float(cand['end_equity'])>=float(ctl['end_equity'])-1e-9
    fresh_improved=float(cand['fresh_return_pct'])>float(ctl['fresh_return_pct'])+1e-9
    dd_not_worse=float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-1e-9
    repair=(int(cand['losses'])<=int(ctl['losses']) and int(cand['flat'])<=int(ctl['flat']) and (int(cand['losses'])<int(ctl['losses']) or int(cand['flat'])<int(ctl['flat'])))
    scope_ok=(routes==34 and families==15 and cells==510)
    promotion_candidate=bool(no_return_regression and fresh_improved and dd_not_worse and repair and scope_ok)
    managed=VEV[VEV.get('management_rule',pd.Series(index=VEV.index,dtype=object)).notna()].copy() if len(VEV) else pd.DataFrame()
    status={'state':'R5_5_LANE1B_FULL_REPLAY_COMPLETE','evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY','selected_lane1a2_rule':rule,'r5_4_control':ctl,'r5_5_candidate':cand,'scope':{'routes':routes,'strategy_families':families,'strategy_cells':cells},'managed_executions':int(len(managed)),'gates':{'end_equity_not_regressed':no_return_regression,'fresh_holdout_improved':fresh_improved,'max_weekly_drawdown_not_worse':dd_not_worse,'loss_flat_repair_without_count_worsening':repair,'full_universe_34x510':scope_ok},'promotion_candidate':promotion_candidate,'governance':'R5.4 selector and files remain unchanged. Only the Lane 1A2 winning causal management rule alters eligible trade exits before the locked replay engine recomputes capital availability, compounding, factor exposure, weekly protection and later selection capacity. Broker degradation certification remains mandatory before R5.5 promotion.'}
    (out/'R5_5_LANE1B_STATUS.json').write_text(json.dumps(status,indent=2,default=str))
    wk=VW.copy();wk.to_csv(out/'R5_5_LANE1B_WEEKLY.csv',index=False)
    if len(managed):managed.to_csv(out/'R5_5_LANE1B_MANAGED_EXECUTIONS.csv',index=False)
    print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
