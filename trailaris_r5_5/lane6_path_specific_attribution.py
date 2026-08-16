#!/usr/bin/env python3
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant
from lane4_market_state_entry_repair import enrich_market_state
from lane6c_winner_payoff_extension import extend_one

R54_END=337.4231242104393
INC_END=343.5447309255116
FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
EPS=1e-12
B3={'behavior':'close_a0.50_r0.20_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.20,'bars':1,'gate_metric':'reliability_win','gate_direction':'le','gate_threshold':0.2655132009803008}
FIELDS=['reliability_score','reliability_expected_r','reliability_win','reliability_strategy_mean_r','reliability_strategy_win','reliability_asset_strategy_mean_r','reliability_asset_strategy_win','expected_r','trailing_win_rate','stability','trailing_giveback','quality','rank_score','cost_r','trend_alignment','h1_strength','compression','signed_z30','ema20_distance_stop_r','stop_atr_ratio','body_frac','signed_impulse_atr','signed_mom6_atr','directional_range20_pos','signed_ret1_atr','adverse_wick_frac']
QGRID=[.10,.20,.30,.40,.60,.70,.80,.90]

def num(x,d=np.nan):
    try:v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def scope_mask(z,t,v):
    if t=='GLOBAL':return pd.Series(True,index=z.index)
    if t=='ASSET_CLASS':return z.asset_class.astype(str).eq(str(v))
    return z.strategy_family.astype(str).eq(str(v))

def scopes(z):
    out=[('GLOBAL','ALL')]
    out += [('ASSET_CLASS',str(v)) for v,c in z.asset_class.value_counts().items() if c>=55]
    out += [('STRATEGY_FAMILY',str(v)) for v,c in z.strategy_family.value_counts().items() if c>=40]
    return out

def cond(z,f,d,th):
    s=pd.to_numeric(z[f],errors='coerce');return s.le(th) if d=='le' else s.ge(th)

def cohort_stats(z,m,label,utility):
    x=z.loc[m];lab=x[label].astype(bool);base=z[label].astype(bool);u=pd.to_numeric(x[utility],errors='coerce').fillna(0)
    rate=float(lab.mean()) if len(x) else 0.;brate=float(base.mean()) if len(z) else 0.
    return {'n':int(len(x)),'share':float(len(x)/max(len(z),1)),'target_n':int(lab.sum()),'target_rate':rate,'baseline_target_rate':brate,'target_lift':rate/max(brate,EPS),'mean_utility':float(u.mean()) if len(x) else 0.,'total_utility':float(u.sum()) if len(x) else 0.}

def clean(e,v):
    return e['n']>=25 and v['n']>=8 and e['share']<=.45 and v['share']<=.50 and e['target_lift']>=1.12 and v['target_lift']>=1.08 and e['mean_utility']>0 and v['mean_utility']>0

def test1(early,valid,label,utility,fields):
    rows=[]
    for st,sv in scopes(early):
        se=scope_mask(early,st,sv);vv=scope_mask(valid,st,sv)
        for f in fields:
            vals=pd.to_numeric(early.loc[se,f],errors='coerce').dropna()
            if len(vals)<35:continue
            for q in QGRID:
                th=float(vals.quantile(q))
                for d in ['le','ge']:
                    e=cohort_stats(early,se&cond(early,f,d,th),label,utility);v=cohort_stats(valid,vv&cond(valid,f,d,th),label,utility)
                    rows.append({'scope_type':st,'scope_value':sv,'rule_type':'1D','feature1':f,'dir1':d,'threshold1':th,'q1':q,'feature2':'','dir2':'','threshold2':np.nan,'q2':np.nan,**{f'disc_{k}':x for k,x in e.items()},**{f'valid_{k}':x for k,x in v.items()},'clean_pass':bool(clean(e,v))})
    return pd.DataFrame(rows)

def test2(early,valid,label,utility,one):
    if one.empty:return pd.DataFrame()
    p=one[(one.disc_n>=20)&(one.valid_n>=6)&(one.disc_target_lift>=1.05)&(one.valid_target_lift>=1.02)&(one.disc_mean_utility>0)].copy()
    if p.empty:return pd.DataFrame()
    p['pre']=4*p.valid_target_lift+2*p.disc_target_lift+2*np.log1p(p.valid_n)+np.maximum(p.valid_mean_utility,0)
    recs=p.sort_values('pre',ascending=False).drop_duplicates(['scope_type','scope_value','feature1','dir1']).head(30).to_dict('records');rows=[]
    for a,b in itertools.combinations(recs,2):
        if (a['scope_type'],a['scope_value'])!=(b['scope_type'],b['scope_value']) or a['feature1']==b['feature1']:continue
        st,sv=a['scope_type'],a['scope_value'];se=scope_mask(early,st,sv);vv=scope_mask(valid,st,sv)
        me=se&cond(early,a['feature1'],a['dir1'],a['threshold1'])&cond(early,b['feature1'],b['dir1'],b['threshold1']);mv=vv&cond(valid,a['feature1'],a['dir1'],a['threshold1'])&cond(valid,b['feature1'],b['dir1'],b['threshold1'])
        e=cohort_stats(early,me,label,utility);v=cohort_stats(valid,mv,label,utility)
        rows.append({'scope_type':st,'scope_value':sv,'rule_type':'2D','feature1':a['feature1'],'dir1':a['dir1'],'threshold1':a['threshold1'],'q1':a['q1'],'feature2':b['feature1'],'dir2':b['dir1'],'threshold2':b['threshold1'],'q2':b['q1'],**{f'disc_{k}':x for k,x in e.items()},**{f'valid_{k}':x for k,x in v.items()},'clean_pass':bool(clean(e,v))})
    return pd.DataFrame(rows)

def extension_label(z,fcache):
    delta=[]
    for _,r in z.iterrows():
        if bool(r.get('is_addon',False)) or str(r.get('exit_reason',''))!='TARGET':delta.append(0.);continue
        h=extend_one(r,fcache.get(str(r.asset)) if fcache else None,.25,4.0,.8,36);delta.append(float(h['delta_r']) if h else 0.)
    return pd.Series(delta,index=z.index,dtype=float)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_l6path_ctl');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 reproduction failed')
    inc,W,EV,DE,PR=eval_variant(ManagedVariant(B3,base,promote),data,cands,opps,fcache,specs,100.0)
    if abs(float(inc['end_equity'])-INC_END)>1e-6:raise RuntimeError('Lane1B2 reproduction failed')
    z=enrich_market_state(EV,fcache);z['decision_time']=pd.to_datetime(z.decision_time,utc=True);z['eval_week']=pd.to_datetime(z.eval_week,utc=True)
    pnl=pd.to_numeric(z.net_pnl,errors='coerce').fillna(0);mfe=pd.to_numeric(z.get('mfe_r'),errors='coerce').fillna(0);mae=pd.to_numeric(z.get('mae_r'),errors='coerce').fillna(np.inf);nr=pd.to_numeric(z.get('net_r'),errors='coerce').fillna(0)
    z['amp_label']=(pnl>EPS)&(mfe>=1.5)&(mae<=.75);z['amp_utility']=np.where(z.amp_label,nr,0.)
    z['extension_delta_r']=extension_label(z,fcache);z['extend_label']=z.extension_delta_r>EPS;z['extend_utility']=z.extension_delta_r
    winner_nr=nr[(pnl>EPS)&(z.decision_time<FRESH0)];priority_floor=float(winner_nr.median()) if len(winner_nr) else .5;z['priority_label']=(pnl>EPS)&(nr>=priority_floor);z['priority_utility']=np.where(z.priority_label,nr,0.)
    weeks=sorted(z.eval_week.unique());cut=weeks[-3];early=z[z.eval_week<cut].copy();valid=z[(z.eval_week>=cut)&(z.decision_time<FRESH0)].copy();fresh=z[z.decision_time>=FRESH0].copy();fields=[f for f in FIELDS if f in z]
    paths={'AMPLIFICATION':('amp_label','amp_utility'),'PAYOFF_EXTENSION':('extend_label','extend_utility'),'PRIORITY_CAPTURE':('priority_label','priority_utility')};status={'state':'R5_5_LANE6_PATH_SPECIFIC_ATTRIBUTION_COMPLETE','evidence_class':'SELECTION_FROZEN_PATH_SPECIFIC_WINNER_ATTRIBUTION','scope':{'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'fresh_outcomes_used_for_selection':False,'priority_net_r_floor':priority_floor,'paths':{}}
    for path,(lab,util) in paths.items():
        one=test1(early,valid,lab,util,fields);two=test2(early,valid,lab,util,one);R=pd.concat([x for x in [one,two] if len(x)],ignore_index=True,sort=False) if len(one) or len(two) else pd.DataFrame()
        if len(R):
            R['score']=8*R.valid_target_lift+4*R.disc_target_lift+3*np.log1p(R.valid_n)+np.log1p(R.disc_n)+4*np.maximum(R.valid_mean_utility,0)+2*np.maximum(R.disc_mean_utility,0);R=R.sort_values(['clean_pass','score','valid_target_lift','valid_mean_utility'],ascending=[False,False,False,False]).reset_index(drop=True);R.to_csv(out/f'R5_5_LANE6_{path}_RULES.csv',index=False)
        best=R[R.clean_pass].iloc[0].to_dict() if len(R) and R.clean_pass.any() else (R.iloc[0].to_dict() if len(R) else {})
        status['paths'][path]={'discovery_target_rate':float(early[lab].mean()),'validation_target_rate':float(valid[lab].mean()),'fresh_target_rate_diagnostic_only':float(fresh[lab].mean()),'rules_tested':int(len(R)),'passing_rules':int(R.clean_pass.sum()) if len(R) else 0,'best_pre_aug13_rule':best}
    status['governance']='Path labels may use realized historical outcomes, MFE/MAE, target reach and post-target path only to train/test discovery and pre-Aug13 validation cohorts. Runtime/replay eligibility for a later trade uses only causal decision-time features. Aug13-14 is reported diagnostically but cannot select a rule. Every nominated rule must return to full 34x15x510 authentic portfolio replay.'
    (out/'R5_5_LANE6_PATH_SPECIFIC_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
