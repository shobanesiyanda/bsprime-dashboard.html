#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant
from lane4_market_state_entry_repair import enrich_market_state

R54_END=337.4231242104393
INCUMBENT_END=343.5447309255116
FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
EPS=1e-12
B3={'behavior':'close_a0.50_r0.20_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.20,'bars':1,'gate_metric':'reliability_win','gate_direction':'le','gate_threshold':0.2655132009803008}
BASE_FIELDS=['reliability_score','reliability_expected_r','reliability_win','reliability_strategy_mean_r','reliability_strategy_win','reliability_asset_strategy_mean_r','reliability_asset_strategy_win','expected_r','trailing_win_rate','stability','trailing_giveback','quality','rank_score','cost_r']
MARKET_FIELDS=['trend_alignment','h1_strength','compression','signed_z30','ema20_distance_stop_r','stop_atr_ratio','body_frac','signed_impulse_atr','signed_mom6_atr','directional_range20_pos','signed_ret1_atr','adverse_wick_frac']
QGRID=[.10,.20,.30,.40,.60,.70,.80,.90]

def n(x,d=np.nan):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def scope_mask(z,typ,val):
    if typ=='GLOBAL':return pd.Series(True,index=z.index)
    if typ=='ASSET_CLASS':return z.asset_class.eq(val)
    return z.strategy_family.eq(val)

def scopes(early):
    out=[('GLOBAL','ALL')]
    for v,c in early.asset_class.value_counts().items():
        if c>=55:out.append(('ASSET_CLASS',v))
    for v,c in early.strategy_family.value_counts().items():
        if c>=40:out.append(('STRATEGY_FAMILY',v))
    return out

def cohort_stats(z,m):
    x=z.loc[m].copy(); pnl=pd.to_numeric(x.net_pnl,errors='coerce').fillna(0)
    base=pd.to_numeric(z.net_pnl,errors='coerce').fillna(0)
    wins=int((pnl>EPS).sum());loss=int((pnl<-EPS).sum());flat=int((pnl.abs()<=EPS).sum())
    wr=wins/max(len(x),1);bwr=float((base>EPS).mean()); mean=float(pnl.mean()) if len(x) else 0.; total=float(pnl.sum())
    pos=float(pnl[pnl>EPS].sum());neg=abs(float(pnl[pnl<-EPS].sum()));pf=pos/max(neg,EPS)
    return {'n':int(len(x)),'share':float(len(x)/max(len(z),1)),'wins':wins,'losses':loss,'flats':flat,'win_rate_all':wr,'baseline_win_rate_all':bwr,'win_lift':wr/max(bwr,EPS),'mean_net_pnl':mean,'total_net_pnl':total,'profit_factor':pf}

def clean(e,v):
    return (e['n']>=25 and v['n']>=8 and e['share']<=.40 and v['share']<=.45 and e['win_lift']>=1.15 and v['win_lift']>=1.10 and e['mean_net_pnl']>0 and v['mean_net_pnl']>0 and e['profit_factor']>=1.35 and v['profit_factor']>=1.20)

def cond(z,f,d,th):
    s=pd.to_numeric(z[f],errors='coerce');return s.le(th) if d=='le' else s.ge(th)

def test_1d(early,valid,fields):
    rows=[]
    for st,sv in scopes(early):
        se=scope_mask(early,st,sv);svm=scope_mask(valid,st,sv)
        for f in fields:
            vals=pd.to_numeric(early.loc[se,f],errors='coerce').dropna()
            if len(vals)<35:continue
            for q in QGRID:
                th=float(vals.quantile(q))
                for d in ['le','ge']:
                    e=cohort_stats(early,se&cond(early,f,d,th));v=cohort_stats(valid,svm&cond(valid,f,d,th))
                    rows.append({'scope_type':st,'scope_value':sv,'rule_type':'1D','feature1':f,'dir1':d,'threshold1':th,'q1':q,'feature2':'','dir2':'','threshold2':np.nan,'q2':np.nan,**{f'disc_{k}':vv for k,vv in e.items()},**{f'valid_{k}':vv for k,vv in v.items()},'clean_pass':bool(clean(e,v))})
    return pd.DataFrame(rows)

def test_2d(early,valid,one):
    if one.empty:return pd.DataFrame()
    pool=one[(one.disc_n>=20)&(one.valid_n>=6)&(one.disc_win_lift>=1.08)&(one.valid_win_lift>=1.03)&(one.disc_mean_net_pnl>0)].copy()
    if pool.empty:return pd.DataFrame()
    pool['pre_rank']=4*pool.valid_win_lift+2*pool.disc_win_lift+2*np.log1p(pool.valid_n)+np.maximum(pool.valid_mean_net_pnl,0)+np.maximum(pool.disc_mean_net_pnl,0)
    pool=pool.sort_values('pre_rank',ascending=False).drop_duplicates(['scope_type','scope_value','feature1','dir1']).head(30)
    recs=pool.to_dict('records');rows=[]
    for a,b in itertools.combinations(recs,2):
        if (a['scope_type'],a['scope_value'])!=(b['scope_type'],b['scope_value']) or a['feature1']==b['feature1']:continue
        st,sv=a['scope_type'],a['scope_value'];se=scope_mask(early,st,sv);svm=scope_mask(valid,st,sv)
        me=se&cond(early,a['feature1'],a['dir1'],a['threshold1'])&cond(early,b['feature1'],b['dir1'],b['threshold1'])
        mv=svm&cond(valid,a['feature1'],a['dir1'],a['threshold1'])&cond(valid,b['feature1'],b['dir1'],b['threshold1'])
        e=cohort_stats(early,me);v=cohort_stats(valid,mv)
        rows.append({'scope_type':st,'scope_value':sv,'rule_type':'2D','feature1':a['feature1'],'dir1':a['dir1'],'threshold1':a['threshold1'],'q1':a['q1'],'feature2':b['feature1'],'dir2':b['dir1'],'threshold2':b['threshold1'],'q2':b['q1'],**{f'disc_{k}':vv for k,vv in e.items()},**{f'valid_{k}':vv for k,vv in v.items()},'clean_pass':bool(clean(e,v))})
    return pd.DataFrame(rows)

def group_table(z,col):
    rows=[]
    for k,g in z.groupby(col,dropna=False):
        pnl=pd.to_numeric(g.net_pnl,errors='coerce').fillna(0);w=g[pnl>EPS]
        rows.append({col:str(k),'trades':len(g),'wins':int((pnl>EPS).sum()),'losses':int((pnl<-EPS).sum()),'flats':int((pnl.abs()<=EPS).sum()),'win_rate_all':float((pnl>EPS).mean()),'net_pnl':float(pnl.sum()),'winner_pnl':float(pd.to_numeric(w.net_pnl,errors='coerce').sum()) if len(w) else 0.,'winner_mean_mfe_r':float(pd.to_numeric(w.get('mfe_r'),errors='coerce').mean()) if len(w) and 'mfe_r' in w else np.nan,'winner_mean_mae_r':float(pd.to_numeric(w.get('mae_r'),errors='coerce').mean()) if len(w) and 'mae_r' in w else np.nan})
    return pd.DataFrame(rows).sort_values(['winner_pnl','net_pnl'],ascending=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane6_ctl');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 control reproduction failed')
    inc,W,EV,DE,PR=eval_variant(ManagedVariant(B3,base,promote),data,cands,opps,fcache,specs,100.0)
    if abs(float(inc['end_equity'])-INCUMBENT_END)>1e-6:raise RuntimeError(f'Lane1B2 incumbent reproduction failed {inc["end_equity"]}')
    z=enrich_market_state(EV,fcache);z['decision_time']=pd.to_datetime(z.decision_time,utc=True);z['eval_week']=pd.to_datetime(z.eval_week,utc=True)
    weeks=sorted(z.eval_week.unique());cut=weeks[-3];early=z[z.eval_week<cut].copy();valid=z[(z.eval_week>=cut)&(z.decision_time<FRESH0)].copy();fresh=z[z.decision_time>=FRESH0].copy()
    fields=[f for f in BASE_FIELDS+MARKET_FIELDS if f in z.columns]
    one=test_1d(early,valid,fields);two=test_2d(early,valid,one);R=pd.concat([x for x in [one,two] if len(x)],ignore_index=True,sort=False) if len(one) or len(two) else pd.DataFrame()
    if len(R):
        R['score']=8*R.valid_win_lift+4*R.disc_win_lift+3*np.log1p(R.valid_n)+np.log1p(R.disc_n)+4*np.maximum(R.valid_mean_net_pnl,0)+2*np.maximum(R.disc_mean_net_pnl,0)+np.log1p(np.maximum(R.valid_profit_factor,0))
        R=R.sort_values(['clean_pass','score','valid_win_lift','valid_mean_net_pnl'],ascending=[False,False,False,False]).reset_index(drop=True);R.to_csv(out/'R5_5_LANE6A_WINNER_RULES.csv',index=False)
    byfam=group_table(z,'strategy_family');byasset=group_table(z,'asset_class');byfam.to_csv(out/'R5_5_LANE6A_WINNERS_BY_FAMILY.csv',index=False);byasset.to_csv(out/'R5_5_LANE6A_WINNERS_BY_ASSET_CLASS.csv',index=False)
    winners=z[pd.to_numeric(z.net_pnl,errors='coerce')>EPS].copy();best=R[R.clean_pass].iloc[0].to_dict() if len(R) and R.clean_pass.any() else (R.iloc[0].to_dict() if len(R) else {})
    status={'state':'R5_5_LANE6A_WINNER_ATTRIBUTION_COMPLETE','evidence_class':'SELECTION_FROZEN_WINNER_ATTRIBUTION','incumbent_lane1b2':inc,'r5_4_control':ctl,'scope':{'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'partition':{'discovery_events':len(early),'validation_pre_aug13_events':len(valid),'fresh_quarantine_events':len(fresh),'fresh_outcomes_used_for_selection':False},'winner_forensics':{'wins':len(winners),'winner_net_pnl':float(pd.to_numeric(winners.net_pnl,errors='coerce').sum()),'mean_mfe_r':float(pd.to_numeric(winners.mfe_r,errors='coerce').mean()) if 'mfe_r' in winners else None,'median_mfe_r':float(pd.to_numeric(winners.mfe_r,errors='coerce').median()) if 'mfe_r' in winners else None,'mean_mae_r':float(pd.to_numeric(winners.mae_r,errors='coerce').mean()) if 'mae_r' in winners else None},'fields_tested':fields,'rules_tested':int(len(R)),'passing_rules':int(R.clean_pass.sum()) if len(R) else 0,'best_rule':best,'top_winner_family':byfam.iloc[0].to_dict() if len(byfam) else {},'top_winner_asset_class':byasset.iloc[0].to_dict() if len(byasset) else {},'next_lane':'WINNER_AMPLIFICATION_FULL_REPLAY','governance':'Post-trade MFE/MAE are attribution diagnostics only. Amplification eligibility uses decision-time fields plus completed market-state bars only, selected on first8 discovery and validated pre-Aug13. Aug13-14 remains quarantined. Any winner amplification must be re-run through the full 34x15 portfolio engine.'}
    (out/'R5_5_LANE6A_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
