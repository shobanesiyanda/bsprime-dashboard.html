#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant
from lane4_market_state_entry_repair import enrich_market_state
from winner_causal_features import causal_decision_features

R54_END=337.4231242104393
INC_END=343.5447309255116
FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
EPS=1e-9
B3={'behavior':'close_a0.50_r0.20_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.20,'bars':1,'gate_metric':'reliability_win','gate_direction':'le','gate_threshold':0.2655132009803008}
FIELDS=['reliability_score','reliability_expected_r','reliability_win','reliability_strategy_mean_r','reliability_strategy_win','reliability_asset_strategy_mean_r','reliability_asset_strategy_win','expected_r','trailing_win_rate','stability','trailing_giveback','quality','rank_score','cost_r','trend_alignment','h1_strength','compression','signed_z30','ema20_distance_stop_r','stop_atr_ratio','body_frac','signed_impulse_atr','signed_mom6_atr','directional_range20_pos','signed_ret1_atr','adverse_wick_frac']

def num(x,d=np.nan):
    try:v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def cond(z,f,d,th):
    if f not in z:return pd.Series(False,index=z.index)
    s=pd.to_numeric(z[f],errors='coerce');return s.le(th) if d=='le' else s.ge(th)

def rule_mask(z,r):
    if str(r['scope_type'])=='GLOBAL':m=pd.Series(True,index=z.index)
    elif str(r['scope_type'])=='ASSET_CLASS':m=z.asset_class.astype(str).eq(str(r['scope_value']))
    else:m=z.strategy_family.astype(str).eq(str(r['scope_value']))
    m &= cond(z,str(r['feature1']),str(r['dir1']),num(r['threshold1']))
    f2=str(r.get('feature2','') or '')
    if f2 and f2.lower()!='nan':m &= cond(z,f2,str(r['dir2']),num(r['threshold2']))
    return m.fillna(False)

def stats(z,m):
    x=z.loc[m].copy();p=pd.to_numeric(x.get('net_pnl',0),errors='coerce').fillna(0)
    return {'n':int(len(x)),'wins':int((p>EPS).sum()),'losses':int((p<-EPS).sum()),'flats':int((p.abs()<=EPS).sum()),'win_rate_all':float((p>EPS).mean()) if len(x) else 0.,'mean_net_pnl':float(p.mean()) if len(x) else 0.,'total_net_pnl':float(p.sum()) if len(x) else 0.}

def shifts(valid,fresh,fields):
    rows=[]
    for f in fields:
        if f not in valid or f not in fresh:continue
        a=pd.to_numeric(valid[f],errors='coerce').dropna();b=pd.to_numeric(fresh[f],errors='coerce').dropna()
        if len(a)<4 or len(b)<2:continue
        meda=float(a.median());medb=float(b.median());iqr=float(a.quantile(.75)-a.quantile(.25));scale=iqr if abs(iqr)>1e-9 else max(abs(meda),1e-6)
        rows.append({'feature':f,'validation_median':meda,'fresh_median':medb,'median_delta':medb-meda,'standardized_shift':(medb-meda)/scale,'validation_n':len(a),'fresh_n':len(b)})
    return pd.DataFrame(rows).sort_values('standardized_shift',key=lambda s:s.abs(),ascending=False) if rows else pd.DataFrame()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_l6e_ctl');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 control reproduction failed')
    inc,W,EV,DE,PR=eval_variant(ManagedVariant(B3,base,promote),data,cands,opps,fcache,specs,100.0)
    if abs(float(inc['end_equity'])-INC_END)>1e-6:raise RuntimeError('Lane1B2 incumbent reproduction failed')
    z=enrich_market_state(EV,fcache);z['decision_time']=pd.to_datetime(z.decision_time,utc=True);z['eval_week']=pd.to_datetime(z.eval_week,utc=True)
    weeks=sorted(z.eval_week.unique());cut=weeks[-3];early=z[z.eval_week<cut].copy();valid=z[(z.eval_week>=cut)&(z.decision_time<FRESH0)].copy();fresh=z[z.decision_time>=FRESH0].copy()
    R=pd.read_csv(a.rules);R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True)
    rule_rows=[]
    for rank,r in R.head(20).iterrows():
        rr=r.to_dict();me=rule_mask(early,rr);mv=rule_mask(valid,rr);mf=rule_mask(fresh,rr)
        row={'rule_rank':int(rank),'scope_type':rr['scope_type'],'scope_value':rr['scope_value'],'feature1':rr['feature1'],'dir1':rr['dir1'],'threshold1':num(rr['threshold1']),'feature2':str(rr.get('feature2','') or ''),'dir2':str(rr.get('dir2','') or ''),'threshold2':num(rr.get('threshold2'))}
        for p,s in [('discovery',stats(early,me)),('validation',stats(valid,mv)),('fresh',stats(fresh,mf))]:row.update({f'{p}_{k}':v for k,v in s.items()})
        rule_rows.append(row)
    rule_df=pd.DataFrame(rule_rows);rule_df.to_csv(out/'R5_5_LANE6E_RULE_FRESH_BEHAVIOUR.csv',index=False)
    best=R.iloc[0].to_dict();mv=rule_mask(valid,best);mf=rule_mask(fresh,best);vq=valid.loc[mv].copy();fq=fresh.loc[mf].copy();drift=shifts(vq,fq,[f for f in FIELDS if f in z.columns]);drift.to_csv(out/'R5_5_LANE6E_BEST_RULE_STATE_DRIFT.csv',index=False)
    ce=causal_decision_features(cands,fcache);ce['decision_time']=pd.to_datetime(ce.decision_time,utc=True);ce['winner_like']=rule_mask(ce,best)
    fresh_c=ce[ce.decision_time>=FRESH0].copy();fresh_parent=fresh_c[~fresh_c.get('is_addon',False).astype(bool)].copy();selected_ids=set(fresh.campaign_id.astype(str)) if 'campaign_id' in fresh else set();fresh_qual=fresh_parent[fresh_parent.winner_like].copy();qual_selected=int(fresh_qual.campaign_id.astype(str).isin(selected_ids).sum()) if 'campaign_id' in fresh_qual else 0
    parent_qual=dict(zip(fresh_parent.campaign_id.astype(str),fresh_parent.winner_like.astype(bool))) if 'campaign_id' in fresh_parent else {};addons=fresh_c[fresh_c.strategy.astype(str).eq('WINNER_ONLY_ADDON')].copy() if 'strategy' in fresh_c else fresh_c.iloc[0:0].copy()
    if len(addons) and 'parent_id' in addons:addons['qualified_parent']=addons.parent_id.astype(str).map(parent_qual).fillna(False)
    else:addons['qualified_parent']=False
    qual_addons=addons[addons.qualified_parent].copy();addon_selected=int(qual_addons.campaign_id.astype(str).isin(selected_ids).sum()) if len(qual_addons) and 'campaign_id' in qual_addons else 0;fresh_exec_qual=fq.copy();target_hits=int((fresh_exec_qual.get('exit_reason',pd.Series('',index=fresh_exec_qual.index)).astype(str)=='TARGET').sum());winner_like_wins=int((pd.to_numeric(fresh_exec_qual.get('net_pnl',0),errors='coerce')>EPS).sum())
    if len(fresh_qual)==0:root='WINNER_SIGNATURE_NOT_PRESENT_IN_FRESH_CANDIDATES'
    elif qual_selected==0:root='WINNER_SIGNATURE_PRESENT_BUT_NOT_SELECTED_CAPITAL_OR_PROMOTION_COMPETITION'
    elif len(qual_addons)==0 and target_hits==0:root='WINNER_SIGNATURE_SELECTED_BUT_NO_CONTINUATION_OR_3R_EXTENSION_TRIGGER'
    elif len(qual_addons)>0 and addon_selected==0:root='CONTINUATION_TRIGGER_PRESENT_BUT_ADDON_NOT_SELECTED_CAPACITY_OR_FACTOR_CONSTRAINT'
    elif target_hits==0:root='WINNER_LIKE_TRADES_SELECTED_BUT_NONE_REACHED_3R_PAYOFF_EXTENSION_GATE'
    elif winner_like_wins==0:root='WINNER_SIGNATURE_PRESENT_BUT_FRESH_REGIME_BROKE_OUTCOME_RELATIONSHIP'
    else:root='WINNER_SIGNATURE_AND_ACTIVATION_PRESENT; MONETIZATION_OR_PORTFOLIO_INTERACTION_IS_BOTTLENECK'
    status={'state':'R5_5_LANE6E_AUG13_14_WINNER_STATE_DIFFERENCE_FORENSIC_COMPLETE','evidence_class':'QUARANTINED_FRESH_DIAGNOSTIC_NOT_SELECTION_EVIDENCE','scope':{'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'partition':{'discovery_events':len(early),'validation_pre_aug13_events':len(valid),'fresh_aug13_14_events':len(fresh)},'best_pre_aug13_rule':best,'best_rule_validation':stats(valid,mv),'best_rule_fresh':stats(fresh,mf),'fresh_mechanism_counts':{'winner_like_parent_candidates':int(len(fresh_qual)),'winner_like_parent_selected':qual_selected,'qualified_winner_only_addon_candidates':int(len(qual_addons)),'qualified_winner_only_addons_selected':addon_selected,'winner_like_executions':int(len(fresh_exec_qual)),'winner_like_execution_wins':winner_like_wins,'winner_like_3R_target_hits':target_hits},'root_cause_class':root,'largest_state_shifts':drift.head(10).to_dict('records') if len(drift) else [],'top20_rule_fresh_survival':rule_df.sort_values(['fresh_total_net_pnl','fresh_win_rate_all','fresh_n'],ascending=False).head(10).to_dict('records') if len(rule_df) else [],'selection_governance':'Candidate winner qualification is reconstructed from weekly-causal reliability state. Aug13-14 outcomes are quarantine evidence used only to diagnose state dependency and cannot nominate or promote a candidate.'}
    (out/'R5_5_LANE6E_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
