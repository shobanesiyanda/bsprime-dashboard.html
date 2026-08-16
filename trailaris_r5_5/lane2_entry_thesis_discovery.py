#!/usr/bin/env python3
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np,pandas as pd

R54_END=337.4231242104393
EPS=1e-12
PREENTRY_FIELDS=[
 'reliability_score','reliability_expected_r','reliability_win',
 'reliability_strategy_mean_r','reliability_strategy_win',
 'reliability_asset_strategy_mean_r','reliability_asset_strategy_win',
 'expected_r','trailing_win_rate','stability','trailing_giveback',
 'quality','rank_score','cost_r'
]
QGRID=[.10,.20,.30,.40,.50,.60,.70,.80,.90]

def b(x): return x.astype(str).str.lower().isin(['true','1','yes']) if getattr(x,'dtype',None)!=bool else x

def enrich(ev):
    z=ev.copy();z['eval_week']=pd.to_datetime(z['eval_week'],utc=True);z['decision_time']=pd.to_datetime(z['decision_time'],utc=True)
    for c in ['net_pnl','net_r','mfe_r','mae_r','giveback_r',*PREENTRY_FIELDS]:
        if c in z:z[c]=pd.to_numeric(z[c],errors='coerce')
    z['decision_hour']=z.decision_time.dt.hour.astype(float)
    return z

def cohort_flags(z):
    loss=z.net_pnl<-EPS
    return {
      'low_mfe_loss':loss & z.mfe_r.lt(.50),
      'brief_fav_loss':loss & z.mfe_r.ge(.50),
      'severe_mae_loss':loss & z.mae_r.le(-1.0),
      'all_loss':loss
    }

def summarize_cohorts(z):
    f=cohort_flags(z);loss=z.net_pnl<-EPS
    grids={}
    for th in [.10,.20,.25,.30,.40,.50,.60,.75,1.0]:grids[f'loss_mfe_lt_{th:.2f}']=int((loss&z.mfe_r.lt(th)).sum())
    for th in [-.75,-.90,-1.0,-1.10,-1.25,-1.50]:grids[f'loss_mae_le_{th:.2f}']=int((loss&z.mae_r.le(th)).sum())
    return {'trades':len(z),'wins':int((z.net_pnl>EPS).sum()),'losses':int(loss.sum()),'flats':int(z.net_pnl.abs().le(EPS).sum()),'cohorts':{k:int(v.sum()) for k,v in f.items()},'threshold_reconciliation':grids}

def eval_exclusion(seg,mask,target):
    x=seg.loc[mask].copy();flags=cohort_flags(seg);target_n=int(flags[target].loc[mask].sum())
    wins=int((x.net_pnl>EPS).sum());losses=int((x.net_pnl<-EPS).sum());flats=int(x.net_pnl.abs().le(EPS).sum())
    delta=-float(x.net_pnl.sum()) if len(x) else 0.
    winner_pnl=float(x.loc[x.net_pnl>EPS,'net_pnl'].sum()) if len(x) else 0.
    loss_pnl=float(x.loc[x.net_pnl<-EPS,'net_pnl'].sum()) if len(x) else 0.
    precision=target_n/max(len(x),1)
    return {'excluded':int(len(x)),'excluded_pct':float(len(x)/max(len(seg),1)),'target_removed':target_n,'wins_removed':wins,'losses_removed':losses,'flats_removed':flats,'selection_frozen_delta_usd':delta,'winner_pnl_removed':winner_pnl,'loss_pnl_removed':loss_pnl,'target_precision':float(precision)}

def one_dim_rules(early,late,target):
    fields=[c for c in [*PREENTRY_FIELDS,'decision_hour'] if c in early and early[c].notna().sum()>=50]
    rows=[]
    for c in fields:
        vals=early[c].dropna().astype(float)
        for q in QGRID:
            th=float(vals.quantile(q))
            for d in ['le','ge']:
                me=early[c].le(th) if d=='le' else early[c].ge(th);ml=late[c].le(th) if d=='le' else late[c].ge(th)
                e=eval_exclusion(early,me,target);h=eval_exclusion(late,ml,target)
                # Discovery safety: narrow intervention, positive discovery and non-negative untouched validation,
                # with no more than 2 winners removed in either segment. This is ranking evidence only.
                safe=(e['excluded_pct']<=.12 and h['excluded_pct']<=.12 and e['selection_frozen_delta_usd']>EPS and h['selection_frozen_delta_usd']>=-EPS and e['target_removed']>0 and h['target_removed']>0 and e['wins_removed']<=2 and h['wins_removed']<=1)
                rows.append({'target':target,'rule_type':'1D','field1':c,'dir1':d,'threshold1':th,'quantile1':q,'field2':'','dir2':'','threshold2':np.nan,'quantile2':np.nan,**{f'disc_{k}':v for k,v in e.items()},**{f'hold_{k}':v for k,v in h.items()},'discovery_pass':bool(safe)})
    return pd.DataFrame(rows)

def two_dim_rules(early,late,target,one):
    # Intersections only from the strongest one-dimensional discovery rules. Thresholds remain frozen from first 8 weeks.
    pool=one[(one.disc_selection_frozen_delta_usd>0)&(one.disc_target_removed>0)&(one.disc_excluded_pct<=.20)].copy()
    if pool.empty:return pd.DataFrame()
    pool['rank']=5*pool.disc_target_removed+2*pool.hold_target_removed+2*np.maximum(pool.hold_selection_frozen_delta_usd,0)+np.maximum(pool.disc_selection_frozen_delta_usd,0)-4*pool.disc_wins_removed-6*pool.hold_wins_removed
    pool=pool.sort_values('rank',ascending=False).drop_duplicates(['field1','dir1']).head(14)
    rows=[]
    recs=pool.to_dict('records')
    for a,c in itertools.combinations(recs,2):
        if a['field1']==c['field1']:continue
        def mk(seg,r):return seg[r['field1']].le(r['threshold1']) if r['dir1']=='le' else seg[r['field1']].ge(r['threshold1'])
        me=mk(early,a)&mk(early,c);ml=mk(late,a)&mk(late,c)
        e=eval_exclusion(early,me,target);h=eval_exclusion(late,ml,target)
        safe=(e['excluded_pct']<=.10 and h['excluded_pct']<=.10 and e['selection_frozen_delta_usd']>EPS and h['selection_frozen_delta_usd']>=-EPS and e['target_removed']>0 and h['target_removed']>0 and e['wins_removed']<=1 and h['wins_removed']==0)
        rows.append({'target':target,'rule_type':'2D','field1':a['field1'],'dir1':a['dir1'],'threshold1':a['threshold1'],'quantile1':a['quantile1'],'field2':c['field1'],'dir2':c['dir1'],'threshold2':c['threshold1'],'quantile2':c['quantile1'],**{f'disc_{k}':v for k,v in e.items()},**{f'hold_{k}':v for k,v in h.items()},'discovery_pass':bool(safe)})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane2_control')
    res,W,EV,DE,PR=eval_variant(mod,data,cands,opps,fcache,specs,100.0)
    if abs(float(res['end_equity'])-R54_END)>1e-9 or len(EV)!=1102:raise RuntimeError('exact R5.4 control reproduction failed')
    z=enrich(EV);weeks=sorted(z.eval_week.dropna().unique());cut=weeks[-3];early=z[z.eval_week<cut].copy();late=z[z.eval_week>=cut].copy()
    allr=[]
    for target in ['low_mfe_loss','severe_mae_loss','all_loss']:
        one=one_dim_rules(early,late,target);two=two_dim_rules(early,late,target,one);r=pd.concat([one,two],ignore_index=True,sort=False);allr.append(r)
    R=pd.concat(allr,ignore_index=True,sort=False)
    R['repair_score']=6*R.hold_target_removed+3*R.disc_target_removed+2*np.maximum(R.hold_selection_frozen_delta_usd,0)+np.maximum(R.disc_selection_frozen_delta_usd,0)-8*R.hold_wins_removed-4*R.disc_wins_removed-.25*(R.disc_excluded+R.hold_excluded)
    R=R.sort_values(['discovery_pass','repair_score','hold_selection_frozen_delta_usd','disc_selection_frozen_delta_usd'],ascending=[False,False,False,False]).reset_index(drop=True)
    R.to_csv(out/'R5_5_LANE2A_ENTRY_GATES.csv',index=False)
    best=R[R.discovery_pass].iloc[0].to_dict() if R.discovery_pass.any() else (R.iloc[0].to_dict() if len(R) else {})
    status={'state':'R5_5_LANE2A_ENTRY_THESIS_DISCOVERY_COMPLETE','evidence_class':'SELECTION_FROZEN_CHRONOLOGICAL_ENTRY_GATE_DISCOVERY_ONLY','r5_4_control':res,'scope':{'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'segment':{'first8':summarize_cohorts(early),'last3_untouched':summarize_cohorts(late),'all':summarize_cohorts(z)},'preentry_fields_tested':[c for c in [*PREENTRY_FIELDS,'decision_hour'] if c in z],'rules_tested':int(len(R)),'passing_rules':int(R.discovery_pass.sum()) if len(R) else 0,'best_rule':best,'governance':'Thresholds are estimated only from the first 8 evaluation weeks. The final 3 evaluation weeks are untouched validation. Only decision-time/pre-entry fields are eligible; net P&L, net R, MFE, MAE, giveback, exit fields and future path information are targets/evaluation only. Selection-frozen results rank hypotheses and cannot promote R5.5. Any selected entry gate must next survive the full causal portfolio replay with R5.4 selector/reliability logic otherwise unchanged.'}
    (out/'R5_5_LANE2A_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
