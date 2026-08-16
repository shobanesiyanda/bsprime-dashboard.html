#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import numpy as np,pandas as pd

R54_END=337.4231242104393
EPS=1e-12
FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
BARELY=.20
SEVERE=1.5063700312482555
QGRID=[.10,.20,.30,.40,.50,.60,.70,.80,.90]
FX={'EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF'}
METALS={'XAUUSD','XAGUSD'};CRYPTO={'BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD'}
INDICES={'US100','US500','US30','GER40','UK100','JP225'};ENERGY={'WTI','BRENT','NATGAS'};SYNTH={'V75','V50','V100','V25','V10'}
FEATURES=['trend_alignment','h1_strength','compression','signed_z30','ema20_distance_stop_r','stop_atr_ratio','body_frac','signed_impulse_atr','signed_mom6_atr','directional_range20_pos','signed_ret1_atr','adverse_wick_frac']

def asset_class(a):
    if a in FX:return 'FX'
    if a in METALS:return 'METALS'
    if a in CRYPTO:return 'CRYPTO'
    if a in INDICES:return 'INDICES'
    if a in ENERGY:return 'ENERGY'
    return 'SYNTH'

def num(x,d=np.nan):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def enrich_market_state(ev,fcache):
    rows=[]
    for _,r in ev.iterrows():
        a=str(r.asset);x=fcache.get(a);dt=pd.Timestamp(r.decision_time);dr=int(num(r.direction,1));sd=max(abs(num(r.stop_distance)),EPS)
        feat={k:np.nan for k in FEATURES}
        if x is not None and len(x):
            ts=pd.to_datetime(x.timestamp,utc=True);i=int(ts.searchsorted(dt,side='right')-1)
            if i>=0:
                b=x.iloc[i];atr=max(num(b.get('atr')),EPS);close=num(b.get('close'));open_=num(b.get('open'));ema=num(b.get('ema20'));trend=num(b.get('h1_trend'),0);strength=num(b.get('h1_strength'));comp=num(b.get('compression'));z=num(b.get('z30'));body=num(b.get('body_frac'));ret1=num(b.get('ret1'))
                feat['trend_alignment']=dr*trend
                feat['h1_strength']=strength;feat['compression']=comp;feat['signed_z30']=dr*z
                feat['ema20_distance_stop_r']=dr*(close-ema)/sd if np.isfinite(close) and np.isfinite(ema) else np.nan
                feat['stop_atr_ratio']=sd/atr;feat['body_frac']=body;feat['signed_impulse_atr']=dr*(close-open_)/atr if np.isfinite(close) and np.isfinite(open_) else np.nan
                if i>=6:feat['signed_mom6_atr']=dr*(close-num(x.close.iloc[i-6]))/atr
                hi=num(b.get('rng20_hi'));lo=num(b.get('rng20_lo'))
                if np.isfinite(hi) and np.isfinite(lo) and hi>lo:
                    p=(close-lo)/(hi-lo);feat['directional_range20_pos']=p if dr==1 else 1-p
                if np.isfinite(ret1) and np.isfinite(close) and close!=0:feat['signed_ret1_atr']=dr*(ret1*close)/atr
                rng=max(num(b.get('range'),num(b.get('high'))-num(b.get('low'))),EPS)
                if dr==1:feat['adverse_wick_frac']=(min(num(b.get('open')),close)-num(b.get('low')))/rng
                else:feat['adverse_wick_frac']=(num(b.get('high'))-max(num(b.get('open')),close))/rng
        q=r.to_dict();q.update(feat);q['asset_class']=asset_class(a);rows.append(q)
    return pd.DataFrame(rows)

def target_mask(z,target):
    loss=pd.to_numeric(z.net_pnl,errors='coerce')<-EPS
    if target=='barely_developed_loss':return loss & pd.to_numeric(z.mfe_r,errors='coerce').lt(BARELY)
    if target=='severe_mae_loss':return loss & pd.to_numeric(z.mae_r,errors='coerce').ge(SEVERE)
    raise KeyError(target)

def segment_stats(seg,mask,target):
    x=seg.loc[mask];tm=target_mask(seg,target);pnl=pd.to_numeric(x.net_pnl,errors='coerce').fillna(0)
    wins=pnl[pnl>EPS];losses=pnl[pnl<-EPS];flats=pnl[pnl.abs()<=EPS]
    tgt=int(tm.loc[mask].sum());base_prev=float(target_mask(seg,target).mean())
    precision=tgt/max(len(x),1);delta=-float(pnl.sum())
    return {'excluded':int(len(x)),'excluded_pct':float(len(x)/max(len(seg),1)),'target_removed':tgt,'target_precision':precision,'target_lift':precision/max(base_prev,EPS),'wins_removed':int(len(wins)),'losses_removed':int(len(losses)),'flats_removed':int(len(flats)),'winner_pnl_removed':float(wins.sum()),'loss_pnl_removed':float(losses.sum()),'selection_frozen_delta_usd':delta}

def scopes(early):
    out=[('GLOBAL','ALL')]
    for c,n in early.asset_class.value_counts().items():
        if n>=60:out.append(('ASSET_CLASS',c))
    for c,n in early.strategy_family.value_counts().items():
        if n>=45:out.append(('STRATEGY_FAMILY',c))
    return out

def scope_mask(z,typ,val):
    if typ=='GLOBAL':return pd.Series(True,index=z.index)
    if typ=='ASSET_CLASS':return z.asset_class.eq(val)
    return z.strategy_family.eq(val)

def test_1d(early,valid,target):
    rows=[]
    for st,sv in scopes(early):
        se=scope_mask(early,st,sv);svm=scope_mask(valid,st,sv)
        for f in FEATURES:
            vals=pd.to_numeric(early.loc[se,f],errors='coerce').dropna()
            if len(vals)<35:continue
            for q in QGRID:
                th=float(vals.quantile(q))
                for d in ['le','ge']:
                    me=se & (pd.to_numeric(early[f],errors='coerce').le(th) if d=='le' else pd.to_numeric(early[f],errors='coerce').ge(th))
                    mv=svm & (pd.to_numeric(valid[f],errors='coerce').le(th) if d=='le' else pd.to_numeric(valid[f],errors='coerce').ge(th))
                    e=segment_stats(early,me,target);v=segment_stats(valid,mv,target)
                    ok=(e['excluded_pct']<=.10 and v['excluded_pct']<=.10 and e['target_removed']>0 and v['target_removed']>0 and e['target_lift']>=1.5 and v['target_lift']>=1.25 and e['selection_frozen_delta_usd']>EPS and v['selection_frozen_delta_usd']>=-EPS and e['winner_pnl_removed']<=abs(e['loss_pnl_removed'])*.85+EPS and v['winner_pnl_removed']<=abs(v['loss_pnl_removed'])+EPS)
                    rows.append({'target':target,'scope_type':st,'scope_value':sv,'rule_type':'1D','feature1':f,'dir1':d,'threshold1':th,'q1':q,'feature2':'','dir2':'','threshold2':np.nan,'q2':np.nan,**{f'disc_{k}':vv for k,vv in e.items()},**{f'valid_{k}':vv for k,vv in v.items()},'clean_pass':bool(ok)})
    return pd.DataFrame(rows)

def test_2d(early,valid,target,one):
    pool=one[(one.disc_selection_frozen_delta_usd>0)&(one.disc_target_removed>0)&(one.disc_target_lift>=1.25)&(one.disc_excluded_pct<=.15)].copy()
    if pool.empty:return pd.DataFrame()
    pool['rank']=5*pool.disc_target_removed+6*pool.valid_target_removed+3*np.maximum(pool.valid_selection_frozen_delta_usd,0)+np.maximum(pool.disc_selection_frozen_delta_usd,0)-2*pool.disc_wins_removed-3*pool.valid_wins_removed
    pool=pool.sort_values('rank',ascending=False).drop_duplicates(['scope_type','scope_value','feature1','dir1']).head(24)
    rows=[];recs=pool.to_dict('records')
    for a,b in itertools.combinations(recs,2):
        if (a['scope_type'],a['scope_value'])!=(b['scope_type'],b['scope_value']) or a['feature1']==b['feature1']:continue
        st,sv=a['scope_type'],a['scope_value'];se=scope_mask(early,st,sv);svm=scope_mask(valid,st,sv)
        def cond(z,r):
            s=pd.to_numeric(z[r['feature1']],errors='coerce');return s.le(r['threshold1']) if r['dir1']=='le' else s.ge(r['threshold1'])
        me=se&cond(early,a)&cond(early,b);mv=svm&cond(valid,a)&cond(valid,b)
        e=segment_stats(early,me,target);v=segment_stats(valid,mv,target)
        ok=(e['excluded_pct']<=.08 and v['excluded_pct']<=.08 and e['target_removed']>0 and v['target_removed']>0 and e['target_lift']>=1.75 and v['target_lift']>=1.35 and e['selection_frozen_delta_usd']>EPS and v['selection_frozen_delta_usd']>=-EPS and e['winner_pnl_removed']<=abs(e['loss_pnl_removed'])*.75+EPS and v['winner_pnl_removed']<=abs(v['loss_pnl_removed'])*.9+EPS)
        rows.append({'target':target,'scope_type':st,'scope_value':sv,'rule_type':'2D','feature1':a['feature1'],'dir1':a['dir1'],'threshold1':a['threshold1'],'q1':a['q1'],'feature2':b['feature1'],'dir2':b['dir1'],'threshold2':b['threshold1'],'q2':b['q1'],**{f'disc_{k}':vv for k,vv in e.items()},**{f'valid_{k}':vv for k,vv in v.items()},'clean_pass':bool(ok)})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy');mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_lane4_control')
    res,W,EV,DE,PR=eval_variant(mod,data,cands,opps,fcache,specs,100.0)
    if abs(float(res['end_equity'])-R54_END)>1e-9 or len(EV)!=1102:raise RuntimeError('R5.4 control reproduction failed')
    z=enrich_market_state(EV,fcache);z['eval_week']=pd.to_datetime(z.eval_week,utc=True);z['decision_time']=pd.to_datetime(z.decision_time,utc=True);weeks=sorted(z.eval_week.unique());cut=weeks[-3]
    early=z[z.eval_week<cut].copy();valid=z[(z.eval_week>=cut)&(z.decision_time<FRESH0)].copy();fresh=z[z.decision_time>=FRESH0].copy()
    allr=[]
    for target in ['barely_developed_loss','severe_mae_loss']:
        one=test_1d(early,valid,target);two=test_2d(early,valid,target,one);allr.append(pd.concat([one,two],ignore_index=True,sort=False))
    R=pd.concat(allr,ignore_index=True,sort=False);R['score']=7*R.valid_target_removed+4*R.disc_target_removed+3*np.maximum(R.valid_selection_frozen_delta_usd,0)+np.maximum(R.disc_selection_frozen_delta_usd,0)+2*R.valid_target_lift+R.disc_target_lift-3*R.valid_wins_removed-2*R.disc_wins_removed
    R=R.sort_values(['clean_pass','score','valid_selection_frozen_delta_usd','disc_selection_frozen_delta_usd'],ascending=[False,False,False,False]).reset_index(drop=True);R.to_csv(out/'R5_5_LANE4A_MARKET_STATE_RULES.csv',index=False)
    best=R[R.clean_pass].iloc[0].to_dict() if R.clean_pass.any() else (R.iloc[0].to_dict() if len(R) else {})
    status={'state':'R5_5_LANE4A_MARKET_STATE_DISCOVERY_COMPLETE','evidence_class':'SELECTION_FROZEN_CAUSAL_MARKET_STATE_DISCOVERY','r5_4_control':res,'scope':{'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'partition':{'discovery_events':len(early),'validation_pre_aug13_events':len(valid),'aug13_14_quarantined_events':len(fresh),'fresh_outcomes_used_for_selection':False},'features':FEATURES,'scope_types':['GLOBAL','ASSET_CLASS','STRATEGY_FAMILY'],'rules_tested':len(R),'passing_rules':int(R.clean_pass.sum()),'passing_by_target':{t:int(((R.target==t)&R.clean_pass).sum()) for t in ['barely_developed_loss','severe_mae_loss']},'best_rule':best,'governance':'Every market-state feature is read from the completed feature-cache bar at or before the immutable R5.4 decision timestamp. No future MFE/MAE/P&L is used as an input. Thresholds are estimated only in the first8 discovery sample, validated only on later pre-Aug13 decisions, and Aug13-14 outcomes remain quarantined. Passing rules are discovery evidence only and must survive authentic full replay before becoming pre-forward candidates.'}
    (out/'R5_5_LANE4A_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
