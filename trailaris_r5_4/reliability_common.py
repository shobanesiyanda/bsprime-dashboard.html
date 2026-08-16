from __future__ import annotations
import numpy as np,pandas as pd
import R5_BASE_CAUSAL_ENGINE as base

def _stats(g):
    n=len(g)
    if not n:return {'n':0,'mean':0.,'win':.5,'sd':1.}
    r=pd.to_numeric(g.net_r,errors='coerce').dropna()
    if len(r)==0:return {'n':0,'mean':0.,'win':.5,'sd':1.}
    return {'n':len(r),'mean':float(r.mean()),'win':float((r>0).mean()),'sd':float(r.std(ddof=1)) if len(r)>1 else 1.}

def _shrink(child,prior,k):
    n=float(child['n']);return {'n':int(n),'mean':(n*child['mean']+k*prior['mean'])/(n+k),'win':(n*child['win']+k*prior['win'])/(n+k),'sd':child['sd']}

def build_reliability(cands,week_start,lookback_days=56):
    h=cands[(cands.exit_time<week_start)&(cands.exit_time>=week_start-pd.Timedelta(days=lookback_days))].copy()
    if h.empty:return {'global':{'n':0,'mean':0.,'win':.5,'sd':1.},'family':{},'strategy':{},'asset_strategy':{},'hour':{}}
    h['decision_time']=pd.to_datetime(h.decision_time,utc=True);h['hour']=h.decision_time.dt.hour
    glob=_stats(h)
    fam={k:_stats(g) for k,g in h.groupby('strategy_family',sort=False)}
    strat={};ast={};hour={}
    for k,g in h.groupby('strategy',sort=False):
        prior=fam.get(str(g.strategy_family.iloc[0]),glob);strat[k]=_shrink(_stats(g),prior,12)
    for k,g in h.groupby(['asset','strategy'],sort=False):
        ast[k]=_shrink(_stats(g),strat.get(k[1],glob),8)
    for k,g in h.groupby('hour',sort=False):hour[int(k)]=_shrink(_stats(g),glob,20)
    return {'global':glob,'family':fam,'strategy':strat,'asset_strategy':ast,'hour':hour}

def annotate(promoted,cands,week_start):
    if promoted.empty:return promoted
    r=build_reliability(cands,week_start);z=promoted.copy();t=pd.to_datetime(z.decision_time,utc=True)
    scores=[];means=[];wins=[];ns=[];hour_means=[];strat_ns=[]
    for (_,x),hr in zip(z.iterrows(),t.dt.hour):
        glob=r['global'];ss=r['strategy'].get(x.strategy,glob);aa=r['asset_strategy'].get((x.asset,x.strategy),ss);hh=r['hour'].get(int(hr),glob)
        # Asset-strategy estimate gets most weight; strategy and market-hour reliability regularize sparse route samples.
        em=.55*aa['mean']+.30*ss['mean']+.15*hh['mean'];ew=.55*aa['win']+.30*ss['win']+.15*hh['win']
        score=.65*np.tanh(em/.50)+.35*np.clip((ew-.50)*2,-1,1)
        scores.append(float(score));means.append(float(em));wins.append(float(ew));ns.append(int(aa['n']));hour_means.append(float(hh['mean']));strat_ns.append(int(ss['n']))
    z['reliability_score']=scores;z['reliability_expected_r']=means;z['reliability_win']=wins;z['reliability_n']=ns;z['reliability_strategy_n']=strat_ns;z['reliability_hour_mean_r']=hour_means
    return z

def promote(cands,week_start,mode):
    p,conf,stats=base.promote_for_week(cands,week_start)
    if p.empty:return p,conf,stats
    z=annotate(p,cands,week_start)
    if mode in {'soft','combined','risk'}:
        z['rank_score']=z.rank_score+.18*z.reliability_score
    if mode in {'gate','combined'}:
        # Fail weak slices only after sufficient prior evidence. No current/future outcome enters the gate.
        bad=((z.reliability_strategy_n>=12)&(z.reliability_expected_r<-.04))|((z.reliability_n>=10)&(z.reliability_expected_r<-.08))
        # A strongly bad market hour can gate only with already negative hierarchical evidence.
        bad |= ((z.reliability_hour_mean_r<-.06)&(z.reliability_expected_r<0)&(z.reliability_strategy_n>=12))
        z=z[~bad].copy()
    if mode=='risk':
        # Only reduce risk on weak evidence; never increase leverage above the frozen baseline.
        weak=z.reliability_score<-.20;z.loc[weak,'risk_fraction']=z.loc[weak,'risk_fraction']*.5
    return z,conf,stats
