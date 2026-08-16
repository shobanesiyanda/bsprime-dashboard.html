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
    for k,g in h.groupby(['asset','strategy'],sort=False):ast[k]=_shrink(_stats(g),strat.get(k[1],glob),8)
    for k,g in h.groupby('hour',sort=False):hour[int(k)]=_shrink(_stats(g),glob,20)
    return {'global':glob,'family':fam,'strategy':strat,'asset_strategy':ast,'hour':hour}

def annotate(promoted,cands,week_start):
    if promoted.empty:return promoted
    r=build_reliability(cands,week_start);z=promoted.copy();t=pd.to_datetime(z.decision_time,utc=True)
    scores=[];means=[];wins=[];ns=[];hour_means=[];strat_ns=[];strat_means=[];strat_wins=[];ast_means=[];ast_wins=[]
    for (_,x),hr in zip(z.iterrows(),t.dt.hour):
        glob=r['global'];ss=r['strategy'].get(x.strategy,glob);aa=r['asset_strategy'].get((x.asset,x.strategy),ss);hh=r['hour'].get(int(hr),glob)
        em=.55*aa['mean']+.30*ss['mean']+.15*hh['mean'];ew=.55*aa['win']+.30*ss['win']+.15*hh['win']
        score=.65*np.tanh(em/.50)+.35*np.clip((ew-.50)*2,-1,1)
        scores.append(float(score));means.append(float(em));wins.append(float(ew));ns.append(int(aa['n']));hour_means.append(float(hh['mean']));strat_ns.append(int(ss['n']));strat_means.append(float(ss['mean']));strat_wins.append(float(ss['win']));ast_means.append(float(aa['mean']));ast_wins.append(float(aa['win']))
    z['reliability_score']=scores;z['reliability_expected_r']=means;z['reliability_win']=wins;z['reliability_n']=ns;z['reliability_strategy_n']=strat_ns;z['reliability_hour_mean_r']=hour_means;z['reliability_strategy_mean_r']=strat_means;z['reliability_strategy_win']=strat_wins;z['reliability_asset_strategy_mean_r']=ast_means;z['reliability_asset_strategy_win']=ast_wins
    return z

def promote(cands,week_start,mode):
    p,conf,stats=base.promote_for_week(cands,week_start)
    if p.empty:return p,conf,stats
    z=annotate(p,cands,week_start)
    if mode in {'soft','combined','risk','high_confidence_rank'}:z['rank_score']=z.rank_score+.18*z.reliability_score
    if mode in {'gate','combined'}:
        bad=((z.reliability_strategy_n>=12)&(z.reliability_expected_r<-.04))|((z.reliability_n>=10)&(z.reliability_expected_r<-.08))
        bad |= ((z.reliability_hour_mean_r<-.06)&(z.reliability_expected_r<0)&(z.reliability_strategy_n>=12))
        z=z[~bad].copy()
    elif mode=='high_confidence_gate':
        # Stronger evidence requirement before suppressing an otherwise baseline-qualified trade.
        # Strategy and route-strategy histories must be materially negative; noisy hours cannot gate alone.
        strat_bad=(z.reliability_strategy_n>=20)&(z.reliability_strategy_mean_r<-.06)&(z.reliability_strategy_win<.43)
        route_bad=(z.reliability_n>=15)&(z.reliability_asset_strategy_mean_r<-.10)&(z.reliability_asset_strategy_win<.40)
        z=z[~(strat_bad|route_bad)].copy()
    elif mode=='strategy_only_gate':
        # No hour or asset-route veto. Requires broad strategy evidence before exclusion.
        bad=(z.reliability_strategy_n>=16)&(z.reliability_strategy_mean_r<-.05)&(z.reliability_strategy_win<.44)
        z=z[~bad].copy()
    elif mode=='high_confidence_rank':
        # Ranking only; never remove a trade from the baseline-qualified set.
        pass
    if mode=='risk':
        weak=z.reliability_score<-.20;z.loc[weak,'risk_fraction']=z.loc[weak,'risk_fraction']*.5
    return z,conf,stats
