from __future__ import annotations
import numpy as np,pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
from reliability_common import annotate

NEW_CONTEXT_FAMILIES={'ORDER_FLOW_MICROSTRUCTURE','CRYPTO_FUNDING_BASIS'}
CONTEXT_FAMILIES={'ORDER_FLOW_MICROSTRUCTURE','CRYPTO_FUNDING_BASIS','EVENT_RESPONSE'}

def _context_table(cands):
    x=cands[cands.strategy_family.isin(CONTEXT_FAMILIES)].copy()
    if x.empty:return x
    x['decision_time']=pd.to_datetime(x.decision_time,utc=True)
    return x.sort_values('decision_time')

def annotate_context(promoted,all_cands):
    if promoted.empty:return promoted
    ctx=_context_table(all_cands);z=promoted.copy();z['decision_time']=pd.to_datetime(z.decision_time,utc=True)
    scores=[];nfac=[];conflicts=[];funding=[];orderflow=[];event=[]
    for _,r in z.iterrows():
        q=ctx[(ctx.asset==r.asset)&(ctx.decision_time<=r.decision_time)&(ctx.decision_time>=r.decision_time-pd.Timedelta(minutes=30))]
        # One most recent observation per family prevents high-frequency order-flow from overpowering slower evidence.
        latest=[]
        for fam,g in q.groupby('strategy_family'):
            latest.append(g.sort_values('decision_time').iloc[-1])
        if latest:
            vals=[];fams={}
            for a in latest:
                w=float(np.clip(a.quality,.55,.90));v=int(a.direction)*w;vals.append((v,w));fams[str(a.strategy_family)]=int(a.direction)*w
            raw=sum(v for v,w in vals)/max(sum(w for v,w in vals),1e-12);align=float(int(r.direction)*raw)
        else:align=0.;fams={}
        scores.append(align);nfac.append(len(latest));conflicts.append(int(sum(1 for a in latest if int(a.direction)!=int(r.direction))));funding.append(float(int(r.direction)*fams.get('CRYPTO_FUNDING_BASIS',0.)));orderflow.append(float(int(r.direction)*fams.get('ORDER_FLOW_MICROSTRUCTURE',0.)));event.append(float(int(r.direction)*fams.get('EVENT_RESPONSE',0.)))
    z['intelligence_alignment']=scores;z['intelligence_family_count']=nfac;z['intelligence_conflicts']=conflicts;z['funding_alignment']=funding;z['orderflow_alignment']=orderflow;z['event_alignment']=event
    return z

def promote(cands,week_start,mode):
    # New funding/order-flow families first inform existing baseline strategies; they do not get independent capital in this experiment.
    tradable=cands[~cands.strategy_family.isin(NEW_CONTEXT_FAMILIES)].copy()
    p,conf,stats=base.promote_for_week(tradable,week_start)
    if p.empty:return p,conf,stats
    if mode.startswith('reliability'):
        p=annotate(p,tradable,week_start)
        bad=((p.reliability_strategy_n>=12)&(p.reliability_expected_r<-.04))|((p.reliability_n>=10)&(p.reliability_expected_r<-.08))
        bad |= ((p.reliability_hour_mean_r<-.06)&(p.reliability_expected_r<0)&(p.reliability_strategy_n>=12))
        p=p[~bad].copy()
        if p.empty:return p,conf,stats
    z=annotate_context(p,cands)
    if mode in {'rank','reliability_rank','reliability_gate'}:
        z['rank_score']=z.rank_score+.10*z.intelligence_alignment
    if mode in {'conflict_gate','reliability_gate'}:
        # Veto only multi-source conflict. A single new source can down-rank but cannot suppress a baseline-qualified trade.
        veto=(z.intelligence_family_count>=2)&(z.intelligence_conflicts>=2)&(z.intelligence_alignment<-.45)
        z=z[~veto].copy()
    return z,conf,stats
