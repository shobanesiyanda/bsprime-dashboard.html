from __future__ import annotations
import heapq
import numpy as np
import pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import trailaris_r4_full_universe_loop as r4
import variant_hierarchical_combined as frozen

# Promotion/reliability/configuration remain exactly frozen R5.4.
promote_for_week = frozen.promote_for_week

# No new thresholds are fitted here. These are frozen R5.4 Tier-B causal evidence
# requirements plus a positive hierarchical reliability sign.
RECOVERY_MAX_RISK_FRACTION = .0025
RECOVERY_MIN_QUALITY = .68
RECOVERY_MIN_EXPECTED_R = .10
RECOVERY_MIN_TRAILING_WIN = .45
RECOVERY_MAX_COST_R = .35


def replay_r5(cands,specs,start=100.,feature_cache=None):
    if cands.empty:return pd.DataFrame(),pd.DataFrame()
    c=cands.copy();c['decision_time']=pd.to_datetime(c.decision_time,utc=True);c['exit_time']=pd.to_datetime(c.exit_time,utc=True)
    c=c.sort_values(['decision_time','rank_score'],ascending=[True,False]).reset_index(drop=True)
    balance=float(start);peak=balance;week=None;week_start=balance;objective=False;floor=False
    openpos={};heap=[];events=[];dec=[];seq=0;factors=np.zeros(len(r4.FACTORS));open_risk=0.;open_margin=0.;config_week_r={};asset_week_r={}

    def mark(p,t):
        x=feature_cache.get(p['asset']) if feature_cache else None
        if x is None:return None
        lo=int(x.timestamp.searchsorted(pd.Timestamp(p['entry_time']),side='left'));hi=int(x.timestamp.searchsorted(pd.Timestamp(t),side='right'))
        if hi<=lo:return None
        z=x.iloc[lo:hi];px=float(z.close.iloc[-1]);ent=float(p['entry_price']);sd=max(float(p['stop_distance']),1e-12);dr=int(p['direction']);cur=(px-ent)*dr/sd-float(p.get('cost_r',0))
        fav=(float(z.high.max())-ent)/sd if dr==1 else (ent-float(z.low.min()))/sd;fl=-1.
        if fav>=.75:fl=0.
        if fav>=1.5:fl=.5
        if fav>=2.:fl=max(fl,fav-.7)
        return dict(mark=px,current_r=cur,mfe=fav,floor=fl)

    def mtm(t):
        q=balance
        for p in openpos.values():
            s=mark(p,t)
            if s:q+=float(p['risk_cash'])*s['current_r']
        return q

    def close_until(t):
        nonlocal balance,peak,factors,open_risk,open_margin
        while heap and heap[0][0]<=t:
            _,_,pid=heapq.heappop(heap);p=openpos.pop(pid,None)
            if p is None:continue
            pnl=float(p['risk_cash'])*float(p['net_r']);before=mtm(p['exit_time']);balance+=pnl;factors-=r4.factor_vec(p['asset'],int(p['direction']));open_risk=max(0,open_risk-float(p['risk_cash']));open_margin=max(0,open_margin-float(p['margin']));eq=mtm(p['exit_time']);peak=max(peak,eq);dd=eq/peak-1
            events.append({**p,'timestamp':p['exit_time'],'net_pnl':pnl,'equity_before':before,'equity':eq,'balance_after':balance,'drawdown':dd})
            wk=r4.week_key(p['exit_time']);config_week_r[(wk,p['asset'],p['strategy'])]=config_week_r.get((wk,p['asset'],p['strategy']),0)+float(p['net_r']);asset_week_r[(wk,p['asset'])]=asset_week_r.get((wk,p['asset']),0)+float(p['net_r'])

    def recycle(new,t,u):
        nonlocal balance,peak,factors,open_risk,open_margin
        choices=[]
        for pid,p in openpos.items():
            if bool(p.get('is_addon',False)):continue
            s=mark(p,t)
            if not s or s['floor']<0:continue
            u0=float(p.get('entry_utility',p.get('rank_score',p.get('quality',0))))
            if u<=u0+.05:continue
            nf=factors-r4.factor_vec(p['asset'],int(p['direction']))+r4.factor_vec(new.asset,int(new.direction))
            if np.abs(nf).max()>1.25:continue
            choices.append((u0,pid,p,s))
        if not choices:return None
        _,pid,p,s=min(choices,key=lambda z:z[0]);openpos.pop(pid,None);rr=float(s['current_r']);before=mtm(t);pnl=float(p['risk_cash'])*rr;balance+=pnl;factors-=r4.factor_vec(p['asset'],int(p['direction']));open_risk=max(0,open_risk-float(p['risk_cash']));open_margin=max(0,open_margin-float(p['margin']));eq=mtm(t);peak=max(peak,eq)
        events.append({**p,'timestamp':pd.Timestamp(t),'exit_time':pd.Timestamp(t),'exit_price':s['mark'],'net_r':rr,'net_pnl':pnl,'equity_before':before,'equity':eq,'balance_after':balance,'drawdown':eq/peak-1,'exit_reason':'DYNAMIC_CAPITAL_RECYCLE_R5'})
        return pid

    def recovery_qualified(r):
        return (float(r.quality)>=RECOVERY_MIN_QUALITY and float(r.expected_r)>RECOVERY_MIN_EXPECTED_R and
                float(r.trailing_win_rate)>=RECOVERY_MIN_TRAILING_WIN and float(r.cost_r)<=RECOVERY_MAX_COST_R and
                float(r.get('reliability_score',0.0))>0.0)

    def recycle_recovery(new,t,u,new_risk,new_margin):
        """Risk- and margin-neutral protected-position substitution only."""
        nonlocal balance,peak,factors,open_risk,open_margin
        choices=[]
        for pid,p in openpos.items():
            if bool(p.get('is_addon',False)):continue
            s=mark(p,t)
            if not s or s['floor']<0:continue
            u0=float(p.get('entry_utility',p.get('rank_score',p.get('quality',0))))
            if u<=u0+.05:continue
            # The incoming recovery candidate cannot consume more risk cash or margin
            # than the protected incumbent it replaces.
            if float(new_risk)>float(p['risk_cash'])+1e-12:continue
            if float(new_margin)>float(p['margin'])+1e-12:continue
            nf=factors-r4.factor_vec(p['asset'],int(p['direction']))+r4.factor_vec(new.asset,int(new.direction))
            if np.abs(nf).max()>1.25:continue
            choices.append((u0,pid,p,s))
        if not choices:return None
        _,pid,p,s=min(choices,key=lambda z:z[0]);openpos.pop(pid,None);rr=float(s['current_r']);before=mtm(t);pnl=float(p['risk_cash'])*rr;balance+=pnl;factors-=r4.factor_vec(p['asset'],int(p['direction']));open_risk=max(0,open_risk-float(p['risk_cash']));open_margin=max(0,open_margin-float(p['margin']));eq=mtm(t);peak=max(peak,eq)
        events.append({**p,'timestamp':pd.Timestamp(t),'exit_time':pd.Timestamp(t),'exit_price':s['mark'],'net_r':rr,'net_pnl':pnl,'equity_before':before,'equity':eq,'balance_after':balance,'drawdown':eq/peak-1,'exit_reason':'RECOVERY_RISK_NEUTRAL_REALLOCATION_R5_5'})
        return pid

    for t,due in c.groupby('decision_time',sort=True):
        wk=r4.week_key(t)
        if week is None:week=wk;week_start=start
        else:
            while t>=week+pd.Timedelta(days=7):
                b=week+pd.Timedelta(days=7);close_until(b);week=b;week_start=mtm(b);objective=False;floor=False
        close_until(t);eq=mtm(t);growth=eq/week_start-1 if week_start else 0
        if growth>=.05:objective=True
        if objective and growth<=.04:floor=True
        rs=0 if floor else (.25 if growth>=.10 else 1.)
        if growth<=-.05:rs=0
        elif growth<=-.035:rs*=.25
        elif growth<=-.02:rs*=.50
        elif growth<=-.01:rs*=.75
        risk_budget=min(.01,max(0,growth-.04)) if objective else .01

        for _,r in due.sort_values('rank_score',ascending=False).iterrows():
            rq=bool(floor and recovery_qualified(r))
            base_rf=float(r.get('risk_fraction',.005))
            rf=min(base_rf,RECOVERY_MAX_RISK_FRACTION) if rq else base_rf*rs
            lot,risk,margin=r4.size_trade(r.asset,eq,rf,float(r.stop_distance),specs);reason='SELECTED';proj=factors+r4.factor_vec(r.asset,int(r.direction));ck=(wk,r.asset,r.strategy);ak=(wk,r.asset)
            if bool(r.is_addon):
                if not r.parent_id or r.parent_id not in openpos:reason='ADDON_PARENT_NOT_OPEN'
                else:
                    ps=mark(openpos[r.parent_id],t)
                    if not ps or ps['floor']<0:reason='ADDON_PARENT_NOT_PROTECTED'
            same=[p for p in openpos.values() if p['asset']==r.asset and not bool(p.get('is_addon',False))]
            if reason=='SELECTED' and same and not bool(r.is_addon):
                protected=[(p,mark(p,t)) for p in same];indep=all(p['strategy_family']!=r.strategy_family for p,_ in protected);same_dir=all(int(p['direction'])==int(r.direction) for p,_ in protected);prot=all(s and s['floor']>=0 for _,s in protected);asset_r=sum(float(p['risk_cash']) for p in same)
                if not (indep and same_dir and prot and (asset_r+risk)/max(eq,1e-12)<=.0075):reason='SAME_ASSET_NOT_PROTECTED_INDEPENDENT'
            if reason=='SELECTED' and config_week_r.get(ck,0)<=-2:reason='CONFIG_WEEKLY_LOSS_BREAKER'
            if reason=='SELECTED' and asset_week_r.get(ak,0)<=-3:reason='ASSET_WEEKLY_LOSS_BREAKER'
            if reason=='SELECTED' and lot<=0:reason='LOT_BELOW_MIN'
            if reason=='SELECTED' and (open_risk+risk)/max(eq,1e-12)>risk_budget+1e-12:reason='OPEN_RISK_BUDGET'
            if reason=='SELECTED' and (open_margin+margin)/max(eq,1e-12)>.60:reason='MARGIN_BUDGET'
            if reason=='SELECTED' and np.abs(proj).max()>1.25:reason='FACTOR_DUPLICATION'
            u=float(r.rank_score)-.20*(margin/max(eq,1e-12));recycled='';recovery_reallocated=False

            # New R5.5 logic: in the protected weekly floor state, no new risk budget is
            # created. A causally strong candidate may replace a protected incumbent only
            # if risk cash, margin, utility and factor conditions all improve/preserve.
            if rq and reason in {'OPEN_RISK_BUDGET','MARGIN_BUDGET','FACTOR_DUPLICATION'} and u>=.12:
                pre_risk=open_risk;pre_margin=open_margin
                pid=recycle_recovery(r,t,u,risk,margin)
                if pid:
                    recycled=pid;eq=mtm(t);lot,risk,margin=r4.size_trade(r.asset,eq,rf,float(r.stop_distance),specs);proj=factors+r4.factor_vec(r.asset,int(r.direction))
                    risk_neutral=(open_risk+risk)<=pre_risk+1e-12;margin_neutral=(open_margin+margin)<=pre_margin+1e-12
                    if lot>0 and risk_neutral and margin_neutral and np.abs(proj).max()<=1.25:
                        reason='SELECTED';recovery_reallocated=True
                    else:reason='RECOVERY_REALLOCATION_NOT_NEUTRAL'
            elif reason in {'OPEN_RISK_BUDGET','MARGIN_BUDGET','FACTOR_DUPLICATION'} and u>=.12:
                pid=recycle(r,t,u)
                if pid:
                    recycled=pid;eq=mtm(t);lot,risk,margin=r4.size_trade(r.asset,eq,rf,float(r.stop_distance),specs);proj=factors+r4.factor_vec(r.asset,int(r.direction))
                    if lot>0 and (open_risk+risk)/max(eq,1e-12)<=risk_budget+1e-12 and (open_margin+margin)/max(eq,1e-12)<=.60 and np.abs(proj).max()<=1.25:reason='SELECTED'
                    else:reason='RECYCLE_DID_NOT_FREE_ENOUGH_CAPACITY'

            dec.append(dict(decision_time=t,campaign_id=r.campaign_id,asset=r.asset,strategy=r.strategy,strategy_family=r.strategy_family,promotion_tier=r.promotion_tier,selected=reason=='SELECTED',reason=reason,quality=float(r.quality),expected_r=float(r.expected_r),rank_score=float(r.rank_score),utility=u,recycled_incumbent_id=recycled,recovery_reallocated=recovery_reallocated,recovery_qualified=rq,equity=eq,week_growth=growth,risk_budget=risk_budget,lot=lot,risk_cash=risk,margin=margin,projected_open_risk_pct=100*(open_risk+risk)/max(eq,1e-12)))
            if reason=='SELECTED':
                p=r.to_dict();p.update(lot=lot,risk_cash=risk,margin=margin,equity_at_entry=eq,balance_at_entry=balance,entry_utility=u,recovery_reallocated=recovery_reallocated);openpos[r.campaign_id]=p;factors=proj;open_risk+=risk;open_margin+=margin;seq+=1;heapq.heappush(heap,(pd.Timestamp(r.exit_time),seq,r.campaign_id))

    close_until(pd.Timestamp.max.tz_localize('UTC'))
    ev=pd.DataFrame(events);de=pd.DataFrame(dec)
    if len(ev):ev=ev.sort_values('timestamp').reset_index(drop=True)
    return ev,de
