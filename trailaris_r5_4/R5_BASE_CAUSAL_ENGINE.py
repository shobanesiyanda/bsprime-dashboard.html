#!/usr/bin/env python3
import argparse, json, os, heapq
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
import trailaris_r4_full_universe_loop as r4

ROUTES=r4.ROUTES
FINE_HOURS=[(0,24),(0,3),(3,7),(7,10),(10,13),(13,16),(16,19),(19,24)]
QGRID=np.array([.55,.60,.64,.68,.72,.76])
CGRID=np.array([.20,.35,.50])


def build_universe(rawdir:Path, mode='research-proxy'):
    data,v=r4.validate_34(rawdir,mode)
    if len(data)!=34: raise RuntimeError(f'34-route gate failed: {len(data)}/34')
    jobs=[(a,str(rawdir/f'{a}_M1_normalized.csv')) for a in ROUTES]
    bundles={};workers=min(8,max(2,(os.cpu_count() or 4)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(r4.build_asset_components,j):j[0] for j in jobs}
        for f in as_completed(futs):
            a,c,xf,o=f.result();bundles[a]=(c,xf,o)
    allc=[];allo=[];fcache={}
    for a in ROUTES:
        c,xf,o=bundles[a];fcache[a]=xf
        if len(c):allc.append(c)
        if len(o):allo.append(o)
    base=pd.concat(allc,ignore_index=True)
    extras=[r4.generate_relative_value_candidates(fcache),r4.generate_intermarket_candidates(fcache)]
    ff=r4.load_factor_feed(rawdir/'Trailaris_R3_FactorFeed.csv')
    extras.append(r4.generate_factor_candidates(fcache,ff,mode))
    cands=pd.concat([x for x in [base,*extras] if x is not None and len(x)],ignore_index=True)
    ens=r4.generate_ensemble_candidates(cands,fcache)
    if len(ens): cands=pd.concat([cands,ens],ignore_index=True)
    if len(cands):
        # same native applicability contract as R4
        import trailaris_r4_asset_adapter as r4a
        m=cands.apply(lambda z:(z.strategy_family=='CAMPAIGN_MANAGEMENT') or r4a.family_applicable(str(z.asset),str(z.strategy_family)),axis=1)
        cands=cands[m].reset_index(drop=True)
    cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    opps=pd.concat(allo,ignore_index=True)
    return data,v,cands,opps,fcache


def cfg_apply(df,cfg):
    if df.empty:return df
    t=pd.to_datetime(df.decision_time,utc=True);m=(df.quality>=cfg['min_quality'])&(df.cost_r<=cfg['max_cost_r'])
    dr=int(cfg['direction'])
    if dr:m &= df.direction.eq(dr)
    h0,h1=int(cfg['hour_start']),int(cfg['hour_end']);m &= t.dt.hour.ge(h0)&t.dt.hour.lt(h1)
    return df[m]


def search_configs(train):
    rows=[]
    for (a,s),g in train.groupby(['asset','strategy'],sort=False):
        qa=g.quality.to_numpy(float);ca=g.cost_r.to_numpy(float);da=g.direction.to_numpy(int);ra=g.net_r.to_numpy(float);ha=g.decision_time.dt.hour.to_numpy(int)
        best=[]
        for q in QGRID:
            mq=qa>=q
            if mq.sum()<6:continue
            for mc in CGRID:
                mb=mq&(ca<=mc)
                if mb.sum()<6:continue
                for dr in (0,1,-1):
                    md=mb if dr==0 else mb&(da==dr)
                    if md.sum()<6:continue
                    for h0,h1 in FINE_HOURS:
                        m=md&(ha>=h0)&(ha<h1);n=int(m.sum())
                        if n<6:continue
                        rv=ra[m];mean=float(rv.mean());med=float(np.median(rv));sd=float(rv.std());win=float((rv>0).mean());cost=float(ca[m].mean())
                        if mean<=.02 or med<=-.30 or win<.34:continue
                        lower=mean-.40*sd/max(np.sqrt(n),1);score=lower+.12*win-.05*cost+.02*np.log1p(n)
                        best.append(dict(asset=a,strategy=s,min_quality=float(q),max_cost_r=float(mc),direction=int(dr),hour_start=h0,hour_end=h1,train_n=n,train_mean_r=mean,train_median_r=med,train_win_rate=win,train_sd_r=sd,lower_bound_r=lower,robust_score=score))
        if best:
            # Keep up to two materially distinct configurations, not a single brittle winner.
            best=sorted(best,key=lambda z:z['robust_score'],reverse=True)
            keep=[]
            for z in best:
                if not any((z['direction']==q['direction'] and z['hour_start']==q['hour_start'] and z['hour_end']==q['hour_end']) for q in keep):keep.append(z)
                if len(keep)==2:break
            rows.extend(keep)
    return pd.DataFrame(rows)


def trailing_stats(hist,asof,lookback_days=35):
    z=hist[(hist.exit_time<asof)&(hist.exit_time>=asof-pd.Timedelta(days=lookback_days))]
    if z.empty:return pd.DataFrame()
    return z.groupby(['asset','strategy']).agg(n=('net_r','size'),mean_r=('net_r','mean'),median_r=('net_r','median'),win=('net_r',lambda s:(s>0).mean()),sd=('net_r','std'),mean_giveback=('giveback_r','mean')).reset_index()


def promote_for_week(cands,week_start):
    # Proposals use old history; latest 14 days are validation only.
    val0=week_start-pd.Timedelta(days=14);old=cands[(cands.exit_time<val0)].copy();recent=cands[(cands.exit_time>=val0)&(cands.exit_time<week_start)].copy()
    target=cands[(cands.decision_time>=week_start)&(cands.decision_time<week_start+pd.Timedelta(days=7))].copy()
    if target.empty:return target,pd.DataFrame(),pd.DataFrame()
    prop=search_configs(old) if len(old) else pd.DataFrame();confirmed=[]
    if len(prop):
        for _,cfg in prop.iterrows():
            z=cfg_apply(recent[(recent.asset==cfg.asset)&(recent.strategy==cfg.strategy)],cfg)
            n=len(z);mean=float(z.net_r.mean()) if n else np.nan;med=float(z.net_r.median()) if n else np.nan;win=float((z.net_r>0).mean()) if n else 0.
            if n>=2 and mean>0 and med>-.25 and win>=.40:
                q=cfg.to_dict();q.update(validation_n=n,validation_mean_r=mean,validation_median_r=med,validation_win_rate=win,tier='A');confirmed.append(q)
    conf=pd.DataFrame(confirmed)
    parts=[]
    if len(conf):
        for _,cfg in conf.iterrows():
            z=cfg_apply(target[(target.asset==cfg.asset)&(target.strategy==cfg.strategy)],cfg).copy()
            if len(z):z['promotion_tier']='A';z['risk_fraction']=.005;parts.append(z)
    promoted=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=target.columns.tolist()+['promotion_tier','risk_fraction'])
    # Tier B: causal trailing-expectancy fallback. Only for asset-strategies without A candidates this week.
    stats=trailing_stats(cands,week_start)
    statmap={(r.asset,r.strategy):r for _,r in stats.iterrows()} if len(stats) else {}
    akeys=set(zip(promoted.asset,promoted.strategy)) if len(promoted) else set()
    b=[]
    for _,r in target.iterrows():
        k=(r.asset,r.strategy)
        if k in akeys:continue
        st=statmap.get(k)
        if st is None or int(st.n)<8:continue
        sd=0 if pd.isna(st.sd) else float(st.sd);lower=float(st.mean_r)-.45*sd/max(np.sqrt(float(st.n)),1)
        # High-quality positive trailing evidence; reduced risk, never a rescue by leverage.
        if float(st.mean_r)>.10 and float(st.median_r)>-.15 and float(st.win)>=.45 and lower>-.05 and float(r.quality)>=.68 and float(r.cost_r)<=.35:
            x=r.copy();x['promotion_tier']='B';x['risk_fraction']=.0025;b.append(x)
    if b:promoted=pd.concat([promoted,pd.DataFrame(b)],ignore_index=True)
    if promoted.empty:return promoted,conf,stats
    # Causal expected-utility rank from trailing exited outcomes only.
    er=[];wr=[];stab=[];gb=[]
    for _,r in promoted.iterrows():
        st=statmap.get((r.asset,r.strategy));mean=0.;win=.5;sd=1.;give=0.
        if st is not None:mean=float(st.mean_r);win=float(st.win);sd=0 if pd.isna(st.sd) else float(st.sd);give=float(st.mean_giveback)
        er.append(mean);wr.append(win);stab.append(mean/max(sd,.25));gb.append(give)
    promoted['expected_r']=er;promoted['trailing_win_rate']=wr;promoted['stability']=stab;promoted['trailing_giveback']=gb
    promoted['rank_score']=(.42*promoted.quality+.28*np.tanh(promoted.expected_r)+.16*promoted.trailing_win_rate+.10*np.tanh(promoted.stability)-.22*promoted.cost_r-.04*np.minimum(promoted.trailing_giveback,3))
    # de-duplicate exact campaign ids, favor A then rank
    promoted['tier_ord']=promoted.promotion_tier.map({'A':0,'B':1}).fillna(9)
    promoted=promoted.sort_values(['campaign_id','tier_ord','rank_score'],ascending=[True,True,False]).drop_duplicates('campaign_id').drop(columns='tier_ord')
    return promoted,conf,stats


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
        if fav>=2.:fl=max(fl,fav-.7) # R5 slightly tighter protected runner after 2R
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
            rf=float(r.get('risk_fraction',.005))*rs;lot,risk,margin=r4.size_trade(r.asset,eq,rf,float(r.stop_distance),specs);reason='SELECTED';proj=factors+r4.factor_vec(r.asset,int(r.direction));ck=(wk,r.asset,r.strategy);ak=(wk,r.asset)
            if bool(r.is_addon):
                if not r.parent_id or r.parent_id not in openpos:reason='ADDON_PARENT_NOT_OPEN'
                else:
                    ps=mark(openpos[r.parent_id],t)
                    if not ps or ps['floor']<0:reason='ADDON_PARENT_NOT_PROTECTED'
            # R5 independent same-asset campaign rule: second foundation only after protection,
            # same direction, different family, and per-asset open risk <=0.75% equity.
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
            u=float(r.rank_score)-.20*(margin/max(eq,1e-12))
            recycled=''
            if reason in {'OPEN_RISK_BUDGET','MARGIN_BUDGET','FACTOR_DUPLICATION'} and u>=.12:
                pid=recycle(r,t,u)
                if pid:
                    recycled=pid;eq=mtm(t);lot,risk,margin=r4.size_trade(r.asset,eq,rf,float(r.stop_distance),specs);proj=factors+r4.factor_vec(r.asset,int(r.direction))
                    if lot>0 and (open_risk+risk)/max(eq,1e-12)<=risk_budget+1e-12 and (open_margin+margin)/max(eq,1e-12)<=.60 and np.abs(proj).max()<=1.25:reason='SELECTED'
                    else:reason='RECYCLE_DID_NOT_FREE_ENOUGH_CAPACITY'
            dec.append(dict(decision_time=t,campaign_id=r.campaign_id,asset=r.asset,strategy=r.strategy,strategy_family=r.strategy_family,promotion_tier=r.promotion_tier,selected=reason=='SELECTED',reason=reason,quality=float(r.quality),expected_r=float(r.expected_r),rank_score=float(r.rank_score),utility=u,recycled_incumbent_id=recycled,equity=eq,week_growth=growth,risk_budget=risk_budget,lot=lot,risk_cash=risk,margin=margin,projected_open_risk_pct=100*(open_risk+risk)/max(eq,1e-12)))
            if reason=='SELECTED':
                p=r.to_dict();p.update(lot=lot,risk_cash=risk,margin=margin,equity_at_entry=eq,balance_at_entry=balance,entry_utility=u);openpos[r.campaign_id]=p;factors=proj;open_risk+=risk;open_margin+=margin;seq+=1;heapq.heappush(heap,(pd.Timestamp(r.exit_time),seq,r.campaign_id))
    close_until(pd.Timestamp.max.tz_localize('UTC'))
    ev=pd.DataFrame(events);de=pd.DataFrame(dec)
    if len(ev):ev=ev.sort_values('timestamp').reset_index(drop=True)
    return ev,de


def run(rawdir:Path,specfile:Path,out:Path,start=100.):
    out.mkdir(parents=True,exist_ok=True);data,v,cands,opps,fcache=build_universe(rawdir);specs=r4.load_specs(specfile,'research-proxy');v.to_csv(out/'01_data_validation.csv',index=False)
    t0=cands.decision_time.min().floor('D');t1=cands.decision_time.max();# rolling weeks after at least four weeks history
    first=(t0+pd.Timedelta(days=28)).to_period('W-SUN').start_time.tz_localize('UTC')
    weeks=[];all_events=[];all_dec=[];all_prom=[];cfg_rows=[];equity=start
    wk=first
    while wk<=t1:
        promoted,conf,stats=promote_for_week(cands,wk)
        if len(promoted):
            ev,de=replay_r5(promoted,specs,equity,fcache)
            # carry actual end equity to next week; replay is week-local candidates only.
            if len(ev): equity=float(ev.equity.iloc[-1])
            all_events.append(ev.assign(eval_week=wk));all_dec.append(de.assign(eval_week=wk));all_prom.append(promoted.assign(eval_week=wk))
        else:ev=pd.DataFrame();de=pd.DataFrame()
        if len(conf):cfg_rows.append(conf.assign(eval_week=wk))
        w=r4.weekly(ev,start=(weeks[-1]['end_equity'] if weeks else start),feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),t1))
        if len(w):rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
        else:
            st=weeks[-1]['end_equity'] if weeks else start;rec=dict(week_start=wk,start_equity=st,end_equity=st,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False);equity=st
        rec['tierA_candidates']=int((promoted.promotion_tier=='A').sum()) if len(promoted) else 0;rec['tierB_candidates']=int((promoted.promotion_tier=='B').sum()) if len(promoted) else 0;rec['confirmed_configs']=len(conf);weeks.append(rec);wk+=pd.Timedelta(days=7)
    W=pd.DataFrame(weeks);EV=pd.concat(all_events,ignore_index=True) if all_events else pd.DataFrame();DE=pd.concat(all_dec,ignore_index=True) if all_dec else pd.DataFrame();PR=pd.concat(all_prom,ignore_index=True) if all_prom else pd.DataFrame();CF=pd.concat(cfg_rows,ignore_index=True) if cfg_rows else pd.DataFrame()
    W.to_csv(out/'R5_walkforward_weekly.csv',index=False);EV.to_csv(out/'R5_events.csv',index=False);DE.to_csv(out/'R5_decisions.csv',index=False);PR.to_csv(out/'R5_promoted_candidates.csv',index=False);CF.to_csv(out/'R5_confirmed_configs.csv',index=False)
    # clean fresh holdout = all data after 2026-08-12; the R5 rules were frozen before these paths were acquired.
    fresh0=pd.Timestamp('2026-08-13T00:00:00Z')
    # Capture over all rolling-OOS events. For the genuinely new Aug13-14 paths, freeze
    # the Aug10 promotion state (which used only data before Aug10), then replay ONLY
    # decisions on/after Aug13 from a clean $100 research account. This prevents open
    # positions from the already-observed Aug10-12 path contaminating the fresh test.
    cap=r4.capture_attribution(EV,r4.opportunity_period(data,first,t1),cands[(cands.decision_time>=first)&(cands.decision_time<=t1)],PR,DE) if len(EV) else pd.DataFrame();cap.to_csv(out/'R5_capture_attribution.csv',index=False)
    aug10=pd.Timestamp('2026-08-10T00:00:00Z');week_prom,_,_=promote_for_week(cands,aug10);fresh_pr=week_prom[(week_prom.decision_time>=fresh0)&(week_prom.decision_time<=t1)].copy()
    fresh_ev,fresh_de=replay_r5(fresh_pr,specs,100.0,fcache) if len(fresh_pr) else (pd.DataFrame(),pd.DataFrame())
    fresh_raw=cands[(cands.decision_time>=fresh0)&(cands.decision_time<=t1)];fresh_opp=r4.opportunity_period(data,fresh0,t1)
    fresh_cap=r4.capture_attribution(fresh_ev,fresh_opp,fresh_raw,fresh_pr,fresh_de) if len(fresh_opp) else pd.DataFrame();fresh_cap.to_csv(out/'R5_fresh_Aug13_14_capture.csv',index=False)
    fresh_w=r4.weekly(fresh_ev,start=100.0,feature_cache=fcache,start_ts=fresh0,end_ts=t1) if len(fresh_ev) else pd.DataFrame();fresh_w.to_csv(out/'R5_fresh_Aug13_14_performance.csv',index=False)
    zero=[]
    for a in ROUTES:
        n=int((CF.asset==a).sum()) if len(CF) else 0;sel=int(((DE.asset==a)&DE.selected).sum()) if len(DE) else 0;zero.append(dict(asset=a,rolling_confirmed_configs=n,selected_campaigns=sel,state='ACTIVE' if (n or sel) else 'NEXT_VARIANT_REQUIRED'))
    pd.DataFrame(zero).to_csv(out/'R5_34_asset_state.csv',index=False)
    status=dict(state='R5_CAUSAL_ROLLING_ORIGIN_EXECUTED',evidence_class='HISTORICAL_MARKET_PROXY_NOT_BROKER_CERTIFIED',routes=34,start_equity=start,end_equity=float(W.end_equity.iloc[-1]) if len(W) else start,weeks=len(W),mean_week_pct=float(W.weekly_return_pct.mean()) if len(W) else 0,median_week_pct=float(W.weekly_return_pct.median()) if len(W) else 0,weeks_ge5=int(W.ge5.sum()) if len(W) else 0,weeks_ge10=int(W.ge10.sum()) if len(W) else 0,max_weekly_dd_pct=float(W.max_drawdown_pct.min()) if len(W) else 0,max_selected_open_risk_pct=float(DE.loc[DE.selected,'projected_open_risk_pct'].max()) if len(DE) and DE.selected.any() else 0,tierA_promoted=int((PR.promotion_tier=='A').sum()) if len(PR) else 0,tierB_promoted=int((PR.promotion_tier=='B').sum()) if len(PR) else 0,assets_with_selected=int(DE.loc[DE.selected,'asset'].nunique()) if len(DE) else 0,fresh_holdout_start=str(fresh0),fresh_holdout_end=str(t1),fresh_events=len(fresh_ev),fresh_start_equity=100.0,fresh_end_equity=float(fresh_w.end_equity.iloc[-1]) if len(fresh_w) else 100.0,fresh_return_pct=float(fresh_w.weekly_return_pct.iloc[-1]) if len(fresh_w) else 0.0,fresh_max_dd_pct=float(fresh_w.max_drawdown_pct.iloc[-1]) if len(fresh_w) else 0.0)
    (out/'R5_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str));return status

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--start',type=float,default=100.);a=ap.parse_args();run(Path(a.rawdir),Path(a.specs),Path(a.outdir),a.start)
