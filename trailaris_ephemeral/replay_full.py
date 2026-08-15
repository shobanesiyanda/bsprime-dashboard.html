#!/usr/bin/env python3
from __future__ import annotations
import json, math, heapq
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

ROOT=Path('trailaris_ephemeral'); RAW=ROOT/'raw'; OUT=ROOT/'results'; OUT.mkdir(parents=True,exist_ok=True)
ROUTES=['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
FX=set(ROUTES[:13]); MET={'XAUUSD','XAGUSD'}; CRYPTO={'BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD'}; INDEX={'US100','US500','US30','GER40','UK100','JP225'}; ENERGY={'WTI','BRENT','NATGAS'}; SYN={'V75','V50','V100','V25','V10'}

def cls(a):
    if a in FX:return 'FX'
    if a in MET:return 'METAL'
    if a in CRYPTO:return 'CRYPTO'
    if a in INDEX:return 'INDEX'
    if a in ENERGY:return 'ENERGY'
    return 'SYNTHETIC'
BASE_COST={'FX':.06,'METAL':.08,'CRYPTO':.10,'INDEX':.08,'ENERGY':.10,'SYNTHETIC':.08}
STRESS_MULT=2.0

def load5(asset):
    p=RAW/f'{asset}.csv'
    if not p.exists(): raise FileNotFoundError(p)
    d=pd.read_csv(p)
    for c in ['timestamp','open','high','low','close','volume']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['timestamp','open','high','low','close'])
    ts=d['timestamp'].astype('int64')
    # all materializers normalize to milliseconds
    d['timestamp']=pd.to_datetime(ts,unit='ms',utc=True)
    d=d.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    return d[['open','high','low','close','volume']]

def m15(d):
    x=d.resample('15min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open','high','low','close'])
    prev=x.close.shift(1)
    tr=pd.concat([(x.high-x.low).abs(),(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x['atr']=tr.rolling(14,min_periods=14).mean()
    x['ema20']=x.close.ewm(span=20,adjust=False).mean();x['ema50']=x.close.ewm(span=50,adjust=False).mean()
    x['sma50']=x.close.rolling(50).mean();x['std50']=x.close.rolling(50).std();x['z']=(x.close-x.sma50)/x.std50.replace(0,np.nan)
    path=x.close.diff().abs().rolling(20).sum();x['eff20']=(x.close-x.close.shift(20)).abs()/path.replace(0,np.nan)
    x['atr_med80']=x.atr.rolling(80).median()
    return x.dropna(subset=['atr','ema20','ema50'])

PARAMS=[]
for lb in (12,24,40): PARAMS.append(('trend',{'lb':lb,'stop':1.5,'target':3.0 if lb<40 else 3.5,'hold':40}))
for stop,target in ((1.25,2.5),(1.6,3.0)): PARAMS.append(('pullback',{'stop':stop,'target':target,'hold':48}))
for lb,comp in ((16,.75),(24,.80),(32,.85)): PARAMS.append(('compression',{'lb':lb,'comp':comp,'stop':1.5,'target':3.25,'hold':48}))
for lb in (16,32): PARAMS.append(('failed_break',{'lb':lb,'stop':1.25,'target':2.25,'hold':32}))
for z in (1.8,2.2,2.6): PARAMS.append(('mean_reversion',{'z':z,'stop':1.35,'target':1.9,'hold':32}))

def signals(x,fam,p):
    c=x.close; h=x.high; l=x.low
    if fam=='trend':
        hh=h.rolling(p['lb']).max().shift(1);ll=l.rolling(p['lb']).min().shift(1)
        lo=(c>hh)&(x.ema20>x.ema50)&(x.eff20>.22); sh=(c<ll)&(x.ema20<x.ema50)&(x.eff20>.22)
        q=np.clip((x.eff20-.20)*2.5,0,1)
    elif fam=='pullback':
        lo=(x.ema20>x.ema50)&(c.shift(1)<=x.ema20.shift(1))&(c>x.ema20)&(x.eff20>.15)
        sh=(x.ema20<x.ema50)&(c.shift(1)>=x.ema20.shift(1))&(c<x.ema20)&(x.eff20>.15)
        q=np.clip(x.eff20*2,0,1)
    elif fam=='compression':
        hh=h.rolling(p['lb']).max().shift(1);ll=l.rolling(p['lb']).min().shift(1); comp=x.atr<x.atr_med80*p['comp']
        # compression must have existed on immediately preceding bar; breakout is current
        lo=comp.shift(1).fillna(False)&(c>hh)&(x.ema20>=x.ema50); sh=comp.shift(1).fillna(False)&(c<ll)&(x.ema20<=x.ema50)
        q=np.clip((x.atr_med80/x.atr.replace(0,np.nan)-1)*.8,0,1).fillna(0)
    elif fam=='failed_break':
        hh=h.rolling(p['lb']).max().shift(1);ll=l.rolling(p['lb']).min().shift(1)
        lo=(l<ll)&(c>ll)&(x.eff20<.55);sh=(h>hh)&(c<hh)&(x.eff20<.55);q=np.clip(1-x.eff20,0,1)
    else:
        lo=(x.z<-p['z'])&(x.eff20<.35);sh=(x.z>p['z'])&(x.eff20<.35);q=np.clip((x.z.abs()-p['z'])/2,0,1)
    direction=pd.Series(0,index=x.index,dtype='int8');direction[lo.fillna(False)]=1;direction[sh.fillna(False)]=-1
    return direction,q.fillna(0)

def simulate_candidate(x,asset,fam,p):
    direction,qual=signals(x,fam,p); idx=np.flatnonzero(direction.to_numpy()!=0)
    o=x.open.to_numpy(float);h=x.high.to_numpy(float);l=x.low.to_numpy(float);c=x.close.to_numpy(float);atr=x.atr.to_numpy(float);ts=x.index
    out=[]; blocked_until=-1
    basecost=BASE_COST[cls(asset)];stress=basecost*STRESS_MULT
    for si in idx:
        if si<=blocked_until or si+1>=len(x):continue
        di=int(direction.iat[si]);ei=si+1;entry=o[ei];sd=p['stop']*atr[si]
        if not np.isfinite(sd) or sd<=0 or entry<=0:continue
        initial_stop=entry-di*sd;target=entry+di*p['target']*sd;curstop=initial_stop
        maxj=min(len(x)-1,ei+p['hold']); exitj=maxj;exitpx=c[maxj];reason='TIME';mfe=0.;mae=0.;addon_i=None
        for j in range(ei,maxj+1):
            favorable=((h[j]-entry)/sd if di>0 else (entry-l[j])/sd); adverse=((entry-l[j])/sd if di>0 else (h[j]-entry)/sd)
            mfe=max(mfe,favorable);mae=max(mae,adverse)
            # adverse-first same-bar ordering using stop valid from previous bar
            stophit=(l[j]<=curstop if di>0 else h[j]>=curstop)
            targethit=(h[j]>=target if di>0 else l[j]<=target)
            if stophit:
                exitj=j;exitpx=curstop;reason='STOP';break
            if targethit:
                exitj=j;exitpx=target;reason='TARGET';break
            progress=di*(c[j]-entry)/sd
            if addon_i is None and progress>=1.0 and j+1<=maxj: addon_i=j+1
            # update protection only after this bar closes
            if favorable>=2.0: curstop=(max(curstop,entry+1.0*sd) if di>0 else min(curstop,entry-1.0*sd))
            elif favorable>=1.0: curstop=(max(curstop,entry+0.05*sd) if di>0 else min(curstop,entry-0.05*sd))
        raw=di*(exitpx-entry)/sd
        addon_raw=0.;addon_trigger=False
        if addon_i is not None and addon_i<=exitj:
            ae=o[addon_i]; afloor=entry+di*.05*sd; ard=di*(ae-afloor)
            if ard>.15*sd:
                addon_trigger=True; astop=afloor;atarget=target;aexit=exitpx
                for j in range(addon_i,exitj+1):
                    if (l[j]<=astop if di>0 else h[j]>=astop):aexit=astop;break
                    if (h[j]>=atarget if di>0 else l[j]<=atarget):aexit=atarget;break
                addon_raw=di*(aexit-ae)/ard
        out.append({'asset':asset,'family':fam,'param':json.dumps(p,sort_keys=True),'signal_time':ts[si],'entry_time':ts[ei],'exit_time':ts[exitj],
                    'raw_r':raw,'base_r':raw-basecost,'stress_r':raw-stress,'addon_raw_r':addon_raw if addon_trigger else 0.,
                    'addon_base_r':addon_raw-basecost if addon_trigger else 0.,'addon_stress_r':addon_raw-stress if addon_trigger else 0.,
                    'addon_trigger':addon_trigger,'mfe_r':mfe,'mae_r':mae,'quality':float(qual.iat[si]),'stop_distance':sd,'exit_reason':reason})
        blocked_until=exitj
    return pd.DataFrame(out)

def stats(r):
    r=pd.Series(r).dropna();n=len(r)
    if not n:return {'n':0,'mean':np.nan,'pf':0,'win':0,'sum':0}
    gp=r[r>0].sum();gl=-r[r<0].sum();pf=float(gp/gl) if gl>0 else (99. if gp>0 else 0.)
    return {'n':n,'mean':float(r.mean()),'pf':pf,'win':float((r>0).mean()),'sum':float(r.sum())}

def choose_route(asset,x,t1,t2):
    cand=[]; frames=[]
    for fam,p in PARAMS:
        tr=simulate_candidate(x,asset,fam,p)
        if tr.empty:continue
        tr['candidate']=fam+'|'+json.dumps(p,sort_keys=True)
        frames.append(tr)
        train=tr[tr.entry_time<t1];val=tr[(tr.entry_time>=t1)&(tr.entry_time<t2)]
        for addon in (False,True):
            col_train=train.stress_r + (.5*train.addon_stress_r if addon else 0)
            col_val=val.stress_r + (.5*val.addon_stress_r if addon else 0)
            st,sv=stats(col_train),stats(col_val)
            ok=st['n']>=20 and sv['n']>=8 and st['mean']>0 and sv['mean']>0 and sv['pf']>1.02
            score=(sv['mean']*math.sqrt(min(sv['n'],50)) + .25*st['mean'] - .02*max(0,1.15-sv['pf'])) if ok else -1e9
            cand.append({'asset':asset,'candidate':tr.candidate.iloc[0],'family':fam,'param':tr.param.iloc[0],'addon':addon,'train_n':st['n'],'train_mean_r':st['mean'],'train_pf':st['pf'],'val_n':sv['n'],'val_mean_r':sv['mean'],'val_pf':sv['pf'],'score':score,'qualified':ok})
    if not cand:return None,pd.DataFrame(cand),pd.DataFrame()
    ct=pd.DataFrame(cand).sort_values('score',ascending=False)
    good=ct[ct.qualified]
    if good.empty:return None,ct,pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    best=good.iloc[0].to_dict(); alltr=pd.concat(frames,ignore_index=True);chosen=alltr[alltr.candidate==best['candidate']].copy();chosen['addon_enabled']=bool(best['addon']);chosen['validated_expectancy']=float(best['val_mean_r']);chosen['selection_score']=float(best['score'])
    return best,ct,chosen

# Load and enforce 34/34 actual history
loaded={};cov=[]
for a in ROUTES:
    d=load5(a);loaded[a]=d;cov.append({'asset':a,'rows_m5':len(d),'first':d.index.min(),'last':d.index.max(),'source_class':cls(a)})
if set(loaded)!=set(ROUTES):raise RuntimeError('34/34 load gate failed')
covdf=pd.DataFrame(cov);common_start=max(covdf['first']);common_end=min(covdf['last'])
if common_end-common_start<pd.Timedelta(days=365): raise RuntimeError(f'common window too short {common_start} {common_end}')
span=common_end-common_start;t1=common_start+span*.60;t2=common_start+span*.80
print('COMMON',common_start,common_end,'TRAIN_END',t1,'VAL_END',t2)

route_rows=[];candidate_rows=[];holdouts=[]
for a in ROUTES:
    x=m15(loaded[a].loc[(loaded[a].index>=common_start)&(loaded[a].index<=common_end)])
    best,ct,tr=choose_route(a,x,t1,t2);candidate_rows.append(ct)
    rr={'asset':a,'class':cls(a),'m15_rows':len(x),'selected':best is not None}
    if best is None:
        rr.update({'strategy':'NONE','addon':False,'train_n':0,'val_n':0,'holdout_n':0,'holdout_stress_mean_r':np.nan,'holdout_base_mean_r':np.nan,'reason':'NO_POSITIVE_TRAIN_VALIDATION_EDGE'})
    else:
        test=tr[tr.entry_time>=t2].copy(); addon=bool(best['addon']);test['portfolio_stress_r']=test.stress_r+(.5*test.addon_stress_r if addon else 0);test['portfolio_base_r']=test.base_r+(.5*test.addon_base_r if addon else 0)
        test['strategy']=best['candidate'];test['addon_enabled']=addon;holdouts.append(test)
        ss=stats(test.portfolio_stress_r);sb=stats(test.portfolio_base_r)
        rr.update({'strategy':best['candidate'],'addon':addon,'train_n':best['train_n'],'train_mean_r':best['train_mean_r'],'val_n':best['val_n'],'val_mean_r':best['val_mean_r'],'val_pf':best['val_pf'],'holdout_n':ss['n'],'holdout_stress_mean_r':ss['mean'],'holdout_stress_pf':ss['pf'],'holdout_base_mean_r':sb['mean'],'holdout_base_pf':sb['pf'],'reason':'SELECTED_FROM_TRAIN_VALIDATION_ONLY'})
    route_rows.append(rr);print('ROUTE',a,rr['selected'],rr.get('strategy'),rr.get('holdout_n'))
route=pd.DataFrame(route_rows);route.to_csv(OUT/'route_results.csv',index=False);pd.concat(candidate_rows,ignore_index=True).to_csv(OUT/'candidate_selection.csv',index=False)
if len(route)!=34:raise RuntimeError('34-route closing research gate failed')
alltr=pd.concat(holdouts,ignore_index=True) if holdouts else pd.DataFrame()
if alltr.empty: raise RuntimeError('no holdout campaigns')
alltr['entry_time']=pd.to_datetime(alltr.entry_time,utc=True);alltr['exit_time']=pd.to_datetime(alltr.exit_time,utc=True);alltr=alltr.sort_values(['entry_time','selection_score'],ascending=[True,False])

@dataclass
class OpenPos:
    exit_ns:int; seq:int; asset:str; bucket:str; risk_cash:float; addon_cash:float; r:float; addon_r:float; reserve_frac:float; entry_time:pd.Timestamp; exit_time:pd.Timestamp; stop_distance:float; lot_units:float

def run_portfolio(start_balance=100.,risk_frac=.0025,cost_mode='stress'):
    bal=start_balance;peak=bal;openheap=[];seq=0;events=[];accepted=[];rejected=[];week_key=None;week_start_bal=bal;latched=False;floor=None
    factor_reserved={k:0. for k in ['FX','METAL','CRYPTO','INDEX','ENERGY','SYNTHETIC']};total_reserved=0.
    def close_until(t):
        nonlocal bal,peak,total_reserved
        while openheap and openheap[0][0]<=t.value:
            _,_,p=heapq.heappop(openheap); pnl=p.risk_cash*p.r + p.addon_cash*p.addon_r;bal+=pnl;peak=max(peak,bal);total_reserved=max(0.,total_reserved-p.reserve_frac);factor_reserved[p.bucket]=max(0.,factor_reserved[p.bucket]-p.reserve_frac);events.append({'time':p.exit_time,'balance':bal,'pnl':pnl,'asset':p.asset,'kind':'EXIT'})
    for _,r in alltr.iterrows():
        t=r.entry_time;wk=(t-pd.Timedelta(days=t.weekday())).normalize()
        close_until(t)
        if week_key is None or wk!=week_key:
            week_key=wk;week_start_bal=bal;latched=False;floor=None
        growth=bal/week_start_bal-1 if week_start_bal else 0
        if growth>=.05 and not latched:latched=True;floor=week_start_bal*1.04
        dd=bal/peak-1 if peak else 0
        ddscale=.25 if dd<=-.05 else (.5 if dd<=-.03 else 1.)
        gov=.25 if growth>=.10 else 1.
        if latched and bal<=floor: gov=0.
        addon_on=bool(r.addon_enabled) and abs(float(r.addon_raw_r))>0
        reserve=risk_frac*(1.5 if addon_on else 1.0)*ddscale*gov
        bucket=cls(r.asset)
        cushion_frac=((bal-floor)/bal if latched and floor is not None and bal>0 else 1.)
        allowed=max(0.,min(.01,cushion_frac))
        if gov<=0 or reserve<=0 or total_reserved+reserve>allowed+1e-12 or factor_reserved[bucket]+reserve>.0075+1e-12:
            rejected.append({'time':t,'asset':r.asset,'reason':'RISK_OR_PROTECTED_FLOOR','balance':bal,'total_reserved':total_reserved,'proposed':reserve});continue
        rc=bal*risk_frac*ddscale*gov; ac=.5*rc if addon_on else 0.; rval=float(r.portfolio_stress_r if cost_mode=='stress' else r.portfolio_base_r)
        # portfolio_r already includes 0.5 addon contribution, split it back to avoid double count
        fr=float(r.stress_r if cost_mode=='stress' else r.base_r); ar=float(r.addon_stress_r if cost_mode=='stress' else r.addon_base_r) if addon_on else 0.
        lot=rc/max(float(r.stop_distance),1e-12)
        p=OpenPos(r.exit_time.value,seq,r.asset,bucket,rc,ac,fr,ar,reserve,t,r.exit_time,float(r.stop_distance),lot);seq+=1;heapq.heappush(openheap,(p.exit_ns,p.seq,p));total_reserved+=reserve;factor_reserved[bucket]+=reserve
        accepted.append({'entry_time':t,'exit_time':r.exit_time,'asset':r.asset,'strategy':r.strategy,'risk_cash':rc,'addon_risk_cash':ac,'lot_units':lot,'balance_at_entry':bal,'foundation_r':fr,'addon_r':ar,'reserved_fraction':reserve,'dd_scale':ddscale,'weekly_growth_at_entry':growth})
    close_until(pd.Timestamp.max.tz_localize('UTC'))
    ev=pd.DataFrame(events).sort_values('time') if events else pd.DataFrame(columns=['time','balance'])
    ac=pd.DataFrame(accepted);rj=pd.DataFrame(rejected)
    # Weekly snapshots across the holdout period
    start=t2.normalize()-pd.Timedelta(days=t2.weekday());end=common_end.normalize();weeks=pd.date_range(start,end,freq='7D',tz='UTC')
    wr=[];prev=start_balance
    for ws in weeks:
        we=ws+pd.Timedelta(days=7)
        sub=ev[ev.time<we]
        endbal=float(sub.balance.iloc[-1]) if len(sub) else start_balance
        # if previous weeks had an event, carry end balance; sub includes all history so correct
        wret=endbal/prev-1 if prev else 0
        camps=int(((ac.entry_time>=ws)&(ac.entry_time<we)).sum()) if len(ac) else 0
        wr.append({'week_start':ws,'week_end':we-pd.Timedelta(seconds=1),'start_equity':prev,'end_equity':endbal,'weekly_return_pct':100*wret,'campaigns':camps,'ge_5pct':wret>=.05,'ge_10pct':wret>=.10})
        prev=endbal
    w=pd.DataFrame(wr);w['month']=w.week_start.dt.strftime('%Y-%m');w['week_in_month']=w.groupby('month').cumcount()+1
    m=[]
    for mo,g in w.groupby('month',sort=True):
        comp=np.prod(1+g.weekly_return_pct/100)-1;m.append({'month':mo,'weeks':len(g),'monthly_compounded_return_pct':100*comp,'average_week_pct':g.weekly_return_pct.mean(),'weeks_ge_5':int(g.ge_5pct.sum()),'weeks_ge_10':int(g.ge_10pct.sum()),'campaigns':int(g.campaigns.sum()),'start_equity':g.start_equity.iloc[0],'end_equity':g.end_equity.iloc[-1]})
    md=pd.DataFrame(m)
    # drawdown on realized-balance path
    if len(ev):
        b=np.r_[start_balance,ev.balance.to_numpy(float)];peaks=np.maximum.accumulate(b);maxdd=float(np.min(b/peaks-1))
    else:maxdd=0.
    metrics={'start_balance':start_balance,'end_balance':bal,'total_return_pct':100*(bal/start_balance-1),'risk_fraction_pct':100*risk_frac,'cost_mode':cost_mode,'campaigns':len(ac),'rejected':len(rj),'mean_week_pct':float(w.weekly_return_pct.mean()),'median_week_pct':float(w.weekly_return_pct.median()),'best_week_pct':float(w.weekly_return_pct.max()),'worst_week_pct':float(w.weekly_return_pct.min()),'weeks':len(w),'weeks_ge_5':int(w.ge_5pct.sum()),'pct_weeks_ge_5':100*float(w.ge_5pct.mean()),'weeks_ge_10':int(w.ge_10pct.sum()),'pct_weeks_ge_10':100*float(w.ge_10pct.mean()),'positive_weeks':int((w.weekly_return_pct>0).sum()),'pct_positive_weeks':100*float((w.weekly_return_pct>0).mean()),'max_realized_balance_dd_pct':100*maxdd,'avg_campaigns_week':len(ac)/max(1,len(w))}
    return metrics,w,md,ac,rj

scenarios=[];canonical=None
for rf in (.0020,.0025,.0033,.0050):
    for cm in ('base','stress'):
        met,w,mo,ac,rj=run_portfolio(100.,rf,cm);scenarios.append(met)
        if abs(rf-.0025)<1e-9 and cm=='stress':canonical=(met,w,mo,ac,rj)
pd.DataFrame(scenarios).to_csv(OUT/'portfolio_scenarios.csv',index=False)
met,w,mo,ac,rj=canonical
w.to_csv(OUT/'weekly_results.csv',index=False);mo.to_csv(OUT/'monthly_results.csv',index=False);ac.to_csv(OUT/'accepted_campaigns.csv',index=False);rj.to_csv(OUT/'rejected_campaigns.csv',index=False)
# $200 proportional replay for explicit small-tier requirement
met200,_,_,_,_=run_portfolio(200.,.0025,'stress')
coverage={'routes_required':34,'routes_loaded':len(loaded),'routes_tested':len(route),'routes_selected':int(route.selected.sum()),'common_start':str(common_start),'common_end':str(common_end),'train_end':str(t1),'validation_end':str(t2),'holdout_start':str(t2),'holdout_weeks':met['weeks'],'evidence_class':'PUBLIC_RESEARCH_PROXY_OR_OFFICIAL_DISCOVERY_DATA; NOT TARGET_BROKER_EXECUTION_PROOF'}
summary={'scope_gate_open':'34/34 PASS','scope_gate_close':'34/34 PASS','coverage':coverage,'canonical_100_stress':met,'canonical_200_stress':met200,'selected_route_holdout':{'mean_stress_r':float(route.holdout_stress_mean_r.dropna().mean()),'routes_positive_holdout':int((route.holdout_stress_mean_r.fillna(-999)>0).sum()),'total_holdout_campaign_candidates':int(route.holdout_n.sum())},'compounding':{'shared_balance':True,'current_balance_risk_sizing':True,'normalized_lot_units_recomputed_each_entry':True,'dynamic_capital_recycling':True,'drawdown_decompounding':True,'winner_addons_validation_gated':True,'weekly_5pct_floor_governor':True,'post_10pct_harvest_scale':0.25,'aggregate_open_risk_cap_pct':1.0},'limits':['Dukascopy conventional histories are research proxies, not target-broker bid/ask execution proof.','Binance histories are official spot OHLCV discovery data, not broker CFD execution.','Deriv synthetic histories are official historical candles.','Normalized lot units compound with equity but broker contract/minimum-lot rounding is not applied in this research replay.','Strategy/parameter selection uses only first 80% train+validation; final 20% is untouched holdout for this run.']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str));covdf.to_csv(OUT/'data_coverage.csv',index=False)
print(json.dumps(summary,indent=2,default=str))
