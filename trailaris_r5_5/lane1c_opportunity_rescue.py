#!/usr/bin/env python3
from __future__ import annotations
import argparse,heapq,json
from pathlib import Path
import numpy as np,pandas as pd

R54_END=337.4231242104393
EPS=1e-12

CONFIGS=[
 {'id':'RW265_M50_C20_G30_U05','gates':[('reliability_win','le',0.2655132009803008)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.05,'week_growth_max':None},
 {'id':'RW265_M50_C20_G30_U10','gates':[('reliability_win','le',0.2655132009803008)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.10,'week_growth_max':None},
 {'id':'RW265_M50_C20_G30_U05_W05','gates':[('reliability_win','le',0.2655132009803008)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.05,'week_growth_max':.05},
 {'id':'RW265_M50_C20_G30_U05_W04','gates':[('reliability_win','le',0.2655132009803008)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.05,'week_growth_max':.04},
 {'id':'TW272_M50_C20_G30_U05','gates':[('trailing_win_rate','le',0.2722646310432570)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.05,'week_growth_max':None},
 {'id':'TW272_M50_C20_G30_U05_W05','gates':[('trailing_win_rate','le',0.2722646310432570)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.05,'week_growth_max':.05},
 {'id':'SW279_M50_C20_G30_U05','gates':[('reliability_strategy_win','le',0.2792262732483457)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.05,'week_growth_max':None},
 {'id':'SW279_M50_C20_G30_U10_W05','gates':[('reliability_strategy_win','le',0.2792262732483457)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.10,'week_growth_max':.05},
 {'id':'SW306_M50_C10_G40_U10_W05','gates':[('reliability_strategy_win','le',0.30622738525995047)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.10,'min_giveback':.40,'utility_delta':.10,'week_growth_max':.05},
 {'id':'RW265_TW272_M50_C20_G30_U05_W05','gates':[('reliability_win','le',0.2655132009803008),('trailing_win_rate','le',0.2722646310432570)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.05,'week_growth_max':.05},
 {'id':'RW265_SW279_M50_C20_G30_U05_W05','gates':[('reliability_win','le',0.2655132009803008),('reliability_strategy_win','le',0.2792262732483457)],'min_mfe':.50,'min_current_r':.02,'max_current_r':.20,'min_giveback':.30,'utility_delta':.05,'week_growth_max':.05},
 {'id':'SW279_M75_C20_G55_U10_W05','gates':[('reliability_strategy_win','le',0.2792262732483457)],'min_mfe':.75,'min_current_r':.02,'max_current_r':.20,'min_giveback':.55,'utility_delta':.10,'week_growth_max':.05},
]

def num(x,d=np.nan):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def gate_pass(p,cfg):
    for metric,direction,th in cfg['gates']:
        v=num(p.get(metric),np.nan)
        if not np.isfinite(v):return False
        if direction=='le' and v>float(th)+EPS:return False
        if direction=='ge' and v<float(th)-EPS:return False
    return True

def weak_rescue_ok(p,s,u,growth,cfg):
    if s is None or s['floor']>=0:return False
    cr=float(s['current_r']);mfe=float(s['mfe'])
    if not (float(cfg['min_current_r'])-EPS<=cr<=float(cfg['max_current_r'])+EPS):return False
    if mfe+EPS<float(cfg['min_mfe']):return False
    if mfe-cr+EPS<float(cfg['min_giveback']):return False
    if not gate_pass(p,cfg):return False
    cap=cfg.get('week_growth_max')
    if cap is not None and float(growth)>=float(cap)-EPS:return False
    u0=float(p.get('entry_utility',p.get('rank_score',p.get('quality',0))))
    return float(u)>u0+float(cfg['utility_delta'])

def replay_r5(cands,specs,start=100.,feature_cache=None,cfg=None):
    import trailaris_r4_full_universe_loop as r4
    if cands.empty:return pd.DataFrame(),pd.DataFrame()
    c=cands.copy();c['decision_time']=pd.to_datetime(c.decision_time,utc=True);c['exit_time']=pd.to_datetime(c.exit_time,utc=True)
    c=c.sort_values(['decision_time','rank_score'],ascending=[True,False]).reset_index(drop=True)
    balance=float(start);peak=balance;week=None;week_start=balance;objective=False;floor=False
    openpos={};heap=[];events=[];dec=[];seq=0;factors=np.zeros(len(r4.FACTORS));open_risk=0.;open_margin=0.;config_week_r={};asset_week_r={}
    rescue_count=0
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
    def recycle(new,t,u,growth):
        nonlocal balance,peak,factors,open_risk,open_margin,rescue_count
        protected=[];weak=[]
        for pid,p in openpos.items():
            if bool(p.get('is_addon',False)):continue
            s=mark(p,t)
            if not s:continue
            u0=float(p.get('entry_utility',p.get('rank_score',p.get('quality',0))))
            nf=factors-r4.factor_vec(p['asset'],int(p['direction']))+r4.factor_vec(new.asset,int(new.direction))
            if np.abs(nf).max()>1.25:continue
            if s['floor']>=0 and u>u0+.05:protected.append((u0,pid,p,s,'DYNAMIC_CAPITAL_RECYCLE_R5'))
            elif cfg is not None and weak_rescue_ok(p,s,u,growth,cfg):weak.append((u0,pid,p,s,'R5_5_OPPORTUNITY_RESCUE_RECYCLE'))
        choices=protected if protected else weak
        if not choices:return None
        _,pid,p,s,why=min(choices,key=lambda z:z[0]);openpos.pop(pid,None);rr=float(s['current_r']);before=mtm(t);pnl=float(p['risk_cash'])*rr;balance+=pnl;factors-=r4.factor_vec(p['asset'],int(p['direction']));open_risk=max(0,open_risk-float(p['risk_cash']));open_margin=max(0,open_margin-float(p['margin']));eq=mtm(t);peak=max(peak,eq)
        if why=='R5_5_OPPORTUNITY_RESCUE_RECYCLE':rescue_count+=1
        events.append({**p,'timestamp':pd.Timestamp(t),'exit_time':pd.Timestamp(t),'exit_price':s['mark'],'net_r':rr,'net_pnl':pnl,'equity_before':before,'equity':eq,'balance_after':balance,'drawdown':eq/peak-1,'exit_reason':why,'management_rule':cfg['id'] if why=='R5_5_OPPORTUNITY_RESCUE_RECYCLE' else p.get('management_rule')})
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
            same=[p for p in openpos.values() if p['asset']==r.asset and not bool(p.get('is_addon',False))]
            if reason=='SELECTED' and same and not bool(r.is_addon):
                protected_now=[(p,mark(p,t)) for p in same];indep=all(p['strategy_family']!=r.strategy_family for p,_ in protected_now);same_dir=all(int(p['direction'])==int(r.direction) for p,_ in protected_now);prot=all(s and s['floor']>=0 for _,s in protected_now);asset_r=sum(float(p['risk_cash']) for p in same)
                if not (indep and same_dir and prot and (asset_r+risk)/max(eq,1e-12)<=.0075):reason='SAME_ASSET_NOT_PROTECTED_INDEPENDENT'
            if reason=='SELECTED' and config_week_r.get(ck,0)<=-2:reason='CONFIG_WEEKLY_LOSS_BREAKER'
            if reason=='SELECTED' and asset_week_r.get(ak,0)<=-3:reason='ASSET_WEEKLY_LOSS_BREAKER'
            if reason=='SELECTED' and lot<=0:reason='LOT_BELOW_MIN'
            if reason=='SELECTED' and (open_risk+risk)/max(eq,1e-12)>risk_budget+1e-12:reason='OPEN_RISK_BUDGET'
            if reason=='SELECTED' and (open_margin+margin)/max(eq,1e-12)>.60:reason='MARGIN_BUDGET'
            if reason=='SELECTED' and np.abs(proj).max()>1.25:reason='FACTOR_DUPLICATION'
            u=float(r.rank_score)-.20*(margin/max(eq,1e-12));recycled=''
            if reason in {'OPEN_RISK_BUDGET','MARGIN_BUDGET','FACTOR_DUPLICATION'} and u>=.12:
                pid=recycle(r,t,u,growth)
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
    if len(de):de['opportunity_rescue_count']=rescue_count
    return ev,de

class OpportunityVariant:
    def __init__(self,cfg,base,promote):self.cfg=cfg;self.base=base;self.promote=promote
    def promote_for_week(self,cands,week_start):return self.promote(cands,week_start,'combined')
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):return replay_r5(cands,specs,start,feature_cache,self.cfg)

def cohort_attribution(control_ev,candidate_ev):
    def prep(x,label):
        z=x.copy();z['eval_week']=pd.to_datetime(z['eval_week'],utc=True);z['key']=z['eval_week'].astype(str)+'|'+z['campaign_id'].astype(str);z=z.sort_values('timestamp').drop_duplicates('key',keep='last');return z.set_index('key')
    c=prep(control_ev,'c');v=prep(candidate_ev,'v');common=c.index.intersection(v.index);removed=c.index.difference(v.index);added=v.index.difference(c.index)
    a=c.loc[common,'net_pnl'].astype(float);b=v.loc[common,'net_pnl'].astype(float)
    ctl_flat=a.abs()<=EPS;ctl_loss=a<-EPS;ctl_win=a>EPS;vflat=b.abs()<=EPS;vloss=b<-EPS;vwin=b>EPS
    az=v.loc[added,'net_pnl'].astype(float) if len(added) else pd.Series(dtype=float);rz=c.loc[removed,'net_pnl'].astype(float) if len(removed) else pd.Series(dtype=float)
    return {
      'common_campaigns':int(len(common)),'baseline_flat_to_profit':int((ctl_flat&vwin).sum()),'baseline_flat_to_loss':int((ctl_flat&vloss).sum()),'baseline_loss_to_nonloss':int((ctl_loss&~vloss).sum()),'baseline_losses_improved':int((ctl_loss&(b>a+EPS)).sum()),'baseline_winner_to_loss':int((ctl_win&vloss).sum()),'baseline_winners_harmed':int((ctl_win&(b<a-EPS)).sum()),'common_pnl_delta':float((b-a).sum()),
      'added_campaigns':int(len(added)),'added_pnl':float(az.sum()) if len(az) else 0.,'added_wins':int((az>EPS).sum()) if len(az) else 0,'added_losses':int((az<-EPS).sum()) if len(az) else 0,'added_flats':int((az.abs()<=EPS).sum()) if len(az) else 0,
      'removed_campaigns':int(len(removed)),'removed_control_pnl':float(rz.sum()) if len(rz) else 0.,'removed_control_wins':int((rz>EPS).sum()) if len(rz) else 0,'removed_control_losses':int((rz<-EPS).sum()) if len(rz) else 0,'removed_control_flats':int((rz.abs()<=EPS).sum()) if len(rz) else 0
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--variant-index',type=int,required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    if not 0<=a.variant_index<len(CONFIGS):raise RuntimeError('variant-index outside CONFIGS')
    cfg=CONFIGS[a.variant_index]
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    lock=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'));scope=lock['scope']
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy')
    if len(data)!=34:raise RuntimeError(f'route universe regressed {len(data)}/34')
    ctl_mod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_l1c_ctl_{a.variant_index}')
    ctl,CW,CEV,CDE,CPR=eval_variant(ctl_mod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 control reproduction failed')
    cand,VW,VEV,VDE,VPR=eval_variant(OpportunityVariant(cfg,base,promote),data,cands,opps,fcache,specs,100.0)
    coh=cohort_attribution(CEV,VEV);rescues=int((VEV.get('exit_reason',pd.Series(index=VEV.index,dtype=object))=='R5_5_OPPORTUNITY_RESCUE_RECYCLE').sum()) if len(VEV) else 0
    ctl_loss_rate=float(ctl['losses'])/max(int(ctl['executed_trades']),1);cand_loss_rate=float(cand['losses'])/max(int(cand['executed_trades']),1);ctl_flat_rate=float(ctl['flat'])/max(int(ctl['executed_trades']),1);cand_flat_rate=float(cand['flat'])/max(int(cand['executed_trades']),1)
    gates={
      'end_equity_not_regressed':float(cand['end_equity'])>=float(ctl['end_equity'])-1e-9,
      'fresh_holdout_improved':float(cand['fresh_return_pct'])>float(ctl['fresh_return_pct'])+1e-9,
      'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=float(ctl['max_weekly_dd_pct'])-1e-9,
      'loss_rate_not_worse':cand_loss_rate<=ctl_loss_rate+1e-12,
      'flat_rate_not_worse':cand_flat_rate<=ctl_flat_rate+1e-12,
      'baseline_cohort_repaired':(coh['baseline_flat_to_profit']+coh['baseline_loss_to_nonloss']>0 and coh['baseline_flat_to_loss']==0 and coh['baseline_winner_to_loss']==0),
      'incremental_campaigns_economic':coh['added_campaigns']==0 or coh['added_pnl']>0,
      'full_universe_34x510':scope['approved_routes']==34 and scope['strategy_families']==15 and scope['route_strategy_cells']==510 and len(data)==34
    }
    promotion=all(gates.values())
    status={'state':'R5_5_LANE1C_OPPORTUNITY_RESCUE_COMPLETE','evidence_class':'END_TO_END_CAUSAL_PORTFOLIO_STATE_REPLAY','variant_index':a.variant_index,'config':cfg,'r5_4_control':ctl,'candidate':cand,'delta_end_equity':float(cand['end_equity'])-float(ctl['end_equity']),'fresh_delta_pct':float(cand['fresh_return_pct'])-float(ctl['fresh_return_pct']),'opportunity_rescue_exits':rescues,'rates':{'control_loss_rate':ctl_loss_rate,'candidate_loss_rate':cand_loss_rate,'control_flat_rate':ctl_flat_rate,'candidate_flat_rate':cand_flat_rate},'cohort_attribution':coh,'gates':gates,'promotion_candidate':promotion,'approved_scope':scope,'governance':'Weak-continuation rescue is allowed only when a superior current opportunity is otherwise blocked by portfolio risk/margin/factor capacity. It uses only current portfolio state, current trade path, pre-entry reliability and current competing opportunity utility. Existing protected recycling remains unchanged and has priority.'}
    (out/f'variant_{a.variant_index:02d}.json').write_text(json.dumps(status,indent=2,default=str));VW.to_csv(out/f'variant_{a.variant_index:02d}_weekly.csv',index=False);print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
