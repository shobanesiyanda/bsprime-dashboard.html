#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane5_pareto_management_stack import StackVariant,apply_stack
from lane6b_winner_amplification import rule_mask
from winner_causal_features import causal_decision_features

R54_END=337.4231242104393
LANE5_END=345.3896887826546
LANE5_LOSSES=228
LANE5_FLATS=421
LANE5_NFWR=0.673352435530086
FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
FRESH_FLOOR=7.261646557142409
DD_FLOOR=-4.637645574907678
FRESH_DD_FLOOR=-0.5196525640746019
LANE5_KEYS=['B3','B5','B8']
EPS=1e-9

class WinnerCaptureRescueVariant:
    """6D subcase: give otherwise-unpromoted, causally winner-like base candidates a conservative Tier-B-risk route into normal portfolio competition.

    The winner rule is frozen before Aug13. Reliability/market-state fields are rebuilt on raw candidates using only history available at the decision week.
    This does not bypass portfolio risk, margin, factor, breaker, lot or same-asset constraints.
    """
    def __init__(self,base,promote,history,causal_features,rule):
        self.base=base;self.promote=promote;self.history=history;self.cf=causal_features;self.rule=rule
        self.rescue_eligible=0;self.rescue_by_week={}
    def promote_for_week(self,cands,week_start):
        p,conf,stats=self.promote(cands,week_start,'combined')
        t0=pd.Timestamp(week_start);t1=t0+pd.Timedelta(days=7)
        target=self.cf[(self.cf.decision_time>=t0)&(self.cf.decision_time<t1)].copy()
        if target.empty:return p,conf,stats
        # Foundation entries only. Add-ons remain governed by the existing post-proof machinery.
        target=target[~target.get('is_addon',pd.Series(False,index=target.index)).astype(bool)].copy()
        m=rule_mask(target,self.rule)
        # Use the engine's existing conservative Tier-B quality/cost envelope; no fresh-tuned threshold is introduced.
        m &= pd.to_numeric(target.quality,errors='coerce').fillna(0).ge(.68)
        m &= pd.to_numeric(target.cost_r,errors='coerce').fillna(99).le(.35)
        rescue=target[m].copy()
        existing=set(p.campaign_id.astype(str)) if len(p) else set()
        rescue=rescue[~rescue.campaign_id.astype(str).isin(existing)].copy()
        if rescue.empty:return p,conf,stats
        statmap={(r.asset,r.strategy):r for _,r in stats.iterrows()} if len(stats) else {}
        er=[];wr=[];stab=[];gb=[]
        for _,r in rescue.iterrows():
            st=statmap.get((r.asset,r.strategy));mean=0.;win=.5;sd=1.;give=0.
            if st is not None:
                mean=float(st.mean_r);win=float(st.win);sd=0 if pd.isna(st.sd) else float(st.sd);give=float(st.mean_giveback)
            er.append(mean);wr.append(win);stab.append(mean/max(sd,.25));gb.append(give)
        rescue['expected_r']=er;rescue['trailing_win_rate']=wr;rescue['stability']=stab;rescue['trailing_giveback']=gb
        rescue['promotion_tier']='W';rescue['risk_fraction']=.0025
        rescue['rank_score']=(.42*pd.to_numeric(rescue.quality,errors='coerce').fillna(0)+.28*np.tanh(rescue.expected_r)+.16*rescue.trailing_win_rate+.10*np.tanh(rescue.stability)-.22*pd.to_numeric(rescue.cost_r,errors='coerce').fillna(0)-.04*np.minimum(rescue.trailing_giveback,3)+.18*pd.to_numeric(rescue.reliability_score,errors='coerce').fillna(0))
        # Same causal reliability rejection used by the combined R5.4 reliability selector.
        bad=((pd.to_numeric(rescue.reliability_strategy_n,errors='coerce').fillna(0)>=12)&(pd.to_numeric(rescue.reliability_expected_r,errors='coerce').fillna(0)<-.04))|((pd.to_numeric(rescue.reliability_n,errors='coerce').fillna(0)>=10)&(pd.to_numeric(rescue.reliability_expected_r,errors='coerce').fillna(0)<-.08))
        bad |= ((pd.to_numeric(rescue.reliability_hour_mean_r,errors='coerce').fillna(0)<-.06)&(pd.to_numeric(rescue.reliability_expected_r,errors='coerce').fillna(0)<0)&(pd.to_numeric(rescue.reliability_strategy_n,errors='coerce').fillna(0)>=12))
        rescue=rescue[~bad].copy();rescue['winner_capture_rescue']=True
        if rescue.empty:return p,conf,stats
        self.rescue_eligible+=len(rescue);self.rescue_by_week[str(t0)]=int(len(rescue))
        if len(p):p=p.copy();p['winner_capture_rescue']=False
        z=pd.concat([p,rescue],ignore_index=True,sort=False)
        z=z.sort_values(['campaign_id','promotion_tier','rank_score'],ascending=[True,True,False]).drop_duplicates('campaign_id')
        return z,conf,stats
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):
        return self.base.replay_r5(apply_stack(cands,feature_cache,LANE5_KEYS),specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--rule-rank',type=int,default=0);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    R=pd.read_csv(a.rules);R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True)
    if a.rule_rank>=len(R):raise RuntimeError(f'winner rule rank unavailable {a.rule_rank}/{len(R)}')
    rule=R.iloc[a.rule_rank].to_dict()
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy');scope=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'))['scope']
    scope_ok=(len(data)==34 and int(scope.get('approved_routes',0))==34 and int(scope.get('strategy_families',0))==15 and int(scope.get('route_strategy_cells',0))==510)
    if not scope_ok:raise RuntimeError(f'full-universe scope failure data={len(data)} scope={scope}')
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'r55_l6drescue_ctl');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.)
    if abs(float(ctl['end_equity'])-R54_END)>EPS:raise RuntimeError(f'R5.4 reproduction failed {ctl["end_equity"]}')
    lane5,*_=eval_variant(StackVariant(LANE5_KEYS,base,promote),data,cands,opps,fcache,specs,100.)
    if abs(float(lane5['end_equity'])-LANE5_END)>1e-6 or int(lane5['losses'])!=LANE5_LOSSES or int(lane5['flat'])!=LANE5_FLATS:raise RuntimeError(f'Lane5 reproduction failed {lane5}')
    cf=causal_decision_features(cands,fcache)
    var=WinnerCaptureRescueVariant(base,promote,cands,cf,rule)
    cand,VW,VEV,VDE,VPR=eval_variant(var,data,cands,opps,fcache,specs,100.)
    flag=VEV.get('winner_capture_rescue',pd.Series(False,index=VEV.index)).fillna(False).astype(bool) if len(VEV) else pd.Series(dtype=bool)
    selected_rescue=int(flag.sum());fresh_mask=pd.to_datetime(VEV.decision_time,utc=True).ge(FRESH0) if len(VEV) and 'decision_time' in VEV else pd.Series(False,index=VEV.index)
    fresh_rescue=int((flag&fresh_mask).sum()) if len(VEV) else 0
    rescue_ev=VEV[flag].copy() if len(VEV) else VEV
    rescue_pnl=float(pd.to_numeric(rescue_ev.get('net_pnl',pd.Series(dtype=float)),errors='coerce').fillna(0).sum()) if len(rescue_ev) else 0.
    fresh_rescue_pnl=float(pd.to_numeric(VEV.loc[flag&fresh_mask].get('net_pnl',pd.Series(dtype=float)),errors='coerce').fillna(0).sum()) if len(VEV) and (flag&fresh_mask).any() else 0.
    gates={'end_equity_above_lane5':float(cand['end_equity'])>LANE5_END+EPS,'fresh_return_above_lane5':float(cand['fresh_return_pct'])>FRESH_FLOOR+EPS,'max_weekly_dd_not_worse':float(cand['max_weekly_dd_pct'])>=DD_FLOOR-EPS,'fresh_dd_not_worse':float(cand['fresh_max_dd_pct'])>=FRESH_DD_FLOOR-EPS,'losses_at_or_below_228':int(cand['losses'])<=LANE5_LOSSES,'flats_at_or_below_421':int(cand['flat'])<=LANE5_FLATS,'nonflat_win_rate_at_least_lane5':float(cand['nonflat_win_rate'])>=LANE5_NFWR-EPS,'assets_at_least_33':int(cand['assets_with_selected'])>=33,'full_universe_34x15x510':scope_ok}
    s={'state':'R5_5_LANE6D_WINNER_CAPTURE_RESCUE_REPLAY_COMPLETE','evidence_class':'CAUSAL_PRE_PROMOTION_WINNER_SIGNATURE_RESCUE_FULL_PORTFOLIO_REPLAY','winner_rule':rule,'mechanism':{'subpath':'6D_WINNER_PRIORITY_CAPTURE','rescue_risk_fraction':.0025,'quality_floor':.68,'cost_r_ceiling':.35,'existing_portfolio_constraints_unchanged':True},'r5_4_control':ctl,'certified_lane5_control':lane5,'candidate':cand,'delta_vs_lane5':float(cand['end_equity'])-LANE5_END,'fresh_delta_vs_lane5':float(cand['fresh_return_pct'])-FRESH_FLOOR,'rescue_counts':{'eligible_pre_selection':var.rescue_eligible,'eligible_by_week':var.rescue_by_week,'selected_executions':selected_rescue,'fresh_selected_executions':fresh_rescue,'selected_rescue_net_pnl':rescue_pnl,'fresh_selected_rescue_net_pnl':fresh_rescue_pnl},'frontier_gates':gates,'frontier_pass':bool(all(gates.values())),'fresh_evidence_status':'REPAIR_DIAGNOSTIC_CONTAMINATED_FOR_THIS_MECHANISM','candidate_freeze':False,'scope':scope,'governance':'The winner signature itself was frozen before Aug13. This rescue mechanism was motivated by quarantined Aug13-14 diagnosis showing profitable winner-like candidates can die before baseline validation promotion, so Aug13-14 cannot serve as independent validation for this mechanism. Historical causal replay may establish non-regression and mechanism value; a new unseen forward holdout after candidate freeze remains mandatory. Rescue is conservative Tier-B risk and never bypasses portfolio risk, margin, factor, lot, breaker or same-asset controls.'}
    (out/'R5_5_LANE6D_RESCUE_STATUS.json').write_text(json.dumps(s,indent=2,default=str));VW.to_csv(out/'weekly.csv',index=False);print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
