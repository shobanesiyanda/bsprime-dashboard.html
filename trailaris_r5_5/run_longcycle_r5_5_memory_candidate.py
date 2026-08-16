#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

CONTROL_END = 1136.4556497760627
CONTROL_TRADES = 2152
CONTROL_WEEKS = 26
EPS = 1e-8


def load_module(path, name):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def rolling_windows(W, n=11):
    r=pd.to_numeric(W.weekly_return_pct,errors='coerce').fillna(0).to_numpy(float); rows=[]
    for i in range(len(W)-n+1):
        z=r[i:i+n]; ret=(np.prod(1+z/100)-1)*100
        rows.append({'window':i+1,'start_week':str(W.week_start.iloc[i]),'end_week':str(W.week_start.iloc[i+n-1]),'compounded_return_pct':float(ret),'mean_week_pct':float(np.mean(z)),'median_week_pct':float(np.median(z)),'positive_weeks':int((z>0).sum()),'weeks_ge5':int((z>=5).sum()),'weeks_ge10':int((z>=10).sum()),'max_weekly_dd_pct':float(pd.to_numeric(W.max_drawdown_pct.iloc[i:i+n],errors='coerce').min())})
    return pd.DataFrame(rows)


def metrics(name, res, W, EV, DE, PR):
    roll=rolling_windows(W,11)
    wins=int((EV.net_pnl>0).sum()) if len(EV) else 0; losses=int((EV.net_pnl<0).sum()) if len(EV) else 0; flats=int((EV.net_pnl==0).sum()) if len(EV) else 0
    gw=float(EV.loc[EV.net_pnl>0,'net_pnl'].sum()) if len(EV) else 0.; gl=float(-EV.loc[EV.net_pnl<0,'net_pnl'].sum()) if len(EV) else 0.
    if len(EV) and 'equity' in EV:
        eq=pd.concat([pd.Series([100.0]),pd.to_numeric(EV.sort_values('timestamp').equity,errors='coerce').dropna()],ignore_index=True); dd=(eq/eq.cummax()-1)*100; event_dd=float(dd.min())
    else: event_dd=0.
    regret={}
    if len(PR) and len(DE):
        q=PR.merge(DE[['campaign_id','eval_week','selected','reason']],on=['campaign_id','eval_week'],how='left')
        for reason in ['OPEN_RISK_BUDGET','FACTOR_DUPLICATION','MARGIN_BUDGET','CONFIG_WEEKLY_LOSS_BREAKER','ASSET_WEEKLY_LOSS_BREAKER','SAME_ASSET_NOT_PROTECTED_INDEPENDENT']:
            g=q[(q.selected==False)&(q.reason==reason)]
            regret[reason]={'rejected':int(len(g)),'ex_post_positive':int((g.net_r>0).sum()) if len(g) else 0,'ex_post_negative':int((g.net_r<0).sum()) if len(g) else 0,'ex_post_flat':int((g.net_r==0).sum()) if len(g) else 0,'ex_post_net_r_sum':float(g.net_r.sum()) if len(g) else 0.}
    return {
      'name':name,'start_equity':100.0,'end_equity':float(W.end_equity.iloc[-1]),'compounded_return_pct':float((W.end_equity.iloc[-1]/100-1)*100),'evaluated_weeks':int(len(W)),
      'positive_weeks':int((W.weekly_return_pct>0).sum()),'negative_weeks':int((W.weekly_return_pct<0).sum()),'mean_week_pct':float(W.weekly_return_pct.mean()),'median_week_pct':float(W.weekly_return_pct.median()),'weeks_ge5':int((W.weekly_return_pct>=5).sum()),'weeks_ge10':int((W.weekly_return_pct>=10).sum()),
      'executed_trades':int(len(EV)),'wins':wins,'losses':losses,'flats':flats,'loss_rate':float(losses/len(EV)) if len(EV) else 0.,'flat_rate':float(flats/len(EV)) if len(EV) else 0.,'nonflat_win_rate':float(wins/(wins+losses)) if wins+losses else 0.,'profit_factor':float(gw/gl) if gl else 999.,
      'max_weekly_drawdown_pct':float(W.max_drawdown_pct.min()),'max_event_equity_drawdown_pct':event_dd,'assets_with_selected':int(DE.loc[DE.selected,'asset'].nunique()) if len(DE) else 0,
      'rolling_11w_windows':int(len(roll)),'rolling_11w_min_return_pct':float(roll.compounded_return_pct.min()),'rolling_11w_median_return_pct':float(roll.compounded_return_pct.median()),'rolling_11w_upper_quartile_return_pct':float(roll.compounded_return_pct.quantile(.75)),'rolling_11w_mean_return_pct':float(roll.compounded_return_pct.mean()),'rolling_11w_max_return_pct':float(roll.compounded_return_pct.max()),
      'opportunity_regret_hindsight_attribution':regret
    }, roll


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rawdir',required=True); ap.add_argument('--specs',required=True); ap.add_argument('--outdir',required=True); a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=['trailaris_r4','trailaris_r5_4','trailaris_r5_4/reliability_variants','trailaris_r5_5']
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from precision_experiment_harness import eval_variant
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir)); specs=r4.load_specs(Path(a.specs),'research-proxy')
    if len(data)!=34: raise RuntimeError(f'full-universe data gate failed {len(data)}/34')
    audit=Path('trailaris_longcycle/results/R4_FACTORY_AUDIT.csv')
    if not audit.exists(): raise RuntimeError('frozen 510-cell audit missing')
    fa=pd.read_csv(audit); fams=int(fa.strategy_family.nunique())
    if len(fa)!=510 or fams!=15: raise RuntimeError(f'full-universe factory lock failed cells={len(fa)} families={fams}')

    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),'frozen_r54_26w_control')
    candmod=load_module(Path('trailaris_r5_5/longcycle_bounded_memory_candidate.py'),'r55_bounded_memory_112d')
    ctl_res,CW,CEV,CDE,CPR=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if len(CW)!=CONTROL_WEEKS or len(CEV)!=CONTROL_TRADES or abs(float(CW.end_equity.iloc[-1])-CONTROL_END)>1e-6:
        raise RuntimeError(f'frozen control reproduction failed end={CW.end_equity.iloc[-1]} trades={len(CEV)} weeks={len(CW)}')
    cand_res,VW,VEV,VDE,VPR=eval_variant(candmod,data,cands,opps,fcache,specs,100.0)
    if len(VW)!=26: raise RuntimeError(f'candidate evaluation drift: {len(VW)} weeks')

    ctl,CR=metrics('R5.4_FROZEN_26W_CONTROL',ctl_res,CW,CEV,CDE,CPR)
    cand,VR=metrics('R5.5_BOUNDED_PROPOSAL_MEMORY_112D',cand_res,VW,VEV,VDE,VPR)
    delta={k:(cand[k]-ctl[k]) for k in ['end_equity','compounded_return_pct','executed_trades','wins','losses','flats','loss_rate','flat_rate','nonflat_win_rate','profit_factor','max_weekly_drawdown_pct','max_event_equity_drawdown_pct','rolling_11w_min_return_pct','rolling_11w_median_return_pct','rolling_11w_upper_quartile_return_pct','rolling_11w_mean_return_pct','rolling_11w_max_return_pct']}
    gates={
      'ending_equity_above_control':cand['end_equity']>ctl['end_equity']+EPS,
      'rolling_11w_min_not_worse':cand['rolling_11w_min_return_pct']>=ctl['rolling_11w_min_return_pct']-EPS,
      'rolling_11w_median_not_worse':cand['rolling_11w_median_return_pct']>=ctl['rolling_11w_median_return_pct']-EPS,
      'rolling_11w_upper_quartile_not_worse':cand['rolling_11w_upper_quartile_return_pct']>=ctl['rolling_11w_upper_quartile_return_pct']-EPS,
      'max_weekly_dd_not_worse':cand['max_weekly_drawdown_pct']>=ctl['max_weekly_drawdown_pct']-EPS,
      'event_dd_not_worse':cand['max_event_equity_drawdown_pct']>=ctl['max_event_equity_drawdown_pct']-EPS,
      'positive_week_consistency_preserved':cand['positive_weeks']>=ctl['positive_weeks'],
      'full_scope_34x15x510':len(data)==34 and fams==15 and len(fa)==510,
    }
    status={
      'state':'R5_5_LONG_CYCLE_CAUSAL_CANDIDATE_REPLAY_COMPLETE','engine_change':'CONFIGURATION_GENERATION_MEMORY_ONLY','proposal_memory_days':112,'reliability_memory_days':56,'validation_memory_days':14,
      'causal_contract':'Each evaluation week uses only candidate outcomes exited before that week. Proposal search sees at most the preceding 112 days ending before the unchanged 14-day confirmation window. No future week is visible to proposal, validation, reliability, ranking, selection, management or sizing.',
      'r5_4_control':ctl,'candidate':cand,'delta_candidate_minus_control':delta,'baseline_promotion_gates':gates,'becomes_new_baseline':bool(all(gates.values())),
      'scope':{'routes':34,'strategy_families':15,'route_strategy_cells':510},'evidence_class':'FULL_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE',
      'governance':'This test changes only configuration-generation memory. Reliability memory, 14-day confirmation, ranking formula, hierarchical reliability overlay, selection, risk fractions, factor caps, management and portfolio risk budgets remain frozen R5.4. Hindsight is used only in the opportunity-regret attribution fields after replay.'
    }
    CW.to_csv(out/'R5_4_CONTROL_WEEKLY.csv',index=False); CR.to_csv(out/'R5_4_CONTROL_ROLLING_11W.csv',index=False)
    VW.to_csv(out/'R5_5_CANDIDATE_WEEKLY.csv',index=False); VR.to_csv(out/'R5_5_CANDIDATE_ROLLING_11W.csv',index=False)
    VEV.to_csv(out/'R5_5_CANDIDATE_EVENTS.csv',index=False); VDE.to_csv(out/'R5_5_CANDIDATE_DECISIONS.csv',index=False); VPR.to_csv(out/'R5_5_CANDIDATE_PROMOTED.csv',index=False)
    (out/'R5_5_LONGCYCLE_MEMORY_STATUS.json').write_text(json.dumps(status,indent=2,default=str))
    print(json.dumps(status,indent=2,default=str))

if __name__=='__main__': main()
