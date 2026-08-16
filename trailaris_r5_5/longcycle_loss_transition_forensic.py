#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# Structural thresholds are inherited from existing engine contracts wherever possible.
TIER_B_MIN_QUALITY = 0.68
TIER_B_MIN_EXPECTED_R = 0.10
TIER_B_MIN_TRAILING_WIN = 0.45
TIER_B_MAX_COST_R = 0.35
RELIABILITY_EVIDENCE_N = 12
MATERIAL_MFE_R = 0.50  # diagnostic attribution only; never used to change historical selection


def pf(g: pd.DataFrame) -> float:
    if g.empty: return 0.0
    r=pd.to_numeric(g.net_r,errors='coerce').fillna(0.0)
    gw=float(r[r>0].sum()); gl=float(-r[r<0].sum())
    return gw/gl if gl else 999.0


def outcome_stats(g: pd.DataFrame) -> dict:
    if g.empty:
        return {'n':0,'wins':0,'losses':0,'flats':0,'loss_rate':0.0,'flat_rate':0.0,'nonflat_win_rate':0.0,'mean_net_r':0.0,'profit_factor':0.0}
    r=pd.to_numeric(g.net_r,errors='coerce').fillna(0.0)
    w=int((r>0).sum()); l=int((r<0).sum()); f=int((r==0).sum())
    return {'n':int(len(g)),'wins':w,'losses':l,'flats':f,'loss_rate':float(l/len(g)),'flat_rate':float(f/len(g)),'nonflat_win_rate':float(w/(w+l)) if w+l else 0.0,'mean_net_r':float(r.mean()),'profit_factor':pf(g)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--events',required=True)
    ap.add_argument('--decisions',required=True)
    ap.add_argument('--promoted',required=True)
    ap.add_argument('--weekly',required=True)
    ap.add_argument('--outdir',required=True)
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)

    ev=pd.read_csv(a.events); de=pd.read_csv(a.decisions); pr=pd.read_csv(a.promoted); wk=pd.read_csv(a.weekly)
    for d in (ev,de,pr):
        for c in ('decision_time','exit_time','timestamp','eval_week'):
            if c in d.columns: d[c]=pd.to_datetime(d[c],utc=True,errors='coerce')
    wk['week_start']=pd.to_datetime(wk.week_start,utc=True,errors='coerce')

    if len(ev)!=2152:
        raise RuntimeError(f'forensic ledger expects certified frozen 26W control executions=2152, got {len(ev)}')

    # Full promoted opportunity set enriched by actual portfolio selection reason.
    q=pr.merge(de[['campaign_id','eval_week','decision_time','selected','reason','week_growth','risk_budget','projected_open_risk_pct']],on=['campaign_id','eval_week'],how='left',suffixes=('','_decision'))
    q['selected']=q.selected.fillna(False).astype(bool)

    # Timestamp-exact opportunity regret. Outcomes are used only after the historical decision.
    rejected_positive=q[(~q.selected)&(q.net_r>0)].copy()
    pos_by_time={}
    for t,g in rejected_positive.groupby('decision_time',sort=False):
        pos_by_time[t]=g[['reason','campaign_id','net_r','rank_score','reliability_score','quality']].to_dict('records')

    ledger=ev.copy()
    ledger['is_loss']=ledger.net_r<0
    ledger['weak_initial_thesis_flag']=ledger.quality<TIER_B_MIN_QUALITY
    ledger['configuration_recent_state_tension_flag']=(ledger.promotion_tier.astype(str).eq('A')) & ((ledger.expected_r<=TIER_B_MIN_EXPECTED_R)|(ledger.trailing_win_rate<TIER_B_MIN_TRAILING_WIN))
    ledger['nonpositive_reliability_state_flag']=(ledger.reliability_strategy_n>=RELIABILITY_EVIDENCE_N)&(ledger.reliability_score<0)
    ledger['execution_cost_pressure_flag']=ledger.cost_r>TIER_B_MAX_COST_R
    ledger['management_rescue_candidate_flag']=ledger.is_loss&(ledger.mfe_r>=MATERIAL_MFE_R)
    ledger['same_timestamp_capacity_regret_flag']=False
    ledger['same_timestamp_factor_regret_flag']=False
    ledger['same_timestamp_other_regret_flag']=False
    ledger['same_timestamp_rejected_positive_count']=0
    ledger['best_rejected_positive_net_r']=np.nan
    ledger['best_rejected_positive_rank_score']=np.nan
    ledger['best_rejected_positive_reason']=''

    for i,r in ledger.iterrows():
        if not bool(r.is_loss): continue
        alts=pos_by_time.get(r.decision_time,[])
        if not alts: continue
        ledger.at[i,'same_timestamp_rejected_positive_count']=len(alts)
        best=max(alts,key=lambda z:float(z['net_r']))
        ledger.at[i,'best_rejected_positive_net_r']=float(best['net_r'])
        ledger.at[i,'best_rejected_positive_rank_score']=float(best['rank_score'])
        ledger.at[i,'best_rejected_positive_reason']=str(best['reason'])
        reasons={str(z['reason']) for z in alts}
        ledger.at[i,'same_timestamp_capacity_regret_flag']='OPEN_RISK_BUDGET' in reasons
        ledger.at[i,'same_timestamp_factor_regret_flag']='FACTOR_DUPLICATION' in reasons
        ledger.at[i,'same_timestamp_other_regret_flag']=bool(reasons-{'OPEN_RISK_BUDGET','FACTOR_DUPLICATION'})

    # Attribution is deliberately non-exclusive: one loss can expose multiple failure surfaces.
    losses=ledger[ledger.is_loss].copy()
    flag_cols=['weak_initial_thesis_flag','configuration_recent_state_tension_flag','nonpositive_reliability_state_flag','execution_cost_pressure_flag','management_rescue_candidate_flag','same_timestamp_capacity_regret_flag','same_timestamp_factor_regret_flag','same_timestamp_other_regret_flag']
    losses['no_identified_avoidable_surface_flag']=~losses[flag_cols].any(axis=1)

    flags={c:{'losses_flagged':int(losses[c].sum()),'share_of_losses':float(losses[c].mean())} for c in flag_cols+['no_identified_avoidable_surface_flag']}

    # Weak-week vs strong-week causal-state comparison. Strong means >=10%; weak means <5% per governance brief.
    week_class=wk[['week_start','weekly_return_pct']].copy(); week_class['bucket']=np.where(week_class.weekly_return_pct<5,'WEAK_LT5',np.where(week_class.weekly_return_pct>=10,'STRONG_GE10','MID'))
    ledger=ledger.merge(week_class[['week_start','bucket']],left_on='eval_week',right_on='week_start',how='left').drop(columns='week_start')
    q=q.merge(week_class[['week_start','bucket']],left_on='eval_week',right_on='week_start',how='left').drop(columns='week_start')
    bucket_rows=[]
    for b in ['WEAK_LT5','MID','STRONG_GE10']:
        e=ledger[ledger.bucket==b]; p=q[q.bucket==b]; sel=p[p.selected]; rej=p[~p.selected]
        bucket_rows.append({
            'bucket':b,'weeks':int((week_class.bucket==b).sum()),'executions':int(len(e)),'loss_rate':float((e.net_r<0).mean()) if len(e) else 0.0,'flat_rate':float((e.net_r==0).mean()) if len(e) else 0.0,'mean_net_r':float(e.net_r.mean()) if len(e) else 0.0,'profit_factor':pf(e),
            'available_promoted_candidates':int(len(p)),'selected_candidates':int(len(sel)),'rejected_candidates':int(len(rej)),'selection_rate':float(len(sel)/len(p)) if len(p) else 0.0,
            'selected_mean_quality':float(sel.quality.mean()) if len(sel) else None,'rejected_mean_quality':float(rej.quality.mean()) if len(rej) else None,'selected_mean_rank_score':float(sel.rank_score.mean()) if len(sel) else None,'rejected_mean_rank_score':float(rej.rank_score.mean()) if len(rej) else None,
            'selected_mean_reliability_score':float(sel.reliability_score.mean()) if len(sel) else None,'rejected_mean_reliability_score':float(rej.reliability_score.mean()) if len(rej) else None,'selected_mean_expected_r':float(sel.expected_r.mean()) if len(sel) else None,
            'open_risk_budget_rejections':int((rej.reason=='OPEN_RISK_BUDGET').sum()),'factor_duplication_rejections':int((rej.reason=='FACTOR_DUPLICATION').sum()),'margin_budget_rejections':int((rej.reason=='MARGIN_BUDGET').sum()),
            'open_risk_budget_rejected_positive':int(((rej.reason=='OPEN_RISK_BUDGET')&(rej.net_r>0)).sum()),'open_risk_budget_rejected_negative':int(((rej.reason=='OPEN_RISK_BUDGET')&(rej.net_r<0)).sum()),'open_risk_budget_rejected_net_r_sum':float(rej.loc[rej.reason=='OPEN_RISK_BUDGET','net_r'].sum())
        })
    buckets=pd.DataFrame(bucket_rows)

    tier={}
    for t,g in ledger.groupby('promotion_tier',sort=False): tier[str(t)]=outcome_stats(g)

    regret={}
    for reason,g in q[~q.selected].groupby('reason',sort=False):
        regret[str(reason)]={'rejected':int(len(g)),'ex_post_wins':int((g.net_r>0).sum()),'ex_post_losses':int((g.net_r<0).sum()),'ex_post_flats':int((g.net_r==0).sum()),'ex_post_net_r_sum':float(g.net_r.sum()),'ex_post_mean_net_r':float(g.net_r.mean()) if len(g) else 0.0}

    report={
        'state':'R5_4_26W_LOSS_TRANSITION_FORENSIC_COMPLETE',
        'executed_trades':int(len(ledger)),'losses':int(losses.shape[0]),'wins':int((ledger.net_r>0).sum()),'flats':int((ledger.net_r==0).sum()),
        'tier_outcome_attribution':tier,'loss_surface_flags_nonexclusive':flags,'weak_vs_strong_week_attribution':buckets.to_dict('records'),'opportunity_regret_by_rejection_reason':regret,
        'causal_vs_hindsight_contract':{
            'causal_at_decision':['quality','promotion_tier','expected_r','trailing_win_rate','reliability_score','reliability_strategy_n','cost_r','rank_score','portfolio rejection reason'],
            'hindsight_only':['realized net_r','mfe_r','mae_r','whether rejected alternative later won','best rejected alternative payoff'],
            'rule':'Hindsight fields classify regret after the fact and MUST NOT be fed backward into historical eligibility, ranking, sizing or management.'
        },
        'structural_threshold_provenance':{
            'quality_0_68':'Frozen R5.4 Tier-B minimum quality','expected_r_0_10':'Frozen R5.4 Tier-B trailing mean requirement','trailing_win_0_45':'Frozen R5.4 Tier-B trailing win-rate requirement','cost_r_0_35':'Frozen R5.4 Tier-B cost ceiling','reliability_n_12':'Frozen hierarchical reliability strategy-evidence gate minimum','mfe_0_50':'Diagnostic favorable-excursion marker only; not an engine promotion threshold.'
        },
        'interpretation':'Flags are failure surfaces, not automatic removal rules. Overlap is expected. The ledger is designed to locate causal repair candidates while preserving asymmetric winners; promotion still requires a complete full-universe causal replay.'
    }

    ledger.to_csv(out/'R5_4_26W_LOSS_TRANSITION_LEDGER.csv',index=False)
    buckets.to_csv(out/'R5_4_26W_WEAK_STRONG_FORENSIC.csv',index=False)
    (out/'R5_4_26W_LOSS_TRANSITION_FORENSIC.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__': main()
