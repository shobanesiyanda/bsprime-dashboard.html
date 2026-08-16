#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane4_market_state_entry_repair import enrich_market_state
from lane6b_winner_amplification import rule_mask

FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
AUG10=pd.Timestamp('2026-08-10T00:00:00Z')
EPS=1e-9

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    import R5_BASE_CAUSAL_ENGINE as base
    from reliability_common import annotate,promote
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    R=pd.read_csv(a.rules);R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True);rule=R.iloc[0].to_dict()
    fresh_raw=cands[cands.decision_time>=FRESH0].copy();fresh=annotate(enrich_market_state(fresh_raw,fcache),cands,AUG10);fresh['winner_like']=rule_mask(fresh,rule)
    parents=fresh[~fresh.get('is_addon',False).astype(bool)].copy();qual=parents[parents.winner_like].copy();qual_ids=set(qual.campaign_id.astype(str))
    base_prom,conf,stats=base.promote_for_week(cands,AUG10);combined,_,_=promote(cands,AUG10,'combined')
    base_ids=set(base_prom.campaign_id.astype(str)) if len(base_prom) else set();comb_ids=set(combined.campaign_id.astype(str)) if len(combined) else set()
    rank_map={}
    if len(combined):
        q=combined.sort_values(['decision_time','rank_score'],ascending=[True,False]).copy();q['_rank']=q.groupby('decision_time').cumcount()+1;q['_n']=q.groupby('decision_time').campaign_id.transform('size');rank_map={str(r.campaign_id):(int(r._rank),int(r._n),float(r.rank_score)) for _,r in q.iterrows()}
    rows=[]
    for _,r in qual.sort_values('decision_time').iterrows():
        cid=str(r.campaign_id);rk=rank_map.get(cid)
        rows.append({'campaign_id':cid,'asset':str(r.asset),'asset_class':str(r.asset_class),'strategy':str(r.strategy),'strategy_family':str(r.strategy_family),'decision_time':str(r.decision_time),'simulated_net_r':float(r.net_r),'simulated_mfe_r':float(r.get('mfe_r',np.nan)),'simulated_mae_r':float(r.get('mae_r',np.nan)),'quality':float(r.quality),'cost_r':float(r.cost_r),'reliability_score':float(r.reliability_score),'reliability_asset_strategy_mean_r':float(r.reliability_asset_strategy_mean_r),'base_weekly_promoted':cid in base_ids,'combined_reliability_promoted':cid in comb_ids,'rank_at_decision':rk[0] if rk else None,'competing_promoted_at_decision':rk[1] if rk else None,'combined_rank_score':rk[2] if rk else None})
    qm=dict(zip(parents.campaign_id.astype(str),parents.winner_like.astype(bool)));addons=fresh[fresh.strategy.astype(str).eq('WINNER_ONLY_ADDON')].copy();addons['winner_like_parent']=addons.parent_id.astype(str).map(qm).fillna(False) if 'parent_id' in addons else False;qa=addons[addons.winner_like_parent].copy();addon_rows=[]
    for _,r in qa.iterrows():
        cid=str(r.campaign_id);rk=rank_map.get(cid);addon_rows.append({'campaign_id':cid,'parent_id':str(r.parent_id),'asset':str(r.asset),'decision_time':str(r.decision_time),'simulated_net_r':float(r.net_r),'base_weekly_promoted':cid in base_ids,'combined_reliability_promoted':cid in comb_ids,'rank_at_decision':rk[0] if rk else None,'competing_promoted_at_decision':rk[1] if rk else None})
    nbase=sum(x['base_weekly_promoted'] for x in rows);ncomb=sum(x['combined_reliability_promoted'] for x in rows)
    if len(rows)==0:root='NO_FRESH_WINNER_SIGNATURE'
    elif nbase<len(rows):root='BASE_OPPORTUNITY_SELECTOR_DROPPED_WINNER_LIKE_CANDIDATES'
    elif ncomb<len(rows):root='RELIABILITY_FILTER_DROPPED_WINNER_LIKE_CANDIDATES'
    else:root='ALL_WINNER_LIKE_CANDIDATES_SURVIVED_WEEKLY_PROMOTION; PORTFOLIO_CAPITAL_FACTOR_OR_MANAGEMENT_STAGE_IS_BOTTLENECK'
    s={'state':'R5_5_LANE6E_FAST_SELECTOR_AUDIT_COMPLETE','evidence_class':'QUARANTINED_FRESH_DIAGNOSTIC_NOT_SELECTION_EVIDENCE','scope':{'routes':len(data),'strategy_families':15,'route_strategy_cells':510},'causal_week_start':str(AUG10),'winner_rule':rule,'winner_like_parent_count':len(rows),'base_promoted_count':nbase,'combined_promoted_count':ncomb,'winner_like_parent_trace':rows,'winner_like_addon_trace':addon_rows,'root_cause_stage':root,'governance':'This traces fixed pre-Aug13 winner-like candidates through the Aug10 causal selector only. Realized Aug13-14 outcomes are diagnostic and cannot tune thresholds or promote a candidate. Portfolio execution remains a separate stage.'}
    Path(a.out).write_text(json.dumps(s,indent=2,default=str));print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
