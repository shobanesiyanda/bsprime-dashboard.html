#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from winner_causal_features import causal_decision_features
from lane6b_winner_amplification import rule_mask

FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
EPS=1e-9

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    import R5_BASE_CAUSAL_ENGINE as base
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir))
    R=pd.read_csv(a.rules);R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True);best=R.iloc[0].to_dict()
    z=causal_decision_features(cands,fcache);z['decision_time']=pd.to_datetime(z.decision_time,utc=True);z['winner_like']=rule_mask(z,best)
    fresh=z[z.decision_time>=FRESH0].copy();parents=fresh[~fresh.get('is_addon',False).astype(bool)].copy();qual=parents[parents.winner_like].copy();qm=dict(zip(parents.campaign_id.astype(str),parents.winner_like.astype(bool)))
    addons=fresh[fresh.strategy.astype(str).eq('WINNER_ONLY_ADDON')].copy();addons['qualified_parent']=addons.parent_id.astype(str).map(qm).fillna(False) if 'parent_id' in addons else False;qa=addons[addons.qualified_parent].copy()
    pnl=pd.to_numeric(qual.get('net_r',pd.Series(dtype=float)),errors='coerce').fillna(0);exit_reason=qual.get('exit_reason',pd.Series('',index=qual.index)).astype(str)
    by_family=qual.groupby('strategy_family').size().sort_values(ascending=False).head(10).to_dict() if len(qual) else {};by_asset=qual.groupby('asset').size().sort_values(ascending=False).head(10).to_dict() if len(qual) else {}
    s={'state':'R5_5_LANE6E_FAST_FRESH_CENSUS_COMPLETE','evidence_class':'QUARANTINED_FRESH_DIAGNOSTIC_NOT_SELECTION_EVIDENCE','scope':{'routes':len(data),'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'best_pre_aug13_winner_rule':best,'fresh_candidate_counts':{'all_candidates':int(len(fresh)),'parent_candidates':int(len(parents)),'winner_like_parent_candidates':int(len(qual)),'winner_like_parent_simulated_wins':int((pnl>EPS).sum()),'winner_like_parent_simulated_losses':int((pnl<-EPS).sum()),'winner_like_parent_simulated_flats':int((pnl.abs()<=EPS).sum()),'winner_like_parent_3R_targets':int((exit_reason=='TARGET').sum()),'winner_only_addon_candidates':int(len(addons)),'winner_only_addons_with_winner_like_parent':int(len(qa))},'winner_like_by_family':{str(k):int(v) for k,v in by_family.items()},'winner_like_by_asset':{str(k):int(v) for k,v in by_asset.items()},'diagnostic_interpretation':('WINNER_SIGNATURE_ABSENT_IN_FRESH_CANDIDATE_UNIVERSE' if len(qual)==0 else ('WINNER_SIGNATURE_PRESENT_BUT_NO_CONTINUATION_ADDON_CANDIDATES' if len(qa)==0 else 'WINNER_SIGNATURE_AND_CONTINUATION_CANDIDATES_PRESENT; FULL_6E_MUST_RESOLVE_SELECTION_OR_MONETIZATION_BOTTLENECK')),'governance':'Uses weekly-causal decision features over the complete 34-route candidate universe. Aug13-14 realized candidate outcomes are diagnostic only and cannot select or promote a rule.'}
    Path(a.out).write_text(json.dumps(s,indent=2,default=str));print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
