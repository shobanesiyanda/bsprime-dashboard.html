#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane4_market_state_entry_repair import enrich_market_state

FRESH0=pd.Timestamp('2026-08-13T00:00:00Z')
EPS=1e-9

def num(x,d=np.nan):
    try:
        v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def cond(z,f,d,th):
    if f not in z:return pd.Series(False,index=z.index)
    s=pd.to_numeric(z[f],errors='coerce');return s.le(th) if d=='le' else s.ge(th)

def rule_mask(z,r):
    if str(r['scope_type'])=='GLOBAL':m=pd.Series(True,index=z.index)
    elif str(r['scope_type'])=='ASSET_CLASS':m=z.asset_class.astype(str).eq(str(r['scope_value']))
    else:m=z.strategy_family.astype(str).eq(str(r['scope_value']))
    m &= cond(z,str(r['feature1']),str(r['dir1']),num(r['threshold1']))
    f2=str(r.get('feature2','') or '')
    if f2 and f2.lower()!='nan':m &= cond(z,f2,str(r['dir2']),num(r['threshold2']))
    return m.fillna(False)

def outcome(z):
    if not len(z):return {'n':0,'wins':0,'losses':0,'flats':0,'target_hits':0,'mean_net_r':0.0,'total_net_r':0.0}
    p=pd.to_numeric(z.get('net_r',z.get('net_pnl',0)),errors='coerce').fillna(0)
    er=z.get('exit_reason',pd.Series('',index=z.index)).astype(str)
    return {'n':int(len(z)),'wins':int((p>EPS).sum()),'losses':int((p<-EPS).sum()),'flats':int((p.abs()<=EPS).sum()),'target_hits':int(er.eq('TARGET').sum()),'mean_net_r':float(p.mean()),'total_net_r':float(p.sum())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    import R5_BASE_CAUSAL_ENGINE as base
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir))
    z=enrich_market_state(cands.copy(),fcache);z['decision_time']=pd.to_datetime(z.decision_time,utc=True)
    R=pd.read_csv(a.rules);R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True)
    fresh=z[z.decision_time>=FRESH0].copy();prior=z[z.decision_time<FRESH0].copy()
    parent_fresh=fresh[~fresh.get('is_addon',False).astype(bool)].copy();parent_prior=prior[~prior.get('is_addon',False).astype(bool)].copy()
    rows=[]
    for rank,r in R.head(20).iterrows():
        rr=r.to_dict();mp=rule_mask(parent_prior,rr);mf=rule_mask(parent_fresh,rr)
        qp=parent_prior.loc[mp].copy();qf=parent_fresh.loc[mf].copy()
        pmap=dict(zip(parent_fresh.campaign_id.astype(str),mf.astype(bool))) if 'campaign_id' in parent_fresh else {}
        addons=fresh[fresh.strategy.astype(str).eq('WINNER_ONLY_ADDON')].copy() if 'strategy' in fresh else fresh.iloc[0:0]
        if len(addons) and 'parent_id' in addons:addons=addons[addons.parent_id.astype(str).map(pmap).fillna(False)]
        else:addons=addons.iloc[0:0]
        rows.append({'rule_rank':int(rank),'scope_type':rr['scope_type'],'scope_value':rr['scope_value'],'feature1':rr['feature1'],'feature2':str(rr.get('feature2','') or ''),'prior_parent_count':int(len(qp)),'fresh_parent_count':int(len(qf)),'fresh_parent_share':float(len(qf)/max(len(parent_fresh),1)),'fresh_qualified_addon_candidates':int(len(addons)),**{f'fresh_{k}':vv for k,vv in outcome(qf).items()}})
    df=pd.DataFrame(rows);df.to_csv(out/'R5_5_LANE6E_FAST_RULE_PRESENCE.csv',index=False)
    best=df.iloc[0].to_dict() if len(df) else {}
    status={'state':'R5_5_LANE6E_FAST_CANDIDATE_PRESENCE_COMPLETE','evidence_class':'QUARANTINED_FRESH_DIAGNOSTIC_ONLY','scope':{'routes_with_data':len(data),'approved_routes':34,'strategy_families':15,'route_strategy_cells':510},'fresh_candidate_population':int(len(fresh)),'fresh_parent_candidate_population':int(len(parent_fresh)),'best_pre_aug13_rule_presence':best,'top20_rules_with_fresh_parent_presence':int((df.fresh_parent_count>0).sum()) if len(df) else 0,'top20_rules_with_fresh_addon_opportunity':int((df.fresh_qualified_addon_candidates>0).sum()) if len(df) else 0,'interpretation':('WINNER_SIGNATURE_PRESENT_IN_FRESH_CANDIDATES' if best.get('fresh_parent_count',0)>0 else 'BEST_WINNER_SIGNATURE_ABSENT_FROM_FRESH_CANDIDATES'),'governance':'This fast diagnostic inspects causal candidate state and simulated candidate path only. It does not measure portfolio selection, realized equity, or promotion. Full Lane6E remains authoritative for selection/capital/portfolio bottleneck classification.'}
    (out/'R5_5_LANE6E_FAST_STATUS.json').write_text(json.dumps(status,indent=2,default=str));print(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
