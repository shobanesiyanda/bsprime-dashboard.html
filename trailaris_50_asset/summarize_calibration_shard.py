#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe,anchor_for,classify

SESSIONS=[('ASIA',0,7),('LONDON',7,13),('NEW_YORK',13,21),('LATE',21,24)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rawdir',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--candidate-list',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
 ev=json.loads((a.rawdir/'ACQUISITION_EVIDENCE.json').read_text());candset={x.strip() for x in a.candidate_list.read_text().splitlines() if x.strip()}
 passed=[e for e in ev.get('evidence',[]) if e.get('status')=='PASS' and e.get('file')]
 meta=pd.DataFrame([{'asset':str(e['canonical_instrument']),'discovery_provider_name':str(e.get('provider_name',''))} for e in passed])
 r4=None
 try:
  import trailaris_r4_full_universe_loop as r4
  install_universe(r4,meta)
 except Exception:
  r4=None
 daily=[];sessions=[];quality=[];opp=[]
 for e in passed:
  asset=str(e['canonical_instrument']);provider=str(e.get('provider_name',''));role=str(e.get('role','CANDIDATE' if asset in candset else 'CORE'))
  z=pd.read_csv(e['file'],usecols=['timestamp','close']);z['timestamp']=pd.to_datetime(z.timestamp,utc=True,errors='coerce');z['close']=pd.to_numeric(z.close,errors='coerce');z=z.dropna().sort_values('timestamp').drop_duplicates('timestamp')
  lr=np.log(z.close).diff().replace([np.inf,-np.inf],np.nan);z['abs_lr']=lr.abs();z['minute_gap']=z.timestamp.diff().dt.total_seconds().div(60)
  dc=z.set_index('timestamp').close.resample('1D').last().dropna();dr=dc.pct_change().dropna()
  for t,v in dr.items():daily.append({'date':t.date().isoformat(),'asset':asset,'return':float(v),'role':role,'asset_class':classify(asset,provider),'anchor':anchor_for(asset,provider)})
  act=[]
  for name,h0,h1 in SESSIONS:
   v=float(z.loc[z.timestamp.dt.hour.ge(h0)&z.timestamp.dt.hour.lt(h1),'abs_lr'].sum(skipna=True));act.append((name,v))
  total=sum(v for _,v in act) or 1.0
  for name,v in act:sessions.append({'asset':asset,'session':name,'activity_share':v/total,'role':role})
  local_gap=z.minute_gap.where(z.minute_gap.le(180));large=float((local_gap>5).mean()) if local_gap.notna().any() else 1.0
  stale=float((lr.abs()<1e-14).mean()) if len(lr)>1 else 1.0
  quality.append({'asset':asset,'role':role,'asset_class':classify(asset,provider),'provider_name':provider,'rows':len(z),'first':str(z.timestamp.iloc[0]),'last':str(z.timestamp.iloc[-1]),'daily_observations':len(dr),'stale_return_share':stale,'local_gap_gt5m_share':large,'coverage_pass':True,'anchor':anchor_for(asset,provider)})
  if asset in candset and r4 is not None:
   try:
    a0,c,xf,o=r4.build_asset_components((asset,str(e['file'])));fam=0;rows=0
    if c is not None and len(c):
     rows=len(c);col='strategy_family' if 'strategy_family' in c.columns else ('strategy' if 'strategy' in c.columns else None);fam=int(c[col].nunique()) if col else 0
    opp.append({'asset':asset,'candidate_rows':rows,'strategy_family_breadth':fam,'candidate_density_per_10000_m1':10000*rows/max(len(z),1),'build_pass':True})
   except Exception as ex:opp.append({'asset':asset,'candidate_rows':0,'strategy_family_breadth':0,'candidate_density_per_10000_m1':0.0,'build_pass':False,'error':str(ex)[:300]})
 pd.DataFrame(daily).to_csv(a.outdir/'DAILY_RETURNS.csv',index=False);pd.DataFrame(sessions).to_csv(a.outdir/'SESSION_PROFILE.csv',index=False);pd.DataFrame(quality).to_csv(a.outdir/'DATA_QUALITY.csv',index=False);pd.DataFrame(opp).to_csv(a.outdir/'OPPORTUNITY_STRUCTURE.csv',index=False)
 status={'state':'STRUCTURAL_CALIBRATION_SUMMARY_COMPLETE','assets_pass':len(passed),'candidate_assets_summarized':sum(str(e['canonical_instrument']) in candset for e in passed),'uses_strategy_outcomes':False,'window_start':ev.get('window_start'),'window_end_exclusive':ev.get('window_end_exclusive')};(a.outdir/'SUMMARY.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
if __name__=='__main__':main()
