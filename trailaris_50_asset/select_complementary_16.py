#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,glob,re
from pathlib import Path
import numpy as np,pandas as pd
CORE=['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
CAPS={'FX':6,'EQUITY':5,'INDEX':3,'CRYPTO':2,'COMMODITY':5,'METAL':5,'OTHER':1}

def pct(s,high=True):
 s=pd.to_numeric(s,errors='coerce');r=s.rank(pct=True,method='average');return r if high else 1-r

def group(a,cls):
 a=str(a).upper()
 if cls=='CRYPTO':return next((x for x in ['BTC','ETH','SOL','BNB','XRP','BCH','LTC','ADA'] if a.startswith(x)),a)
 if cls=='METAL':return 'XAU' if a.startswith('XAU') else ('XAG' if a.startswith('XAG') else a)
 if cls=='INDEX':
  if a.startswith(('US','DOLLAR')):return 'US_INDEX'
  if a.startswith(('FRA','NLD','CHE')):return 'EU_INDEX'
  return a
 if cls=='COMMODITY':return 'ENERGY' if any(x in a for x in ['DIESEL','BRENT','WTI','GAS']) else a
 return a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--summaries-root',type=Path,required=True);ap.add_argument('--candidate-list',type=Path,required=True);ap.add_argument('--candidate-manifest',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
 cand=[x.strip() for x in a.candidate_list.read_text().splitlines() if x.strip()];assert len(cand)==66 and len(set(cand))==66
 dfs=[];ss=[];qq=[];oo=[]
 for f in a.summaries_root.rglob('DAILY_RETURNS.csv'):
  try:dfs.append(pd.read_csv(f))
  except:pass
 for f in a.summaries_root.rglob('SESSION_PROFILE.csv'):
  try:ss.append(pd.read_csv(f))
  except:pass
 for f in a.summaries_root.rglob('DATA_QUALITY.csv'):
  try:qq.append(pd.read_csv(f))
  except:pass
 for f in a.summaries_root.rglob('OPPORTUNITY_STRUCTURE.csv'):
  try:
   z=pd.read_csv(f)
   if len(z):oo.append(z)
  except:pass
 D=pd.concat(dfs,ignore_index=True).drop_duplicates(['date','asset']);S=pd.concat(ss,ignore_index=True).drop_duplicates(['asset','session']);Q=pd.concat(qq,ignore_index=True).drop_duplicates('asset');O=pd.concat(oo,ignore_index=True).drop_duplicates('asset') if oo else pd.DataFrame(columns=['asset'])
 have_core=set(D.asset)&set(CORE)
 if have_core!=set(CORE):raise RuntimeError(f'core structural reference incomplete {len(have_core)}/34 missing={sorted(set(CORE)-have_core)}')
 eligible=set(Q.loc[Q.coverage_pass.astype(bool),'asset'])&set(cand)
 if 'build_pass' in O.columns:eligible&=set(O.loc[O.build_pass.astype(bool),'asset'])
 if len(eligible)<16:raise RuntimeError(f'only {len(eligible)} structurally eligible candidates')
 D['date']=pd.to_datetime(D.date);W=D.pivot(index='date',columns='asset',values='return').sort_index();W=W[W.index.dayofweek<5]
 core=W[CORE].copy();core_port=core.mean(axis=1,skipna=True);stress=set(core_port.nsmallest(max(10,int(.10*core_port.notna().sum()))).index)
 X=core.copy();X=X.apply(lambda s:(s-s.mean())/(s.std(ddof=0) if s.std(ddof=0)>1e-12 else 1.)).fillna(0.).to_numpy(float);U,sv,Vt=np.linalg.svd(X,full_matrices=False);var=sv**2;cum=np.cumsum(var)/max(var.sum(),1e-12);k=max(1,min(10,int(np.searchsorted(cum,.85)+1)));PC=U[:,:k]*sv[:k]
 core_sess=S[S.asset.isin(CORE)].groupby('session').activity_share.mean();core_sess=core_sess/core_sess.sum()
 rows=[]
 for asset in cand:
  if asset not in eligible:continue
  y=W[asset] if asset in W else pd.Series(index=W.index,dtype=float);valid=y.notna();cors=[]
  for c in CORE:
   z=pd.concat([y,core[c]],axis=1).dropna();
   if len(z)>=40:cors.append(abs(float(z.iloc[:,0].corr(z.iloc[:,1]))))
  if not cors:continue
  yy=y.loc[valid].astype(float);pc=PC[valid.to_numpy(),:];ys=(yy-yy.mean())/(yy.std(ddof=0) if yy.std(ddof=0)>1e-12 else 1.);coef=np.linalg.lstsq(np.c_[np.ones(len(pc)),pc],ys.to_numpy(),rcond=None)[0];res=ys.to_numpy()-np.c_[np.ones(len(pc)),pc]@coef;resid=float(np.var(res)/max(np.var(ys.to_numpy()),1e-12))
  st=[d for d in stress if d in y.index and pd.notna(y.loc[d])];cod=float(np.mean([y.loc[d]<0 for d in st])) if st else 1.
  rv=y.rolling(10,min_periods=6).std();cv=core_port.rolling(10,min_periods=6).std();zv=pd.concat([rv,cv],axis=1).dropna();volcorr=abs(float(zv.iloc[:,0].corr(zv.iloc[:,1]))) if len(zv)>=20 else 1.
  er=(y.rolling(20,min_periods=12).sum().abs()/y.abs().rolling(20,min_periods=12).sum().replace(0,np.nan)).mean();core_er=[]
  for c in CORE:core_er.append((core[c].rolling(20,min_periods=12).sum().abs()/core[c].abs().rolling(20,min_periods=12).sum().replace(0,np.nan)).mean())
  trenddiff=abs(float(er)-float(np.nanmedian(core_er))) if pd.notna(er) else 0.
  sp=S[S.asset.eq(asset)].set_index('session').activity_share;idx=sorted(set(core_sess.index)|set(sp.index));cs=np.array([core_sess.get(i,0.) for i in idx]);asv=np.array([sp.get(i,0.) for i in idx]);sessionnov=.5*float(np.abs(cs-asv).sum())
  q=Q[Q.asset.eq(asset)].iloc[0];oc=O[O.asset.eq(asset)].iloc[0] if asset in set(O.asset) else pd.Series(dtype=object);cls=str(q.asset_class);g=group(asset,cls);dup=1.0 if any(group(x,('CRYPTO' if x in ['BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD'] else 'METAL' if x in ['XAUUSD','XAGUSD'] else 'INDEX' if x in ['US100','US500','US30','GER40','UK100','JP225'] else 'COMMODITY' if x in ['WTI','BRENT','NATGAS'] else 'FX' if x in CORE[:13] else 'SYNTHETIC'))==g for x in CORE) else 0.0
  rows.append({'asset':asset,'asset_class':cls,'exposure_group':g,'mean_abs_core_corr':float(np.mean(cors)),'max_abs_core_corr':float(np.max(cors)),'residual_variance_ratio':resid,'tail_codrawdown_frequency':cod,'rolling_vol_corr_abs':volcorr,'trend_chop_difference':trenddiff,'session_novelty':sessionnov,'stale_return_share':float(q.stale_return_share),'local_gap_gt5m_share':float(q.local_gap_gt5m_share),'daily_observations':int(q.daily_observations),'candidate_density_per_10000_m1':float(oc.get('candidate_density_per_10000_m1',0.)),'strategy_family_breadth':int(oc.get('strategy_family_breadth',0)),'core_underlying_duplicate':dup,'anchor':str(q.get('anchor',''))})
 R=pd.DataFrame(rows);assert len(R)>=16
 R['structural_score']=.20*pct(R.mean_abs_core_corr,False)+.12*pct(R.max_abs_core_corr,False)+.18*pct(R.residual_variance_ratio,True)+.10*pct(R.tail_codrawdown_frequency,False)+.08*pct(R.rolling_vol_corr_abs,False)+.07*pct(R.trend_chop_difference,True)+.08*pct(R.session_novelty,True)+.06*pct(R.stale_return_share,False)+.03*pct(R.local_gap_gt5m_share,False)+.04*pct(np.log1p(R.candidate_density_per_10000_m1),True)+.04*pct(R.strategy_family_breadth,True)-.12*R.core_underlying_duplicate
 corr=W[[x for x in cand if x in W]].corr().abs();selected=[];counts={}
 while len(selected)<16:
  best=None
  for _,r in R[~R.asset.isin(selected)].iterrows():
   cls=r.asset_class
   if counts.get(cls,0)>=CAPS.get(cls,2):continue
   if selected:
    vals=[corr.loc[r.asset,s] for s in selected if r.asset in corr.index and s in corr.columns and pd.notna(corr.loc[r.asset,s])];pair=float(np.mean(vals)) if vals else .5;grp=sum(group(s,str(R.loc[R.asset.eq(s),'asset_class'].iloc[0]))==r.exposure_group for s in selected)
   else:pair=0.;grp=0
   diversity_bonus=.025 if counts.get(cls,0)==0 else 0.;joint=float(r.structural_score)-.22*pair-.06*grp+diversity_bonus
   if best is None or joint>best[0]:best=(joint,r,pair,grp)
  if best is None:raise RuntimeError('class caps prevented selection of 16')
  _,r,pair,grp=best;selected.append(r.asset);counts[r.asset_class]=counts.get(r.asset_class,0)+1;R.loc[R.asset.eq(r.asset),'joint_selection_score']=best[0];R.loc[R.asset.eq(r.asset),'pairwise_selected_corr_at_entry']=pair;R.loc[R.asset.eq(r.asset),'selected_order']=len(selected)
 if len(set(R.loc[R.asset.isin(selected),'asset_class']))<3:raise RuntimeError('selected additions do not span at least 3 asset classes')
 R['selected']=R.asset.isin(selected);R=R.sort_values(['selected','selected_order','structural_score'],ascending=[False,True,False]);R.to_csv(a.outdir/'QV2_50_CANDIDATE_STRUCTURAL_SCORES.csv',index=False)
 M=pd.read_csv(a.candidate_manifest,sep='\t');M=M[M.canonical_instrument.isin(selected)].copy();M['selected_order']=M.canonical_instrument.map({x:i+1 for i,x in enumerate(selected)});M=M.sort_values('selected_order');M.to_csv(a.outdir/'QV2_50_SELECTED_16.tsv',sep='\t',index=False);(a.outdir/'QV2_50_SELECTED_16.txt').write_text('\n'.join(selected)+'\n')
 sel=R[R.selected].sort_values('selected_order');e={'state':'COMPLEMENTARY_16_SELECTED_PRE_REPLAY','selection_window':'2025-07-21T00:00:00Z to 2026-01-19T00:00:00Z','evaluation_window_not_used_for_selection':True,'strategy_outcomes_used':False,'core_assets':34,'candidate_pool_predeclared':66,'structurally_eligible_candidates':len(R),'selected_assets':selected,'selected_class_mix':sel.asset_class.value_counts().to_dict(),'mean_selected_core_correlation':float(sel.mean_abs_core_corr.mean()),'mean_selected_residual_variance_ratio':float(sel.residual_variance_ratio.mean()),'mean_selected_tail_codrawdown_frequency':float(sel.tail_codrawdown_frequency.mean()),'mean_selected_pairwise_corr_at_entry':float(sel.pairwise_selected_corr_at_entry.fillna(0).mean()),'selection_method':'joint minimum-redundancy structural complementarity; no evaluation-period P&L or strategy outcome used','automatic_50_asset_replay_authorized':False}
 (a.outdir/'QV2_50_SELECTION_EVIDENCE.json').write_text(json.dumps(e,indent=2));print(json.dumps(e,indent=2))
if __name__=='__main__':main()
