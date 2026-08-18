#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe

FACTORY_FAMILIES=[
 'SUPPLY_DEMAND','LIQUIDITY_STRUCTURE','TREND_FOLLOWING','MOMENTUM_PULLBACK','BREAKOUT_RETEST','MEAN_REVERSION','VOLATILITY_REGIME','SESSION_TIME','ORDER_FLOW_MICROSTRUCTURE','RELATIVE_VALUE_PAIRS','INTERMARKET','CARRY_VALUE','CRYPTO_FUNDING_BASIS','EVENT_RESPONSE','ENSEMBLE_REGIME'
]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle-root',type=Path,required=True);ap.add_argument('--factor-feed',type=Path,required=True);ap.add_argument('--approved16',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    approved=[x.strip() for x in a.approved16.read_text().splitlines() if x.strip()]
    if len(approved)!=16 or len(set(approved))!=16:raise RuntimeError('approved-16 identity gate failed')
    idx=[];parts=[];fcache={}
    for f in sorted(a.bundle_root.rglob('FEATURE_INDEX.csv')):
        z=pd.read_csv(f)
        for _,r in z.iterrows():
            asset=str(r.asset);fp=f.parent/str(r.feature_file)
            if not fp.exists():raise RuntimeError(f'missing feature file {asset}: {fp}')
            x=pd.read_csv(fp);x['timestamp']=pd.to_datetime(x.timestamp,utc=True,errors='coerce')
            if x['timestamp'].isna().any():raise RuntimeError(f'invalid feature timestamp {asset}')
            fcache[asset]=x.sort_values('timestamp').reset_index(drop=True)
            idx.append({'asset':asset,'discovery_provider_name':str(r.discovery_provider_name),'instrument_id':str(r.instrument_id),'source_feature_file':str(fp)})
    for f in sorted(a.bundle_root.rglob('CANDIDATES.csv.gz')):
        try:z=pd.read_csv(f)
        except pd.errors.EmptyDataError:continue
        if len(z):parts.append(z)
    meta=pd.DataFrame(idx).drop_duplicates('asset').sort_values('asset').reset_index(drop=True)
    if len(meta)!=50:raise RuntimeError(f'50-asset feature integrity failed: {len(meta)}/50')
    original=['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
    expected=set(original+approved)
    if set(meta.asset.astype(str))!=expected:raise RuntimeError(f'exact 50 identity mismatch missing={sorted(expected-set(meta.asset.astype(str)))} extra={sorted(set(meta.asset.astype(str))-expected)}')
    if not parts:raise RuntimeError('no asset-local candidates')
    base=pd.concat(parts,ignore_index=True)
    import trailaris_r4_full_universe_loop as r4
    mapping=install_universe(r4,meta)
    if len(set(r4.ROUTES))!=50:raise RuntimeError(f'extended route registry failed: {len(set(r4.ROUTES))}/50')
    extras=[]
    for fn in (r4.generate_relative_value_candidates,r4.generate_intermarket_candidates):
        z=fn(fcache)
        if z is not None and len(z):extras.append(z)
    ff=r4.load_factor_feed(a.factor_feed);z=r4.generate_factor_candidates(fcache,ff,'research-proxy')
    if z is not None and len(z):extras.append(z)
    cands=pd.concat([base,*extras],ignore_index=True) if extras else base.copy();ens=r4.generate_ensemble_candidates(cands,fcache)
    if ens is not None and len(ens):cands=pd.concat([cands,ens],ignore_index=True)
    import trailaris_r4_asset_adapter as r4a
    for asset in meta.asset.astype(str):
        for family in FACTORY_FAMILIES:
            v=r4a.family_applicable(asset,family)
            if not isinstance(v,(bool,int)):raise RuntimeError(f'non-boolean applicability {asset}/{family}: {v!r}')
    m=cands.apply(lambda q:(q.strategy_family=='CAMPAIGN_MANAGEMENT') or r4a.family_applicable(str(q.asset),str(q.strategy_family)),axis=1)
    cands=cands[m].reset_index(drop=True);cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True);cands=cands.sort_values(['decision_time','campaign_id']).reset_index(drop=True)
    if int(cands.asset.nunique())!=50:raise RuntimeError(f'candidate asset integrity {int(cands.asset.nunique())}/50')
    cands.to_csv(a.outdir/'ALL_CANDIDATES.csv.gz',index=False,compression='gzip')
    outidx=[]
    for _,r in meta.iterrows():
        asset=str(r.asset);src=Path(str(r.source_feature_file));fn=f'{asset.replace("/","_").replace(".","_")}_{src.name}'
        dst=a.outdir/fn;dst.write_bytes(src.read_bytes());outidx.append({'asset':asset,'discovery_provider_name':str(r.discovery_provider_name),'instrument_id':str(r.instrument_id),'feature_file':fn})
    pd.DataFrame(outidx).to_csv(a.outdir/'FEATURE_INDEX.csv',index=False)
    pd.DataFrame([{'asset':k,'anchor':v} for k,v in sorted(mapping.items())]).to_csv(a.outdir/'UNIVERSE_ADAPTER_MAP.csv',index=False)
    observed=sorted(set(cands.strategy_family.astype(str))) if 'strategy_family' in cands else [];unknown=sorted(set(observed)-set(FACTORY_FAMILIES)-{'CAMPAIGN_MANAGEMENT'})
    if unknown:raise RuntimeError(f'unknown strategy families in candidates: {unknown}')
    out={'state':'EXACT_50_ASSET_UNIVERSE_ASSEMBLED','feature_assets':len(meta),'asset_local_candidate_rows':len(base),'total_candidate_rows':len(cands),'candidate_assets':int(cands.asset.nunique()),'strategy_families':15,'factory_strategy_family_names':FACTORY_FAMILIES,'triggered_candidate_strategy_family_names':observed,'global_relative_intermarket_factor_ensemble_generation':True,'approved_additions':approved,'quant_v2_modified':False}
    (a.outdir/'ASSEMBLY_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
