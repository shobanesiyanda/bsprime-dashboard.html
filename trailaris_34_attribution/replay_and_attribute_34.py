#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from trailaris_full_universe.qv2_asset_universe_adapter import install_universe, extend_research_specs
from trailaris_100_asset.select_replay_100 import metrics

CLASS_MAP={
'EURUSD':'FX','GBPUSD':'FX','USDJPY':'FX','AUDUSD':'FX','USDCAD':'FX','USDCHF':'FX','NZDUSD':'FX','EURJPY':'FX','GBPJPY':'FX','EURGBP':'FX','AUDJPY':'FX','CADJPY':'FX','GBPCHF':'FX',
'XAUUSD':'METAL','XAGUSD':'METAL','BTCUSD':'CRYPTO','ETHUSD':'CRYPTO','SOLUSD':'CRYPTO','BNBUSD':'CRYPTO','XRPUSD':'CRYPTO','US100':'INDEX','US500':'INDEX','US30':'INDEX','GER40':'INDEX','UK100':'INDEX','JP225':'INDEX','WTI':'ENERGY','BRENT':'ENERGY','NATGAS':'ENERGY','V75':'SYNTHETIC','V50':'SYNTHETIC','V100':'SYNTHETIC','V25':'SYNTHETIC','V10':'SYNTHETIC'}

def safe_pf(x):
    w=float(x.loc[x.net_pnl>0,'net_pnl'].sum());l=float(-x.loc[x.net_pnl<0,'net_pnl'].sum());return w/l if l>0 else (999. if w>0 else 0.)

def group_summary(df,col,total_net):
    rows=[]
    for k,g in df.groupby(col,dropna=False):
        wins=int((g.net_pnl>0).sum());losses=int((g.net_pnl<0).sum());flats=int((g.net_pnl==0).sum());nf=wins+losses
        net=float(g.net_pnl.sum());gw=float(g.loc[g.net_pnl>0,'net_pnl'].sum());gl=float(-g.loc[g.net_pnl<0,'net_pnl'].sum())
        rows.append({col:str(k),'trades':int(len(g)),'wins':wins,'losses':losses,'flats':flats,'nonflat_win_rate':wins/nf if nf else 0.,'profit_factor':gw/gl if gl else (999. if gw else 0.),'net_pnl_usd':net,'net_profit_contribution_pct':100*net/total_net if total_net else 0.,'expectancy_r':float(pd.to_numeric(g.net_r,errors='coerce').mean()),'mean_net_pnl_usd':float(g.net_pnl.mean()),'mean_nominal_risk_fraction':float(pd.to_numeric(g.risk_fraction,errors='coerce').mean()),'mean_actual_risk_pct_equity':float((100*pd.to_numeric(g.risk_cash,errors='coerce')/pd.to_numeric(g.equity_at_entry,errors='coerce')).mean())})
    return pd.DataFrame(rows).sort_values('net_pnl_usd',ascending=False).reset_index(drop=True)

def rejection_summary(df,stage_col):
    rows=[]
    if df.empty:return pd.DataFrame()
    for k,g in df.groupby(stage_col,dropna=False):
        r=pd.to_numeric(g.net_r,errors='coerce');w=int((r>0).sum());l=int((r<0).sum());nf=w+l;gw=float(r[r>0].sum());gl=float(-r[r<0].sum())
        rows.append({stage_col:str(k),'opportunities':int(len(g)),'expost_mean_net_r':float(r.mean()),'expost_median_net_r':float(r.median()),'expost_nonflat_win_rate':w/nf if nf else 0.,'expost_r_profit_factor':gw/gl if gl else (999. if gw else 0.),'expost_sum_net_r':float(r.sum())})
    return pd.DataFrame(rows).sort_values('opportunities',ascending=False).reset_index(drop=True)

def mark_r(row,t,fcache):
    x=fcache.get(str(row.asset));
    if x is None or len(x)==0:return np.nan
    ts=x.timestamp;lo=int(ts.searchsorted(pd.Timestamp(row.entry_time),side='left'));hi=int(ts.searchsorted(pd.Timestamp(t),side='right'))
    if hi<=lo:return np.nan
    px=float(x.iloc[hi-1].close);ent=float(row.entry_price);sd=max(float(row.stop_distance),1e-12);dr=int(row.direction)
    return (px-ent)*dr/sd-float(row.cost_r)

def concurrency_and_factor(ev,r4):
    rows=[];events=[]
    for i,r in ev.reset_index(drop=True).iterrows():
        events.append((pd.Timestamp(r.exit_time),0,i));events.append((pd.Timestamp(r.entry_time),1,i))
    events.sort(key=lambda z:(z[0],z[1],z[2]));active={};factor=np.zeros(len(r4.FACTORS));maxc=0;maxfa=0.;entry_counts=[];entry_risk=[];factor_entry=[]
    for t,kind,i in events:
        r=ev.iloc[i]
        if kind==0:
            if i in active:
                factor-=r4.factor_vec(str(r.asset),int(r.direction));active.pop(i,None)
        else:
            active[i]=r;factor+=r4.factor_vec(str(r.asset),int(r.direction));maxc=max(maxc,len(active));fa=float(np.abs(factor).max()) if len(factor) else 0.;maxfa=max(maxfa,fa);eq=max(float(r.equity_at_entry),1e-12);risk=sum(float(q.risk_cash) for q in active.values())/eq
            entry_counts.append(len(active));entry_risk.append(risk);factor_entry.append(fa)
    return {'max_concurrent_positions':int(maxc),'median_concurrent_positions_at_entry':float(np.median(entry_counts)) if entry_counts else 0.,'p95_concurrent_positions_at_entry':float(np.quantile(entry_counts,.95)) if entry_counts else 0.,'max_open_nominal_risk_pct_equity_at_entry':float(100*max(entry_risk)) if entry_risk else 0.,'median_open_nominal_risk_pct_equity_at_entry':float(100*np.median(entry_risk)) if entry_risk else 0.,'max_absolute_factor_state':float(maxfa),'pct_entries_factor_state_ge_1':float(100*np.mean(np.array(factor_entry)>=1.0)) if factor_entry else 0.}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--universe-dir',type=Path,required=True);ap.add_argument('--promoted-root',type=Path,required=True);ap.add_argument('--specs',type=Path,required=True);ap.add_argument('--control',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    cands=pd.read_csv(a.universe_dir/'ALL_CANDIDATES.csv.gz');cands['decision_time']=pd.to_datetime(cands.decision_time,utc=True);cands['exit_time']=pd.to_datetime(cands.exit_time,utc=True)
    idx=pd.read_csv(a.universe_dir/'FEATURE_INDEX.csv');fcache={}
    for _,r in idx.iterrows():
        x=pd.read_csv(a.universe_dir/str(r.feature_file));x['timestamp']=pd.to_datetime(x.timestamp,utc=True);fcache[str(r.asset)]=x.sort_values('timestamp').reset_index(drop=True)
    if len(fcache)!=34 or int(cands.asset.nunique())!=34:raise RuntimeError('exact 34 data integrity failed')
    import trailaris_r4_full_universe_loop as r4
    meta=idx[['asset','discovery_provider_name']].drop_duplicates().copy();install_universe(r4,meta)
    bs=r4.load_specs(a.specs,'research-proxy');bsf=bs.reset_index() if 'asset' not in bs.columns else bs.copy();specs=extend_research_specs(bsf,meta).set_index('asset',drop=True)
    import R5_BASE_CAUSAL_ENGINE as base;base.r4.factor_vec=r4.factor_vec
    import quantitative_integrated_v2 as qv2
    pfiles=sorted(a.promoted_root.rglob('QV2_34_PROMOTED_SHARD_*.csv.gz'));gfiles=sorted(a.promoted_root.rglob('QV2_34_GATE_REJECTIONS_SHARD_*.csv.gz'))
    pp=[];gg=[]
    for f in pfiles:
        try:z=pd.read_csv(f)
        except pd.errors.EmptyDataError:continue
        if len(z):pp.append(z)
    for f in gfiles:
        try:z=pd.read_csv(f)
        except pd.errors.EmptyDataError:continue
        if len(z):gg.append(z)
    P=pd.concat(pp,ignore_index=True);G=pd.concat(gg,ignore_index=True) if gg else pd.DataFrame()
    for col in ['eval_week','decision_time','exit_time']:P[col]=pd.to_datetime(P[col],utc=True)
    pairs=P[['eval_week_index','eval_week']].drop_duplicates();
    if len(pairs)!=26 or sorted(pairs.eval_week_index.astype(int).tolist())!=list(range(26)):raise RuntimeError('promotion week integrity failed')
    weeks=[pd.Timestamp(x) for x in pairs.sort_values('eval_week_index').eval_week.tolist()];P=P.sort_values(['eval_week','decision_time','campaign_id']).reset_index(drop=True)
    equity=100.;Wrows=[];events=[];decisions=[];raw_end=max(pd.Timestamp(x.timestamp.iloc[-1]) for x in fcache.values())
    for wk in weeks:
        pr=P[P.eval_week.eq(wk)].copy();ev,de=qv2.replay_r5(pr,specs,equity,fcache) if len(pr) else (pd.DataFrame(),pd.DataFrame())
        if len(ev):equity=float(ev.equity.iloc[-1]);events.append(ev.assign(eval_week=wk));decisions.append(de.assign(eval_week=wk))
        prev=Wrows[-1]['end_equity'] if Wrows else 100.;w=r4.weekly(ev,start=prev,feature_cache=fcache,start_ts=wk,end_ts=min(wk+pd.Timedelta(days=7),raw_end))
        if len(w):rec=w.iloc[0].to_dict();equity=float(rec['end_equity'])
        else:rec=dict(week_start=wk,start_equity=prev,end_equity=prev,weekly_return_pct=0.,campaigns_entered=0,campaigns_closed=0,max_drawdown_pct=0.,ge5=False,ge10=False);equity=prev
        Wrows.append(rec)
    W=pd.DataFrame(Wrows);EV=pd.concat(events,ignore_index=True);DE=pd.concat(decisions,ignore_index=True)
    for col in ['entry_time','decision_time','exit_time','timestamp']:EV[col]=pd.to_datetime(EV[col],utc=True)
    DE['decision_time']=pd.to_datetime(DE.decision_time,utc=True);DE['eval_week']=pd.to_datetime(DE.eval_week,utc=True)
    m=metrics(W,EV,DE,cands,P,fcache);control=json.loads(a.control.read_text())['realized_closed_balance_replay']
    if abs(float(m['end_balance'])-float(control['end_equity']))>1e-6 or int(m['executed_trades'])!=int(control['executed_trades']) or abs(float(m['profit_factor'])-float(control['profit_factor']))>1e-8:raise RuntimeError(f'34 control mismatch reconstructed={m} control={control}')
    total_net=float(EV.net_pnl.sum());EV['asset_class']=EV.asset.map(CLASS_MAP).fillna('OTHER')
    asset=group_summary(EV,'asset',total_net);family=group_summary(EV,'strategy_family',total_net);tier=group_summary(EV,'promotion_tier',total_net);classes=group_summary(EV,'asset_class',total_net)
    asset.to_csv(a.outdir/'QV2_34_ATTRIBUTION_BY_ASSET.csv',index=False);family.to_csv(a.outdir/'QV2_34_ATTRIBUTION_BY_STRATEGY_FAMILY.csv',index=False);tier.to_csv(a.outdir/'QV2_34_ATTRIBUTION_BY_TIER.csv',index=False);classes.to_csv(a.outdir/'QV2_34_ATTRIBUTION_BY_ASSET_CLASS.csv',index=False)
    # right-tail concentration
    winners=EV[EV.net_pnl>0].sort_values('net_pnl',ascending=False);gw=float(winners.net_pnl.sum());n10=max(1,int(math.ceil(.10*len(winners))));n20=max(1,int(math.ceil(.20*len(winners))))
    right_tail={'winner_count':int(len(winners)),'gross_winner_pnl_usd':gw,'top_10pct_winners_share_of_gross_winner_pnl_pct':float(100*winners.head(n10).net_pnl.sum()/gw),'top_20pct_winners_share_of_gross_winner_pnl_pct':float(100*winners.head(n20).net_pnl.sum()/gw),'trades_net_r_ge_2':int((EV.net_r>=2).sum()),'trades_net_r_ge_3':int((EV.net_r>=3).sum()),'net_pnl_from_net_r_ge_2_usd':float(EV.loc[EV.net_r>=2,'net_pnl'].sum()),'net_pnl_from_net_r_ge_3_usd':float(EV.loc[EV.net_r>=3,'net_pnl'].sum())}
    # management and recycle
    repaired=EV['r5_5_management_repaired'].fillna(False).astype(bool) if 'r5_5_management_repaired' in EV else pd.Series(False,index=EV.index);recycle=EV.exit_reason.astype(str).str.contains('DYNAMIC_CAPITAL_RECYCLE_R5',na=False)
    management={'protected_runner_trades':int((EV.management.astype(str)=='PROTECTED_RUNNER').sum()),'r5_5_partial_repair_trades':int(repaired.sum()),'r5_5_partial_repair_net_pnl_usd':float(EV.loc[repaired,'net_pnl'].sum()),'r5_5_partial_repair_profit_factor':safe_pf(EV.loc[repaired]) if repaired.any() else 0.,'dynamic_recycle_exits':int(recycle.sum()),'dynamic_recycle_net_pnl_usd':float(EV.loc[recycle,'net_pnl'].sum()),'dynamic_recycle_positive_exit_rate':float((EV.loc[recycle,'net_pnl']>0).mean()) if recycle.any() else 0.}
    # gate cascade totals and efficacy
    totals={'target_candidate_rows':0,'bounded_promoted':0,'after_r5_5_negative_gate':0,'quant_v2_promoted':0,'r5_5_removed':0,'quant_v2_removed':0}
    for f in a.promoted_root.rglob('QV2_34_PROMOTION_SHARD_*_STATUS.json'):
        s=json.loads(f.read_text())
        for w in s.get('weeks',[]):
            for k in totals:totals[k]+=int(w.get(k,0))
    gate_eff=rejection_summary(G,'gate_stage') if len(G) else pd.DataFrame();gate_eff.to_csv(a.outdir/'QV2_34_GATE_REJECTION_EFFICACY.csv',index=False)
    # portfolio decision rejection efficacy using promoted outcomes
    pm=P[['campaign_id','eval_week','net_r']].copy();pm['eval_week']=pd.to_datetime(pm.eval_week,utc=True);dm=DE.merge(pm,on=['campaign_id','eval_week'],how='left',suffixes=('','_promoted'))
    port_eff=rejection_summary(dm[dm.reason.astype(str)!='SELECTED'],'reason');port_eff.to_csv(a.outdir/'QV2_34_PORTFOLIO_REJECTION_EFFICACY.csv',index=False)
    decision_counts=DE.reason.value_counts().to_dict()
    # drawdown forensic
    trough_i=int(pd.to_numeric(EV.drawdown,errors='coerce').idxmin());tr=EV.loc[trough_i];trough_t=pd.Timestamp(tr.timestamp);prior=EV.loc[EV.timestamp<=trough_t].copy();peak_val=max(100.,float(prior.equity.max()));peak_rows=prior[np.isclose(prior.equity.astype(float),peak_val)] if peak_val>100 else prior.iloc[0:0];peak_t=pd.Timestamp(peak_rows.iloc[-1].timestamp) if len(peak_rows) else pd.Timestamp(weeks[0])
    ep=EV[(EV.timestamp>peak_t)&(EV.timestamp<=trough_t)].copy();active=EV[(EV.entry_time<=trough_t)&(EV.exit_time>trough_t)].copy();active['current_r_at_trough']=[mark_r(r,trough_t,fcache) for _,r in active.iterrows()];active['unrealized_pnl_at_trough']=active.risk_cash*active.current_r_at_trough
    real=ep.groupby('asset').net_pnl.sum().rename('realized_pnl_peak_to_trough');unr=active.groupby('asset').unrealized_pnl_at_trough.sum().rename('unrealized_pnl_at_trough');dc=pd.concat([real,unr],axis=1).fillna(0);dc['combined_pnl_to_trough']=dc.sum(axis=1);dc=dc.sort_values('combined_pnl_to_trough');dc.to_csv(a.outdir/'QV2_34_MAX_DD_ATTRIBUTION_BY_ASSET.csv')
    fv=np.zeros(len(r4.FACTORS))
    for _,r in active.iterrows():fv+=r4.factor_vec(str(r.asset),int(r.direction))
    factor_state={str(r4.FACTORS[i]):float(fv[i]) for i in range(len(fv))}
    drawdown={'peak_timestamp':str(peak_t),'peak_equity_usd':peak_val,'trough_timestamp':str(trough_t),'trough_equity_usd':float(tr.equity),'event_drawdown_pct':float(100*tr.drawdown),'realized_closures_peak_to_trough':int(len(ep)),'realized_pnl_peak_to_trough_usd':float(ep.net_pnl.sum()),'open_positions_at_trough':int(len(active)),'unrealized_pnl_open_at_trough_usd':float(active.unrealized_pnl_at_trough.sum()),'active_assets_at_trough':sorted(active.asset.astype(str).unique().tolist()),'factor_state_at_trough':factor_state,'max_abs_factor_state_at_trough':float(np.abs(fv).max()) if len(fv) else 0.}
    active[['asset','strategy_family','promotion_tier','risk_cash','current_r_at_trough','unrealized_pnl_at_trough','entry_time','exit_time']].to_csv(a.outdir/'QV2_34_MAX_DD_ACTIVE_POSITIONS.csv',index=False)
    exposure=concurrency_and_factor(EV,r4)
    profitable_assets=int((asset.net_pnl_usd>0).sum());pf_gt1_assets=int((asset.profit_factor>1).sum());top5=float(asset.head(5).net_pnl_usd.sum()/total_net*100) if total_net else 0.
    sel=dm[dm.reason.astype(str)=='SELECTED'];rej=dm[dm.reason.astype(str)!='SELECTED'];selection_edge={'promoted_opportunities':int(len(P)),'selected_trades':int(len(EV)),'capture_pct':float(100*len(EV)/len(P)),'selected_expost_mean_net_r':float(pd.to_numeric(sel.net_r_promoted,errors='coerce').mean()) if 'net_r_promoted' in sel else float(pd.to_numeric(sel.net_r,errors='coerce').mean()),'rejected_expost_mean_net_r':float(pd.to_numeric(rej.net_r_promoted,errors='coerce').mean()) if 'net_r_promoted' in rej else float(pd.to_numeric(rej.net_r,errors='coerce').mean())}
    summary={'state':'QV2_34_SUCCESS_ATTRIBUTION_COMPLETE','control_reproduction':{'end_balance':float(m['end_balance']),'profit_factor':float(m['profit_factor']),'executed_trades':int(m['executed_trades']),'positive_weeks':int(m['positive_weeks']),'max_event_equity_drawdown_pct':float(m['max_event_equity_drawdown_pct']),'max_weekly_drawdown_pct':float(m['max_weekly_drawdown_pct'])},'profit_breadth':{'profitable_assets':profitable_assets,'assets_pf_gt_1':pf_gt1_assets,'assets_with_selected':int(EV.asset.nunique()),'top_5_assets_share_of_net_profit_pct':top5,'profitable_strategy_families':int((family.net_pnl_usd>0).sum()),'strategy_families_with_trades':int(len(family))},'gate_cascade':{**totals,'bounded_promotion_rate_from_target_pct':100*totals['bounded_promoted']/totals['target_candidate_rows'] if totals['target_candidate_rows'] else 0.,'r5_5_removal_rate_from_bounded_pct':100*totals['r5_5_removed']/totals['bounded_promoted'] if totals['bounded_promoted'] else 0.,'quant_v2_removal_rate_from_post_r5_5_pct':100*totals['quant_v2_removed']/totals['after_r5_5_negative_gate'] if totals['after_r5_5_negative_gate'] else 0.},'portfolio_decision_reason_counts':{str(k):int(v) for k,v in decision_counts.items()},'selection_edge':selection_edge,'right_tail':right_tail,'management':management,'exposure_geometry':exposure,'max_drawdown_forensic':drawdown,'interpretation_guardrail':'Attribution is diagnostic on the already-observed 26-week control. It identifies mechanisms associated with the 34 success but does not by itself establish causal generalization to unseen periods.','strategy_blob':'120390ab52e80f51bafedc42389ceb7420a410f2'}
    (a.outdir/'QV2_34_SUCCESS_ATTRIBUTION_SUMMARY.json').write_text(json.dumps(summary,indent=2,default=str));W.to_csv(a.outdir/'QV2_34_WEEKLY_REPRODUCED.csv',index=False);EV.to_csv(a.outdir/'QV2_34_EVENTS_REPRODUCED.csv.gz',index=False,compression='gzip');DE.to_csv(a.outdir/'QV2_34_DECISIONS_REPRODUCED.csv.gz',index=False,compression='gzip');P.to_csv(a.outdir/'QV2_34_PROMOTED_REPRODUCED.csv.gz',index=False,compression='gzip');print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__':main()
