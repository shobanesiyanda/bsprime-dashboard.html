#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import R5_BASE_CAUSAL_ENGINE as base
import trailaris_r4_full_universe_loop as r4
from precision_experiment_harness import eval_variant
from non_regression_gate import evaluate,BASE

# Pre-specified management variants for MEAN_REVERSION_EXTREME only.
# Entry signal/direction remain untouched; only post-entry management changes.
VARIANTS={
 'baseline':{},
 'mre_early_partial':{'p1_r':.60,'p1_frac':.35,'p2_r':1.10,'p2_frac':.25,'p1_floor':0.0,'p2_floor':.35,'trail_start':1.60,'trail_gap':.65,'max_bars':30},
 'mre_profit_lock':{'p1_r':.75,'p1_frac':.30,'p2_r':1.25,'p2_frac':.25,'p1_floor':.15,'p2_floor':.50,'trail_start':1.60,'trail_gap':.60,'max_bars':30},
 'mre_short_horizon':{'p1_r':.75,'p1_frac':.25,'p2_r':1.25,'p2_frac':.25,'p1_floor':0.0,'p2_floor':.40,'trail_start':1.75,'trail_gap':.70,'max_bars':18},
 'mre_combined':{'p1_r':.60,'p1_frac':.35,'p2_r':1.10,'p2_frac':.25,'p1_floor':.10,'p2_floor':.45,'trail_start':1.50,'trail_gap':.60,'max_bars':18},
}

def simulate(x,i,direction,cfg):
    if i>=len(x)-2:return None
    ent_i=i+1;ent=float(x.open.iloc[ent_i]);a=float(x.atr.iloc[i]);
    if not np.isfinite(a) or a<=0:return None
    sd=max(a*1.15,float(x.spread.iloc[i])*4,1e-12);stop=ent-direction*sd;target=ent+direction*3.0*sd
    remaining=1.0;real=0.;floor=-1.;mfe=0.;mae=0.;addon=None;reason='TIME';limit=min(len(x),ent_i+int(cfg['max_bars'])+1);exit_idx=limit-1;exit_time=x.timestamp.iloc[exit_idx];exit_px=float(x.close.iloc[exit_idx]);p1=False;p2=False
    for j in range(ent_i,limit):
        b=x.iloc[j];fav=(float(b.high)-ent) if direction==1 else (ent-float(b.low));adv=(ent-float(b.low)) if direction==1 else (float(b.high)-ent);mfe=max(mfe,fav/sd);mae=max(mae,adv/sd)
        if not p1 and mfe>=cfg['p1_r']:
            real+=cfg['p1_frac']*cfg['p1_r'];remaining-=cfg['p1_frac'];floor=max(floor,cfg['p1_floor']);p1=True
        if not p2 and mfe>=cfg['p2_r']:
            real+=cfg['p2_frac']*cfg['p2_r'];remaining-=cfg['p2_frac'];floor=max(floor,cfg['p2_floor']);p2=True
        if mfe>=cfg['trail_start']:floor=max(floor,mfe-cfg['trail_gap'])
        if addon is None and mfe>=.80 and floor>=0 and j>=ent_i+4:
            prev=x.iloc[max(ent_i,j-5):j]
            if direction==1 and float(b.close)>float(prev.high.max()) and b.body_frac>=.45:addon=j
            if direction==-1 and float(b.close)<float(prev.low.min()) and b.body_frac>=.45:addon=j
        eff=ent+direction*floor*sd if floor>-1 else stop
        stopped=(float(b.low)<=eff if direction==1 else float(b.high)>=eff);targeted=(float(b.high)>=target if direction==1 else float(b.low)<=target)
        if stopped:
            rr=(eff-ent)*direction/sd;real+=remaining*rr;remaining=0.;exit_px=eff;exit_time=b.timestamp;reason='PROTECTED_STOP' if floor>=0 else 'STOP';break
        if targeted:
            real+=remaining*3.0;remaining=0.;exit_px=target;exit_time=b.timestamp;reason='TARGET';break
    if remaining>0:
        rr=(exit_px-ent)*direction/sd;real+=remaining*rr
    cost=max(0,float(x.spread.iloc[i])/sd);net=real-cost;give=max(0,mfe-max(0,real))
    return dict(entry_time=x.timestamp.iloc[ent_i],decision_time=x.timestamp.iloc[i],exit_time=exit_time,direction=int(direction),entry_price=ent,exit_price=float(exit_px),stop_distance=sd,gross_r=float(real),cost_r=float(cost),net_r=float(net),mfe_r=float(mfe),mae_r=float(mae),giveback_r=float(give),exit_reason=reason,addon_trigger_index=addon)

def transform(cands,fcache,cfg):
    if not cfg:return cands.copy()
    z=cands.copy();mask=z.strategy.eq('MEAN_REVERSION_EXTREME');idxs=z.index[mask].tolist()
    for ix in idxs:
        row=z.loc[ix];x=fcache[row.asset];i=r4.nearest_feature_index(x,pd.Timestamp(row.decision_time));o=simulate(x,i,int(row.direction),cfg)
        if not o:continue
        for k,v in o.items():z.at[ix,k]=v
    return z

def payoff_stats(ev):
    g=ev[ev.strategy.eq('MEAN_REVERSION_EXTREME')] if len(ev) else pd.DataFrame()
    if g.empty:return {}
    w=g[g.net_pnl>0];l=g[g.net_pnl<0]
    return {'trades':len(g),'wins':len(w),'losses':len(l),'net_pnl':float(g.net_pnl.sum()),'nonflat_win_rate':float(len(w)/(len(w)+len(l))) if len(w)+len(l) else 0.,'avg_win_usd':float(w.net_pnl.mean()) if len(w) else 0.,'avg_loss_usd':float(l.net_pnl.mean()) if len(l) else 0.,'payoff_ratio':float(w.net_pnl.mean()/abs(l.net_pnl.mean())) if len(w) and len(l) and l.net_pnl.mean()!=0 else None,'avg_giveback_r':float(g.giveback_r.mean())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy');rows=[]
    for name,cfg in VARIANTS.items():
        cz=transform(cands,fcache,cfg);res,W,EV,DE,PR=eval_variant(base,data,cz,opps,fcache,specs);res['variant']=name;res['mre']=payoff_stats(EV);strict=evaluate({'end_equity':res['end_equity'],'mean_week_pct':res['mean_week_pct'],'median_week_pct':res['median_week_pct'],'weeks_ge5':res['weeks_ge5'],'weeks_ge10':res['weeks_ge10'],'max_weekly_dd_pct':res['max_weekly_dd_pct'],'fresh_return_pct':res['fresh_return_pct'],'fresh_max_dd_pct':res['fresh_max_dd_pct'],'assets_with_selected':res['assets_with_selected'],'losses':res['losses'],'nonflat_win_rate':res['nonflat_win_rate'],'profit_factor':BASE['profit_factor']});res['strict_nonregression']=strict['behavioral_non_regression_pass'];res['promotion_allowed']=strict['promotion_allowed'];rows.append(res);W.to_csv(out/f'{name}_weekly.csv',index=False)
    flat=[]
    for r in rows:
        q={k:v for k,v in r.items() if k!='mre'};q.update({f'mre_{k}':v for k,v in r['mre'].items()});flat.append(q)
    df=pd.DataFrame(flat).sort_values(['promotion_allowed','end_equity'],ascending=[False,False]);df.to_csv(out/'R5_4_MRE_MANAGEMENT_EXPERIMENTS.csv',index=False);s={'variants_tested':len(df),'promotion_candidates':int(df.promotion_allowed.sum()),'best':df.iloc[0].to_dict()};(out/'R5_4_MRE_MANAGEMENT_STATUS.json').write_text(json.dumps(s,indent=2,default=str));print(json.dumps(s,indent=2,default=str));b=df[df.variant=='baseline'].iloc[0];assert abs(float(b.end_equity)-BASE['end_equity'])<=1e-9

if __name__=='__main__':main()
