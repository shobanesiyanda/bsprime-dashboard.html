#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from lane1b_full_replay import ManagedVariant,apply_rule
from winner_causal_features import causal_decision_features

R54_END=337.4231242104393
INC_END=343.5447309255116
INC_FRESH=7.261646557142409
INC_DD=-4.637645574907678
INC_NFWR=0.6671428571428571
CHALLENGE_END=345.3896887826546
CHALLENGE_NFWR=0.673352435530086
CHALLENGE_LOSSES=228
CHALLENGE_FLATS=421
EPS=1e-9
B3={'behavior':'close_a0.50_r0.20_1b','kind':'close','activate_mfe_r':0.50,'parameter_r':0.20,'bars':1,'gate_metric':'reliability_win','gate_direction':'le','gate_threshold':0.2655132009803008}

def num(x,d=np.nan):
    try:v=float(x);return v if np.isfinite(v) else d
    except Exception:return d

def rcond(z,f,d,th):
    if f not in z.columns:return pd.Series(False,index=z.index)
    s=pd.to_numeric(z[f],errors='coerce');return s.le(th) if d=='le' else s.ge(th)

def rule_mask(z,r):
    if str(r['scope_type'])=='GLOBAL':m=pd.Series(True,index=z.index)
    elif str(r['scope_type'])=='ASSET_CLASS':m=z.asset_class.eq(str(r['scope_value']))
    else:m=z.strategy_family.eq(str(r['scope_value']))
    m &= rcond(z,str(r['feature1']),str(r['dir1']),num(r['threshold1']))
    f2=str(r.get('feature2','') or '')
    if f2 and f2.lower()!='nan':m &= rcond(z,f2,str(r['dir2']),num(r['threshold2']))
    return m.fillna(False)

def extend_one(row,frame,runner_frac,target_r,trail_gap,max_extra_bars):
    if frame is None or len(frame)==0:return None
    t=pd.to_datetime(frame.timestamp,utc=True);start=int(t.searchsorted(pd.Timestamp(row.exit_time),side='right'))
    if start>=len(frame):return None
    ent=num(row.entry_price);sd=max(abs(num(row.stop_distance)),1e-12);dr=int(num(row.direction,1));floor_r=2.0;mfe=3.0;rr=3.0;xt=pd.Timestamp(row.exit_time);px=num(row.exit_price)
    for j in range(start,min(len(frame),start+int(max_extra_bars))):
        b=frame.iloc[j];hi=num(b.high);lo=num(b.low);cl=num(b.close);op=num(b.open,cl);ts=pd.Timestamp(b.timestamp)
        stop=ent+dr*floor_r*sd;stopped=(lo<=stop+EPS if dr==1 else hi>=stop-EPS)
        if stopped:
            px=op if (dr==1 and op<stop) or (dr==-1 and op>stop) else stop;rr=(px-ent)*dr/sd;xt=ts;break
        fav=(hi-ent)/sd if dr==1 else (ent-lo)/sd;mfe=max(mfe,fav)
        targeted=(hi>=ent+target_r*sd-EPS if dr==1 else lo<=ent-target_r*sd+EPS)
        if targeted:rr=float(target_r);px=ent+dr*target_r*sd;xt=ts;break
        floor_r=max(floor_r,mfe-float(trail_gap));rr=(cl-ent)*dr/sd;px=cl;xt=ts
    gross=(1-float(runner_frac))*3.0+float(runner_frac)*rr;net=gross-max(0.,num(row.cost_r,0.));orig=num(row.net_r)
    return {'gross_r':gross,'net_r':net,'exit_time':xt,'exit_price':px,'mfe_r':max(num(row.get('mfe_r'),3.0),mfe),'delta_r':net-orig,'runner_r':rr}

def extend_candidates(cands,fcache,runner_frac,target_r,trail_gap,max_extra_bars):
    z=cands.copy();extended=0;delta=0.
    if 'winner_extend_qualifies' not in z:return z,extended,delta
    for i,row in z.iterrows():
        if not bool(row.get('winner_extend_qualifies',False)) or bool(row.get('is_addon',False)):continue
        if str(row.get('exit_reason',''))!='TARGET' or pd.notna(row.get('management_rule',np.nan)):continue
        hit=extend_one(row,fcache.get(str(row.asset)) if fcache else None,runner_frac,target_r,trail_gap,max_extra_bars)
        if hit is None:continue
        for k in ['gross_r','net_r','exit_time','exit_price','mfe_r']:z.at[i,k]=hit[k]
        z.at[i,'winner_extension']=f'runner{runner_frac}_target{target_r}_trail{trail_gap}';z.at[i,'winner_extension_original_net_r']=num(row.net_r);z.at[i,'winner_extension_delta_r']=hit['delta_r'];extended+=1;delta+=hit['delta_r']
    return z,extended,delta

class WinnerExtensionVariant:
    def __init__(self,base,promote,runner_frac,target_r,trail_gap,max_extra):self.base=base;self.promote=promote;self.runner_frac=runner_frac;self.target_r=target_r;self.trail_gap=trail_gap;self.max_extra=max_extra;self.extended=0;self.delta_r=0.
    def promote_for_week(self,cands,week_start):return self.promote(cands,week_start,'combined')
    def replay_r5(self,cands,specs,start=100.,feature_cache=None):
        managed=apply_rule(cands,feature_cache,B3);z,n,d=extend_candidates(managed,feature_cache,self.runner_frac,self.target_r,self.trail_gap,self.max_extra);self.extended=n;self.delta_r=d;return self.base.replay_r5(z,specs,start,feature_cache)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rawdir',required=True);ap.add_argument('--specs',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--rule-rank',type=int,required=True);ap.add_argument('--runner-frac',type=float,required=True);ap.add_argument('--target-r',type=float,required=True);ap.add_argument('--trail-gap',type=float,default=.8);ap.add_argument('--max-extra-bars',type=int,default=36);ap.add_argument('--outdir',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    R=pd.read_csv(a.rules);R=R[R.clean_pass.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True)
    if a.rule_rank>=len(R):
        s={'state':'R5_5_LANE6C_NOT_RUN_RULE_RANK_UNAVAILABLE','rule_rank':a.rule_rank,'passing_rules':len(R)};(out/'result.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));return
    rule=R.iloc[a.rule_rank].to_dict()
    import R5_BASE_CAUSAL_ENGINE as base
    import trailaris_r4_full_universe_loop as r4
    from reliability_common import promote
    from precision_experiment_harness import eval_variant,load_module
    data,v,cands,opps,fcache=base.build_universe(Path(a.rawdir));specs=r4.load_specs(Path(a.specs),'research-proxy');scope=json.load(open('trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json'))['scope']
    ctlmod=load_module(Path('trailaris_r5_4/reliability_variants/variant_hierarchical_combined.py'),f'r55_lane6c_ctl_{a.rule_rank}_{a.runner_frac}_{a.target_r}');ctl,*_=eval_variant(ctlmod,data,cands,opps,fcache,specs,100.0)
    if abs(float(ctl['end_equity'])-R54_END)>1e-9:raise RuntimeError('R5.4 control reproduction failed')
    inc,*_=eval_variant(ManagedVariant(B3,base,promote),data,cands,opps,fcache,specs,100.0)
    if abs(float(inc['end_equity'])-INC_END)>1e-6:raise RuntimeError('Lane1B2 incumbent reproduction failed')
    z=causal_decision_features(cands,fcache);z['winner_extend_qualifies']=rule_mask(z,rule).to_numpy();var=WinnerExtensionVariant(base,promote,a.runner_frac,a.target_r,a.trail_gap,a.max_extra_bars);cand,VW,VEV,VDE,VPR=eval_variant(var,data,z,opps,fcache,specs,100.0)
    selected_ext=int(VEV.get('winner_extension',pd.Series(index=VEV.index,dtype=object)).notna().sum()) if len(VEV) else 0;scope_ok=(len(data)==34 and scope=={'approved_routes':34,'strategy_families':15,'route_strategy_cells':510})
    alpha={'end_equity_beats_1b2':float(cand['end_equity'])>INC_END+EPS,'fresh_not_worse':float(cand['fresh_return_pct'])>=INC_FRESH-EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=INC_DD-EPS,'asset_coverage_not_worse':int(cand['assets_with_selected'])>=33,'full_universe_34x15x510':scope_ok}
    frontier={'end_equity_beats_lane5':float(cand['end_equity'])>CHALLENGE_END+EPS,'fresh_improved':float(cand['fresh_return_pct'])>INC_FRESH+EPS,'drawdown_not_worse':float(cand['max_weekly_dd_pct'])>=INC_DD-EPS,'fresh_dd_not_worse':float(cand['fresh_max_dd_pct'])>=-0.5196525640746019-EPS,'losses_at_or_below_lane5':int(cand['losses'])<=CHALLENGE_LOSSES,'flats_at_or_below_lane5':int(cand['flat'])<=CHALLENGE_FLATS,'nonflat_win_rate_at_least_lane5':float(cand['nonflat_win_rate'])>=CHALLENGE_NFWR-EPS,'asset_coverage_not_worse':int(cand['assets_with_selected'])>=33,'full_universe_34x15x510':scope_ok}
    s={'state':'R5_5_LANE6C_WINNER_PAYOFF_EXTENSION_COMPLETE','evidence_class':'END_TO_END_CAUSAL_FULL_REPLAY','rule_rank':a.rule_rank,'winner_rule':rule,'runner_frac':a.runner_frac,'target_r':a.target_r,'trail_gap_r':a.trail_gap,'max_extra_bars':a.max_extra_bars,'qualifying_candidates':int(z.winner_extend_qualifies.sum()),'extended_candidates_pre_replay':var.extended,'selected_extended_executions':selected_ext,'candidate_delta_r_before_portfolio_interaction':var.delta_r,'r5_4_control':ctl,'lane1b2_incumbent':inc,'candidate':cand,'delta_vs_1b2':float(cand['end_equity'])-INC_END,'delta_vs_lane5_frontier':float(cand['end_equity'])-CHALLENGE_END,'alpha_gates':alpha,'alpha_candidate':bool(all(alpha.values())),'frontier_gates':frontier,'frontier_candidate':bool(all(frontier.values())),'promotion_candidate':False,'scope':scope,'governance':'Winner cohort qualification is reconstructed at each decision week using only prior exits. Extension activates only after an unmodified selected base trade has causally reached 3R; initial risk is unchanged and full portfolio opportunity displacement is replayed.'}
    (out/'result.json').write_text(json.dumps(s,indent=2,default=str));VW.to_csv(out/'weekly.csv',index=False);print(json.dumps(s,indent=2,default=str))
if __name__=='__main__':main()
