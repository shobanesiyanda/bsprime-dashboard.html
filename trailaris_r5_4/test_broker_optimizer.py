#!/usr/bin/env python3
import pandas as pd
from broker_optimizer import optimize,ROUTES

def row(broker,route='EURUSD',cost=1.0,tail=2.0,cert=True,lot=.01,minlot=.01,step=.01,sample=5000):
    return dict(broker_id=broker,route=route,available=True,sample_n=sample,trade_lot=lot,spread_cost_median_usd=cost,spread_cost_p95_usd=tail,commission_round_turn_usd=0.0,slippage_mean_usd=0.0,slippage_p95_usd=0.0,financing_long_usd=0.0,financing_short_usd=0.0,contract_size=100000,tick_size=.00001,tick_value=1,min_lot=minlot,lot_step=step,margin_required_usd=1,fill_rate=.999,reject_rate=.001,latency_median_ms=20,latency_p95_ms=40,spec_hash='abc',captured_from_target_server=True,broker_certified=cert)

def test_cheapest_wins():
    _,w,_=optimize(pd.DataFrame([row('A',cost=2,tail=3),row('B',cost=1,tail=2)]));x=w[w.route=='EURUSD'].iloc[0];assert x.primary_broker=='B'

def test_uncertified_excluded():
    r,w,_=optimize(pd.DataFrame([row('A',cost=.1,cert=False),row('B',cost=1,cert=True)]));assert w[w.route=='EURUSD'].iloc[0].primary_broker=='B';assert r[r.broker_id=='A'].iloc[0].eligibility_reason=='NOT_BROKER_CERTIFIED'

def test_lot_infeasible_excluded():
    r,_,_=optimize(pd.DataFrame([row('A',lot=.01,minlot=.1),row('B')]));assert r[r.broker_id=='A'].iloc[0].eligibility_reason=='BASELINE_TRADE_BELOW_MIN_LOT'

def test_full_gate_fails_without_34_certified_routes():
    _,_,s=optimize(pd.DataFrame([row('A')]));assert s['routes_required']==34 and not s['broker_optimization_gate_pass']

def test_ib_never_scores():
    _,_,s=optimize(pd.DataFrame([row('A'),row('B',cost=2)]));assert s['ib_revenue_used_in_score'] is False

if __name__=='__main__':
    for f in [test_cheapest_wins,test_uncertified_excluded,test_lot_infeasible_excluded,test_full_gate_fails_without_34_certified_routes,test_ib_never_scores]:f()
    print('5/5 PASS')
