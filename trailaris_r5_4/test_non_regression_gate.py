from non_regression_gate import BASE,evaluate

def exact():
 d=dict(BASE);d['fresh_max_dd_pct']=BASE['fresh_max_dd_pct'];return d

def test_exact_not_promotion():
 r=evaluate(exact());assert r['behavioral_non_regression_pass'] and not r['material_improvement_present'] and not r['promotion_allowed']

def test_improved_losses_promotes():
 d=exact();d['losses']=373;r=evaluate(d);assert r['promotion_allowed']

def test_more_return_but_bad_median_rejected():
 d=exact();d['end_equity']=300;d['median_week_pct']=8;r=evaluate(d);assert not r['promotion_allowed']

def test_more_return_but_fewer_assets_rejected():
 d=exact();d['end_equity']=300;d['assets_with_selected']=32;r=evaluate(d);assert not r['promotion_allowed']

def test_better_losses_but_worse_dd_rejected():
 d=exact();d['losses']=300;d['max_weekly_dd_pct']=-8;r=evaluate(d);assert not r['promotion_allowed']

if __name__=='__main__':
 for f in [test_exact_not_promotion,test_improved_losses_promotes,test_more_return_but_bad_median_rejected,test_more_return_but_fewer_assets_rejected,test_better_losses_but_worse_dd_rejected]:f()
 print('5/5 PASS')
