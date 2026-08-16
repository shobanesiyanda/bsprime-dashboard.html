#!/usr/bin/env python3
BASE={
 'end_equity':254.4465598357797,
 'compounded_return_pct':154.4465598357797,
 'mean_week_pct':9.004861168026705,
 'median_week_pct':10.760554729262605,
 'weeks_ge5':8,
 'weeks_ge10':6,
 'max_weekly_dd_pct':-7.083721208160998,
 'fresh_return_pct':7.826996335713843,
 'fresh_max_dd_pct':-0.9205380450303946,
 'assets_with_selected':33,
 'losses':374,
 'nonflat_win_rate':0.5930359085963003,
 'profit_factor':1.9080991024100367
}

def evaluate(m):
    checks={
      'end_equity':float(m['end_equity'])>=BASE['end_equity']-1e-9,
      'mean_week_pct':float(m['mean_week_pct'])>=BASE['mean_week_pct']-1e-9,
      'median_week_pct':float(m['median_week_pct'])>=BASE['median_week_pct']-1e-9,
      'weeks_ge5':int(m['weeks_ge5'])>=BASE['weeks_ge5'],
      'weeks_ge10':int(m['weeks_ge10'])>=BASE['weeks_ge10'],
      'max_weekly_dd':float(m['max_weekly_dd_pct'])>=BASE['max_weekly_dd_pct']-1e-9,
      'fresh_return':float(m['fresh_return_pct'])>=BASE['fresh_return_pct']-1e-9,
      'fresh_dd':float(m.get('fresh_max_dd_pct',BASE['fresh_max_dd_pct']))>=BASE['fresh_max_dd_pct']-1e-9,
      'asset_participation':int(m['assets_with_selected'])>=BASE['assets_with_selected']
    }
    improvements={
      'fewer_losses':int(m.get('losses',BASE['losses']))<BASE['losses'],
      'higher_nonflat_win_rate':float(m.get('nonflat_win_rate',BASE['nonflat_win_rate']))>BASE['nonflat_win_rate']+1e-12,
      'higher_profit_factor':float(m.get('profit_factor',BASE['profit_factor']))>BASE['profit_factor']+1e-12,
      'higher_end_equity':float(m['end_equity'])>BASE['end_equity']+1e-9,
      'better_max_weekly_dd':float(m['max_weekly_dd_pct'])>BASE['max_weekly_dd_pct']+1e-9
    }
    return {'checks':checks,'improvements':improvements,'behavioral_non_regression_pass':all(checks.values()),'material_improvement_present':any(improvements.values()),'promotion_allowed':all(checks.values()) and any(improvements.values())}

if __name__=='__main__':
    import json,sys
    d=json.load(open(sys.argv[1]));print(json.dumps(evaluate(d),indent=2))
