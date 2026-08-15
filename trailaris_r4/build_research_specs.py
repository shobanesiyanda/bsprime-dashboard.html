from pathlib import Path
import pandas as pd
ROUTES=['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
FX=set(ROUTES[:13]);MET={'XAUUSD','XAGUSD'};CR={'BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD'};IDX={'US100','US500','US30','GER40','UK100','JP225'};ENE={'WTI','BRENT','NATGAS'}
rows=[]
for a in ROUTES:
 if a in FX:cs=100000;tick=.001 if 'JPY' in a else .00001;tv=1.;vmin=.001;step=.001;margin=.30
 elif a in MET:cs=100;tick=.01;tv=1.;vmin=.01;step=.01;margin=5.
 elif a in CR:cs=1;tick=.01;tv=.01;vmin=.01;step=.01;margin=2.
 elif a in IDX:cs=1;tick=.1;tv=.1;vmin=.01;step=.01;margin=2.
 elif a in ENE:cs=100;tick=.01;tv=1.;vmin=.01;step=.01;margin=4.
 else:cs=1;tick=.01;tv=.01;vmin=.001;step=.001;margin=.5
 rows.append(dict(asset=a,contract_size=cs,tick_size=tick,tick_value=tv,volume_min=vmin,volume_step=step,volume_max=100,margin_per_lot=margin,evidence_state='RESEARCH_PROXY_SIZING_NOT_BROKER_CERTIFIED'))
p=Path('trailaris_r4/research_proxy_specs.csv');p.parent.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False)
