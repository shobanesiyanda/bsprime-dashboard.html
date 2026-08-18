from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np
import pandas as pd

ORIGINAL_ROUTES = [
    'EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF',
    'XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS',
    'V75','V50','V100','V25','V10'
]
FX_ROUTES = ORIGINAL_ROUTES[:13]
CRYPTO_ROUTES = ['BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD']
CURRENCIES = {
    'USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','CNH','CNY','HKD','SGD','NOK','SEK','DKK','PLN','CZK','HUF','RON','TRY','ZAR','MXN','BRL','ILS','AED','SAR','THB'
}

# Runtime-only adapter lineage. This does not change Quant V2 strategy logic;
# it lets R4 applicability/session helpers resolve an expanded asset through
# the same canonical 34-route anchor already used by the universe adapter.
_ADAPTER_ANCHORS: dict[str, str] = {}


def _tokens(asset: str, provider_name: str = '') -> tuple[str, str]:
    a = str(asset).upper().replace('-', '').replace('_', '')
    p = str(provider_name).upper()
    return a, p


def classify(asset: str, provider_name: str = '') -> str:
    a, p = _tokens(asset, provider_name)
    if a in ORIGINAL_ROUTES:
        if a in FX_ROUTES: return 'FX'
        if a in {'XAUUSD','XAGUSD'}: return 'METAL'
        if a in CRYPTO_ROUTES: return 'CRYPTO'
        if a in {'US100','US500','US30','GER40','UK100','JP225'}: return 'INDEX'
        if a in {'WTI','BRENT','NATGAS'}: return 'COMMODITY'
        return 'SYNTHETIC'
    if '.IDX' in a or '.IDX/' in p: return 'INDEX'
    if '.CMD' in a or '.CMD/' in p: return 'COMMODITY'
    if re.fullmatch(r'[A-Z]{6}', a) and a[:3] in CURRENCIES and a[3:] in CURRENCIES: return 'FX'
    if a.startswith(('XAU','XAG','XPD','XPT')): return 'METAL'
    if a.startswith(('BTC','ETH','SOL','BNB','XRP','BCH','LTC','DOGE','ADA','DOT','AVAX','LINK')): return 'CRYPTO'
    if re.search(r'\.[A-Z]{2,3}$', str(asset).upper()): return 'EQUITY'
    return 'OTHER'


def anchor_for(asset: str, provider_name: str = '') -> str:
    a, p = _tokens(asset, provider_name)
    if a in ORIGINAL_ROUTES: return a
    kind = classify(asset, provider_name)
    if kind == 'FX':
        base, quote = a[:3], a[3:]
        scored=[]
        for r in FX_ROUTES:
            rb,rq=r[:3],r[3:]
            score=4*(rb==base and rq==quote)+2*(rb==base)+2*(rq==quote)+(rq==base)+(rb==quote)
            scored.append((score,r))
        return max(scored)[1]
    if kind == 'METAL': return 'XAGUSD' if a.startswith('XAG') else 'XAUUSD'
    if kind == 'CRYPTO':
        for r in CRYPTO_ROUTES:
            if a.startswith(r[:3]): return r
        return 'BTCUSD'
    if kind in {'INDEX','EQUITY'}:
        u=(a+' '+p)
        if any(x in u for x in ['.JP','JPY','JPN.IDX']): return 'JP225'
        if any(x in u for x in ['.GB','GBP','GBR.IDX']): return 'UK100'
        if any(x in u for x in ['.DE','EUR','DEU.IDX','FRA.IDX','ESP.IDX','ITA.IDX','EUS.IDX']): return 'GER40'
        return 'US500'
    if kind == 'COMMODITY':
        u=a+' '+p
        if 'GAS' in u: return 'NATGAS'
        if 'BRENT' in u or 'DIESEL' in u: return 'BRENT'
        if a.startswith(('XAU','XAG','XPD','XPT')): return 'XAUUSD'
        return 'WTI'
    return 'US500'


def _copy_anchor_keys(module: Any, asset: str, anchor: str) -> None:
    for _, obj in list(vars(module).items()):
        if isinstance(obj, dict) and anchor in obj and asset not in obj:
            try: obj[asset]=copy.deepcopy(obj[anchor])
            except Exception: pass


def _ensure_r4_asset_class_fallback() -> None:
    import trailaris_r4_asset_adapter as r4a
    if getattr(r4a, '_qv2_anchor_asset_class_fallback', False):
        return
    original = r4a.asset_class
    def extended_asset_class(asset):
        a=str(asset)
        try:
            return original(a)
        except KeyError:
            anchor=_ADAPTER_ANCHORS.get(a)
            if anchor is None:
                raise
            return original(anchor)
    r4a.asset_class=extended_asset_class
    r4a._qv2_anchor_asset_class_fallback=True


def install_asset(r4: Any, asset: str, provider_name: str = '') -> str:
    asset=str(asset); anchor=anchor_for(asset,provider_name)
    if asset in ORIGINAL_ROUTES: return anchor
    _ADAPTER_ANCHORS[asset]=anchor
    _copy_anchor_keys(r4,asset,anchor)
    routes=getattr(r4,'ROUTES',None)
    if isinstance(routes,list) and asset not in routes: routes.append(asset)
    elif isinstance(routes,tuple) and asset not in routes: setattr(r4,'ROUTES',routes+(asset,))
    try:
        import trailaris_r4_asset_adapter as r4a
        _copy_anchor_keys(r4a,asset,anchor)
        _ensure_r4_asset_class_fallback()
    except Exception:
        pass
    return anchor


def install_universe(r4: Any, metadata: pd.DataFrame) -> dict[str,str]:
    mapping={}
    if metadata is None or len(metadata)==0: return mapping
    for _,r in metadata.drop_duplicates('asset').iterrows():
        asset=str(r['asset']); provider=str(r.get('discovery_provider_name',''))
        mapping[asset]=install_asset(r4,asset,provider)
    original_factor_vec=getattr(r4,'factor_vec')
    def factor_vec(asset: str, direction: int):
        a=str(asset)
        if a in ORIGINAL_ROUTES:
            return original_factor_vec(a,direction)
        provider=''
        z=metadata.loc[metadata.asset.astype(str).eq(a)]
        if len(z): provider=str(z.iloc[0].get('discovery_provider_name',''))
        anchor=mapping.get(a) or install_asset(r4,a,provider)
        try: return original_factor_vec(a,direction)
        except Exception: return np.asarray(original_factor_vec(anchor,direction),dtype=float)
    r4.factor_vec=factor_vec
    return mapping


def extend_research_specs(base_specs: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    out=base_specs.copy()
    have=set(out.asset.astype(str)) if len(out) else set()
    rows=[]
    for _,r in metadata.drop_duplicates('asset').iterrows():
        a=str(r['asset'])
        if a in have: continue
        provider=str(r.get('discovery_provider_name','')); kind=classify(a,provider)
        if kind=='FX': cs=100000; tick=.001 if a.endswith('JPY') else .00001; tv=1.; vmin=.001; step=.001; margin=.30
        elif kind=='METAL': cs=100; tick=.01; tv=1.; vmin=.01; step=.01; margin=5.
        elif kind=='CRYPTO': cs=1; tick=.01; tv=.01; vmin=.01; step=.01; margin=2.
        elif kind=='INDEX': cs=1; tick=.1; tv=.1; vmin=.01; step=.01; margin=2.
        elif kind=='COMMODITY': cs=100; tick=.01; tv=1.; vmin=.01; step=.01; margin=4.
        else: cs=1; tick=.01; tv=.01; vmin=.001; step=.001; margin=.5
        rows.append(dict(asset=a,contract_size=cs,tick_size=tick,tick_value=tv,volume_min=vmin,volume_step=step,volume_max=100,margin_per_lot=margin,evidence_state='R4_RESEARCH_PROXY_RULE_EXTENDED_ASSET_UNIVERSE'))
    return pd.concat([out,pd.DataFrame(rows)],ignore_index=True) if rows else out
