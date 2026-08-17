#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import requests

META_URL = "https://raw.githubusercontent.com/Leo4815162342/dukascopy-node/refs/heads/master/src/utils/instrument-meta-data/generated/instrument-meta-data.json"
START = "2026-01-19T00:00:00Z"

ALIASES = {
    "USA30.IDX/USD": "US30",
    "USA500.IDX/USD": "US500",
    "USATECH.IDX/USD": "US100",
    "DEU.IDX/EUR": "GER40",
    "GBR.IDX/GBP": "UK100",
    "JPN.IDX/JPY": "JP225",
    "LIGHT.CMD/USD": "WTI",
    "BRENT.CMD/USD": "BRENT",
    "GAS.CMD/USD": "NATGAS",
    "XAU/USD": "XAUUSD",
    "XAG/USD": "XAGUSD",
}


def canonical(name: str, instrument_id: str) -> str:
    n = (name or "").upper().strip()
    if n in ALIASES:
        return ALIASES[n]
    if re.fullmatch(r"[A-Z]{3}/[A-Z]{3}", n):
        return n.replace("/", "")
    m = re.fullmatch(r"([A-Z0-9._-]+)\.([A-Z]{2})/([A-Z]{3})", n)
    if m:
        # preserve venue/country to avoid ticker collisions across exchanges
        return f"{m.group(1)}.{m.group(2)}"
    m = re.fullmatch(r"([A-Z0-9._-]+)\.IDX/([A-Z]{3})", n)
    if m:
        return f"{m.group(1)}.IDX"
    m = re.fullmatch(r"([A-Z0-9._-]+)\.CMD/([A-Z]{3})", n)
    if m:
        return f"{m.group(1)}.CMD"
    x = re.sub(r"[^A-Z0-9._-]", "", n)
    return x or instrument_id.upper()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    r = requests.get(META_URL, timeout=90)
    r.raise_for_status()
    meta = r.json()

    import trailaris_r4_full_universe_loop as r4

    fields = [
        "instrument_id", "canonical_instrument", "provider_name", "provider_code",
        "description", "minute_history_start", "has_full_26w_window", "factor_compatible",
        "factor_reason"
    ]
    rows = []
    compatible = []
    for instrument_id, m in sorted(meta.items()):
        name = str(m.get("name", ""))
        asset = canonical(name, instrument_id)
        start = str(m.get("startDayForMinuteCandles", ""))
        has_window = bool(start and start <= START)
        ok = False
        reason = "NO_26W_M1_HISTORY"
        if has_window:
            try:
                a = np.asarray(r4.factor_vec(asset, 1), dtype=float)
                b = np.asarray(r4.factor_vec(asset, -1), dtype=float)
                if a.size == len(r4.FACTORS) and b.size == len(r4.FACTORS) and np.isfinite(a).all() and np.isfinite(b).all():
                    ok = True
                    reason = "PASS"
                else:
                    reason = "FACTOR_VECTOR_CONTRACT_INVALID"
            except Exception as e:
                reason = f"FACTOR_CONTRACT_REJECT:{type(e).__name__}:{str(e)[:180]}"
        row = {
            "instrument_id": instrument_id,
            "canonical_instrument": asset,
            "provider_name": name,
            "provider_code": str(m.get("code", "")),
            "description": str(m.get("description", "")),
            "minute_history_start": start,
            "has_full_26w_window": str(has_window).lower(),
            "factor_compatible": str(ok).lower(),
            "factor_reason": reason,
        }
        rows.append(row)
        if ok:
            compatible.append(row)

    with (args.outdir / "DISCOVERED_UNIVERSE.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with (args.outdir / "FACTOR_COMPATIBLE_UNIVERSE.tsv").open("w", newline="", encoding="utf-8") as f:
        f.write("instrument_id\tcanonical_instrument\tprovider_name\n")
        for x in compatible:
            f.write(f"{x['instrument_id']}\t{x['canonical_instrument']}\t{x['provider_name']}\n")

    summary = {
        "state": "DISCOVERED",
        "provider_metadata_instruments": len(rows),
        "with_full_26w_m1_window": sum(x["has_full_26w_window"] == "true" for x in rows),
        "frozen_factor_contract_compatible": len(compatible),
        "window_start": START,
        "window_end_exclusive": "2026-08-15T00:00:00Z",
        "outcome_information_used": False,
        "quant_v2_modified": False,
    }
    (args.outdir / "DISCOVERY_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
