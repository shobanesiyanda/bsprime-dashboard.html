#!/usr/bin/env python3
"""Export the frozen 26-week replay window from one exact logged-in MT5 account.

Run on Windows where MetaTrader 5 and the Python MetaTrader5 package are available.
The script is read-only: it requests symbol metadata/history and places no orders.

Frozen comparison window:
  2026-01-19T00:00:00Z <= bar time < 2026-08-15T00:00:00Z
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

START = datetime(2026, 1, 19, 0, 0, tzinfo=timezone.utc)
END_EXCLUSIVE = datetime(2026, 8, 15, 0, 0, tzinfo=timezone.utc)

REGISTRY_FIELDS = [
    "canonical_instrument","asset_class","broker_legal_entity","jurisdiction","server",
    "account_type","data_source","data_start_utc","data_end_utc","bar_interval","rows",
    "coverage_state","ohlc_state","duplicate_state","causality_state","data_hash","evidence_ref"
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"coverage_state":"FAIL","ohlc_state":"FAIL","duplicate_state":"FAIL","causality_state":"FAIL"}
    dup = int(df["timestamp"].duplicated().sum())
    bad_ohlc = int(((df["high"] < df[["open","close","low"]].max(axis=1)) |
                    (df["low"] > df[["open","close","high"]].min(axis=1)) |
                    (df[["open","high","low","close"]].le(0).any(axis=1))).sum())
    ordered = bool(df["timestamp"].is_monotonic_increasing)
    start_ok = pd.Timestamp(df["timestamp"].min()) >= pd.Timestamp(START)
    end_ok = pd.Timestamp(df["timestamp"].max()) < pd.Timestamp(END_EXCLUSIVE)
    span_days = (pd.Timestamp(df["timestamp"].max()) - pd.Timestamp(df["timestamp"].min())).total_seconds()/86400
    return {
        "coverage_state": "PASS" if len(df) >= 1000 and span_days >= 120 else "FAIL",
        "ohlc_state": "PASS" if bad_ohlc == 0 else "FAIL",
        "duplicate_state": "PASS" if dup == 0 else "FAIL",
        "causality_state": "PASS" if ordered and start_ok and end_ok else "FAIL",
        "duplicate_rows": dup,
        "invalid_ohlc_rows": bad_ohlc,
        "span_days": span_days,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, required=True, help="Trailaris broker symbol catalog CSV exported from the same exact MT5 account")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--jurisdiction", required=True)
    p.add_argument("--account-type", required=True)
    p.add_argument("--terminal-path", default="")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-symbols", type=int, default=100)
    p.add_argument("--evidence-ref", default="")
    args = p.parse_args()

    if not mt5.initialize(path=args.terminal_path or None):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        ai = mt5.account_info()
        if ai is None:
            raise SystemExit("No logged-in MT5 account")
        broker = str(ai.company)
        server = str(ai.server)

        with args.catalog.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        batch = rows[args.start_index: args.start_index + args.max_symbols]
        args.outdir.mkdir(parents=True, exist_ok=True)

        data_registry = []
        diagnostics = []
        for i, r in enumerate(batch, start=args.start_index):
            raw = (r.get("raw_symbol") or r.get("symbol") or r.get("name") or "").strip()
            canonical = (r.get("canonical_instrument") or "").strip()
            asset_class = (r.get("asset_class") or "").strip()
            if not raw or not canonical:
                diagnostics.append({"index":i,"raw_symbol":raw,"canonical_instrument":canonical,"state":"SKIP_UNMAPPED"})
                continue

            if not mt5.symbol_select(raw, True):
                diagnostics.append({"index":i,"raw_symbol":raw,"canonical_instrument":canonical,"state":"SYMBOL_SELECT_FAIL","error":str(mt5.last_error())})
                continue

            rates = mt5.copy_rates_range(raw, mt5.TIMEFRAME_M1, START, END_EXCLUSIVE)
            if rates is None or len(rates) == 0:
                diagnostics.append({"index":i,"raw_symbol":raw,"canonical_instrument":canonical,"state":"NO_HISTORY","error":str(mt5.last_error())})
                continue

            d = pd.DataFrame(rates)
            d["timestamp"] = pd.to_datetime(d["time"], unit="s", utc=True)
            d = d[(d["timestamp"] >= pd.Timestamp(START)) & (d["timestamp"] < pd.Timestamp(END_EXCLUSIVE))].copy()
            d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            d["bid"] = d["close"]
            d["ask"] = ""
            d["spread"] = pd.to_numeric(d.get("spread", 0), errors="coerce").fillna(0)
            d["volume"] = pd.to_numeric(d.get("real_volume", d.get("tick_volume", 0)), errors="coerce").fillna(0)
            d["source"] = f"MT5_BROKER_M1::{broker}::{server}::{args.account_type}"
            d["asset"] = canonical
            out = d[["timestamp","open","high","low","close","bid","ask","spread","volume","source","asset"]].copy()
            check = validate(out)

            path = args.outdir / f"{canonical}__{raw}_M1_normalized.csv.gz"
            with gzip.open(path, "wt", newline="", encoding="utf-8") as gz:
                out.to_csv(gz, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
            digest = sha256(path)

            state = "PASS" if all(check[k] == "PASS" for k in ("coverage_state","ohlc_state","duplicate_state","causality_state")) else "FAIL"
            diagnostics.append({"index":i,"raw_symbol":raw,"canonical_instrument":canonical,"state":state,**check,"rows":len(out),"data_hash":digest})
            data_registry.append({
                "canonical_instrument":canonical,
                "asset_class":asset_class,
                "broker_legal_entity":broker,
                "jurisdiction":args.jurisdiction,
                "server":server,
                "account_type":args.account_type,
                "data_source":f"MT5_BROKER_M1::{broker}::{server}::{args.account_type}",
                "data_start_utc":str(out["timestamp"].min()),
                "data_end_utc":str(out["timestamp"].max()),
                "bar_interval":"M1",
                "rows":str(len(out)),
                "coverage_state":check["coverage_state"],
                "ohlc_state":check["ohlc_state"],
                "duplicate_state":check["duplicate_state"],
                "causality_state":check["causality_state"],
                "data_hash":digest,
                "evidence_ref":args.evidence_ref,
            })

        reg = args.outdir / f"HISTORICAL_DATA_REGISTRY__{args.start_index:05d}.csv"
        with reg.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS); w.writeheader(); w.writerows(data_registry)
        (args.outdir / f"HISTORY_EXPORT_DIAGNOSTIC__{args.start_index:05d}.json").write_text(json.dumps({
            "broker_legal_entity":broker,
            "server":server,
            "jurisdiction":args.jurisdiction,
            "account_type":args.account_type,
            "window_start":START.isoformat(),
            "window_end_exclusive":END_EXCLUSIVE.isoformat(),
            "batch_start_index":args.start_index,
            "batch_max_symbols":args.max_symbols,
            "processed_catalog_rows":len(batch),
            "history_registry_rows":len(data_registry),
            "diagnostics":diagnostics,
            "places_orders":False,
        }, indent=2), encoding="utf-8")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
