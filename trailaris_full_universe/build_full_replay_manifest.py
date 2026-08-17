#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

READY_STATES = {"REPLAY_READY", "LIVE_CERTIFIED"}
MAPPING_STATES = {"EXACT_CANONICAL", "EXPLICIT_ALIAS"}
PASS_STATES = {"PASS", "VALID", "OK"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def stable_hash(rows: List[Dict[str, str]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def tuple_key(r: Dict[str, str]) -> Tuple[str, ...]:
    return (
        r.get("broker_legal_entity", ""),
        r.get("jurisdiction", ""),
        r.get("server", ""),
        r.get("account_type", ""),
        r.get("canonical_instrument", ""),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol-registry", type=Path, required=True)
    p.add_argument("--data-registry", type=Path, required=True)
    p.add_argument("--reference-universe", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--shard-size", type=int, default=100)
    args = p.parse_args()

    symbols = read_csv(args.symbol_registry)
    data = read_csv(args.data_registry)
    reference = read_csv(args.reference_universe)

    reference_assets = {r["canonical_instrument"] for r in reference if r.get("canonical_instrument")}
    if len(reference_assets) != 34:
        raise SystemExit(f"frozen reference universe integrity failure: {len(reference_assets)}/34")

    data_by_key = {tuple_key(r): r for r in data if r.get("canonical_instrument")}
    eligible: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []

    for r in symbols:
        canonical = r.get("canonical_instrument", "")
        reason = ""
        if not canonical:
            reason = "NO_CANONICAL_MAPPING"
        elif r.get("mapping_state") not in MAPPING_STATES:
            reason = "MAPPING_NOT_APPROVED"
        elif r.get("certification_state") not in READY_STATES:
            reason = "BROKER_SYMBOL_NOT_REPLAY_READY"
        else:
            d = data_by_key.get(tuple_key(r))
            if d is None:
                reason = "NO_26W_DATA_REGISTRY_ROW"
            elif d.get("coverage_state") not in PASS_STATES:
                reason = "COVERAGE_GATE_NOT_PASS"
            elif d.get("ohlc_state") not in PASS_STATES:
                reason = "OHLC_GATE_NOT_PASS"
            elif d.get("duplicate_state") not in PASS_STATES:
                reason = "DUPLICATE_GATE_NOT_PASS"
            elif d.get("causality_state") not in PASS_STATES:
                reason = "CAUSALITY_GATE_NOT_PASS"
            elif not d.get("data_hash"):
                reason = "MISSING_DATA_HASH"

        if reason:
            rejected.append({**r, "replay_rejection_reason": reason})
            continue

        execution_domain = "DERIV_OPTIONAL" if str(r.get("asset_class", "")).upper() == "SYNTHETIC" else "CORE"
        d = data_by_key[tuple_key(r)]
        eligible.append({
            "canonical_instrument": canonical,
            "asset_class": r.get("asset_class", ""),
            "execution_domain": execution_domain,
            "broker_legal_entity": r.get("broker_legal_entity", ""),
            "jurisdiction": r.get("jurisdiction", ""),
            "server": r.get("server", ""),
            "account_type": r.get("account_type", ""),
            "raw_symbol": r.get("raw_symbol", ""),
            "data_source": d.get("data_source", ""),
            "data_start_utc": d.get("data_start_utc", ""),
            "data_end_utc": d.get("data_end_utc", ""),
            "rows": d.get("rows", ""),
            "data_hash": d.get("data_hash", ""),
            "evidence_ref": d.get("evidence_ref", "") or r.get("evidence_ref", ""),
        })

    # Deduplicate canonical instruments without outcome information. Prefer CORE over optional,
    # then deterministic lexical broker tuple. This is an evidence-selection rule only.
    eligible.sort(key=lambda r: (
        r["canonical_instrument"],
        1 if r["execution_domain"] == "DERIV_OPTIONAL" else 0,
        r["broker_legal_entity"], r["jurisdiction"], r["server"], r["account_type"], r["raw_symbol"]
    ))
    canonical: Dict[str, Dict[str, str]] = {}
    for r in eligible:
        canonical.setdefault(r["canonical_instrument"], r)

    full = sorted(canonical.values(), key=lambda r: r["canonical_instrument"])
    core = [r for r in full if r["execution_domain"] == "CORE"]
    plus_deriv = list(full)

    args.outdir.mkdir(parents=True, exist_ok=True)

    def write_manifest(name: str, rows: List[Dict[str, str]]) -> None:
        path = args.outdir / f"{name}.csv"
        fields = list(rows[0].keys()) if rows else [
            "canonical_instrument","asset_class","execution_domain","broker_legal_entity",
            "jurisdiction","server","account_type","raw_symbol","data_source","data_start_utc",
            "data_end_utc","rows","data_hash","evidence_ref"
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

    write_manifest("CORE_FULL_UNIVERSE_MANIFEST", core)
    write_manifest("FULL_UNIVERSE_PLUS_DERIV_OPTIONAL_MANIFEST", plus_deriv)

    rejected_fields = sorted({k for r in rejected for k in r}) if rejected else ["replay_rejection_reason"]
    with (args.outdir / "REJECTED_REGISTRY_ROWS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rejected_fields)
        w.writeheader(); w.writerows(rejected)

    shards = []
    for lane_name, rows in (("CORE_FULL_UNIVERSE", core), ("FULL_UNIVERSE_PLUS_DERIV_OPTIONAL", plus_deriv)):
        for i in range(0, len(rows), args.shard_size):
            part = rows[i:i + args.shard_size]
            shards.append({
                "lane": lane_name,
                "shard_id": f"{lane_name}__{i // args.shard_size:04d}",
                "asset_count": len(part),
                "first_asset": part[0]["canonical_instrument"] if part else "",
                "last_asset": part[-1]["canonical_instrument"] if part else "",
                "manifest_sha256": stable_hash(part),
                "assets": "|".join(r["canonical_instrument"] for r in part),
            })

    with (args.outdir / "REPLAY_SHARDS.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["lane","shard_id","asset_count","first_asset","last_asset","manifest_sha256","assets"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(shards)

    readiness = {
        "state": "READY" if len(plus_deriv) > 34 and len(core) > 29 else "NOT_READY",
        "frozen_reference_assets": len(reference_assets),
        "symbol_registry_rows": len(symbols),
        "historical_data_registry_rows": len(data),
        "eligible_unique_core_assets": len(core),
        "eligible_unique_full_assets": len(plus_deriv),
        "rejected_registry_rows": len(rejected),
        "shard_size": args.shard_size,
        "shards": len(shards),
        "core_manifest_sha256": stable_hash(core),
        "full_manifest_sha256": stable_hash(plus_deriv),
        "quant_v2_modified": False,
    }
    (args.outdir / "FULL_UNIVERSE_REPLAY_READINESS.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    print(json.dumps(readiness, indent=2))


if __name__ == "__main__":
    main()
