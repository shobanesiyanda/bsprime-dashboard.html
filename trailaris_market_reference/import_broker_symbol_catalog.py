#!/usr/bin/env python3
"""Import a complete broker/server/account symbol catalogue into the Trailaris Market Reference Registry.

This importer does NOT modify Quant V2 and does NOT make an instrument trade-eligible.
Every broker symbol is retained. Canonical mapping is fail-closed: only exact canonical
matches or explicit alias mappings are accepted. All other symbols remain UNMAPPED.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "QUANT_V2_REFERENCE_UNIVERSE.csv"
OUTPUT_HEADER = [
    "broker_legal_entity","jurisdiction","server","account_type","raw_symbol",
    "canonical_instrument","asset_class","description","currency_base","currency_profit",
    "trade_mode","digits","point","tick_size","tick_value","contract_size","volume_min",
    "volume_step","volume_max","stops_level","freeze_level","margin_initial","swap_long",
    "swap_short","mapping_state","certification_state","evidence_ref"
]

FIELD_ALIASES = {
    "raw_symbol": ["raw_symbol", "symbol", "name"],
    "description": ["description", "symbol_description"],
    "currency_base": ["currency_base", "base_currency"],
    "currency_profit": ["currency_profit", "profit_currency", "quote_currency"],
    "trade_mode": ["trade_mode", "symbol_trade_mode"],
    "digits": ["digits"],
    "point": ["point"],
    "tick_size": ["tick_size", "trade_tick_size"],
    "tick_value": ["tick_value", "trade_tick_value"],
    "contract_size": ["contract_size", "trade_contract_size"],
    "volume_min": ["volume_min", "lots_min"],
    "volume_step": ["volume_step", "lots_step"],
    "volume_max": ["volume_max", "lots_max"],
    "stops_level": ["stops_level", "trade_stops_level"],
    "freeze_level": ["freeze_level", "trade_freeze_level"],
    "margin_initial": ["margin_initial"],
    "swap_long": ["swap_long"],
    "swap_short": ["swap_short"],
}


def first(row: Dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name]).strip()
    return ""


def load_reference() -> Dict[str, str]:
    out: Dict[str, str] = {}
    with REFERENCE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["canonical_instrument"].strip()] = row["asset_class"].strip()
    return out


def load_aliases(path: Path | None) -> Dict[str, str]:
    if not path:
        return {}
    aliases: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = (row.get("raw_symbol") or "").strip()
            canonical = (row.get("canonical_instrument") or "").strip()
            if raw and canonical:
                aliases[raw] = canonical
    return aliases


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="CSV export containing the full symbol catalogue for one exact broker/server/account type")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--broker-legal-entity", required=True)
    p.add_argument("--jurisdiction", required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--account-type", required=True)
    p.add_argument("--aliases", type=Path, help="Controlled CSV with raw_symbol,canonical_instrument")
    p.add_argument("--evidence-ref", default="")
    args = p.parse_args()

    reference = load_reference()
    aliases = load_aliases(args.aliases)
    rows: List[Dict[str, str]] = []

    with args.input.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("input has no header")
        for src in reader:
            raw = first(src, FIELD_ALIASES["raw_symbol"])
            if not raw:
                continue

            if raw in aliases:
                canonical = aliases[raw]
                mapping_state = "EXPLICIT_ALIAS"
            elif raw in reference:
                canonical = raw
                mapping_state = "EXACT_CANONICAL"
            else:
                canonical = ""
                mapping_state = "UNMAPPED"

            asset_class = reference.get(canonical, "") if canonical else ""
            out = {k: "" for k in OUTPUT_HEADER}
            out.update({
                "broker_legal_entity": args.broker_legal_entity,
                "jurisdiction": args.jurisdiction,
                "server": args.server,
                "account_type": args.account_type,
                "raw_symbol": raw,
                "canonical_instrument": canonical,
                "asset_class": asset_class,
                "mapping_state": mapping_state,
                "certification_state": "PENDING_BROKER_ACCOUNT_CERTIFICATION",
                "evidence_ref": args.evidence_ref,
            })
            for dst, names in FIELD_ALIASES.items():
                if dst == "raw_symbol":
                    continue
                out[dst] = first(src, names)
            rows.append(out)

    rows.sort(key=lambda r: (r["raw_symbol"], r["canonical_instrument"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    mapped = sum(r["mapping_state"] != "UNMAPPED" for r in rows)
    print(f"catalogue_rows={len(rows)} mapped={mapped} unmapped={len(rows)-mapped}")
    print("No row is production-authorised by this import. Broker/account certification remains mandatory.")


if __name__ == "__main__":
    main()
