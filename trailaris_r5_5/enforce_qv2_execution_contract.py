#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def git_blob(path: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], text=True).strip()


def fail(msg: str) -> None:
    raise SystemExit(f"QV2_EXECUTION_CONTRACT_FAIL: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="trailaris_r5_5/QV2_EXECUTION_CONTRACT_LOCK.json")
    ap.add_argument("--comparison-harness", action="append", default=[])
    ap.add_argument("--declared-delta", default="ASSET_UNIVERSE_ONLY")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get("state") != "HARD_LOCK_FAIL_CLOSED":
        fail("manifest is not fail-closed")
    if manifest.get("currency") != "USD" or float(manifest.get("start_equity", -1)) != 100.0:
        fail("USD100 canonical convention changed")
    if args.declared_delta != manifest.get("declared_experiment_delta"):
        fail(f"undeclared experiment delta: {args.declared_delta!r}")

    for path, expected in manifest["protected_git_blobs"].items():
        actual = git_blob(path)
        if actual != expected:
            fail(f"protected blob drift {path}: {actual} != {expected}")

    baseline = json.loads(Path("trailaris_r5_5/R5_5_V2_26W_CANONICAL_BASELINE.json").read_text())
    if baseline.get("baseline_id") != manifest.get("baseline_id"):
        fail("canonical baseline id drift")
    if baseline.get("state") != "LOCKED_CANONICAL_BASELINE":
        fail("canonical baseline is not locked")
    expected = manifest["canonical_identity"]
    observed = baseline["performance"]
    exact_int = ["positive_weeks", "executed_trades", "wins", "losses", "flats"]
    for k in exact_int:
        if int(observed[k]) != int(expected[k]):
            fail(f"canonical identity metadata drift {k}")
    for k in ["end_equity", "compounded_return_pct", "profit_factor", "max_weekly_drawdown_pct", "max_event_equity_drawdown_pct"]:
        if abs(float(observed[k]) - float(expected[k])) > 1e-12:
            fail(f"canonical identity metadata drift {k}")

    forbidden = [
        (re.compile(r"\b(?:base\.)?r4\.size_trade\s*="), "runtime size_trade monkey-patch"),
        (re.compile(r"\bnormalized_size_trade\b"), "normalized sizing substitution"),
        (re.compile(r"NORMALIZED_R", re.I), "normalized-R comparison semantics"),
        (re.compile(r"broker_contract_sizing_used['\"\s:]*false", re.I), "broker/spec sizing disabled"),
    ]
    for harness in args.comparison_harness:
        p = Path(harness)
        if not p.exists():
            fail(f"comparison harness missing: {harness}")
        text = p.read_text(errors="replace")
        for pattern, reason in forbidden:
            if pattern.search(text):
                fail(f"{reason} in {harness}")

    print("QV2_EXECUTION_CONTRACT PASS")
    print(json.dumps({
        "contract_id": manifest["contract_id"],
        "baseline_id": manifest["baseline_id"],
        "currency": "USD",
        "start_equity": 100.0,
        "declared_delta": args.declared_delta,
        "protected_blob_count": len(manifest["protected_git_blobs"]),
        "comparison_harnesses_checked": args.comparison_harness,
        "golden_replay_required": True
    }, sort_keys=True))


if __name__ == "__main__":
    main()
