# Trailaris R5.5 — Canonical Active Research Map

This file exists to prevent legacy branches, historical workflows and failed research artifacts from being mistaken for the active Market Execution Engine build.

## Canonical branches

- `main` — repository baseline; not the live R5.5 research branch.
- `trailaris-ephemeral-research-20260815` — **canonical active R5.5 research branch**.
- `trailaris-r5-5-lane1b2-frozen` — **immutable frozen Lane 1B-2 evidence branch**; never mutate.
- `trailaris-data-runtime-20260813` — historical acquisition/runtime branch with unique acquisition commits; evidence/reference only until its unique work is formally reconciled.
- `trailaris-r5-5-candidate-20260816` — superseded early R5.5 candidate branch; **do not use for active research**. Its unique tip is preserved under `archive/trailaris-r5-5-candidate-20260816` pending branch-ref deletion capability.

Any other `archive/*` ref is non-canonical and must never be used for execution, research selection, promotion, deployment or production.

## Immutable controls and governance

- `trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json`
- `trailaris_r5_5/R5_5_PROMOTION_CONTRACT.json`
- `trailaris_r5_5/R5_5_FRONTIER_LOCK.json`
- `trailaris_r5_5/R5_5_HANDOVER_GAP_REGISTER.json`
- `trailaris_r5_5/R5_5_PRODUCTION_SAFETY_CONTRACT.json`

## Current performance references

- R5.4 frozen control: `$100 -> $337.4231242104393`.
- Lane 1B-2 incumbent return benchmark: `$100 -> $343.5447309255116`; not promotable because repair/fresh gates fail.
- Lane 1G clean repair candidate: `$100 -> $337.90138673186806`, 451 wins / 228 losses / 423 flats; pre-forward only.
- Lane 4B market-state gate: rejected after full replay (`$322.8413858993665`).

The active objective is to clear the dual frontier: exceed Lane 1B-2 return while simultaneously meeting or improving R5.4 risk/repair quality and all research-completeness gates.

## Active research lanes

- Lane 5 — Pareto management stacking.
- Lane 6A — winner attribution.
- Lane 6B — winner amplification/pyramiding after proof.
- Lane 6C — winner payoff extension / protected runner beyond the legacy 3R ceiling.
- Lane 6D — winner-priority opportunity capture.
- Lane 3 V3 — payoff-gated weak-strategy rebuild where still running/valid.

Completed or failed lanes remain evidence, but they are not the active system simply because their files remain in the repository.

## Legacy estate

`trailaris_ephemeral/` is evidence-only. Its obsolete executable acquisition/replay code and pre-R4 workflows have been removed from the active branch. Remaining result artifacts are retained solely for auditability.

## Hard locks

Every material research, replay, attribution, optimization and promotion result must preserve the approved 34 routes, 15 strategy families and 510 route-strategy cells. Diagnostic isolation does not count as a system result.
