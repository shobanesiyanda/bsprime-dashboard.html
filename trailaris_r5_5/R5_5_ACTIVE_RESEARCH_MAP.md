# Trailaris R5.5 — Canonical Active Research Map

This file prevents historical branches, workflows and failed research artifacts from being mistaken for the active Market Execution Engine build.

## Canonical branches

- `main` — repository baseline; not the live R5.5 research branch.
- `trailaris-ephemeral-research-20260815` — **canonical active R5.5 research branch**.
- `trailaris-r5-5-lane1b2-frozen` — **immutable frozen Lane 1B-2 evidence branch**; never mutate.
- `archive/*` refs — historical evidence only; never use for active research selection, promotion, deployment or production.

## Immutable controls and governance

- `trailaris_r5_4/R5_4_MARKET_EXECUTION_ENGINE_LOCK.json`
- `trailaris_r5_5/R5_5_PROMOTION_CONTRACT.json`
- `trailaris_r5_5/R5_5_FRONTIER_LOCK.json`
- `trailaris_r5_5/R5_5_HANDOVER_GAP_REGISTER.json`
- `trailaris_r5_5/R5_5_PRODUCTION_SAFETY_CONTRACT.json`
- `trailaris_r5_5/R5_5_WINNER_ARCHETYPE_MAP.json`

## Current performance frontier

- **R5.4 frozen control:** `$100 -> $337.4231242104393`; 446 wins / 231 losses / 425 flats; 65.8789% non-flat WR; max weekly DD -4.6376456%; fresh +7.2616466%.
- **Lane 1B-2 frozen evidence:** `$100 -> $343.5447309255116`; 467 wins / 233 losses / 426 flats; historical return improvement but repair/fresh gates fail.
- **Lane 5 certified full-universe historical frontier:** `$100 -> $345.3896887826546`; 470 wins / 228 losses / 421 flats; 67.3352% non-flat WR; max weekly DD unchanged at -4.6376456%; fresh remains +7.2616466%.
  - Equivalent certified stacks: `B3_B4_B8` and `B3_B5_B8`.
  - Exact certification: 34 routes / 15 approved strategy families / 510 research cells, exact R5.4 control reproduction.
  - Lane 5 is **not yet the frozen R5.5 candidate** because strict fresh improvement has not been demonstrated.

### Active challenge frontier

Any challenger must simultaneously achieve:

- ending equity **> $345.3896887826546**;
- fresh return **> +7.261646557142409%**;
- losses **<= 228**;
- flats **<= 421**;
- non-flat WR **>= 67.3352435530%**;
- max weekly DD no worse than **-4.6376455749%**;
- fresh DD no worse than **-0.5196525641%**;
- selected-asset coverage >= 33;
- exact 34 × 15 = 510 approved research estate.

## Active winner program

- **Lane 6A — Winner Attribution:** COMPLETE. 467 winners analyzed; 3,935 causal rules tested; 501 clean discovery + pre-Aug13 validation rules. This lane answers why winners win.
- **Lane 6B — Winner Amplification:** ACTIVE authentic replay. Tests extra exposure only after a validated winner-like parent has proved its thesis in-market.
- **Lane 6C — Winner Payoff Extension:** ACTIVE authentic replay. Tests protected right-tail realization beyond the ordinary 3R close.
- **Lane 6D — Winner Priority Capture:** ACTIVE authentic replay. Tests whether scarce portfolio capacity should preferentially go to validated winner-like opportunities.
- **Path-Specific Winner Attribution:** ACTIVE/QUEUED. Separates broad win probability from amplification potential, right-tail continuation potential and capital-priority value so one high-WR cohort is not incorrectly used for every winner mechanism.
- **Lane 6E — Aug13–14 Winner State Difference Forensic:** ACTIVE diagnostic. Determines whether fresh under-response is caused by missing winner signature, missing confirmation, promotion/capital competition, add-on capacity/factor blocking, failure to reach 3R, or post-3R monetization. Aug13–14 remains quarantine evidence and cannot select/promote a candidate.
- **Lane 7 — Winner + Management Fusion:** PREPARED, not yet launched. It will start from the certified Lane 5 frontier and add only a winner mechanism that independently demonstrates value.

## Closed / rejected research lanes

- **Lane 3 V3 strategy rebuild:** REJECTED/CLOSED. Six authentic replays completed. Best variant `SD_STRICT_ONLY` ended at `$334.06966928043823`, fresh remained +7.2616466%, selected-asset coverage fell to 32. No independent alpha eligible for fusion. Recovered evidence: `lane3_v3_results/R5_5_LANE3_V3_RECOVERED_STATUS.json`.
- **Lane 4B market-state entry gate:** REJECTED. Loss/flat repair did not compensate for material ending-equity regression.
- **Lane 1C:** REJECTED; equal to R5.4 with no rescue value.
- **Lane 1D discovery / Lane 1E authentic replay:** REJECTED after authentic replay materially regressed return.
- **Lane 2A V3:** CLOSED NULL RESULT; generic reliability/quality/rank entry filtering produced zero passing rules.

Completed and failed lanes remain auditable evidence, but they are not active system components simply because their files remain in the repository.

## Decision path from here

1. Finish 6B/6C/6D authentic replays.
2. Match winner-selection rules to the economic objective of each path using path-specific attribution.
3. If a winner mechanism produces strict fresh improvement without violating the Lane 5 quality frontier, replay it through Lane 7 on top of the certified Lane 5 management stack.
4. If none improves fresh return, stop historical threshold tuning and use Lane 6E to identify and repair the Aug13–14 state dependency.
5. Only a fused full 34×15×510 candidate that clears the frontier proceeds to the forensic-completeness gates and candidate freeze.
6. Only after candidate freeze may new unseen forward data be acquired/opened. Target-server broker execution certification remains required after forward validation.

## Legacy estate

`trailaris_ephemeral/` is evidence-only. Obsolete executable acquisition/replay code and pre-R4 workflows have been removed from the active branch. Remaining result artifacts exist solely for auditability.

## Hard lock

Every material research, replay, attribution, optimization and promotion result must preserve the approved 34 routes, 15 strategy families and 510 route-strategy cells. Diagnostic isolation does not count as a system result.
