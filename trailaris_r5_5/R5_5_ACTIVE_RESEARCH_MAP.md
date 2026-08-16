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
- `trailaris_r5_5/R5_5_LANE5_CERTIFIED_LEADER.json`

## Current performance frontier

- **R5.4 frozen control:** `$100 -> $337.4231242104393`; 446 wins / 231 losses / 425 flats; 65.8789% non-flat WR; max weekly DD -4.6376456%; fresh +7.2616466%.
- **Lane 1B-2 frozen evidence:** `$100 -> $343.5447309255116`; 467 wins / 233 losses / 426 flats; historical return improvement but repair/fresh gates fail.
- **Lane 5 certified full-universe historical frontier:** `$100 -> $345.3896887826546`; 470 wins / 228 losses / 421 flats; 67.3352% non-flat WR; max weekly DD unchanged at -4.6376456%; fresh remains +7.2616466%.
  - Equivalent certified stacks: `B3_B4_B8` and `B3_B5_B8`.
  - Canonical simpler baseline: `B3_B5_B8`.
  - Exact certification: 34 routes / 15 approved strategy families / 510 research cells, exact R5.4 control reproduction.
  - Lane 5 broad stacking is CLOSED. It is **not yet the frozen R5.5 candidate** because strict fresh improvement has not been demonstrated.

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
- **Original Lane 6B/6C/6D broad parallel replay:** INVALIDATED/CANCELLED as architecture evidence, not as an economic winner-path failure. The first 6B job proved the generic runner tried to read `reliability_asset_strategy_mean_r` from raw candidates before the causal reliability layer had created that field. The entire 34-variant run was cancelled to prevent invalid/duplicated compute.
- **Corrected winner qualification stage:** `baseline market candidate -> weekly causal promotion -> causal reliability annotation -> market-state enrichment -> winner-rule qualification -> winner monetization action -> portfolio selection/replay`. This reuses the exact R5.4 reliability logic and does not use future outcomes.
- **Corrected targeted winner replay:** ACTIVE. `lane6_targeted_causal_winner_replay.py` tests AMPLIFY / EXTEND / PRIORITY both standalone on B3 and directly on the certified Lane 5 baseline. Amplification copies are created only after an already-promoted winner-only add-on opportunity and are capped at 0.25% add-on risk; priority changes ordering only after promotion eligibility is fixed; extension activates only after a qualified base trade reaches the original 3R target.
- **Path-Specific Winner Attribution:** ACTIVE/restarted. It separately learns pre-Aug13 rules for amplification potential, post-3R right-tail extension value and high economic contribution, rather than forcing the same high-WR cohort onto every path.
- **Single-build path-targeted matrix:** PREPARED. `lane6_path_targeted_matrix.py` will build the 34-route universe once, reproduce R5.4/B3/Lane5 once, and then test the best path-specific rule for AMPLIFY, EXTEND and PRIORITY sequentially against the same evidence state.
- **Lane 6E — Aug13–14 Winner State Difference Forensic:** ACTIVE/restarted diagnostic. Determines whether fresh under-response is caused by missing winner signature, missing confirmation, promotion/capital competition, add-on capacity/factor blocking, failure to reach 3R, or post-3R monetization. Aug13–14 remains quarantine evidence and cannot select/promote a candidate.
- **Lane 6E fast candidate-presence diagnostic:** ACTIVE. Separately answers whether validated winner-like opportunities exist in Aug13–14 before portfolio selection.
- **Lane 7 — Winner + Management Fusion:** PREPARED, not yet launched. It starts from the certified Lane 5 frontier and may add only a causally valid winner mechanism.

## Winner archetype guidance

- Mean Reversion is the strongest high-hit-rate cohort, but its winner MFE is only about 1.85R; do not automatically use it for right-tail extension.
- Ensemble Regime is the primary payoff-extension/amplification pool with about 2.50R winner MFE and the largest strategy-family winner contribution.
- FX is the largest absolute winner pool.
- Crypto has stronger right-tail characteristics at about 2.86R winner MFE.
- Session Time has an extreme ~4R winner MFE but only four winners and remains research-only due to low support.

## Closed / rejected research lanes

- **Lane 5 broad management stacking:** CLOSED with certified leader `$345.3896887826546`; replay workflow retired from active execution estate.
- **Lane 3 V3 strategy rebuild:** REJECTED/CLOSED. Six authentic replays completed. Best variant `SD_STRICT_ONLY` ended at `$334.06966928043823`, fresh remained +7.2616466%, selected-asset coverage fell to 32. No independent alpha eligible for fusion.
- **Lane 4B market-state entry gate:** REJECTED. Loss/flat repair did not compensate for material ending-equity regression.
- **Lane 1C:** REJECTED; equal to R5.4 with no rescue value.
- **Lane 1D discovery / Lane 1E authentic replay:** REJECTED after authentic replay materially regressed return.
- **Lane 2A V3:** CLOSED NULL RESULT; generic reliability/quality/rank entry filtering produced zero passing rules.

Completed and failed lanes remain auditable evidence, but they are not active system components simply because their files remain in the repository.

## Decision path from here

1. Complete the corrected stage-valid targeted amplification smoke replay.
2. Complete path-specific winner attribution and Lane 6E fresh-state diagnosis.
3. Trigger the single-build path-targeted matrix using the best pre-Aug13 rule for each economic winner path.
4. If any mechanism improves fresh return and adds value on top of Lane 5 without violating the frontier, test parameter robustness only around that mechanism; do not reopen broad tuning.
5. If none improves fresh return, use Lane 6E to repair the identified Aug13–14 state dependency and replay the complete 34×15×510 engine.
6. Only a fused full 34×15×510 candidate that clears the frontier proceeds to the forensic-completeness gates and candidate freeze.
7. Only after candidate freeze may new unseen forward data be acquired/opened. Target-server broker execution certification remains required after forward validation.

## Legacy estate

`trailaris_ephemeral/` is evidence-only. Obsolete executable acquisition/replay code and pre-R4 workflows have been removed from the active branch. Invalid/rejected result artifacts remain solely for auditability; invalid executable launchers are retired after replacement.

## Hard lock

Every material research, replay, attribution, optimization and promotion result must preserve the approved 34 routes, 15 strategy families and 510 route-strategy cells. Diagnostic isolation does not count as a system result.
