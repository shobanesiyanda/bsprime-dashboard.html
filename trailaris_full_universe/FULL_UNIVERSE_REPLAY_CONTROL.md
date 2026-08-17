# Trailaris Quant V2 — Full Asset Universe Replay Control

Change reference: TR-MEE-QV2-FUR-20260817-001

## Purpose
Run the frozen Quant V2 logic against the expanded, broker-derived Trailaris asset universe without modifying the frozen Quant V2 release or its 34-route evidentiary baseline.

## Immutable control
- Frozen Quant V2 source commit: `194031e05594e6921b5410e3eb328a5fb53fde94`.
- Frozen Quant V2 model blob SHA-1: `120390ab52e80f51bafedc42389ceb7420a410f2`.
- The original 34-route Quant V2 replay remains the immutable control benchmark.
- No threshold, ranking, veto, sizing, management, factor, strategy, cost or execution rule in Quant V2 may be changed to accommodate the larger universe.

## Replay lanes
1. `CORE_FULL_UNIVERSE`: all replay-ready canonical instruments on prescribed conventional broker/account tuples. Deriv proprietary synthetics excluded.
2. `FULL_UNIVERSE_PLUS_DERIV_OPTIONAL`: lane 1 plus replay-ready Deriv synthetic instruments.
3. `QUANT_V2_34_CONTROL`: the frozen 34-route control, used only for comparison and regression verification.

## Eligibility gate
An instrument may enter a full-universe replay lane only when all of the following exist:
- exact prescribed broker legal entity and jurisdiction;
- exact MT5 server and account type;
- broker symbol captured in `BROKER_SYMBOL_REGISTRY.csv`;
- canonical mapping is `EXACT_CANONICAL` or `EXPLICIT_ALIAS`;
- registry certification state is `REPLAY_READY` or `LIVE_CERTIFIED`;
- 26-week historical data is present and passes timestamp, duplicate, OHLC, coverage and causality checks;
- contract specification required for sizing/cost modelling is present;
- no look-ahead or future-data leakage.

Presence in a broker catalogue does not create replay eligibility.

## Scale architecture
The expanded universe is replayed in deterministic shards rather than one monolithic runner. Each shard is immutable once generated. Shard outputs are aggregated only after every scheduled shard has completed and passed its own data and model-integrity gates.

Default planning shard size: 100 canonical instruments. The shard generator may reduce this for heavy asset classes, but it may not select assets based on observed replay performance.

## Evidence separation
Full-universe results are successor-line research evidence. They do not rewrite, replace or retroactively broaden the evidence claims of frozen Quant V2. Promotion to a production-authorised asset remains separately controlled through broker/account certification and live execution evidence.

## No outcome-driven repair
If a newly added asset fails the frozen model's applicability or data contract, it is excluded with a recorded reason. The frozen model is not altered to make the asset pass. Any proposed model extension requires a separately versioned successor quantitative release and Founder-approved change control.
