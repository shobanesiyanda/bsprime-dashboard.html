# Trailaris Broker–Account Qualification Gates

No broker brand is approved by name alone. Approval is granted only to an exact broker legal entity + jurisdiction + MT5 server + account type + account currency + leverage configuration + position mode + contract specification tuple.

## Gate 1 — Legal / entity identity
- Broker legal entity verified from official legal documents.
- Regulator and licence verified.
- Client jurisdiction eligibility verified.
- Contracting entity stored exactly.

## Gate 2 — Account / server identity
- MT5 account type exact match.
- Live server identifier captured from the account.
- Demo server may be separately certified but never substituted for live certification.
- Account currency, leverage and hedging/netting mode captured.

## Gate 3 — Full symbol catalogue ingestion
- Export every symbol exposed to the exact account/server.
- Preserve raw broker symbol names.
- Import all symbols into BROKER_SYMBOL_REGISTRY.csv.
- No fuzzy symbol mapping. Exact canonical match or explicit controlled alias only.

## Gate 4 — Contract specification
For every mapped symbol capture and validate:
- digits and point
- tick size and tick value
- contract size
- minimum/maximum/step volume
- stops level and freeze level
- initial margin / leverage behaviour
- long and short swap
- trading/session status

## Gate 5 — Cost model
- live spread distribution
- commissions
- swap / financing
- conversion costs
- minimum effective transaction cost
- rollover and weekend treatment

## Gate 6 — Execution behaviour
- order types accepted
- Expert Advisor / algorithmic execution permission
- market execution behaviour
- slippage distribution
- reject / requote rate
- partial-fill behaviour
- latency distribution
- stop/limit handling

## Gate 7 — Data integrity
- M1 timestamp/session integrity
- missing and duplicate bars
- invalid OHLC rate
- price discontinuities
- corporate-action / symbol-roll treatment where relevant
- timezone / daylight-saving handling

## Gate 8 — Quant V2 compatibility
- Frozen Quant V2 remains unchanged.
- Existing 34-route reference universe is used as the control where symbols overlap.
- Additional registry instruments are not assumed strategy-compatible merely because they exist.
- Any extension of Quant V2 decision logic to new instruments is a successor quantitative release, never an in-place mutation.

## Gate 9 — Demo execution certification
- Exact broker/account/server tuple exercised under live market conditions using demo where technically representative.
- Order-state, fill-state, risk, recovery and fail-closed behaviour verified.

## Gate 10 — Small-capital live certification
- Exact production tuple tested with controlled capital.
- Costs, slippage, latency and operational behaviour reconciled against the qualification model.

## Final states
- CANDIDATE
- DATA_INGESTED
- CONTRACT_SPEC_VERIFIED
- DEMO_CERTIFIED
- LIVE_CERTIFIED
- APPROVED
- SUSPENDED
- REVOKED

Production execution is allowed only for APPROVED tuples and approved canonical instruments. Any mismatch fails closed.
