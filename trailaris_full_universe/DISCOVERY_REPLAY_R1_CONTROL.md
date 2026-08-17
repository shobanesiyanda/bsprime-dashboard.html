# Quant V2 Full-Universe Discovery Replay R1

Change reference: TR-MEE-QV2-FUR-DISC-R1-20260817-001

This is a non-certifying research sidecar to the locked broker-derived full-universe replay. It exists to test frozen Quant V2 across the widest independently obtainable M1 universe now, without waiting for broker/server/account catalogue ingestion.

Hard constraints:
- `trailaris_r5_5/quantitative_integrated_v2.py` remains byte-identical to blob `120390ab52e80f51bafedc42389ceb7420a410f2`.
- Replay window is fixed at 2026-01-19T00:00:00Z through 2026-08-15T00:00:00Z, matching the frozen 26-week research window.
- Instruments are discovered from independent historical-data metadata, not selected using replay outcomes.
- Frozen R4 factor compatibility is checked before expensive data acquisition; incompatible instruments are recorded and excluded rather than repaired.
- Each accepted instrument must pass timestamp, duplicate and OHLC validity gates before candidate generation.
- Frozen Quant V2 promotion/veto/rank/risk-fraction logic is used unchanged.
- Because exact prescribed broker contract specifications are not yet populated, final portfolio accounting uses a normalized-R sizing harness. This lane therefore cannot certify broker execution, margin, lots, commissions or production eligibility.
- The locked broker-derived `CORE_FULL_UNIVERSE` and `FULL_UNIVERSE_PLUS_DERIV_OPTIONAL` lanes remain unchanged and still require exact prescribed broker legal entity, jurisdiction, server, account type, symbol mapping, contract specifications and execution-cost evidence.
- No discovery result may retroactively change the 34-route Quant V2 evidence base.

The discovery lane is for opportunity breadth, frozen-model applicability, campaign selection and normalized portfolio behaviour only. Production claims remain prohibited until the broker-derived replay and live certification gates are passed.
