# Trailaris Market Reference Registry — successor-line control

Change reference: TR-MEE-QV2-MRR-20260817-001

## Immutable source
- Frozen Quant V2 source commit: `194031e05594e6921b5410e3eb328a5fb53fde94`.
- Frozen Quant V2 model blob: `120390ab52e80f51bafedc42389ceb7420a410f2`.
- The 34-route Quant V2 reference universe remains intact for historical/replay comparability and is not edited in place.

## Successor branch
`trailaris-quant-v2-full-asset-registry-20260817`

This branch separates the Market Reference Registry from the frozen Quant V2 reference universe. The registry is allowed to contain the entire instrument catalogue exposed by a prescribed broker/server/account type without making those instruments automatically eligible for Quant V2 execution.

## Authority rules
1. A broker is identified by legal entity and jurisdiction, not brand name alone.
2. A broker environment is identified by exact platform, server and account type.
3. Every broker symbol is stored, including symbols not yet mapped to a canonical Trailaris instrument.
4. Canonical mapping is fail-closed. Automatic fuzzy/suffix stripping is prohibited. Mapping requires exact canonical equality or an explicit controlled alias.
5. Instrument presence does not equal execution approval.
6. Production eligibility requires a certified broker/account tuple plus contract-spec, cost, execution, data and strategy/model evidence.
7. Deriv synthetic routes remain an optional specialist domain; they are not a dependency of the core conventional-market registry.
8. No registry import may modify frozen Quant V2 logic, thresholds, parameters or historical evidence.

## Population rule
The authoritative full catalogue is pulled from the exact prescribed broker/server/account environment because broker websites and generic brand-level catalogues do not define the exact symbols, suffixes, contract specifications or account availability used by execution.

`import_broker_symbol_catalog.py` ingests the full exported catalogue and retains unmapped symbols as `UNMAPPED` / `PENDING_BROKER_ACCOUNT_CERTIFICATION` rather than guessing.
