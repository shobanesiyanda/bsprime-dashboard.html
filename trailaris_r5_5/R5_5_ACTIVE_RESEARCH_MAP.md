# Trailaris R5.5 — Canonical Active Research Map

This file is the authoritative continuation map for the active Trailaris Market Execution Engine research branch.

## Canonical active branch

- `trailaris-ephemeral-research-20260815` — active R5.5 research branch.
- Frozen historical branches, prior lanes, scripts, and result files are evidence only. They are not active launch targets.

## Immutable primary control

- Control ID: `R5_4_FROZEN_26W_20260816`
- Starting equity: `$100`
- Ending equity: `$1,136.4556497760627`
- Compounded return: `+1,036.4556497760627%`
- Evaluated weeks: `26`
- Positive weeks: `26/26`
- Executed trades: `2,152`
- Wins: `856`
- Losses: `505`
- Flats: `791`
- Non-flat win rate: `62.8949301984%`
- Profit factor: `2.7918406371`
- Max weekly drawdown: `-2.9475910078%`
- Max event/equity drawdown: `-3.1202037867%`
- Required universe: `34 routes × 15 strategy families = 510 cells`

The old 11-week `$337-$345` R5.4/R5.5 frontiers are historical evidence only and must never replace this 26-week control.

## Single active lane

**R5.5 26-Week Forensic Attribution** is the only active lane.

Its job is to mine the already-completed frozen 26-week evidence estate and determine:

1. why the 856 winners won;
2. why the 505 losses lost;
3. why the 791 trades finished flat;
4. which pre-entry causal states separate those populations;
5. which management decisions preserved or surrendered favorable excursion;
6. which losses are avoidable versus economically unavoidable;
7. which flats represent unrealized opportunity versus correct risk neutralization;
8. where portfolio capacity/ranking caused opportunity regret;
9. which winner characteristics can be strengthened without sacrificing diversification, drawdown quality, or the right tail.

## Actions / compute policy

- No R5.4 26-week baseline rerun.
- No candidate portfolio replay.
- No acquisition run.
- No optimization sweep.
- No old lane may be reopened merely because its code or results remain in the repository.
- GitHub Actions may be used only after an evidence-backed new candidate exists and replay is explicitly authorized.
- The sole workflow launcher is manual-only and contains a forensic-only guard. It is not automatically triggered.

## Evidence preservation

Historical scripts and result artifacts remain available for forensic audit. Removing obsolete workflow launchers does not delete historical evidence.

## Hard locks

- Frozen R5.4 control is immutable.
- Full approved scope remains 34 × 15 = 510 cells for any future promotion replay.
- Hindsight fields may label historical outcomes but may not be fed backward into historical decisions.
- No material R5.5 architecture or replay is promoted without evidence and explicit approval.

## Continuation rule

`continue` means advance from the frozen 26-week control through forensic attribution. It does not mean rerun, regress to an earlier lane, or restart an obsolete frontier.
