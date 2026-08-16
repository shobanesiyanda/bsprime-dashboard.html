# Trailaris R5.5 — Canonical Final Release Map

## Final locked release

**Trailaris R5.5 is closed for refinement.** The canonical final R5.5 engine is **R5.5 Integrated Negative Refinement V2**.

- Release ID: `TRAILARIS_R5_5_FINAL_20260817`
- Canonical engine ID: `R5_5_INTEGRATED_NEGATIVE_REFINEMENT_V2_26W_LOCKED_20260816`
- Scope: **26 weeks × 34 routes × 15 strategy families = 510 route-strategy cells**
- Starting equity: **$100**
- Ending equity: **$1,366.8938578865148**
- Compounded return: **+1,266.8938578865148%**
- Positive / negative weeks: **26 / 0**
- Mean / median weekly return: **10.6240% / 11.5154%**
- Weeks ≥5% / ≥10%: **22 / 21**
- Executions: **2,136**
- Wins / losses / flats: **1,221 / 478 / 437**
- Non-flat win rate: **71.8658%**
- Profit Factor: **2.9195**
- Max weekly drawdown: **-3.7441%**
- Max event-equity drawdown: **-2.7024%**
- Rolling 11-week minimum / median / mean / maximum return: **175.6827% / 198.2718% / 200.3838% / 223.8214%**

The final R5.5 architecture retains the validated 112-day bounded proposal memory, 56-day reliability memory and 14-day validation layer; applies causal negative-edge entry exclusions; monetizes the validated weak-state flat population with a 25% realization at first +0.75R; preserves the remaining open-ended protected runner; does not credit early released capacity; and has **no global 3R ceiling**.

## Closure governance

R5.5 is now immutable as a released research version. There is no active R5.5 refinement mandate and no active R5.5 candidate lane.

Any future feature, model, threshold, risk, execution, data, liquidity/runway, loss-repair, flat-repair or architecture change must be developed under a **separately versioned Trailaris successor or update**. It must not silently mutate this final R5.5 release.

The full-universe standard remains preserved for any future claimed full-system successor: **34 routes × 15 strategy families = 510 cells**, unless the user explicitly authorizes a different scope.

## Preserved research evidence

### Frozen R5.4 historical control

- End equity: **$1,136.4556497760627**
- Return: **+1,036.4556497760627%**
- Wins / losses / flats: **856 / 505 / 791**
- Profit Factor: **2.7918406371**
- Max weekly drawdown: **-2.9475910078%**
- Max event-equity drawdown: **-3.1202037867%**

R5.4 remains immutable historical evidence.

### Bounded-memory 112-day predecessor

- End equity: **$1,272.6945454530814**
- Return: **+1,172.6945454530814%**
- Wins / losses / flats: **879 / 505 / 822**
- Profit Factor: **2.8370396693**
- Max weekly/event drawdown: **-2.6697987101% / -2.6697987101%**

This remains preserved as predecessor evidence.

### Liquidity/Runway V3 research challenger — rejected for release replacement

V3 demonstrated that causal runway/continuation management can materially reduce flats, but it did not beat V2 as a complete portfolio architecture. It reduced flats from 437 to 26 and losses from 478 to 460, but reduced ending equity to **$891.8805650248761**, lowered Profit Factor to **2.7279656969756694**, weakened rolling-window performance and worsened event-equity drawdown to **-3.6442412114453537%**. Its research findings remain available for future separately versioned development, but **none of its rejected release-level changes are incorporated into final R5.5**.

## Evidence classification

`FINAL_LOCKED_RESEARCH_RELEASE; FULL_26_WEEK_CAUSAL_WALK_FORWARD_RESEARCH_PROXY; BROKER_PRODUCTION_CERTIFICATION_SEPARATE`
