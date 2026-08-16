import R5_BASE_CAUSAL_ENGINE as base
from reliability_common import promote
replay_r5=base.replay_r5
def promote_for_week(cands,week_start):return promote(cands,week_start,'strategy_only_gate')
