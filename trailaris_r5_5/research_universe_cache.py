from __future__ import annotations
import pickle
from pathlib import Path

SNAPSHOT_VERSION='R5_5_BASE_UNIVERSE_34_20260504_20260814_V1'

def load_or_build(base_module, rawdir:Path, snapshot:Path|None=None):
    """Load the immutable research universe snapshot when available, otherwise build it exactly once.

    This is a research-harness optimization only. The snapshot contains the direct output of
    R5_BASE_CAUSAL_ENGINE.build_universe(rawdir): data validation state, candidates, opportunities and
    completed feature caches. No replay, promotion, selection, management or outcome state is cached.
    """
    if snapshot is not None:
        snapshot=Path(snapshot)
        if snapshot.exists():
            with snapshot.open('rb') as h:payload=pickle.load(h)
            if payload.get('snapshot_version')!=SNAPSHOT_VERSION:
                raise RuntimeError(f"research universe snapshot version mismatch: {payload.get('snapshot_version')}")
            return payload['universe']
    universe=base_module.build_universe(Path(rawdir))
    if snapshot is not None:
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        tmp=snapshot.with_suffix(snapshot.suffix+'.tmp')
        with tmp.open('wb') as h:pickle.dump({'snapshot_version':SNAPSHOT_VERSION,'universe':universe},h,protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(snapshot)
    return universe
