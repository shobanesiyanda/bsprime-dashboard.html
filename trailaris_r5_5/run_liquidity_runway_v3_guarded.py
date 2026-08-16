#!/usr/bin/env python3
from __future__ import annotations
import json
import run_liquidity_runway_v3_from_memory112_promoted as v3

_original = v3.apply_adaptive_banking


def protected_runner_only(r, hist, fcache):
    if str(r.get('management', '')) != 'PROTECTED_RUNNER':
        out = r.copy()
        out['r5_5_v3_adaptive_bank_fraction'] = 0.0
        out['r5_5_v3_runner_fraction'] = 1.0
        out['r5_5_v3_management_trace'] = '[]'
        return out
    return _original(r, hist, fcache)


v3.apply_adaptive_banking = protected_runner_only

if __name__ == '__main__':
    v3.main()
