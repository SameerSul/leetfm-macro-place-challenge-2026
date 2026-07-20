"""Verify batched hard-endpoint swap scores against scalar trials.

Usage:
  uv run python test/verification/_verify_batch_hard_swap_scoring.py ibm10
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from placer import (  # noqa: E402
    IncrementalScorer,
    _patch_plc_congestion,
    _patch_plc_density,
    _patch_plc_wirelength,
)


def _state(scorer):
    return (
        scorer.H_flat.copy(),
        scorer.V_flat.copy(),
        scorer.H_macro_flat.copy(),
        scorer.V_macro_flat.copy(),
        scorer.H_smoothed.copy(),
        scorer.V_smoothed.copy(),
        scorer.grid_occupied.copy(),
        scorer.plc._last_pos_cache.copy(),
    )


def _unchanged(before, scorer):
    after = (
        scorer.H_flat,
        scorer.V_flat,
        scorer.H_macro_flat,
        scorer.V_macro_flat,
        scorer.H_smoothed,
        scorer.V_smoothed,
        scorer.grid_occupied,
        scorer.plc._last_pos_cache,
    )
    return all(np.array_equal(a, b) for a, b in zip(before, after))


def run(bench_name):
    bench, plc = load_benchmark_from_dir(
        str(ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench_name)
    )
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, bench)
    _patch_plc_density(plc, bench)
    placement = bench.macro_positions.cpu().numpy().astype(np.float64)
    scorer = IncrementalScorer(plc, bench, placement)

    hard_with_nets = [
        hard_k
        for hard_k, module in enumerate(scorer.hard_indices)
        if scorer._macro_nets(int(module)).size
    ]
    soft_with_nets = [
        soft_k
        for soft_k, module in enumerate(scorer.soft_indices)
        if scorer._macro_nets(int(module)).size
    ]
    ok = True

    for hard_a in hard_with_nets[:4]:
        candidates = np.asarray(
            [hard_b for hard_b in hard_with_nets if hard_b != hard_a][:24],
            dtype=np.int64,
        )
        before = _state(scorer)
        batch = scorer.score_swap_hard_hard_many(hard_a, candidates)
        prepared = scorer.prepare_swap_hard_hard_source(hard_a, candidates)
        split = min(4, candidates.size)
        prepared_scores = np.concatenate(
            [
                scorer.score_prepared_swap_hard_hard(prepared, 0, split),
                scorer.score_prepared_swap_hard_hard(prepared, split, candidates.size),
            ]
        )
        state_exact = _unchanged(before, scorer)
        scalar = np.asarray(
            [scorer.score_swap_hard_hard(hard_a, int(hard_b)) for hard_b in candidates]
        )
        delta = float(np.max(np.abs(batch - scalar), initial=0.0))
        prepared_delta = float(np.max(np.abs(batch - prepared_scores), initial=0.0))
        passed = delta < 1.0e-12 and prepared_delta == 0.0 and state_exact
        print(
            f"  hard-hard source={hard_a}: max score delta={delta:.2e}, "
            f"prepared delta={prepared_delta:.2e}, "
            f"state_exact={int(state_exact)} {'PASS' if passed else 'FAIL'}"
        )
        ok = ok and passed

    for hard_a in hard_with_nets[:4]:
        candidates = np.asarray(soft_with_nets[:24], dtype=np.int64)
        before = _state(scorer)
        batch = scorer.score_swap_hard_soft_many(hard_a, candidates)
        prepared = scorer.prepare_swap_hard_soft_source(hard_a, candidates)
        split = min(8, candidates.size)
        prepared_scores = np.concatenate(
            [
                scorer.score_prepared_swap_hard_soft(prepared, 0, split),
                scorer.score_prepared_swap_hard_soft(prepared, split, candidates.size),
            ]
        )
        state_exact = _unchanged(before, scorer)
        scalar = np.asarray(
            [scorer.score_swap_hard_soft(hard_a, int(soft_b)) for soft_b in candidates]
        )
        delta = float(np.max(np.abs(batch - scalar), initial=0.0))
        prepared_delta = float(np.max(np.abs(batch - prepared_scores), initial=0.0))
        passed = delta < 1.0e-12 and prepared_delta == 0.0 and state_exact
        print(
            f"  hard-soft source={hard_a}: max score delta={delta:.2e}, "
            f"prepared delta={prepared_delta:.2e}, "
            f"state_exact={int(state_exact)} {'PASS' if passed else 'FAIL'}"
        )
        ok = ok and passed
    return ok


if __name__ == "__main__":
    benchmark_name = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    if not run(benchmark_name):
        raise SystemExit(1)
