"""Verify batched soft relocation scores against scalar incremental trials.

Usage:
  uv run python test/verification/_verify_batch_soft_scoring.py ibm10
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


def run(bench_name):
    bench, plc = load_benchmark_from_dir(
        str(ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench_name)
    )
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, bench)
    _patch_plc_density(plc, bench)
    placement = bench.macro_positions.cpu().numpy().astype(np.float64)
    scorer = IncrementalScorer(plc, bench, placement)
    rng = np.random.RandomState(89)

    soft_with_nets = [
        soft_k
        for soft_k, module in enumerate(scorer.soft_indices)
        if scorer._macro_nets(int(module)).size
    ]
    ok = True
    for soft_k in soft_with_nets[:8]:
        current = scorer.committed_soft_pos[soft_k]
        targets = current + rng.uniform(-20.0, 20.0, size=(32, 2))
        targets[:, 0] = np.clip(targets[:, 0], 0.0, float(plc.width))
        targets[:, 1] = np.clip(targets[:, 1], 0.0, float(plc.height))

        prep = scorer._prepare_move_soft(soft_k)
        state_before = (
            scorer.H_flat.copy(),
            scorer.V_flat.copy(),
            scorer.H_smoothed.copy(),
            scorer.V_smoothed.copy(),
            scorer.grid_occupied.copy(),
        )
        batch = scorer._trial_many_at_soft(prep, targets)
        state_after_batch = (
            scorer.H_flat,
            scorer.V_flat,
            scorer.H_smoothed,
            scorer.V_smoothed,
            scorer.grid_occupied,
        )
        state_exact = all(
            np.array_equal(before, after) for before, after in zip(state_before, state_after_batch)
        )
        scalar = np.asarray([scorer._trial_at_soft(prep, xy) for xy in targets])
        delta = float(np.max(np.abs(batch - scalar), initial=0.0))
        passed = delta < 1e-12 and state_exact
        print(
            f"  soft={soft_k}: max score delta={delta:.2e}, "
            f"state_exact={int(state_exact)} {'PASS' if passed else 'FAIL'}"
        )
        ok = ok and passed
        scorer._revert_prep_soft(prep)

    swap_scorer = IncrementalScorer(plc, bench, placement)
    swap_soft_with_nets = [
        soft_k
        for soft_k, module in enumerate(swap_scorer.soft_indices)
        if swap_scorer._macro_nets(int(module)).size
    ]
    for soft_a in swap_soft_with_nets[:4]:
        candidates = np.asarray(
            [soft_b for soft_b in swap_soft_with_nets if soft_b != soft_a][:32],
            dtype=np.int64,
        )
        state_before = (
            swap_scorer.H_flat.copy(),
            swap_scorer.V_flat.copy(),
            swap_scorer.H_smoothed.copy(),
            swap_scorer.V_smoothed.copy(),
            swap_scorer.grid_occupied.copy(),
            swap_scorer.plc._last_pos_cache.copy(),
        )
        batch = swap_scorer.score_swap_soft_soft_many(soft_a, candidates)
        prepared = swap_scorer.prepare_swap_soft_soft_source(soft_a, candidates)
        split = min(12, candidates.size)
        prepared_scores = np.concatenate(
            [
                swap_scorer.score_prepared_swap_soft_soft(prepared, 0, split),
                swap_scorer.score_prepared_swap_soft_soft(prepared, split, candidates.size),
            ]
        )
        state_after_batch = (
            swap_scorer.H_flat,
            swap_scorer.V_flat,
            swap_scorer.H_smoothed,
            swap_scorer.V_smoothed,
            swap_scorer.grid_occupied,
            swap_scorer.plc._last_pos_cache,
        )
        state_exact = all(
            np.array_equal(before, after) for before, after in zip(state_before, state_after_batch)
        )
        scalar = np.asarray(
            [swap_scorer.score_swap_soft_soft(soft_a, int(soft_b)) for soft_b in candidates]
        )
        delta = float(np.max(np.abs(batch - scalar), initial=0.0))
        prepared_delta = float(np.max(np.abs(batch - prepared_scores), initial=0.0))
        passed = delta < 1e-12 and prepared_delta == 0.0 and state_exact
        print(
            f"  swap soft={soft_a}: max score delta={delta:.2e}, "
            f"prepared delta={prepared_delta:.2e}, "
            f"state_exact={int(state_exact)} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        ok = ok and passed
    return ok


if __name__ == "__main__":
    benchmark_name = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    if not run(benchmark_name):
        raise SystemExit(1)
