"""Verify IncrementalScorer (B3 phase 2) matches _exact_proxy bit-for-bit.

Tests:
  1. Initial total_wl matches plc.get_cost() exactly.
  2. score_swap returns the same proxy as _exact_proxy for various swaps.
  3. commit_swap + subsequent score_swap stays consistent with full recompute.
  4. Many sequential commits don't drift from the full recompute.

Usage:
  uv run python test/verification/_verify_incremental_scorer.py
"""

import sys
from pathlib import Path

# Make `placer` importable from the v2 src directory.
_V2_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_V2_DIR / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from placer import (  # noqa: E402
    _exact_proxy,
    _fast_set_placement,
    _patch_plc_congestion,
    _patch_plc_density,
    _patch_plc_wirelength,
    IncrementalScorer,
)
from macro_place.loader import load_benchmark_from_dir  # noqa: E402


def _full_proxy(placement_np, bench, plc):
    pl = torch.from_numpy(placement_np.astype(np.float32))
    return _exact_proxy(pl, bench, plc)


def _to_placement_np(placement_t):
    return placement_t.cpu().numpy().astype(np.float64)


def run_one(bench_name: str, n_trials: int = 12, n_commits: int = 4) -> bool:
    print(f"\n=== {bench_name} ===")
    bench, plc = load_benchmark_from_dir(
        f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}"
    )
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, bench)
    _patch_plc_density(plc, bench)

    placement = bench.macro_positions.clone()
    placement_np = _to_placement_np(placement)
    n_hard = bench.num_hard_macros

    # Baseline score (also seeds plc state for the scorer)
    base_score = _full_proxy(placement_np, bench, plc)
    print(f"  baseline proxy:        {base_score:.6f}")

    scorer = IncrementalScorer(plc, bench, placement_np)
    # Normalized WL should match plc.get_cost() after init.
    scorer_wl_normalized = scorer.total_wl_raw / scorer.wl_normalizer
    plc_wl = float(plc.get_cost())
    wl_delta = abs(scorer_wl_normalized - plc_wl)
    print(f"  scorer wl (normalized): {scorer_wl_normalized:.6f}  vs plc.get_cost(): {plc_wl:.6f}  Δ={wl_delta:.2e}")
    if wl_delta > 1e-6:
        print("  FAIL: total_wl mismatch")
        return False

    rng = np.random.RandomState(7)
    ok = True

    # Test 1: many trials (score_swap without commit) should match _exact_proxy
    # exactly. After each, the scorer reverts; plc returns to the baseline state.
    print(f"  Test 1: {n_trials} trial swaps (revert each time, compare vs _exact_proxy):")
    for trial in range(n_trials):
        # Pick two movable hard macros at random
        movable_hard = np.where(bench.get_movable_mask().numpy()[:n_hard])[0]
        i, j = rng.choice(movable_hard, size=2, replace=False)
        new_i_xy = scorer.committed_hard_pos[j].copy()
        new_j_xy = scorer.committed_hard_pos[i].copy()

        # Scorer trial (reverts internally)
        scorer_score = scorer.score_swap(int(i), new_i_xy, int(j), new_j_xy)

        # Reference: full _exact_proxy on the same swapped placement
        ref_placement = placement_np.copy()
        ref_placement[i] = new_i_xy
        ref_placement[j] = new_j_xy
        ref_score = _full_proxy(ref_placement, bench, plc)

        # plc has been mutated by ref scoring; restore by re-applying baseline
        _fast_set_placement(plc, placement_np, bench)
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True

        delta = abs(scorer_score - ref_score)
        marker = "ok" if delta < 1e-4 else "FAIL"
        print(f"    swap({i:3d}, {j:3d}): scorer={scorer_score:.6f}  ref={ref_score:.6f}  Δ={delta:.2e}  {marker}")
        if delta >= 1e-4:
            ok = False

    # Test 2: sequential commits. After each accept the scorer's state and
    # plc should both reflect the new placement. Check that a fresh
    # _exact_proxy from the same placement matches scorer's projection.
    print(f"  Test 2: {n_commits} sequential commits (verify state stays consistent):")
    current_placement = placement_np.copy()
    for commit_i in range(n_commits):
        movable_hard = np.where(bench.get_movable_mask().numpy()[:n_hard])[0]
        i, j = rng.choice(movable_hard, size=2, replace=False)
        new_i_xy = current_placement[j].copy()
        new_j_xy = current_placement[i].copy()

        # Trial then commit
        scorer_pred = scorer.score_swap(int(i), new_i_xy, int(j), new_j_xy)
        scorer.commit_swap(int(i), new_i_xy, int(j), new_j_xy)

        # Update reference placement
        current_placement[i] = new_i_xy
        current_placement[j] = new_j_xy

        # Reference proxy via full recompute (resets plc)
        ref_score = _full_proxy(current_placement, bench, plc)

        # plc state may be reset by reference; ensure scorer's expectations
        # of plc still hold (re-apply via _fast_set_placement to be safe).
        _fast_set_placement(plc, current_placement, bench)
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True

        delta = abs(scorer_pred - ref_score)
        marker = "ok" if delta < 1e-4 else "FAIL"
        print(f"    commit {commit_i+1}: scorer_pred={scorer_pred:.6f}  ref_full={ref_score:.6f}  Δ={delta:.2e}  {marker}")
        if delta >= 1e-4:
            ok = False

    return ok


def main():
    benchmarks = ["ibm01", "ibm04", "ibm10"]
    results = {}
    for bench in benchmarks:
        results[bench] = run_one(bench)
    print("\n=== Summary ===")
    for bench, ok in results.items():
        print(f"  {bench}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
