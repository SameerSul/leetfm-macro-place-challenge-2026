"""Verify IncrementalScorer.score_move / commit_move match _exact_proxy.

score_move is the single-macro relocation analogue of score_swap (used by the
congestion-directed relocation pass). It must equal a full _exact_proxy recompute
of the same single-macro-moved placement, and sequential commits must not drift.

    uv run python test/verification/_verify_score_move.py
"""
import sys
from pathlib import Path

_V2_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_V2_DIR / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from placer import (  # noqa: E402
    _exact_proxy, _fast_set_placement,
    _patch_plc_congestion, _patch_plc_density, _patch_plc_wirelength,
    IncrementalScorer,
)
from macro_place.loader import load_benchmark_from_dir  # noqa: E402


def run_one(bench_name, n_trials=12, n_commits=5):
    print(f"\n=== {bench_name} ===")
    bench, plc = load_benchmark_from_dir(
        f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}")
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, bench)
    _patch_plc_density(plc, bench)

    placement = bench.macro_positions.clone()
    placement_np = placement.cpu().numpy().astype(np.float64)
    n = bench.num_hard_macros
    cw, ch = bench.canvas_width, bench.canvas_height
    sizes = bench.macro_sizes[:n].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2

    _exact_proxy(placement, bench, plc)
    scorer = IncrementalScorer(plc, bench, placement_np)
    movable = np.where((bench.get_movable_mask() & bench.get_hard_macro_mask())[:n].numpy())[0]
    rng = np.random.RandomState(11)
    ok = True

    print(f"  Test 1: {n_trials} trial relocations (revert each, vs _exact_proxy):")
    for _ in range(n_trials):
        i = int(rng.choice(movable))
        # random in-bounds target center
        nx = float(rng.uniform(hw[i], cw - hw[i]))
        ny = float(rng.uniform(hh[i], ch - hh[i]))
        s = scorer.score_move(i, (nx, ny))
        ref = placement_np.copy()
        ref[i] = (nx, ny)
        r = _exact_proxy(torch.from_numpy(ref.astype(np.float32)), bench, plc)
        _fast_set_placement(plc, placement_np, bench)
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True
        d = abs(s - r)
        mark = "ok" if d < 1e-4 else "FAIL"
        print(f"    move({i:4d}): scorer={s:.6f} ref={r:.6f} Δ={d:.2e} {mark}")
        if d >= 1e-4:
            ok = False

    print(f"  Test 2: {n_commits} sequential commits (state consistency):")
    cur = placement_np.copy()
    for c in range(n_commits):
        i = int(rng.choice(movable))
        nx = float(rng.uniform(hw[i], cw - hw[i]))
        ny = float(rng.uniform(hh[i], ch - hh[i]))
        pred = scorer.score_move(i, (nx, ny))
        scorer.commit_move(i, (nx, ny))
        cur[i] = (nx, ny)
        ref = _exact_proxy(torch.from_numpy(cur.astype(np.float32)), bench, plc)
        _fast_set_placement(plc, cur, bench)
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True
        d = abs(pred - ref)
        mark = "ok" if d < 1e-4 else "FAIL"
        print(f"    commit {c+1}: pred={pred:.6f} ref={ref:.6f} Δ={d:.2e} {mark}")
        if d >= 1e-4:
            ok = False
    return ok


def main():
    res = {b: run_one(b) for b in ("ibm01", "ibm04", "ibm10")}
    print("\n=== Summary ===")
    for b, ok in res.items():
        print(f"  {b}: {'PASS' if ok else 'FAIL'}")
    if not all(res.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
