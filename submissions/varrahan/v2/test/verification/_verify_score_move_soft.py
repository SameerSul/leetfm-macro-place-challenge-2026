"""Verify IncrementalScorer.score_move_soft / commit_move_soft match _exact_proxy.

Soft relocation touches WL + net-routing congestion + density but NOT macro-
routing blockage (only hard macros block) - a distinct code path from the hard
score_move. It must equal a full _exact_proxy recompute of the same soft-moved
placement, with no drift over sequential commits.

    uv run python submissions/varrahan/v2/test/verification/_verify_score_move_soft.py
"""
import sys
from pathlib import Path

_V2_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_V2_DIR))

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
    nsoft = bench.num_soft_macros
    cw, ch = bench.canvas_width, bench.canvas_height

    _exact_proxy(placement, bench, plc)
    scorer = IncrementalScorer(plc, bench, placement_np)
    rng = np.random.RandomState(13)
    ok = True

    print(f"  n_soft={nsoft}; Test 1: {n_trials} trial soft relocations:")
    for _ in range(n_trials):
        k = int(rng.randint(0, nsoft))
        nx = float(rng.uniform(0.0, cw))
        ny = float(rng.uniform(0.0, ch))
        s = scorer.score_move_soft(k, (nx, ny))
        ref = placement_np.copy()
        ref[n + k] = (nx, ny)
        r = _exact_proxy(torch.from_numpy(ref.astype(np.float32)), bench, plc)
        _fast_set_placement(plc, placement_np, bench)
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True
        d = abs(s - r)
        mark = "ok" if d < 1e-4 else "FAIL"
        print(f"    soft({k:5d}): scorer={s:.6f} ref={r:.6f} Δ={d:.2e} {mark}")
        if d >= 1e-4:
            ok = False

    print(f"  Test 2: {n_commits} sequential soft commits:")
    cur = placement_np.copy()
    for c in range(n_commits):
        k = int(rng.randint(0, nsoft))
        nx = float(rng.uniform(0.0, cw))
        ny = float(rng.uniform(0.0, ch))
        pred = scorer.score_move_soft(k, (nx, ny))
        scorer.commit_move_soft(k, (nx, ny))
        cur[n + k] = (nx, ny)
        r = _exact_proxy(torch.from_numpy(cur.astype(np.float32)), bench, plc)
        _fast_set_placement(plc, cur, bench)
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True
        d = abs(pred - r)
        mark = "ok" if d < 1e-4 else "FAIL"
        print(f"    commit {c+1}: pred={pred:.6f} ref={r:.6f} Δ={d:.2e} {mark}")
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
