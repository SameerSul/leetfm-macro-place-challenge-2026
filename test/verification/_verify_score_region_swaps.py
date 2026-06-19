"""Verify hierarchy swap scorer methods against full _exact_proxy.

uv run python test/verification/_verify_score_region_swaps.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from placer import (  # noqa: E402
    IncrementalScorer,
    _exact_proxy,
    _fast_set_placement,
    _patch_plc_congestion,
    _patch_plc_density,
    _patch_plc_wirelength,
)


def _setup(bench_name):
    bench, plc = load_benchmark_from_dir(
        str(ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench_name)
    )
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, bench)
    _patch_plc_density(plc, bench)
    placement = bench.macro_positions.clone()
    placement_np = placement.cpu().numpy().astype(np.float64)
    _exact_proxy(placement, bench, plc)
    return bench, plc, placement_np


def _check_trial(bench, plc, scorer, cur, kind, a, b):
    n = bench.num_hard_macros
    ref = cur.copy()
    if kind == "hh":
        pred = scorer.score_swap_hard_hard(a, b)
        ref[[a, b]] = ref[[b, a]]
        label = f"hh({a},{b})"
    elif kind == "ss":
        pred = scorer.score_swap_soft_soft(a, b)
        ref[[n + a, n + b]] = ref[[n + b, n + a]]
        label = f"ss({a},{b})"
    else:
        pred = scorer.score_swap_hard_soft(a, b)
        ref[a], ref[n + b] = ref[n + b].copy(), ref[a].copy()
        label = f"hs({a},{b})"
    actual = float(_exact_proxy(torch.from_numpy(ref.astype(np.float32)), bench, plc))
    _fast_set_placement(plc, cur, bench)
    plc.FLAG_UPDATE_DENSITY = True
    plc.FLAG_UPDATE_CONGESTION = True
    return label, abs(pred - actual), pred, actual


def _commit(bench, scorer, cur, kind, a, b):
    n = bench.num_hard_macros
    if kind == "hh":
        pred = scorer.score_swap_hard_hard(a, b)
        scorer.commit_swap_hard_hard(a, b)
        cur[[a, b]] = cur[[b, a]]
        label = f"hh({a},{b})"
    elif kind == "ss":
        pred = scorer.score_swap_soft_soft(a, b)
        scorer.commit_swap_soft_soft(a, b)
        cur[[n + a, n + b]] = cur[[n + b, n + a]]
        label = f"ss({a},{b})"
    else:
        pred = scorer.score_swap_hard_soft(a, b)
        scorer.commit_swap_hard_soft(a, b)
        cur[a], cur[n + b] = cur[n + b].copy(), cur[a].copy()
        label = f"hs({a},{b})"
    return label, pred


def run_one(bench_name, n_trials=8, n_commits=4):
    print(f"\n=== {bench_name} ===")
    bench, plc, placement_np = _setup(bench_name)
    n = bench.num_hard_macros
    ns = bench.num_soft_macros
    movable = np.where((bench.get_movable_mask() & bench.get_hard_macro_mask())[:n].numpy())[0]
    soft_movable = np.arange(ns)
    sm = bench.get_movable_mask().numpy()[n : n + ns]
    if sm.size:
        soft_movable = soft_movable[sm]
    rng = np.random.RandomState(23)
    ok = True
    scorer = IncrementalScorer(plc, bench, placement_np)

    cases = []
    for _ in range(n_trials):
        if movable.size >= 2:
            a, b = rng.choice(movable, size=2, replace=False)
            cases.append(("hh", int(a), int(b)))
        if soft_movable.size >= 2:
            a, b = rng.choice(soft_movable, size=2, replace=False)
            cases.append(("ss", int(a), int(b)))
        if movable.size and soft_movable.size:
            cases.append(("hs", int(rng.choice(movable)), int(rng.choice(soft_movable))))

    print("  Trial swaps:")
    for kind, a, b in cases:
        label, delta, pred, actual = _check_trial(bench, plc, scorer, placement_np, kind, a, b)
        mark = "ok" if delta < 1e-4 else "FAIL"
        print(f"    {label}: scorer={pred:.6f} ref={actual:.6f} delta={delta:.2e} {mark}")
        ok = ok and delta < 1e-4

    print("  Sequential commits:")
    cur = placement_np.copy()
    scorer = IncrementalScorer(plc, bench, cur)
    commit_cases = cases[:n_commits]
    for kind, a, b in commit_cases:
        label, pred = _commit(bench, scorer, cur, kind, a, b)
        actual = float(_exact_proxy(torch.from_numpy(cur.astype(np.float32)), bench, plc))
        _fast_set_placement(plc, cur, bench)
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True
        delta = abs(pred - actual)
        mark = "ok" if delta < 1e-4 else "FAIL"
        print(f"    {label}: scorer={pred:.6f} ref={actual:.6f} delta={delta:.2e} {mark}")
        ok = ok and delta < 1e-4
    return ok


def main():
    benches = sys.argv[1:] or ["ibm01", "ibm04", "ibm10"]
    results = {b: run_one(b) for b in benches}
    print("\n=== Summary ===")
    for b, ok in results.items():
        print(f"  {b}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
