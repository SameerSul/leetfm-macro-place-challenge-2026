"""Verify IncrementalScorer.congestion_field matches full PLC routing fields.

uv run python test/verification/_verify_scorer_congestion_field.py
"""

import sys
from pathlib import Path

import numpy as np

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
from placer.local_search.fields import _congestion_field  # noqa: E402


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


def _full_from_scorer(scorer):
    return np.vstack([scorer.committed_hard_pos, scorer.committed_soft_pos]).astype(np.float64)


def _check(label, bench, plc, scorer, tol=1e-9):
    nr, nc = int(bench.grid_rows), int(bench.grid_cols)
    placement = _full_from_scorer(scorer)
    _fast_set_placement(plc, placement, bench)
    plc.FLAG_UPDATE_CONGESTION = True
    plc_field = _congestion_field(plc, nr, nc)
    scorer_field = _congestion_field(scorer, nr, nc)
    delta = float(np.max(np.abs(plc_field - scorer_field)))
    mark = "ok" if delta <= tol else "FAIL"
    print(f"    {label}: max_delta={delta:.3e} {mark}")
    return delta <= tol


def run_one(bench_name):
    print(f"\n=== {bench_name} ===")
    bench, plc, placement_np = _setup(bench_name)
    scorer = IncrementalScorer(plc, bench, placement_np)
    ok = _check("initial", bench, plc, scorer)

    n = bench.num_hard_macros
    movable = np.where((bench.get_movable_mask() & bench.get_hard_macro_mask())[:n].numpy())[0]
    if movable.size:
        i = int(movable[0])
        hw = float(bench.macro_sizes[i, 0]) * 0.5
        hh = float(bench.macro_sizes[i, 1]) * 0.5
        cell_w = float(bench.canvas_width) / max(int(bench.grid_cols), 1)
        cell_h = float(bench.canvas_height) / max(int(bench.grid_rows), 1)
        x = min(max(float(placement_np[i, 0]) + 0.25 * cell_w, hw), bench.canvas_width - hw)
        y = min(
            max(float(placement_np[i, 1]) + 0.25 * cell_h, hh),
            bench.canvas_height - hh,
        )
        scorer.commit_move(i, (x, y))
        ok = _check("after hard commit", bench, plc, scorer) and ok

    if bench.num_soft_macros:
        k = 0
        idx = n + k
        hw = float(bench.macro_sizes[idx, 0]) * 0.5
        hh = float(bench.macro_sizes[idx, 1]) * 0.5
        cell_w = float(bench.canvas_width) / max(int(bench.grid_cols), 1)
        cell_h = float(bench.canvas_height) / max(int(bench.grid_rows), 1)
        x = min(
            max(float(placement_np[idx, 0]) + 0.25 * cell_w, hw),
            bench.canvas_width - hw,
        )
        y = min(
            max(float(placement_np[idx, 1]) + 0.25 * cell_h, hh),
            bench.canvas_height - hh,
        )
        scorer.commit_move_soft(k, (x, y))
        ok = _check("after soft commit", bench, plc, scorer) and ok

    return ok


def main():
    benches = sys.argv[1:] or ["ibm01", "ibm04", "ibm10"]
    results = {bench: run_one(bench) for bench in benches}
    print("\n=== Summary ===")
    for bench, ok in results.items():
        print(f"  {bench}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
