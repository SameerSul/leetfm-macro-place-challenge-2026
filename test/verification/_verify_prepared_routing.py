"""Verify prepared Numba routing structs against the NumPy reference path.

Usage:
  uv run python test/verification/_verify_prepared_routing.py ibm10
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from placer import (  # noqa: E402
    _fast_set_placement,
    _patch_plc_congestion,
    _patch_plc_density,
    _patch_plc_wirelength,
)
from placer.routing import apply as routing_apply  # noqa: E402


def _setup(bench_name):
    bench, plc = load_benchmark_from_dir(
        str(ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench_name)
    )
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, bench)
    _patch_plc_density(plc, bench)
    placement = bench.macro_positions.cpu().numpy().astype(np.float64)
    _fast_set_placement(plc, placement, bench)
    plc.get_congestion_cost()
    return bench, plc, placement


def _apply(plc, struct, use_numba, weight):
    n_cells = int(plc.grid_row * plc.grid_col)
    horizontal = np.zeros(n_cells, dtype=np.float64)
    vertical = np.zeros(n_cells, dtype=np.float64)
    original = routing_apply.HAS_NUMBA
    routing_apply.HAS_NUMBA = use_numba
    try:
        bbox = routing_apply._apply_net_routing_struct(plc, struct, weight, horizontal, vertical)
    finally:
        routing_apply.HAS_NUMBA = original
    return horizontal, vertical, bbox


def run(bench_name):
    bench, plc, placement = _setup(bench_name)
    n_nets = int(plc._wl_vec_cache["n_nets"])
    rng = np.random.RandomState(41)
    subsets = [
        np.arange(n_nets, dtype=np.int64),
        np.sort(rng.choice(n_nets, size=min(300, n_nets), replace=False)).astype(np.int64),
    ]

    ok = True
    for moved in (False, True):
        if moved:
            trial = placement.copy()
            movable = np.where(bench.get_movable_mask().numpy())[0]
            chosen = rng.choice(movable, size=min(40, movable.size), replace=False)
            trial[chosen] += rng.uniform(-5.0, 5.0, size=(chosen.size, 2))
            trial[:, 0] = np.clip(trial[:, 0], 0.0, float(plc.width))
            trial[:, 1] = np.clip(trial[:, 1], 0.0, float(plc.height))
            _fast_set_placement(plc, trial, bench)

        for subset_idx, net_indices in enumerate(subsets):
            struct = routing_apply._build_net_routing_struct(plc, net_indices)
            for weight in (-1.0, 1.0):
                h_jit, v_jit, bbox_jit = _apply(plc, struct, True, weight)
                h_ref, v_ref, bbox_ref = _apply(plc, struct, False, weight)
                delta_h = float(np.max(np.abs(h_jit - h_ref), initial=0.0))
                delta_v = float(np.max(np.abs(v_jit - v_ref), initial=0.0))
                passed = delta_h < 1e-12 and delta_v < 1e-12 and bbox_jit == bbox_ref
                print(
                    f"  moved={int(moved)} subset={subset_idx} weight={weight:+.0f}: "
                    f"dH={delta_h:.2e} dV={delta_v:.2e} bbox={bbox_jit} "
                    f"{'PASS' if passed else 'FAIL'}"
                )
                ok = ok and passed
    return ok


if __name__ == "__main__":
    benchmark_name = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    if not run(benchmark_name):
        raise SystemExit(1)
