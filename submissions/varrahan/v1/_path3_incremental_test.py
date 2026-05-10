"""Path 3 test: how fast is plc.get_cost() when use_incremental_cost is enabled?

If incremental rescoring is fast enough on large benchmarks (ibm10/15/18), we can
ditch the surrogate entirely and run SA directly against the real evaluator.

Run:
    uv run python submissions/varrahan/v1/_path3_incremental_test.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from macro_place.loader import load_benchmark_from_dir


ICCAD_DIR = "/home/varrahan/Development/hackathon/external/MacroPlacement/Testcases/ICCAD04"


def _displace_macros(plc, indices, dx_dy):
    """Apply (dx, dy) shift to each plc index in `indices`."""
    for idx, (dx, dy) in zip(indices, dx_dy):
        x, y = plc.get_node_location(idx)
        plc.update_node_coords(idx, x + dx, y + dy)


def _proxy(plc):
    wl = plc.get_cost()
    d = plc.get_density_cost()
    c = plc.get_congestion_cost()
    return wl + 0.5 * d + 0.5 * c, wl, d, c


def _time(label, fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    dt = time.perf_counter() - t0
    if isinstance(out, tuple) and len(out) == 4:
        proxy, wl, d, c = out
        print(f"  {label:<52s} {dt:7.2f}s  proxy={proxy:.4f} wl={wl:.4f} d={d:.4f} c={c:.4f}")
    else:
        print(f"  {label:<52s} {dt:7.2f}s")
    return dt, out


def run(benchmark_name: str):
    print(f"\n=== {benchmark_name} ===")
    bm, plc = load_benchmark_from_dir(f"{ICCAD_DIR}/{benchmark_name}")
    print(f"  n_macros={bm.num_macros}  hard={bm.num_hard_macros}  soft={bm.num_macros - bm.num_hard_macros}")

    n_total = len(plc.modules_w_pins)
    movable = list(plc.hard_macro_indices) + list(plc.soft_macro_indices)
    movable = [i for i in movable if not plc.is_node_fixed(i)]
    print(f"  movable={len(movable)} of {n_total}")

    # Cold first call (without incremental)
    _time("cold proxy (incremental=OFF)", lambda: _proxy(plc))
    _time("warm proxy (incremental=OFF, no change)", lambda: _proxy(plc))

    # Enable incremental
    plc.set_use_incremental_cost(True)
    _time("first proxy (incremental=ON, no change since last)", lambda: _proxy(plc))
    _time("rescore no-change (incremental=ON)", lambda: _proxy(plc))

    rng = np.random.default_rng(0)

    # Real 1-macro shift
    idx = movable[0]
    x, y = plc.get_node_location(idx)
    plc.update_node_coords(idx, x + 5.0, y + 5.0)
    _time("rescore after 1-macro real shift", lambda: _proxy(plc))

    # 10 macros
    if len(movable) >= 10:
        picks = rng.choice(movable, size=10, replace=False)
        for i in picks:
            x, y = plc.get_node_location(int(i))
            plc.update_node_coords(int(i), x + rng.uniform(-3, 3), y + rng.uniform(-3, 3))
        _time("rescore after 10-macro shift", lambda: _proxy(plc))

    # 50 macros
    if len(movable) >= 50:
        picks = rng.choice(movable, size=50, replace=False)
        for i in picks:
            x, y = plc.get_node_location(int(i))
            plc.update_node_coords(int(i), x + rng.uniform(-3, 3), y + rng.uniform(-3, 3))
        _time("rescore after 50-macro shift", lambda: _proxy(plc))

    # All movable
    for i in movable:
        x, y = plc.get_node_location(i)
        plc.update_node_coords(i, x + rng.uniform(-1, 1), y + rng.uniform(-1, 1))
    _time(f"rescore after all-{len(movable)} shift", lambda: _proxy(plc))


if __name__ == "__main__":
    # ibm01 = small; ibm10 = first big one; ibm15/18 = pathologically slow
    import sys
    names = sys.argv[1:] or ["ibm01", "ibm10"]
    for name in names:
        try:
            run(name)
        except Exception as e:
            print(f"  ERR {name}: {type(e).__name__}: {e}")
