"""Time ibm15 baseline scoring in isolation (clean CPU).

If t_score < 100s, EXACT_GRID_CELL_LIMIT can be raised to 2200 so ibm15
(grid=2166) goes through the cong-grad pipeline. SLOW_SCORE_THRESHOLD_S
(=100s) in placer.py acts as a runtime safety net even if our estimate is wrong.

Run:
    uv run python submissions/varrahan/v2/tests/diagnostic/_ibm15_timing_test.py
    uv run python submissions/varrahan/v2/tests/diagnostic/_ibm15_timing_test.py ibm18  # for comparison
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPO_ROOT))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

SAMEER_DIR = REPO_ROOT / "submissions" / "sameer_v1"
sys.path.insert(0, str(SAMEER_DIR))
from placer import _will_legalize  # type: ignore  # noqa: E402

ICCAD_DIR = "/home/varrahan/Development/hackathon/external/MacroPlacement/Testcases/ICCAD04"


def run(name: str):
    print(f"\n=== {name} ===")
    bm, plc = load_benchmark_from_dir(f"{ICCAD_DIR}/{name}")
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    grid_cells = bm.grid_rows * bm.grid_cols
    print(f"  n={n}  grid={bm.grid_rows}x{bm.grid_cols}={grid_cells}")

    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    init_pos = bm.macro_positions[:n].numpy().copy().astype(np.float64)

    t0 = time.perf_counter()
    leg = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
    t_leg = time.perf_counter() - t0
    print(f"  legalize: {t_leg:.2f}s")

    pl = bm.macro_positions.clone()
    pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)

    # Three calls so we can see if there's a warm-up cost (caches, JIT, etc.).
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        out = compute_proxy_cost(pl, bm, plc)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  score #{i+1}: {dt:.2f}s  proxy={out['proxy_cost']:.4f} "
              f"(wl={out['wirelength_cost']:.4f} d={out['density_cost']:.4f} c={out['congestion_cost']:.4f})")

    print(f"  min={min(times):.2f}s  max={max(times):.2f}s  mean={sum(times)/len(times):.2f}s")
    print(f"  100s threshold: {'PASS' if max(times) < 100.0 else 'FAIL'} (max={max(times):.1f}s)")
    print(f"  budget for restarts (200s - max_score - leg): "
          f"{200.0 - max(times) - t_leg:.0f}s "
          f"=> ~{int((200.0 - max(times) - t_leg) / max(times))} restart slots")


if __name__ == "__main__":
    names = sys.argv[1:] or ["ibm15"]
    for name in names:
        try:
            run(name)
        except Exception as e:
            print(f"  ERR {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
