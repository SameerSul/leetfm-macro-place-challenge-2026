"""Measure the per-PASS fixed overhead of the R2 interleave: IncrementalScorer
.__init__ (full routing build + density scatter + full WL) and _exact_proxy
(full proxy on plc). The interleave does ~4 passes/round × ≤6 rounds, each
calling BOTH on best_pl. P5 (_profile_move.py) measured per-move cost; this
measures the fixed per-pass cost the shared-scorer refactor would eliminate.

    uv run python submissions/varrahan/v2/test/diagnostic/_profile_init.py ibm09 ibm15 ibm10
"""
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
V2_DIR = THIS.parents[2]
REPO_ROOT = THIS.parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2_DIR))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2_placer", str(V2_DIR / "placer.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_will_legalize = _v2._will_legalize
_exact_proxy = _v2._exact_proxy
_load_plc = _v2._load_plc
IncrementalScorer = _v2.IncrementalScorer


def _t(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return 1e3 * (time.perf_counter() - t0) / n  # ms/call


def run(name, n_reps=8):
    bm, _ = load_benchmark_from_dir(
        str(REPO_ROOT / f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    )
    plc = _load_plc(name, bm)
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    leg = _will_legalize(bm.macro_positions[:n].numpy().astype(np.float64),
                         movable, sizes, hw, hh, cw, ch, n)
    pl = bm.macro_positions.clone()
    pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)
    pl_np = pl.cpu().numpy().astype(np.float64)

    # Force a position change every call so plc's cost cache MISSES — this is the
    # real interleave path (each pass scores a CHANGED best_pl). A static-position
    # _exact_proxy returns the cached cost (~ms) and badly underestimates the cost.
    rng = np.random.RandomState(0)
    mov = np.where(movable)[0]
    cell_w = cw / bm.grid_cols

    def _proxy_dirty():
        pl2 = pl.clone()
        j = int(mov[rng.randint(0, mov.size)])
        pl2[j, 0] = float(np.clip(pl[j, 0].item() + cell_w, hw[j], cw - hw[j]))
        return _exact_proxy(pl2, bm, plc)

    def _init_dirty():
        p = pl_np.copy()
        j = int(mov[rng.randint(0, mov.size)])
        p[j, 0] = float(np.clip(p[j, 0] + cell_w, hw[j], cw - hw[j]))
        return IncrementalScorer(plc, bm, p)

    _proxy_dirty()  # warm caches/dispatch
    t_proxy = _t(_proxy_dirty, n_reps)
    _init_dirty()
    t_init = _t(_init_dirty, n_reps)

    ncell = bm.grid_rows * bm.grid_cols
    # The loop calls (per improving pass) base _exact_proxy + init + re-score
    # _exact_proxy; 4 passes/round. Lower-bound per round = 4*(proxy+init), and
    # up to +4*proxy for re-scores. Report both the unit costs and a 6-round
    # projection of the FIXED overhead (no per-move work).
    per_round_min = 4 * (t_proxy + t_init)
    per_round_max = 4 * (2 * t_proxy + t_init)
    print(f"{name:6s} n={n:4d} soft={bm.num_soft_macros:5d} grid={ncell:5d}  "
          f"_exact_proxy={t_proxy:7.1f}ms  init={t_init:7.1f}ms  "
          f"| fixed/round={per_round_min/1e3:5.2f}-{per_round_max/1e3:5.2f}s  "
          f"6-round={6*per_round_min/1e3:5.1f}-{6*per_round_max/1e3:5.1f}s")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["ibm09", "ibm15", "ibm10"]):
        try:
            run(nm)
        except Exception as e:
            import traceback
            print(f"{nm}: ERROR {e}")
            traceback.print_exc()
