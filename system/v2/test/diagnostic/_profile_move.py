"""Profile the per-trial-move hot path (for the speedup pass). score_move /
score_move_soft are called thousands of times per relocation pass; each runs
_compute_cong_cost (full re-smooth + top-5% partition over all cells) and
_compute_density_cost (top-10% partition). This breaks down where the ms go on a
large-grid benchmark so we optimize the right thing.

    uv run python system/v2/test/diagnostic/_profile_move.py ibm15 ibm17
"""
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
V2_DIR = THIS.parents[2]
REPO_ROOT = THIS.parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2_DIR / "src"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2_placer", str(V2_DIR / "src" / "main.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_will_legalize = _v2._will_legalize
_exact_proxy = _v2._exact_proxy
_load_plc = _v2._load_plc
IncrementalScorer = _v2.IncrementalScorer
_smooth = _v2._smooth_routing_cong_vec


def _t(fn, n=200):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return 1e3 * (time.perf_counter() - t0) / n  # ms/call


def run(name, n_moves=300):
    bm, _ = load_benchmark_from_dir(str(REPO_ROOT / f"external/MacroPlacement/Testcases/ICCAD04/{name}"))
    plc = _load_plc(name, bm)
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    leg = _will_legalize(bm.macro_positions[:n].numpy().astype(np.float64), movable, sizes, hw, hh, cw, ch, n)
    pl = bm.macro_positions.clone()
    pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)
    _exact_proxy(pl, bm, plc)
    sc = IncrementalScorer(plc, bm, pl.cpu().numpy().astype(np.float64))
    ncell = sc.grid_row * sc.grid_col

    # component timings
    t_cong = _t(sc._compute_cong_cost)
    t_dens = _t(sc._compute_density_cost)
    V = sc.V_flat / sc.grid_v_routes
    t_smooth1 = _t(lambda: _smooth(V, sc.grid_row, sc.grid_col, sc.smooth_range, axis_h=False))

    # full soft score_move (end to end)
    ns = bm.num_soft_macros
    rng = np.random.RandomState(0)
    t0 = time.perf_counter()
    for _ in range(n_moves):
        k = int(rng.randint(0, ns))
        sc.score_move_soft(k, (float(rng.uniform(0, cw)), float(rng.uniform(0, ch))))
    t_full = 1e3 * (time.perf_counter() - t0) / n_moves

    print(f"{name:6s} grid={sc.grid_col}x{sc.grid_row}={ncell:5d}  "
          f"score_move_soft={t_full:6.3f}ms | cong_cost={t_cong:6.3f} "
          f"(smooth×2≈{2*t_smooth1:5.3f}) density_cost={t_dens:6.3f}ms  "
          f"[cong {100*t_cong/t_full:4.1f}%, smooth {100*2*t_smooth1/t_full:4.1f}%, "
          f"dens {100*t_dens/t_full:4.1f}%]")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["ibm15", "ibm17"]):
        try:
            run(nm)
        except Exception as e:
            import traceback; print(f"{nm}: ERROR {e}"); traceback.print_exc()
