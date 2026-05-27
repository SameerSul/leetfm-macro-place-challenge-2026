"""Profile what fraction of a 2-opt score_swap is the (full-recompute) density.

WL and congestion are already incremental in IncrementalScorer; density is the
only full recompute (plc.get_density_cost over ALL soft+hard macros). This tells
us whether making density incremental (P3) is worth it on small vs large
benchmarks.

    uv run python submissions/varrahan/v2/test/diagnostic/_profile_density.py ibm04 ibm10
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

from macro_place.loader import load_benchmark_from_dir  # type: ignore  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2_placer", str(V2_DIR / "placer.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_will_legalize = _v2._will_legalize
_exact_proxy = _v2._exact_proxy
IncrementalScorer = _v2.IncrementalScorer

ICCAD_DIR = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


def run(name: str, n_trials: int = 300):
    bm, plc = load_benchmark_from_dir(str(ICCAD_DIR / name))
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    init = bm.macro_positions[:n].numpy().copy().astype(np.float64)
    leg = _will_legalize(init, movable, sizes, hw, hh, cw, ch, n)
    pl = bm.macro_positions.clone()
    pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)

    _exact_proxy(pl, bm, plc)  # warm the patches + caches
    scorer = IncrementalScorer(plc, bm, pl.cpu().numpy().astype(np.float64))

    mov_idx = np.where(movable)[0]
    rng = np.random.RandomState(0)

    # Time a full score_swap, and the density-only portion within it.
    t_swap = 0.0
    t_dens = 0.0
    done = 0
    for _ in range(n_trials):
        i, j = rng.choice(mov_idx, size=2, replace=False)
        xi = (float(scorer.committed_hard_pos[j, 0]), float(scorer.committed_hard_pos[j, 1]))
        xj = (float(scorer.committed_hard_pos[i, 0]), float(scorer.committed_hard_pos[i, 1]))
        t0 = time.perf_counter()
        scorer.score_swap(int(i), xi, int(j), xj)
        t_swap += time.perf_counter() - t0
        # Density alone: force recompute, time it.
        plc.FLAG_UPDATE_DENSITY = True
        t1 = time.perf_counter()
        plc.get_density_cost()
        t_dens += time.perf_counter() - t1
        done += 1

    n_soft = bm.num_soft_macros
    print(f"{name:7s} n_hard={n:4d} n_soft={n_soft:5d} n_mod={n+n_soft:5d}  "
          f"score_swap={1e3*t_swap/done:6.3f}ms  density={1e3*t_dens/done:6.3f}ms  "
          f"({100*t_dens/t_swap:4.1f}% of swap)")


if __name__ == "__main__":
    names = sys.argv[1:] or ["ibm04", "ibm10"]
    for nm in names:
        try:
            run(nm)
        except Exception as e:
            import traceback
            print(f"{nm}: ERROR {e}")
            traceback.print_exc()
