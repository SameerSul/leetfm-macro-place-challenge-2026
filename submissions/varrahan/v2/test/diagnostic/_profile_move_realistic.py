"""Measure score_move_soft in the REAL relocation-pass access pattern vs the
random-k pattern, to isolate the idea-1/idea-2 benefit. The real pass tries the
SAME hot macro at many NEARBY targets consecutively (so the per-module routing-
struct cache HITS and the touched bbox stays small); the random-k profile picks a
random macro + a random far target each call (cache MISS + wide bbox), which is
the worst case for both optimizations.

    uv run python submissions/varrahan/v2/test/diagnostic/_profile_move_realistic.py ibm10 ibm15
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
sys.path.insert(0, str(V2_DIR / "src"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2_placer", str(V2_DIR / "src" / "main.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_will_legalize = _v2._will_legalize
_exact_proxy = _v2._exact_proxy
_load_plc = _v2._load_plc
IncrementalScorer = _v2.IncrementalScorer


def _setup(name):
    bm, _ = load_benchmark_from_dir(
        str(REPO_ROOT / f"external/MacroPlacement/Testcases/ICCAD04/{name}"))
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
    _exact_proxy(pl, bm, plc)
    return bm, plc, pl.cpu().numpy().astype(np.float64), n, cw, ch


def run(name, n_macros=40, n_targets=24):
    bm, plc, pl_np, n, cw, ch = _setup(name)
    ns = bm.num_soft_macros
    rng = np.random.RandomState(0)
    soft_x = pl_np[n:n + ns, 0]
    soft_y = pl_np[n:n + ns, 1]
    hot = rng.choice(ns, size=min(n_macros, ns), replace=False)
    # Nearby targets: small displacements around the macro's current position
    # (mimics relocation to a nearby cold cell).
    cellw, cellh = cw / bm.grid_cols, ch / bm.grid_rows

    # --- realistic: same macro, many nearby targets (cache HITS) ---
    sc = IncrementalScorer(plc, bm, pl_np)
    t0 = time.perf_counter()
    nm = 0
    for k in hot:
        k = int(k)
        for _ in range(n_targets):
            dx = float(rng.uniform(-3, 3) * cellw)
            dy = float(rng.uniform(-3, 3) * cellh)
            x = float(np.clip(soft_x[k] + dx, 0, cw))
            y = float(np.clip(soft_y[k] + dy, 0, ch))
            sc.score_move_soft(k, (x, y))
            nm += 1
    t_real = 1e3 * (time.perf_counter() - t0) / nm

    # --- random-k far (cache MISSES) ---
    sc2 = IncrementalScorer(plc, bm, pl_np)
    t0 = time.perf_counter()
    nm2 = 0
    for _ in range(n_macros * n_targets):
        k = int(rng.randint(0, ns))
        sc2.score_move_soft(k, (float(rng.uniform(0, cw)), float(rng.uniform(0, ch))))
        nm2 += 1
    t_rand = 1e3 * (time.perf_counter() - t0) / nm2

    print(f"{name:6s} grid={bm.grid_rows*bm.grid_cols:5d}  "
          f"realistic(same-macro,nearby)={t_real:6.3f}ms  "
          f"random(far,cache-miss)={t_rand:6.3f}ms  "
          f"realistic is {100*(1-t_real/t_rand):4.1f}% faster")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["ibm10", "ibm15"]):
        run(nm)
