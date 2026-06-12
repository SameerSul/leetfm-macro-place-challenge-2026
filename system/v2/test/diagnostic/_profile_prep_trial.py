"""A/B the S1 prep+trial pattern against the legacy score_move_soft method in
the realistic same-macro / nearby-target pattern that the relocation passes
use. score_move_soft per-call hasn't changed; the prep/trial pattern amortizes
the subtract-old across n_targets candidates.

    uv run python system/v2/test/diagnostic/_profile_prep_trial.py
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
    cellw, cellh = cw / bm.grid_cols, ch / bm.grid_rows

    # Pre-generate candidate targets per macro (so both A and B test the same set)
    def gen_targets(k):
        return [
            (float(np.clip(soft_x[int(k)] + rng.uniform(-3, 3) * cellw, 0, cw)),
             float(np.clip(soft_y[int(k)] + rng.uniform(-3, 3) * cellh, 0, ch)))
            for _ in range(n_targets)
        ]

    targets_per_macro = [gen_targets(k) for k in hot]

    # A) Legacy score_move_soft: per-call subtract-old + add-new
    sc_a = IncrementalScorer(plc, bm, pl_np)
    t0 = time.perf_counter()
    n_a = 0
    for k, tgts in zip(hot, targets_per_macro):
        for tx, ty in tgts:
            sc_a.score_move_soft(int(k), (tx, ty))
            n_a += 1
    t_a = 1e3 * (time.perf_counter() - t0) / n_a

    # B) S1 prep + trial: subtract-old once per macro
    sc_b = IncrementalScorer(plc, bm, pl_np)
    t0 = time.perf_counter()
    n_b = 0
    for k, tgts in zip(hot, targets_per_macro):
        prep = sc_b._prepare_move_soft(int(k))
        try:
            for tx, ty in tgts:
                sc_b._trial_at_soft(prep, (tx, ty))
                n_b += 1
        finally:
            sc_b._revert_prep_soft(prep)
    t_b = 1e3 * (time.perf_counter() - t0) / n_b

    print(f"{name:6s} grid={bm.grid_rows*bm.grid_cols:5d}  "
          f"legacy score_move_soft={t_a:6.3f}ms  "
          f"S1 prep+trial         ={t_b:6.3f}ms  "
          f"S1 is {100*(1-t_b/t_a):5.1f}% faster")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["ibm10", "ibm15", "ibm17"]):
        run(nm)
