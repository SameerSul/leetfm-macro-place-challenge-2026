"""Correctness check for the soft-relocation WL-delta prefilter (`wl_delta_move_soft`).

Two properties the prefilter relies on:
  1. No side effects — calling it leaves the scorer state untouched.
  2. It equals the TRUE committed WL change — wl_delta_move_soft(k, xy) must match
     the normalized total_wl_raw delta that relocating soft k to xy actually
     produces (via _prepare_move_soft + _commit_after_prep_soft).

    uv run python system/v2/test/verification/_verify_wl_delta_move_soft.py
"""
import sys
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

TOL = 1e-9


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
    return bm, plc, pl.cpu().numpy().astype(np.float64), n


def _run(name, n_trials=12):
    bm, plc, pl_np, n = _setup(name)
    ns = bm.num_soft_macros
    rng = np.random.RandomState(0)
    sc = IncrementalScorer(plc, bm, pl_np)
    cw, ch = bm.canvas_width, bm.canvas_height
    p0 = float(sc.total_wl_raw)
    fail = 0
    print(f"\n=== {name} (n_soft={ns}) ===")
    for _ in range(n_trials):
        k = int(rng.randint(0, ns))
        old_xy = (float(sc.committed_soft_pos[k, 0]), float(sc.committed_soft_pos[k, 1]))
        new_xy = (float(rng.uniform(0, cw)), float(rng.uniform(0, ch)))

        wl_before = sc.total_wl_raw / sc.wl_normalizer
        wl_d = sc.wl_delta_move_soft(k, new_xy)
        side = abs(float(sc.total_wl_raw) - p0)   # property 1: no side effects

        prep = sc._prepare_move_soft(k)
        sc._commit_after_prep_soft(prep, new_xy)
        wl_after = sc.total_wl_raw / sc.wl_normalizer
        true_d = wl_after - wl_before
        prep2 = sc._prepare_move_soft(k)          # revert
        sc._commit_after_prep_soft(prep2, old_xy)
        p0 = float(sc.total_wl_raw)

        d = abs(wl_d - true_d)
        ok = d <= TOL and side <= TOL
        fail += 0 if ok else 1
        print(f"  move soft({k:4d}) wl_d={wl_d:+.3e} true={true_d:+.3e} "
              f"Δ={d:.1e} side={side:.1e} {'ok' if ok else 'FAIL'}")
    return fail


if __name__ == "__main__":
    total = 0
    for nm in (sys.argv[1:] or ["ibm01", "ibm10", "ibm13"]):
        total += _run(nm)
    print(f"\n=== {'PASS' if total == 0 else f'{total} FAILURES'} ===")
    sys.exit(1 if total else 0)
