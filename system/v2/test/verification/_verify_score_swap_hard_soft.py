"""Bit-exactness check for HXS's score_swap_hard_soft /
commit_swap_hard_soft against the full _exact_proxy. Mirrors
_verify_score_swap_soft.py.

    uv run python system/v2/test/verification/_verify_score_swap_hard_soft.py
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

TOL = 1e-8


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
    return bm, plc, pl, pl.cpu().numpy().astype(np.float64), n, cw, ch


def _run(name, n_trials=10):
    bm, plc, pl_torch, pl_np, n, cw, ch = _setup(name)
    ns = bm.num_soft_macros
    rng = np.random.RandomState(0)
    print(f"\n=== {name} ===")

    # Test 1: trial cross-swap scores match _exact_proxy.
    print(f"  Test 1: {n_trials} trial hard-soft cross-swaps")
    sc = IncrementalScorer(plc, bm, pl_np)
    fail = 0
    for _ in range(n_trials):
        i_h = int(rng.randint(0, n))
        k_s = int(rng.randint(0, ns))
        # Swap: exchange positions
        old_hx = float(pl_np[i_h, 0]); old_hy = float(pl_np[i_h, 1])
        old_sx = float(pl_np[n + k_s, 0]); old_sy = float(pl_np[n + k_s, 1])
        new_hard_xy = (old_sx, old_sy)
        new_soft_xy = (old_hx, old_hy)
        got = sc.score_swap_hard_soft(i_h, new_hard_xy, k_s, new_soft_xy)
        pl_swapped = pl_torch.clone()
        pl_swapped[i_h, 0] = old_sx; pl_swapped[i_h, 1] = old_sy
        pl_swapped[n + k_s, 0] = old_hx; pl_swapped[n + k_s, 1] = old_hy
        ref = float(_exact_proxy(pl_swapped, bm, plc))
        _exact_proxy(pl_torch, bm, plc)
        d = abs(got - ref)
        status = "ok" if d <= TOL else "FAIL"
        if d > TOL:
            fail += 1
        print(f"    swap hard({i_h:4d}) <-> soft({k_s:4d}) got={got:.6f} "
              f"ref={ref:.6f} Δ={d:.2e} {status}")

    # Test 2: sequential commits leave the scorer's state consistent.
    print("  Test 2: 3 sequential hard-soft commits")
    sc = IncrementalScorer(plc, bm, pl_np)
    pl_running = pl_np.copy()
    for ic in range(3):
        i_h = int(rng.randint(0, n))
        k_s = int(rng.randint(0, ns))
        old_hx = float(pl_running[i_h, 0]); old_hy = float(pl_running[i_h, 1])
        old_sx = float(pl_running[n + k_s, 0]); old_sy = float(pl_running[n + k_s, 1])
        new_hard_xy = (old_sx, old_sy)
        new_soft_xy = (old_hx, old_hy)
        pred = sc.score_swap_hard_soft(i_h, new_hard_xy, k_s, new_soft_xy)
        sc.commit_swap_hard_soft(i_h, new_hard_xy, k_s, new_soft_xy)
        pl_running[i_h, 0] = old_sx; pl_running[i_h, 1] = old_sy
        pl_running[n + k_s, 0] = old_hx; pl_running[n + k_s, 1] = old_hy
        pl_t = pl_torch.clone()
        pl_t[:n, 0] = torch.tensor(pl_running[:n, 0], dtype=torch.float32)
        pl_t[:n, 1] = torch.tensor(pl_running[:n, 1], dtype=torch.float32)
        pl_t[n:n + ns, 0] = torch.tensor(pl_running[n:n + ns, 0], dtype=torch.float32)
        pl_t[n:n + ns, 1] = torch.tensor(pl_running[n:n + ns, 1], dtype=torch.float32)
        ref = float(_exact_proxy(pl_t, bm, plc))
        d = abs(pred - ref)
        status = "ok" if d <= TOL else "FAIL"
        if d > TOL:
            fail += 1
        print(f"    commit {ic+1}: pred={pred:.6f} ref={ref:.6f} Δ={d:.2e} {status}")
    return fail


if __name__ == "__main__":
    total_fail = 0
    for nm in (sys.argv[1:] or ["ibm01", "ibm04", "ibm10"]):
        total_fail += _run(nm)
    print(f"\n=== Summary: {'PASS' if total_fail == 0 else f'{total_fail} FAILURES'} ===")
    sys.exit(1 if total_fail else 0)
