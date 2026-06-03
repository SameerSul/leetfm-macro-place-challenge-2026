"""Bit-exactness check for HS3's score_cycle_hard_soft_soft /
commit_cycle_hard_soft_soft against the full _exact_proxy.

    uv run python submissions/varrahan/v2/test/verification/_verify_score_cycle_hard_soft_soft.py
"""
import sys
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

    # Test 1: trial cycle scores match _exact_proxy of the modified placement.
    print(f"  Test 1: {n_trials} trial hard-soft-soft 3-cycles")
    sc = IncrementalScorer(plc, bm, pl_np)
    fail = 0
    for _ in range(n_trials):
        i_h = int(rng.randint(0, n))
        # Pick 2 distinct softs.
        k1, k2 = rng.choice(ns, size=2, replace=False)
        k1, k2 = int(k1), int(k2)
        old_hx = float(pl_np[i_h, 0]); old_hy = float(pl_np[i_h, 1])
        old_s1x = float(pl_np[n + k1, 0]); old_s1y = float(pl_np[n + k1, 1])
        old_s2x = float(pl_np[n + k2, 0]); old_s2y = float(pl_np[n + k2, 1])
        # Cycle H → S1 → S2 → H:
        #   H takes S1's old pos, S1 takes S2's old pos, S2 takes H's old pos.
        new_hard_xy = (old_s1x, old_s1y)
        new_k1_xy = (old_s2x, old_s2y)
        new_k2_xy = (old_hx, old_hy)
        got = sc.score_cycle_hard_soft_soft(
            i_h, new_hard_xy, k1, new_k1_xy, k2, new_k2_xy
        )
        # Full reference: build placement with the cycle applied, score it.
        pl_cyc = pl_torch.clone()
        pl_cyc[i_h, 0] = old_s1x;     pl_cyc[i_h, 1] = old_s1y
        pl_cyc[n + k1, 0] = old_s2x;  pl_cyc[n + k1, 1] = old_s2y
        pl_cyc[n + k2, 0] = old_hx;   pl_cyc[n + k2, 1] = old_hy
        ref = float(_exact_proxy(pl_cyc, bm, plc))
        # Restore plc state for next iteration.
        _exact_proxy(pl_torch, bm, plc)
        d = abs(got - ref)
        status = "ok" if d <= TOL else "FAIL"
        if d > TOL:
            fail += 1
        print(f"    cycle hard({i_h:4d}) → soft({k1:4d}) → soft({k2:4d}) "
              f"got={got:.6f} ref={ref:.6f} Δ={d:.2e} {status}")

    # Test 2: sequential commits leave scorer state consistent.
    print(f"  Test 2: 3 sequential hard-soft-soft commits")
    sc = IncrementalScorer(plc, bm, pl_np)
    pl_running = pl_np.copy()
    for ic in range(3):
        i_h = int(rng.randint(0, n))
        k1, k2 = rng.choice(ns, size=2, replace=False)
        k1, k2 = int(k1), int(k2)
        old_hx = float(pl_running[i_h, 0]); old_hy = float(pl_running[i_h, 1])
        old_s1x = float(pl_running[n + k1, 0]); old_s1y = float(pl_running[n + k1, 1])
        old_s2x = float(pl_running[n + k2, 0]); old_s2y = float(pl_running[n + k2, 1])
        new_hard_xy = (old_s1x, old_s1y)
        new_k1_xy = (old_s2x, old_s2y)
        new_k2_xy = (old_hx, old_hy)
        pred = sc.score_cycle_hard_soft_soft(
            i_h, new_hard_xy, k1, new_k1_xy, k2, new_k2_xy
        )
        sc.commit_cycle_hard_soft_soft(
            i_h, new_hard_xy, k1, new_k1_xy, k2, new_k2_xy
        )
        pl_running[i_h, 0] = old_s1x;     pl_running[i_h, 1] = old_s1y
        pl_running[n + k1, 0] = old_s2x;  pl_running[n + k1, 1] = old_s2y
        pl_running[n + k2, 0] = old_hx;   pl_running[n + k2, 1] = old_hy
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
