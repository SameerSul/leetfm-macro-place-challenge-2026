"""Path 2: surrogate calibration test.

Generate N legal placements (baseline + perturbations at varying scales), score
each with both real proxy and surrogate, report rank correlation. If Spearman
correlation > ~0.6 the surrogate is useful as a *relative* ranker even if
its absolute values are off. Anything <= 0.3 means we should rewrite components.

Run:
    uv run python submissions/varrahan/v1/_calibration_test.py ibm10
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make sibling files importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

# Pull legalizer + plc loader directly from sameer_v1 (read-only allowed)
SAMEER_DIR = HERE.parent.parent / "sameer_v1"
sys.path.insert(0, str(SAMEER_DIR))
from placer import _will_legalize  # type: ignore  # noqa: E402

from surrogate import surrogate_components, surrogate_usable  # noqa: E402

ICCAD_DIR = "/home/varrahan/Development/hackathon/external/MacroPlacement/Testcases/ICCAD04"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. -1..1; 1 = same order, 0 = no relation."""
    a_rk = np.argsort(np.argsort(a))
    b_rk = np.argsort(np.argsort(b))
    a_rk = a_rk - a_rk.mean()
    b_rk = b_rk - b_rk.mean()
    denom = float(np.sqrt((a_rk ** 2).sum() * (b_rk ** 2).sum()))
    if denom == 0:
        return 0.0
    return float((a_rk * b_rk).sum() / denom)


def _real_proxy(pos: np.ndarray, benchmark, plc, n: int) -> dict:
    pl = benchmark.macro_positions.clone()
    pl[:n, 0] = torch.tensor(pos[:, 0], dtype=torch.float32)
    pl[:n, 1] = torch.tensor(pos[:, 1], dtype=torch.float32)
    out = compute_proxy_cost(pl, benchmark, plc)
    return out


def _setup(name: str):
    bm, plc = load_benchmark_from_dir(f"{ICCAD_DIR}/{name}")
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    init_pos = bm.macro_positions[:n].numpy().copy().astype(np.float64)
    return bm, plc, n, cw, ch, sizes, hw, hh, movable, init_pos


def _perturb(init_pos, movable, hw, hh, cw, ch, scale_frac, rng):
    pos = init_pos.copy()
    if scale_frac == 0.0:
        return pos
    sx = scale_frac * cw
    sy = scale_frac * ch
    for i in range(pos.shape[0]):
        if not movable[i]:
            continue
        pos[i, 0] = np.clip(pos[i, 0] + rng.normal(0, sx), hw[i], cw - hw[i])
        pos[i, 1] = np.clip(pos[i, 1] + rng.normal(0, sy), hh[i], ch - hh[i])
    return pos


def run(name: str):
    print(f"\n=== {name} ===")
    bm, plc, n, cw, ch, sizes, hw, hh, movable, init_pos = _setup(name)
    print(f"  n={n}  movable={int(movable.sum())}  cw,ch=({cw:.0f},{ch:.0f})  grid={bm.grid_rows}x{bm.grid_cols}")

    if not surrogate_usable(bm, plc=plc):
        print("  surrogate not usable (no nets) -- aborting")
        return

    rng = np.random.default_rng(0)
    # 6 candidates: baseline + 5 noise scales. Dropped redundant repeats from
    # the original 10-scale sweep so ibm11/ibm15 stay under ~30 min.
    scales = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
    rows = []
    for k, s in enumerate(scales):
        t0 = time.perf_counter()
        pos = _perturb(init_pos, movable, hw, hh, cw, ch, s, rng)
        leg = _will_legalize(pos, movable, sizes, hw, hh, cw, ch, n)
        t_leg = time.perf_counter() - t0

        t0 = time.perf_counter()
        real = _real_proxy(leg, bm, plc, n)
        t_real = time.perf_counter() - t0

        t0 = time.perf_counter()
        sur = surrogate_components(leg, bm, n, cw, ch, plc=plc)
        t_sur = time.perf_counter() - t0

        rows.append({
            "k": k, "scale": s, "real": real["proxy_cost"], "real_wl": real["wirelength_cost"],
            "real_d": real["density_cost"], "real_c": real["congestion_cost"],
            "surr_legacy": sur["score_legacy"], "surr_default": sur["score_default"],
            "sur_wl": sur["wl"], "sur_d": sur["density"], "sur_c": sur["congestion"],
            "t_leg": t_leg, "t_real": t_real, "t_sur": t_sur,
        })
        print(f"  k={k} scale={s:.3f}  REAL proxy={real['proxy_cost']:.4f} (wl={real['wirelength_cost']:.4f} "
              f"d={real['density_cost']:.4f} c={real['congestion_cost']:.4f})  "
              f"SUR_legacy={sur['score_legacy']:.4f} (wl={sur['wl']:.4f} d={sur['density']:.4f} c={sur['congestion']:.4f})  "
              f"t={t_leg:.1f}/{t_real:.1f}/{t_sur:.2f}s")

    real_arr = np.array([r["real"] for r in rows])
    surr_arr = np.array([r["surr_legacy"] for r in rows])
    # Variants: which weighting of surrogate components ranks real proxy best?
    sur_wl = np.array([r["sur_wl"] for r in rows])
    sur_d = np.array([r["sur_d"] for r in rows])
    sur_c = np.array([r["sur_c"] for r in rows])
    surr_no_d = 1.0 * sur_wl + 0.5 * sur_c
    surr_c_only = sur_c
    surr_w1c1 = sur_wl + sur_c

    print(f"\n  Spearman rank corr (overall, current 1*wl+0.5*d+0.5*c):  {_spearman(real_arr, surr_arr):+.3f}")
    print(f"  Spearman rank corr (no density,    1*wl+0.5*c):          {_spearman(real_arr, surr_no_d):+.3f}")
    print(f"  Spearman rank corr (congestion only):                    {_spearman(real_arr, surr_c_only):+.3f}")
    print(f"  Spearman rank corr (1*wl+1*c):                           {_spearman(real_arr, surr_w1c1):+.3f}")
    print(f"  Spearman rank corr (WL):         {_spearman(np.array([r['real_wl'] for r in rows]), np.array([r['sur_wl'] for r in rows])):+.3f}")
    print(f"  Spearman rank corr (density):    {_spearman(np.array([r['real_d'] for r in rows]), np.array([r['sur_d'] for r in rows])):+.3f}")
    print(f"  Spearman rank corr (congestion): {_spearman(np.array([r['real_c'] for r in rows]), np.array([r['sur_c'] for r in rows])):+.3f}")

    # Component-vs-overall: which surrogate component best predicts real proxy?
    print(f"\n  Spearman SUR_WL vs REAL proxy:   {_spearman(np.array([r['sur_wl'] for r in rows]), real_arr):+.3f}")
    print(f"  Spearman SUR_D  vs REAL proxy:   {_spearman(np.array([r['sur_d'] for r in rows]), real_arr):+.3f}")
    print(f"  Spearman SUR_C  vs REAL proxy:   {_spearman(np.array([r['sur_c'] for r in rows]), real_arr):+.3f}")

    print(f"\n  Best (real):           #{int(np.argmin(real_arr))} proxy={real_arr.min():.4f}")
    print(f"  Best (surr legacy):    #{int(np.argmin(surr_arr))} surr={surr_arr.min():.4f}  "
          f"-> real proxy at that pick = {real_arr[int(np.argmin(surr_arr))]:.4f}")
    print(f"  Best (surr default):   #{int(np.argmin(sur_wl))} sur_wl={sur_wl.min():.4f}  "
          f"-> real proxy at that pick = {real_arr[int(np.argmin(sur_wl))]:.4f}")
    print(f"  Worst (real):  #{int(np.argmax(real_arr))} proxy={real_arr.max():.4f}")
    print(f"  Range (real):  {real_arr.max() - real_arr.min():.4f}")
    print(f"  Range (surr):  {surr_arr.max() - surr_arr.min():.4f}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["ibm10"]
    for n in names:
        try:
            run(n)
        except Exception as e:
            print(f"  ERR {n}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
