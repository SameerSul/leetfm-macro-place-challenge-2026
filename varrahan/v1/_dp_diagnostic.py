"""Score DREAMPlace outputs across benchmarks with WL/D/C decomposition.

For each benchmark with a saved DP output in /tmp/dreamplace_v1/{name}/results,
loads the .gp.pl, legalizes the hard placement, and scores with full
component breakdown. Compares to:
  (a) baseline legalize (initial.plc hards)
  (b) recorded v15 best from --all log

Goal: identify whether DP's loss pattern is dominated by congestion (the
non-WL component DP can't see), density, or wirelength.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from placer import _will_legalize, _load_plc


def score(pl, b, plc):
    from macro_place.objective import compute_proxy_cost
    return compute_proxy_cost(pl, b, plc)


def evaluate_dp(bname, v15_best, v15_components=None):
    from macro_place.loader import load_benchmark_from_dir
    from dreamplace_bridge.bookshelf_to_pb import read_dreamplace_positions_full

    bdir = Path("external/MacroPlacement/Testcases/ICCAD04") / bname
    scratch = Path(f"/tmp/dreamplace_v1/{bname}")
    if not bdir.exists() or not scratch.exists():
        print(f"  [{bname}] missing data, skipping")
        return None

    b, plc = load_benchmark_from_dir(bdir.as_posix())
    n_hard, n_soft = b.num_hard_macros, b.num_soft_macros
    init = b.macro_positions.numpy().astype(np.float64).copy()
    sizes = b.macro_sizes.numpy().astype(np.float64)
    cw, ch = b.canvas_width, b.canvas_height
    hw = sizes[:n_hard, 0] / 2; hh = sizes[:n_hard, 1] / 2
    mov_h = (b.get_movable_mask() & b.get_hard_macro_mask()).numpy()[:n_hard]

    # (a) Baseline legalize from initial.plc - for the WL/D/C reference
    leg = _will_legalize(init[:n_hard], mov_h, sizes[:n_hard], hw, hh, cw, ch, n_hard)
    pl_base = b.macro_positions.clone()
    pl_base[:n_hard, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    pl_base[:n_hard, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)
    c_base = score(pl_base, b, plc)

    # (b) DP placement: read from .gp.pl, legalize hards, plug in soft positions
    try:
        dp_hard, dp_soft = read_dreamplace_positions_full(plc, str(scratch), bname)
    except Exception as e:
        print(f"  [{bname}] DP read failed: {type(e).__name__}: {e}")
        return None

    dp_hard_clip = dp_hard.copy()
    dp_hard_clip[:, 0] = np.clip(dp_hard_clip[:, 0], hw, cw - hw)
    dp_hard_clip[:, 1] = np.clip(dp_hard_clip[:, 1], hh, ch - hh)
    dp_leg = _will_legalize(dp_hard_clip, mov_h, sizes[:n_hard], hw, hh, cw, ch, n_hard)
    pl_dp = b.macro_positions.clone()
    pl_dp[:n_hard, 0] = torch.tensor(dp_leg[:, 0], dtype=torch.float32)
    pl_dp[:n_hard, 1] = torch.tensor(dp_leg[:, 1], dtype=torch.float32)
    n_soft_use = int(min(dp_soft.shape[0], n_soft))
    if n_soft_use > 0:
        pl_dp[n_hard:n_hard + n_soft_use, 0] = torch.tensor(dp_soft[:n_soft_use, 0], dtype=torch.float32)
        pl_dp[n_hard:n_hard + n_soft_use, 1] = torch.tensor(dp_soft[:n_soft_use, 1], dtype=torch.float32)
    c_dp = score(pl_dp, b, plc)

    return {
        "bench": bname, "v15_best": v15_best, "v15_c": v15_components,
        "base": c_base, "dp": c_dp,
    }


def main():
    targets = [
        ("ibm01", 1.1521, (0.069, 0.910, 1.256)),   # DP WINS
        ("ibm02", 1.5923, (0.080, 0.810, 2.215)),   # DP LOSES (+0.051)
        ("ibm03", 1.3603, (0.084, 0.783, 1.769)),   # DP LOSES (+0.048)
        ("ibm04", 1.3196, (0.073, 0.805, 1.689)),   # DP WINS
        ("ibm06", 1.6684, (0.065, 0.746, 2.460)),   # DP LOSES (+0.042)
        ("ibm07", 1.4950, (0.066, 0.853, 2.005)),   # DP LOSES (+0.029)
        ("ibm08", 1.5251, (0.072, 0.892, 2.015)),   # DP LOSES (+0.019)
        ("ibm09", 1.1304, (0.059, 0.890, 1.254)),   # DP LOSES (+0.004)
        ("ibm10", 1.3661, None),                     # DP WINS (baseline-only)
        ("ibm11", 1.2354, (0.054, 0.906, 1.455)),   # DP LOSES (+0.032)
        ("ibm12", 1.6506, (0.060, 0.809, 2.372)),   # DP LOSES (+0.080)
        ("ibm13", 1.4006, (0.054, 0.915, 1.778)),   # DP LOSES (+0.009)
    ]

    print(f"{'bench':>8s}  {'win?':>5s}  | {'base_WL':>8s} {'base_D':>7s} {'base_C':>7s} {'base_P':>7s}"
          f"  | {'DP_WL':>7s} {'DP_D':>6s} {'DP_C':>6s} {'DP_P':>7s}"
          f"  | {'best_WL':>8s} {'best_D':>7s} {'best_C':>7s} {'best_P':>7s}"
          f"  | {'dWL':>7s} {'dD':>7s} {'dC':>7s}")
    print("-" * 175)

    rows = []
    for bname, v15_best, v15_c in targets:
        r = evaluate_dp(bname, v15_best, v15_c)
        if r is None:
            continue
        base = r["base"]; dp = r["dp"]
        won = dp["proxy_cost"] <= r["v15_best"] + 0.0005
        if r["v15_c"]:
            bwl, bd, bc = r["v15_c"]; bp = r["v15_best"]
        else:
            bwl, bd, bc, bp = float("nan"), float("nan"), float("nan"), r["v15_best"]
        # DP vs WINNER component deltas (negative = DP better)
        dwl = dp["wirelength_cost"] - bwl if not np.isnan(bwl) else float("nan")
        dd  = dp["density_cost"]    - bd  if not np.isnan(bd)  else float("nan")
        dc  = dp["congestion_cost"] - bc  if not np.isnan(bc)  else float("nan")
        print(f"{bname:>8s}  {'YES' if won else 'no':>5s}  |"
              f" {base['wirelength_cost']:>8.4f} {base['density_cost']:>7.4f} {base['congestion_cost']:>7.4f} {base['proxy_cost']:>7.4f}"
              f"  | {dp['wirelength_cost']:>7.4f} {dp['density_cost']:>6.4f} {dp['congestion_cost']:>6.4f} {dp['proxy_cost']:>7.4f}"
              f"  | {bwl:>8.4f} {bd:>7.4f} {bc:>7.4f} {bp:>7.4f}"
              f"  | {dwl:>+7.4f} {dd:>+7.4f} {dc:>+7.4f}")
        rows.append((bname, won, dp, (bwl, bd, bc, bp)))
    return rows


if __name__ == "__main__":
    main()
