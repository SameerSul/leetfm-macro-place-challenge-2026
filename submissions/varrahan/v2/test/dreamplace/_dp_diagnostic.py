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

HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[2]
REPO_ROOT = HERE.parents[5]
for p in (str(REPO_ROOT), str(V2_DIR / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from placer import (
    _will_legalize,
    _load_plc,
    _patch_plc_wirelength,
    _patch_plc_density,
    _patch_plc_congestion,
    _fast_set_placement,
)


def _ensure_patches(plc, b):
    """Apply the B3-vectorized scoring patches so this script runs fast.

    Without these, each plc.get_*_cost call falls back to the unpatched
    Python loop path (10s+ per call on ibm15+) — the diagnostic timed out
    at 15 min producing zero rows.
    """
    _patch_plc_wirelength(plc)
    _patch_plc_density(plc, b)
    _patch_plc_congestion(plc, b)


def score(pl, b, plc):
    """Decomposed proxy score using the patched (fast) plc methods.

    Returns wirelength_cost / density_cost / congestion_cost / proxy_cost
    in the same dict shape as macro_place.objective.compute_proxy_cost,
    but skips the expensive O(n²) overlap-metric computation that the
    diagnostic doesn't actually need.
    """
    placement_np = pl.cpu().numpy().astype(np.float64)
    _fast_set_placement(plc, placement_np, b)
    wl = float(plc.get_cost())
    dens = float(plc.get_density_cost())
    cong = float(plc.get_congestion_cost())
    return {
        "wirelength_cost": wl,
        "density_cost": dens,
        "congestion_cost": cong,
        "proxy_cost": wl + 0.5 * dens + 0.5 * cong,
    }


def evaluate_dp(bname, v15_best, v15_components=None, tag="hi"):
    """Evaluate DP output for one (benchmark, target_density tag) combo.

    tag: "hi" → /tmp/dreamplace_v1_hi (target_density=0.85)
         "lo" → /tmp/dreamplace_v1_lo (target_density=0.65)
    """
    from macro_place.loader import load_benchmark_from_dir
    from dreamplace_bridge.bookshelf_to_pb import read_dreamplace_positions_full

    bdir = Path("external/MacroPlacement/Testcases/ICCAD04") / bname
    # Multi-DP layout (2026-05-23): the placer now launches at two
    # target_density values into separate scratch roots. Pick the requested
    # one for this evaluation.
    scratch = Path(f"/tmp/dreamplace_v1_{tag}/{bname}")
    if not bdir.exists() or not scratch.exists():
        print(f"  [{bname}/{tag}] missing data ({scratch}), skipping")
        return None

    b, plc = load_benchmark_from_dir(bdir.as_posix())
    _ensure_patches(plc, b)
    n_hard, n_soft = b.num_hard_macros, b.num_soft_macros
    init = b.macro_positions.numpy().astype(np.float64).copy()
    sizes = b.macro_sizes.numpy().astype(np.float64)
    cw, ch = b.canvas_width, b.canvas_height
    hw = sizes[:n_hard, 0] / 2; hh = sizes[:n_hard, 1] / 2
    mov_h = (b.get_movable_mask() & b.get_hard_macro_mask()).numpy()[:n_hard]

    # (a) Baseline legalize from initial.plc — for the WL/D/C reference
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
    # Current best per-benchmark (post-B3p3 --all, 2026-05-23). Updated from
    # the v15 entries below since the session's wins changed every benchmark.
    # Components (WL, D, C) where known come from the post-B3p3 --all output.
    targets = [
        ("ibm01", 1.1352, (0.068, 0.891, 1.243)),   # was 1.1521 in v15
        ("ibm02", 1.5712, (0.080, 0.797, 2.186)),   # was 1.5923
        ("ibm03", 1.3531, (0.084, 0.786, 1.753)),   # was 1.3603
        ("ibm04", 1.2969, (0.076, 0.840, 1.601)),   # was 1.3196
        ("ibm06", 1.6744, (0.064, 0.761, 2.460)),   # was 1.6684 (slight regr)
        ("ibm07", 1.4855, (0.066, 0.848, 1.991)),   # was 1.4950
        ("ibm08", 1.5141, (0.071, 0.880, 2.006)),   # was 1.5251
        ("ibm09", 1.1037, (0.060, 0.871, 1.217)),   # was 1.1304
        ("ibm10", 1.3728, (0.060, 0.788, 1.851)),   # was 1.3661 (use_exact pipeline now)
        ("ibm11", 1.2283, (0.055, 0.900, 1.447)),   # was 1.2354
        ("ibm12", 1.6472, (0.060, 0.809, 2.367)),   # was 1.6506
        ("ibm13", 1.3884, (0.056, 0.889, 1.779)),   # was 1.4006
        ("ibm14", 1.5891, (0.053, 0.970, 2.103)),
        ("ibm15", 1.6040, (0.059, 0.936, 2.158)),
        ("ibm16", 1.5164, (0.050, 0.857, 2.088)),
        ("ibm17", 1.7408, (0.054, 0.952, 2.427)),
        ("ibm18", 1.7869, (0.053, 1.043, 2.438)),
    ]

    header = (
        f"{'bench':>8s}  {'tag':>3s}  {'win?':>5s}  | "
        f"{'base_WL':>8s} {'base_D':>7s} {'base_C':>7s} {'base_P':>7s}  | "
        f"{'DP_WL':>7s} {'DP_D':>6s} {'DP_C':>6s} {'DP_P':>7s}  | "
        f"{'best_WL':>8s} {'best_D':>7s} {'best_C':>7s} {'best_P':>7s}  | "
        f"{'dWL':>7s} {'dD':>7s} {'dC':>7s}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    rows = []
    for bname, v15_best, v15_c in targets:
        for tag in ("hi", "lo"):
            r = evaluate_dp(bname, v15_best, v15_c, tag=tag)
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
            print(f"{bname:>8s}  {tag:>3s}  {'YES' if won else 'no':>5s}  |"
                  f" {base['wirelength_cost']:>8.4f} {base['density_cost']:>7.4f} {base['congestion_cost']:>7.4f} {base['proxy_cost']:>7.4f}"
                  f"  | {dp['wirelength_cost']:>7.4f} {dp['density_cost']:>6.4f} {dp['congestion_cost']:>6.4f} {dp['proxy_cost']:>7.4f}"
                  f"  | {bwl:>8.4f} {bd:>7.4f} {bc:>7.4f} {bp:>7.4f}"
                  f"  | {dwl:>+7.4f} {dd:>+7.4f} {dc:>+7.4f}",
                  flush=True)
            rows.append((bname, tag, won, dp, (bwl, bd, bc, bp)))
    return rows


if __name__ == "__main__":
    main()
