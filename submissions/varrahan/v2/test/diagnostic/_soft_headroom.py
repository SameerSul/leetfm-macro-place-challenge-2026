"""O3 diagnostic: how much proxy is recoverable by repositioning SOFT macros?

Soft macros are stand-ins for std-cell clusters; the v2 pipeline leaves them
at their initial.plc positions while hard macros move. Two competing effects:
  - Following net centroids REDUCES wirelength.
  - Clustering softs near centroids SPIKES density + congestion (which dominate
    proxy ~30x). The 2026-05-22 centroid re-snap at blend=1.0 was a disaster
    (ibm04 1.3079 -> 1.6465).

This probe fixes a representative HARD placement (baseline legalization) and
sweeps soft positions along (1-a)*initial + a*net_centroid_target for several
blend factors a. If no a < 1 beats a=0 (initial), O3 is not worth pursuing with
a centroid method and we should stop. If some small a helps, that's the lever.

Run:
    uv run python submissions/varrahan/v2/test/diagnostic/_soft_headroom.py ibm04
    uv run python submissions/varrahan/v2/test/diagnostic/_soft_headroom.py ibm04 ibm10 ibm16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
V2_DIR = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2_DIR))

import importlib.util  # noqa: E402

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

SAMEER_DIR = REPO_ROOT / "submissions" / "sameer_v1"
sys.path.insert(0, str(SAMEER_DIR))
from placer import _will_legalize  # type: ignore  # noqa: E402

# Both sameer_v1 and v2 expose a module named `placer`; load v2's by path
# under a distinct name to avoid the sys.modules cache collision above.
_spec = importlib.util.spec_from_file_location("v2_placer", str(V2_DIR / "placer.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_build_wl_cache = _v2._build_wl_cache
_ensure_pos_cache = _v2._ensure_pos_cache
_fast_set_placement = _v2._fast_set_placement
_exact_proxy = _v2._exact_proxy


def _proxy_breakdown(pl_tensor, bm, plc):
    """Fast proxy via v2's vectorized scorer; returns (proxy, wl, den, cong)."""
    p = _exact_proxy(pl_tensor, bm, plc)
    wl = float(plc.get_cost())
    den = float(plc.get_density_cost())
    cong = float(plc.get_congestion_cost())
    return p, wl, den, cong

ICCAD_DIR = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"

BLENDS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def soft_centroid_targets(plc, bm, placement_np: np.ndarray) -> np.ndarray:
    """Jacobi net-centroid target for each soft macro, given fixed positions.

    target[i] = mean over nets touching soft i of (mean pin position of that net).
    Returns [num_soft, 2]; softs with no nets get NaN (caller keeps initial).
    """
    n = bm.num_hard_macros
    num_soft = bm.num_soft_macros
    soft_idx = np.asarray(bm.soft_macro_indices, dtype=np.int64)

    # Populate plc's global pos cache with this placement so pin positions read true.
    _fast_set_placement(plc, placement_np, bm)
    pos_cache = _ensure_pos_cache(plc)

    wl = _build_wl_cache(plc)
    ref_idx = wl["ref_idx"]          # [n_pins] parent module per pin
    x_off, y_off = wl["x_off"], wl["y_off"]
    net_starts, net_ends = wl["net_starts"], wl["net_ends"]
    pin_to_net = wl["pin_to_net"]    # [n_pins] net index per pin
    n_nets = wl["n_nets"]

    # Pin world positions = node pos + offset.
    pin_x = pos_cache[ref_idx, 0] + x_off
    pin_y = pos_cache[ref_idx, 1] + y_off

    # Net center = mean pin position per net (star model).
    sums_x = np.add.reduceat(pin_x, net_starts)
    sums_y = np.add.reduceat(pin_y, net_starts)
    counts = (net_ends - net_starts).astype(np.float64)
    counts[counts == 0] = 1.0
    net_cx = sums_x / counts
    net_cy = sums_y / counts

    # Map each soft module index -> contiguous soft row 0..num_soft-1.
    mod_to_soft = {int(m): i for i, m in enumerate(soft_idx)}
    soft_pin_mask = np.isin(ref_idx, soft_idx)
    spin_mod = ref_idx[soft_pin_mask]
    spin_net = pin_to_net[soft_pin_mask]
    spin_soft = np.array([mod_to_soft[int(m)] for m in spin_mod], dtype=np.int64)

    tgt_sum = np.zeros((num_soft, 2), dtype=np.float64)
    tgt_cnt = np.zeros(num_soft, dtype=np.float64)
    np.add.at(tgt_sum[:, 0], spin_soft, net_cx[spin_net])
    np.add.at(tgt_sum[:, 1], spin_soft, net_cy[spin_net])
    np.add.at(tgt_cnt, spin_soft, 1.0)

    target = np.full((num_soft, 2), np.nan, dtype=np.float64)
    nz = tgt_cnt > 0
    target[nz, 0] = tgt_sum[nz, 0] / tgt_cnt[nz]
    target[nz, 1] = tgt_sum[nz, 1] / tgt_cnt[nz]
    return target


def run(name: str):
    print(f"\n=== {name} ===")
    bm, plc = load_benchmark_from_dir(str(ICCAD_DIR / name))
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    init_pos = bm.macro_positions[:n].numpy().copy().astype(np.float64)

    leg = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)

    soft_init = bm.macro_positions[n:].numpy().copy().astype(np.float64)
    print(f"  n_hard={n}  n_soft={bm.num_soft_macros}  canvas={cw:.0f}x{ch:.0f}")

    # Fixed-hard placement with initial softs (for target computation + a=0).
    base_pl = bm.macro_positions.clone()
    base_pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    base_pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)

    target = soft_centroid_targets(plc, bm, base_pl.cpu().numpy())
    has_tgt = ~np.isnan(target[:, 0])
    print(f"  softs with >=1 net: {int(has_tgt.sum())}/{bm.num_soft_macros}")

    best = None
    for a in BLENDS:
        soft_new = soft_init.copy()
        soft_new[has_tgt] = (1.0 - a) * soft_init[has_tgt] + a * target[has_tgt]
        # clip softs into canvas (centroids can land out of bounds)
        soft_new[:, 0] = np.clip(soft_new[:, 0], 0.0, cw)
        soft_new[:, 1] = np.clip(soft_new[:, 1], 0.0, ch)

        pl = base_pl.clone()
        pl[n:, 0] = torch.tensor(soft_new[:, 0], dtype=torch.float32)
        pl[n:, 1] = torch.tensor(soft_new[:, 1], dtype=torch.float32)

        p, wl_c, den_c, cong_c = _proxy_breakdown(pl, bm, plc)
        tag = " <- initial (a=0)" if a == 0.0 else ""
        marker = ""
        if a == 0.0:
            base_p = p
        else:
            marker = "  *BEATS a=0*" if p < base_p else ""
        print(f"  a={a:<4}: proxy={p:.4f}  wl={wl_c:.4f} "
              f"den={den_c:.4f} cong={cong_c:.4f}{tag}{marker}")
        if best is None or p < best[1]:
            best = (a, p)

    print(f"  -> best blend a={best[0]} proxy={best[1]:.4f} "
          f"(a=0 was {base_p:.4f}, delta {best[1]-base_p:+.4f})")


def _cong_grad_move(p, field, nr, nc, cw, ch, frac, threshold):
    """Finite-diff gradient move of points `p` AGAINST a per-cell `field`
    (toward lower field value), scaled by local field magnitude. Only points
    in cells with field >= threshold move. Mirrors _routing_congestion_perturb
    but for arbitrary (soft) points and with no legalization."""
    cell_w, cell_h = cw / nc, ch / nr
    scale = frac * min(cw, ch)
    c = np.minimum((p[:, 0] / cell_w).astype(np.int64), nc - 1)
    r = np.minimum((p[:, 1] / cell_h).astype(np.int64), nr - 1)
    c = np.maximum(c, 0)
    r = np.maximum(r, 0)
    local = field[r, c]
    cl, cr = np.maximum(c - 1, 0), np.minimum(c + 1, nc - 1)
    rd, ru = np.maximum(r - 1, 0), np.minimum(r + 1, nr - 1)
    gx = (field[r, cr] - field[r, cl]) / 2.0
    gy = (field[ru, c] - field[rd, c]) / 2.0
    gl = np.sqrt(gx ** 2 + gy ** 2) + 1e-10
    ms = scale * local
    mv = np.zeros_like(p)
    m = local >= threshold
    mv[m, 0] = -(gx[m] / gl[m]) * ms[m]
    mv[m, 1] = -(gy[m] / gl[m]) * ms[m]
    return mv


def run_proxy_probe(name: str, mode: str = "cong"):
    """Greedy descent moving SOFT macros down the proxy via a congestion- or
    density-spread gradient. Accept-only on the true proxy (cannot regress).
    Tests whether soft placement has headroom on the DOMINANT terms (unlike
    the WL-centroid sweep in run(), which targets the ~5%-of-proxy WL term)."""
    print(f"\n=== {name}  [proxy-aware soft probe, mode={mode}] ===")
    bm, plc = load_benchmark_from_dir(str(ICCAD_DIR / name))
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    nr, nc = bm.grid_rows, bm.grid_cols
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    init_pos = bm.macro_positions[:n].numpy().copy().astype(np.float64)
    leg = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)

    soft = bm.macro_positions[n:].numpy().copy().astype(np.float64)
    base_pl = bm.macro_positions.clone()
    base_pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    base_pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)

    def proxy_with_soft(soft_arr):
        pl = base_pl.clone()
        pl[n:, 0] = torch.tensor(soft_arr[:, 0], dtype=torch.float32)
        pl[n:, 1] = torch.tensor(soft_arr[:, 1], dtype=torch.float32)
        return _proxy_breakdown(pl, bm, plc)

    p0, wl0, d0, c0 = proxy_with_soft(soft)
    print(f"  n_soft={bm.num_soft_macros}  init proxy={p0:.4f} "
          f"(wl={wl0:.4f} den={d0:.4f} cong={c0:.4f})")

    best_p, best_soft = p0, soft.copy()
    for frac in (0.02, 0.05, 0.1):
        cur = best_soft.copy()
        cur_p = best_p
        for it in range(1, 7):
            # Refresh plc grids for the current (hard, soft) placement.
            proxy_with_soft(cur)
            if mode == "cong":
                h = np.asarray(plc.get_horizontal_routing_congestion())
                v = np.asarray(plc.get_vertical_routing_congestion())
                if h.size != nr * nc:
                    break
                field = np.maximum(h.reshape(nr, nc), v.reshape(nr, nc))
                thr = 0.5
            else:  # density spread: soft-occupancy histogram
                cidx = np.clip((cur[:, 0] / (cw / nc)).astype(int), 0, nc - 1)
                ridx = np.clip((cur[:, 1] / (ch / nr)).astype(int), 0, nr - 1)
                field = np.zeros((nr, nc), dtype=np.float64)
                np.add.at(field, (ridx, cidx), 1.0)
                thr = field.mean() + 1e-9
            mv = _cong_grad_move(cur, field, nr, nc, cw, ch, frac, thr)
            cand = cur + mv
            cand[:, 0] = np.clip(cand[:, 0], 0.0, cw)
            cand[:, 1] = np.clip(cand[:, 1], 0.0, ch)
            p, wl, d, c = proxy_with_soft(cand)
            moved = int((np.abs(mv).sum(axis=1) > 0).sum())
            improved = p < cur_p - 1e-6
            print(f"  frac={frac:<4} it={it}: proxy={p:.4f} "
                  f"(wl={wl:.4f} den={d:.4f} cong={c:.4f}) moved={moved}"
                  f"{'  +' if improved else ''}")
            if improved:
                cur_p, cur = p, cand
                if p < best_p:
                    best_p, best_soft = p, cand.copy()
            else:
                break

    print(f"  -> best proxy={best_p:.4f}  (init {p0:.4f}, delta {best_p-p0:+.4f})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--proxy":
        mode = "cong"
        rest = args[1:]
        if rest and rest[0] in ("cong", "spread"):
            mode, rest = rest[0], rest[1:]
        for nm in (rest or ["ibm10"]):
            run_proxy_probe(nm, mode=mode)
    else:
        for nm in (args or ["ibm04"]):
            run(nm)
