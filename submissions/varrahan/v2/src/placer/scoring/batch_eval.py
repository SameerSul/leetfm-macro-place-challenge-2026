"""S2 — batched candidate evaluation for hard-macro relocation (reference only).

Scores all K candidate positions of one macro at once instead of K sequential
`_trial_at`s, sharing the prep base state (macro removed): WL is vectorized over
K, congestion uses the base+delta decomposition (box-filter smoothing is linear),
density batches the footprint occupancy.

Verified bit-exact vs the sequential loop (`_verify_batch_eval.py`, Δ ≤ 4.4e-16,
identical argmin), but NOT a win on the small IBM grids — kept as isolated,
zero-impact reference code, not wired into the pipeline. See ARCHITECTURE.md §6.2
for the full investigation.
"""

import numpy as np

from placer.plc.placement import _ensure_pos_cache
from placer.routing.apply import (
    _apply_macro_routing_subset,
    _apply_net_routing_struct,
)


def _batch_wl_norm(sc, i_module, touched, cand_xy):
    """Normalized wirelength for each of the K candidates, vectorized.

    Returns wl_norm[K]. Only macro i's pins move with the candidate; every other
    pin on the touched nets is fixed, so per net the bbox is max/min over
    (fixed pins) and (cand + i's pin offsets), broadcast across K.
    """
    K = cand_xy.shape[0]
    if len(touched) == 0:
        return np.full(K, sc.total_wl_raw / sc.wl_normalizer)

    starts_t = sc.net_starts[touched]
    lengths_t = sc.net_lengths[touched]
    total = int(lengths_t.sum())
    # Per-pin gather over the touched nets (same construction as
    # _compute_per_net_hpwl_subset).
    cum = np.concatenate([[0], np.cumsum(lengths_t)[:-1]])
    pin_indices = np.repeat(starts_t, lengths_t) + (np.arange(total) - np.repeat(cum, lengths_t))
    sub_starts = cum  # reduceat segment starts in the compact array

    pos_cache = _ensure_pos_cache(sc.plc)
    node_x = pos_cache[sc.unique_ref, 0]
    node_y = pos_cache[sc.unique_ref, 1]
    ref_local = sc.ref_inv[pin_indices]
    # macro i's pins among the gathered pins (their node is i_module).
    is_i = sc.unique_ref[ref_local] == i_module

    base_px = node_x[ref_local] + sc.x_off[pin_indices]
    base_py = node_y[ref_local] + sc.y_off[pin_indices]
    off_x = sc.x_off[pin_indices]
    off_y = sc.y_off[pin_indices]

    # pin positions broadcast to [n_pins, K]; overwrite i's pins with cand + offset.
    px = np.broadcast_to(base_px[:, None], (total, K)).copy()
    py = np.broadcast_to(base_py[:, None], (total, K)).copy()
    if is_i.any():
        px[is_i, :] = cand_xy[:, 0][None, :] + off_x[is_i][:, None]
        py[is_i, :] = cand_xy[:, 1][None, :] + off_y[is_i][:, None]

    max_x = np.maximum.reduceat(px, sub_starts, axis=0)
    min_x = np.minimum.reduceat(px, sub_starts, axis=0)
    max_y = np.maximum.reduceat(py, sub_starts, axis=0)
    min_y = np.minimum.reduceat(py, sub_starts, axis=0)
    hpwl = (max_x - min_x) + (max_y - min_y)            # [n_nets, K]

    w = sc.net_weights[touched][:, None]
    base_hpwl = sc.per_net_hpwl[touched][:, None]
    delta = np.sum((hpwl - base_hpwl) * w, axis=0)      # [K]
    return (sc.total_wl_raw + delta) / sc.wl_normalizer


def _score_candidates_hard(sc, prep, cand_xy):
    """Return scores[K] for K candidate (x, y) positions of the prepped macro.

    Bit-close to [sc._trial_at(prep, xy) for xy in cand_xy] (WL is exact; cong /
    density use the verified base+delta decomposition).
    """
    cand_xy = np.asarray(cand_xy, dtype=np.float64).reshape(-1, 2)
    K = cand_xy.shape[0]
    i_module = prep["i_module"]
    struct = prep["struct"]
    macro_subset = prep["macro_subset"]

    # --- WL (batched, exact) ---
    touched = sc._macro_nets(i_module)
    wl_norm = _batch_wl_norm(sc, i_module, touched, cand_xy)

    # --- cong + density: per-candidate fill on the shared base, then top-k ---
    base_H_sm = sc.H_smoothed
    base_V_sm = sc.V_smoothed
    base_Hm = sc.H_macro_flat
    base_Vm = sc.V_macro_flat
    grid_row, grid_col = sc.grid_row, sc.grid_col

    Hd = np.empty(sc.H_flat.shape, dtype=np.float64)
    Vd = np.empty(sc.V_flat.shape, dtype=np.float64)
    Hmd = np.empty(sc.H_macro_flat.shape, dtype=np.float64)
    Vmd = np.empty(sc.V_macro_flat.shape, dtype=np.float64)
    from placer.routing.apply import _smooth_routing_cong_vec

    cong = np.empty(K)
    dens = np.empty(K)
    for k in range(K):
        nx, ny = float(cand_xy[k, 0]), float(cand_xy[k, 1])
        sc._apply_pos(i_module, nx, ny)
        Hd.fill(0.0); Vd.fill(0.0)
        _apply_net_routing_struct(sc.plc, struct, 1.0, Hd, Vd)
        if macro_subset.size:
            Hmd.fill(0.0); Vmd.fill(0.0)
            _apply_macro_routing_subset(sc.plc, macro_subset, 1.0, Vmd, Hmd)
        sc._apply_pos(i_module, prep["old_ix"], prep["old_iy"])

        sm_H = _smooth_routing_cong_vec(
            Hd / sc.grid_h_routes, grid_row, grid_col, sc.smooth_range, axis_h=True
        ).reshape(grid_row, grid_col)
        sm_V = _smooth_routing_cong_vec(
            Vd / sc.grid_v_routes, grid_row, grid_col, sc.smooth_range, axis_h=False
        ).reshape(grid_row, grid_col)
        Hm = (base_Hm + (Hmd if macro_subset.size else 0.0)) / sc.grid_h_routes
        Vm = (base_Vm + (Vmd if macro_subset.size else 0.0)) / sc.grid_v_routes
        xx = np.concatenate([(base_V_sm + sm_V).ravel() + Vm,
                             (base_H_sm + sm_H).ravel() + Hm])
        n = xx.size
        cnt = int(n * 0.05)
        cong[k] = float(xx.max()) if cnt == 0 else float(np.partition(xx, n - cnt)[n - cnt:].sum() / cnt)

        n_idx, n_area = sc._macro_occ(i_module, nx, ny)
        go = sc.grid_occupied
        if n_idx.size:
            np.add.at(go, n_idx, n_area)
        dens[k] = sc._compute_density_cost()
        if n_idx.size:
            np.subtract.at(go, n_idx, n_area)

    return wl_norm + 0.5 * dens + 0.5 * cong


# ---------------------------------------------------------------------------
# GPU batched smooth + congestion (matches the CPU reference; not wired in)
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from placer.config import _GPU_DEVICE  # noqa: E402


def _smooth_batch_gpu(delta, lp, up, cnt, axis):
    """Box-filter smooth a [K, R, C] tensor, matching _resmooth_h/v semantics:
    smoothed[p] = sum_{j in [lp[p], up[p]]} val[j] / cnt[j], along `axis`
    (axis=1 rows for H, axis=2 cols for V). lp/up/cnt are 1-D over that axis.
    """
    shape = [1, 1, 1]
    shape[axis] = -1
    w = delta / cnt.reshape(shape)                       # val[j]/cnt[j]
    cs = torch.cumsum(w, dim=axis)
    zero = torch.zeros_like(delta.index_select(axis, torch.zeros(1, dtype=torch.long, device=delta.device)))
    cs = torch.cat([zero, cs], dim=axis)                 # cs[0]=0, cs[j]=sum(w[0..j-1])
    hi = cs.index_select(axis, up + 1)
    lo = cs.index_select(axis, lp)
    return hi - lo


class _GpuBatchState:
    """Per-scorer cached GPU tensors (grid windows + routes) for the batch path."""
    def __init__(self, sc):
        d = _GPU_DEVICE
        sr = sc.smooth_range
        self.dev = d
        self.R, self.C = sc.grid_row, sc.grid_col
        self.h_routes = float(sc.grid_h_routes)
        self.v_routes = float(sc.grid_v_routes)
        self.sr = sr
        if sr > 0:
            self.row_lp = torch.as_tensor(sc._sm_row_lp, device=d)
            self.row_up = torch.as_tensor(sc._sm_row_up, device=d)
            self.row_cnt = torch.as_tensor(sc._sm_row_cnt, device=d)
            self.col_lp = torch.as_tensor(sc._sm_col_lp, device=d)
            self.col_up = torch.as_tensor(sc._sm_col_up, device=d)
            self.col_cnt = torch.as_tensor(sc._sm_col_cnt, device=d)


def _score_candidates_hard_gpu(sc, prep, cand_xy):
    """GPU batch: per-candidate fill (CPU), then batched smooth + cong top-5% +
    density top-10% on GPU. Matches _score_candidates_hard (the reference)."""
    cand_xy = np.asarray(cand_xy, dtype=np.float64).reshape(-1, 2)
    K = cand_xy.shape[0]
    i_module = prep["i_module"]
    struct = prep["struct"]
    macro_subset = prep["macro_subset"]
    R, C = sc.grid_row, sc.grid_col
    if getattr(sc, "_gpu_batch_state", None) is None:
        sc._gpu_batch_state = _GpuBatchState(sc)

    wl_norm = _batch_wl_norm(sc, i_module, sc._macro_nets(i_module), cand_xy)

    # Per-candidate raw deltas (fill loops; the expensive smooth/topk batch below).
    Hbuf = np.zeros((K, sc.H_flat.size), dtype=np.float64)
    Vbuf = np.zeros((K, sc.V_flat.size), dtype=np.float64)
    Hmbuf = np.zeros((K, sc.H_macro_flat.size), dtype=np.float64)
    Vmbuf = np.zeros((K, sc.V_macro_flat.size), dtype=np.float64)
    dens = np.empty(K)
    for k in range(K):
        nx, ny = float(cand_xy[k, 0]), float(cand_xy[k, 1])
        sc._apply_pos(i_module, nx, ny)
        _apply_net_routing_struct(sc.plc, struct, 1.0, Hbuf[k], Vbuf[k])
        if macro_subset.size:
            _apply_macro_routing_subset(sc.plc, macro_subset, 1.0, Vmbuf[k], Hmbuf[k])
        sc._apply_pos(i_module, prep["old_ix"], prep["old_iy"])
        n_idx, n_area = sc._macro_occ(i_module, nx, ny)
        go = sc.grid_occupied
        if n_idx.size:
            np.add.at(go, n_idx, n_area)
        dens[k] = sc._compute_density_cost()
        if n_idx.size:
            np.subtract.at(go, n_idx, n_area)

    st = sc._gpu_batch_state
    d = st.dev
    Ht = torch.as_tensor(Hbuf, device=d).reshape(K, R, C)
    Vt = torch.as_tensor(Vbuf, device=d).reshape(K, R, C)
    if st.sr > 0:
        smH = _smooth_batch_gpu(Ht / st.h_routes, st.row_lp, st.row_up, st.row_cnt, axis=1)
        smV = _smooth_batch_gpu(Vt / st.v_routes, st.col_lp, st.col_up, st.col_cnt, axis=2)
    else:
        smH = Ht / st.h_routes
        smV = Vt / st.v_routes
    baseH = torch.as_tensor(sc.H_smoothed, device=d)
    baseV = torch.as_tensor(sc.V_smoothed, device=d)
    Hm = (torch.as_tensor(sc.H_macro_flat, device=d).reshape(R, C)[None] +
          torch.as_tensor(Hmbuf, device=d).reshape(K, R, C)) / st.h_routes
    Vm = (torch.as_tensor(sc.V_macro_flat, device=d).reshape(R, C)[None] +
          torch.as_tensor(Vmbuf, device=d).reshape(K, R, C)) / st.v_routes
    H_total = (baseH[None] + smH + Hm).reshape(K, -1)
    V_total = (baseV[None] + smV + Vm).reshape(K, -1)
    xx = torch.cat([V_total, H_total], dim=1)            # [K, 2*cells]
    ncell = xx.shape[1]
    cnt = int(ncell * 0.05)
    if cnt == 0:
        cong = xx.max(dim=1).values
    else:
        cong = torch.topk(xx, cnt, dim=1).values.sum(dim=1) / cnt
    cong = cong.cpu().numpy()
    return wl_norm + 0.5 * dens + 0.5 * cong
