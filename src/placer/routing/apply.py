"""Vectorized routing demand and smoothing helpers."""

import numpy as np
import torch
from macro_place.benchmark import Benchmark

from utils import constants as const
from utils.config import HAS_NUMBA, _GPU_DEVICE, _USE_GPU, _numba_njit
from placer.plc.placement import _ensure_pos_cache
from placer.scoring.wirelength import _build_wl_cache


def _build_cong_cache(plc, benchmark: Benchmark):
    """Build routing arrays that do not change with placement."""
    if hasattr(plc, "_cong_cache"):
        return plc._cong_cache
    wl = _build_wl_cache(plc)

    # Pin range for each net.
    starts = wl["net_starts"]
    n_nets = len(starts)
    n_pins = wl["n_pins"]
    if n_nets == 0:
        ends = np.zeros(0, dtype=np.int64)
    else:
        ends = np.concatenate([starts[1:], np.array([n_pins], dtype=np.int64)])
    lengths = ends - starts

    # Hard macro sizes.
    hard_indices = list(plc.hard_macro_indices)
    n_hard = len(hard_indices)
    hard_half_w = np.empty(n_hard, dtype=np.float64)
    hard_half_h = np.empty(n_hard, dtype=np.float64)
    for k, idx in enumerate(hard_indices):
        m = plc.modules_w_pins[idx]
        hard_half_w[k] = float(m.get_width()) * 0.5
        hard_half_h[k] = float(m.get_height()) * 0.5

    # Group nets by pin count once.
    idx2_cache = np.where(lengths == 2)[0]
    s2_cache = starts[idx2_cache] if idx2_cache.size else np.zeros(0, dtype=np.int64)
    s2p1_cache = s2_cache + 1

    idx3_cache = np.where(lengths == 3)[0]
    if idx3_cache.size:
        s3_cache = starts[idx3_cache]
        s3p1_cache = s3_cache + 1
        s3p2_cache = s3_cache + 2
    else:
        s3_cache = s3p1_cache = s3p2_cache = np.zeros(0, dtype=np.int64)

    idx_big_cache = np.where(lengths >= 4)[0]
    if idx_big_cache.size:
        starts_big_cache = starts[idx_big_cache]
        lengths_big_cache = lengths[idx_big_cache]
        sink_lens_cache = lengths_big_cache - 1
        sink_total_cache = int(sink_lens_cache.sum())
        B_cache = idx_big_cache.size
        if sink_total_cache > 0:
            net_local_ids_cache = np.repeat(np.arange(B_cache, dtype=np.int64), sink_lens_cache)
            cum_sink_starts_cache = np.zeros(B_cache + 1, dtype=np.int64)
            np.cumsum(sink_lens_cache, out=cum_sink_starts_cache[1:])
            offset_in_sinks_cache = np.arange(sink_total_cache, dtype=np.int64) - np.repeat(
                cum_sink_starts_cache[:-1], sink_lens_cache
            )
            global_pin_idx_cache = (starts_big_cache + 1)[
                net_local_ids_cache
            ] + offset_in_sinks_cache
        else:
            net_local_ids_cache = np.zeros(0, dtype=np.int64)
            cum_sink_starts_cache = np.zeros(1, dtype=np.int64)
            global_pin_idx_cache = np.zeros(0, dtype=np.int64)
    else:
        starts_big_cache = np.zeros(0, dtype=np.int64)
        lengths_big_cache = np.zeros(0, dtype=np.int64)
        sink_lens_cache = np.zeros(0, dtype=np.int64)
        sink_total_cache = 0
        B_cache = 0
        net_local_ids_cache = np.zeros(0, dtype=np.int64)
        cum_sink_starts_cache = np.zeros(1, dtype=np.int64)
        global_pin_idx_cache = np.zeros(0, dtype=np.int64)

    plc._cong_cache = {
        "starts": starts,
        "ends": ends,
        "lengths": lengths,
        "n_nets": n_nets,
        "hard_indices": hard_indices,
        "hard_half_w": hard_half_w,
        "hard_half_h": hard_half_h,
        "n_hard": n_hard,
        # Net groups used by the routing dispatcher.
        "idx2": idx2_cache,
        "s2": s2_cache,
        "s2p1": s2p1_cache,
        "idx3": idx3_cache,
        "s3": s3_cache,
        "s3p1": s3p1_cache,
        "s3p2": s3p2_cache,
        "idx_big": idx_big_cache,
        "starts_big": starts_big_cache,
        "lengths_big": lengths_big_cache,
        "sink_lens": sink_lens_cache,
        "sink_total": sink_total_cache,
        "B_big": B_cache,
        "net_local_ids": net_local_ids_cache,
        "cum_sink_starts": cum_sink_starts_cache,
        "global_pin_idx": global_pin_idx_cache,
    }
    return plc._cong_cache


if HAS_NUMBA:

    @_numba_njit(cache=True, fastmath=False)
    def _apply_h_strips_batch_jit(H_flat, row, col_lo, col_hi, weight, grid_col):
        """Add horizontal routing strips with numba."""
        n = row.shape[0]
        for k in range(n):
            r = row[k]
            lo = col_lo[k]
            hi = col_hi[k]
            w = weight[k]
            base = r * grid_col
            for c in range(lo, hi):
                H_flat[base + c] += w

    @_numba_njit(cache=True, fastmath=False)
    def _apply_v_strips_batch_jit(V_flat, col, row_lo, row_hi, weight, grid_col):
        """Add vertical routing strips with numba."""
        n = col.shape[0]
        for k in range(n):
            c = col[k]
            lo = row_lo[k]
            hi = row_hi[k]
            w = weight[k]
            for r in range(lo, hi):
                V_flat[r * grid_col + c] += w


def _apply_h_strips_batch(
    H_flat: np.ndarray,
    row: np.ndarray,
    col_lo: np.ndarray,
    col_hi: np.ndarray,
    weight: np.ndarray,
    grid_row: int,
    grid_col: int,
) -> None:
    """Add a batch of horizontal routing strips."""
    if row.size == 0:
        return
    if HAS_NUMBA and const.ROUTE_STRUCT_JIT:
        _apply_h_strips_batch_jit(
            H_flat,
            np.ascontiguousarray(row, dtype=np.int64),
            np.ascontiguousarray(col_lo, dtype=np.int64),
            np.ascontiguousarray(col_hi, dtype=np.int64),
            np.ascontiguousarray(weight, dtype=np.float64),
            int(grid_col),
        )
        return
    # Numpy fallback.
    uniq, inv = np.unique(row, return_inverse=True)
    inv = inv.ravel()
    nU = uniq.size
    stride = grid_col + 1
    base = inv * stride
    all_idx = np.concatenate([base + col_lo, base + col_hi])
    all_w = np.concatenate([weight, -weight])
    h_events = np.bincount(all_idx, weights=all_w, minlength=nU * stride).reshape(nU, stride)
    cs = np.cumsum(h_events, axis=1)[:, :grid_col]
    H_flat.reshape(grid_row, grid_col)[uniq] += cs


def _apply_v_strips_batch(
    V_flat: np.ndarray,
    col: np.ndarray,
    row_lo: np.ndarray,
    row_hi: np.ndarray,
    weight: np.ndarray,
    grid_row: int,
    grid_col: int,
) -> None:
    """Add a batch of vertical routing strips."""
    if col.size == 0:
        return
    if HAS_NUMBA:
        _apply_v_strips_batch_jit(
            V_flat,
            np.ascontiguousarray(col, dtype=np.int64),
            np.ascontiguousarray(row_lo, dtype=np.int64),
            np.ascontiguousarray(row_hi, dtype=np.int64),
            np.ascontiguousarray(weight, dtype=np.float64),
            int(grid_col),
        )
        return
    # Numpy fallback.
    uniq, inv = np.unique(col, return_inverse=True)
    inv = inv.ravel()
    nU = uniq.size
    stride = grid_row + 1
    base = inv * stride
    all_idx = np.concatenate([base + row_lo, base + row_hi])
    all_w = np.concatenate([weight, -weight])
    v_events = np.bincount(all_idx, weights=all_w, minlength=nU * stride).reshape(nU, stride)
    cs = np.cumsum(v_events, axis=1)[:, :grid_row]
    V_flat.reshape(grid_row, grid_col)[:, uniq] += cs.T


def _apply_2pin_routing(
    H_flat: np.ndarray,
    V_flat: np.ndarray,
    src_row: np.ndarray,
    src_col: np.ndarray,
    snk_row: np.ndarray,
    snk_col: np.ndarray,
    weight: np.ndarray,
    grid_row: int,
    grid_col: int,
) -> None:
    """Route 2-pin nets with one L-shaped path per net."""
    if src_row.size == 0:
        return
    col_min = np.minimum(src_col, snk_col)
    col_max = np.maximum(src_col, snk_col)
    _apply_h_strips_batch(H_flat, src_row, col_min, col_max, weight, grid_row, grid_col)
    row_min = np.minimum(src_row, snk_row)
    row_max = np.maximum(src_row, snk_row)
    _apply_v_strips_batch(V_flat, snk_col, row_min, row_max, weight, grid_row, grid_col)


if HAS_NUMBA:

    @_numba_njit(cache=True, fastmath=False)
    def _apply_3pin_routing_vec_jit(H_flat, V_flat, g0_flat, g1_flat, g2_flat, weights, grid_col):
        """Route 3-pin nets with numba."""
        n = g0_flat.shape[0]
        for k in range(n):
            # Decode each pin's grid cell.
            ya = g0_flat[k] // grid_col
            xa = g0_flat[k] % grid_col
            yb = g1_flat[k] // grid_col
            xb = g1_flat[k] % grid_col
            yc = g2_flat[k] // grid_col
            xc = g2_flat[k] % grid_col
            w = weights[k]

            # Sort three points by column, then row.
            x1 = xa
            y1 = ya
            x2 = xb
            y2 = yb
            x3 = xc
            y3 = yc
            if x1 > x2 or (x1 == x2 and y1 > y2):
                tx = x1
                x1 = x2
                x2 = tx
                ty = y1
                y1 = y2
                y2 = ty
            if x2 > x3 or (x2 == x3 and y2 > y3):
                tx = x2
                x2 = x3
                x3 = tx
                ty = y2
                y2 = y3
                y3 = ty
            if x1 > x2 or (x1 == x2 and y1 > y2):
                tx = x1
                x1 = x2
                x2 = tx
                ty = y1
                y1 = y2
                y2 = ty

            # Case 1: L-routing - x1<x2<x3 AND y2 strictly between y1 and y3.
            if x1 < x2 and x2 < x3:
                ymn13 = y1 if y1 < y3 else y3
                ymx13 = y1 if y1 > y3 else y3
                if ymn13 < y2 and ymx13 > y2:
                    # H y1 [x1..x2], y2 [x2..x3]
                    base1 = y1 * grid_col
                    for c in range(x1, x2):
                        H_flat[base1 + c] += w
                    base2 = y2 * grid_col
                    for c in range(x2, x3):
                        H_flat[base2 + c] += w
                    # V x2 [min(y1,y2)..max(y1,y2)]
                    r_lo = y1 if y1 < y2 else y2
                    r_hi = y1 if y1 > y2 else y2
                    for r in range(r_lo, r_hi):
                        V_flat[r * grid_col + x2] += w
                    # V x3 [min(y2,y3)..max(y2,y3)]
                    r_lo = y2 if y2 < y3 else y3
                    r_hi = y2 if y2 > y3 else y3
                    for r in range(r_lo, r_hi):
                        V_flat[r * grid_col + x3] += w
                    continue

            # Case 2: x2==x3, x1<x2, y1 < min(y2, y3).  NOT case1.
            mny23 = y2 if y2 < y3 else y3
            if x2 == x3 and x1 < x2 and y1 < mny23:
                # H y1 [x1..x2]
                base1 = y1 * grid_col
                for c in range(x1, x2):
                    H_flat[base1 + c] += w
                # V x2 [y1..max(y2,y3)]
                mxy23 = y2 if y2 > y3 else y3
                for r in range(y1, mxy23):
                    V_flat[r * grid_col + x2] += w
                continue

            # Case 3: y2 == y3.  NOT case1, NOT case2.
            if y2 == y3:
                base1 = y1 * grid_col
                for c in range(x1, x2):
                    H_flat[base1 + c] += w
                base2 = y2 * grid_col
                for c in range(x2, x3):
                    H_flat[base2 + c] += w
                # V x2 [min(y2,y1)..max(y2,y1)]
                r_lo = y2 if y2 < y1 else y1
                r_hi = y2 if y2 > y1 else y1
                for r in range(r_lo, r_hi):
                    V_flat[r * grid_col + x2] += w
                continue

            # Case 4: T-routing - re-sort by (row asc, col asc).
            # Restart from the ORIGINAL (xa, ya), (xb, yb), (xc, yc).
            x1t = xa
            y1t = ya
            x2t = xb
            y2t = yb
            x3t = xc
            y3t = yc
            if y1t > y2t or (y1t == y2t and x1t > x2t):
                tx = x1t
                x1t = x2t
                x2t = tx
                ty = y1t
                y1t = y2t
                y2t = ty
            if y2t > y3t or (y2t == y3t and x2t > x3t):
                tx = x2t
                x2t = x3t
                x3t = tx
                ty = y2t
                y2t = y3t
                y3t = ty
            if y1t > y2t or (y1t == y2t and x1t > x2t):
                tx = x1t
                x1t = x2t
                x2t = tx
                ty = y1t
                y1t = y2t
                y2t = ty

            # xmin/xmax over all 3 (= over the original 3 = the row-sorted 3).
            xmin = x1t if x1t < x2t else x2t
            if x3t < xmin:
                xmin = x3t
            xmax = x1t if x1t > x2t else x2t
            if x3t > xmax:
                xmax = x3t
            # H y2t [xmin..xmax]
            base = y2t * grid_col
            for c in range(xmin, xmax):
                H_flat[base + c] += w
            # V x1t [y1t..y2t]  (rows are sorted, so y1t <= y2t <= y3t)
            for r in range(y1t, y2t):
                V_flat[r * grid_col + x1t] += w
            # V x3t [y2t..y3t]
            for r in range(y2t, y3t):
                V_flat[r * grid_col + x3t] += w


def _apply_3pin_routing_vec(
    H_flat: np.ndarray,
    V_flat: np.ndarray,
    g0_flat: np.ndarray,
    g1_flat: np.ndarray,
    g2_flat: np.ndarray,
    weights: np.ndarray,
    grid_row: int,
    grid_col: int,
) -> None:
    """Route 3-pin nets using numba when available."""
    if g0_flat.size == 0:
        return
    if HAS_NUMBA:
        _apply_3pin_routing_vec_jit(
            H_flat,
            V_flat,
            np.ascontiguousarray(g0_flat, dtype=np.int64),
            np.ascontiguousarray(g1_flat, dtype=np.int64),
            np.ascontiguousarray(g2_flat, dtype=np.int64),
            np.ascontiguousarray(weights, dtype=np.float64),
            int(grid_col),
        )
        return
    _apply_3pin_routing_vec_numpy(
        H_flat, V_flat, g0_flat, g1_flat, g2_flat, weights, grid_row, grid_col
    )


def _apply_3pin_routing_vec_numpy(
    H_flat: np.ndarray,
    V_flat: np.ndarray,
    g0_flat: np.ndarray,
    g1_flat: np.ndarray,
    g2_flat: np.ndarray,
    weights: np.ndarray,
    grid_row: int,
    grid_col: int,
) -> None:
    """Numpy fallback for three-pin routing."""
    if g0_flat.size == 0:
        return
    # Convert flat cell ids into row/column pairs.
    y_all = np.stack(
        [g0_flat // grid_col, g1_flat // grid_col, g2_flat // grid_col], axis=1
    ).astype(np.int64)
    x_all = np.stack([g0_flat % grid_col, g1_flat % grid_col, g2_flat % grid_col], axis=1).astype(
        np.int64
    )
    w = np.asarray(weights, dtype=np.float64)
    # Sort each net's three pins by column, then row.
    BIG = int(max(grid_row, grid_col)) + 16
    key = x_all * BIG + y_all
    order = np.argsort(key, axis=1, kind="stable")
    y = np.take_along_axis(y_all, order, axis=1)
    x = np.take_along_axis(x_all, order, axis=1)
    y1 = y[:, 0]
    y2 = y[:, 1]
    y3 = y[:, 2]
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    # Case 1: L-routing - x1<x2<x3 AND y2 strictly between y1 and y3
    case1 = (x1 < x2) & (x2 < x3) & (np.minimum(y1, y3) < y2) & (np.maximum(y1, y3) > y2)
    # Case 2: x2==x3, x1<x2, y1 < min(y2,y3), NOT case1
    case2 = (~case1) & (x2 == x3) & (x1 < x2) & (y1 < np.minimum(y2, y3))
    # Case 3: y2==y3, NOT case1, NOT case2
    case3 = (~case1) & (~case2) & (y2 == y3)
    case4 = ~(case1 | case2 | case3)

    h_rows: "list[np.ndarray]" = []
    h_los: "list[np.ndarray]" = []
    h_his: "list[np.ndarray]" = []
    h_ws: "list[np.ndarray]" = []
    v_cols: "list[np.ndarray]" = []
    v_los: "list[np.ndarray]" = []
    v_his: "list[np.ndarray]" = []
    v_ws: "list[np.ndarray]" = []

    if case1.any():
        m = case1
        wm = w[m]
        # H y1 [x1..x2], y2 [x2..x3]
        h_rows.append(y1[m])
        h_los.append(x1[m])
        h_his.append(x2[m])
        h_ws.append(wm)
        h_rows.append(y2[m])
        h_los.append(x2[m])
        h_his.append(x3[m])
        h_ws.append(wm)
        # V x2 [min(y1,y2)..max(y1,y2)], x3 [min(y2,y3)..max(y2,y3)]
        v_cols.append(x2[m])
        v_los.append(np.minimum(y1[m], y2[m]))
        v_his.append(np.maximum(y1[m], y2[m]))
        v_ws.append(wm)
        v_cols.append(x3[m])
        v_los.append(np.minimum(y2[m], y3[m]))
        v_his.append(np.maximum(y2[m], y3[m]))
        v_ws.append(wm)

    if case2.any():
        m = case2
        wm = w[m]
        h_rows.append(y1[m])
        h_los.append(x1[m])
        h_his.append(x2[m])
        h_ws.append(wm)
        v_cols.append(x2[m])
        v_los.append(y1[m])
        v_his.append(np.maximum(y2[m], y3[m]))
        v_ws.append(wm)

    if case3.any():
        m = case3
        wm = w[m]
        h_rows.append(y1[m])
        h_los.append(x1[m])
        h_his.append(x2[m])
        h_ws.append(wm)
        h_rows.append(y2[m])
        h_los.append(x2[m])
        h_his.append(x3[m])
        h_ws.append(wm)
        v_cols.append(x2[m])
        v_los.append(np.minimum(y2[m], y1[m]))
        v_his.append(np.maximum(y2[m], y1[m]))
        v_ws.append(wm)

    if case4.any():
        m = case4
        wm = w[m]
        # T-routes use row order.
        y_t = y_all[m]
        x_t = x_all[m]
        key_t = y_t * BIG + x_t
        order_t = np.argsort(key_t, axis=1, kind="stable")
        y_t = np.take_along_axis(y_t, order_t, axis=1)
        x_t = np.take_along_axis(x_t, order_t, axis=1)
        y1t = y_t[:, 0]
        y2t = y_t[:, 1]
        y3t = y_t[:, 2]
        x1t = x_t[:, 0]
        x2t = x_t[:, 1]
        x3t = x_t[:, 2]
        xmin_t = np.minimum(np.minimum(x1t, x2t), x3t)
        xmax_t = np.maximum(np.maximum(x1t, x2t), x3t)
        h_rows.append(y2t)
        h_los.append(xmin_t)
        h_his.append(xmax_t)
        h_ws.append(wm)
        v_cols.append(x1t)
        v_los.append(np.minimum(y1t, y2t))
        v_his.append(np.maximum(y1t, y2t))
        v_ws.append(wm)
        v_cols.append(x3t)
        v_los.append(np.minimum(y2t, y3t))
        v_his.append(np.maximum(y2t, y3t))
        v_ws.append(wm)

    if h_rows:
        rows = np.concatenate(h_rows)
        los = np.concatenate(h_los)
        his = np.concatenate(h_his)
        ws_h = np.concatenate(h_ws)
        nz = los != his
        if nz.any():
            _apply_h_strips_batch(H_flat, rows[nz], los[nz], his[nz], ws_h[nz], grid_row, grid_col)
    if v_cols:
        cols = np.concatenate(v_cols)
        rlos = np.concatenate(v_los)
        rhis = np.concatenate(v_his)
        ws_v = np.concatenate(v_ws)
        nz = rlos != rhis
        if nz.any():
            _apply_v_strips_batch(
                V_flat, cols[nz], rlos[nz], rhis[nz], ws_v[nz], grid_row, grid_col
            )


if HAS_NUMBA:

    @_numba_njit(cache=True, fastmath=False)
    def _apply_route_struct_l2_l3_jit(
        H_flat,
        V_flat,
        pin_gcell,
        starts2,
        weights2,
        starts3,
        weights3,
        grid_col,
    ):
        """Apply 2/3-pin prepared route-struct nets without Python buckets."""
        for k in range(starts2.shape[0]):
            s = starts2[k]
            src = pin_gcell[s]
            snk = pin_gcell[s + 1]
            if src == snk:
                continue
            src_row = src // grid_col
            src_col = src % grid_col
            snk_row = snk // grid_col
            snk_col = snk % grid_col
            w = weights2[k]
            c_lo = src_col if src_col < snk_col else snk_col
            c_hi = src_col if src_col > snk_col else snk_col
            base = src_row * grid_col
            for c in range(c_lo, c_hi):
                H_flat[base + c] += w
            r_lo = src_row if src_row < snk_row else snk_row
            r_hi = src_row if src_row > snk_row else snk_row
            for r in range(r_lo, r_hi):
                V_flat[r * grid_col + snk_col] += w

        for k in range(starts3.shape[0]):
            s = starts3[k]
            g0 = pin_gcell[s]
            g1 = pin_gcell[s + 1]
            g2 = pin_gcell[s + 2]
            eq01 = g0 == g1
            eq02 = g0 == g2
            eq12 = g1 == g2
            eq_count = 0
            if eq01:
                eq_count += 1
            if eq02:
                eq_count += 1
            if eq12:
                eq_count += 1
            if eq_count == 3:
                continue
            w = weights3[k]
            if eq_count == 1:
                src = g0
                snk = g2 if eq01 else g1
                src_row = src // grid_col
                src_col = src % grid_col
                snk_row = snk // grid_col
                snk_col = snk % grid_col
                c_lo = src_col if src_col < snk_col else snk_col
                c_hi = src_col if src_col > snk_col else snk_col
                base = src_row * grid_col
                for c in range(c_lo, c_hi):
                    H_flat[base + c] += w
                r_lo = src_row if src_row < snk_row else snk_row
                r_hi = src_row if src_row > snk_row else snk_row
                for r in range(r_lo, r_hi):
                    V_flat[r * grid_col + snk_col] += w
                continue

            ya = g0 // grid_col
            xa = g0 % grid_col
            yb = g1 // grid_col
            xb = g1 % grid_col
            yc = g2 // grid_col
            xc = g2 % grid_col

            x1 = xa
            y1 = ya
            x2 = xb
            y2 = yb
            x3 = xc
            y3 = yc
            if x1 > x2 or (x1 == x2 and y1 > y2):
                tx = x1
                x1 = x2
                x2 = tx
                ty = y1
                y1 = y2
                y2 = ty
            if x2 > x3 or (x2 == x3 and y2 > y3):
                tx = x2
                x2 = x3
                x3 = tx
                ty = y2
                y2 = y3
                y3 = ty
            if x1 > x2 or (x1 == x2 and y1 > y2):
                tx = x1
                x1 = x2
                x2 = tx
                ty = y1
                y1 = y2
                y2 = ty

            if x1 < x2 and x2 < x3:
                ymn13 = y1 if y1 < y3 else y3
                ymx13 = y1 if y1 > y3 else y3
                if ymn13 < y2 and ymx13 > y2:
                    base1 = y1 * grid_col
                    for c in range(x1, x2):
                        H_flat[base1 + c] += w
                    base2 = y2 * grid_col
                    for c in range(x2, x3):
                        H_flat[base2 + c] += w
                    r_lo = y1 if y1 < y2 else y2
                    r_hi = y1 if y1 > y2 else y2
                    for r in range(r_lo, r_hi):
                        V_flat[r * grid_col + x2] += w
                    r_lo = y2 if y2 < y3 else y3
                    r_hi = y2 if y2 > y3 else y3
                    for r in range(r_lo, r_hi):
                        V_flat[r * grid_col + x3] += w
                    continue

            mny23 = y2 if y2 < y3 else y3
            if x2 == x3 and x1 < x2 and y1 < mny23:
                base1 = y1 * grid_col
                for c in range(x1, x2):
                    H_flat[base1 + c] += w
                mxy23 = y2 if y2 > y3 else y3
                for r in range(y1, mxy23):
                    V_flat[r * grid_col + x2] += w
                continue

            if y2 == y3:
                base1 = y1 * grid_col
                for c in range(x1, x2):
                    H_flat[base1 + c] += w
                base2 = y2 * grid_col
                for c in range(x2, x3):
                    H_flat[base2 + c] += w
                r_lo = y2 if y2 < y1 else y1
                r_hi = y2 if y2 > y1 else y1
                for r in range(r_lo, r_hi):
                    V_flat[r * grid_col + x2] += w
                continue

            x1t = xa
            y1t = ya
            x2t = xb
            y2t = yb
            x3t = xc
            y3t = yc
            if y1t > y2t or (y1t == y2t and x1t > x2t):
                tx = x1t
                x1t = x2t
                x2t = tx
                ty = y1t
                y1t = y2t
                y2t = ty
            if y2t > y3t or (y2t == y3t and x2t > x3t):
                tx = x2t
                x2t = x3t
                x3t = tx
                ty = y2t
                y2t = y3t
                y3t = ty
            if y1t > y2t or (y1t == y2t and x1t > x2t):
                tx = x1t
                x1t = x2t
                x2t = tx
                ty = y1t
                y1t = y2t
                y2t = ty

            xmin = x1t if x1t < x2t else x2t
            if x3t < xmin:
                xmin = x3t
            xmax = x1t if x1t > x2t else x2t
            if x3t > xmax:
                xmax = x3t
            base = y2t * grid_col
            for c in range(xmin, xmax):
                H_flat[base + c] += w
            for r in range(y1t, y2t):
                V_flat[r * grid_col + x1t] += w
            for r in range(y2t, y3t):
                V_flat[r * grid_col + x3t] += w


def _smooth_routing_cong_vec(
    routing_flat: np.ndarray, grid_row: int, grid_col: int, smooth_range: int, axis_h: bool
) -> np.ndarray:
    """Smooth routing demand across nearby rows or columns."""
    grid_2d = routing_flat.reshape(grid_row, grid_col)
    sr = smooth_range
    if _USE_GPU:
        with torch.no_grad():
            dev = _GPU_DEVICE
            g2d = torch.from_numpy(grid_2d).to(dev)
            if axis_h:
                # H demand spreads along rows.
                rows = torch.arange(grid_row, dtype=torch.int64, device=dev)
                cnts = (
                    torch.clamp(rows + sr, max=grid_row - 1) - torch.clamp(rows - sr, min=0) + 1
                ).to(g2d.dtype)
                w = g2d / cnts[:, None]
                zero_row = torch.zeros(1, grid_col, dtype=g2d.dtype, device=dev)
                cs = torch.cumsum(torch.cat([zero_row, w], dim=0), dim=0)
                lo_idx = torch.clamp(rows - sr, min=0)
                hi_idx = torch.clamp(rows + sr + 1, max=grid_row)
                smoothed = cs[hi_idx] - cs[lo_idx]
            else:
                # V demand spreads along columns.
                cols = torch.arange(grid_col, dtype=torch.int64, device=dev)
                cnts = (
                    torch.clamp(cols + sr, max=grid_col - 1) - torch.clamp(cols - sr, min=0) + 1
                ).to(g2d.dtype)
                w = g2d / cnts[None, :]
                zero_col = torch.zeros(grid_row, 1, dtype=g2d.dtype, device=dev)
                cs = torch.cumsum(torch.cat([zero_col, w], dim=1), dim=1)
                lo_idx = torch.clamp(cols - sr, min=0)
                hi_idx = torch.clamp(cols + sr + 1, max=grid_col)
                smoothed = cs[:, hi_idx] - cs[:, lo_idx]
            return smoothed.contiguous().cpu().numpy().ravel()
    if axis_h:
        # H demand spreads along rows.
        rows = np.arange(grid_row, dtype=np.int64)
        lp = np.maximum(rows - sr, 0)
        up = np.minimum(rows + sr, grid_row - 1)
        cnts = (up - lp + 1).astype(np.float64)
        weighted = grid_2d / cnts[:, None]
        events = np.zeros((grid_row + 1, grid_col), dtype=np.float64)
        np.add.at(events, lp, weighted)
        np.subtract.at(events, up + 1, weighted)
        smoothed = np.cumsum(events, axis=0)[:grid_row]
    else:
        # V demand spreads along columns.
        cols = np.arange(grid_col, dtype=np.int64)
        lp = np.maximum(cols - sr, 0)
        up = np.minimum(cols + sr, grid_col - 1)
        cnts = (up - lp + 1).astype(np.float64)
        weighted = grid_2d / cnts[None, :]
        events = np.zeros((grid_row, grid_col + 1), dtype=np.float64)
        row_idx = np.broadcast_to(
            np.arange(grid_row, dtype=np.int64)[:, None], (grid_row, grid_col)
        )
        col_lp = np.broadcast_to(lp[None, :], (grid_row, grid_col))
        col_up = np.broadcast_to((up + 1)[None, :], (grid_row, grid_col))
        np.add.at(events, (row_idx, col_lp), weighted)
        np.subtract.at(events, (row_idx, col_up), weighted)
        smoothed = np.cumsum(events, axis=1)[:, :grid_col]
    return smoothed.ravel()


if HAS_NUMBA:

    @_numba_njit(cache=True, fastmath=False)
    def _apply_macro_routing_scatter_jit(
        V_macro_flat,
        H_macro_flat,
        bl_row,
        bl_col,
        ur_row,
        ur_col,
        x_min,
        x_max,
        y_min,
        y_max,
        grid_w,
        grid_h,
        grid_col,
        valloc,
        halloc,
    ):
        """Add hard-macro routing blockage with numba."""
        n = bl_row.shape[0]
        tol = 1e-5
        # Add overlap-based blockage.
        for m in range(n):
            r0 = bl_row[m]
            r1 = ur_row[m]
            c0 = bl_col[m]
            c1 = ur_col[m]
            xmn = x_min[m]
            xmx = x_max[m]
            ymn = y_min[m]
            ymx = y_max[m]
            for rr in range(r0, r1 + 1):
                cy0 = grid_h * rr
                cy1 = grid_h * (rr + 1)
                yd = min(cy1, ymx) - max(cy0, ymn)
                if yd < 0.0:
                    yd = 0.0
                base = rr * grid_col
                for cc in range(c0, c1 + 1):
                    cx0 = grid_w * cc
                    cx1 = grid_w * (cc + 1)
                    xd = min(cx1, xmx) - max(cx0, xmn)
                    if xd < 0.0:
                        xd = 0.0
                    idx = base + cc
                    V_macro_flat[idx] += xd * valloc
                    H_macro_flat[idx] += yd * halloc
        # Correct partially covered top rows.
        for m in range(n):
            r0 = bl_row[m]
            r1 = ur_row[m]
            if r1 == r0:
                continue
            ymn = y_min[m]
            ymx = y_max[m]
            if not (
                abs((grid_h * (r0 + 1) - ymn) - grid_h) > tol
                or abs((ymx - grid_h * r1) - grid_h) > tol
            ):
                continue
            c0 = bl_col[m]
            c1 = ur_col[m]
            xmn = x_min[m]
            xmx = x_max[m]
            base = r1 * grid_col
            for cc in range(c0, c1 + 1):
                cx0 = grid_w * cc
                cx1 = grid_w * (cc + 1)
                xd = min(cx1, xmx) - max(cx0, xmn)
                if xd < 0.0:
                    xd = 0.0
                V_macro_flat[base + cc] -= xd * valloc
        # Correct partially covered right columns.
        for m in range(n):
            c0 = bl_col[m]
            c1 = ur_col[m]
            if c1 == c0:
                continue
            xmn = x_min[m]
            xmx = x_max[m]
            if not (
                abs((grid_w * (c0 + 1) - xmn) - grid_w) > tol
                or abs((xmx - grid_w * c1) - grid_w) > tol
            ):
                continue
            r0 = bl_row[m]
            r1 = ur_row[m]
            ymn = y_min[m]
            ymx = y_max[m]
            for rr in range(r0, r1 + 1):
                cy0 = grid_h * rr
                cy1 = grid_h * (rr + 1)
                yd = min(cy1, ymx) - max(cy0, ymn)
                if yd < 0.0:
                    yd = 0.0
                H_macro_flat[rr * grid_col + c1] -= yd * halloc


def _apply_macro_routing(
    V_macro_flat: np.ndarray,
    H_macro_flat: np.ndarray,
    hard_x: np.ndarray,
    hard_y: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    grid_w: float,
    grid_h: float,
    grid_row: int,
    grid_col: int,
    vrouting_alloc: float,
    hrouting_alloc: float,
) -> None:
    """Add hard-macro routing blockage to the congestion grids."""
    x_min = hard_x - half_w
    x_max = hard_x + half_w
    y_min = hard_y - half_h
    y_max = hard_y + half_h
    bl_col = np.floor(x_min / grid_w).astype(np.int64)
    bl_row = np.floor(y_min / grid_h).astype(np.int64)
    ur_col = np.floor(x_max / grid_w).astype(np.int64)
    ur_row = np.floor(y_max / grid_h).astype(np.int64)
    # Skip macros fully outside the grid.
    in_bounds = (ur_row >= 0) & (ur_col >= 0) & (bl_row <= grid_row - 1) & (bl_col <= grid_col - 1)
    bl_col = np.clip(bl_col, 0, grid_col - 1)
    bl_row = np.clip(bl_row, 0, grid_row - 1)
    ur_col = np.clip(ur_col, 0, grid_col - 1)
    ur_row = np.clip(ur_row, 0, grid_row - 1)

    if not in_bounds.any():
        return

    # Work only on macros that touch the grid.
    sel = np.where(in_bounds)[0]
    bl_col_s = bl_col[sel]
    bl_row_s = bl_row[sel]
    ur_col_s = ur_col[sel]
    ur_row_s = ur_row[sel]
    x_min_s = x_min[sel]
    x_max_s = x_max[sel]
    y_min_s = y_min[sel]
    y_max_s = y_max[sel]

    if HAS_NUMBA:
        _apply_macro_routing_scatter_jit(
            V_macro_flat,
            H_macro_flat,
            np.ascontiguousarray(bl_row_s),
            np.ascontiguousarray(bl_col_s),
            np.ascontiguousarray(ur_row_s),
            np.ascontiguousarray(ur_col_s),
            np.ascontiguousarray(x_min_s),
            np.ascontiguousarray(x_max_s),
            np.ascontiguousarray(y_min_s),
            np.ascontiguousarray(y_max_s),
            float(grid_w),
            float(grid_h),
            int(grid_col),
            float(vrouting_alloc),
            float(hrouting_alloc),
        )
        return

    n_rows_per = (ur_row_s - bl_row_s + 1).astype(np.int64)
    n_cols_per = (ur_col_s - bl_col_s + 1).astype(np.int64)
    n_cells_per = n_rows_per * n_cols_per
    total = int(n_cells_per.sum())
    if total == 0:
        return

    # Enumerate every grid cell touched by each macro.
    macro_idx = np.repeat(np.arange(sel.size, dtype=np.int64), n_cells_per)
    cum = np.zeros(sel.size + 1, dtype=np.int64)
    np.cumsum(n_cells_per, out=cum[1:])
    local_idx = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], n_cells_per)
    n_cols_per_cell = n_cols_per[macro_idx]
    row_off = local_idx // n_cols_per_cell
    col_off = local_idx - row_off * n_cols_per_cell

    rr_g = bl_row_s[macro_idx] + row_off
    cc_g = bl_col_s[macro_idx] + col_off
    flat_idx = rr_g * grid_col + cc_g

    # Measure overlap with each touched cell.
    cell_xmin = grid_w * cc_g.astype(np.float64)
    cell_xmax = grid_w * (cc_g + 1).astype(np.float64)
    cell_ymin = grid_h * rr_g.astype(np.float64)
    cell_ymax = grid_h * (rr_g + 1).astype(np.float64)
    x_max_pc = x_max_s[macro_idx]
    x_min_pc = x_min_s[macro_idx]
    y_max_pc = y_max_s[macro_idx]
    y_min_pc = y_min_s[macro_idx]
    x_dist = np.minimum(cell_xmax, x_max_pc) - np.maximum(cell_xmin, x_min_pc)
    y_dist = np.minimum(cell_ymax, y_max_pc) - np.maximum(cell_ymin, y_min_pc)
    np.maximum(x_dist, 0.0, out=x_dist)
    np.maximum(y_dist, 0.0, out=y_dist)

    np.add.at(V_macro_flat, flat_idx, x_dist * vrouting_alloc)
    np.add.at(H_macro_flat, flat_idx, y_dist * hrouting_alloc)

    # Match the reference correction for partially covered edge cells.
    tol = 1e-5
    spans_rows = ur_row_s != bl_row_s
    bot_partial = np.abs((grid_h * (bl_row_s + 1) - y_min_s) - grid_h) > tol
    top_partial = np.abs((y_max_s - grid_h * ur_row_s) - grid_h) > tol
    partial_v = spans_rows & (bot_partial | top_partial)
    if partial_v.any():
        ur_off_per_macro = (ur_row_s - bl_row_s).astype(np.int64)
        mask = partial_v[macro_idx] & (row_off == ur_off_per_macro[macro_idx])
        if mask.any():
            np.subtract.at(
                V_macro_flat,
                ur_row_s[macro_idx[mask]] * grid_col + cc_g[mask],
                x_dist[mask] * vrouting_alloc,
            )

    spans_cols = ur_col_s != bl_col_s
    left_partial = np.abs((grid_w * (bl_col_s + 1) - x_min_s) - grid_w) > tol
    right_partial = np.abs((x_max_s - grid_w * ur_col_s) - grid_w) > tol
    partial_h = spans_cols & (left_partial | right_partial)
    if partial_h.any():
        ur_coff_per_macro = (ur_col_s - bl_col_s).astype(np.int64)
        mask = partial_h[macro_idx] & (col_off == ur_coff_per_macro[macro_idx])
        if mask.any():
            np.subtract.at(
                H_macro_flat,
                rr_g[mask] * grid_col + ur_col_s[macro_idx[mask]],
                y_dist[mask] * hrouting_alloc,
            )


def _build_net_routing_struct(plc, net_indices: np.ndarray):
    """Precompute routing data for a set of nets."""
    if len(net_indices) == 0:
        return None
    cache = plc._cong_cache
    wl_cache = plc._wl_vec_cache
    starts = cache["starts"]
    lengths = cache["lengths"]
    net_weights = wl_cache["net_weights"]
    inv = wl_cache["ref_inv"]

    starts_s = starts[net_indices]
    lengths_s = lengths[net_indices]
    total_pins = int(lengths_s.sum())
    if total_pins == 0:
        return None
    cumsum_lens = np.concatenate([[0], np.cumsum(lengths_s)[:-1]]).astype(np.int64)
    sub_pin_idx_in_flat = np.repeat(starts_s, lengths_s) + (
        np.arange(total_pins, dtype=np.int64) - np.repeat(cumsum_lens, lengths_s)
    )
    struct = {
        "sub_pin_idx_in_flat": sub_pin_idx_in_flat,
        "pin_ref_local": inv[sub_pin_idx_in_flat],
        "weights_unsigned": net_weights[net_indices],
        "mask_l2": lengths_s == 2,
        "mask_l3": lengths_s == 3,
    }
    struct["local_starts_l2"] = cumsum_lens[struct["mask_l2"]] if struct["mask_l2"].any() else None
    struct["local_starts_l3"] = cumsum_lens[struct["mask_l3"]] if struct["mask_l3"].any() else None

    struct["l4"] = None
    mask_l4 = lengths_s >= 4
    if mask_l4.any():
        sub_idx_big = np.where(mask_l4)[0]
        starts_big_local = cumsum_lens[sub_idx_big]
        sink_lens_local = lengths_s[sub_idx_big] - 1
        sink_total_local = int(sink_lens_local.sum())
        if sink_total_local > 0:
            B_local = sub_idx_big.size
            net_local_ids_local = np.repeat(np.arange(B_local, dtype=np.int64), sink_lens_local)
            cum_sink_starts_local = np.zeros(B_local + 1, dtype=np.int64)
            np.cumsum(sink_lens_local, out=cum_sink_starts_local[1:])
            offset_in_sinks_local = np.arange(sink_total_local, dtype=np.int64) - np.repeat(
                cum_sink_starts_local[:-1], sink_lens_local
            )
            global_pin_idx_local = (starts_big_local + 1)[
                net_local_ids_local
            ] + offset_in_sinks_local
            struct["l4"] = {
                "sub_idx_big": sub_idx_big,
                "starts_big_local": starts_big_local,
                "B_local": B_local,
                "net_local_ids_local": net_local_ids_local,
                "global_pin_idx_local": global_pin_idx_local,
            }
    return struct


def _apply_net_routing_struct(
    plc, struct, weight_mult: float, H_flat: np.ndarray, V_flat: np.ndarray
):
    """Apply routing for a precomputed net set at current positions."""
    if struct is None:
        return None
    wl_cache = plc._wl_vec_cache
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)

    sub_pin_idx_in_flat = struct["sub_pin_idx_in_flat"]
    pin_ref_local = struct["pin_ref_local"]
    pos_cache = _ensure_pos_cache(plc)
    unique_ref = wl_cache["unique_ref"]
    x_off = wl_cache["x_off"]
    y_off = wl_cache["y_off"]
    pin_x = pos_cache[unique_ref[pin_ref_local], 0] + x_off[sub_pin_idx_in_flat]
    pin_y = pos_cache[unique_ref[pin_ref_local], 1] + y_off[sub_pin_idx_in_flat]
    pin_col = np.clip((pin_x / grid_w).astype(np.int64), 0, grid_col - 1)
    pin_row = np.clip((pin_y / grid_h).astype(np.int64), 0, grid_row - 1)
    pin_gcell = pin_row * grid_col + pin_col

    weights_sub = struct["weights_unsigned"] * weight_mult

    bucket_2_src: list = []
    bucket_2_snk: list = []
    bucket_2_w: list = []
    bucket_3_g0: list = []
    bucket_3_g1: list = []
    bucket_3_g2: list = []
    bucket_3_w: list = []
    used_l2_l3_jit = False

    if HAS_NUMBA:
        starts2_jit = (
            struct["local_starts_l2"]
            if struct["local_starts_l2"] is not None
            else np.empty(0, dtype=np.int64)
        )
        starts3_jit = (
            struct["local_starts_l3"]
            if struct["local_starts_l3"] is not None
            else np.empty(0, dtype=np.int64)
        )
        if starts2_jit.size or starts3_jit.size:
            _apply_route_struct_l2_l3_jit(
                H_flat,
                V_flat,
                np.ascontiguousarray(pin_gcell, dtype=np.int64),
                np.ascontiguousarray(starts2_jit, dtype=np.int64),
                np.ascontiguousarray(weights_sub[struct["mask_l2"]], dtype=np.float64),
                np.ascontiguousarray(starts3_jit, dtype=np.int64),
                np.ascontiguousarray(weights_sub[struct["mask_l3"]], dtype=np.float64),
                int(grid_col),
            )
            used_l2_l3_jit = True

    mask_l2 = struct["mask_l2"]
    local_starts_l2 = struct["local_starts_l2"]
    if local_starts_l2 is not None and not used_l2_l3_jit:
        src2 = pin_gcell[local_starts_l2]
        snk2 = pin_gcell[local_starts_l2 + 1]
        sub_mask = src2 != snk2
        if sub_mask.any():
            bucket_2_src.append(src2[sub_mask])
            bucket_2_snk.append(snk2[sub_mask])
            bucket_2_w.append(weights_sub[mask_l2][sub_mask])

    mask_l3 = struct["mask_l3"]
    local_starts_l3 = struct["local_starts_l3"]
    if local_starts_l3 is not None and not used_l2_l3_jit:
        g0 = pin_gcell[local_starts_l3]
        g1 = pin_gcell[local_starts_l3 + 1]
        g2 = pin_gcell[local_starts_l3 + 2]
        eq01 = g0 == g1
        eq02 = g0 == g2
        eq12 = g1 == g2
        eq_count = eq01.astype(np.int64) + eq02.astype(np.int64) + eq12.astype(np.int64)
        uniq2 = eq_count == 1
        uniq3 = eq_count == 0
        if uniq2.any():
            src_2 = g0[uniq2]
            sink_2 = np.where(eq01[uniq2], g2[uniq2], g1[uniq2])
            bucket_2_src.append(src_2)
            bucket_2_snk.append(sink_2)
            bucket_2_w.append(weights_sub[mask_l3][uniq2])
        if uniq3.any():
            bucket_3_g0.append(g0[uniq3])
            bucket_3_g1.append(g1[uniq3])
            bucket_3_g2.append(g2[uniq3])
            bucket_3_w.append(weights_sub[mask_l3][uniq3])

    l4 = struct["l4"]
    if l4 is not None:
        sub_idx_big = l4["sub_idx_big"]
        starts_big_local = l4["starts_big_local"]
        B_local = l4["B_local"]
        net_local_ids_local = l4["net_local_ids_local"]
        global_pin_idx_local = l4["global_pin_idx_local"]
        src_gcells_big = pin_gcell[starts_big_local]
        sink_gcells = pin_gcell[global_pin_idx_local]
        mask_not_src = sink_gcells != src_gcells_big[net_local_ids_local]
        if mask_not_src.any():
            nli_ns = net_local_ids_local[mask_not_src]
            sg_ns = sink_gcells[mask_not_src]
            order = np.lexsort((sg_ns, nli_ns))
            nli_sorted = nli_ns[order]
            sg_sorted = sg_ns[order]
            keep = np.empty(sg_sorted.size, dtype=bool)
            keep[0] = True
            if sg_sorted.size > 1:
                keep[1:] = (nli_sorted[1:] != nli_sorted[:-1]) | (sg_sorted[1:] != sg_sorted[:-1])
            nli_uniq = nli_sorted[keep]
            sg_uniq = sg_sorted[keep]
            uniq_sink_counts = np.bincount(nli_uniq, minlength=B_local)
            n_uniq_total = 1 + uniq_sink_counts
            net_is_3 = n_uniq_total == 3
            net_is_starlike = ~net_is_3
            mask_starlike = net_is_starlike[nli_uniq]
            if mask_starlike.any():
                nli_emit = nli_uniq[mask_starlike]
                bucket_2_src.append(src_gcells_big[nli_emit])
                bucket_2_snk.append(sg_uniq[mask_starlike])
                bucket_2_w.append(weights_sub[sub_idx_big[nli_emit]])
            if net_is_3.any():
                cum_counts = np.cumsum(uniq_sink_counts)
                net3_ids = np.where(net_is_3)[0]
                ends = cum_counts[net3_ids]
                bucket_3_g0.append(src_gcells_big[net3_ids])
                bucket_3_g1.append(sg_uniq[ends - 2])
                bucket_3_g2.append(sg_uniq[ends - 1])
                bucket_3_w.append(weights_sub[sub_idx_big[net3_ids]])

    if bucket_2_src:
        src_flat = np.concatenate(bucket_2_src)
        snk_flat = np.concatenate(bucket_2_snk)
        w_arr = np.concatenate(bucket_2_w)
        _apply_2pin_routing(
            H_flat,
            V_flat,
            src_flat // grid_col,
            src_flat % grid_col,
            snk_flat // grid_col,
            snk_flat % grid_col,
            w_arr,
            grid_row,
            grid_col,
        )
    if bucket_3_g0:
        g0_arr = np.concatenate(bucket_3_g0)
        g1_arr = np.concatenate(bucket_3_g1)
        g2_arr = np.concatenate(bucket_3_g2)
        w_arr3 = np.concatenate(bucket_3_w)
        _apply_3pin_routing_vec(H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col)

    return (int(pin_row.min()), int(pin_row.max()), int(pin_col.min()), int(pin_col.max()))


def _apply_net_routing_subset(
    plc,
    net_indices: np.ndarray,
    weight_mult: float,
    H_flat: np.ndarray,
    V_flat: np.ndarray,
) -> None:
    """Add or remove routing for only the given nets."""
    if len(net_indices) == 0:
        return None

    cache = plc._cong_cache
    wl_cache = plc._wl_vec_cache
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)

    starts = cache["starts"]
    lengths = cache["lengths"]
    net_weights = wl_cache["net_weights"]

    # Gather only pins from touched nets.
    starts_s = starts[net_indices]
    lengths_s = lengths[net_indices]
    total_pins = int(lengths_s.sum())
    if total_pins == 0:
        return None
    cumsum_lens = np.concatenate([[0], np.cumsum(lengths_s)[:-1]]).astype(np.int64)
    sub_pin_idx_in_flat = np.repeat(starts_s, lengths_s) + (
        np.arange(total_pins, dtype=np.int64) - np.repeat(cumsum_lens, lengths_s)
    )

    # Convert touched pins to grid cells.
    pos_cache = _ensure_pos_cache(plc)
    unique_ref = wl_cache["unique_ref"]
    inv = wl_cache["ref_inv"]
    x_off = wl_cache["x_off"]
    y_off = wl_cache["y_off"]
    pin_ref_local = inv[sub_pin_idx_in_flat]
    pin_x = pos_cache[unique_ref[pin_ref_local], 0] + x_off[sub_pin_idx_in_flat]
    pin_y = pos_cache[unique_ref[pin_ref_local], 1] + y_off[sub_pin_idx_in_flat]
    pin_col = np.clip((pin_x / grid_w).astype(np.int64), 0, grid_col - 1)
    pin_row = np.clip((pin_y / grid_h).astype(np.int64), 0, grid_row - 1)
    pin_gcell = pin_row * grid_col + pin_col

    weights_sub = net_weights[net_indices] * weight_mult

    bucket_2_src: list = []
    bucket_2_snk: list = []
    bucket_2_w: list = []
    bucket_3_g0: list = []
    bucket_3_g1: list = []
    bucket_3_g2: list = []
    bucket_3_w: list = []

    # Two-pin nets.
    mask_l2 = lengths_s == 2
    if mask_l2.any():
        local_starts_l2 = cumsum_lens[mask_l2]
        src2 = pin_gcell[local_starts_l2]
        snk2 = pin_gcell[local_starts_l2 + 1]
        sub_mask = src2 != snk2
        if sub_mask.any():
            bucket_2_src.append(src2[sub_mask])
            bucket_2_snk.append(snk2[sub_mask])
            bucket_2_w.append(weights_sub[mask_l2][sub_mask])

    # Three-pin nets.
    mask_l3 = lengths_s == 3
    if mask_l3.any():
        local_starts_l3 = cumsum_lens[mask_l3]
        g0 = pin_gcell[local_starts_l3]
        g1 = pin_gcell[local_starts_l3 + 1]
        g2 = pin_gcell[local_starts_l3 + 2]
        eq01 = g0 == g1
        eq02 = g0 == g2
        eq12 = g1 == g2
        eq_count = eq01.astype(np.int64) + eq02.astype(np.int64) + eq12.astype(np.int64)
        uniq2 = eq_count == 1
        uniq3 = eq_count == 0
        if uniq2.any():
            src_2 = g0[uniq2]
            sink_2 = np.where(eq01[uniq2], g2[uniq2], g1[uniq2])
            bucket_2_src.append(src_2)
            bucket_2_snk.append(sink_2)
            bucket_2_w.append(weights_sub[mask_l3][uniq2])
        if uniq3.any():
            bucket_3_g0.append(g0[uniq3])
            bucket_3_g1.append(g1[uniq3])
            bucket_3_g2.append(g2[uniq3])
            bucket_3_w.append(weights_sub[mask_l3][uniq3])

    # Larger nets.
    mask_l4 = lengths_s >= 4
    if mask_l4.any():
        sub_idx_big = np.where(mask_l4)[0]
        starts_big_local = cumsum_lens[sub_idx_big]
        lengths_big_local = lengths_s[sub_idx_big]
        sink_lens_local = lengths_big_local - 1
        sink_total_local = int(sink_lens_local.sum())
        src_gcells_big = pin_gcell[starts_big_local]
        if sink_total_local > 0:
            B_local = sub_idx_big.size
            net_local_ids_local = np.repeat(np.arange(B_local, dtype=np.int64), sink_lens_local)
            cum_sink_starts_local = np.zeros(B_local + 1, dtype=np.int64)
            np.cumsum(sink_lens_local, out=cum_sink_starts_local[1:])
            offset_in_sinks_local = np.arange(sink_total_local, dtype=np.int64) - np.repeat(
                cum_sink_starts_local[:-1], sink_lens_local
            )
            global_pin_idx_local = (starts_big_local + 1)[
                net_local_ids_local
            ] + offset_in_sinks_local
            sink_gcells = pin_gcell[global_pin_idx_local]
            mask_not_src = sink_gcells != src_gcells_big[net_local_ids_local]
            if mask_not_src.any():
                nli_ns = net_local_ids_local[mask_not_src]
                sg_ns = sink_gcells[mask_not_src]
                order = np.lexsort((sg_ns, nli_ns))
                nli_sorted = nli_ns[order]
                sg_sorted = sg_ns[order]
                keep = np.empty(sg_sorted.size, dtype=bool)
                keep[0] = True
                if sg_sorted.size > 1:
                    keep[1:] = (nli_sorted[1:] != nli_sorted[:-1]) | (
                        sg_sorted[1:] != sg_sorted[:-1]
                    )
                nli_uniq = nli_sorted[keep]
                sg_uniq = sg_sorted[keep]
                uniq_sink_counts = np.bincount(nli_uniq, minlength=B_local)
                n_uniq_total = 1 + uniq_sink_counts
                net_is_3 = n_uniq_total == 3
                net_is_starlike = ~net_is_3
                mask_starlike = net_is_starlike[nli_uniq]
                if mask_starlike.any():
                    nli_emit = nli_uniq[mask_starlike]
                    bucket_2_src.append(src_gcells_big[nli_emit])
                    bucket_2_snk.append(sg_uniq[mask_starlike])
                    bucket_2_w.append(weights_sub[sub_idx_big[nli_emit]])
                if net_is_3.any():
                    cum_counts = np.cumsum(uniq_sink_counts)
                    net3_ids = np.where(net_is_3)[0]
                    ends = cum_counts[net3_ids]
                    bucket_3_g0.append(src_gcells_big[net3_ids])
                    bucket_3_g1.append(sg_uniq[ends - 2])
                    bucket_3_g2.append(sg_uniq[ends - 1])
                    bucket_3_w.append(weights_sub[sub_idx_big[net3_ids]])

    if bucket_2_src:
        src_flat = np.concatenate(bucket_2_src)
        snk_flat = np.concatenate(bucket_2_snk)
        w_arr = np.concatenate(bucket_2_w)
        _apply_2pin_routing(
            H_flat,
            V_flat,
            src_flat // grid_col,
            src_flat % grid_col,
            snk_flat // grid_col,
            snk_flat % grid_col,
            w_arr,
            grid_row,
            grid_col,
        )
    if bucket_3_g0:
        g0_arr = np.concatenate(bucket_3_g0)
        g1_arr = np.concatenate(bucket_3_g1)
        g2_arr = np.concatenate(bucket_3_g2)
        w_arr3 = np.concatenate(bucket_3_w)
        _apply_3pin_routing_vec(H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col)

    return (int(pin_row.min()), int(pin_row.max()), int(pin_col.min()), int(pin_col.max()))


def _apply_macro_routing_subset(
    plc,
    macro_subset: np.ndarray,
    weight_mult: float,
    V_macro_flat: np.ndarray,
    H_macro_flat: np.ndarray,
) -> None:
    """Add or remove blockage for selected hard macros."""
    if len(macro_subset) == 0:
        return
    cache = plc._cong_cache
    if cache["n_hard"] == 0:
        return
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)

    pos_cache = _ensure_pos_cache(plc)
    hard_indices_arr = cache.get("hard_indices_arr")
    if hard_indices_arr is None:
        hard_indices_arr = np.asarray(cache["hard_indices"], dtype=np.int64)
        cache["hard_indices_arr"] = hard_indices_arr
    sub_module_indices = hard_indices_arr[macro_subset]
    hard_x = pos_cache[sub_module_indices, 0]
    hard_y = pos_cache[sub_module_indices, 1]
    hw_sub = cache["hard_half_w"][macro_subset]
    hh_sub = cache["hard_half_h"][macro_subset]

    # Flip the capacity sign to subtract blockage.
    _apply_macro_routing(
        V_macro_flat,
        H_macro_flat,
        hard_x,
        hard_y,
        hw_sub,
        hh_sub,
        grid_w,
        grid_h,
        grid_row,
        grid_col,
        float(plc.vrouting_alloc) * weight_mult,
        float(plc.hrouting_alloc) * weight_mult,
    )


def _vectorized_get_routing(plc) -> None:
    """Compute routing congestion with vectorized numpy code."""
    cache = plc._cong_cache
    wl = plc._wl_vec_cache

    # Refresh grid geometry.
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)
    plc.grid_width = grid_w
    plc.grid_height = grid_h
    grid_v_routes = grid_w * plc.vroutes_per_micron
    grid_h_routes = grid_h * plc.hroutes_per_micron
    plc.grid_v_routes = grid_v_routes
    plc.grid_h_routes = grid_h_routes

    n_cells = grid_row * grid_col
    H_flat = np.zeros(n_cells, dtype=np.float64)
    V_flat = np.zeros(n_cells, dtype=np.float64)
    H_macro_flat = np.zeros(n_cells, dtype=np.float64)
    V_macro_flat = np.zeros(n_cells, dtype=np.float64)

    n_nets = wl["n_nets"]
    if n_nets > 0:
        # Read pin positions from the cached module positions.
        unique_ref = wl["unique_ref"]
        pos_cache = _ensure_pos_cache(plc)
        node_x = pos_cache[unique_ref, 0]
        node_y = pos_cache[unique_ref, 1]
        inv = wl["ref_inv"]
        pin_x = node_x[inv] + wl["x_off"]
        pin_y = node_y[inv] + wl["y_off"]
        # Map pins to grid cells.
        pin_col = np.clip((pin_x / grid_w).astype(np.int64), 0, grid_col - 1)
        pin_row = np.clip((pin_y / grid_h).astype(np.int64), 0, grid_row - 1)
        pin_gcell = pin_row * grid_col + pin_col

        net_weights = wl["net_weights"]

        # Group nets by how many distinct grid cells they touch.
        bucket_2_src_flat: "list[np.ndarray]" = []
        bucket_2_snk_flat: "list[np.ndarray]" = []
        bucket_2_w_arrs: "list[np.ndarray]" = []

        bucket_3_g0: "list[np.ndarray]" = []
        bucket_3_g1: "list[np.ndarray]" = []
        bucket_3_g2: "list[np.ndarray]" = []
        bucket_3_w_arrs: "list[np.ndarray]" = []

        # Two-pin nets.
        idx2 = cache["idx2"]
        if idx2.size > 0:
            src2 = pin_gcell[cache["s2"]]
            snk2 = pin_gcell[cache["s2p1"]]
            mask = src2 != snk2
            if mask.any():
                bucket_2_src_flat.append(src2[mask])
                bucket_2_snk_flat.append(snk2[mask])
                bucket_2_w_arrs.append(net_weights[idx2][mask])

        # Three-pin nets may collapse to two cells.
        idx3 = cache["idx3"]
        if idx3.size > 0:
            g0 = pin_gcell[cache["s3"]]
            g1 = pin_gcell[cache["s3p1"]]
            g2 = pin_gcell[cache["s3p2"]]
            eq01 = g0 == g1
            eq02 = g0 == g2
            eq12 = g1 == g2
            eq_count = eq01.astype(np.int64) + eq02.astype(np.int64) + eq12.astype(np.int64)
            uniq2 = eq_count == 1
            uniq3 = eq_count == 0
            mask2 = uniq2
            if mask2.any():
                src_2 = g0[mask2]
                sink_2 = np.where(eq01[mask2], g2[mask2], g1[mask2])
                bucket_2_src_flat.append(src_2)
                bucket_2_snk_flat.append(sink_2)
                bucket_2_w_arrs.append(net_weights[idx3][mask2])
            if uniq3.any():
                idx3_uniq3 = idx3[uniq3]
                bucket_3_g0.append(g0[uniq3])
                bucket_3_g1.append(g1[uniq3])
                bucket_3_g2.append(g2[uniq3])
                bucket_3_w_arrs.append(net_weights[idx3_uniq3])

        # Larger nets route from the source to each unique sink cell.
        idx_big = cache["idx_big"]
        if idx_big.size > 0:
            starts_big = cache["starts_big"]
            sink_total = cache["sink_total"]
            src_gcells_big = pin_gcell[starts_big]
            if sink_total > 0:
                B = cache["B_big"]
                net_local_ids = cache["net_local_ids"]
                global_pin_idx = cache["global_pin_idx"]
                sink_gcells = pin_gcell[global_pin_idx]
                # Ignore sinks already in the source cell.
                mask_not_src = sink_gcells != src_gcells_big[net_local_ids]
                if mask_not_src.any():
                    nli_ns = net_local_ids[mask_not_src]
                    sg_ns = sink_gcells[mask_not_src]
                    # Deduplicate sink cells per net.
                    order = np.lexsort((sg_ns, nli_ns))
                    nli_sorted = nli_ns[order]
                    sg_sorted = sg_ns[order]
                    keep = np.empty(sg_sorted.size, dtype=bool)
                    keep[0] = True
                    if sg_sorted.size > 1:
                        keep[1:] = (nli_sorted[1:] != nli_sorted[:-1]) | (
                            sg_sorted[1:] != sg_sorted[:-1]
                        )
                    nli_uniq = nli_sorted[keep]
                    sg_uniq = sg_sorted[keep]
                    uniq_sink_counts = np.bincount(nli_uniq, minlength=B)
                    n_uniq_total = 1 + uniq_sink_counts
                    net_is_3 = n_uniq_total == 3
                    net_is_starlike = ~net_is_3
                    mask_starlike = net_is_starlike[nli_uniq]
                    if mask_starlike.any():
                        nli_emit = nli_uniq[mask_starlike]
                        bucket_2_src_flat.append(src_gcells_big[nli_emit])
                        bucket_2_snk_flat.append(sg_uniq[mask_starlike])
                        bucket_2_w_arrs.append(net_weights[idx_big[nli_emit]])
                    if net_is_3.any():
                        cum_counts = np.cumsum(uniq_sink_counts)
                        net3_ids = np.where(net_is_3)[0]
                        ends = cum_counts[net3_ids]
                        bucket_3_g0.append(src_gcells_big[net3_ids])
                        bucket_3_g1.append(sg_uniq[ends - 2])
                        bucket_3_g2.append(sg_uniq[ends - 1])
                        bucket_3_w_arrs.append(net_weights[idx_big[net3_ids]])

        if bucket_2_src_flat:
            src_flat = np.concatenate(bucket_2_src_flat)
            snk_flat = np.concatenate(bucket_2_snk_flat)
            w_arr = np.concatenate(bucket_2_w_arrs)
            _apply_2pin_routing(
                H_flat,
                V_flat,
                src_flat // grid_col,
                src_flat % grid_col,
                snk_flat // grid_col,
                snk_flat % grid_col,
                w_arr,
                grid_row,
                grid_col,
            )
        if bucket_3_g0:
            g0_arr = np.concatenate(bucket_3_g0)
            g1_arr = np.concatenate(bucket_3_g1)
            g2_arr = np.concatenate(bucket_3_g2)
            w_arr3 = np.concatenate(bucket_3_w_arrs)
            _apply_3pin_routing_vec(
                H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col
            )

    # Hard-macro blockage.
    n_hard = cache["n_hard"]
    if n_hard > 0:
        hard_indices = cache["hard_indices"]
        hard_indices_arr = cache.get("hard_indices_arr")
        if hard_indices_arr is None:
            hard_indices_arr = np.asarray(hard_indices, dtype=np.int64)
            cache["hard_indices_arr"] = hard_indices_arr
        pos_cache = _ensure_pos_cache(plc)
        hard_x = pos_cache[hard_indices_arr, 0]
        hard_y = pos_cache[hard_indices_arr, 1]
        _apply_macro_routing(
            V_macro_flat,
            H_macro_flat,
            hard_x,
            hard_y,
            cache["hard_half_w"],
            cache["hard_half_h"],
            grid_w,
            grid_h,
            grid_row,
            grid_col,
            float(plc.vrouting_alloc),
            float(plc.hrouting_alloc),
        )

    # Normalize by routing capacity.
    H_flat /= grid_h_routes
    V_flat /= grid_v_routes
    H_macro_flat /= grid_h_routes
    V_macro_flat /= grid_v_routes

    # Smooth net routing and add macro blockage.
    smooth_range = int(plc.smooth_range)
    if smooth_range > 0:
        V_flat = _smooth_routing_cong_vec(V_flat, grid_row, grid_col, smooth_range, axis_h=False)
        H_flat = _smooth_routing_cong_vec(H_flat, grid_row, grid_col, smooth_range, axis_h=True)

    V_total = V_flat + V_macro_flat
    H_total = H_flat + H_macro_flat

    # Downstream code consumes numpy arrays directly.
    plc.V_routing_cong = V_total
    plc.H_routing_cong = H_total
    plc.V_macro_routing_cong = V_macro_flat
    plc.H_macro_routing_cong = H_macro_flat
    plc.FLAG_UPDATE_CONGESTION = False
