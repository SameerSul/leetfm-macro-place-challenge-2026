"""Vectorized routing demand and smoothing helpers."""

import numpy as np
import torch
from macro_place.benchmark import Benchmark

from placer.config import HAS_NUMBA, _GPU_DEVICE, _USE_GPU, _numba_njit
from placer.plc.placement import _ensure_pos_cache
from placer.scoring.wirelength import _build_wl_cache

def _build_cong_cache(plc, benchmark: Benchmark):
    """One-time precomputation for vectorized get_routing.

    Reuses _wl_vec_cache for per-pin (ref_idx, offset) arrays. Adds:
      - per-net weight (derived from driver's get_weight)
      - per-net pin-range starts/lengths
      - hard macro arrays (idx, half_w, half_h)
    """
    if hasattr(plc, "_cong_cache"):
        return plc._cong_cache
    wl = _build_wl_cache(plc)

    # Per-net pin lengths (end - start). Last net runs to n_pins.
    starts = wl["net_starts"]
    n_nets = len(starts)
    n_pins = wl["n_pins"]
    if n_nets == 0:
        ends = np.zeros(0, dtype=np.int64)
    else:
        ends = np.concatenate([starts[1:], np.array([n_pins], dtype=np.int64)])
    lengths = ends - starts

    # Hard macro arrays
    hard_indices = list(plc.hard_macro_indices)
    n_hard = len(hard_indices)
    hard_half_w = np.empty(n_hard, dtype=np.float64)
    hard_half_h = np.empty(n_hard, dtype=np.float64)
    for k, idx in enumerate(hard_indices):
        m = plc.modules_w_pins[idx]
        hard_half_w[k] = float(m.get_width()) * 0.5
        hard_half_h[k] = float(m.get_height()) * 0.5

    # Pre-compute the dispatch structures that depend only on net topology (not
    # positions), so _vectorized_get_routing doesn't rebuild them every call.
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
            net_local_ids_cache = np.repeat(
                np.arange(B_cache, dtype=np.int64), sink_lens_cache
            )
            cum_sink_starts_cache = np.zeros(B_cache + 1, dtype=np.int64)
            np.cumsum(sink_lens_cache, out=cum_sink_starts_cache[1:])
            offset_in_sinks_cache = (
                np.arange(sink_total_cache, dtype=np.int64)
                - np.repeat(cum_sink_starts_cache[:-1], sink_lens_cache)
            )
            global_pin_idx_cache = (
                (starts_big_cache + 1)[net_local_ids_cache] + offset_in_sinks_cache
            )
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
        # B4 dispatch caches:
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
        """JIT explicit-loop H-strip add: H_flat[r, lo:hi] += w for each entry.
        ~3-5× the numpy bincount+cumsum version on typical strip-batch sizes.
        Same accumulation order as a sequential add (np.bincount can differ
        at float precision; this matches a left-to-right scalar accumulation).
        Within the move-verifier tolerance (≤4.4e-16) on all tested
        benchmarks."""
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
        """JIT explicit-loop V-strip add: V_flat[lo:hi, c] += w for each entry."""
        n = col.shape[0]
        for k in range(n):
            c = col[k]
            lo = row_lo[k]
            hi = row_hi[k]
            w = weight[k]
            for r in range(lo, hi):
                V_flat[r * grid_col + c] += w


def _apply_h_strips_batch(H_flat: np.ndarray, row: np.ndarray,
                           col_lo: np.ndarray, col_hi: np.ndarray,
                           weight: np.ndarray, grid_row: int, grid_col: int) -> None:
    """Batched H-strip add: for each entry, H_flat[row, col_lo:col_hi] += weight.
    Dispatches to a numba-JIT explicit-loop version when available (3-5×
    faster on typical batches), otherwise falls back to a vectorized numpy
    bincount+cumsum (the previous S3 implementation).

    The JIT path's float-accumulation order differs from the bincount path
    (left-to-right vs separate +/− bincount sums), but both are within
    machine-eps of the same true value; the move verifier accepts ≤4.4e-16
    drift and this is well under that."""
    if row.size == 0:
        return
    if HAS_NUMBA:
        # Numba dispatch - ensures contiguous int64 / float64 arrays.
        _apply_h_strips_batch_jit(
            H_flat,
            np.ascontiguousarray(row, dtype=np.int64),
            np.ascontiguousarray(col_lo, dtype=np.int64),
            np.ascontiguousarray(col_hi, dtype=np.int64),
            np.ascontiguousarray(weight, dtype=np.float64),
            int(grid_col),
        )
        return
    # Numpy fallback (S3 bincount path).
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


def _apply_v_strips_batch(V_flat: np.ndarray, col: np.ndarray,
                           row_lo: np.ndarray, row_hi: np.ndarray,
                           weight: np.ndarray, grid_row: int, grid_col: int) -> None:
    """Batched V-strip add: for each entry, V_flat[row_lo:row_hi, col] += weight.
    See `_apply_h_strips_batch` for the JIT-vs-numpy dispatch rationale."""
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
    # Numpy fallback (S3 bincount path).
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


def _apply_2pin_routing(H_flat: np.ndarray, V_flat: np.ndarray,
                         src_row: np.ndarray, src_col: np.ndarray,
                         snk_row: np.ndarray, snk_col: np.ndarray,
                         weight: np.ndarray, grid_row: int, grid_col: int) -> None:
    """Batched 2-pin L-routing via difference-array prefix-sum.

    Mirrors __two_pin_net_routing exactly:
      H_routing[source_row, col_min : col_max] += weight
      V_routing[row_min : row_max, sink_col]  += weight
    """
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
    def _apply_3pin_routing_vec_jit(H_flat, V_flat, g0_flat, g1_flat, g2_flat,
                                      weights, grid_col):
        """JIT explicit-per-net 3-pin routing.

        Bit-equivalent to `_apply_3pin_routing_vec_numpy`. For each net: decode
        the 3 flat gcells to (row, col), sort by (col asc, row asc), classify
        into the 4 routing cases (case 4 / T-route needs a row-sorted re-sort),
        and apply H/V strip adds directly into the flat arrays. Collapsing the
        numpy version's per-case mask/concat/strip-batch fanout into one tight
        JIT'd per-net loop removes its overhead (it was 38% of per-move time);
        accumulation order stays within the verifier tolerance (<=4.4e-16).
        """
        n = g0_flat.shape[0]
        for k in range(n):
            # Decode flat → (row, col) for each of the 3 pins.
            ya = g0_flat[k] // grid_col; xa = g0_flat[k] % grid_col
            yb = g1_flat[k] // grid_col; xb = g1_flat[k] % grid_col
            yc = g2_flat[k] // grid_col; xc = g2_flat[k] % grid_col
            w = weights[k]

            # Sort 3 (x, y) pairs by (x asc, y asc). Manual 3-pass swap -
            # equivalent to a 3-element insertion / bubble sort.
            x1 = xa; y1 = ya
            x2 = xb; y2 = yb
            x3 = xc; y3 = yc
            if x1 > x2 or (x1 == x2 and y1 > y2):
                tx = x1; x1 = x2; x2 = tx
                ty = y1; y1 = y2; y2 = ty
            if x2 > x3 or (x2 == x3 and y2 > y3):
                tx = x2; x2 = x3; x3 = tx
                ty = y2; y2 = y3; y3 = ty
            if x1 > x2 or (x1 == x2 and y1 > y2):
                tx = x1; x1 = x2; x2 = tx
                ty = y1; y1 = y2; y2 = ty

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
            x1t = xa; y1t = ya
            x2t = xb; y2t = yb
            x3t = xc; y3t = yc
            if y1t > y2t or (y1t == y2t and x1t > x2t):
                tx = x1t; x1t = x2t; x2t = tx
                ty = y1t; y1t = y2t; y2t = ty
            if y2t > y3t or (y2t == y3t and x2t > x3t):
                tx = x2t; x2t = x3t; x3t = tx
                ty = y2t; y2t = y3t; y3t = ty
            if y1t > y2t or (y1t == y2t and x1t > x2t):
                tx = x1t; x1t = x2t; x2t = tx
                ty = y1t; y1t = y2t; y2t = ty

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


def _apply_3pin_routing_vec(H_flat: np.ndarray, V_flat: np.ndarray,
                             g0_flat: np.ndarray, g1_flat: np.ndarray,
                             g2_flat: np.ndarray, weights: np.ndarray,
                             grid_row: int, grid_col: int) -> None:
    """Dispatcher: JIT explicit-per-net when numba is available, else the
    vectorized numpy gather/scatter (see `_apply_3pin_routing_vec_numpy`)."""
    if g0_flat.size == 0:
        return
    if HAS_NUMBA:
        _apply_3pin_routing_vec_jit(
            H_flat, V_flat,
            np.ascontiguousarray(g0_flat, dtype=np.int64),
            np.ascontiguousarray(g1_flat, dtype=np.int64),
            np.ascontiguousarray(g2_flat, dtype=np.int64),
            np.ascontiguousarray(weights, dtype=np.float64),
            int(grid_col),
        )
        return
    _apply_3pin_routing_vec_numpy(H_flat, V_flat, g0_flat, g1_flat, g2_flat,
                                    weights, grid_row, grid_col)


def _apply_3pin_routing_vec_numpy(H_flat: np.ndarray, V_flat: np.ndarray,
                                    g0_flat: np.ndarray, g1_flat: np.ndarray,
                                    g2_flat: np.ndarray, weights: np.ndarray,
                                    grid_row: int, grid_col: int) -> None:
    """Numpy gather/scatter fallback for three-pin routing.

    Each net's 3 gcells are first sorted by (col, row). Cases 1-3 use that
    ordering; case 4 (T-routing) requires a second sort by (row, col).
    """
    if g0_flat.size == 0:
        return
    n = g0_flat.size
    # Convert flat → (row, col) and stack
    y_all = np.stack([g0_flat // grid_col, g1_flat // grid_col, g2_flat // grid_col], axis=1).astype(np.int64)
    x_all = np.stack([g0_flat % grid_col, g1_flat % grid_col, g2_flat % grid_col], axis=1).astype(np.int64)
    w = np.asarray(weights, dtype=np.float64)
    # Sort each net's 3 points by (col asc, row asc)
    BIG = int(max(grid_row, grid_col)) + 16
    key = x_all * BIG + y_all
    order = np.argsort(key, axis=1, kind="stable")
    y = np.take_along_axis(y_all, order, axis=1)
    x = np.take_along_axis(x_all, order, axis=1)
    y1 = y[:, 0]; y2 = y[:, 1]; y3 = y[:, 2]
    x1 = x[:, 0]; x2 = x[:, 1]; x3 = x[:, 2]

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
        h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
        h_rows.append(y2[m]); h_los.append(x2[m]); h_his.append(x3[m]); h_ws.append(wm)
        # V x2 [min(y1,y2)..max(y1,y2)], x3 [min(y2,y3)..max(y2,y3)]
        v_cols.append(x2[m]); v_los.append(np.minimum(y1[m], y2[m])); v_his.append(np.maximum(y1[m], y2[m])); v_ws.append(wm)
        v_cols.append(x3[m]); v_los.append(np.minimum(y2[m], y3[m])); v_his.append(np.maximum(y2[m], y3[m])); v_ws.append(wm)

    if case2.any():
        m = case2
        wm = w[m]
        h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
        v_cols.append(x2[m]); v_los.append(y1[m]); v_his.append(np.maximum(y2[m], y3[m])); v_ws.append(wm)

    if case3.any():
        m = case3
        wm = w[m]
        h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
        h_rows.append(y2[m]); h_los.append(x2[m]); h_his.append(x3[m]); h_ws.append(wm)
        v_cols.append(x2[m]); v_los.append(np.minimum(y2[m], y1[m])); v_his.append(np.maximum(y2[m], y1[m])); v_ws.append(wm)

    if case4.any():
        m = case4
        wm = w[m]
        # Re-sort by (row asc, col asc) - matches scalar's `sorted(temp)` which
        # sorts tuples lexicographically by (row, col).
        y_t = y_all[m]; x_t = x_all[m]
        key_t = y_t * BIG + x_t
        order_t = np.argsort(key_t, axis=1, kind="stable")
        y_t = np.take_along_axis(y_t, order_t, axis=1)
        x_t = np.take_along_axis(x_t, order_t, axis=1)
        y1t = y_t[:, 0]; y2t = y_t[:, 1]; y3t = y_t[:, 2]
        x1t = x_t[:, 0]; x2t = x_t[:, 1]; x3t = x_t[:, 2]
        xmin_t = np.minimum(np.minimum(x1t, x2t), x3t)
        xmax_t = np.maximum(np.maximum(x1t, x2t), x3t)
        h_rows.append(y2t); h_los.append(xmin_t); h_his.append(xmax_t); h_ws.append(wm)
        v_cols.append(x1t); v_los.append(np.minimum(y1t, y2t)); v_his.append(np.maximum(y1t, y2t)); v_ws.append(wm)
        v_cols.append(x3t); v_los.append(np.minimum(y2t, y3t)); v_his.append(np.maximum(y2t, y3t)); v_ws.append(wm)

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
            _apply_v_strips_batch(V_flat, cols[nz], rlos[nz], rhis[nz], ws_v[nz], grid_row, grid_col)


def _smooth_routing_cong_vec(routing_flat: np.ndarray, grid_row: int,
                              grid_col: int, smooth_range: int,
                              axis_h: bool) -> np.ndarray:
    """Vectorized __smooth_routing_cong. For each cell, distribute its value
    across a 1D window of `2*smooth_range + 1` cells (clipped at the grid
    edges, divided by the window size). axis_h=True smooths along columns
    (V-routing-style); axis_h=False smooths along rows (H-routing-style).

    Implemented via difference-array prefix-sum trick - O(grid + events)
    rather than O(grid × window).

    NOTE on the reference's quirk: __smooth_routing_cong smooths V along
    COLUMNS (the V loop iterates `for ptr in range(lp, rp+1)` with lp/rp
    clamped to grid_col), and smooths H along ROWS (H loop iterates rows
    via `for ptr in range(lp, up+1)`). The naming is swapped vs intuition
    - V_routing gets a column-axis smooth, H_routing gets a row-axis smooth.
    `axis_h=False` means smooth along the column axis (V-style behavior);
    `axis_h=True` means smooth along the row axis (H-style).

    GPU path (gpu-testing branch): uses pure torch.cumsum + fancy indexing in
    place of np.add.at + np.cumsum. Mathematically equivalent to the difference-
    array approach: smoothed[p] = sum of w[r] for all source rows r whose window
    contains p, where w[r] = grid_2d[r] / window_width[r].

    index_add_ is NOT used: on DirectML it falls back to CPU with a UserWarning.
    torch.cumsum and fancy integer indexing (cs[hi] - cs[lo]) are DirectML-native.
    """
    grid_2d = routing_flat.reshape(grid_row, grid_col)
    sr = smooth_range
    if _USE_GPU:
        with torch.no_grad():
            dev = _GPU_DEVICE
            g2d = torch.from_numpy(grid_2d).to(dev)
            if axis_h:
                # H-style: smooth along rows. Window for source row r is
                # [max(0,r-sr), min(gr-1,r+sr)]; cumsum gives
                # smoothed[p] = cs[min(gr,p+sr+1)] - cs[max(0,p-sr)].
                rows = torch.arange(grid_row, dtype=torch.int64, device=dev)
                cnts = (torch.clamp(rows + sr, max=grid_row - 1)
                        - torch.clamp(rows - sr, min=0) + 1).to(g2d.dtype)
                w = g2d / cnts[:, None]
                zero_row = torch.zeros(1, grid_col, dtype=g2d.dtype, device=dev)
                cs = torch.cumsum(torch.cat([zero_row, w], dim=0), dim=0)  # [gr+1, gc]
                lo_idx = torch.clamp(rows - sr, min=0)
                hi_idx = torch.clamp(rows + sr + 1, max=grid_row)
                smoothed = cs[hi_idx] - cs[lo_idx]  # [gr, gc]
            else:
                # V-style: smooth along cols (axis 1).
                # For source col c the window spans [max(0,c-sr), min(gc-1,c+sr)].
                cols = torch.arange(grid_col, dtype=torch.int64, device=dev)
                cnts = (torch.clamp(cols + sr, max=grid_col - 1)
                        - torch.clamp(cols - sr, min=0) + 1).to(g2d.dtype)
                w = g2d / cnts[None, :]
                zero_col = torch.zeros(grid_row, 1, dtype=g2d.dtype, device=dev)
                cs = torch.cumsum(torch.cat([zero_col, w], dim=1), dim=1)  # [gr, gc+1]
                lo_idx = torch.clamp(cols - sr, min=0)
                hi_idx = torch.clamp(cols + sr + 1, max=grid_col)
                smoothed = cs[:, hi_idx] - cs[:, lo_idx]  # [gr, gc]
            return smoothed.contiguous().cpu().numpy().ravel()
    if axis_h:
        # H-style: each cell spreads across rows [max(0,row-sr), min(gr-1,row+sr)]
        # via a difference array along axis 0 (events +=/-= weighted at lp/up+1).
        # Edge clipping makes lp/up collide, so np.add.at accumulates duplicates.
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
        # V-style: each cell spreads across cols [max(0,col-sr), min(gc-1,col+sr)]
        # via a difference array along axis 1, vectorized with advanced indexing.
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


def _apply_macro_routing(V_macro_flat: np.ndarray, H_macro_flat: np.ndarray,
                          hard_x: np.ndarray, hard_y: np.ndarray,
                          half_w: np.ndarray, half_h: np.ndarray,
                          grid_w: float, grid_h: float,
                          grid_row: int, grid_col: int,
                          vrouting_alloc: float, hrouting_alloc: float) -> None:
    """Per-hard-macro routing contribution. Matches __macro_route_over_grid_cell.

    For each cell the macro overlaps:
      x_dist = horizontal overlap between macro and cell
      y_dist = vertical overlap between macro and cell
      V_macro_cong[r,c] += x_dist * vrouting_alloc
      H_macro_cong[r,c] += y_dist * hrouting_alloc
    Plus PARTIAL_OVERLAP corrections (subtract from top row / right col)
    that fire when the macro's bounding-box doesn't fully cover the boundary
    cell along the relevant axis.
    """
    x_min = hard_x - half_w
    x_max = hard_x + half_w
    y_min = hard_y - half_h
    y_max = hard_y + half_h
    bl_col = np.floor(x_min / grid_w).astype(np.int64)
    bl_row = np.floor(y_min / grid_h).astype(np.int64)
    ur_col = np.floor(x_max / grid_w).astype(np.int64)
    ur_row = np.floor(y_max / grid_h).astype(np.int64)
    # Mirror reference's OOB skip
    in_bounds = (ur_row >= 0) & (ur_col >= 0) & (bl_row <= grid_row - 1) & (bl_col <= grid_col - 1)
    bl_col = np.clip(bl_col, 0, grid_col - 1)
    bl_row = np.clip(bl_row, 0, grid_row - 1)
    ur_col = np.clip(ur_col, 0, grid_col - 1)
    ur_row = np.clip(ur_row, 0, grid_row - 1)

    if not in_bounds.any():
        return

    # Restrict to in-bounds macros
    sel = np.where(in_bounds)[0]
    bl_col_s = bl_col[sel]
    bl_row_s = bl_row[sel]
    ur_col_s = ur_col[sel]
    ur_row_s = ur_row[sel]
    x_min_s = x_min[sel]
    x_max_s = x_max[sel]
    y_min_s = y_min[sel]
    y_max_s = y_max[sel]

    n_rows_per = (ur_row_s - bl_row_s + 1).astype(np.int64)
    n_cols_per = (ur_col_s - bl_col_s + 1).astype(np.int64)
    n_cells_per = n_rows_per * n_cols_per
    total = int(n_cells_per.sum())
    if total == 0:
        return

    # Per-cell (macro_idx, row_offset, col_offset) via flat enumeration
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

    # Per-cell overlap distances (x varies with col, y varies with row)
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

    # ----- PARTIAL_OVERLAP corrections ----------------------------------
    # Mirror the scalar reference: a macro spanning >1 row with a partial top/
    # bottom row (y_dist != grid_h) subtracts per-column x_dist from V at ur_row.
    # Only bl_row/ur_row can be partial (middle rows are full). Symmetric for H.
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
    """Idea 2 (2026-05-29): the position-INDEPENDENT structure of the subset
    routing apply - which flat pins the touched nets span, their macro refs,
    net-length classification (2/3/≥4-pin), and the ≥4-pin sink index layout.

    Depends only on the netlist topology + `net_indices`, NOT on placement, so it
    can be built once per net-set and reused across the −1/+1 applies of a move
    AND across moves of the same macro (the scorer caches it per module). The
    position-dependent fill is done by `_apply_net_routing_struct`. Returns None
    for an empty/pinless set. Mirrors the bookkeeping of `_apply_net_routing_subset`
    exactly (same arrays), just hoisted out of the per-call path.
    """
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
    sub_pin_idx_in_flat = (
        np.repeat(starts_s, lengths_s)
        + (np.arange(total_pins, dtype=np.int64) - np.repeat(cumsum_lens, lengths_s))
    )
    struct = {
        "sub_pin_idx_in_flat": sub_pin_idx_in_flat,
        "pin_ref_local": inv[sub_pin_idx_in_flat],
        "weights_unsigned": net_weights[net_indices],
        "mask_l2": lengths_s == 2,
        "mask_l3": lengths_s == 3,
    }
    struct["local_starts_l2"] = (
        cumsum_lens[struct["mask_l2"]] if struct["mask_l2"].any() else None)
    struct["local_starts_l3"] = (
        cumsum_lens[struct["mask_l3"]] if struct["mask_l3"].any() else None)

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
            offset_in_sinks_local = (
                np.arange(sink_total_local, dtype=np.int64)
                - np.repeat(cum_sink_starts_local[:-1], sink_lens_local)
            )
            global_pin_idx_local = (starts_big_local + 1)[net_local_ids_local] + offset_in_sinks_local
            struct["l4"] = {
                "sub_idx_big": sub_idx_big,
                "starts_big_local": starts_big_local,
                "B_local": B_local,
                "net_local_ids_local": net_local_ids_local,
                "global_pin_idx_local": global_pin_idx_local,
            }
    return struct


def _apply_net_routing_struct(plc, struct, weight_mult: float,
                              H_flat: np.ndarray, V_flat: np.ndarray):
    """Idea 2: position-DEPENDENT routing apply using a prebuilt topology struct.
    Computes pin_gcell from current positions, dispatches the 2/3/≥4-pin fills
    with a signed weight, and returns the touched-pin bbox (or None). The math is
    identical to `_apply_net_routing_subset`'s second half - only the
    position-independent bookkeeping has been hoisted into the struct.
    """
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

    mask_l2 = struct["mask_l2"]
    local_starts_l2 = struct["local_starts_l2"]
    if local_starts_l2 is not None:
        src2 = pin_gcell[local_starts_l2]
        snk2 = pin_gcell[local_starts_l2 + 1]
        sub_mask = src2 != snk2
        if sub_mask.any():
            bucket_2_src.append(src2[sub_mask])
            bucket_2_snk.append(snk2[sub_mask])
            bucket_2_w.append(weights_sub[mask_l2][sub_mask])

    mask_l3 = struct["mask_l3"]
    local_starts_l3 = struct["local_starts_l3"]
    if local_starts_l3 is not None:
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
                keep[1:] = (
                    (nli_sorted[1:] != nli_sorted[:-1])
                    | (sg_sorted[1:] != sg_sorted[:-1])
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
            H_flat, V_flat,
            src_flat // grid_col, src_flat % grid_col,
            snk_flat // grid_col, snk_flat % grid_col,
            w_arr, grid_row, grid_col,
        )
    if bucket_3_g0:
        g0_arr = np.concatenate(bucket_3_g0)
        g1_arr = np.concatenate(bucket_3_g1)
        g2_arr = np.concatenate(bucket_3_g2)
        w_arr3 = np.concatenate(bucket_3_w)
        _apply_3pin_routing_vec(H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col)

    return (int(pin_row.min()), int(pin_row.max()),
            int(pin_col.min()), int(pin_col.max()))


def _apply_net_routing_subset(
    plc,
    net_indices: np.ndarray,
    weight_mult: float,
    H_flat: np.ndarray,
    V_flat: np.ndarray,
) -> None:
    """B3 phase 4 (2026-05-24): per-net routing contribution for a SUBSET of
    nets, applied to in-place flat arrays with a signed weight multiplier.

    Mirrors `_vectorized_get_routing`'s per-net dispatch (2-pin / 3-pin /
    ≥4-pin steiner), but operates on `net_indices` only. `weight_mult=+1`
    adds contributions; `weight_mult=-1` subtracts them (for delta updates
    when a swap changes the touched-net set's routing).

    Does NOT touch macro routing (use `_apply_macro_routing_subset`) and
    does NOT smooth (caller handles smoothing once per swap).

    Pin positions read from `plc._global_pos_cache` (B3 phase 1). For
    efficient subset processing, pin_gcell is computed only for the
    touched pins (a small fraction of total pins).

    Returns the (r_lo, r_hi, c_lo, c_hi) grid-cell bounding box of the touched
    pins (or None if nothing was applied). Every cell this call modifies lies
    within that box (routing fill stays inside the pin bbox), so the caller can
    re-smooth only those columns/rows for the incremental congestion cost.
    """
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

    # Gather just the touched-net pin indices in the flat pin array.
    starts_s = starts[net_indices]
    lengths_s = lengths[net_indices]
    total_pins = int(lengths_s.sum())
    if total_pins == 0:
        return None
    cumsum_lens = np.concatenate([[0], np.cumsum(lengths_s)[:-1]]).astype(np.int64)
    sub_pin_idx_in_flat = (
        np.repeat(starts_s, lengths_s)
        + (np.arange(total_pins, dtype=np.int64) - np.repeat(cumsum_lens, lengths_s))
    )

    # Compute pin_gcell ONLY for touched pins.
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
    pin_gcell = pin_row * grid_col + pin_col  # in the COMPACT subset pin order

    weights_sub = net_weights[net_indices] * weight_mult

    bucket_2_src: list = []
    bucket_2_snk: list = []
    bucket_2_w: list = []
    bucket_3_g0: list = []
    bucket_3_g1: list = []
    bucket_3_g2: list = []
    bucket_3_w: list = []

    # ------ length-2 nets in subset ------
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

    # ------ length-3 nets in subset ------
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

    # ------ length≥4 nets in subset ------
    mask_l4 = lengths_s >= 4
    if mask_l4.any():
        sub_idx_big = np.where(mask_l4)[0]
        starts_big_local = cumsum_lens[sub_idx_big]  # offsets in the SUBSET pin order
        lengths_big_local = lengths_s[sub_idx_big]
        sink_lens_local = lengths_big_local - 1
        sink_total_local = int(sink_lens_local.sum())
        src_gcells_big = pin_gcell[starts_big_local]
        if sink_total_local > 0:
            B_local = sub_idx_big.size
            net_local_ids_local = np.repeat(np.arange(B_local, dtype=np.int64), sink_lens_local)
            cum_sink_starts_local = np.zeros(B_local + 1, dtype=np.int64)
            np.cumsum(sink_lens_local, out=cum_sink_starts_local[1:])
            offset_in_sinks_local = (
                np.arange(sink_total_local, dtype=np.int64)
                - np.repeat(cum_sink_starts_local[:-1], sink_lens_local)
            )
            global_pin_idx_local = (starts_big_local + 1)[net_local_ids_local] + offset_in_sinks_local
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
                    keep[1:] = (
                        (nli_sorted[1:] != nli_sorted[:-1])
                        | (sg_sorted[1:] != sg_sorted[:-1])
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
            H_flat, V_flat,
            src_flat // grid_col, src_flat % grid_col,
            snk_flat // grid_col, snk_flat % grid_col,
            w_arr, grid_row, grid_col,
        )
    if bucket_3_g0:
        g0_arr = np.concatenate(bucket_3_g0)
        g1_arr = np.concatenate(bucket_3_g1)
        g2_arr = np.concatenate(bucket_3_g2)
        w_arr3 = np.concatenate(bucket_3_w)
        _apply_3pin_routing_vec(H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col)

    return (int(pin_row.min()), int(pin_row.max()),
            int(pin_col.min()), int(pin_col.max()))


def _apply_macro_routing_subset(
    plc,
    macro_subset: np.ndarray,
    weight_mult: float,
    V_macro_flat: np.ndarray,
    H_macro_flat: np.ndarray,
) -> None:
    """B3 phase 4: per-macro routing contribution for a SUBSET of hard macros.

    `macro_subset` is an int array of indices into `cong_cache["hard_indices"]`
    (i.e., hard-macro slot indices, not module indices).
    """
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

    # Apply with the requested sign. _apply_macro_routing uses
    # `vrouting_alloc * weight` style multipliers; we just flip the alloc
    # sign for subtraction. (Same effect as -1 on the additive output.)
    _apply_macro_routing(
        V_macro_flat, H_macro_flat, hard_x, hard_y,
        hw_sub, hh_sub,
        grid_w, grid_h, grid_row, grid_col,
        float(plc.vrouting_alloc) * weight_mult,
        float(plc.hrouting_alloc) * weight_mult,
    )


def _vectorized_get_routing(plc) -> None:
    """Drop-in replacement for plc.get_routing().

    Replaces the inner ~25-second Python loop on ibm10 with a vectorized
    numpy pipeline. Sets plc.V_routing_cong / H_routing_cong as Python lists
    (matching the reference's API - `get_horizontal/vertical_routing_congestion`
    return them directly).
    """
    cache = plc._cong_cache
    wl = plc._wl_vec_cache

    # Geometry refresh (matches reference)
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
        # Use the global pos cache instead of a per-node get_pos loop.
        unique_ref = wl["unique_ref"]
        pos_cache = _ensure_pos_cache(plc)
        node_x = pos_cache[unique_ref, 0]
        node_y = pos_cache[unique_ref, 1]
        inv = wl["ref_inv"]
        pin_x = node_x[inv] + wl["x_off"]
        pin_y = node_y[inv] + wl["y_off"]
        # Apply the patched grid-cell location: floor + clamp
        pin_col = np.clip((pin_x / grid_w).astype(np.int64), 0, grid_col - 1)
        pin_row = np.clip((pin_y / grid_h).astype(np.int64), 0, grid_row - 1)
        pin_gcell = pin_row * grid_col + pin_col  # flat per-pin gcell idx

        starts = cache["starts"]
        lengths = cache["lengths"]
        net_weights = wl["net_weights"]

        # Per-net dispatch (v1's np.unique-per-net loop was 663ms on ibm10).
        # Partition nets by pin count and vectorize the common cases:
        #   - length 2 (most nets): pure-numpy fast path.
        #   - length 3: vectorized unique-count classification → per-bucket batch.
        #   - length ≥4: batched dispatch (per-net Python for the rare tail).
        bucket_2_src_flat: "list[np.ndarray]" = []  # accumulators of flat src gcells
        bucket_2_snk_flat: "list[np.ndarray]" = []
        bucket_2_w_arrs: "list[np.ndarray]" = []

        # 3-pin buckets: 3 flat-gcell arrays + weights (all parallel).
        bucket_3_g0: "list[np.ndarray]" = []
        bucket_3_g1: "list[np.ndarray]" = []
        bucket_3_g2: "list[np.ndarray]" = []
        bucket_3_w_arrs: "list[np.ndarray]" = []

        # --- length-2 vectorized fast path -----------------------------------
        # B4: read pre-cached idx2/s2/s2p1 from cong_cache (topology-fixed).
        idx2 = cache["idx2"]
        if idx2.size > 0:
            src2 = pin_gcell[cache["s2"]]
            snk2 = pin_gcell[cache["s2p1"]]
            mask = src2 != snk2  # same-cell pins → no routing
            if mask.any():
                bucket_2_src_flat.append(src2[mask])
                bucket_2_snk_flat.append(snk2[mask])
                bucket_2_w_arrs.append(net_weights[idx2][mask])

        # --- length-3 vectorized classification ------------------------------
        # Count unique gcells among (g0, g1, g2): all equal → skip; two distinct
        # → 2-pin edge; all distinct → 3-pin handler. (idx3/s3* from cache.)
        idx3 = cache["idx3"]
        if idx3.size > 0:
            g0 = pin_gcell[cache["s3"]]      # driver
            g1 = pin_gcell[cache["s3p1"]]
            g2 = pin_gcell[cache["s3p2"]]
            eq01 = g0 == g1
            eq02 = g0 == g2
            eq12 = g1 == g2
            # eq_count = #equal-pairs: 3 → all equal (skip); 1 → 2-pin edge;
            # 0 → all distinct (3-pin).
            eq_count = eq01.astype(np.int64) + eq02.astype(np.int64) + eq12.astype(np.int64)
            uniq2 = eq_count == 1
            uniq3 = eq_count == 0
            # 2-uniq: driver is g0; sink is whichever of g1/g2 differs from it.
            mask2 = uniq2
            if mask2.any():
                src_2 = g0[mask2]
                # Sink: g2 when eq01[mask2], else g1
                sink_2 = np.where(eq01[mask2], g2[mask2], g1[mask2])
                bucket_2_src_flat.append(src_2)
                bucket_2_snk_flat.append(sink_2)
                bucket_2_w_arrs.append(net_weights[idx3][mask2])
            # 3-uniq case: pass to vectorized 3-pin handler - directly append
            # the per-axis flat gcell arrays (no per-net Python loop).
            if uniq3.any():
                idx3_uniq3 = idx3[uniq3]
                bucket_3_g0.append(g0[uniq3])
                bucket_3_g1.append(g1[uniq3])
                bucket_3_g2.append(g2[uniq3])
                bucket_3_w_arrs.append(net_weights[idx3_uniq3])

        # --- length ≥4: vectorized batch dispatch ----------------------------
        # Per-net np.unique was 28k calls on ibm10 (~62ms). Instead build flat
        # (net_local_id, sink_gcell) pairs for ALL big nets, dedup via lexsort,
        # then dispatch by per-net unique count (source filtered out of sinks
        # first; n_uniq = 1 + #unique_sinks). Cache holds idx_big/starts_big/etc.
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
                # Drop sinks that equal the source gcell
                mask_not_src = sink_gcells != src_gcells_big[net_local_ids]
                if mask_not_src.any():
                    nli_ns = net_local_ids[mask_not_src]
                    sg_ns = sink_gcells[mask_not_src]
                    # Dedup per net via lexsort
                    order = np.lexsort((sg_ns, nli_ns))
                    nli_sorted = nli_ns[order]
                    sg_sorted = sg_ns[order]
                    keep = np.empty(sg_sorted.size, dtype=bool)
                    keep[0] = True
                    if sg_sorted.size > 1:
                        keep[1:] = (
                            (nli_sorted[1:] != nli_sorted[:-1])
                            | (sg_sorted[1:] != sg_sorted[:-1])
                        )
                    nli_uniq = nli_sorted[keep]
                    sg_uniq = sg_sorted[keep]
                    uniq_sink_counts = np.bincount(nli_uniq, minlength=B)
                    n_uniq_total = 1 + uniq_sink_counts
                    # Dispatch:
                    #   n_uniq_total == 3 → 3-pin steiner handler (Python loop on few nets)
                    #   n_uniq_total != 3 (covers 2 and ≥4) → emit (src, sink) edges
                    net_is_3 = n_uniq_total == 3
                    net_is_starlike = ~net_is_3
                    mask_starlike = net_is_starlike[nli_uniq]
                    if mask_starlike.any():
                        nli_emit = nli_uniq[mask_starlike]
                        bucket_2_src_flat.append(src_gcells_big[nli_emit])
                        bucket_2_snk_flat.append(sg_uniq[mask_starlike])
                        bucket_2_w_arrs.append(net_weights[idx_big[nli_emit]])
                    if net_is_3.any():
                        # The 2 unique sinks for each 3-pin net live in sg_uniq
                        # at positions [cum_count-2, cum_count-1]. Vectorize the
                        # gather instead of looping.
                        cum_counts = np.cumsum(uniq_sink_counts)
                        net3_ids = np.where(net_is_3)[0]
                        ends = cum_counts[net3_ids]
                        bucket_3_g0.append(src_gcells_big[net3_ids])
                        bucket_3_g1.append(sg_uniq[ends - 2])
                        bucket_3_g2.append(sg_uniq[ends - 1])
                        bucket_3_w_arrs.append(net_weights[idx_big[net3_ids]])

        # --- Apply 2-pin batch via difference-array --------------------------
        if bucket_2_src_flat:
            src_flat = np.concatenate(bucket_2_src_flat)
            snk_flat = np.concatenate(bucket_2_snk_flat)
            w_arr = np.concatenate(bucket_2_w_arrs)
            _apply_2pin_routing(
                H_flat, V_flat,
                src_flat // grid_col, src_flat % grid_col,
                snk_flat // grid_col, snk_flat % grid_col,
                w_arr, grid_row, grid_col,
            )
        # Apply 3-pin (vectorized batch)
        if bucket_3_g0:
            g0_arr = np.concatenate(bucket_3_g0)
            g1_arr = np.concatenate(bucket_3_g1)
            g2_arr = np.concatenate(bucket_3_g2)
            w_arr3 = np.concatenate(bucket_3_w_arrs)
            _apply_3pin_routing_vec(H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col)

    # Hard-macro routing contributions
    n_hard = cache["n_hard"]
    if n_hard > 0:
        # Use the global pos cache instead of a per-macro get_pos loop.
        hard_indices = cache["hard_indices"]
        hard_indices_arr = cache.get("hard_indices_arr")
        if hard_indices_arr is None:
            hard_indices_arr = np.asarray(hard_indices, dtype=np.int64)
            cache["hard_indices_arr"] = hard_indices_arr
        pos_cache = _ensure_pos_cache(plc)
        hard_x = pos_cache[hard_indices_arr, 0]
        hard_y = pos_cache[hard_indices_arr, 1]
        _apply_macro_routing(
            V_macro_flat, H_macro_flat, hard_x, hard_y,
            cache["hard_half_w"], cache["hard_half_h"],
            grid_w, grid_h, grid_row, grid_col,
            float(plc.vrouting_alloc), float(plc.hrouting_alloc),
        )

    # Normalize by routes-per-cell capacity
    H_flat /= grid_h_routes
    V_flat /= grid_v_routes
    H_macro_flat /= grid_h_routes
    V_macro_flat /= grid_v_routes

    # Smooth + combine
    smooth_range = int(plc.smooth_range)
    if smooth_range > 0:
        V_flat = _smooth_routing_cong_vec(V_flat, grid_row, grid_col, smooth_range, axis_h=False)
        H_flat = _smooth_routing_cong_vec(H_flat, grid_row, grid_col, smooth_range, axis_h=True)

    V_total = V_flat + V_macro_flat
    H_total = H_flat + H_macro_flat

    # Store as numpy arrays, not Python lists (saves ~2ms/call on .tolist()).
    # The patched get_congestion_cost and _routing_congestion_perturb both consume
    # them via numpy ops, so arrays work transparently.
    plc.V_routing_cong = V_total
    plc.H_routing_cong = H_total
    plc.V_macro_routing_cong = V_macro_flat
    plc.H_macro_routing_cong = H_macro_flat
    plc.FLAG_UPDATE_CONGESTION = False
