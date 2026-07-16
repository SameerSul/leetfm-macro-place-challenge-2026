"""Incremental proxy scorer used by local-search moves."""

import numpy as np
from macro_place.benchmark import Benchmark

from utils.config import HAS_NUMBA, _numba_njit
from placer.plc.placement import _ensure_pos_cache, _fast_set_placement
from placer.routing.apply import (
    _apply_macro_routing_subset,
    _apply_net_routing_struct,
    _apply_net_routing_subset,
    _build_net_routing_struct,
    _smooth_routing_cong_vec,
)
from placer.scoring.density import _build_density_cache, _vectorized_get_grid_cells_density
from placer.scoring.wirelength import _build_wl_cache

if HAS_NUMBA:
    from placer.routing.apply import _apply_route_struct_prepared_jit

    @_numba_njit(cache=True, fastmath=False)
    def _hpwl_subset_jit(
        net_indices, net_starts, net_lengths, ref_inv, x_off, y_off, node_x, node_y
    ):
        """Compute HPWL for selected nets with numba."""
        n = net_indices.shape[0]
        out = np.empty(n, dtype=np.float64)
        for k in range(n):
            net = net_indices[k]
            s = net_starts[net]
            L = net_lengths[net]
            if L == 0:
                out[k] = 0.0
                continue
            px = node_x[ref_inv[s]] + x_off[s]
            py = node_y[ref_inv[s]] + y_off[s]
            minx = px
            maxx = px
            miny = py
            maxy = py
            for j in range(1, L):
                p = s + j
                qx = node_x[ref_inv[p]] + x_off[p]
                qy = node_y[ref_inv[p]] + y_off[p]
                if qx < minx:
                    minx = qx
                if qx > maxx:
                    maxx = qx
                if qy < miny:
                    miny = qy
                if qy > maxy:
                    maxy = qy
            out[k] = (maxx - minx) + (maxy - miny)
        return out

    @_numba_njit(cache=True, fastmath=False)
    def _macro_occ_jit(bl_row, bl_col, ur_row, ur_col, x_min, x_max, y_min, y_max, gw, gh, gcol):
        """Return grid cells and overlap area for one macro."""
        nr = ur_row - bl_row + 1
        nc = ur_col - bl_col + 1
        flat = np.empty(nr * nc, dtype=np.int64)
        area = np.empty(nr * nc, dtype=np.float64)
        i = 0
        for r in range(bl_row, ur_row + 1):
            oy = min(gh * (r + 1), y_max) - max(gh * r, y_min)
            if oy < 0.0:
                oy = 0.0
            base = r * gcol
            for c in range(bl_col, ur_col + 1):
                ox = min(gw * (c + 1), x_max) - max(gw * c, x_min)
                if ox < 0.0:
                    ox = 0.0
                flat[i] = base + c
                area[i] = oy * ox
                i += 1
        return flat, area

    @_numba_njit(cache=True, fastmath=False)
    def _resmooth_h_cols_jit(
        raw_flat,
        smoothed,
        c_lo,
        c_hi,
        grid_row,
        grid_col,
        route_capacity,
        window_lo,
        window_hi,
        window_count,
        prefix,
    ):
        """Re-smooth selected horizontal columns with reusable prefix storage."""
        for c in range(c_lo, c_hi + 1):
            prefix[0] = 0.0
            for r in range(grid_row):
                value = raw_flat[r * grid_col + c] / route_capacity
                prefix[r + 1] = prefix[r] + value / window_count[r]
            for r in range(grid_row):
                smoothed[r, c] = prefix[window_hi[r] + 1] - prefix[window_lo[r]]

    @_numba_njit(cache=True, fastmath=False)
    def _resmooth_v_rows_jit(
        raw_flat,
        smoothed,
        r_lo,
        r_hi,
        grid_col,
        route_capacity,
        window_lo,
        window_hi,
        window_count,
        prefix,
    ):
        """Re-smooth selected vertical rows with reusable prefix storage."""
        for r in range(r_lo, r_hi + 1):
            base = r * grid_col
            prefix[0] = 0.0
            for c in range(grid_col):
                value = raw_flat[base + c] / route_capacity
                prefix[c + 1] = prefix[c] + value / window_count[c]
            for c in range(grid_col):
                smoothed[r, c] = prefix[window_hi[c] + 1] - prefix[window_lo[c]]

    @_numba_njit(cache=True, fastmath=False)
    def _batch_soft_route_grids_jit(
        xy,
        module,
        pos_cache,
        base_h,
        base_v,
        out_h,
        out_v,
        bboxes,
        pin_gcell,
        pin_module,
        pin_x_off,
        pin_y_off,
        starts2,
        weights2,
        starts3,
        weights3,
        starts4,
        lengths4,
        weights4,
        unique_sinks,
        three_g0,
        three_g1,
        three_g2,
        three_weights,
        grid_w,
        grid_h,
        grid_row,
        grid_col,
    ):
        """Build trial routing grids for a batch of soft-macro targets."""
        old_x = pos_cache[module, 0]
        old_y = pos_cache[module, 1]
        for batch_idx in range(xy.shape[0]):
            for cell in range(base_h.shape[0]):
                out_h[batch_idx, cell] = base_h[cell]
                out_v[batch_idx, cell] = base_v[cell]
            pos_cache[module, 0] = xy[batch_idx, 0]
            pos_cache[module, 1] = xy[batch_idx, 1]
            bbox = _apply_route_struct_prepared_jit(
                out_h[batch_idx],
                out_v[batch_idx],
                pos_cache,
                pin_gcell,
                pin_module,
                pin_x_off,
                pin_y_off,
                starts2,
                weights2,
                starts3,
                weights3,
                starts4,
                lengths4,
                weights4,
                1.0,
                unique_sinks,
                three_g0,
                three_g1,
                three_g2,
                three_weights,
                grid_w,
                grid_h,
                grid_row,
                grid_col,
            )
            bboxes[batch_idx, 0] = bbox[0]
            bboxes[batch_idx, 1] = bbox[1]
            bboxes[batch_idx, 2] = bbox[2]
            bboxes[batch_idx, 3] = bbox[3]
        pos_cache[module, 0] = old_x
        pos_cache[module, 1] = old_y

    @_numba_njit(cache=True, fastmath=False)
    def _batch_soft_swap_route_grids_jit(
        a_module,
        b_modules,
        a_xy,
        b_xy,
        pos_cache,
        base_h,
        base_v,
        out_h,
        out_v,
        bboxes,
        pin_offsets,
        pin_gcell,
        pin_module,
        pin_x_off,
        pin_y_off,
        starts2_offsets,
        starts2,
        weights2,
        starts3_offsets,
        starts3,
        weights3,
        starts4_offsets,
        starts4,
        lengths4,
        weights4,
        unique_sinks,
        three_g0,
        three_g1,
        three_g2,
        three_weights,
        grid_w,
        grid_h,
        grid_row,
        grid_col,
    ):
        """Build routing grids for a batch sharing one soft-swap endpoint."""
        for batch_idx in range(b_modules.shape[0]):
            for cell in range(base_h.shape[0]):
                out_h[batch_idx, cell] = base_h[cell]
                out_v[batch_idx, cell] = base_v[cell]

            pin_lo = pin_offsets[batch_idx]
            pin_hi = pin_offsets[batch_idx + 1]
            starts2_lo = starts2_offsets[batch_idx]
            starts2_hi = starts2_offsets[batch_idx + 1]
            starts3_lo = starts3_offsets[batch_idx]
            starts3_hi = starts3_offsets[batch_idx + 1]
            starts4_lo = starts4_offsets[batch_idx]
            starts4_hi = starts4_offsets[batch_idx + 1]
            bbox_old = _apply_route_struct_prepared_jit(
                out_h[batch_idx],
                out_v[batch_idx],
                pos_cache,
                pin_gcell[pin_lo:pin_hi],
                pin_module[pin_lo:pin_hi],
                pin_x_off[pin_lo:pin_hi],
                pin_y_off[pin_lo:pin_hi],
                starts2[starts2_lo:starts2_hi],
                weights2[starts2_lo:starts2_hi],
                starts3[starts3_lo:starts3_hi],
                weights3[starts3_lo:starts3_hi],
                starts4[starts4_lo:starts4_hi],
                lengths4[starts4_lo:starts4_hi],
                weights4[starts4_lo:starts4_hi],
                -1.0,
                unique_sinks,
                three_g0,
                three_g1,
                three_g2,
                three_weights,
                grid_w,
                grid_h,
                grid_row,
                grid_col,
            )

            b_module = b_modules[batch_idx]
            old_b_x = pos_cache[b_module, 0]
            old_b_y = pos_cache[b_module, 1]
            pos_cache[a_module, 0] = b_xy[batch_idx, 0]
            pos_cache[a_module, 1] = b_xy[batch_idx, 1]
            pos_cache[b_module, 0] = a_xy[0]
            pos_cache[b_module, 1] = a_xy[1]
            bbox_new = _apply_route_struct_prepared_jit(
                out_h[batch_idx],
                out_v[batch_idx],
                pos_cache,
                pin_gcell[pin_lo:pin_hi],
                pin_module[pin_lo:pin_hi],
                pin_x_off[pin_lo:pin_hi],
                pin_y_off[pin_lo:pin_hi],
                starts2[starts2_lo:starts2_hi],
                weights2[starts2_lo:starts2_hi],
                starts3[starts3_lo:starts3_hi],
                weights3[starts3_lo:starts3_hi],
                starts4[starts4_lo:starts4_hi],
                lengths4[starts4_lo:starts4_hi],
                weights4[starts4_lo:starts4_hi],
                1.0,
                unique_sinks,
                three_g0,
                three_g1,
                three_g2,
                three_weights,
                grid_w,
                grid_h,
                grid_row,
                grid_col,
            )
            pos_cache[a_module, 0] = a_xy[0]
            pos_cache[a_module, 1] = a_xy[1]
            pos_cache[b_module, 0] = old_b_x
            pos_cache[b_module, 1] = old_b_y

            bboxes[batch_idx, 0] = min(bbox_old[0], bbox_new[0])
            bboxes[batch_idx, 1] = max(bbox_old[1], bbox_new[1])
            bboxes[batch_idx, 2] = min(bbox_old[2], bbox_new[2])
            bboxes[batch_idx, 3] = max(bbox_old[3], bbox_new[3])

    @_numba_njit(cache=True, fastmath=False)
    def _apply_pair_macro_blockage_jit(
        v_grid,
        h_grid,
        a_x,
        a_y,
        a_half_w,
        a_half_h,
        b_x,
        b_y,
        b_half_w,
        b_half_h,
        has_b,
        weight,
        grid_w,
        grid_h,
        grid_row,
        grid_col,
        valloc,
        halloc,
    ):
        """Apply one or two hard-macro blockage rectangles in reference order."""
        tol = 1.0e-5

        # Match routing.apply._apply_macro_routing_scatter_jit: all overlap
        # contributions precede the top-row and right-column corrections.
        for macro_idx in range(2):
            if macro_idx == 1 and not has_b:
                break
            if macro_idx == 0:
                cx = a_x
                cy = a_y
                half_w = a_half_w
                half_h = a_half_h
            else:
                cx = b_x
                cy = b_y
                half_w = b_half_w
                half_h = b_half_h
            x_min = cx - half_w
            x_max = cx + half_w
            y_min = cy - half_h
            y_max = cy + half_h
            bl_col = int(np.floor(x_min / grid_w))
            bl_row = int(np.floor(y_min / grid_h))
            ur_col = int(np.floor(x_max / grid_w))
            ur_row = int(np.floor(y_max / grid_h))
            if ur_row < 0 or ur_col < 0 or bl_row >= grid_row or bl_col >= grid_col:
                continue
            bl_col = min(max(bl_col, 0), grid_col - 1)
            bl_row = min(max(bl_row, 0), grid_row - 1)
            ur_col = min(max(ur_col, 0), grid_col - 1)
            ur_row = min(max(ur_row, 0), grid_row - 1)
            for row in range(bl_row, ur_row + 1):
                cell_y_min = grid_h * row
                cell_y_max = grid_h * (row + 1)
                y_dist = min(cell_y_max, y_max) - max(cell_y_min, y_min)
                if y_dist < 0.0:
                    y_dist = 0.0
                base = row * grid_col
                for col in range(bl_col, ur_col + 1):
                    cell_x_min = grid_w * col
                    cell_x_max = grid_w * (col + 1)
                    x_dist = min(cell_x_max, x_max) - max(cell_x_min, x_min)
                    if x_dist < 0.0:
                        x_dist = 0.0
                    cell = base + col
                    v_grid[cell] += x_dist * valloc * weight
                    h_grid[cell] += y_dist * halloc * weight

        for macro_idx in range(2):
            if macro_idx == 1 and not has_b:
                break
            if macro_idx == 0:
                cx = a_x
                cy = a_y
                half_w = a_half_w
                half_h = a_half_h
            else:
                cx = b_x
                cy = b_y
                half_w = b_half_w
                half_h = b_half_h
            x_min = cx - half_w
            x_max = cx + half_w
            y_min = cy - half_h
            y_max = cy + half_h
            bl_col = int(np.floor(x_min / grid_w))
            bl_row = int(np.floor(y_min / grid_h))
            ur_col = int(np.floor(x_max / grid_w))
            ur_row = int(np.floor(y_max / grid_h))
            if ur_row < 0 or ur_col < 0 or bl_row >= grid_row or bl_col >= grid_col:
                continue
            bl_col = min(max(bl_col, 0), grid_col - 1)
            bl_row = min(max(bl_row, 0), grid_row - 1)
            ur_col = min(max(ur_col, 0), grid_col - 1)
            ur_row = min(max(ur_row, 0), grid_row - 1)
            if ur_row == bl_row:
                continue
            bottom_partial = abs((grid_h * (bl_row + 1) - y_min) - grid_h) > tol
            top_partial = abs((y_max - grid_h * ur_row) - grid_h) > tol
            if not (bottom_partial or top_partial):
                continue
            base = ur_row * grid_col
            for col in range(bl_col, ur_col + 1):
                cell_x_min = grid_w * col
                cell_x_max = grid_w * (col + 1)
                x_dist = min(cell_x_max, x_max) - max(cell_x_min, x_min)
                if x_dist < 0.0:
                    x_dist = 0.0
                v_grid[base + col] -= x_dist * valloc * weight

        for macro_idx in range(2):
            if macro_idx == 1 and not has_b:
                break
            if macro_idx == 0:
                cx = a_x
                cy = a_y
                half_w = a_half_w
                half_h = a_half_h
            else:
                cx = b_x
                cy = b_y
                half_w = b_half_w
                half_h = b_half_h
            x_min = cx - half_w
            x_max = cx + half_w
            y_min = cy - half_h
            y_max = cy + half_h
            bl_col = int(np.floor(x_min / grid_w))
            bl_row = int(np.floor(y_min / grid_h))
            ur_col = int(np.floor(x_max / grid_w))
            ur_row = int(np.floor(y_max / grid_h))
            if ur_row < 0 or ur_col < 0 or bl_row >= grid_row or bl_col >= grid_col:
                continue
            bl_col = min(max(bl_col, 0), grid_col - 1)
            bl_row = min(max(bl_row, 0), grid_row - 1)
            ur_col = min(max(ur_col, 0), grid_col - 1)
            ur_row = min(max(ur_row, 0), grid_row - 1)
            if ur_col == bl_col:
                continue
            left_partial = abs((grid_w * (bl_col + 1) - x_min) - grid_w) > tol
            right_partial = abs((x_max - grid_w * ur_col) - grid_w) > tol
            if not (left_partial or right_partial):
                continue
            for row in range(bl_row, ur_row + 1):
                cell_y_min = grid_h * row
                cell_y_max = grid_h * (row + 1)
                y_dist = min(cell_y_max, y_max) - max(cell_y_min, y_min)
                if y_dist < 0.0:
                    y_dist = 0.0
                h_grid[row * grid_col + ur_col] -= y_dist * halloc * weight

    @_numba_njit(cache=True, fastmath=False)
    def _batch_hard_swap_macro_grids_jit(
        a_xy,
        b_xy,
        a_half_w,
        a_half_h,
        b_half_w,
        b_half_h,
        b_is_hard,
        base_v,
        base_h,
        grid_w,
        grid_h,
        grid_row,
        grid_col,
        valloc,
        halloc,
        out_v,
        out_h,
    ):
        """Build per-candidate hard blockage grids for HH or HS swaps."""
        for batch_idx in range(b_xy.shape[0]):
            for cell in range(base_v.shape[0]):
                out_v[batch_idx, cell] = base_v[cell]
                out_h[batch_idx, cell] = base_h[cell]
            _apply_pair_macro_blockage_jit(
                out_v[batch_idx],
                out_h[batch_idx],
                a_xy[0],
                a_xy[1],
                a_half_w,
                a_half_h,
                b_xy[batch_idx, 0],
                b_xy[batch_idx, 1],
                b_half_w[batch_idx],
                b_half_h[batch_idx],
                b_is_hard,
                -1.0,
                grid_w,
                grid_h,
                grid_row,
                grid_col,
                valloc,
                halloc,
            )
            _apply_pair_macro_blockage_jit(
                out_v[batch_idx],
                out_h[batch_idx],
                b_xy[batch_idx, 0],
                b_xy[batch_idx, 1],
                a_half_w,
                a_half_h,
                a_xy[0],
                a_xy[1],
                b_half_w[batch_idx],
                b_half_h[batch_idx],
                b_is_hard,
                1.0,
                grid_w,
                grid_h,
                grid_row,
                grid_col,
                valloc,
                halloc,
            )

    @_numba_njit(cache=True, fastmath=False)
    def _batch_soft_congestion_values_jit(
        raw_h,
        raw_v,
        bboxes,
        base_h_smoothed,
        base_v_smoothed,
        h_macro,
        v_macro,
        h_capacity,
        v_capacity,
        h_window_lo,
        h_window_hi,
        h_window_count,
        v_window_lo,
        v_window_hi,
        v_window_count,
        smooth_range,
        grid_row,
        grid_col,
        out,
    ):
        """Create exact per-target congestion values from trial raw grids."""
        n_cells = grid_row * grid_col
        prefix_h = np.empty(grid_row + 1, dtype=np.float64)
        prefix_v = np.empty(grid_col + 1, dtype=np.float64)
        for batch_idx in range(raw_h.shape[0]):
            for cell in range(n_cells):
                out[batch_idx, cell] = base_v_smoothed[cell] + v_macro[cell] / v_capacity
                out[batch_idx, n_cells + cell] = base_h_smoothed[cell] + h_macro[cell] / h_capacity

            r_lo = bboxes[batch_idx, 0]
            r_hi = bboxes[batch_idx, 1]
            c_lo = bboxes[batch_idx, 2]
            c_hi = bboxes[batch_idx, 3]
            if smooth_range <= 0:
                for col in range(c_lo, c_hi + 1):
                    for row in range(grid_row):
                        cell = row * grid_col + col
                        out[batch_idx, n_cells + cell] = (
                            raw_h[batch_idx, cell] + h_macro[cell]
                        ) / h_capacity
                for row in range(r_lo, r_hi + 1):
                    base = row * grid_col
                    for col in range(grid_col):
                        cell = base + col
                        out[batch_idx, cell] = (raw_v[batch_idx, cell] + v_macro[cell]) / v_capacity
                continue

            for col in range(c_lo, c_hi + 1):
                prefix_h[0] = 0.0
                for row in range(grid_row):
                    cell = row * grid_col + col
                    value = raw_h[batch_idx, cell] / h_capacity
                    prefix_h[row + 1] = prefix_h[row] + value / h_window_count[row]
                for row in range(grid_row):
                    cell = row * grid_col + col
                    smoothed = prefix_h[h_window_hi[row] + 1] - prefix_h[h_window_lo[row]]
                    out[batch_idx, n_cells + cell] = smoothed + h_macro[cell] / h_capacity

            for row in range(r_lo, r_hi + 1):
                base = row * grid_col
                prefix_v[0] = 0.0
                for col in range(grid_col):
                    value = raw_v[batch_idx, base + col] / v_capacity
                    prefix_v[col + 1] = prefix_v[col] + value / v_window_count[col]
                for col in range(grid_col):
                    cell = base + col
                    smoothed = prefix_v[v_window_hi[col] + 1] - prefix_v[v_window_lo[col]]
                    out[batch_idx, cell] = smoothed + v_macro[cell] / v_capacity

    @_numba_njit(cache=True, fastmath=False)
    def _batch_hard_swap_congestion_values_jit(
        raw_h,
        raw_v,
        bboxes,
        base_h_smoothed,
        base_v_smoothed,
        h_macro,
        v_macro,
        h_capacity,
        v_capacity,
        h_window_lo,
        h_window_hi,
        h_window_count,
        v_window_lo,
        v_window_hi,
        v_window_count,
        smooth_range,
        grid_row,
        grid_col,
        out,
    ):
        """Create exact congestion values with candidate-specific hard blockage."""
        n_cells = grid_row * grid_col
        prefix_h = np.empty(grid_row + 1, dtype=np.float64)
        prefix_v = np.empty(grid_col + 1, dtype=np.float64)
        for batch_idx in range(raw_h.shape[0]):
            for cell in range(n_cells):
                out[batch_idx, cell] = base_v_smoothed[cell] + v_macro[batch_idx, cell] / v_capacity
                out[batch_idx, n_cells + cell] = (
                    base_h_smoothed[cell] + h_macro[batch_idx, cell] / h_capacity
                )

            r_lo = bboxes[batch_idx, 0]
            r_hi = bboxes[batch_idx, 1]
            c_lo = bboxes[batch_idx, 2]
            c_hi = bboxes[batch_idx, 3]
            if smooth_range <= 0:
                for col in range(c_lo, c_hi + 1):
                    for row in range(grid_row):
                        cell = row * grid_col + col
                        out[batch_idx, n_cells + cell] = (
                            raw_h[batch_idx, cell] + h_macro[batch_idx, cell]
                        ) / h_capacity
                for row in range(r_lo, r_hi + 1):
                    base = row * grid_col
                    for col in range(grid_col):
                        cell = base + col
                        out[batch_idx, cell] = (
                            raw_v[batch_idx, cell] + v_macro[batch_idx, cell]
                        ) / v_capacity
                continue

            for col in range(c_lo, c_hi + 1):
                prefix_h[0] = 0.0
                for row in range(grid_row):
                    cell = row * grid_col + col
                    value = raw_h[batch_idx, cell] / h_capacity
                    prefix_h[row + 1] = prefix_h[row] + value / h_window_count[row]
                for row in range(grid_row):
                    cell = row * grid_col + col
                    smoothed = prefix_h[h_window_hi[row] + 1] - prefix_h[h_window_lo[row]]
                    out[batch_idx, n_cells + cell] = (
                        smoothed + h_macro[batch_idx, cell] / h_capacity
                    )

            for row in range(r_lo, r_hi + 1):
                base = row * grid_col
                prefix_v[0] = 0.0
                for col in range(grid_col):
                    value = raw_v[batch_idx, base + col] / v_capacity
                    prefix_v[col + 1] = prefix_v[col] + value / v_window_count[col]
                for col in range(grid_col):
                    cell = base + col
                    smoothed = prefix_v[v_window_hi[col] + 1] - prefix_v[v_window_lo[col]]
                    out[batch_idx, cell] = smoothed + v_macro[batch_idx, cell] / v_capacity

    @_numba_njit(cache=True, fastmath=False)
    def _batch_soft_wirelength_jit(
        xy,
        module,
        touched,
        net_starts,
        net_lengths,
        ref_inv,
        unique_ref,
        x_off,
        y_off,
        pos_cache,
        per_net_hpwl,
        net_weights,
        total_wl_raw,
        normalizer,
        out,
    ):
        """Compute exact touched-net HPWL for every soft target."""
        for batch_idx in range(xy.shape[0]):
            delta = 0.0
            for touched_idx in range(touched.shape[0]):
                net = touched[touched_idx]
                start = net_starts[net]
                length = net_lengths[net]
                if length == 0:
                    continue
                pin = start
                pin_module_idx = unique_ref[ref_inv[pin]]
                pin_x = (
                    xy[batch_idx, 0] if pin_module_idx == module else pos_cache[pin_module_idx, 0]
                ) + x_off[pin]
                pin_y = (
                    xy[batch_idx, 1] if pin_module_idx == module else pos_cache[pin_module_idx, 1]
                ) + y_off[pin]
                min_x = pin_x
                max_x = pin_x
                min_y = pin_y
                max_y = pin_y
                for pin_offset in range(1, length):
                    pin = start + pin_offset
                    pin_module_idx = unique_ref[ref_inv[pin]]
                    pin_x = (
                        xy[batch_idx, 0]
                        if pin_module_idx == module
                        else pos_cache[pin_module_idx, 0]
                    ) + x_off[pin]
                    pin_y = (
                        xy[batch_idx, 1]
                        if pin_module_idx == module
                        else pos_cache[pin_module_idx, 1]
                    ) + y_off[pin]
                    if pin_x < min_x:
                        min_x = pin_x
                    if pin_x > max_x:
                        max_x = pin_x
                    if pin_y < min_y:
                        min_y = pin_y
                    if pin_y > max_y:
                        max_y = pin_y
                hpwl = (max_x - min_x) + (max_y - min_y)
                delta += (hpwl - per_net_hpwl[net]) * net_weights[net]
            out[batch_idx] = (total_wl_raw + delta) / normalizer

    @_numba_njit(cache=True, fastmath=False)
    def _batch_soft_swap_wirelength_jit(
        a_module,
        b_modules,
        a_xy,
        b_xy,
        touched_offsets,
        touched,
        net_starts,
        net_lengths,
        ref_inv,
        unique_ref,
        x_off,
        y_off,
        pos_cache,
        per_net_hpwl,
        net_weights,
        total_wl_raw,
        normalizer,
        out,
    ):
        """Compute touched-net HPWL for a batch of soft-soft swaps."""
        for batch_idx in range(b_modules.shape[0]):
            b_module = b_modules[batch_idx]
            delta = 0.0
            for touched_idx in range(touched_offsets[batch_idx], touched_offsets[batch_idx + 1]):
                net = touched[touched_idx]
                start = net_starts[net]
                length = net_lengths[net]
                if length == 0:
                    continue
                min_x = np.inf
                max_x = -np.inf
                min_y = np.inf
                max_y = -np.inf
                for pin_offset in range(length):
                    pin = start + pin_offset
                    pin_module_idx = unique_ref[ref_inv[pin]]
                    if pin_module_idx == a_module:
                        center_x = b_xy[batch_idx, 0]
                        center_y = b_xy[batch_idx, 1]
                    elif pin_module_idx == b_module:
                        center_x = a_xy[0]
                        center_y = a_xy[1]
                    else:
                        center_x = pos_cache[pin_module_idx, 0]
                        center_y = pos_cache[pin_module_idx, 1]
                    pin_x = center_x + x_off[pin]
                    pin_y = center_y + y_off[pin]
                    if pin_x < min_x:
                        min_x = pin_x
                    if pin_x > max_x:
                        max_x = pin_x
                    if pin_y < min_y:
                        min_y = pin_y
                    if pin_y > max_y:
                        max_y = pin_y
                hpwl = (max_x - min_x) + (max_y - min_y)
                delta += (hpwl - per_net_hpwl[net]) * net_weights[net]
            out[batch_idx] = (total_wl_raw + delta) / normalizer

    @_numba_njit(cache=True, fastmath=False)
    def _add_density_rect_jit(
        grid,
        cx,
        cy,
        half_w,
        half_h,
        weight,
        grid_w,
        grid_h,
        grid_row,
        grid_col,
    ):
        """Add one weighted rectangle's occupancy to a density grid."""
        x_min = cx - half_w
        x_max = cx + half_w
        y_min = cy - half_h
        y_max = cy + half_h
        bl_col = int(np.floor(x_min / grid_w))
        ur_col = int(np.floor(x_max / grid_w))
        bl_row = int(np.floor(y_min / grid_h))
        ur_row = int(np.floor(y_max / grid_h))
        if ur_row < 0 or ur_col < 0 or bl_row >= grid_row or bl_col >= grid_col:
            return
        bl_col = min(max(bl_col, 0), grid_col - 1)
        ur_col = min(max(ur_col, 0), grid_col - 1)
        bl_row = min(max(bl_row, 0), grid_row - 1)
        ur_row = min(max(ur_row, 0), grid_row - 1)
        for row in range(bl_row, ur_row + 1):
            overlap_y = min(grid_h * (row + 1), y_max) - max(grid_h * row, y_min)
            if overlap_y < 0.0:
                overlap_y = 0.0
            for col in range(bl_col, ur_col + 1):
                overlap_x = min(grid_w * (col + 1), x_max) - max(grid_w * col, x_min)
                if overlap_x < 0.0:
                    overlap_x = 0.0
                grid[row * grid_col + col] += weight * overlap_x * overlap_y

    @_numba_njit(cache=True, fastmath=False)
    def _batch_soft_swap_density_grids_jit(
        a_xy,
        b_xy,
        a_half_w,
        a_half_h,
        b_half_w,
        b_half_h,
        base_grid,
        grid_w,
        grid_h,
        grid_row,
        grid_col,
        out,
    ):
        """Build exact density grids for a batch of soft-soft swaps."""
        for batch_idx in range(b_xy.shape[0]):
            for cell in range(base_grid.shape[0]):
                out[batch_idx, cell] = base_grid[cell]
            grid = out[batch_idx]
            _add_density_rect_jit(
                grid,
                a_xy[0],
                a_xy[1],
                a_half_w,
                a_half_h,
                -1.0,
                grid_w,
                grid_h,
                grid_row,
                grid_col,
            )
            _add_density_rect_jit(
                grid,
                b_xy[batch_idx, 0],
                b_xy[batch_idx, 1],
                b_half_w[batch_idx],
                b_half_h[batch_idx],
                -1.0,
                grid_w,
                grid_h,
                grid_row,
                grid_col,
            )
            _add_density_rect_jit(
                grid,
                b_xy[batch_idx, 0],
                b_xy[batch_idx, 1],
                a_half_w,
                a_half_h,
                1.0,
                grid_w,
                grid_h,
                grid_row,
                grid_col,
            )
            _add_density_rect_jit(
                grid,
                a_xy[0],
                a_xy[1],
                b_half_w[batch_idx],
                b_half_h[batch_idx],
                1.0,
                grid_w,
                grid_h,
                grid_row,
                grid_col,
            )

    @_numba_njit(cache=True, fastmath=False)
    def _batch_soft_density_grids_jit(
        xy,
        half_w,
        half_h,
        base_grid,
        grid_w,
        grid_h,
        grid_row,
        grid_col,
        out,
    ):
        """Add one soft macro's occupancy to every target density grid."""
        for batch_idx in range(xy.shape[0]):
            for cell in range(base_grid.shape[0]):
                out[batch_idx, cell] = base_grid[cell]
            x_min = xy[batch_idx, 0] - half_w
            x_max = xy[batch_idx, 0] + half_w
            y_min = xy[batch_idx, 1] - half_h
            y_max = xy[batch_idx, 1] + half_h
            bl_col = int(np.floor(x_min / grid_w))
            ur_col = int(np.floor(x_max / grid_w))
            bl_row = int(np.floor(y_min / grid_h))
            ur_row = int(np.floor(y_max / grid_h))
            if ur_row < 0 or ur_col < 0 or bl_row >= grid_row or bl_col >= grid_col:
                continue
            bl_col = min(max(bl_col, 0), grid_col - 1)
            ur_col = min(max(ur_col, 0), grid_col - 1)
            bl_row = min(max(bl_row, 0), grid_row - 1)
            ur_row = min(max(ur_row, 0), grid_row - 1)
            for row in range(bl_row, ur_row + 1):
                overlap_y = min(grid_h * (row + 1), y_max) - max(grid_h * row, y_min)
                if overlap_y < 0.0:
                    overlap_y = 0.0
                for col in range(bl_col, ur_col + 1):
                    overlap_x = min(grid_w * (col + 1), x_max) - max(grid_w * col, x_min)
                    if overlap_x < 0.0:
                        overlap_x = 0.0
                    out[batch_idx, row * grid_col + col] += overlap_x * overlap_y


class IncrementalScorer:
    """Fast proxy scorer for small local-search moves."""

    def __init__(self, plc, benchmark: Benchmark, current_placement_np: np.ndarray):
        self.plc = plc
        self.benchmark = benchmark
        self.n_hard = benchmark.num_hard_macros
        self.hard_indices = list(benchmark.hard_macro_indices)

        # Force a full placement set before building baseline scores.
        plc._last_pos_cache = None
        _fast_set_placement(plc, current_placement_np, benchmark)

        wl_cache = _build_wl_cache(plc)
        self.wl_cache = wl_cache
        self.net_weights = wl_cache["net_weights"]
        self.net_starts = wl_cache["net_starts"]
        self.net_ends = wl_cache["net_ends"]
        self.net_lengths = wl_cache["net_lengths"]
        self.ref_inv = wl_cache["ref_inv"]
        self.x_off = wl_cache["x_off"]
        self.y_off = wl_cache["y_off"]
        self.unique_ref = wl_cache["unique_ref"]
        self.n_pins = wl_cache["n_pins"]
        self.n_nets = wl_cache["n_nets"]

        # Map each macro to the nets it touches.
        self._build_macro_to_nets()

        # Match the evaluator's wirelength scaling.
        cw_, ch_ = plc.get_canvas_width_height()
        self.wl_normalizer = float((cw_ + ch_) * max(plc.net_cnt, 1))

        # Baseline per-net wirelength.
        self.per_net_hpwl = self._compute_per_net_hpwl_full()
        self.total_wl_raw = float(np.sum(self.per_net_hpwl * self.net_weights))

        self.committed_hard_pos = current_placement_np[: self.n_hard].astype(np.float64).copy()

        # Soft macros affect WL, net routing, and density, but not blockage.
        self.soft_indices = list(benchmark.soft_macro_indices)
        self.num_soft = len(self.soft_indices)
        self.committed_soft_pos = (
            current_placement_np[self.n_hard : self.n_hard + self.num_soft]
            .astype(np.float64)
            .copy()
        )

        # Congestion state.
        cong_cache = plc._cong_cache
        self.grid_col = int(plc.grid_col)
        self.grid_row = int(plc.grid_row)
        self.grid_w = float(plc.width / self.grid_col)
        self.grid_h = float(plc.height / self.grid_row)
        self.grid_v_routes = self.grid_w * plc.vroutes_per_micron
        self.grid_h_routes = self.grid_h * plc.hroutes_per_micron
        self.smooth_range = int(plc.smooth_range)
        n_cells = self.grid_row * self.grid_col

        # Build raw routing grids for the current placement.
        plc.get_congestion_cost()  # ensure routing populated
        self.H_flat = np.zeros(n_cells, dtype=np.float64)
        self.V_flat = np.zeros(n_cells, dtype=np.float64)
        self.H_macro_flat = np.zeros(n_cells, dtype=np.float64)
        self.V_macro_flat = np.zeros(n_cells, dtype=np.float64)
        if self.n_nets > 0:
            _apply_net_routing_subset(
                plc,
                np.arange(self.n_nets, dtype=np.int64),
                +1.0,
                self.H_flat,
                self.V_flat,
            )
        n_hard_cache = cong_cache["n_hard"]
        if n_hard_cache > 0:
            _apply_macro_routing_subset(
                plc,
                np.arange(n_hard_cache, dtype=np.int64),
                +1.0,
                self.V_macro_flat,
                self.H_macro_flat,
            )

        # Cache smoothed routing so moves only re-smooth touched rows/columns.
        sr = self.smooth_range
        if sr > 0:
            _rows = np.arange(self.grid_row, dtype=np.int64)
            self._sm_row_lp = np.maximum(_rows - sr, 0)
            self._sm_row_up = np.minimum(_rows + sr, self.grid_row - 1)
            self._sm_row_cnt = (self._sm_row_up - self._sm_row_lp + 1).astype(np.float64)
            _cols = np.arange(self.grid_col, dtype=np.int64)
            self._sm_col_lp = np.maximum(_cols - sr, 0)
            self._sm_col_up = np.minimum(_cols + sr, self.grid_col - 1)
            self._sm_col_cnt = (self._sm_col_up - self._sm_col_lp + 1).astype(np.float64)
            self._resmooth_h_prefix = np.empty(self.grid_row + 1, dtype=np.float64)
            self._resmooth_v_prefix = np.empty(self.grid_col + 1, dtype=np.float64)
            self.H_smoothed = _smooth_routing_cong_vec(
                self.H_flat / self.grid_h_routes,
                self.grid_row,
                self.grid_col,
                sr,
                axis_h=True,
            ).reshape(self.grid_row, self.grid_col)
            self.V_smoothed = _smooth_routing_cong_vec(
                self.V_flat / self.grid_v_routes,
                self.grid_row,
                self.grid_col,
                sr,
                axis_h=False,
            ).reshape(self.grid_row, self.grid_col)
        else:
            # No smoothing: normalized flats are the cache.
            self.H_smoothed = (self.H_flat / self.grid_h_routes).reshape(
                self.grid_row, self.grid_col
            )
            self.V_smoothed = (self.V_flat / self.grid_v_routes).reshape(
                self.grid_row, self.grid_col
            )

        # Map module id to its hard-macro slot.
        self._module_to_hard_slot: "dict[int, int]" = {
            int(m): k for k, m in enumerate(cong_cache["hard_indices"])
        }

        # Cache per-macro net routing structures.
        self._route_struct_cache: "dict[int, object]" = {}
        # Cache placement-independent routing structures for repeated swap module sets.
        self._route_struct_many_cache: "dict[tuple[int, ...], object]" = {}
        # Cache touched nets for frequently repeated module sets.
        self._touched_cache2: "dict[tuple[int, int], np.ndarray]" = {}
        self._touched_cache_many: "dict[tuple[int, ...], np.ndarray]" = {}

        # Density state.
        dens_cache = _build_density_cache(plc, benchmark)
        self.dens_grid_col = int(plc.grid_col)
        self.dens_grid_row = int(plc.grid_row)
        self.dens_grid_w = float(plc.width / self.dens_grid_col)
        self.dens_grid_h = float(plc.height / self.dens_grid_row)
        self.dens_grid_area = self.dens_grid_w * self.dens_grid_h
        self.dens_n_cells = self.dens_grid_col * self.dens_grid_row
        self.dens_density_cnt = int(np.floor(self.dens_n_cells * 0.1))
        # Half sizes by module id.
        self._dens_half: "dict[int, tuple[float, float]]" = {
            int(m): (float(dens_cache["half_w"][k]), float(dens_cache["half_h"][k]))
            for k, m in enumerate(dens_cache["macro_indices"])
        }
        # Baseline occupancy grid.
        _vectorized_get_grid_cells_density(plc)
        self.grid_occupied = np.asarray(plc.grid_occupied, dtype=np.float64)
        self._dens_empty_idx = np.empty(0, dtype=np.int64)
        self._dens_empty_area = np.empty(0, dtype=np.float64)

    def _macro_occ(self, module_idx: int, cx: float, cy: float):
        """Return cells and occupied area for one macro."""
        hw_, hh_ = self._dens_half[int(module_idx)]
        gw, gh = self.dens_grid_w, self.dens_grid_h
        gcol, grow = self.dens_grid_col, self.dens_grid_row
        x_min = cx - hw_
        x_max = cx + hw_
        y_min = cy - hh_
        y_max = cy + hh_
        bl_col = int(np.floor(x_min / gw))
        bl_row = int(np.floor(y_min / gh))
        ur_col = int(np.floor(x_max / gw))
        ur_row = int(np.floor(y_max / gh))
        # Skip macros fully outside the grid.
        if not (ur_row >= 0 and ur_col >= 0 and bl_row <= grow - 1 and bl_col <= gcol - 1):
            return self._dens_empty_idx, self._dens_empty_area
        bl_col = min(max(bl_col, 0), gcol - 1)
        ur_col = min(max(ur_col, 0), gcol - 1)
        bl_row = min(max(bl_row, 0), grow - 1)
        ur_row = min(max(ur_row, 0), grow - 1)
        if HAS_NUMBA:
            return _macro_occ_jit(
                bl_row, bl_col, ur_row, ur_col, x_min, x_max, y_min, y_max, gw, gh, gcol
            )
        cols = np.arange(bl_col, ur_col + 1)
        rows = np.arange(bl_row, ur_row + 1)
        ox = np.minimum(gw * (cols + 1), x_max) - np.maximum(gw * cols, x_min)
        oy = np.minimum(gh * (rows + 1), y_max) - np.maximum(gh * rows, y_min)
        np.maximum(ox, 0.0, out=ox)
        np.maximum(oy, 0.0, out=oy)
        area = np.outer(oy, ox).ravel()
        flat = (rows[:, None] * gcol + cols[None, :]).ravel()
        return flat, area

    def _compute_density_cost(self) -> float:
        """Compute density cost from the maintained occupancy grid."""
        cnt = self.dens_density_cnt
        go = self.grid_occupied
        nz = go[go != 0.0]
        if nz.size == 0:
            return 0.0
        if self.dens_n_cells < 10:
            return 0.5 * float(nz.mean() / self.dens_grid_area)
        k = min(cnt, nz.size)
        # CPU partition is faster than GPU topk for these small grids.
        top = np.partition(nz, nz.size - k)[nz.size - k :]
        return 0.5 * float(top.sum()) / self.dens_grid_area / cnt

    def _compute_cong_cost(self) -> float:
        """Compute congestion cost from cached routing grids."""
        Hm = self.H_macro_flat / self.grid_h_routes
        Vm = self.V_macro_flat / self.grid_v_routes
        xx = np.concatenate([self.V_smoothed.ravel() + Vm, self.H_smoothed.ravel() + Hm])
        n = xx.size
        cnt = int(n * 0.05)
        if cnt == 0:
            return float(xx.max())
        top = np.partition(xx, n - cnt)[n - cnt :]
        return float(top.sum() / cnt)

    def congestion_field(self) -> np.ndarray:
        """Return the current max(H, V) routing congestion grid."""
        Hm = (self.H_macro_flat / self.grid_h_routes).reshape(self.grid_row, self.grid_col)
        Vm = (self.V_macro_flat / self.grid_v_routes).reshape(self.grid_row, self.grid_col)
        return np.maximum(self.H_smoothed + Hm, self.V_smoothed + Vm)

    @staticmethod
    def _union_bbox(*bbs):
        """Union non-empty row/column boxes."""
        bbs = [b for b in bbs if b is not None]
        if not bbs:
            return None, None, None, None
        return (
            min(b[0] for b in bbs),
            max(b[1] for b in bbs),
            min(b[2] for b in bbs),
            max(b[3] for b in bbs),
        )

    def _resmooth_h_cols(self, c_lo: int, c_hi: int) -> None:
        """Re-smooth affected H columns from raw routing."""
        H2d = self.H_flat.reshape(self.grid_row, self.grid_col)
        if self.smooth_range <= 0:
            self.H_smoothed[:, c_lo : c_hi + 1] = H2d[:, c_lo : c_hi + 1] / self.grid_h_routes
            return
        if HAS_NUMBA:
            _resmooth_h_cols_jit(
                self.H_flat,
                self.H_smoothed,
                int(c_lo),
                int(c_hi),
                int(self.grid_row),
                int(self.grid_col),
                float(self.grid_h_routes),
                self._sm_row_lp,
                self._sm_row_up,
                self._sm_row_cnt,
                self._resmooth_h_prefix,
            )
            return
        sub = H2d[:, c_lo : c_hi + 1] / self.grid_h_routes
        weighted = sub / self._sm_row_cnt[:, None]
        ncols = sub.shape[1]
        cs = np.empty((self.grid_row + 1, ncols), dtype=np.float64)
        cs[0, :] = 0.0
        np.cumsum(weighted, axis=0, out=cs[1:, :])
        self.H_smoothed[:, c_lo : c_hi + 1] = cs[self._sm_row_up + 1] - cs[self._sm_row_lp]

    def _resmooth_v_rows(self, r_lo: int, r_hi: int) -> None:
        """Re-smooth affected V rows from raw routing."""
        V2d = self.V_flat.reshape(self.grid_row, self.grid_col)
        if self.smooth_range <= 0:
            self.V_smoothed[r_lo : r_hi + 1, :] = V2d[r_lo : r_hi + 1, :] / self.grid_v_routes
            return
        if HAS_NUMBA:
            _resmooth_v_rows_jit(
                self.V_flat,
                self.V_smoothed,
                int(r_lo),
                int(r_hi),
                int(self.grid_col),
                float(self.grid_v_routes),
                self._sm_col_lp,
                self._sm_col_up,
                self._sm_col_cnt,
                self._resmooth_v_prefix,
            )
            return
        sub = V2d[r_lo : r_hi + 1, :] / self.grid_v_routes
        nrows = sub.shape[0]
        weighted = sub / self._sm_col_cnt[None, :]
        cs = np.empty((nrows, self.grid_col + 1), dtype=np.float64)
        cs[:, 0] = 0.0
        np.cumsum(weighted, axis=1, out=cs[:, 1:])
        self.V_smoothed[r_lo : r_hi + 1, :] = cs[:, self._sm_col_up + 1] - cs[:, self._sm_col_lp]

    def _resmooth_bbox(self, r_lo, r_hi, c_lo, c_hi) -> None:
        """Re-smooth rows and columns touched by a move."""
        if c_lo is None:
            return
        self._resmooth_h_cols(c_lo, c_hi)
        self._resmooth_v_rows(r_lo, r_hi)

    def _build_macro_to_nets(self):
        """Build macro-to-net lookup tables."""
        ref_idx = self.wl_cache["ref_idx"]
        pin_to_net = self.wl_cache["pin_to_net"]
        # Sort pins by macro, then split into runs.
        order = np.argsort(ref_idx, kind="stable")
        sorted_macros = ref_idx[order]
        sorted_nets = pin_to_net[order]
        boundaries = np.flatnonzero(np.diff(sorted_macros) != 0) + 1
        macro_segments = np.split(sorted_nets, boundaries)
        macro_keys = (
            sorted_macros[np.concatenate([[0], boundaries])]
            if len(sorted_macros)
            else np.array([], dtype=ref_idx.dtype)
        )
        self.macro_to_nets: "dict[int, np.ndarray]" = {}
        for k, nets_for_macro in zip(macro_keys, macro_segments):
            uniq = np.unique(nets_for_macro)
            self.macro_to_nets[int(k)] = uniq

    def _compute_per_net_hpwl_full(self) -> np.ndarray:
        """Compute HPWL for every net."""
        if self.n_nets == 0:
            return np.empty(0, dtype=np.float64)
        pos_cache = _ensure_pos_cache(self.plc)
        node_x = pos_cache[self.unique_ref, 0]
        node_y = pos_cache[self.unique_ref, 1]
        pin_x = node_x[self.ref_inv] + self.x_off
        pin_y = node_y[self.ref_inv] + self.y_off
        starts = self.net_starts
        max_x = np.maximum.reduceat(pin_x, starts)
        min_x = np.minimum.reduceat(pin_x, starts)
        max_y = np.maximum.reduceat(pin_y, starts)
        min_y = np.minimum.reduceat(pin_y, starts)
        return (max_x - min_x) + (max_y - min_y)

    def _compute_per_net_hpwl_subset(self, net_indices: np.ndarray) -> np.ndarray:
        """Compute HPWL for selected nets."""
        if len(net_indices) == 0:
            return np.empty(0, dtype=np.float64)

        if HAS_NUMBA:
            pos_cache = _ensure_pos_cache(self.plc)
            return _hpwl_subset_jit(
                np.ascontiguousarray(net_indices),
                self.net_starts,
                self.net_lengths,
                self.ref_inv,
                self.x_off,
                self.y_off,
                np.ascontiguousarray(pos_cache[self.unique_ref, 0]),
                np.ascontiguousarray(pos_cache[self.unique_ref, 1]),
            )

        starts_t = self.net_starts[net_indices]
        lengths_t = self.net_lengths[net_indices]
        total = int(lengths_t.sum())
        if total == 0:
            return np.zeros(len(net_indices), dtype=np.float64)

        # Gather pins from the selected nets only.
        pin_indices = np.repeat(starts_t, lengths_t) + (
            np.arange(total)
            - np.repeat(np.concatenate([[0], np.cumsum(lengths_t)[:-1]]), lengths_t)
        )

        pos_cache = _ensure_pos_cache(self.plc)
        node_x = pos_cache[self.unique_ref, 0]
        node_y = pos_cache[self.unique_ref, 1]
        pin_x = node_x[self.ref_inv[pin_indices]] + self.x_off[pin_indices]
        pin_y = node_y[self.ref_inv[pin_indices]] + self.y_off[pin_indices]

        sub_starts = np.concatenate([[0], np.cumsum(lengths_t)[:-1]])
        max_x = np.maximum.reduceat(pin_x, sub_starts)
        min_x = np.minimum.reduceat(pin_x, sub_starts)
        max_y = np.maximum.reduceat(pin_y, sub_starts)
        min_y = np.minimum.reduceat(pin_y, sub_starts)
        return (max_x - min_x) + (max_y - min_y)

    def _touched_nets(self, i_module: int, j_module: int) -> np.ndarray:
        i0 = int(i_module)
        j0 = int(j_module)
        key = (i0, j0) if i0 <= j0 else (j0, i0)
        cached = self._touched_cache2.get(key)
        if cached is not None:
            return cached
        a = self.macro_to_nets.get(i0)
        b = self.macro_to_nets.get(j0)
        if a is None and b is None:
            return np.empty(0, dtype=np.int64)
        if a is None:
            self._touched_cache2[key] = b
            return b
        if b is None:
            self._touched_cache2[key] = a
            return a
        out = np.union1d(a, b)
        self._touched_cache2[key] = out
        return out

    def _touched_nets3(self, m1: int, m2: int, m3: int) -> np.ndarray:
        """Return nets touched by three modules."""
        key = tuple(sorted({int(m1), int(m2), int(m3)}))
        if len(key) == 1:
            return self._touched_nets_only(key[0])
        if len(key) == 2:
            return self._touched_nets(key[0], key[1])
        cached = self._touched_cache_many.get(key)
        if cached is not None:
            return cached
        parts = [self._macro_nets(int(m)) for m in key]
        if not parts:
            return np.empty(0, dtype=np.int64)
        if len(parts) == 1:
            return parts[0]
        out = np.unique(np.concatenate(parts))
        self._touched_cache_many[key] = out
        return out

    def _touched_nets_only(self, module: int) -> np.ndarray:
        """Return nets touched by exactly one module."""
        module = int(module)
        cached = self._touched_cache2.get((module, module))
        if cached is not None:
            return cached
        nets = self._macro_nets(module)
        self._touched_cache2[(module, module)] = nets
        return nets

    def _touched_nets_many(self, modules) -> np.ndarray:
        """Return the union of nets touched by the given modules."""
        key = tuple(sorted({int(m) for m in modules}))
        if not key:
            return np.empty(0, dtype=np.int64)
        cached = self._touched_cache_many.get(key)
        if cached is not None:
            return cached
        parts = [self._macro_nets(int(m)) for m in key]
        parts = [p for p in parts if p.size]
        if not parts:
            return np.empty(0, dtype=np.int64)
        if len(parts) == 1:
            self._touched_cache_many[key] = parts[0]
            return parts[0]
        out = np.unique(np.concatenate(parts))
        self._touched_cache_many[key] = out
        return out

    def _apply_pos(self, module_idx: int, x: float, y: float) -> None:
        """Set a module position and mark cached costs dirty."""
        self.plc.modules_w_pins[module_idx].set_pos(float(x), float(y))
        pos_cache = _ensure_pos_cache(self.plc)
        pos_cache[module_idx, 0] = float(x)
        pos_cache[module_idx, 1] = float(y)
        # We compute WL here; plc must recompute density/congestion if asked.
        self.plc.FLAG_UPDATE_DENSITY = True
        self.plc.FLAG_UPDATE_CONGESTION = True

    def _macro_nets(self, i_module: int) -> np.ndarray:
        a = self.macro_to_nets.get(i_module)
        return a if a is not None else np.empty(0, dtype=np.int64)

    def _route_struct(self, module_idx: int):
        """Cached topology routing-struct for one macro's touched nets. Built once
        per module (placement-independent), reused across calls."""
        cache = self._route_struct_cache
        m = int(module_idx)
        if m not in cache:
            cache[m] = _build_net_routing_struct(self.plc, self._macro_nets(m))
        return cache[m]

    def _route_struct_many(self, modules):
        """Cached topology routing-struct for a repeated multi-macro move."""
        key = tuple(sorted({int(m) for m in modules}))
        if not key:
            return None
        cache = self._route_struct_many_cache
        if key not in cache:
            cache[key] = _build_net_routing_struct(self.plc, self._touched_nets_many(key))
        return cache[key]

    def soft_net_centroids(self) -> np.ndarray:
        """Return a connection-centered target point for each soft macro."""
        pos = _ensure_pos_cache(self.plc)
        ref_idx = self.wl_cache["ref_idx"]
        pin_to_net = self.wl_cache["pin_to_net"]
        starts = self.net_starts
        out = np.empty((self.num_soft, 2), dtype=np.float64)
        for k in range(self.num_soft):
            out[k, 0] = self.committed_soft_pos[k, 0]
            out[k, 1] = self.committed_soft_pos[k, 1]
        if ref_idx.size == 0 or starts.size == 0:
            return out
        pin_x = pos[ref_idx, 0] + self.x_off
        pin_y = pos[ref_idx, 1] + self.y_off
        counts = (self.net_ends - self.net_starts).astype(np.float64)
        counts[counts == 0] = 1.0
        net_cx = np.add.reduceat(pin_x, starts) / counts
        net_cy = np.add.reduceat(pin_y, starts) / counts
        n_mod = int(ref_idx.max()) + 1
        msx = np.zeros(n_mod, dtype=np.float64)
        msy = np.zeros(n_mod, dtype=np.float64)
        mc = np.zeros(n_mod, dtype=np.float64)
        np.add.at(msx, ref_idx, net_cx[pin_to_net])
        np.add.at(msy, ref_idx, net_cy[pin_to_net])
        np.add.at(mc, ref_idx, 1.0)
        for k, m in enumerate(self.soft_indices):
            if m < n_mod and mc[m] > 0:
                out[k, 0] = msx[m] / mc[m]
                out[k, 1] = msy[m] / mc[m]
        return out

    def score_move(self, i_hard: int, new_xy) -> float:
        """Score a hard relocation, then restore the old state."""
        i_module = self.hard_indices[i_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])

        touched = self._touched_nets_only(i_module)
        restore_nets = touched.size > 0
        restore_macros = False
        struct = self._route_struct(i_module)
        macro_subset = (
            np.array([i_slot], dtype=np.int64)
            if i_slot is not None
            else np.empty(0, dtype=np.int64)
        )
        if macro_subset.size > 0:
            restore_macros = True

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0, self.V_macro_flat, self.H_macro_flat
            )
        self._apply_pos(i_module, new_ix, new_iy)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0, self.V_macro_flat, self.H_macro_flat
            )

        # Re-smooth touched rows/columns.
        r_lo, r_hi, c_lo, c_hi = self._union_bbox(bb_old, bb_new)
        if c_lo is not None:
            Hs_snap = self.H_smoothed[:, c_lo : c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo : r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer
        cong = self._compute_cong_cost()

        o_idx, o_area = self._macro_occ(i_module, old_ix, old_iy)
        n_idx, n_area = self._macro_occ(i_module, new_ix, new_iy)
        go = self.grid_occupied
        if o_idx.size:
            np.subtract.at(go, o_idx, o_area)
        if n_idx.size:
            np.add.at(go, n_idx, n_area)
        dens = self._compute_density_cost()

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        # Revert trial state.
        if n_idx.size:
            np.subtract.at(go, n_idx, n_area)
        if o_idx.size:
            np.add.at(go, o_idx, o_area)
        if restore_nets:
            self._apply_pos(i_module, new_ix, new_iy)
            _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
            self._apply_pos(i_module, old_ix, old_iy)
            _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        else:
            self._apply_pos(i_module, old_ix, old_iy)
        if restore_macros:
            self._apply_pos(i_module, new_ix, new_iy)
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0, self.V_macro_flat, self.H_macro_flat
            )
            self._apply_pos(i_module, old_ix, old_iy)
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0, self.V_macro_flat, self.H_macro_flat
            )
        if c_lo is not None:
            self.H_smoothed[:, c_lo : c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo : r_hi + 1, :] = Vs_snap
        return score

    def commit_move(self, i_hard: int, new_xy) -> None:
        """Commit a hard relocation."""
        i_module = self.hard_indices[i_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])

        touched = self._macro_nets(i_module)
        struct = self._route_struct(i_module)
        macro_subset = (
            np.array([i_slot], dtype=np.int64)
            if i_slot is not None
            else np.empty(0, dtype=np.int64)
        )

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0, self.V_macro_flat, self.H_macro_flat
            )
        self._apply_pos(i_module, new_ix, new_iy)

        go = self.grid_occupied
        o_idx, o_area = self._macro_occ(i_module, old_ix, old_iy)
        n_idx, n_area = self._macro_occ(i_module, new_ix, new_iy)
        if o_idx.size:
            np.subtract.at(go, o_idx, o_area)
        if n_idx.size:
            np.add.at(go, n_idx, n_area)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0, self.V_macro_flat, self.H_macro_flat
            )

        self._resmooth_bbox(*self._union_bbox(bb_old, bb_new))

        self.committed_hard_pos[i_hard, 0] = new_ix
        self.committed_hard_pos[i_hard, 1] = new_iy
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def score_move_soft(self, soft_k: int, new_xy) -> float:
        """Score a soft relocation, then restore the old state."""
        s_module = self.soft_indices[soft_k]
        old_x = float(self.committed_soft_pos[soft_k, 0])
        old_y = float(self.committed_soft_pos[soft_k, 1])
        new_x, new_y = float(new_xy[0]), float(new_xy[1])

        touched = self._touched_nets_only(s_module)
        restore_nets = touched.size > 0
        struct = self._route_struct(s_module)

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        self._apply_pos(s_module, new_x, new_y)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)

        # Re-smooth touched rows/columns.
        r_lo, r_hi, c_lo, c_hi = self._union_bbox(bb_old, bb_new)
        if c_lo is not None:
            Hs_snap = self.H_smoothed[:, c_lo : c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo : r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer
        cong = self._compute_cong_cost()

        o_idx, o_area = self._macro_occ(s_module, old_x, old_y)
        n_idx, n_area = self._macro_occ(s_module, new_x, new_y)
        go = self.grid_occupied
        if o_idx.size:
            np.subtract.at(go, o_idx, o_area)
        if n_idx.size:
            np.add.at(go, n_idx, n_area)
        dens = self._compute_density_cost()

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        # Revert trial state.
        if n_idx.size:
            np.subtract.at(go, n_idx, n_area)
        if o_idx.size:
            np.add.at(go, o_idx, o_area)
        if restore_nets:
            self._apply_pos(s_module, new_x, new_y)
            _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
            self._apply_pos(s_module, old_x, old_y)
            _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        else:
            self._apply_pos(s_module, old_x, old_y)
        if c_lo is not None:
            self.H_smoothed[:, c_lo : c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo : r_hi + 1, :] = Vs_snap
        return score

    def commit_move_soft(self, soft_k: int, new_xy) -> None:
        """Commit a soft relocation."""
        s_module = self.soft_indices[soft_k]
        old_x = float(self.committed_soft_pos[soft_k, 0])
        old_y = float(self.committed_soft_pos[soft_k, 1])
        new_x, new_y = float(new_xy[0]), float(new_xy[1])
        touched = self._macro_nets(s_module)
        struct = self._route_struct(s_module)

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        self._apply_pos(s_module, new_x, new_y)

        go = self.grid_occupied
        o_idx, o_area = self._macro_occ(s_module, old_x, old_y)
        n_idx, n_area = self._macro_occ(s_module, new_x, new_y)
        if o_idx.size:
            np.subtract.at(go, o_idx, o_area)
        if n_idx.size:
            np.add.at(go, n_idx, n_area)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)

        self._resmooth_bbox(*self._union_bbox(bb_old, bb_new))

        self.committed_soft_pos[soft_k, 0] = new_x
        self.committed_soft_pos[soft_k, 1] = new_y
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def _score_multi_move(self, modules, old_xy, new_xy, hard_slots) -> float:
        """Score a small multi-macro move, then restore committed state."""
        modules = [int(m) for m in modules]
        old_xy = [(float(x), float(y)) for x, y in old_xy]
        new_xy = [(float(x), float(y)) for x, y in new_xy]
        hard_slots = np.asarray(hard_slots, dtype=np.int64)

        touched = self._touched_nets_many(modules)
        struct = self._route_struct_many(modules) if touched.size else None
        H_snap = self.H_flat.copy() if touched.size else None
        V_snap = self.V_flat.copy() if touched.size else None
        Hm_snap = self.H_macro_flat.copy() if hard_slots.size else None
        Vm_snap = self.V_macro_flat.copy() if hard_slots.size else None
        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if hard_slots.size:
            _apply_macro_routing_subset(
                self.plc, hard_slots, -1.0, self.V_macro_flat, self.H_macro_flat
            )

        old_occ = [self._macro_occ(m, x, y) for m, (x, y) in zip(modules, old_xy)]
        go = self.grid_occupied
        for idx, area in old_occ:
            if idx.size:
                np.subtract.at(go, idx, area)

        for m, (x, y) in zip(modules, new_xy):
            self._apply_pos(m, x, y)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if hard_slots.size:
            _apply_macro_routing_subset(
                self.plc, hard_slots, +1.0, self.V_macro_flat, self.H_macro_flat
            )

        r_lo, r_hi, c_lo, c_hi = self._union_bbox(bb_old, bb_new)
        Hs_snap = Vs_snap = None
        if c_lo is not None:
            Hs_snap = self.H_smoothed[:, c_lo : c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo : r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        if touched.size:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer
        cong = self._compute_cong_cost()

        new_occ = [self._macro_occ(m, x, y) for m, (x, y) in zip(modules, new_xy)]
        for idx, area in new_occ:
            if idx.size:
                np.add.at(go, idx, area)
        dens = self._compute_density_cost()

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        for idx, area in new_occ:
            if idx.size:
                np.subtract.at(go, idx, area)
        for idx, area in old_occ:
            if idx.size:
                np.add.at(go, idx, area)
        for m, (x, y) in zip(modules, old_xy):
            self._apply_pos(m, x, y)
        if touched.size:
            self.H_flat[:] = H_snap
            self.V_flat[:] = V_snap
        if Hm_snap is not None:
            self.H_macro_flat[:] = Hm_snap
            self.V_macro_flat[:] = Vm_snap
        if c_lo is not None:
            self.H_smoothed[:, c_lo : c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo : r_hi + 1, :] = Vs_snap
        return score

    def _commit_multi_move(self, modules, old_xy, new_xy, hard_slots, hard_updates, soft_updates):
        """Commit a small multi-macro move."""
        modules = [int(m) for m in modules]
        old_xy = [(float(x), float(y)) for x, y in old_xy]
        new_xy = [(float(x), float(y)) for x, y in new_xy]
        hard_slots = np.asarray(hard_slots, dtype=np.int64)
        touched = self._touched_nets_many(modules)
        struct = self._route_struct_many(modules) if touched.size else None

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if hard_slots.size:
            _apply_macro_routing_subset(
                self.plc, hard_slots, -1.0, self.V_macro_flat, self.H_macro_flat
            )

        go = self.grid_occupied
        for m, (x, y) in zip(modules, old_xy):
            idx, area = self._macro_occ(m, x, y)
            if idx.size:
                np.subtract.at(go, idx, area)

        for m, (x, y) in zip(modules, new_xy):
            self._apply_pos(m, x, y)
            idx, area = self._macro_occ(m, x, y)
            if idx.size:
                np.add.at(go, idx, area)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if hard_slots.size:
            _apply_macro_routing_subset(
                self.plc, hard_slots, +1.0, self.V_macro_flat, self.H_macro_flat
            )
        self._resmooth_bbox(*self._union_bbox(bb_old, bb_new))

        for i, xy in hard_updates.items():
            self.committed_hard_pos[int(i), 0] = float(xy[0])
            self.committed_hard_pos[int(i), 1] = float(xy[1])
        for k, xy in soft_updates.items():
            self.committed_soft_pos[int(k), 0] = float(xy[0])
            self.committed_soft_pos[int(k), 1] = float(xy[1])

        if touched.size:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def _hard_slot_array(self, *i_hard) -> np.ndarray:
        slots = []
        for i in i_hard:
            slot = self._module_to_hard_slot.get(int(self.hard_indices[int(i)]))
            if slot is not None:
                slots.append(int(slot))
        return np.asarray(slots, dtype=np.int64)

    def score_swap_hard_hard(self, i_hard: int, j_hard: int) -> float:
        """Score exchanging two hard macro centers."""
        i_hard, j_hard = int(i_hard), int(j_hard)
        im, jm = int(self.hard_indices[i_hard]), int(self.hard_indices[j_hard])
        i_xy = tuple(self.committed_hard_pos[i_hard])
        j_xy = tuple(self.committed_hard_pos[j_hard])
        return self._score_multi_move(
            [im, jm],
            [i_xy, j_xy],
            [j_xy, i_xy],
            self._hard_slot_array(i_hard, j_hard),
        )

    def score_swap_hard_hard_many(self, i_hard: int, candidates: np.ndarray) -> np.ndarray:
        """Score hard-hard swaps for one hard macro, preserving candidate order."""
        cand = np.asarray(candidates, dtype=np.int64).reshape(-1)
        b_modules = np.asarray([self.hard_indices[int(j_hard)] for j_hard in cand], dtype=np.int64)
        b_xy = self.committed_hard_pos[cand]
        batch = self._score_hard_endpoint_swaps_many(int(i_hard), b_modules, b_xy, b_is_hard=True)
        if batch is not None:
            return batch
        out = np.empty(cand.size, dtype=np.float64)
        for k, j_hard in enumerate(cand):
            out[k] = self.score_swap_hard_hard(i_hard, int(j_hard))
        return out

    def commit_swap_hard_hard(self, i_hard: int, j_hard: int) -> None:
        """Commit exchanging two hard macro centers."""
        i_hard, j_hard = int(i_hard), int(j_hard)
        im, jm = int(self.hard_indices[i_hard]), int(self.hard_indices[j_hard])
        i_xy = tuple(self.committed_hard_pos[i_hard])
        j_xy = tuple(self.committed_hard_pos[j_hard])
        self._commit_multi_move(
            [im, jm],
            [i_xy, j_xy],
            [j_xy, i_xy],
            self._hard_slot_array(i_hard, j_hard),
            {i_hard: j_xy, j_hard: i_xy},
            {},
        )

    def score_swap_soft_soft(self, soft_a: int, soft_b: int) -> float:
        """Score exchanging two soft macro centers."""
        soft_a, soft_b = int(soft_a), int(soft_b)
        am, bm = int(self.soft_indices[soft_a]), int(self.soft_indices[soft_b])
        a_xy = tuple(self.committed_soft_pos[soft_a])
        b_xy = tuple(self.committed_soft_pos[soft_b])
        return self._score_multi_move([am, bm], [a_xy, b_xy], [b_xy, a_xy], [])

    def _prepare_pair_swap_batch(self, a_module: int, b_modules: np.ndarray):
        """Flatten cached pair topology for compiled swap batch scoring."""
        a_module = int(a_module)
        b_modules = np.ascontiguousarray(b_modules, dtype=np.int64)
        structs = [self._route_struct_many([a_module, int(b_module)]) for b_module in b_modules]
        if any(struct is None for struct in structs):
            return None
        jit_structs = [struct["jit"] for struct in structs]

        def _offsets(key):
            out = np.zeros(len(jit_structs) + 1, dtype=np.int64)
            for idx, jit in enumerate(jit_structs):
                out[idx + 1] = out[idx] + int(jit[key].size)
            return out

        def _concat(key, dtype):
            arrays = [np.asarray(jit[key], dtype=dtype) for jit in jit_structs if jit[key].size]
            if not arrays:
                return np.empty(0, dtype=dtype)
            return np.ascontiguousarray(np.concatenate(arrays), dtype=dtype)

        pin_offsets = _offsets("pin_module")
        starts2_offsets = _offsets("starts2")
        starts3_offsets = _offsets("starts3")
        starts4_offsets = _offsets("starts4")
        touched_parts = [self._touched_nets(a_module, int(b_module)) for b_module in b_modules]
        touched_offsets = np.zeros(len(touched_parts) + 1, dtype=np.int64)
        for idx, touched in enumerate(touched_parts):
            touched_offsets[idx + 1] = touched_offsets[idx] + int(touched.size)
        touched_arrays = [part for part in touched_parts if part.size]
        touched = (
            np.ascontiguousarray(np.concatenate(touched_arrays), dtype=np.int64)
            if touched_arrays
            else np.empty(0, dtype=np.int64)
        )
        max_unique = max(int(jit["unique_sinks"].size) for jit in jit_structs)
        max_three = max(int(jit["three_g0"].size) for jit in jit_structs)
        return {
            "a_module": a_module,
            "b_modules": b_modules,
            "pin_offsets": pin_offsets,
            "pin_gcell": np.empty(int(pin_offsets[-1]), dtype=np.int64),
            "pin_module": _concat("pin_module", np.int64),
            "pin_x_off": _concat("pin_x_off", np.float64),
            "pin_y_off": _concat("pin_y_off", np.float64),
            "starts2_offsets": starts2_offsets,
            "starts2": _concat("starts2", np.int64),
            "weights2": _concat("weights2", np.float64),
            "starts3_offsets": starts3_offsets,
            "starts3": _concat("starts3", np.int64),
            "weights3": _concat("weights3", np.float64),
            "starts4_offsets": starts4_offsets,
            "starts4": _concat("starts4", np.int64),
            "lengths4": _concat("lengths4", np.int64),
            "weights4": _concat("weights4", np.float64),
            "unique_sinks": np.empty(max_unique, dtype=np.int64),
            "three_g0": np.empty(max_three, dtype=np.int64),
            "three_g1": np.empty(max_three, dtype=np.int64),
            "three_g2": np.empty(max_three, dtype=np.int64),
            "three_weights": np.empty(max_three, dtype=np.float64),
            "touched_offsets": touched_offsets,
            "touched": touched,
        }

    def _score_hard_endpoint_swaps_many(
        self,
        i_hard: int,
        b_modules: np.ndarray,
        b_xy: np.ndarray,
        *,
        b_is_hard: bool,
    ) -> "np.ndarray | None":
        """Batch exact HH or HS swaps sharing one hard endpoint."""
        b_modules = np.ascontiguousarray(b_modules, dtype=np.int64)
        b_xy = np.ascontiguousarray(b_xy, dtype=np.float64)
        if not HAS_NUMBA or b_modules.size < 2:
            return None

        a_module = int(self.hard_indices[int(i_hard)])
        batch = self._prepare_pair_swap_batch(a_module, b_modules)
        if batch is None:
            return None

        n_batch = int(b_modules.size)
        n_cells = self.grid_row * self.grid_col
        raw_h = np.empty((n_batch, n_cells), dtype=np.float64)
        raw_v = np.empty((n_batch, n_cells), dtype=np.float64)
        bboxes = np.empty((n_batch, 4), dtype=np.int64)
        a_xy = np.ascontiguousarray(self.committed_hard_pos[int(i_hard)], dtype=np.float64)
        pos_cache = np.ascontiguousarray(_ensure_pos_cache(self.plc), dtype=np.float64)
        _batch_soft_swap_route_grids_jit(
            batch["a_module"],
            batch["b_modules"],
            a_xy,
            b_xy,
            pos_cache,
            self.H_flat,
            self.V_flat,
            raw_h,
            raw_v,
            bboxes,
            batch["pin_offsets"],
            batch["pin_gcell"],
            batch["pin_module"],
            batch["pin_x_off"],
            batch["pin_y_off"],
            batch["starts2_offsets"],
            batch["starts2"],
            batch["weights2"],
            batch["starts3_offsets"],
            batch["starts3"],
            batch["weights3"],
            batch["starts4_offsets"],
            batch["starts4"],
            batch["lengths4"],
            batch["weights4"],
            batch["unique_sinks"],
            batch["three_g0"],
            batch["three_g1"],
            batch["three_g2"],
            batch["three_weights"],
            self.grid_w,
            self.grid_h,
            self.grid_row,
            self.grid_col,
        )

        cong_cache = self.plc._cong_cache
        a_slot = self._module_to_hard_slot.get(a_module)
        if a_slot is None:
            return None
        if b_is_hard:
            b_slots = np.asarray(
                [self._module_to_hard_slot.get(int(module), -1) for module in b_modules],
                dtype=np.int64,
            )
            if np.any(b_slots < 0):
                return None
            b_half_w = np.ascontiguousarray(cong_cache["hard_half_w"][b_slots], dtype=np.float64)
            b_half_h = np.ascontiguousarray(cong_cache["hard_half_h"][b_slots], dtype=np.float64)
        else:
            b_half_w = np.zeros(n_batch, dtype=np.float64)
            b_half_h = np.zeros(n_batch, dtype=np.float64)
        h_macro = np.empty((n_batch, n_cells), dtype=np.float64)
        v_macro = np.empty((n_batch, n_cells), dtype=np.float64)
        _batch_hard_swap_macro_grids_jit(
            a_xy,
            b_xy,
            float(cong_cache["hard_half_w"][a_slot]),
            float(cong_cache["hard_half_h"][a_slot]),
            b_half_w,
            b_half_h,
            bool(b_is_hard),
            self.V_macro_flat,
            self.H_macro_flat,
            self.grid_w,
            self.grid_h,
            self.grid_row,
            self.grid_col,
            float(self.plc.vrouting_alloc),
            float(self.plc.hrouting_alloc),
            v_macro,
            h_macro,
        )

        congestion_values = np.empty((n_batch, 2 * n_cells), dtype=np.float64)
        _batch_hard_swap_congestion_values_jit(
            raw_h,
            raw_v,
            bboxes,
            np.ascontiguousarray(self.H_smoothed.ravel()),
            np.ascontiguousarray(self.V_smoothed.ravel()),
            h_macro,
            v_macro,
            self.grid_h_routes,
            self.grid_v_routes,
            self._sm_row_lp if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_row_up if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_row_cnt if self.smooth_range > 0 else np.empty(0, dtype=np.float64),
            self._sm_col_lp if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_col_up if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_col_cnt if self.smooth_range > 0 else np.empty(0, dtype=np.float64),
            self.smooth_range,
            self.grid_row,
            self.grid_col,
            congestion_values,
        )
        congestion_count = int(congestion_values.shape[1] * 0.05)
        if congestion_count == 0:
            congestion = congestion_values.max(axis=1)
        else:
            top = np.partition(
                congestion_values,
                congestion_values.shape[1] - congestion_count,
                axis=1,
            )[:, -congestion_count:]
            congestion = top.sum(axis=1) / congestion_count

        wirelength = np.empty(n_batch, dtype=np.float64)
        _batch_soft_swap_wirelength_jit(
            batch["a_module"],
            batch["b_modules"],
            a_xy,
            b_xy,
            batch["touched_offsets"],
            batch["touched"],
            self.net_starts,
            self.net_lengths,
            self.ref_inv,
            self.unique_ref,
            self.x_off,
            self.y_off,
            pos_cache,
            self.per_net_hpwl,
            self.net_weights,
            self.total_wl_raw,
            self.wl_normalizer,
            wirelength,
        )

        density_grids = np.empty((n_batch, self.dens_n_cells), dtype=np.float64)
        a_half_w, a_half_h = self._dens_half[a_module]
        b_half = np.asarray(
            [self._dens_half[int(module)] for module in b_modules], dtype=np.float64
        )
        _batch_soft_swap_density_grids_jit(
            a_xy,
            b_xy,
            a_half_w,
            a_half_h,
            np.ascontiguousarray(b_half[:, 0]),
            np.ascontiguousarray(b_half[:, 1]),
            self.grid_occupied,
            self.dens_grid_w,
            self.dens_grid_h,
            self.dens_grid_row,
            self.dens_grid_col,
            density_grids,
        )
        density = np.empty(n_batch, dtype=np.float64)
        for batch_idx in range(n_batch):
            occupied = density_grids[batch_idx]
            nonzero = occupied[occupied != 0.0]
            if nonzero.size == 0:
                density[batch_idx] = 0.0
            elif self.dens_n_cells < 10:
                density[batch_idx] = 0.5 * float(nonzero.mean() / self.dens_grid_area)
            else:
                count = min(self.dens_density_cnt, nonzero.size)
                top = np.partition(nonzero, nonzero.size - count)[nonzero.size - count :]
                density[batch_idx] = (
                    0.5 * float(top.sum()) / self.dens_grid_area / self.dens_density_cnt
                )
        return wirelength + 0.5 * density + 0.5 * congestion

    def score_swap_soft_soft_many(self, soft_a: int, candidates: np.ndarray) -> np.ndarray:
        """Score soft-soft swaps for one soft macro, preserving candidate order."""
        cand = np.asarray(candidates, dtype=np.int64).reshape(-1)
        if not HAS_NUMBA or cand.size < 2:
            out = np.empty(cand.size, dtype=np.float64)
            for k, soft_b in enumerate(cand):
                out[k] = self.score_swap_soft_soft(soft_a, int(soft_b))
            return out

        a_module = int(self.soft_indices[int(soft_a)])
        b_modules = np.asarray([self.soft_indices[int(soft_b)] for soft_b in cand], dtype=np.int64)
        batch = self._prepare_pair_swap_batch(a_module, b_modules)
        if batch is None:
            out = np.empty(cand.size, dtype=np.float64)
            for k, soft_b in enumerate(cand):
                out[k] = self.score_swap_soft_soft(soft_a, int(soft_b))
            return out

        n_cells = self.grid_row * self.grid_col
        raw_h = np.empty((cand.size, n_cells), dtype=np.float64)
        raw_v = np.empty((cand.size, n_cells), dtype=np.float64)
        bboxes = np.empty((cand.size, 4), dtype=np.int64)
        a_xy = np.ascontiguousarray(self.committed_soft_pos[int(soft_a)], dtype=np.float64)
        b_xy = np.ascontiguousarray(self.committed_soft_pos[cand], dtype=np.float64)
        pos_cache = np.ascontiguousarray(_ensure_pos_cache(self.plc), dtype=np.float64)
        _batch_soft_swap_route_grids_jit(
            batch["a_module"],
            batch["b_modules"],
            a_xy,
            b_xy,
            pos_cache,
            self.H_flat,
            self.V_flat,
            raw_h,
            raw_v,
            bboxes,
            batch["pin_offsets"],
            batch["pin_gcell"],
            batch["pin_module"],
            batch["pin_x_off"],
            batch["pin_y_off"],
            batch["starts2_offsets"],
            batch["starts2"],
            batch["weights2"],
            batch["starts3_offsets"],
            batch["starts3"],
            batch["weights3"],
            batch["starts4_offsets"],
            batch["starts4"],
            batch["lengths4"],
            batch["weights4"],
            batch["unique_sinks"],
            batch["three_g0"],
            batch["three_g1"],
            batch["three_g2"],
            batch["three_weights"],
            self.grid_w,
            self.grid_h,
            self.grid_row,
            self.grid_col,
        )

        congestion_values = np.empty((cand.size, 2 * n_cells), dtype=np.float64)
        _batch_soft_congestion_values_jit(
            raw_h,
            raw_v,
            bboxes,
            np.ascontiguousarray(self.H_smoothed.ravel()),
            np.ascontiguousarray(self.V_smoothed.ravel()),
            self.H_macro_flat,
            self.V_macro_flat,
            self.grid_h_routes,
            self.grid_v_routes,
            self._sm_row_lp if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_row_up if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_row_cnt if self.smooth_range > 0 else np.empty(0, dtype=np.float64),
            self._sm_col_lp if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_col_up if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_col_cnt if self.smooth_range > 0 else np.empty(0, dtype=np.float64),
            self.smooth_range,
            self.grid_row,
            self.grid_col,
            congestion_values,
        )
        congestion_count = int(congestion_values.shape[1] * 0.05)
        if congestion_count == 0:
            congestion = congestion_values.max(axis=1)
        else:
            top = np.partition(
                congestion_values,
                congestion_values.shape[1] - congestion_count,
                axis=1,
            )[:, -congestion_count:]
            congestion = top.sum(axis=1) / congestion_count

        wirelength = np.empty(cand.size, dtype=np.float64)
        _batch_soft_swap_wirelength_jit(
            batch["a_module"],
            batch["b_modules"],
            a_xy,
            b_xy,
            batch["touched_offsets"],
            batch["touched"],
            self.net_starts,
            self.net_lengths,
            self.ref_inv,
            self.unique_ref,
            self.x_off,
            self.y_off,
            pos_cache,
            self.per_net_hpwl,
            self.net_weights,
            self.total_wl_raw,
            self.wl_normalizer,
            wirelength,
        )

        density_grids = np.empty((cand.size, self.dens_n_cells), dtype=np.float64)
        a_half_w, a_half_h = self._dens_half[int(batch["a_module"])]
        b_half = np.asarray(
            [self._dens_half[int(module)] for module in batch["b_modules"]], dtype=np.float64
        )
        _batch_soft_swap_density_grids_jit(
            a_xy,
            b_xy,
            a_half_w,
            a_half_h,
            np.ascontiguousarray(b_half[:, 0]),
            np.ascontiguousarray(b_half[:, 1]),
            self.grid_occupied,
            self.dens_grid_w,
            self.dens_grid_h,
            self.dens_grid_row,
            self.dens_grid_col,
            density_grids,
        )
        density = np.empty(cand.size, dtype=np.float64)
        for batch_idx in range(cand.size):
            occupied = density_grids[batch_idx]
            nonzero = occupied[occupied != 0.0]
            if nonzero.size == 0:
                density[batch_idx] = 0.0
            elif self.dens_n_cells < 10:
                density[batch_idx] = 0.5 * float(nonzero.mean() / self.dens_grid_area)
            else:
                count = min(self.dens_density_cnt, nonzero.size)
                top = np.partition(nonzero, nonzero.size - count)[nonzero.size - count :]
                density[batch_idx] = (
                    0.5 * float(top.sum()) / self.dens_grid_area / self.dens_density_cnt
                )
        return wirelength + 0.5 * density + 0.5 * congestion

    def commit_swap_soft_soft(self, soft_a: int, soft_b: int) -> None:
        """Commit exchanging two soft macro centers."""
        soft_a, soft_b = int(soft_a), int(soft_b)
        am, bm = int(self.soft_indices[soft_a]), int(self.soft_indices[soft_b])
        a_xy = tuple(self.committed_soft_pos[soft_a])
        b_xy = tuple(self.committed_soft_pos[soft_b])
        self._commit_multi_move(
            [am, bm],
            [a_xy, b_xy],
            [b_xy, a_xy],
            [],
            {},
            {soft_a: b_xy, soft_b: a_xy},
        )

    def score_swap_hard_soft(self, i_hard: int, soft_k: int) -> float:
        """Score exchanging a hard macro center with a soft macro center."""
        i_hard, soft_k = int(i_hard), int(soft_k)
        hm, sm = int(self.hard_indices[i_hard]), int(self.soft_indices[soft_k])
        h_xy = tuple(self.committed_hard_pos[i_hard])
        s_xy = tuple(self.committed_soft_pos[soft_k])
        return self._score_multi_move(
            [hm, sm],
            [h_xy, s_xy],
            [s_xy, h_xy],
            self._hard_slot_array(i_hard),
        )

    def score_swap_hard_soft_many(self, i_hard: int, candidates: np.ndarray) -> np.ndarray:
        """Score hard-soft swaps for one hard macro, preserving candidate order."""
        cand = np.asarray(candidates, dtype=np.int64).reshape(-1)
        b_modules = np.asarray([self.soft_indices[int(soft_k)] for soft_k in cand], dtype=np.int64)
        b_xy = self.committed_soft_pos[cand]
        batch = self._score_hard_endpoint_swaps_many(int(i_hard), b_modules, b_xy, b_is_hard=False)
        if batch is not None:
            return batch
        out = np.empty(cand.size, dtype=np.float64)
        for k, soft_k in enumerate(cand):
            out[k] = self.score_swap_hard_soft(i_hard, int(soft_k))
        return out

    def commit_swap_hard_soft(self, i_hard: int, soft_k: int) -> None:
        """Commit exchanging a hard macro center with a soft macro center."""
        i_hard, soft_k = int(i_hard), int(soft_k)
        hm, sm = int(self.hard_indices[i_hard]), int(self.soft_indices[soft_k])
        h_xy = tuple(self.committed_hard_pos[i_hard])
        s_xy = tuple(self.committed_soft_pos[soft_k])
        self._commit_multi_move(
            [hm, sm],
            [h_xy, s_xy],
            [s_xy, h_xy],
            self._hard_slot_array(i_hard),
            {i_hard: s_xy},
            {soft_k: h_xy},
        )

    # Relocation prep removes the old macro once, then trials many targets.

    def _prepare_move(self, i_hard: int) -> dict:
        """Remove one hard macro before trying target positions."""
        i_module = self.hard_indices[i_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        struct = self._route_struct(i_module)
        macro_subset = (
            np.array([i_slot], dtype=np.int64)
            if i_slot is not None
            else np.empty(0, dtype=np.int64)
        )

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0, self.V_macro_flat, self.H_macro_flat
            )
        o_idx, o_area = self._macro_occ(i_module, old_ix, old_iy)
        if o_idx.size:
            np.subtract.at(self.grid_occupied, o_idx, o_area)
        if bb_old is not None:
            self._resmooth_bbox(*bb_old)

        return {
            "kind": "hard",
            "i_module": i_module,
            "i_hard": i_hard,
            "i_slot": i_slot,
            "old_ix": old_ix,
            "old_iy": old_iy,
            "struct": struct,
            "macro_subset": macro_subset,
            "old_dens_idx": o_idx,
            "old_dens_area": o_area,
            "bb_old": bb_old,
        }

    def _trial_at(self, prep: dict, new_xy) -> float:
        """Score one target after `_prepare_move`."""
        i_module = prep["i_module"]
        struct = prep["struct"]
        macro_subset = prep["macro_subset"]
        old_ix, old_iy = prep["old_ix"], prep["old_iy"]
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])
        touched = self._touched_nets_only(i_module)

        H_snap = self.H_flat.copy() if touched.size else None
        V_snap = self.V_flat.copy() if touched.size else None
        Hm_snap = self.H_macro_flat.copy() if macro_subset.size else None
        Vm_snap = self.V_macro_flat.copy() if macro_subset.size else None

        self._apply_pos(i_module, new_ix, new_iy)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0, self.V_macro_flat, self.H_macro_flat
            )

        Hs_snap = Vs_snap = None
        r_lo = r_hi = c_lo = c_hi = None
        if bb_new is not None:
            r_lo, r_hi, c_lo, c_hi = bb_new
            Hs_snap = self.H_smoothed[:, c_lo : c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo : r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        touched = self._macro_nets(i_module)
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer
        cong = self._compute_cong_cost()

        n_idx, n_area = self._macro_occ(i_module, new_ix, new_iy)
        go = self.grid_occupied
        if n_idx.size:
            np.add.at(go, n_idx, n_area)
        dens = self._compute_density_cost()
        if n_idx.size:
            np.subtract.at(go, n_idx, n_area)

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        if bb_new is not None:
            self.H_smoothed[:, c_lo : c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo : r_hi + 1, :] = Vs_snap
        self._apply_pos(i_module, old_ix, old_iy)
        if touched.size:
            self.H_flat[:] = H_snap
            self.V_flat[:] = V_snap
        if Hm_snap is not None:
            self.H_macro_flat[:] = Hm_snap
            self.V_macro_flat[:] = Vm_snap

        return score

    def _trial_many_at(self, prep: dict, xy_array: np.ndarray) -> np.ndarray:
        """Score several hard targets after `_prepare_move`, preserving order."""
        xy = np.asarray(xy_array, dtype=np.float64).reshape(-1, 2)
        out = np.empty(xy.shape[0], dtype=np.float64)
        for k in range(xy.shape[0]):
            out[k] = self._trial_at(prep, xy[k])
        return out

    def _commit_after_prep(self, prep: dict, new_xy) -> None:
        """Commit the winning target after `_prepare_move`."""
        i_module = prep["i_module"]
        i_hard = prep["i_hard"]
        struct = prep["struct"]
        macro_subset = prep["macro_subset"]
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])

        self._apply_pos(i_module, new_ix, new_iy)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0, self.V_macro_flat, self.H_macro_flat
            )

        n_idx, n_area = self._macro_occ(i_module, new_ix, new_iy)
        if n_idx.size:
            np.add.at(self.grid_occupied, n_idx, n_area)

        if bb_new is not None:
            self._resmooth_bbox(*bb_new)

        self.committed_hard_pos[i_hard, 0] = new_ix
        self.committed_hard_pos[i_hard, 1] = new_iy
        touched = self._macro_nets(i_module)
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def _revert_prep(self, prep: dict) -> None:
        """Undo `_prepare_move` when no target wins."""
        struct = prep["struct"]
        macro_subset = prep["macro_subset"]

        bb_old = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0, self.V_macro_flat, self.H_macro_flat
            )
        o_idx = prep["old_dens_idx"]
        o_area = prep["old_dens_area"]
        if o_idx.size:
            np.add.at(self.grid_occupied, o_idx, o_area)
        if bb_old is not None:
            self._resmooth_bbox(*bb_old)

    # Soft versions skip hard-macro blockage.

    def _prepare_move_soft(self, soft_k: int) -> dict:
        """Remove one soft macro before trying target positions."""
        s_module = self.soft_indices[soft_k]
        old_x = float(self.committed_soft_pos[soft_k, 0])
        old_y = float(self.committed_soft_pos[soft_k, 1])
        struct = self._route_struct(s_module)

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        o_idx, o_area = self._macro_occ(s_module, old_x, old_y)
        if o_idx.size:
            np.subtract.at(self.grid_occupied, o_idx, o_area)
        if bb_old is not None:
            self._resmooth_bbox(*bb_old)

        return {
            "kind": "soft",
            "s_module": s_module,
            "soft_k": soft_k,
            "old_x": old_x,
            "old_y": old_y,
            "struct": struct,
            "old_dens_idx": o_idx,
            "old_dens_area": o_area,
            "bb_old": bb_old,
        }

    def _trial_at_soft(self, prep: dict, new_xy) -> float:
        """Score one soft target after `_prepare_move_soft`."""
        s_module = prep["s_module"]
        struct = prep["struct"]
        old_x, old_y = prep["old_x"], prep["old_y"]
        new_x, new_y = float(new_xy[0]), float(new_xy[1])
        touched = self._touched_nets_only(s_module)

        H_snap = self.H_flat.copy() if touched.size else None
        V_snap = self.V_flat.copy() if touched.size else None

        self._apply_pos(s_module, new_x, new_y)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)

        Hs_snap = Vs_snap = None
        r_lo = r_hi = c_lo = c_hi = None
        if bb_new is not None:
            r_lo, r_hi, c_lo, c_hi = bb_new
            Hs_snap = self.H_smoothed[:, c_lo : c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo : r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        touched = self._macro_nets(s_module)
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer
        cong = self._compute_cong_cost()

        n_idx, n_area = self._macro_occ(s_module, new_x, new_y)
        go = self.grid_occupied
        if n_idx.size:
            np.add.at(go, n_idx, n_area)
        dens = self._compute_density_cost()
        if n_idx.size:
            np.subtract.at(go, n_idx, n_area)

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        if bb_new is not None:
            self.H_smoothed[:, c_lo : c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo : r_hi + 1, :] = Vs_snap
        self._apply_pos(s_module, old_x, old_y)
        if touched.size:
            self.H_flat[:] = H_snap
            self.V_flat[:] = V_snap

        return score

    def _trial_many_at_soft(self, prep: dict, xy_array: np.ndarray) -> np.ndarray:
        """Score several soft targets after `_prepare_move_soft`, preserving order."""
        xy = np.asarray(xy_array, dtype=np.float64).reshape(-1, 2)
        if not HAS_NUMBA or xy.shape[0] < 2 or prep["struct"] is None:
            out = np.empty(xy.shape[0], dtype=np.float64)
            for k in range(xy.shape[0]):
                out[k] = self._trial_at_soft(prep, xy[k])
            return out

        batch_size = xy.shape[0]
        n_cells = self.grid_row * self.grid_col
        raw_h = np.empty((batch_size, n_cells), dtype=np.float64)
        raw_v = np.empty((batch_size, n_cells), dtype=np.float64)
        bboxes = np.empty((batch_size, 4), dtype=np.int64)
        struct_jit = prep["struct"]["jit"]
        pos_cache = np.ascontiguousarray(_ensure_pos_cache(self.plc), dtype=np.float64)
        _batch_soft_route_grids_jit(
            xy,
            int(prep["s_module"]),
            pos_cache,
            self.H_flat,
            self.V_flat,
            raw_h,
            raw_v,
            bboxes,
            struct_jit["pin_gcell"],
            struct_jit["pin_module"],
            struct_jit["pin_x_off"],
            struct_jit["pin_y_off"],
            struct_jit["starts2"],
            struct_jit["weights2"],
            struct_jit["starts3"],
            struct_jit["weights3"],
            struct_jit["starts4"],
            struct_jit["lengths4"],
            struct_jit["weights4"],
            struct_jit["unique_sinks"],
            struct_jit["three_g0"],
            struct_jit["three_g1"],
            struct_jit["three_g2"],
            struct_jit["three_weights"],
            self.grid_w,
            self.grid_h,
            self.grid_row,
            self.grid_col,
        )

        congestion_values = np.empty((batch_size, 2 * n_cells), dtype=np.float64)
        _batch_soft_congestion_values_jit(
            raw_h,
            raw_v,
            bboxes,
            np.ascontiguousarray(self.H_smoothed.ravel()),
            np.ascontiguousarray(self.V_smoothed.ravel()),
            self.H_macro_flat,
            self.V_macro_flat,
            self.grid_h_routes,
            self.grid_v_routes,
            self._sm_row_lp if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_row_up if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_row_cnt if self.smooth_range > 0 else np.empty(0, dtype=np.float64),
            self._sm_col_lp if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_col_up if self.smooth_range > 0 else np.empty(0, dtype=np.int64),
            self._sm_col_cnt if self.smooth_range > 0 else np.empty(0, dtype=np.float64),
            self.smooth_range,
            self.grid_row,
            self.grid_col,
            congestion_values,
        )
        congestion_count = int(congestion_values.shape[1] * 0.05)
        if congestion_count == 0:
            congestion = congestion_values.max(axis=1)
        else:
            top = np.partition(
                congestion_values,
                congestion_values.shape[1] - congestion_count,
                axis=1,
            )[:, -congestion_count:]
            congestion = top.sum(axis=1) / congestion_count

        touched = self._macro_nets(int(prep["s_module"]))
        wirelength = np.empty(batch_size, dtype=np.float64)
        _batch_soft_wirelength_jit(
            xy,
            int(prep["s_module"]),
            np.ascontiguousarray(touched, dtype=np.int64),
            self.net_starts,
            self.net_lengths,
            self.ref_inv,
            self.unique_ref,
            self.x_off,
            self.y_off,
            pos_cache,
            self.per_net_hpwl,
            self.net_weights,
            self.total_wl_raw,
            self.wl_normalizer,
            wirelength,
        )

        density_grids = np.empty((batch_size, self.dens_n_cells), dtype=np.float64)
        half_w, half_h = self._dens_half[int(prep["s_module"])]
        _batch_soft_density_grids_jit(
            xy,
            half_w,
            half_h,
            self.grid_occupied,
            self.dens_grid_w,
            self.dens_grid_h,
            self.dens_grid_row,
            self.dens_grid_col,
            density_grids,
        )
        density = np.empty(batch_size, dtype=np.float64)
        for batch_idx in range(batch_size):
            occupied = density_grids[batch_idx]
            nonzero = occupied[occupied != 0.0]
            if nonzero.size == 0:
                density[batch_idx] = 0.0
            elif self.dens_n_cells < 10:
                density[batch_idx] = 0.5 * float(nonzero.mean() / self.dens_grid_area)
            else:
                count = min(self.dens_density_cnt, nonzero.size)
                top = np.partition(nonzero, nonzero.size - count)[nonzero.size - count :]
                density[batch_idx] = (
                    0.5 * float(top.sum()) / self.dens_grid_area / self.dens_density_cnt
                )

        return wirelength + 0.5 * density + 0.5 * congestion

    def _commit_after_prep_soft(self, prep: dict, new_xy) -> None:
        """Commit the winning soft target after `_prepare_move_soft`."""
        s_module = prep["s_module"]
        soft_k = prep["soft_k"]
        struct = prep["struct"]
        new_x, new_y = float(new_xy[0]), float(new_xy[1])

        self._apply_pos(s_module, new_x, new_y)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)

        n_idx, n_area = self._macro_occ(s_module, new_x, new_y)
        if n_idx.size:
            np.add.at(self.grid_occupied, n_idx, n_area)

        if bb_new is not None:
            self._resmooth_bbox(*bb_new)

        self.committed_soft_pos[soft_k, 0] = new_x
        self.committed_soft_pos[soft_k, 1] = new_y
        touched = self._macro_nets(s_module)
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(
                np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
            )
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def _revert_prep_soft(self, prep: dict) -> None:
        """Undo `_prepare_move_soft` when no target wins."""
        struct = prep["struct"]
        bb_old = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        o_idx = prep["old_dens_idx"]
        o_area = prep["old_dens_area"]
        if o_idx.size:
            np.add.at(self.grid_occupied, o_idx, o_area)
        if bb_old is not None:
            self._resmooth_bbox(*bb_old)

    def wl_delta_move_soft(self, soft_k: int, new_xy) -> float:
        """Return the wirelength-only delta for a soft relocation."""
        s_module = self.soft_indices[soft_k]
        touched = self._macro_nets(s_module)
        if len(touched) == 0:
            return 0.0
        pos_cache = _ensure_pos_cache(self.plc)
        sx = float(pos_cache[s_module, 0])
        sy = float(pos_cache[s_module, 1])
        pos_cache[s_module, 0] = float(new_xy[0])
        pos_cache[s_module, 1] = float(new_xy[1])
        new_per_net = self._compute_per_net_hpwl_subset(touched)
        delta = float(
            np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched])
        )
        pos_cache[s_module, 0] = sx
        pos_cache[s_module, 1] = sy
        return delta / self.wl_normalizer
