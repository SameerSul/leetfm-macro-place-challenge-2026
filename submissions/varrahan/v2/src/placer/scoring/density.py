"""Vectorized density helpers."""

import numpy as np
from macro_place.benchmark import Benchmark

from placer.plc.placement import _ensure_pos_cache

def _build_density_cache(plc, benchmark: Benchmark):
    """One-time precomputation per plc for the vectorized density path.

    Density depends on (macro half-widths/heights) which are immutable, and
    macro center positions which change. Cache the immutable arrays once;
    the per-call work is a vectorized scatter-add into grid_occupied.
    """
    if hasattr(plc, "_density_cache"):
        return plc._density_cache
    macro_indices = list(plc.soft_macro_indices) + list(plc.hard_macro_indices)
    n_mod = len(macro_indices)
    half_w = np.empty(n_mod, dtype=np.float64)
    half_h = np.empty(n_mod, dtype=np.float64)
    for k, idx in enumerate(macro_indices):
        m = plc.modules_w_pins[idx]
        half_w[k] = float(m.get_width()) * 0.5
        half_h[k] = float(m.get_height()) * 0.5
    plc._density_cache = {
        "macro_indices": macro_indices,
        "half_w": half_w,
        "half_h": half_h,
        "n_mod": n_mod,
    }
    return plc._density_cache


def _vectorized_get_grid_cells_density(plc) -> "list[float]":
    """Drop-in numpy replacement for plc.get_grid_cells_density().

    Matches the reference: for each soft+hard macro, distribute the area it
    overlaps with each grid cell into grid_occupied, then normalize by
    grid_area to produce grid_cells. Uses cached half-sizes; reads positions
    fresh each call.

    Out-of-canvas behavior is the same as the reference: a macro whose
    bounding box doesn't intersect any in-canvas cell is skipped. The
    reference's row/col clamping logic is reproduced via np.clip.
    """
    cache = plc._density_cache
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)
    plc.grid_width = grid_w
    plc.grid_height = grid_h

    n_cells = grid_col * grid_row
    grid_occupied = np.zeros(n_cells, dtype=np.float64)

    n_mod = cache["n_mod"]
    if n_mod == 0:
        plc.grid_occupied = grid_occupied.tolist()
        plc.grid_cells = [0.0] * n_cells
        return plc.grid_cells

    # B3 (2026-05-23): use global pos cache instead of per-macro get_pos loop.
    macro_indices = cache["macro_indices"]
    pos_cache = _ensure_pos_cache(plc)
    macro_indices_arr = (
        cache.get("macro_indices_arr") if isinstance(cache, dict) else None
    )
    if macro_indices_arr is None:
        macro_indices_arr = np.asarray(macro_indices, dtype=np.int64)
        cache["macro_indices_arr"] = macro_indices_arr
    pos_x = pos_cache[macro_indices_arr, 0]
    pos_y = pos_cache[macro_indices_arr, 1]

    half_w = cache["half_w"]
    half_h = cache["half_h"]
    x_min = pos_x - half_w
    x_max = pos_x + half_w
    y_min = pos_y - half_h
    y_max = pos_y + half_h

    # Mirror the reference's grid cell location (floor, then clamp). Floor
    # at edges gets corrected by the clamping step below; OOB modules
    # (both corners outside canvas) are filtered out.
    bl_col = np.floor(x_min / grid_w).astype(np.int64)
    bl_row = np.floor(y_min / grid_h).astype(np.int64)
    ur_col = np.floor(x_max / grid_w).astype(np.int64)
    ur_row = np.floor(y_max / grid_h).astype(np.int64)

    # OOB skip: if either corner pair places the macro entirely outside
    # the canvas, the reference skips the macro. Mirror via a mask.
    in_bounds = (ur_row >= 0) & (ur_col >= 0) & (bl_row <= grid_row - 1) & (bl_col <= grid_col - 1)
    bl_col = np.clip(bl_col, 0, grid_col - 1)
    bl_row = np.clip(bl_row, 0, grid_row - 1)
    ur_col = np.clip(ur_col, 0, grid_col - 1)
    ur_row = np.clip(ur_row, 0, grid_row - 1)

    # Fully-batched scatter via np.bincount (weights). Mirrors the structure
    # of _apply_macro_routing's vectorized rectangle expansion:
    #   1. Filter to in-bounds macros.
    #   2. Per-cell (macro_idx, row_offset, col_offset) via flat enumeration.
    #   3. Compute per-cell overlap area = ox * oy.
    #   4. bincount over flat cell indices.
    if not in_bounds.any():
        grid_area = grid_w * grid_h
        plc.grid_occupied = grid_occupied.tolist()
        plc.grid_cells = (grid_occupied / grid_area).tolist()
        return plc.grid_cells

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
        grid_area = grid_w * grid_h
        plc.grid_occupied = grid_occupied.tolist()
        plc.grid_cells = (grid_occupied / grid_area).tolist()
        return plc.grid_cells

    macro_idx = np.repeat(np.arange(sel.size, dtype=np.int64), n_cells_per)
    cum = np.zeros(sel.size + 1, dtype=np.int64)
    np.cumsum(n_cells_per, out=cum[1:])
    local_idx = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], n_cells_per)
    n_cols_per_cell = n_cols_per[macro_idx]
    row_off = local_idx // n_cols_per_cell
    col_off = local_idx - row_off * n_cols_per_cell
    rr_g = bl_row_s[macro_idx] + row_off
    cc_g = bl_col_s[macro_idx] + col_off
    flat_idx_cells = rr_g * grid_col + cc_g

    cell_xmin = grid_w * cc_g.astype(np.float64)
    cell_xmax = grid_w * (cc_g + 1).astype(np.float64)
    cell_ymin = grid_h * rr_g.astype(np.float64)
    cell_ymax = grid_h * (rr_g + 1).astype(np.float64)
    x_max_pc = x_max_s[macro_idx]
    x_min_pc = x_min_s[macro_idx]
    y_max_pc = y_max_s[macro_idx]
    y_min_pc = y_min_s[macro_idx]
    ox = np.minimum(cell_xmax, x_max_pc) - np.maximum(cell_xmin, x_min_pc)
    oy = np.minimum(cell_ymax, y_max_pc) - np.maximum(cell_ymin, y_min_pc)
    np.maximum(ox, 0.0, out=ox)
    np.maximum(oy, 0.0, out=oy)
    ov = ox * oy

    # scatter_add_ on DirectML has a last-write-wins bug for duplicate indices
    # (does not accumulate as expected), so the GPU scatter path is removed.
    # np.bincount with weights handles duplicate indices correctly and is fast
    # enough for all benchmarks (even ibm18 with ~4K entries runs in < 1ms).
    grid_occupied = np.bincount(flat_idx_cells, weights=ov, minlength=n_cells)
    grid_area = grid_w * grid_h
    grid_cells = grid_occupied / grid_area
    plc.grid_occupied = grid_occupied.tolist()
    plc.grid_cells = grid_cells.tolist()
    return plc.grid_cells


def _patch_plc_density(plc, benchmark: Benchmark) -> None:
    """Install vectorized density on this plc instance (idempotent)."""
    if getattr(plc, "_density_vec_installed", False):
        return
    _build_density_cache(plc, benchmark)
    plc.get_grid_cells_density = lambda _plc=plc: _vectorized_get_grid_cells_density(_plc)
    plc._density_vec_installed = True


# ---------------------------------------------------------------------------
# Vectorized congestion (get_routing)
# ---------------------------------------------------------------------------
# On ibm10 the scalar plc.get_routing takes ~24.6s per call — dominant cost
# of every scoring call. The native Python loop processes ~50000 nets serially
# with per-net Python overhead (4+ method calls per pin, set-build, branchy
# L/T routing). This vectorized replacement:
#   1. Reuses the per-pin cache built for wirelength (ref_idx + offset
#      arrays) to compute all pin grid cells in one numpy gather.
#   2. Buckets nets by unique-gcell count (1/2/3/many). 2-pin nets — the
#      majority — get batched into flat (source_row/col, sink_row/col,
#      weight) arrays and applied via the difference-array prefix-sum trick
#      (O(strips + grid) rather than O(strip_length × strips)). 3-pin nets
#      are rare; they get a small Python loop matching __three_pin_net_routing
#      exactly. ≥4-gcell nets fan out into source→sink 2-pin edges.
#   3. Hard-macro routing: vectorized per-cell overlap × vrouting/hrouting
#      alloc, then partial-overlap correction.
#   4. Smoothing: 1-D box-blur via cumsum.
# Goal: 24.6s → <5s on ibm10; matches scalar output exactly (integer grid-
# cell indices + sum-of-weights — no FP order sensitivity).
# ---------------------------------------------------------------------------
