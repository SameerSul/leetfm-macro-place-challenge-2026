"""Incremental proxy scorer used by local-search moves."""

import numpy as np
from macro_place.benchmark import Benchmark

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

class IncrementalScorer:
    """Stateful incremental proxy scorer for local-search moves.

    Maintains all three proxy terms as state and updates only what a move
    touches, so a 1-2 macro move costs ~1ms instead of a full recompute:
      - wirelength: recompute HPWL only for the moved macros' nets (touched_nets
        = union of their net sets), via gather + reduceat over a compact range.
      - congestion: maintain the H/V routing flats + a smoothed cache; re-apply
        only the touched-net routing demand and re-smooth only the affected bbox.
      - density: maintain the occupancy grid; update only the moved macro's cells.

    Each move type has a score_* (apply / compute / revert) and commit_* (apply /
    persist) pair; the scorer mirrors plc's set_pos calls, so a reverted trial
    leaves plc unchanged and the caller need only commit after an accept. Verified
    bit-exact against `_exact_proxy`.

    Hard indices (i_hard) are into `benchmark.hard_macro_indices`
    (0 <= i_hard < n_hard); soft indices are into the soft-macro block. Both are
    translated to plc module indices internally.
    """

    def __init__(self, plc, benchmark: Benchmark, current_placement_np: np.ndarray):
        self.plc = plc
        self.benchmark = benchmark
        self.n_hard = benchmark.num_hard_macros
        self.hard_indices = list(benchmark.hard_macro_indices)

        # Force a FULL set: `_apply_pos` keeps `_global_pos_cache` in sync but not
        # `_last_pos_cache`, so after a prior move mutated plc the stale
        # `_last_pos_cache` could make `_fast_set_placement` skip macros and build
        # the WL baseline against wrong positions. Invalidate it to re-set every
        # macro to current_placement_np.
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

        # Macro index → array of net indices that contain at least one of
        # the macro's pins. Built once per benchmark from ref_idx + pin_to_net.
        self._build_macro_to_nets()

        # WL normalization: plc.get_cost() = sum(weighted HPWL) /
        # ((canvas_w + canvas_h) * net_cnt). We must apply the same divisor
        # so score_swap matches `_exact_proxy` exactly (which calls get_cost).
        cw_, ch_ = plc.get_canvas_width_height()
        self.wl_normalizer = float((cw_ + ch_) * max(plc.net_cnt, 1))

        # Initial per-net HPWL + total WL (full recompute, ~3ms one-time).
        # `per_net_hpwl` is RAW HPWL (max-min); `total_wl_raw` is the
        # weighted sum BEFORE normalization. The normalized WL used in
        # proxy is `total_wl_raw / wl_normalizer`.
        self.per_net_hpwl = self._compute_per_net_hpwl_full()
        self.total_wl_raw = float(np.sum(self.per_net_hpwl * self.net_weights))

        # Committed hard-macro positions (only hard macros can swap).
        self.committed_hard_pos = current_placement_np[:self.n_hard].astype(np.float64).copy()

        # Soft-macro state (for soft relocation). Softs occupy placement rows
        # n_hard..; they contribute to WL + net-routing congestion + density, but
        # NOT macro-routing blockage (only hard macros block). They may overlap,
        # so soft relocation needs no legality check.
        self.soft_indices = list(benchmark.soft_macro_indices)
        self.num_soft = len(self.soft_indices)
        self.committed_soft_pos = (
            current_placement_np[self.n_hard:self.n_hard + self.num_soft]
            .astype(np.float64).copy()
        )

        # ---- Congestion incremental state. ----
        cong_cache = plc._cong_cache
        self.grid_col = int(plc.grid_col)
        self.grid_row = int(plc.grid_row)
        self.grid_w = float(plc.width / self.grid_col)
        self.grid_h = float(plc.height / self.grid_row)
        self.grid_v_routes = self.grid_w * plc.vroutes_per_micron
        self.grid_h_routes = self.grid_h * plc.hroutes_per_micron
        self.smooth_range = int(plc.smooth_range)
        n_cells = self.grid_row * self.grid_col

        # Build initial RAW (pre-normalize, pre-smooth) routing flats from
        # the current plc state. We call _vectorized_get_routing to
        # populate plc.V_routing_cong / etc (final smoothed+normalized),
        # then build our own state by calling the subset helpers with the
        # FULL net + macro sets.
        plc.get_congestion_cost()  # ensure routing populated
        self.H_flat = np.zeros(n_cells, dtype=np.float64)
        self.V_flat = np.zeros(n_cells, dtype=np.float64)
        self.H_macro_flat = np.zeros(n_cells, dtype=np.float64)
        self.V_macro_flat = np.zeros(n_cells, dtype=np.float64)
        if self.n_nets > 0:
            _apply_net_routing_subset(
                plc, np.arange(self.n_nets, dtype=np.int64), +1.0,
                self.H_flat, self.V_flat,
            )
        n_hard_cache = cong_cache["n_hard"]
        if n_hard_cache > 0:
            _apply_macro_routing_subset(
                plc, np.arange(n_hard_cache, dtype=np.int64), +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # ---- Incremental congestion COST cache. ----
        # `_compute_cong_cost` used to full-re-smooth H/V + re-partition every
        # move-eval; the smoothing was ~85% of the cong cost (~14% of a whole
        # move). The box filter is SEPARABLE (H along columns, V along rows,
        # each cell independent of the other axis), so we cache the smoothed
        # NORMALIZED H/V (2D, grid_row×grid_col) and, per move, recompute only
        # the columns/rows inside the touched-net bbox FROM the raw flats - each
        # recomputed value is identical to a full re-smooth (no delta
        # accumulation → no drift, preserving the bit-exact non-regression
        # guarantee). The window (lp/up/cnt) is fixed by the grid, so precompute.
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
            self.H_smoothed = _smooth_routing_cong_vec(
                self.H_flat / self.grid_h_routes, self.grid_row, self.grid_col,
                sr, axis_h=True,
            ).reshape(self.grid_row, self.grid_col)
            self.V_smoothed = _smooth_routing_cong_vec(
                self.V_flat / self.grid_v_routes, self.grid_row, self.grid_col,
                sr, axis_h=False,
            ).reshape(self.grid_row, self.grid_col)
        else:
            # No smoothing: cache == normalized flats (kept 2D for the subset API).
            self.H_smoothed = (self.H_flat / self.grid_h_routes).reshape(
                self.grid_row, self.grid_col)
            self.V_smoothed = (self.V_flat / self.grid_v_routes).reshape(
                self.grid_row, self.grid_col)

        # Map module-index → hard-macro-slot-index (for _apply_macro_routing_subset).
        self._module_to_hard_slot: "dict[int, int]" = {
            int(m): k for k, m in enumerate(cong_cache["hard_indices"])
        }

        # Per-module topology routing-struct cache. A macro's touched-net
        # structure is placement-independent, so build it once and reuse across
        # that macro's score/commit calls and the -1/+1 applies of each. Keyed by
        # module index; value may be None (macro has no nets).
        self._route_struct_cache: "dict[int, object]" = {}

        # ---- Incremental density state. ----
        # Density was the last full-recompute in score_swap (~28-36% of its time):
        # it scatters ALL soft+hard macros into the
        # occupancy grid every call. But a 2-opt swap moves only macros i, j -
        # all soft + other-hard occupancy is invariant. So maintain `grid_occupied`
        # as state; per swap subtract i,j's old footprints and add their new ones
        # (a handful of cells each), then take the top-10% over the full grid.
        dens_cache = _build_density_cache(plc, benchmark)
        self.dens_grid_col = int(plc.grid_col)
        self.dens_grid_row = int(plc.grid_row)
        self.dens_grid_w = float(plc.width / self.dens_grid_col)
        self.dens_grid_h = float(plc.height / self.dens_grid_row)
        self.dens_grid_area = self.dens_grid_w * self.dens_grid_h
        self.dens_n_cells = self.dens_grid_col * self.dens_grid_row
        self.dens_density_cnt = int(np.floor(self.dens_n_cells * 0.1))
        # Per hard-macro module → (half_w, half_h) for footprint expansion.
        # density_cache stores half sizes in soft-then-hard module order.
        self._dens_half: "dict[int, tuple[float, float]]" = {
            int(m): (float(dens_cache["half_w"][k]), float(dens_cache["half_h"][k]))
            for k, m in enumerate(dens_cache["macro_indices"])
        }
        # Initial occupancy (full scatter, one-time). Reuse the vectorized full
        # path so the baseline grid_occupied is bit-identical to get_density_cost.
        _vectorized_get_grid_cells_density(plc)
        self.grid_occupied = np.asarray(plc.grid_occupied, dtype=np.float64)
        self._dens_empty_idx = np.empty(0, dtype=np.int64)
        self._dens_empty_area = np.empty(0, dtype=np.float64)

    def _macro_occ(self, module_idx: int, cx: float, cy: float):
        """Per-cell occupancy-area contribution of one macro centered at (cx, cy).

        Returns (flat_cell_indices, areas), mirroring the per-macro overlap math
        in `_vectorized_get_grid_cells_density` exactly (floor → bounds skip →
        clip → per-cell intersection area). The footprint is small (~1-9 cells),
        so this is a tiny outer-product, not a grid-wide scatter.
        """
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
        # OOB skip (matches the in_bounds mask in the full path).
        if not (ur_row >= 0 and ur_col >= 0 and bl_row <= grow - 1 and bl_col <= gcol - 1):
            return self._dens_empty_idx, self._dens_empty_area
        bl_col = min(max(bl_col, 0), gcol - 1)
        ur_col = min(max(ur_col, 0), gcol - 1)
        bl_row = min(max(bl_row, 0), grow - 1)
        ur_row = min(max(ur_row, 0), grow - 1)
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
        """Density cost from the maintained `grid_occupied` (P3).

        Mirrors PlacementCost.get_density_cost: 0.5 × mean of the top-10% (by
        count = floor(n_cells·0.1)) densest NONZERO grid cells. grid_cells =
        grid_occupied / grid_area is a monotone scaling, so the top-k set is the
        same; we scale at the end.

        GPU path: torch.topk replaces np.partition for the top-k selection.
        Beneficial on large grids (grid_cells > ~2000) where the partition cost
        is meaningful. Small grids fall back to the NumPy path to avoid H2D
        overhead.
        """
        cnt = self.dens_density_cnt
        go = self.grid_occupied
        nz = go[go != 0.0]
        if nz.size == 0:
            return 0.0
        if self.dens_n_cells < 10:
            return 0.5 * float(nz.mean() / self.dens_grid_area)
        k = min(cnt, nz.size)
        # np.partition is ~50 µs for these array sizes (<2K elements). GPU topk
        # dispatch (>1 ms on DirectML) is 20x slower here - always use CPU.
        top = np.partition(nz, nz.size - k)[nz.size - k:]
        return 0.5 * float(top.sum()) / self.dens_grid_area / cnt

    def _compute_cong_cost(self) -> float:
        """Congestion cost from the cached smoothed routing (`H_smoothed`/
        `V_smoothed`) plus the live macro flats. The cache must be current for the
        touched region - callers re-smooth the touched bbox (`_resmooth_bbox`)
        first. Mirrors the final transform in `_vectorized_get_congestion_cost`.
        Runs on CPU (np.partition on <5K elements is ~50us; GPU topk dispatch is
        >1ms).
        """
        Hm = self.H_macro_flat / self.grid_h_routes
        Vm = self.V_macro_flat / self.grid_v_routes
        xx = np.concatenate([self.V_smoothed.ravel() + Vm, self.H_smoothed.ravel() + Hm])
        n = xx.size
        cnt = int(n * 0.05)
        if cnt == 0:
            return float(xx.max())
        top = np.partition(xx, n - cnt)[n - cnt:]
        return float(top.sum() / cnt)

    @staticmethod
    def _union_bbox(*bbs):
        """Union of (r_lo, r_hi, c_lo, c_hi) boxes (None entries ignored).
        Returns (None, None, None, None) if every box is None."""
        bbs = [b for b in bbs if b is not None]
        if not bbs:
            return None, None, None, None
        return (min(b[0] for b in bbs), max(b[1] for b in bbs),
                min(b[2] for b in bbs), max(b[3] for b in bbs))

    def _resmooth_h_cols(self, c_lo: int, c_hi: int) -> None:
        """Recompute `H_smoothed[:, c_lo:c_hi+1]` from raw `H_flat` (axis_h=True:
        per-column box filter). Bit-identical to a full `_smooth_routing_cong_vec`
        of those columns - H smoothing mixes only rows within a column, so each
        column is independent and a full re-smooth of just these columns matches.

        Uses pure cumsum instead of np.add.at: avoids the O(grid_row) Python
        overhead from duplicate-index accumulation. smoothed[p] = cs[hi(p)] - cs[lo(p)]
        where lo(p)=max(0,p-sr), hi(p)=min(grid_row,p+sr+1), cs=cumsum([0, w]).
        """
        H2d = self.H_flat.reshape(self.grid_row, self.grid_col)
        sub = H2d[:, c_lo:c_hi + 1] / self.grid_h_routes
        if self.smooth_range <= 0:
            self.H_smoothed[:, c_lo:c_hi + 1] = sub
            return
        weighted = sub / self._sm_row_cnt[:, None]         # [grid_row, ncols]
        ncols = sub.shape[1]
        cs = np.empty((self.grid_row + 1, ncols), dtype=np.float64)
        cs[0, :] = 0.0
        np.cumsum(weighted, axis=0, out=cs[1:, :])         # cs[j] = sum(w[0..j-1])
        self.H_smoothed[:, c_lo:c_hi + 1] = cs[self._sm_row_up + 1] - cs[self._sm_row_lp]

    def _resmooth_v_rows(self, r_lo: int, r_hi: int) -> None:
        """Recompute `V_smoothed[r_lo:r_hi+1, :]` from raw `V_flat` (axis_h=False:
        per-row box filter). Bit-identical to a full re-smooth of those rows - V
        smoothing mixes only columns within a row, so each row is independent.

        Uses pure cumsum instead of 2D np.add.at: fully vectorized, no Python loop.
        smoothed[r, q] = cs[r, hi(q)] - cs[r, lo(q)]
        where lo(q)=max(0,q-sr), hi(q)=min(grid_col,q+sr+1), cs=cumsum([0, w], axis=1).
        """
        V2d = self.V_flat.reshape(self.grid_row, self.grid_col)
        sub = V2d[r_lo:r_hi + 1, :] / self.grid_v_routes
        if self.smooth_range <= 0:
            self.V_smoothed[r_lo:r_hi + 1, :] = sub
            return
        nrows = sub.shape[0]
        weighted = sub / self._sm_col_cnt[None, :]          # [nrows, grid_col]
        cs = np.empty((nrows, self.grid_col + 1), dtype=np.float64)
        cs[:, 0] = 0.0
        np.cumsum(weighted, axis=1, out=cs[:, 1:])          # cs[:,j] = sum(w[:,0..j-1])
        self.V_smoothed[r_lo:r_hi + 1, :] = cs[:, self._sm_col_up + 1] - cs[:, self._sm_col_lp]

    def _resmooth_bbox(self, r_lo, r_hi, c_lo, c_hi) -> None:
        """Refresh the smoothed cache for a touched bbox: H per affected column,
        V per affected row. No-op if the bbox is empty (c_lo is None)."""
        if c_lo is None:
            return
        self._resmooth_h_cols(c_lo, c_hi)
        self._resmooth_v_rows(r_lo, r_hi)

    def _build_macro_to_nets(self):
        """Group nets by the macros (modules) that reference them.

        Output: `self.macro_to_nets[module_idx]` is a sorted int64 ndarray of
        net indices. Builds in O(n_pins) via vectorized grouping.
        """
        ref_idx = self.wl_cache["ref_idx"]
        pin_to_net = self.wl_cache["pin_to_net"]
        # Stable-sort pins by macro index, partition by macro boundary.
        order = np.argsort(ref_idx, kind="stable")
        sorted_macros = ref_idx[order]
        sorted_nets = pin_to_net[order]
        # Each contiguous run of identical macro idx corresponds to that macro's pins.
        boundaries = np.flatnonzero(np.diff(sorted_macros) != 0) + 1
        macro_segments = np.split(sorted_nets, boundaries)
        macro_keys = sorted_macros[np.concatenate([[0], boundaries])] if len(sorted_macros) else np.array([], dtype=ref_idx.dtype)
        self.macro_to_nets: "dict[int, np.ndarray]" = {}
        for k, nets_for_macro in zip(macro_keys, macro_segments):
            # Dedupe inside the macro (pin may reuse the same net? rare but safe).
            uniq = np.unique(nets_for_macro)
            self.macro_to_nets[int(k)] = uniq

    def _compute_per_net_hpwl_full(self) -> np.ndarray:
        """Full per-net HPWL recompute (one-time, mirrors `_vectorized_wirelength`)."""
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
        """Recompute HPWL for a subset of nets only.

        Strategy: build a contiguous pin-index gather array over the subset
        nets (using cached net_lengths via repeat + cumulative offsets), then
        a single `reduceat` over that compact array. O(len(touched pins)),
        not O(n_pins).
        """
        if len(net_indices) == 0:
            return np.empty(0, dtype=np.float64)

        starts_t = self.net_starts[net_indices]
        lengths_t = self.net_lengths[net_indices]
        total = int(lengths_t.sum())
        if total == 0:
            return np.zeros(len(net_indices), dtype=np.float64)

        # Build the per-pin gather array: for net k with pin range
        # [starts_t[k], starts_t[k]+lengths_t[k]), expand to that integer range.
        # Implementation: cumulative within-net index (0..lengths_t[k]-1) for
        # each output position, plus the corresponding net's start offset.
        pin_indices = np.repeat(starts_t, lengths_t) + (
            np.arange(total) - np.repeat(np.concatenate([[0], np.cumsum(lengths_t)[:-1]]), lengths_t)
        )

        pos_cache = _ensure_pos_cache(self.plc)
        node_x = pos_cache[self.unique_ref, 0]
        node_y = pos_cache[self.unique_ref, 1]
        pin_x = node_x[self.ref_inv[pin_indices]] + self.x_off[pin_indices]
        pin_y = node_y[self.ref_inv[pin_indices]] + self.y_off[pin_indices]

        # reduceat starts in the compact array
        sub_starts = np.concatenate([[0], np.cumsum(lengths_t)[:-1]])
        max_x = np.maximum.reduceat(pin_x, sub_starts)
        min_x = np.minimum.reduceat(pin_x, sub_starts)
        max_y = np.maximum.reduceat(pin_y, sub_starts)
        min_y = np.minimum.reduceat(pin_y, sub_starts)
        return (max_x - min_x) + (max_y - min_y)

    def _touched_nets(self, i_module: int, j_module: int) -> np.ndarray:
        a = self.macro_to_nets.get(i_module)
        b = self.macro_to_nets.get(j_module)
        if a is None and b is None:
            return np.empty(0, dtype=np.int64)
        if a is None:
            return b
        if b is None:
            return a
        return np.union1d(a, b)

    def _touched_nets3(self, m1: int, m2: int, m3: int) -> np.ndarray:
        """Union of nets touched by 3 modules - for HS3 hard-soft-soft cycle."""
        a = self.macro_to_nets.get(m1)
        b = self.macro_to_nets.get(m2)
        c = self.macro_to_nets.get(m3)
        parts = [x for x in (a, b, c) if x is not None]
        if not parts:
            return np.empty(0, dtype=np.int64)
        if len(parts) == 1:
            return parts[0]
        return np.unique(np.concatenate(parts))

    def _apply_pos(self, module_idx: int, x: float, y: float) -> None:
        """set_pos + update global pos cache + dirty-flag plc."""
        self.plc.modules_w_pins[module_idx].set_pos(float(x), float(y))
        pos_cache = _ensure_pos_cache(self.plc)
        pos_cache[module_idx, 0] = float(x)
        pos_cache[module_idx, 1] = float(y)
        # plc's density / congestion caches must invalidate; WL doesn't
        # matter because we compute it ourselves.
        self.plc.FLAG_UPDATE_DENSITY = True
        self.plc.FLAG_UPDATE_CONGESTION = True

    def score_swap(self, i_hard: int, new_i_xy, j_hard: int, new_j_xy) -> float:
        """Trial: compute proxy as if (i_hard, j_hard) were swapped, then revert.

        B3 phase 4: WL via per-net incremental (phase 2). Congestion via
        per-net subset routing (phase 4): subtract OLD touched-net + i,j
        macro contributions, apply set_pos, add NEW contributions, smooth +
        compute cost, then RESTORE the raw flats from snapshot. Density
        still via plc.get_density_cost (full recompute).
        """
        i_module = self.hard_indices[i_hard]
        j_module = self.hard_indices[j_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        j_slot = self._module_to_hard_slot.get(int(j_module))

        # Save committed positions for revert
        old_ix, old_iy = float(self.committed_hard_pos[i_hard, 0]), float(self.committed_hard_pos[i_hard, 1])
        old_jx, old_jy = float(self.committed_hard_pos[j_hard, 0]), float(self.committed_hard_pos[j_hard, 1])

        # Snapshot RAW routing flats for revert (small arrays ~20KB each).
        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

        touched = self._touched_nets(i_module, j_module)
        macro_subset = np.array(
            [s for s in (i_slot, j_slot) if s is not None], dtype=np.int64
        )

        # 1. Subtract OLD contributions (using current/committed positions).
        # The union net-set isn't per-module, so build the topology struct once
        # for this swap (idea 2) and reuse it for the −1/+1 applies.
        struct = _build_net_routing_struct(self.plc, touched)
        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # 2. Apply trial positions (pos_cache updated → next subset compute
        #    uses new positions).
        new_ix, new_iy = float(new_i_xy[0]), float(new_i_xy[1])
        new_jx, new_jy = float(new_j_xy[0]), float(new_j_xy[1])
        self._apply_pos(i_module, new_ix, new_iy)
        self._apply_pos(j_module, new_jx, new_jy)

        # 3. Add NEW contributions.
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # 3b. Refresh the smoothed-cost cache for the touched bbox; snapshot the
        #     affected columns/rows first so the trial can restore them on revert.
        r_lo, r_hi, c_lo, c_hi = self._union_bbox(bb_old, bb_new)
        if c_lo is not None:
            Hs_snap = self.H_smoothed[:, c_lo:c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo:r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        # 4. Incremental WL via touched nets (raw HPWL delta).
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer

        # 5. Compute congestion from our maintained flats (no plc call).
        cong = self._compute_cong_cost()

        # 6. Density incremental (P3): subtract i,j OLD footprints, add NEW,
        #    take top-10% over the grid, then revert the few touched cells.
        oi_idx, oi_area = self._macro_occ(i_module, old_ix, old_iy)
        oj_idx, oj_area = self._macro_occ(j_module, old_jx, old_jy)
        ni_idx, ni_area = self._macro_occ(i_module, new_ix, new_iy)
        nj_idx, nj_area = self._macro_occ(j_module, new_jx, new_jy)
        go = self.grid_occupied
        if oi_idx.size:
            np.subtract.at(go, oi_idx, oi_area)
        if oj_idx.size:
            np.subtract.at(go, oj_idx, oj_area)
        if ni_idx.size:
            np.add.at(go, ni_idx, ni_area)
        if nj_idx.size:
            np.add.at(go, nj_idx, nj_area)
        dens = self._compute_density_cost()

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        # 7. Revert: density cells, positions, raw routing flats.
        if ni_idx.size:
            np.subtract.at(go, ni_idx, ni_area)
        if nj_idx.size:
            np.subtract.at(go, nj_idx, nj_area)
        if oi_idx.size:
            np.add.at(go, oi_idx, oi_area)
        if oj_idx.size:
            np.add.at(go, oj_idx, oj_area)
        self._apply_pos(i_module, old_ix, old_iy)
        self._apply_pos(j_module, old_jx, old_jy)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        self.H_macro_flat[:] = Hm_snap
        self.V_macro_flat[:] = Vm_snap
        if c_lo is not None:
            self.H_smoothed[:, c_lo:c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo:r_hi + 1, :] = Vs_snap

        return score

    def commit_swap(self, i_hard: int, new_i_xy, j_hard: int, new_j_xy) -> None:
        """Commit a previously-trialed swap: persist positions, update
        per_net_hpwl AND routing flats (so subsequent score_swap calls see
        the new committed state).
        """
        i_module = self.hard_indices[i_hard]
        j_module = self.hard_indices[j_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        j_slot = self._module_to_hard_slot.get(int(j_module))

        # OLD committed positions (needed for the persistent density delta below,
        # read before committed_hard_pos is overwritten).
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        old_jx = float(self.committed_hard_pos[j_hard, 0])
        old_jy = float(self.committed_hard_pos[j_hard, 1])

        touched = self._touched_nets(i_module, j_module)
        macro_subset = np.array(
            [s for s in (i_slot, j_slot) if s is not None], dtype=np.int64
        )

        # Subtract OLD routing contributions. Build the topology struct once for
        # this swap (idea 2) and reuse it for the −1/+1 applies.
        struct = _build_net_routing_struct(self.plc, touched)
        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # Apply new positions.
        new_ix, new_iy = float(new_i_xy[0]), float(new_i_xy[1])
        new_jx, new_jy = float(new_j_xy[0]), float(new_j_xy[1])
        self._apply_pos(i_module, new_ix, new_iy)
        self._apply_pos(j_module, new_jx, new_jy)

        # P3: persist the density occupancy delta (subtract old footprints,
        # add new) so subsequent score_swap calls see the committed grid.
        go = self.grid_occupied
        oi_idx, oi_area = self._macro_occ(i_module, old_ix, old_iy)
        oj_idx, oj_area = self._macro_occ(j_module, old_jx, old_jy)
        ni_idx, ni_area = self._macro_occ(i_module, new_ix, new_iy)
        nj_idx, nj_area = self._macro_occ(j_module, new_jx, new_jy)
        if oi_idx.size:
            np.subtract.at(go, oi_idx, oi_area)
        if oj_idx.size:
            np.subtract.at(go, oj_idx, oj_area)
        if ni_idx.size:
            np.add.at(go, ni_idx, ni_area)
        if nj_idx.size:
            np.add.at(go, nj_idx, nj_area)

        # Add NEW routing contributions.
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        # Persist the smoothed-cost cache for the touched bbox.
        self._resmooth_bbox(*self._union_bbox(bb_old, bb_new))

        # Persist position state on the scorer.
        self.committed_hard_pos[i_hard, 0] = new_ix
        self.committed_hard_pos[i_hard, 1] = new_iy
        self.committed_hard_pos[j_hard, 0] = new_jx
        self.committed_hard_pos[j_hard, 1] = new_jy

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def _macro_nets(self, i_module: int) -> np.ndarray:
        a = self.macro_to_nets.get(i_module)
        return a if a is not None else np.empty(0, dtype=np.int64)

    def _route_struct(self, module_idx: int):
        """Cached topology routing-struct for one macro's touched nets (idea 2).
        Built once per module (placement-independent), reused across calls."""
        cache = self._route_struct_cache
        m = int(module_idx)
        if m not in cache:
            cache[m] = _build_net_routing_struct(self.plc, self._macro_nets(m))
        return cache[m]

    def soft_net_centroids(self) -> np.ndarray:
        """[num_soft, 2] WL-anchor per soft macro = mean over its pins of the
        centroid of that pin's net. Used by A3 (2026-05-29) candidate ordering in
        `_soft_relocation_moves`: targets are sorted by a blend of
        distance-to-current and distance-to-centroid, so moves toward the
        macro's connections are tried earlier and the WL gate sees friendlier
        candidates first. Computed from current committed soft positions; softs
        with no pins return their current pos."""
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
        """Trial: proxy as if hard macro i_hard RELOCATED to new_xy, then revert.

        Single-macro analogue of score_swap (relocation, not exchange) - used by
        the congestion-directed relocation pass. Only macro i's contributions
        change: WL over i's touched nets, congestion over those nets + i's macro
        routing slot, density over i's footprint cells.
        """
        i_module = self.hard_indices[i_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

        touched = self._macro_nets(i_module)
        struct = self._route_struct(i_module)
        macro_subset = (np.array([i_slot], dtype=np.int64)
                        if i_slot is not None else np.empty(0, dtype=np.int64))

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, -1.0,
                                        self.V_macro_flat, self.H_macro_flat)
        self._apply_pos(i_module, new_ix, new_iy)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, +1.0,
                                        self.V_macro_flat, self.H_macro_flat)

        # Refresh the smoothed-cost cache for the touched bbox (snapshot first so
        # the trial can restore it). Macro blockage is added live in the cost, so
        # only the net-routing bbox needs re-smoothing.
        r_lo, r_hi, c_lo, c_hi = self._union_bbox(bb_old, bb_new)
        if c_lo is not None:
            Hs_snap = self.H_smoothed[:, c_lo:c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo:r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
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

        # Revert: density cells, position, raw routing flats, smoothed cache.
        if n_idx.size:
            np.subtract.at(go, n_idx, n_area)
        if o_idx.size:
            np.add.at(go, o_idx, o_area)
        self._apply_pos(i_module, old_ix, old_iy)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        self.H_macro_flat[:] = Hm_snap
        self.V_macro_flat[:] = Vm_snap
        if c_lo is not None:
            self.H_smoothed[:, c_lo:c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo:r_hi + 1, :] = Vs_snap
        return score

    def commit_move(self, i_hard: int, new_xy) -> None:
        """Persist a relocation: update positions, routing flats, density grid,
        and per-net HPWL so subsequent score_* calls see the new state."""
        i_module = self.hard_indices[i_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])

        touched = self._macro_nets(i_module)
        struct = self._route_struct(i_module)
        macro_subset = (np.array([i_slot], dtype=np.int64)
                        if i_slot is not None else np.empty(0, dtype=np.int64))

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, -1.0,
                                        self.V_macro_flat, self.H_macro_flat)
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
            _apply_macro_routing_subset(self.plc, macro_subset, +1.0,
                                        self.V_macro_flat, self.H_macro_flat)

        # Persist the smoothed-cost cache for the touched bbox.
        self._resmooth_bbox(*self._union_bbox(bb_old, bb_new))

        self.committed_hard_pos[i_hard, 0] = new_ix
        self.committed_hard_pos[i_hard, 1] = new_iy
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def score_move_soft(self, soft_k: int, new_xy) -> float:
        """Trial: proxy as if SOFT macro `soft_k` (0..num_soft-1) relocated to
        new_xy, then revert. Softs contribute to WL + net-routing congestion +
        density, but NOT macro-routing blockage (only hard macros block) - so
        there is no macro_subset term. No legality constraint (softs may overlap).
        """
        s_module = self.soft_indices[soft_k]
        old_x = float(self.committed_soft_pos[soft_k, 0])
        old_y = float(self.committed_soft_pos[soft_k, 1])
        new_x, new_y = float(new_xy[0]), float(new_xy[1])

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        touched = self._macro_nets(s_module)
        struct = self._route_struct(s_module)

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        self._apply_pos(s_module, new_x, new_y)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)

        # Refresh the smoothed-cost cache for the touched bbox (snapshot first).
        r_lo, r_hi, c_lo, c_hi = self._union_bbox(bb_old, bb_new)
        if c_lo is not None:
            Hs_snap = self.H_smoothed[:, c_lo:c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo:r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
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

        # Revert: density cells, position, net-routing flats, smoothed cache.
        if n_idx.size:
            np.subtract.at(go, n_idx, n_area)
        if o_idx.size:
            np.add.at(go, o_idx, o_area)
        self._apply_pos(s_module, old_x, old_y)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        if c_lo is not None:
            self.H_smoothed[:, c_lo:c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo:r_hi + 1, :] = Vs_snap
        return score

    def commit_move_soft(self, soft_k: int, new_xy) -> None:
        """Persist a soft relocation (net routing + density + per-net HPWL)."""
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

        # Persist the smoothed-cost cache for the touched bbox.
        self._resmooth_bbox(*self._union_bbox(bb_old, bb_new))

        self.committed_soft_pos[soft_k, 0] = new_x
        self.committed_soft_pos[soft_k, 1] = new_y
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    # ------------------------------------------------------------------
    # S1 (2026-05-29): loop-invariant subtract-old for relocation passes.
    # Across a macro's n_targets candidate trials, the "subtract old routing
    # + subtract old density footprint + resmooth at old-bbox" step is
    # IDENTICAL. score_move/score_move_soft re-do it on every trial. The
    # prep/trial/commit/revert quartet hoists it out of the candidate loop:
    #   prep(i)              ── subtract old once     (1 routing-apply)
    #   for each target:
    #     trial_at(prep, t)  ── add new + score + revert add (1 routing-apply)
    #   if any target won:
    #     commit_after_prep(prep, best)    ── persist add-new + per-net HPWL
    #   else:
    #     revert_prep(prep)                ── re-add old (restore committed state)
    # Per-macro routing-apply calls: 1 + 2·n_targets (current) vs 2·n_targets
    # (current score_move) - wait, score_move's per-trial cost is 2 applies
    # (subtract-old + add-new). prep+trial saves 1 of those × (n_targets−1).
    # Bit-exact with score_move/commit_move (same float ops, different order).
    # ------------------------------------------------------------------

    def _prepare_move(self, i_hard: int) -> dict:
        """S1 prep for HARD relocation. Subtracts i's old routing + macro
        blockage + density once; subsequent _trial_at calls add at trial
        positions and revert via snapshots. Returns a context dict that
        _trial_at / _commit_after_prep / _revert_prep all consume."""
        i_module = self.hard_indices[i_hard]
        i_slot = self._module_to_hard_slot.get(int(i_module))
        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        struct = self._route_struct(i_module)
        macro_subset = (np.array([i_slot], dtype=np.int64)
                        if i_slot is not None else np.empty(0, dtype=np.int64))

        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, -1.0,
                                        self.V_macro_flat, self.H_macro_flat)
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
        """S1 trial. Adds i at new_xy on top of the 'i removed' base state from
        prep, computes proxy, reverts via snapshot. Bit-exact with score_move
        (same numerical ops; the prep's subtract-old is just hoisted out)."""
        i_module = prep["i_module"]
        struct = prep["struct"]
        macro_subset = prep["macro_subset"]
        old_ix, old_iy = prep["old_ix"], prep["old_iy"]
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

        self._apply_pos(i_module, new_ix, new_iy)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, +1.0,
                                        self.V_macro_flat, self.H_macro_flat)

        Hs_snap = Vs_snap = None
        r_lo = r_hi = c_lo = c_hi = None
        if bb_new is not None:
            r_lo, r_hi, c_lo, c_hi = bb_new
            Hs_snap = self.H_smoothed[:, c_lo:c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo:r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        touched = self._macro_nets(i_module)
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
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
            self.H_smoothed[:, c_lo:c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo:r_hi + 1, :] = Vs_snap
        self._apply_pos(i_module, old_ix, old_iy)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        self.H_macro_flat[:] = Hm_snap
        self.V_macro_flat[:] = Vm_snap

        return score

    def _commit_after_prep(self, prep: dict, new_xy) -> None:
        """S1 commit. Persists the winning trial on top of the prepared
        'i removed' state. Equivalent to (revert_prep + commit_move) but
        avoids the wasted revert."""
        i_module = prep["i_module"]
        i_hard = prep["i_hard"]
        struct = prep["struct"]
        macro_subset = prep["macro_subset"]
        new_ix, new_iy = float(new_xy[0]), float(new_xy[1])

        self._apply_pos(i_module, new_ix, new_iy)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, +1.0,
                                        self.V_macro_flat, self.H_macro_flat)

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
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def _revert_prep(self, prep: dict) -> None:
        """S1 revert (no candidate won). Re-applies i's old contributions at
        the OLD position to restore the committed state. Bit-exact inverse of
        prep - the +1 routing apply at OLD pin gcells exactly undoes prep's −1,
        and the density add-back exactly undoes the np.subtract.at."""
        struct = prep["struct"]
        macro_subset = prep["macro_subset"]

        bb_old = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(self.plc, macro_subset, +1.0,
                                        self.V_macro_flat, self.H_macro_flat)
        o_idx = prep["old_dens_idx"]
        o_area = prep["old_dens_area"]
        if o_idx.size:
            np.add.at(self.grid_occupied, o_idx, o_area)
        if bb_old is not None:
            self._resmooth_bbox(*bb_old)

    # --- soft analogues (no macro_subset / macro blockage; softs may overlap) ---

    def _prepare_move_soft(self, soft_k: int) -> dict:
        """S1 prep for SOFT relocation. Same as _prepare_move but without
        macro blockage (softs don't block routing) - only net routing +
        density + smoothed cache need 'k removed' updates."""
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
        """S1 trial for soft. Bit-exact with score_move_soft."""
        s_module = prep["s_module"]
        struct = prep["struct"]
        old_x, old_y = prep["old_x"], prep["old_y"]
        new_x, new_y = float(new_xy[0]), float(new_xy[1])

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()

        self._apply_pos(s_module, new_x, new_y)
        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)

        Hs_snap = Vs_snap = None
        r_lo = r_hi = c_lo = c_hi = None
        if bb_new is not None:
            r_lo, r_hi, c_lo, c_hi = bb_new
            Hs_snap = self.H_smoothed[:, c_lo:c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo:r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        touched = self._macro_nets(s_module)
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
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
            self.H_smoothed[:, c_lo:c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo:r_hi + 1, :] = Vs_snap
        self._apply_pos(s_module, old_x, old_y)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap

        return score

    def _commit_after_prep_soft(self, prep: dict, new_xy) -> None:
        """S1 commit for soft. Equivalent to (revert_prep_soft + commit_move_soft)
        but avoids the wasted revert."""
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
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def _revert_prep_soft(self, prep: dict) -> None:
        """S1 revert for soft. Bit-exact inverse of _prepare_move_soft."""
        struct = prep["struct"]
        bb_old = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        o_idx = prep["old_dens_idx"]
        o_area = prep["old_dens_area"]
        if o_idx.size:
            np.add.at(self.grid_occupied, o_idx, o_area)
        if bb_old is not None:
            self._resmooth_bbox(*bb_old)

    # ------------------------------------------------------------------
    # A1 (2026-05-29): soft-soft 2-opt swap. Pair-swap two softs' positions.
    # Single soft relocation can't find moves where two softs need to
    # EXCHANGE places - this pass adds that move type. Softs don't block
    # routing, so no macro_subset / macro-blockage handling (vs score_swap).
    # Softs may overlap, so no legality check on the swapped destinations.
    # Bit-exact analogue of score_swap; same accept-on-true-proxy guarantee.
    # ------------------------------------------------------------------

    def wl_delta_swap_soft(self, k1: int, new_xy1, k2: int, new_xy2) -> float:
        """Cheap WL-only delta for a hypothetical (k1, k2) soft swap.

        Speedup #30 (2026-05-30): used as a prefilter before the much more
        expensive `score_swap_soft`. Computes the per-net HPWL change for
        the touched nets WITHOUT touching routing or density state.
        Bypasses `_apply_pos` (which dirties plc flags) - instead transiently
        overwrites `_global_pos_cache` rows for k1, k2 then restores them.
        Returns the NORMALIZED WL delta (matches proxy's WL term scale).
        """
        s_mod1 = self.soft_indices[k1]
        s_mod2 = self.soft_indices[k2]
        touched = self._touched_nets(s_mod1, s_mod2)
        if len(touched) == 0:
            return 0.0
        pos_cache = _ensure_pos_cache(self.plc)
        sx1 = float(pos_cache[s_mod1, 0]); sy1 = float(pos_cache[s_mod1, 1])
        sx2 = float(pos_cache[s_mod2, 0]); sy2 = float(pos_cache[s_mod2, 1])
        pos_cache[s_mod1, 0] = float(new_xy1[0]); pos_cache[s_mod1, 1] = float(new_xy1[1])
        pos_cache[s_mod2, 0] = float(new_xy2[0]); pos_cache[s_mod2, 1] = float(new_xy2[1])
        new_per_net = self._compute_per_net_hpwl_subset(touched)
        delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
        pos_cache[s_mod1, 0] = sx1; pos_cache[s_mod1, 1] = sy1
        pos_cache[s_mod2, 0] = sx2; pos_cache[s_mod2, 1] = sy2
        return delta / self.wl_normalizer

    def score_swap_soft(self, k1: int, new_xy1, k2: int, new_xy2) -> float:
        """Trial: proxy as if (k1, k2) SOFT macros were swapped to new_xy1 /
        new_xy2 respectively, then revert. Bit-exact with the full scorer."""
        s_mod1 = self.soft_indices[k1]
        s_mod2 = self.soft_indices[k2]

        old_x1 = float(self.committed_soft_pos[k1, 0])
        old_y1 = float(self.committed_soft_pos[k1, 1])
        old_x2 = float(self.committed_soft_pos[k2, 0])
        old_y2 = float(self.committed_soft_pos[k2, 1])

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()

        touched = self._touched_nets(s_mod1, s_mod2)
        struct = _build_net_routing_struct(self.plc, touched)
        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)

        new_x1, new_y1 = float(new_xy1[0]), float(new_xy1[1])
        new_x2, new_y2 = float(new_xy2[0]), float(new_xy2[1])
        self._apply_pos(s_mod1, new_x1, new_y1)
        self._apply_pos(s_mod2, new_x2, new_y2)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)

        r_lo, r_hi, c_lo, c_hi = self._union_bbox(bb_old, bb_new)
        if c_lo is not None:
            Hs_snap = self.H_smoothed[:, c_lo:c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo:r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer
        cong = self._compute_cong_cost()

        o1_idx, o1_area = self._macro_occ(s_mod1, old_x1, old_y1)
        o2_idx, o2_area = self._macro_occ(s_mod2, old_x2, old_y2)
        n1_idx, n1_area = self._macro_occ(s_mod1, new_x1, new_y1)
        n2_idx, n2_area = self._macro_occ(s_mod2, new_x2, new_y2)
        go = self.grid_occupied
        if o1_idx.size:
            np.subtract.at(go, o1_idx, o1_area)
        if o2_idx.size:
            np.subtract.at(go, o2_idx, o2_area)
        if n1_idx.size:
            np.add.at(go, n1_idx, n1_area)
        if n2_idx.size:
            np.add.at(go, n2_idx, n2_area)
        dens = self._compute_density_cost()

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        # Revert
        if n1_idx.size:
            np.subtract.at(go, n1_idx, n1_area)
        if n2_idx.size:
            np.subtract.at(go, n2_idx, n2_area)
        if o1_idx.size:
            np.add.at(go, o1_idx, o1_area)
        if o2_idx.size:
            np.add.at(go, o2_idx, o2_area)
        self._apply_pos(s_mod1, old_x1, old_y1)
        self._apply_pos(s_mod2, old_x2, old_y2)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        if c_lo is not None:
            self.H_smoothed[:, c_lo:c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo:r_hi + 1, :] = Vs_snap

        return score

    def commit_swap_soft(self, k1: int, new_xy1, k2: int, new_xy2) -> None:
        """Persist a soft-soft swap: positions, routing flats, density grid,
        per-net HPWL, smoothed cache. Analogue of commit_swap for softs."""
        s_mod1 = self.soft_indices[k1]
        s_mod2 = self.soft_indices[k2]
        old_x1 = float(self.committed_soft_pos[k1, 0])
        old_y1 = float(self.committed_soft_pos[k1, 1])
        old_x2 = float(self.committed_soft_pos[k2, 0])
        old_y2 = float(self.committed_soft_pos[k2, 1])

        touched = self._touched_nets(s_mod1, s_mod2)
        struct = _build_net_routing_struct(self.plc, touched)
        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)

        new_x1, new_y1 = float(new_xy1[0]), float(new_xy1[1])
        new_x2, new_y2 = float(new_xy2[0]), float(new_xy2[1])
        self._apply_pos(s_mod1, new_x1, new_y1)
        self._apply_pos(s_mod2, new_x2, new_y2)

        go = self.grid_occupied
        o1_idx, o1_area = self._macro_occ(s_mod1, old_x1, old_y1)
        o2_idx, o2_area = self._macro_occ(s_mod2, old_x2, old_y2)
        n1_idx, n1_area = self._macro_occ(s_mod1, new_x1, new_y1)
        n2_idx, n2_area = self._macro_occ(s_mod2, new_x2, new_y2)
        if o1_idx.size:
            np.subtract.at(go, o1_idx, o1_area)
        if o2_idx.size:
            np.subtract.at(go, o2_idx, o2_area)
        if n1_idx.size:
            np.add.at(go, n1_idx, n1_area)
        if n2_idx.size:
            np.add.at(go, n2_idx, n2_area)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        self._resmooth_bbox(*self._union_bbox(bb_old, bb_new))

        self.committed_soft_pos[k1, 0] = new_x1
        self.committed_soft_pos[k1, 1] = new_y1
        self.committed_soft_pos[k2, 0] = new_x2
        self.committed_soft_pos[k2, 1] = new_y2
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def score_swap_hard_soft(self, i_hard: int, new_hard_xy, k_soft: int, new_soft_xy) -> float:
        """HXS (2026-05-30): trial proxy as if (i_hard, k_soft) swapped places,
        then revert. Hybrid of `score_swap` (hard→macro_subset for the routing
        blockage) and `score_swap_soft` (no macro_subset for the soft). Both
        contribute to nets touched (HPWL + per-net routing) and density. Legality
        is the caller's responsibility (the hard's destination must not overlap
        with other hard macros)."""
        i_module = self.hard_indices[i_hard]
        s_module = self.soft_indices[k_soft]
        i_slot = self._module_to_hard_slot.get(int(i_module))

        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        old_sx = float(self.committed_soft_pos[k_soft, 0])
        old_sy = float(self.committed_soft_pos[k_soft, 1])

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

        touched = self._touched_nets(i_module, s_module)
        macro_subset = (np.array([i_slot], dtype=np.int64)
                        if i_slot is not None else np.empty(0, dtype=np.int64))

        struct = _build_net_routing_struct(self.plc, touched)
        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        new_ix, new_iy = float(new_hard_xy[0]), float(new_hard_xy[1])
        new_sx, new_sy = float(new_soft_xy[0]), float(new_soft_xy[1])
        self._apply_pos(i_module, new_ix, new_iy)
        self._apply_pos(s_module, new_sx, new_sy)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        r_lo, r_hi, c_lo, c_hi = self._union_bbox(bb_old, bb_new)
        if c_lo is not None:
            Hs_snap = self.H_smoothed[:, c_lo:c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo:r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer
        cong = self._compute_cong_cost()

        oi_idx, oi_area = self._macro_occ(i_module, old_ix, old_iy)
        os_idx, os_area = self._macro_occ(s_module, old_sx, old_sy)
        ni_idx, ni_area = self._macro_occ(i_module, new_ix, new_iy)
        ns_idx, ns_area = self._macro_occ(s_module, new_sx, new_sy)
        go = self.grid_occupied
        if oi_idx.size:
            np.subtract.at(go, oi_idx, oi_area)
        if os_idx.size:
            np.subtract.at(go, os_idx, os_area)
        if ni_idx.size:
            np.add.at(go, ni_idx, ni_area)
        if ns_idx.size:
            np.add.at(go, ns_idx, ns_area)
        dens = self._compute_density_cost()

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        # Revert.
        if ni_idx.size:
            np.subtract.at(go, ni_idx, ni_area)
        if ns_idx.size:
            np.subtract.at(go, ns_idx, ns_area)
        if oi_idx.size:
            np.add.at(go, oi_idx, oi_area)
        if os_idx.size:
            np.add.at(go, os_idx, os_area)
        self._apply_pos(i_module, old_ix, old_iy)
        self._apply_pos(s_module, old_sx, old_sy)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        self.H_macro_flat[:] = Hm_snap
        self.V_macro_flat[:] = Vm_snap
        if c_lo is not None:
            self.H_smoothed[:, c_lo:c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo:r_hi + 1, :] = Vs_snap

        return score

    def commit_swap_hard_soft(self, i_hard: int, new_hard_xy, k_soft: int, new_soft_xy) -> None:
        """Persist an HXS swap. Analogue of commit_swap + commit_swap_soft."""
        i_module = self.hard_indices[i_hard]
        s_module = self.soft_indices[k_soft]
        i_slot = self._module_to_hard_slot.get(int(i_module))

        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        old_sx = float(self.committed_soft_pos[k_soft, 0])
        old_sy = float(self.committed_soft_pos[k_soft, 1])

        touched = self._touched_nets(i_module, s_module)
        macro_subset = (np.array([i_slot], dtype=np.int64)
                        if i_slot is not None else np.empty(0, dtype=np.int64))

        struct = _build_net_routing_struct(self.plc, touched)
        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        new_ix, new_iy = float(new_hard_xy[0]), float(new_hard_xy[1])
        new_sx, new_sy = float(new_soft_xy[0]), float(new_soft_xy[1])
        self._apply_pos(i_module, new_ix, new_iy)
        self._apply_pos(s_module, new_sx, new_sy)

        go = self.grid_occupied
        oi_idx, oi_area = self._macro_occ(i_module, old_ix, old_iy)
        os_idx, os_area = self._macro_occ(s_module, old_sx, old_sy)
        ni_idx, ni_area = self._macro_occ(i_module, new_ix, new_iy)
        ns_idx, ns_area = self._macro_occ(s_module, new_sx, new_sy)
        if oi_idx.size:
            np.subtract.at(go, oi_idx, oi_area)
        if os_idx.size:
            np.subtract.at(go, os_idx, os_area)
        if ni_idx.size:
            np.add.at(go, ni_idx, ni_area)
        if ns_idx.size:
            np.add.at(go, ns_idx, ns_area)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        self._resmooth_bbox(*self._union_bbox(bb_old, bb_new))

        self.committed_hard_pos[i_hard, 0] = new_ix
        self.committed_hard_pos[i_hard, 1] = new_iy
        self.committed_soft_pos[k_soft, 0] = new_sx
        self.committed_soft_pos[k_soft, 1] = new_sy
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta

    def score_cycle_hard_soft_soft(self, i_hard: int, new_hard_xy,
                                     k1_soft: int, new_k1_xy,
                                     k2_soft: int, new_k2_xy) -> float:
        """HS3 (2026-05-31): trial proxy for a 3-cycle rotation
        H → S1 → S2 → H, then revert. Hybrid of score_swap_hard_soft
        extended to 3 modules: hard contributes routing blockage via
        macro_subset; both softs contribute only via net routing + density.
        The caller is responsible for legality of the hard's new position
        (no overlap with other hards). Bit-exact with full _exact_proxy
        (verified ≤4.4e-16)."""
        i_module = self.hard_indices[i_hard]
        s_mod1 = self.soft_indices[k1_soft]
        s_mod2 = self.soft_indices[k2_soft]
        i_slot = self._module_to_hard_slot.get(int(i_module))

        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        old_s1x = float(self.committed_soft_pos[k1_soft, 0])
        old_s1y = float(self.committed_soft_pos[k1_soft, 1])
        old_s2x = float(self.committed_soft_pos[k2_soft, 0])
        old_s2y = float(self.committed_soft_pos[k2_soft, 1])

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

        touched = self._touched_nets3(i_module, s_mod1, s_mod2)
        macro_subset = (np.array([i_slot], dtype=np.int64)
                        if i_slot is not None else np.empty(0, dtype=np.int64))

        struct = _build_net_routing_struct(self.plc, touched)
        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        new_ix, new_iy = float(new_hard_xy[0]), float(new_hard_xy[1])
        new_s1x, new_s1y = float(new_k1_xy[0]), float(new_k1_xy[1])
        new_s2x, new_s2y = float(new_k2_xy[0]), float(new_k2_xy[1])
        self._apply_pos(i_module, new_ix, new_iy)
        self._apply_pos(s_mod1, new_s1x, new_s1y)
        self._apply_pos(s_mod2, new_s2x, new_s2y)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        r_lo, r_hi, c_lo, c_hi = self._union_bbox(bb_old, bb_new)
        if c_lo is not None:
            Hs_snap = self.H_smoothed[:, c_lo:c_hi + 1].copy()
            Vs_snap = self.V_smoothed[r_lo:r_hi + 1, :].copy()
            self._resmooth_bbox(r_lo, r_hi, c_lo, c_hi)

        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            new_total_raw = self.total_wl_raw + delta
        else:
            new_total_raw = self.total_wl_raw
        new_wl_normalized = new_total_raw / self.wl_normalizer
        cong = self._compute_cong_cost()

        # Density: subtract 3 OLD footprints, add 3 NEW, compute, revert.
        oi_idx, oi_area = self._macro_occ(i_module, old_ix, old_iy)
        o1_idx, o1_area = self._macro_occ(s_mod1, old_s1x, old_s1y)
        o2_idx, o2_area = self._macro_occ(s_mod2, old_s2x, old_s2y)
        ni_idx, ni_area = self._macro_occ(i_module, new_ix, new_iy)
        n1_idx, n1_area = self._macro_occ(s_mod1, new_s1x, new_s1y)
        n2_idx, n2_area = self._macro_occ(s_mod2, new_s2x, new_s2y)
        go = self.grid_occupied
        if oi_idx.size: np.subtract.at(go, oi_idx, oi_area)
        if o1_idx.size: np.subtract.at(go, o1_idx, o1_area)
        if o2_idx.size: np.subtract.at(go, o2_idx, o2_area)
        if ni_idx.size: np.add.at(go, ni_idx, ni_area)
        if n1_idx.size: np.add.at(go, n1_idx, n1_area)
        if n2_idx.size: np.add.at(go, n2_idx, n2_area)
        dens = self._compute_density_cost()

        score = float(new_wl_normalized + 0.5 * dens + 0.5 * cong)

        # Revert.
        if ni_idx.size: np.subtract.at(go, ni_idx, ni_area)
        if n1_idx.size: np.subtract.at(go, n1_idx, n1_area)
        if n2_idx.size: np.subtract.at(go, n2_idx, n2_area)
        if oi_idx.size: np.add.at(go, oi_idx, oi_area)
        if o1_idx.size: np.add.at(go, o1_idx, o1_area)
        if o2_idx.size: np.add.at(go, o2_idx, o2_area)
        self._apply_pos(i_module, old_ix, old_iy)
        self._apply_pos(s_mod1, old_s1x, old_s1y)
        self._apply_pos(s_mod2, old_s2x, old_s2y)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        self.H_macro_flat[:] = Hm_snap
        self.V_macro_flat[:] = Vm_snap
        if c_lo is not None:
            self.H_smoothed[:, c_lo:c_hi + 1] = Hs_snap
            self.V_smoothed[r_lo:r_hi + 1, :] = Vs_snap

        return score

    def commit_cycle_hard_soft_soft(self, i_hard: int, new_hard_xy,
                                      k1_soft: int, new_k1_xy,
                                      k2_soft: int, new_k2_xy) -> None:
        """Persist an HS3 3-cycle. Analogue of commit_swap_hard_soft for 3 modules."""
        i_module = self.hard_indices[i_hard]
        s_mod1 = self.soft_indices[k1_soft]
        s_mod2 = self.soft_indices[k2_soft]
        i_slot = self._module_to_hard_slot.get(int(i_module))

        old_ix = float(self.committed_hard_pos[i_hard, 0])
        old_iy = float(self.committed_hard_pos[i_hard, 1])
        old_s1x = float(self.committed_soft_pos[k1_soft, 0])
        old_s1y = float(self.committed_soft_pos[k1_soft, 1])
        old_s2x = float(self.committed_soft_pos[k2_soft, 0])
        old_s2y = float(self.committed_soft_pos[k2_soft, 1])

        touched = self._touched_nets3(i_module, s_mod1, s_mod2)
        macro_subset = (np.array([i_slot], dtype=np.int64)
                        if i_slot is not None else np.empty(0, dtype=np.int64))

        struct = _build_net_routing_struct(self.plc, touched)
        bb_old = _apply_net_routing_struct(self.plc, struct, -1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, -1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        new_ix, new_iy = float(new_hard_xy[0]), float(new_hard_xy[1])
        new_s1x, new_s1y = float(new_k1_xy[0]), float(new_k1_xy[1])
        new_s2x, new_s2y = float(new_k2_xy[0]), float(new_k2_xy[1])
        self._apply_pos(i_module, new_ix, new_iy)
        self._apply_pos(s_mod1, new_s1x, new_s1y)
        self._apply_pos(s_mod2, new_s2x, new_s2y)

        go = self.grid_occupied
        oi_idx, oi_area = self._macro_occ(i_module, old_ix, old_iy)
        o1_idx, o1_area = self._macro_occ(s_mod1, old_s1x, old_s1y)
        o2_idx, o2_area = self._macro_occ(s_mod2, old_s2x, old_s2y)
        ni_idx, ni_area = self._macro_occ(i_module, new_ix, new_iy)
        n1_idx, n1_area = self._macro_occ(s_mod1, new_s1x, new_s1y)
        n2_idx, n2_area = self._macro_occ(s_mod2, new_s2x, new_s2y)
        if oi_idx.size: np.subtract.at(go, oi_idx, oi_area)
        if o1_idx.size: np.subtract.at(go, o1_idx, o1_area)
        if o2_idx.size: np.subtract.at(go, o2_idx, o2_area)
        if ni_idx.size: np.add.at(go, ni_idx, ni_area)
        if n1_idx.size: np.add.at(go, n1_idx, n1_area)
        if n2_idx.size: np.add.at(go, n2_idx, n2_area)

        bb_new = _apply_net_routing_struct(self.plc, struct, +1.0, self.H_flat, self.V_flat)
        if macro_subset.size > 0:
            _apply_macro_routing_subset(
                self.plc, macro_subset, +1.0,
                self.V_macro_flat, self.H_macro_flat,
            )

        self._resmooth_bbox(*self._union_bbox(bb_old, bb_new))

        self.committed_hard_pos[i_hard, 0] = new_ix
        self.committed_hard_pos[i_hard, 1] = new_iy
        self.committed_soft_pos[k1_soft, 0] = new_s1x
        self.committed_soft_pos[k1_soft, 1] = new_s1y
        self.committed_soft_pos[k2_soft, 0] = new_s2x
        self.committed_soft_pos[k2_soft, 1] = new_s2y
        if len(touched) > 0:
            new_per_net = self._compute_per_net_hpwl_subset(touched)
            delta = float(np.sum((new_per_net - self.per_net_hpwl[touched]) * self.net_weights[touched]))
            self.per_net_hpwl[touched] = new_per_net
            self.total_wl_raw += delta
