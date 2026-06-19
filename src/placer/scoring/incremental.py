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
        sub = H2d[:, c_lo : c_hi + 1] / self.grid_h_routes
        if self.smooth_range <= 0:
            self.H_smoothed[:, c_lo : c_hi + 1] = sub
            return
        weighted = sub / self._sm_row_cnt[:, None]
        ncols = sub.shape[1]
        cs = np.empty((self.grid_row + 1, ncols), dtype=np.float64)
        cs[0, :] = 0.0
        np.cumsum(weighted, axis=0, out=cs[1:, :])
        self.H_smoothed[:, c_lo : c_hi + 1] = cs[self._sm_row_up + 1] - cs[self._sm_row_lp]

    def _resmooth_v_rows(self, r_lo: int, r_hi: int) -> None:
        """Re-smooth affected V rows from raw routing."""
        V2d = self.V_flat.reshape(self.grid_row, self.grid_col)
        sub = V2d[r_lo : r_hi + 1, :] / self.grid_v_routes
        if self.smooth_range <= 0:
            self.V_smoothed[r_lo : r_hi + 1, :] = sub
            return
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
        """Return nets touched by three modules."""
        a = self.macro_to_nets.get(m1)
        b = self.macro_to_nets.get(m2)
        c = self.macro_to_nets.get(m3)
        parts = [x for x in (a, b, c) if x is not None]
        if not parts:
            return np.empty(0, dtype=np.int64)
        if len(parts) == 1:
            return parts[0]
        return np.unique(np.concatenate(parts))

    def _touched_nets_many(self, modules) -> np.ndarray:
        """Return the union of nets touched by the given modules."""
        parts = [self._macro_nets(int(m)) for m in modules]
        parts = [p for p in parts if p.size]
        if not parts:
            return np.empty(0, dtype=np.int64)
        if len(parts) == 1:
            return parts[0]
        return np.unique(np.concatenate(parts))

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

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

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
        self._apply_pos(i_module, old_ix, old_iy)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
        self.H_macro_flat[:] = Hm_snap
        self.V_macro_flat[:] = Vm_snap
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

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        touched = self._macro_nets(s_module)
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
        self._apply_pos(s_module, old_x, old_y)
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
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

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

        touched = self._touched_nets_many(modules)
        bb_old = _apply_net_routing_subset(self.plc, touched, -1.0, self.H_flat, self.V_flat)
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

        bb_new = _apply_net_routing_subset(self.plc, touched, +1.0, self.H_flat, self.V_flat)
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
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
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

        bb_old = _apply_net_routing_subset(self.plc, touched, -1.0, self.H_flat, self.V_flat)
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

        bb_new = _apply_net_routing_subset(self.plc, touched, +1.0, self.H_flat, self.V_flat)
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

    def score_swap_soft_soft_many(self, soft_a: int, candidates: np.ndarray) -> np.ndarray:
        """Score soft-soft swaps for one soft macro, preserving candidate order."""
        cand = np.asarray(candidates, dtype=np.int64).reshape(-1)
        out = np.empty(cand.size, dtype=np.float64)
        for k, soft_b in enumerate(cand):
            out[k] = self.score_swap_soft_soft(soft_a, int(soft_b))
        return out

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

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()
        Hm_snap = self.H_macro_flat.copy()
        Vm_snap = self.V_macro_flat.copy()

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
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap
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

        H_snap = self.H_flat.copy()
        V_snap = self.V_flat.copy()

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
        self.H_flat[:] = H_snap
        self.V_flat[:] = V_snap

        return score

    def _trial_many_at_soft(self, prep: dict, xy_array: np.ndarray) -> np.ndarray:
        """Score several soft targets after `_prepare_move_soft`, preserving order."""
        xy = np.asarray(xy_array, dtype=np.float64).reshape(-1, 2)
        out = np.empty(xy.shape[0], dtype=np.float64)
        for k in range(xy.shape[0]):
            out[k] = self._trial_at_soft(prep, xy[k])
        return out

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
