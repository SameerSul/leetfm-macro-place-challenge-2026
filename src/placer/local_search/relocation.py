"""Relocation moves for local search."""

import os
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from utils import constants as const
from utils.config import HAS_NUMBA, _numba_njit

from placer.shared.geometry import separation_matrices
from placer.local_search.fields import _congestion_field, _density_field, weighted_congestion_field
from placer.local_search.region_rules import accepts_region_score, point_in_region

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark


def _structural_weights() -> tuple[float, float, float]:
    return (
        float(const.HIER_KEEP_OUT_WEIGHT),
        float(const.HIER_GRID_ALIGN_WEIGHT),
        float(const.HIER_NOTCH_WEIGHT),
    )


def _hierarchy_structural_weight() -> float:
    """Weight for BeyondPPA-style structure inside hierarchy candidate ordering."""
    return max(0.0, float(const.HIER_OBJECTIVE_STRUCTURAL_WEIGHT))


def _full_committed_pos(incremental_scorer) -> np.ndarray:
    return np.vstack(
        [
            incremental_scorer.committed_hard_pos,
            incremental_scorer.committed_soft_pos,
        ]
    ).astype(np.float64, copy=True)


def _full_macro_sizes(incremental_scorer) -> np.ndarray:
    return incremental_scorer.benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)


def _dedupe_targets_xy(targets) -> np.ndarray:
    arr = np.asarray(targets, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 1:
        if arr.size == 0:
            return arr.reshape(0, 2)
        arr = arr.reshape(-1, 2)
    if arr.shape[1] != 2:
        raise ValueError("target coordinates must be [n,2]")
    _, keep = np.unique(arr, axis=0, return_index=True)
    return arr[np.sort(keep)]


def _structural_local_penalty(
    full_pos: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    benchmark: "Benchmark",
    module_index: int,
) -> float:
    pos = np.asarray(full_pos, dtype=np.float64)
    i = int(module_index)
    values = _structural_local_penalty_batch(
        pos,
        np.asarray(sizes, dtype=np.float64),
        cw,
        ch,
        benchmark,
        module_index=i,
        target_x=np.array([float(pos[i, 0])]),
        target_y=np.array([float(pos[i, 1])]),
    )
    if values.size == 0:
        return 0.0
    return float(values[0])


def _structural_local_penalty_batch(
    full_pos: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    benchmark: "Benchmark",
    *,
    module_index: int,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    kw, gw, nw = _structural_weights()
    pos = np.asarray(full_pos, dtype=np.float64)
    sz = np.asarray(sizes, dtype=np.float64)
    i = int(module_index)
    tx = np.asarray(target_x, dtype=np.float64).reshape(-1)
    ty = np.asarray(target_y, dtype=np.float64).reshape(-1)
    if tx.size == 0 or ty.size == 0:
        return np.zeros(0, dtype=np.float64)
    if tx.shape != ty.shape:
        raise ValueError("target_x and target_y must be the same shape")

    half = sz / 2.0
    hw_i = float(half[i, 0])
    hh_i = float(half[i, 1])
    keepout = max(
        min(float(np.median(np.minimum(sz[:, 0], sz[:, 1]))), min(cw, ch) * 0.02),
        1e-9,
    )
    clear = np.minimum.reduce(
        [
            tx - hw_i,
            cw - (tx + hw_i),
            ty - hh_i,
            ch - (ty + hh_i),
        ]
    )
    edge = np.maximum((keepout - clear) / keepout, 0.0) ** 2

    px = cw / max(int(benchmark.grid_cols), 1)
    py = ch / max(int(benchmark.grid_rows), 1)
    gx = (np.floor(tx / px) + 0.5) * px
    gy = (np.floor(ty / py) + 0.5) * py
    dx = np.minimum(np.abs(tx - gx) / max(0.5 * px, 1e-9), 1.0)
    dy = np.minimum(np.abs(ty - gy) / max(0.5 * py, 1e-9), 1.0)
    grid = 0.5 * (dx * dx + dy * dy)

    left = pos[:, 0] - half[:, 0]
    right = pos[:, 0] + half[:, 0]
    bottom = pos[:, 1] - half[:, 1]
    top = pos[:, 1] + half[:, 1]
    left_t = tx - hw_i
    right_t = tx + hw_i
    bottom_t = ty - hh_i
    top_t = ty + hh_i
    x_gap = np.maximum.reduce(
        [
            left[None, :] - right_t[:, None],
            left_t[:, None] - right[None, :],
            np.zeros((tx.size, pos.shape[0]), dtype=np.float64),
        ]
    )
    y_gap = np.maximum.reduce(
        [
            bottom[None, :] - top_t[:, None],
            bottom_t[:, None] - top[None, :],
            np.zeros((tx.size, pos.shape[0]), dtype=np.float64),
        ]
    )
    x_overlap = np.minimum(right_t[:, None], right[None, :]) - np.maximum(
        left_t[:, None], left[None, :]
    )
    y_overlap = np.minimum(top_t[:, None], top[None, :]) - np.maximum(
        bottom_t[:, None], bottom[None, :]
    )
    valid_x = (x_gap > 0.0) & (x_gap < keepout) & (y_overlap > 0.0)
    valid_y = (y_gap > 0.0) & (y_gap < keepout) & (x_overlap > 0.0)
    if i < pos.shape[0]:
        valid_x[:, i] = False
        valid_y[:, i] = False
    terms = np.zeros((tx.size, pos.shape[0]), dtype=np.float64)
    if valid_x.any():
        cover = np.minimum(1.0, y_overlap[valid_x] / max(float(sz[i, 1]), 1e-9))
        terms[valid_x] = ((keepout - x_gap[valid_x]) / keepout) ** 2 * cover
    if valid_y.any():
        cover = np.minimum(1.0, x_overlap[valid_y] / max(float(sz[i, 0]), 1e-9))
        terms[valid_y] = ((keepout - y_gap[valid_y]) / keepout) ** 2 * cover
    term_sum = np.sum(terms, axis=1)
    term_count = np.count_nonzero(terms, axis=1)
    notch = np.divide(term_sum, term_count, out=np.zeros_like(term_sum), where=term_count > 0)
    return kw * edge + gw * grid + nw * notch


def _structural_candidate_order(
    *,
    cand: np.ndarray,
    base_rank: np.ndarray,
    module_index: int,
    target_x: np.ndarray,
    target_y: np.ndarray,
    full_pos: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    benchmark: "Benchmark",
) -> np.ndarray:
    structural_weight = _hierarchy_structural_weight()
    if cand.size == 0 or structural_weight <= 0.0:
        return np.argsort(base_rank)
    base_struct = _structural_local_penalty(full_pos, sizes, cw, ch, benchmark, module_index)
    span2 = max(float(max(cw, ch)) ** 2, 1.0)
    adjusted = np.asarray(base_rank, dtype=np.float64).copy()
    local_scores = _structural_local_penalty_batch(
        full_pos,
        sizes,
        cw,
        ch,
        benchmark,
        module_index=module_index,
        target_x=target_x[cand],
        target_y=target_y[cand],
    )
    adjusted += structural_weight * span2 * (local_scores - base_struct)
    return np.argsort(adjusted, kind="stable")


def _apply_structural_proposal_scores(
    proposals: list[dict],
    *,
    full_pos: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    benchmark: "Benchmark",
    n_hard: int,
    soft: bool = False,
) -> None:
    structural_weight = _hierarchy_structural_weight()
    if not proposals or structural_weight <= 0.0:
        return
    proposals_by_module: dict[int, list[tuple[int, float, float]]] = {}
    for p_i, proposal in enumerate(proposals):
        idx = int(proposal["i"])
        module_index = n_hard + idx if soft else idx
        proposals_by_module.setdefault(module_index, []).append(
            (p_i, float(proposal["xy"][0]), float(proposal["xy"][1]))
        )

    for module_index, entries in proposals_by_module.items():
        base_pos = _structural_local_penalty_batch(
            full_pos,
            sizes,
            cw,
            ch,
            benchmark,
            module_index=module_index,
            target_x=np.array([float(full_pos[module_index, 0])]),
            target_y=np.array([float(full_pos[module_index, 1])]),
        )
        base_struct = float(base_pos[0]) if base_pos.size else 0.0
        idxs = np.array([e[0] for e in entries], dtype=np.int64)
        tx = np.array([e[1] for e in entries], dtype=np.float64)
        ty = np.array([e[2] for e in entries], dtype=np.float64)
        if tx.size == 0:
            continue
        local_scores = _structural_local_penalty_batch(
            full_pos,
            sizes,
            cw,
            ch,
            benchmark,
            module_index=module_index,
            target_x=tx,
            target_y=ty,
        )
        deltas = local_scores - base_struct
        for k, p_i in enumerate(idxs):
            proposals[p_i]["structural_delta"] = float(deltas[k])
            proposals[p_i]["score"] = float(proposals[p_i]["score"]) + structural_weight * float(
                deltas[k]
            )


if HAS_NUMBA:

    @_numba_njit(cache=True, fastmath=False)
    def _legal_candidate_mask(
        cand_x: np.ndarray,
        cand_y: np.ndarray,
        blocked_x: np.ndarray,
        blocked_y: np.ndarray,
        blocked_sx: np.ndarray,
        blocked_sy: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Numba-accelerated legality mask for candidate points."""
        n = cand_x.shape[0]
        m = blocked_x.shape[0]
        out = np.ones(n, dtype=np.bool_)
        for i in range(n):
            cx = cand_x[i]
            cy = cand_y[i]
            ok = True
            for j in range(m):
                dx = cx - blocked_x[j]
                if dx < 0.0:
                    dx = -dx
                if dx < (blocked_sx[j] + eps):
                    dy = cy - blocked_y[j]
                    if dy < 0.0:
                        dy = -dy
                    if dy < (blocked_sy[j] + eps):
                        ok = False
                        break
            out[i] = ok
        return out

else:

    def _legal_candidate_mask(
        cand_x: np.ndarray,
        cand_y: np.ndarray,
        blocked_x: np.ndarray,
        blocked_y: np.ndarray,
        blocked_sx: np.ndarray,
        blocked_sy: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Fallback overlap test when numba is unavailable."""
        if cand_x.size == 0:
            return np.ones(0, dtype=np.bool_)
        if blocked_x.size == 0:
            return np.ones(cand_x.size, dtype=np.bool_)
        overlap = (np.abs(cand_x[:, None] - blocked_x[None, :]) < (blocked_sx[None, :] + eps)) & (
            np.abs(cand_y[:, None] - blocked_y[None, :]) < (blocked_sy[None, :] + eps)
        )
        return ~np.any(overlap, axis=1)


def _region_penalty(d2, tgt_x, tgt_y, cand, region_bbox, i, span2, region_bias):
    """Push out-of-region candidate cells to the back of the distance ranking.

    Soft region lock: in-region cells fill `n_targets` first; out-of-region cells
    are reached only when in-region options run out. Returns the penalized d2.
    """
    rb = region_bbox[i]
    out = (
        (tgt_x[cand] < rb[0])
        | (tgt_x[cand] > rb[2])
        | (tgt_y[cand] < rb[1])
        | (tgt_y[cand] > rb[3])
    )
    return d2 + region_bias * span2 * out


def _hierarchy_aware_target_filter(
    cand: np.ndarray,
    *,
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_field: np.ndarray,
    source_field: float,
    region_bbox,
    region_index: int,
    region_mask,
    cw: float,
    ch: float,
    field_span: float,
    enabled: bool,
) -> np.ndarray:
    """Prefer in-region targets unless an outside target has much better relief."""
    if not enabled or region_bbox is None or cand.size <= 1:
        return cand
    rb = region_bbox[int(region_index)]
    inside = (
        (target_x[cand] >= rb[0])
        & (target_x[cand] <= rb[2])
        & (target_y[cand] >= rb[1])
        & (target_y[cand] <= rb[3])
    )
    if region_mask is not None and inside.any():
        mask = np.asarray(region_mask, dtype=bool)
        if mask.ndim == 2 and mask.size:
            nr, nc = mask.shape
            cols = np.clip((target_x[cand] / (cw / nc)).astype(np.int64), 0, nc - 1)
            rows = np.clip((target_y[cand] / (ch / nr)).astype(np.int64), 0, nr - 1)
            inside &= mask[rows, cols]
    if not inside.any():
        return cand
    relief = float(source_field) - target_field[cand]
    best_inside = float(np.max(relief[inside]))
    margin = float(const.HIER_PROPOSAL_OUTSIDE_RELIEF_MARGIN) * max(float(field_span), 1e-12)
    keep = inside | (relief >= best_inside + margin)
    filtered = cand[keep]
    return filtered if filtered.size else cand


def _point_in_region_mask(region_mask, x: float, y: float, cw: float, ch: float) -> bool:
    """Return whether a target center falls inside an optional grid-cell mask."""
    if region_mask is None:
        return True
    mask = np.asarray(region_mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        return True
    nr, nc = mask.shape
    c = int(np.clip(float(x) / (cw / nc), 0, nc - 1))
    r = int(np.clip(float(y) / (ch / nr), 0, nr - 1))
    return bool(mask[r, c])


def _target_pool_from_override(
    target_pool, flat, n_targets: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return sanitized flat grid target indices and optional component penalties."""
    if target_pool is None:
        return np.zeros(0, dtype=np.int64), None
    penalty_by_flat = None
    if isinstance(target_pool, dict):
        pool = np.asarray(target_pool.get("indices", []), dtype=np.int64).reshape(-1)
        penalties = np.asarray(target_pool.get("penalty", []), dtype=np.float64).reshape(-1)
        if pool.size and penalties.size == pool.size:
            penalty_by_flat = {int(idx): float(pen) for idx, pen in zip(pool, penalties)}
    else:
        pool = np.asarray(target_pool, dtype=np.int64).reshape(-1)
    if pool.size == 0:
        return np.zeros(0, dtype=np.int64), None
    valid = pool[(pool >= 0) & (pool < flat.size)]
    if valid.size == 0:
        return np.zeros(0, dtype=np.int64), None
    _, keep = np.unique(valid, return_index=True)
    pool = valid[np.sort(keep)]
    if pool.size == 0:
        return np.zeros(0, dtype=np.int64), None
    if penalty_by_flat is None:
        penalties = None
        order = np.argsort(flat[pool])
    else:
        penalties = np.asarray(
            [penalty_by_flat.get(int(idx), 1.0) for idx in pool], dtype=np.float64
        )
        order = np.argsort(
            penalties + 1e-9 * np.asarray(flat[pool], dtype=np.float64), kind="stable"
        )
    limit = max(pool.size, max(n_targets, 64))
    pool = pool[order[:limit]]
    if penalties is not None:
        penalties = penalties[order[:limit]]
    return pool, penalties


def _micro_shift_polish(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    movable_h: np.ndarray,
    soft_movable: "np.ndarray | None",
    n: int,
    plc,
    benchmark: "Benchmark",
    incremental_scorer,
    initial_score: float,
    hard_region: "np.ndarray | None" = None,
    soft_region: "np.ndarray | None" = None,
    deadline: "float | None" = None,
    radius_cells: int = 2,
    top_hot: int = 96,
    min_gain: float = 1e-5,
    use_density: bool = False,
    hard_candidate_allowed: "Callable[[int, float, float], bool] | None" = None,
    soft_candidate_allowed: "Callable[[int, float, float], bool] | None" = None,
) -> "tuple[np.ndarray, np.ndarray, int, float]":
    """Try exact-gated one/two-cell moves for hot macros inside their regions."""
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    field = (
        _density_field(incremental_scorer, nr, nc)
        if use_density
        else _congestion_field(incremental_scorer, nr, nc)
    )
    if field is None:
        return hard_pos, soft_pos, 0, float(initial_score)
    cell_w, cell_h = cw / nc, ch / nr
    best_score = float(initial_score)
    accepts = 0
    radius_cells = max(1, int(radius_cells))
    offsets = []
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dc == 0 and dr == 0:
                continue
            if max(abs(dc), abs(dr)) > radius_cells:
                continue
            offsets.append((dc * cell_w, dr * cell_h, abs(dc) + abs(dr)))
    offsets.sort(key=lambda v: (v[2], abs(v[0]) + abs(v[1])))

    ci = np.clip((hard_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((hard_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    hard_local = field[ri, ci]
    hard_idx = np.where(movable_h)[0]
    hard_hot = hard_idx[np.argsort(-hard_local[hard_idx])][: min(top_hot, hard_idx.size)]
    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    all_h = np.arange(n)

    for i_raw in hard_hot:
        if deadline is not None and time.monotonic() > deadline:
            break
        i = int(i_raw)
        best_xy = None
        try:
            mask = all_h != i
            ox = hard_pos[mask, 0]
            oy = hard_pos[mask, 1]
            sxi = sep_x_mat[i, mask]
            syi = sep_y_mat[i, mask]
            dxs = np.array([d[0] for d in offsets], dtype=np.float64)
            dys = np.array([d[1] for d in offsets], dtype=np.float64)
            if dxs.size:
                cand_x = hard_pos[i, 0] + dxs
                cand_y = hard_pos[i, 1] + dys
                in_bounds = (
                    (cand_x - hw[i] >= 0.0)
                    & (cand_x + hw[i] <= cw)
                    & (cand_y - hh[i] >= 0.0)
                    & (cand_y + hh[i] <= ch)
                )
                legal = np.zeros(dxs.size, dtype=np.bool_)
                if in_bounds.any():
                    legal_in = _legal_candidate_mask(
                        cand_x[in_bounds],
                        cand_y[in_bounds],
                        ox,
                        oy,
                        sxi,
                        syi,
                        0.05,
                    )
                    legal[in_bounds] = legal_in
            else:
                in_bounds = np.zeros(0, dtype=np.bool_)
                legal = np.zeros(0, dtype=np.bool_)

            targets = []
            for idx, (dx, dy, _dist) in enumerate(offsets):
                nx = float(hard_pos[i, 0] + dx)
                ny = float(hard_pos[i, 1] + dy)
                if idx >= legal.size or not in_bounds[idx] or not legal[idx]:
                    continue
                if hard_region is not None:
                    if not point_in_region(hard_region, i, nx, ny):
                        continue
                if hard_candidate_allowed is not None and not bool(
                    hard_candidate_allowed(i, nx, ny)
                ):
                    continue
                targets.append((nx, ny))
            targets = _dedupe_targets_xy(targets)
            if targets.size:
                prep = incremental_scorer._prepare_move(i)
                try:
                    scores = incremental_scorer._trial_many_at(prep, targets)
                finally:
                    incremental_scorer._revert_prep(prep)
            else:
                scores = np.empty(0, dtype=np.float64)
            for (nx, ny), score in zip(targets, scores):
                if float(score) < best_score - float(min_gain):
                    best_score = float(score)
                    best_xy = (nx, ny)
            if best_xy is not None:
                prep = incremental_scorer._prepare_move(i)
                incremental_scorer._commit_after_prep(prep, best_xy)
                hard_pos[i, 0], hard_pos[i, 1] = best_xy
                accepts += 1
        except Exception:
            raise

    if soft_pos.shape[0] == 0:
        return hard_pos, soft_pos, accepts, best_score
    sci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    sri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    soft_local = field[sri, sci]
    soft_idx = np.arange(soft_pos.shape[0])
    if soft_movable is not None:
        sm = np.asarray(soft_movable, dtype=bool)
        soft_idx = soft_idx[sm[soft_idx]]
    soft_hot = soft_idx[np.argsort(-soft_local[soft_idx])][: min(top_hot, soft_idx.size)]
    for k_raw in soft_hot:
        if deadline is not None and time.monotonic() > deadline:
            break
        k = int(k_raw)
        best_xy = None
        try:
            targets = []
            for dx, dy, _dist in offsets:
                nx = float(np.clip(soft_pos[k, 0] + dx, soft_hw[k], cw - soft_hw[k]))
                ny = float(np.clip(soft_pos[k, 1] + dy, soft_hh[k], ch - soft_hh[k]))
                if soft_region is not None and not point_in_region(soft_region, k, nx, ny):
                    continue
                if soft_candidate_allowed is not None and not bool(
                    soft_candidate_allowed(k, nx, ny)
                ):
                    continue
                targets.append((nx, ny))
            targets = _dedupe_targets_xy(targets)
            if targets.size:
                prep = incremental_scorer._prepare_move_soft(k)
                try:
                    scores = incremental_scorer._trial_many_at_soft(prep, targets)
                finally:
                    incremental_scorer._revert_prep_soft(prep)
            else:
                scores = np.empty(0, dtype=np.float64)
            for (nx, ny), score in zip(targets, scores):
                if float(score) < best_score - float(min_gain):
                    best_score = float(score)
                    best_xy = (nx, ny)
            if best_xy is not None:
                prep = incremental_scorer._prepare_move_soft(k)
                incremental_scorer._commit_after_prep_soft(prep, best_xy)
                soft_pos[k, 0], soft_pos[k, 1] = best_xy
                accepts += 1
        except Exception:
            raise
    return hard_pos, soft_pos, accepts, best_score


def _relocation_moves_propose_all(
    *,
    pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    n: int,
    plc,
    benchmark: "Benchmark",
    incremental_scorer,
    initial_score: float,
    deadline: "float | None",
    n_targets: int,
    net_centroid: "np.ndarray | None",
    wl_blend: float,
    hot: np.ndarray,
    pool: np.ndarray,
    tgt_x: np.ndarray,
    tgt_y: np.ndarray,
    tgt_cong: np.ndarray,
    local_cong: np.ndarray,
    sep_x_mat: np.ndarray,
    sep_y_mat: np.ndarray,
    propose_top_m: "int | None",
    eps: float,
    field: str,
    region_bbox: "np.ndarray | None" = None,
    region_bias: float = 0.0,
    region_escape_min: float = 0.0,
    accept_min_gain: float = 0.0,
    target_pool: "np.ndarray | None" = None,
    region_mask: "np.ndarray | None" = None,
    tgt_component_penalty: "np.ndarray | None" = None,
    candidate_allowed: "Callable[[int, float, float], bool] | None" = None,
    max_scored: "int | None" = None,
) -> "tuple[np.ndarray, int, float]":
    """Rank all hard-relocation proposals, then exact-check the best ones."""
    best_score = initial_score
    accepts = 0
    all_idx = np.arange(n)
    _span2 = float(max(cw, ch)) ** 2
    proposals = []
    full_pos_for_struct = _full_committed_pos(incremental_scorer)
    sizes_for_struct = _full_macro_sizes(incremental_scorer)
    t0 = time.monotonic()
    legal_count = 0
    frozen_scores = 0
    verify_scores = 0
    hierarchy_rejects = 0
    score_limit = None if max_scored is None else max(0, int(max_scored))

    for hot_rank, i_raw in enumerate(hot):
        if score_limit is not None and frozen_scores >= score_limit:
            break
        if deadline is not None and time.monotonic() > deadline:
            break
        i = int(i_raw)
        if not movable[i]:
            continue
        local_field = float(local_cong[i])
        cand_field = tgt_cong
        cand = np.where(cand_field < local_field - 1e-9)[0]
        if cand.size == 0:
            continue
        cand = _hierarchy_aware_target_filter(
            cand,
            target_x=tgt_x,
            target_y=tgt_y,
            target_field=cand_field,
            source_field=local_field,
            region_bbox=region_bbox,
            region_index=i,
            region_mask=region_mask,
            cw=cw,
            ch=ch,
            field_span=max(float(np.max(tgt_cong) - np.min(tgt_cong)), 1e-12),
            enabled=field == "weighted_congestion",
        )
        d2 = (tgt_x[cand] - pos[i, 0]) ** 2 + (tgt_y[cand] - pos[i, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[i, 0]) ** 2 + (tgt_y[cand] - net_centroid[i, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        if region_bbox is not None and region_bias > 0.0:
            d2 = _region_penalty(d2, tgt_x, tgt_y, cand, region_bbox, i, _span2, region_bias)
        if tgt_component_penalty is not None and tgt_component_penalty.size == tgt_cong.size:
            d2 = (
                d2
                + float(const.HIER_COLD_COMPONENT_RANK_WEIGHT)
                * _span2
                * tgt_component_penalty[cand]
            )
        cand = cand[np.argsort(d2)][:n_targets]

        mask = all_idx != i
        sxi = sep_x_mat[i, mask]
        syi = sep_y_mat[i, mask]
        ox = pos[mask, 0]
        oy = pos[mask, 1]
        legal = []
        cand_x = tgt_x[cand].astype(np.float64, copy=False)
        cand_y = tgt_y[cand].astype(np.float64, copy=False)
        in_bounds = (
            (cand_x - hw[i] >= 0.0)
            & (cand_x + hw[i] <= cw)
            & (cand_y - hh[i] >= 0.0)
            & (cand_y + hh[i] <= ch)
        )
        legal_mask = np.zeros(cand.size, dtype=np.bool_)
        if in_bounds.any():
            legal_mask[in_bounds] = _legal_candidate_mask(
                cand_x[in_bounds],
                cand_y[in_bounds],
                ox,
                oy,
                sxi,
                syi,
                eps,
            )
        for candidate_rank, t in enumerate(cand):
            nx, ny = float(tgt_x[t]), float(tgt_y[t])
            if (
                candidate_rank >= legal_mask.size
                or not in_bounds[candidate_rank]
                or not legal_mask[candidate_rank]
            ):
                continue
            if candidate_allowed is not None and not bool(candidate_allowed(i, nx, ny)):
                hierarchy_rejects += 1
                continue
            legal.append((int(candidate_rank), int(t), nx, ny))
        if not legal:
            continue
        legal_count += len(legal)
        if score_limit is not None:
            legal = legal[: max(0, score_limit - frozen_scores)]
        if not legal:
            break

        prep = incremental_scorer._prepare_move(i)
        try:
            targets = np.asarray([(nx, ny) for _, _, nx, ny in legal], dtype=np.float64)
            scores = incremental_scorer._trial_many_at(prep, targets)
            for (candidate_rank, target_index, nx, ny), score in zip(legal, scores):
                if deadline is not None and time.monotonic() > deadline:
                    break
                frozen_scores += 1
                proposals.append(
                    {
                        "score": float(score),
                        "i": i,
                        "hot_rank": int(hot_rank),
                        "candidate_rank": int(candidate_rank),
                        "target_index": int(target_index),
                        "local_field": local_field,
                        "target_field": float(cand_field[target_index]),
                        "component_penalty": (
                            float(tgt_component_penalty[target_index])
                            if tgt_component_penalty is not None
                            and tgt_component_penalty.size == tgt_cong.size
                            else 0.0
                        ),
                        "xy": (nx, ny),
                    }
                )
        finally:
            incremental_scorer._revert_prep(prep)

    _apply_structural_proposal_scores(
        proposals,
        full_pos=full_pos_for_struct,
        sizes=sizes_for_struct,
        cw=cw,
        ch=ch,
        benchmark=benchmark,
        n_hard=n,
    )

    if not proposals:
        _relocation_moves_propose_all.last_stats = {
            "candidates": 0,
            "legal": int(legal_count),
            "scored": int(frozen_scores),
            "verify_scores": int(verify_scores),
            "hierarchy_rejects": int(hierarchy_rejects),
            "score_limit": score_limit,
            "quota_exhausted": bool(score_limit is not None and frozen_scores >= score_limit),
            "accepts": 0,
        }
        return pos, 0, best_score

    proposals.sort(
        key=lambda p: (
            p["score"],
            p["hot_rank"],
            p["candidate_rank"],
            p["i"],
            p["target_index"],
        )
    )
    if propose_top_m is not None and propose_top_m > 0:
        top_m = int(propose_top_m)
        additive_extra = max(0, int(const.HIER_ADDITIVE_RELOC_EXTRA_TOP_K))
        proposals = proposals[: top_m + additive_extra]

    moved = set()
    for p in proposals:
        if deadline is not None and time.monotonic() > deadline:
            break
        i = int(p["i"])
        if i in moved or not movable[i]:
            continue
        nx, ny = p["xy"]
        # Keep the whole macro inside the canvas.
        if nx - hw[i] < 0 or nx + hw[i] > cw or ny - hh[i] < 0 or ny + hh[i] > ch:
            continue
        mask = all_idx != i
        if not _legal_candidate_mask(
            np.array([nx], dtype=np.float64),
            np.array([ny], dtype=np.float64),
            pos[mask, 0],
            pos[mask, 1],
            sep_x_mat[i, mask],
            sep_y_mat[i, mask],
            eps,
        )[0]:
            continue

        prep = incremental_scorer._prepare_move(i)
        try:
            score = incremental_scorer._trial_at(prep, (nx, ny))
            verify_scores += 1
            outside = not point_in_region(region_bbox, i, nx, ny)
            min_gain = max(
                1e-9,
                float(accept_min_gain),
                float(region_escape_min) if outside else 0.0,
            )
            if float(score) < float(best_score) - min_gain:
                incremental_scorer._commit_after_prep(prep, (nx, ny))
                pos[i, 0], pos[i, 1] = nx, ny
                best_score = float(score)
                accepts += 1
                moved.add(i)
            else:
                incremental_scorer._revert_prep(prep)
        except Exception:
            incremental_scorer._revert_prep(prep)
            raise

    if os.environ.get("RELOC_PROPOSE_LOG", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
        "on",
        "ON",
    }:
        print(
            "  R2 propose-all[%s]: hot=%d legal=%d frozen_scores=%d "
            "selected=%d verify_scores=%d accepts=%d elapsed=%.3fs"
            % (
                field,
                len(hot),
                legal_count,
                frozen_scores,
                len(proposals),
                verify_scores,
                accepts,
                time.monotonic() - t0,
            ),
            flush=True,
        )
    _relocation_moves_propose_all.last_stats = {
        "candidates": int(len(proposals)),
        "legal": int(legal_count),
        "scored": int(frozen_scores),
        "verify_scores": int(verify_scores),
        "hierarchy_rejects": int(hierarchy_rejects),
        "score_limit": score_limit,
        "quota_exhausted": bool(score_limit is not None and frozen_scores >= score_limit),
        "accepts": int(accepts),
    }
    return pos, accepts, best_score


def _relocation_moves(
    pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    n: int,
    plc,
    benchmark: "Benchmark",
    incremental_scorer,
    initial_score: float,
    deadline: "float | None" = None,
    top_hot: int = 24,
    n_targets: int = 12,
    net_centroid: "np.ndarray | None" = None,
    wl_blend: float = 0.0,
    use_density: bool = False,
    use_combined: bool = False,
    propose_all: bool = False,
    propose_top_m: "int | None" = None,
    region_bbox: "np.ndarray | None" = None,
    region_bias: float = 0.0,
    region_escape_min: float = 0.0,
    propose_accept_min_gain: float = 0.0,
    target_pool: "np.ndarray | None" = None,
    region_mask: "np.ndarray | None" = None,
    candidate_allowed: "Callable[[int, float, float], bool] | None" = None,
    max_scored: "int | None" = None,
) -> "tuple[np.ndarray, int, float]":
    """Move hot hard macros to colder legal spots.

    When `region_bbox` (per-macro center-feasible box [n,4]) and `region_bias>0`
    are given, out-of-region candidate cells get a ranking penalty so a macro
    strongly prefers staying within its cluster region — a SOFT region lock (it
    can still exit when in-region cold cells run out). Bit-identical when
    `region_bbox is None`.
    """
    _relocation_moves.last_stats = {
        "candidates": 0,
        "legal": 0,
        "scored": 0,
        "hierarchy_rejects": 0,
        "score_limit": None if max_scored is None else max(0, int(max_scored)),
        "quota_exhausted": bool(max_scored is not None and int(max_scored) <= 0),
        "accepts": 0,
    }
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    weighted_rank = not use_density and not use_combined
    trace_field = (
        "weighted_congestion"
        if weighted_rank
        else ("combined" if use_combined else ("density" if use_density else "congestion"))
    )
    if use_combined:
        # A cell is hot only when both fields are high.
        cong_field = _congestion_field(incremental_scorer, nr, nc)
        dens_field = _density_field(incremental_scorer, nr, nc)
        if cong_field is None or dens_field is None:
            return pos, 0, initial_score
        cong_max = max(float(cong_field.max()), 1e-12)
        dens_max = max(float(dens_field.max()), 1e-12)
        cell_cong = np.sqrt((cong_field / cong_max) * (dens_field / dens_max))
    else:
        cell_cong = (
            _density_field(incremental_scorer, nr, nc)
            if use_density
            else (
                weighted_congestion_field(incremental_scorer, nr, nc)
                if weighted_rank
                else _congestion_field(incremental_scorer, nr, nc)
            )
        )
    if cell_cong is None:
        return pos, 0, initial_score
    cell_w, cell_h = cw / nc, ch / nr
    # Pick the hottest movable macros.
    ci_all = np.clip((pos[:n, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri_all = np.clip((pos[:n, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_cong[ri_all, ci_all]
    mov_idx = np.where(movable)[0]
    if mov_idx.size == 0:
        return pos, 0, initial_score
    hot = mov_idx[np.argsort(-local_cong[mov_idx])][:top_hot]

    # Use low-field cells as possible targets.
    flat = cell_cong.ravel()
    pool, tgt_component_penalty = _target_pool_from_override(target_pool, flat, n_targets)
    if pool.size == 0:
        _thr = np.percentile(flat, 55)
        pool = np.where(flat < _thr)[0]
        if pool.size < max(n_targets, 64):
            pool = np.argsort(flat)[: max(n_targets, 64)]
        tgt_component_penalty = None
    tgt_c = (pool % nc).astype(np.float64)
    tgt_r = (pool // nc).astype(np.float64)
    tgt_x = (tgt_c + 0.5) * cell_w
    tgt_y = (tgt_r + 0.5) * cell_h
    tgt_cong = flat[pool]

    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    EPS = 0.05

    if propose_all:
        result = _relocation_moves_propose_all(
            pos=pos,
            sizes=sizes,
            hw=hw,
            hh=hh,
            cw=cw,
            ch=ch,
            movable=movable,
            n=n,
            plc=plc,
            benchmark=benchmark,
            incremental_scorer=incremental_scorer,
            initial_score=initial_score,
            deadline=deadline,
            n_targets=n_targets,
            net_centroid=net_centroid,
            wl_blend=wl_blend,
            hot=hot,
            pool=pool,
            tgt_x=tgt_x,
            tgt_y=tgt_y,
            tgt_cong=tgt_cong,
            local_cong=local_cong,
            sep_x_mat=sep_x_mat,
            sep_y_mat=sep_y_mat,
            propose_top_m=propose_top_m,
            eps=EPS,
            field=trace_field,
            region_bbox=region_bbox,
            region_bias=region_bias,
            region_escape_min=region_escape_min,
            accept_min_gain=propose_accept_min_gain,
            target_pool=target_pool,
            region_mask=region_mask,
            tgt_component_penalty=tgt_component_penalty,
            candidate_allowed=candidate_allowed,
            max_scored=max_scored,
        )
        _relocation_moves.last_stats = dict(
            getattr(_relocation_moves_propose_all, "last_stats", {})
        )
        return result

    best_score = initial_score
    accepts = 0
    candidate_count = 0
    legal_count = 0
    scored_count = 0
    hierarchy_rejects = 0
    score_limit = None if max_scored is None else max(0, int(max_scored))
    all_idx = np.arange(n)
    _span2 = float(max(cw, ch)) ** 2
    full_pos_for_struct = _full_committed_pos(incremental_scorer)
    sizes_for_struct = _full_macro_sizes(incremental_scorer)
    for i in hot:
        i = int(i)
        if score_limit is not None and scored_count >= score_limit:
            break
        if deadline is not None and time.monotonic() > deadline:
            break
        if not movable[i]:
            continue
        # Try colder targets first, capped per macro.
        cand_field = tgt_cong
        cand = np.where(cand_field < local_cong[i] - 1e-9)[0]
        if cand.size == 0:
            continue
        candidate_count += int(cand.size)
        cand = _hierarchy_aware_target_filter(
            cand,
            target_x=tgt_x,
            target_y=tgt_y,
            target_field=cand_field,
            source_field=float(local_cong[i]),
            region_bbox=region_bbox,
            region_index=i,
            region_mask=region_mask,
            cw=cw,
            ch=ch,
            field_span=max(float(np.max(flat) - np.min(flat)), 1e-12),
            enabled=weighted_rank,
        )
        d2 = (tgt_x[cand] - pos[i, 0]) ** 2 + (tgt_y[cand] - pos[i, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[i, 0]) ** 2 + (tgt_y[cand] - net_centroid[i, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        if region_bbox is not None and region_bias > 0.0:
            d2 = _region_penalty(d2, tgt_x, tgt_y, cand, region_bbox, i, _span2, region_bias)
        if tgt_component_penalty is not None and tgt_component_penalty.size == tgt_cong.size:
            d2 = (
                d2
                + float(const.HIER_COLD_COMPONENT_RANK_WEIGHT)
                * _span2
                * tgt_component_penalty[cand]
            )
        order = _structural_candidate_order(
            cand=cand,
            base_rank=d2,
            module_index=i,
            target_x=tgt_x,
            target_y=tgt_y,
            full_pos=full_pos_for_struct,
            sizes=sizes_for_struct,
            cw=cw,
            ch=ch,
            benchmark=benchmark,
        )
        cand = cand[order][:n_targets]

        mask = all_idx != i
        sxi = sep_x_mat[i, mask]
        syi = sep_y_mat[i, mask]
        ox = pos[mask, 0]
        oy = pos[mask, 1]
        best_i_xy = None
        try:
            targets = []
            for candidate_rank, t in enumerate(cand):
                nx, ny = float(tgt_x[t]), float(tgt_y[t])
                # Keep the whole macro inside the canvas.
                if nx - hw[i] < 0 or nx + hw[i] > cw or ny - hh[i] < 0 or ny + hh[i] > ch:
                    continue
                if not _point_in_region_mask(region_mask, nx, ny, cw, ch):
                    continue
                # Hard macros cannot overlap.
                if ((np.abs(nx - ox) < sxi + EPS) & (np.abs(ny - oy) < syi + EPS)).any():
                    continue
                if candidate_allowed is not None and not bool(candidate_allowed(i, nx, ny)):
                    hierarchy_rejects += 1
                    continue
                targets.append((nx, ny))
            if targets:
                targets_arr = np.asarray(targets, dtype=np.float64)
                if score_limit is not None:
                    targets_arr = targets_arr[: max(0, score_limit - scored_count)]
                prep = incremental_scorer._prepare_move(i)
                try:
                    scores = incremental_scorer._trial_many_at(prep, targets_arr)
                finally:
                    incremental_scorer._revert_prep(prep)
            else:
                scores = np.empty(0, dtype=np.float64)
            legal_count += int(len(targets))
            scored_count += int(scores.size)
            for nx, ny, s in [(x, y, score) for (x, y), score in zip(targets, scores)]:
                outside = not (
                    point_in_region(region_bbox, i, nx, ny)
                    and _point_in_region_mask(region_mask, nx, ny, cw, ch)
                )
                if accepts_region_score(s, best_score, outside, region_escape_min):
                    best_score = s
                    best_i_xy = (nx, ny)
            if best_i_xy is not None:
                prep = incremental_scorer._prepare_move(i)
                incremental_scorer._commit_after_prep(prep, best_i_xy)
                pos[i, 0], pos[i, 1] = best_i_xy
                full_pos_for_struct[i, 0], full_pos_for_struct[i, 1] = best_i_xy
                accepts += 1
        except Exception:
            # Restore committed state if a trial failed.
            raise
    _relocation_moves.last_stats = {
        "candidates": int(candidate_count),
        "legal": int(legal_count),
        "scored": int(scored_count),
        "hierarchy_rejects": int(hierarchy_rejects),
        "score_limit": score_limit,
        "quota_exhausted": bool(score_limit is not None and scored_count >= score_limit),
        "accepts": int(accepts),
    }
    return pos, accepts, best_score


def _soft_relocation_moves(
    soft_pos: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    plc,
    benchmark: "Benchmark",
    incremental_scorer,
    initial_score: float,
    deadline: "float | None" = None,
    top_hot: int = 48,
    n_targets: int = 16,
    soft_movable: "np.ndarray | None" = None,
    use_density: bool = False,
    net_centroid: "np.ndarray | None" = None,
    wl_blend: float = 0.0,
    wl_prefilter: float = 1e-4,
    region_bbox: "np.ndarray | None" = None,
    region_bias: float = 0.0,
    region_escape_min: float = 0.0,
    accept_min_gain: float = 0.0,
    target_pool: "np.ndarray | None" = None,
    region_mask: "np.ndarray | None" = None,
    candidate_allowed: "Callable[[int, float, float], bool] | None" = None,
    max_scored: "int | None" = None,
) -> "tuple[np.ndarray, int, float]":
    """Move hot soft macros to colder cells."""
    _soft_relocation_moves.last_stats = {
        "candidates": 0,
        "legal": 0,
        "scored": 0,
        "hierarchy_rejects": 0,
        "score_limit": None if max_scored is None else max(0, int(max_scored)),
        "quota_exhausted": bool(max_scored is not None and int(max_scored) <= 0),
        "accepts": 0,
    }
    num_soft = incremental_scorer.num_soft
    if num_soft == 0:
        return soft_pos, 0, initial_score
    # Skip full scoring when wirelength alone is already too costly.
    _env_wl = os.environ.get("SOFT_RELOC_WL_PREFILTER")
    if _env_wl not in (None, ""):
        wl_prefilter = float(_env_wl)
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    weighted_rank = not use_density
    cell_field = (
        _density_field(incremental_scorer, nr, nc)
        if use_density
        else (
            weighted_congestion_field(incremental_scorer, nr, nc)
            if weighted_rank
            else _congestion_field(incremental_scorer, nr, nc)
        )
    )
    if cell_field is None:
        return soft_pos, 0, initial_score
    cell_w, cell_h = cw / nc, ch / nr

    ci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_field[ri, ci]
    # Fixed soft macros must stay put.
    order = np.argsort(-local_cong)
    if soft_movable is not None:
        sm = np.asarray(soft_movable, dtype=bool)
        order = order[sm[order]]
    hot = order[:top_hot]

    flat = cell_field.ravel()
    # Use low-field cells as possible targets.
    pool, tgt_component_penalty = _target_pool_from_override(target_pool, flat, n_targets)
    if pool.size == 0:
        _thr = np.percentile(flat, 55)
        pool = np.where(flat < _thr)[0]
        if pool.size < max(n_targets, 64):
            pool = np.argsort(flat)[: max(n_targets, 64)]
        tgt_component_penalty = None
    tgt_x = ((pool % nc).astype(np.float64) + 0.5) * cell_w
    tgt_y = ((pool // nc).astype(np.float64) + 0.5) * cell_h
    tgt_cong = flat[pool]

    best_score = initial_score
    accepts = 0
    candidate_count = 0
    legal_count = 0
    scored_count = 0
    hierarchy_rejects = 0
    score_limit = None if max_scored is None else max(0, int(max_scored))
    full_pos_for_struct = _full_committed_pos(incremental_scorer)
    sizes_for_struct = _full_macro_sizes(incremental_scorer)
    for k in hot:
        k = int(k)
        if score_limit is not None and scored_count >= score_limit:
            break
        if deadline is not None and time.monotonic() > deadline:
            break
        cand = np.where(tgt_cong < local_cong[k] - 1e-9)[0]
        if cand.size == 0:
            continue
        candidate_count += int(cand.size)
        cand = _hierarchy_aware_target_filter(
            cand,
            target_x=tgt_x,
            target_y=tgt_y,
            target_field=tgt_cong,
            source_field=float(local_cong[k]),
            region_bbox=region_bbox,
            region_index=k,
            region_mask=region_mask,
            cw=cw,
            ch=ch,
            field_span=max(float(np.max(flat) - np.min(flat)), 1e-12),
            enabled=weighted_rank,
        )
        # Order targets by distance and optional wirelength anchor.
        d2 = (tgt_x[cand] - soft_pos[k, 0]) ** 2 + (tgt_y[cand] - soft_pos[k, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[k, 0]) ** 2 + (tgt_y[cand] - net_centroid[k, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        if region_bbox is not None and region_bias > 0.0:
            _span2 = float(max(cw, ch)) ** 2
            d2 = _region_penalty(d2, tgt_x, tgt_y, cand, region_bbox, k, _span2, region_bias)
        if tgt_component_penalty is not None and tgt_component_penalty.size == tgt_cong.size:
            _span2 = float(max(cw, ch)) ** 2
            d2 = (
                d2
                + float(const.HIER_COLD_COMPONENT_RANK_WEIGHT)
                * _span2
                * tgt_component_penalty[cand]
            )
        order = _structural_candidate_order(
            cand=cand,
            base_rank=d2,
            module_index=n + k,
            target_x=tgt_x,
            target_y=tgt_y,
            full_pos=full_pos_for_struct,
            sizes=sizes_for_struct,
            cw=cw,
            ch=ch,
            benchmark=benchmark,
        )
        cand = cand[order][:n_targets]
        best_k_xy = None
        try:
            targets = []
            for t in cand:
                nx = float(np.clip(tgt_x[t], soft_hw[k], cw - soft_hw[k]))
                ny = float(np.clip(tgt_y[t], soft_hh[k], ch - soft_hh[k]))
                if not _point_in_region_mask(region_mask, nx, ny, cw, ch):
                    continue
                if candidate_allowed is not None and not bool(candidate_allowed(k, nx, ny)):
                    hierarchy_rejects += 1
                    continue
                # Cheaply skip targets with too much wirelength damage.
                wl_d = 0.0
                if wl_prefilter > 0.0:
                    wl_d = incremental_scorer.wl_delta_move_soft(k, (nx, ny))
                if wl_prefilter > 0.0 and wl_d > wl_prefilter:
                    continue
                targets.append((nx, ny))
            targets = _dedupe_targets_xy(targets)
            if score_limit is not None:
                targets = targets[: max(0, score_limit - scored_count)]
            if targets.size:
                prep = incremental_scorer._prepare_move_soft(k)
                try:
                    scores = incremental_scorer._trial_many_at_soft(prep, targets)
                finally:
                    incremental_scorer._revert_prep_soft(prep)
            else:
                scores = np.empty(0, dtype=np.float64)
            legal_count += int(targets.shape[0])
            scored_count += int(scores.size)
            for nx, ny, s in [
                (float(p[0]), float(p[1]), float(score)) for p, score in zip(targets, scores)
            ]:
                outside = not (
                    point_in_region(region_bbox, k, nx, ny)
                    and _point_in_region_mask(region_mask, nx, ny, cw, ch)
                )
                min_gain = max(
                    1e-9,
                    float(accept_min_gain),
                    float(region_escape_min) if outside else 0.0,
                )
                if float(s) < float(best_score) - min_gain:
                    best_score = s
                    best_k_xy = (nx, ny)
            if best_k_xy is not None:
                prep = incremental_scorer._prepare_move_soft(k)
                incremental_scorer._commit_after_prep_soft(prep, best_k_xy)
                soft_pos[k, 0], soft_pos[k, 1] = best_k_xy
                full_pos_for_struct[n + k, 0], full_pos_for_struct[n + k, 1] = best_k_xy
                accepts += 1
        except Exception:
            raise
    _soft_relocation_moves.last_stats = {
        "candidates": int(candidate_count),
        "legal": int(legal_count),
        "scored": int(scored_count),
        "hierarchy_rejects": int(hierarchy_rejects),
        "score_limit": score_limit,
        "quota_exhausted": bool(score_limit is not None and scored_count >= score_limit),
        "accepts": int(accepts),
    }
    return soft_pos, accepts, best_score
