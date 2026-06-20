"""Relocation moves for local search."""

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from utils import constants as const
from utils.config import (
    HAS_NUMBA,
    _CUDA_DEVICE_REQUESTED,
    _GPU_BACKEND,
    _GPU_DEVICE,
    _GPU_DEVICE_NAME,
    _numba_njit,
)
from placer.shared.geometry import separation_matrices
from placer.local_search.fields import _congestion_field, _density_field
from placer.local_search.gnn_trace import gnn_trace_limit, log_gnn_event
from placer.local_search.region_rules import accepts_region_score, point_in_region
from placer.local_search.structural_fields import combined_structural_penalty
from placer.plc.placement import _ensure_pos_cache

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark


def _proposal_scorer_mode() -> str:
    mode = const.RELOC_PROPOSE_SCORER.strip().lower()
    return mode if mode in {"exact", "tensor", "cuda_delta"} else "cuda_delta"


def _structural_weights() -> tuple[float, float, float]:
    return (
        float(const.HIER_KEEP_OUT_WEIGHT),
        float(const.HIER_GRID_ALIGN_WEIGHT),
        float(const.HIER_NOTCH_WEIGHT),
    )


def _hierarchy_structural_weight() -> float:
    """Weight for BeyondPPA-style structure inside hierarchy candidate ordering."""
    return max(0.0, float(const.HIER_OBJECTIVE_STRUCTURAL_WEIGHT))


def _candidate_trace_sample(proposals: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    out = []
    for p in proposals[:limit]:
        out.append(
            {
                "macro": int(p["i"]),
                "hot_rank": int(p.get("hot_rank", -1)),
                "candidate_rank": int(p.get("candidate_rank", -1)),
                "target_index": int(p.get("target_index", -1)),
                "score": float(p.get("score", 0.0)),
                "local_field": float(p.get("local_field", 0.0)),
                "target_field": float(p.get("target_field", 0.0)),
                "structural_delta": float(p.get("structural_delta", 0.0)),
                "x": float(p["xy"][0]),
                "y": float(p["xy"][1]),
                "gnn_score": float(p["gnn_score"]) if "gnn_score" in p else None,
                "gnn_rank_error": p.get("gnn_rank_error"),
            }
        )
    return out


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


def _structural_penalty(
    full_pos: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    benchmark: "Benchmark",
) -> float:
    kw, gw, nw = _structural_weights()
    return combined_structural_penalty(
        full_pos,
        sizes,
        cw,
        ch,
        grid_cols=int(benchmark.grid_cols),
        grid_rows=int(benchmark.grid_rows),
        keepout_weight=kw,
        grid_align_weight=gw,
        notch_weight=nw,
    )


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


def _score_relocation_proposals_tensor(
    proposals: list[dict],
    *,
    pos: np.ndarray,
    cw: float,
    ch: float,
    local_cong: np.ndarray,
    tgt_cong: np.ndarray,
) -> None:
    """Give each proposal a cheap ranking score."""
    if not proposals:
        return
    dev = _GPU_DEVICE
    idx = np.asarray([p["i"] for p in proposals], dtype=np.int64)
    candidate_rank = np.asarray([p["candidate_rank"] for p in proposals], dtype=np.float32)
    nx = np.asarray([p["xy"][0] for p in proposals], dtype=np.float32)
    ny = np.asarray([p["xy"][1] for p in proposals], dtype=np.float32)
    local_field = np.asarray(
        [p.get("local_field", local_cong[p["i"]]) for p in proposals],
        dtype=np.float32,
    )
    target_field = np.asarray(
        [p.get("target_field", tgt_cong[p["target_index"]]) for p in proposals],
        dtype=np.float32,
    )

    with torch.inference_mode():
        i_t = torch.as_tensor(idx, device=dev, dtype=torch.long)
        nx_t = torch.as_tensor(nx, device=dev)
        ny_t = torch.as_tensor(ny, device=dev)
        rank_t = torch.as_tensor(candidate_rank, device=dev)

        pos_t = torch.as_tensor(pos, device=dev, dtype=torch.float32)
        local_t = torch.as_tensor(local_field, device=dev, dtype=torch.float32)
        tgt_t = torch.as_tensor(target_field, device=dev, dtype=torch.float32)

        field_max = torch.clamp(torch.max(local_t), min=1e-12)
        relief = torch.clamp((local_t - tgt_t) / field_max, min=0.0)
        dx = (nx_t - pos_t[i_t, 0]) / max(float(cw), 1e-12)
        dy = (ny_t - pos_t[i_t, 1]) / max(float(ch), 1e-12)
        dist = torch.sqrt(dx * dx + dy * dy)
        rank_norm = rank_t / torch.clamp(torch.max(rank_t), min=1.0)

        # Prefer more relief, shorter moves, and earlier heuristic picks.
        score = -(1.0 * relief - 0.08 * dist - 0.02 * rank_norm)
        scores = score.detach().cpu().numpy().astype(np.float64)
    for proposal, score_value in zip(proposals, scores):
        proposal["score"] = float(score_value)


def _macro_routing_contrib(
    incremental_scorer,
    module_idx: int,
    cx: float,
    cy: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return one hard macro's routing blockage as sparse arrays."""
    half = incremental_scorer._dens_half.get(int(module_idx))
    if half is None:
        empty_i = np.empty(0, dtype=np.int64)
        empty_v = np.empty(0, dtype=np.float32)
        return empty_i, empty_v, empty_v

    half_w, half_h = half
    grid_w = float(incremental_scorer.grid_w)
    grid_h = float(incremental_scorer.grid_h)
    grid_row = int(incremental_scorer.grid_row)
    grid_col = int(incremental_scorer.grid_col)
    x_min = float(cx) - float(half_w)
    x_max = float(cx) + float(half_w)
    y_min = float(cy) - float(half_h)
    y_max = float(cy) + float(half_h)

    bl_col = int(np.floor(x_min / grid_w))
    bl_row = int(np.floor(y_min / grid_h))
    ur_col = int(np.floor(x_max / grid_w))
    ur_row = int(np.floor(y_max / grid_h))
    if not (ur_row >= 0 and ur_col >= 0 and bl_row <= grid_row - 1 and bl_col <= grid_col - 1):
        empty_i = np.empty(0, dtype=np.int64)
        empty_v = np.empty(0, dtype=np.float32)
        return empty_i, empty_v, empty_v

    bl_col = min(max(bl_col, 0), grid_col - 1)
    ur_col = min(max(ur_col, 0), grid_col - 1)
    bl_row = min(max(bl_row, 0), grid_row - 1)
    ur_row = min(max(ur_row, 0), grid_row - 1)

    cols = np.arange(bl_col, ur_col + 1, dtype=np.int64)
    rows = np.arange(bl_row, ur_row + 1, dtype=np.int64)
    cc_g = np.broadcast_to(cols[None, :], (rows.size, cols.size)).ravel()
    rr_g = np.broadcast_to(rows[:, None], (rows.size, cols.size)).ravel()
    flat = rr_g * grid_col + cc_g

    cell_xmin = grid_w * cc_g.astype(np.float64)
    cell_xmax = grid_w * (cc_g + 1).astype(np.float64)
    cell_ymin = grid_h * rr_g.astype(np.float64)
    cell_ymax = grid_h * (rr_g + 1).astype(np.float64)
    x_dist = np.minimum(cell_xmax, x_max) - np.maximum(cell_xmin, x_min)
    y_dist = np.minimum(cell_ymax, y_max) - np.maximum(cell_ymin, y_min)
    np.maximum(x_dist, 0.0, out=x_dist)
    np.maximum(y_dist, 0.0, out=y_dist)

    v_val = x_dist * float(incremental_scorer.plc.vrouting_alloc)
    h_val = y_dist * float(incremental_scorer.plc.hrouting_alloc)

    tol = 1e-5
    if ur_row != bl_row:
        bot_partial = abs((grid_h * (bl_row + 1) - y_min) - grid_h) > tol
        top_partial = abs((y_max - grid_h * ur_row) - grid_h) > tol
        if bot_partial or top_partial:
            top_mask = rr_g == ur_row
            v_val[top_mask] -= x_dist[top_mask] * float(incremental_scorer.plc.vrouting_alloc)
    if ur_col != bl_col:
        left_partial = abs((grid_w * (bl_col + 1) - x_min) - grid_w) > tol
        right_partial = abs((x_max - grid_w * ur_col) - grid_w) > tol
        if left_partial or right_partial:
            right_mask = cc_g == ur_col
            h_val[right_mask] -= y_dist[right_mask] * float(incremental_scorer.plc.hrouting_alloc)

    return (
        flat.astype(np.int64, copy=False),
        v_val.astype(np.float32, copy=False),
        h_val.astype(np.float32, copy=False),
    )


def _net_routing_2pin_contrib(
    incremental_scorer,
    module_idx: int,
    cx: float,
    cy: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return 2-pin routing for nets touching one module."""
    empty_i = np.empty(0, dtype=np.int64)
    empty_v = np.empty(0, dtype=np.float32)
    nets = incremental_scorer.macro_to_nets.get(int(module_idx))
    if nets is None or len(nets) == 0:
        return empty_i, empty_v, empty_v

    lengths = incremental_scorer.net_lengths[nets]
    nets_2 = nets[lengths == 2]
    if len(nets_2) == 0:
        return empty_i, empty_v, empty_v

    starts = incremental_scorer.net_starts[nets_2]
    pin0 = starts
    pin1 = starts + 1
    pin_indices = np.concatenate([pin0, pin1])
    pos_cache = _ensure_pos_cache(incremental_scorer.plc)
    ref_local = incremental_scorer.ref_inv[pin_indices]
    refs = incremental_scorer.unique_ref[ref_local]
    pin_x = pos_cache[refs, 0] + incremental_scorer.x_off[pin_indices]
    pin_y = pos_cache[refs, 1] + incremental_scorer.y_off[pin_indices]
    moved = refs == int(module_idx)
    if moved.any():
        pin_x[moved] = float(cx) + incremental_scorer.x_off[pin_indices[moved]]
        pin_y[moved] = float(cy) + incremental_scorer.y_off[pin_indices[moved]]

    n2 = len(nets_2)
    src_x = pin_x[:n2]
    src_y = pin_y[:n2]
    snk_x = pin_x[n2:]
    snk_y = pin_y[n2:]
    grid_col = int(incremental_scorer.grid_col)
    grid_row = int(incremental_scorer.grid_row)
    grid_w = float(incremental_scorer.grid_w)
    grid_h = float(incremental_scorer.grid_h)
    src_col = np.clip((src_x / grid_w).astype(np.int64), 0, grid_col - 1)
    src_row = np.clip((src_y / grid_h).astype(np.int64), 0, grid_row - 1)
    snk_col = np.clip((snk_x / grid_w).astype(np.int64), 0, grid_col - 1)
    snk_row = np.clip((snk_y / grid_h).astype(np.int64), 0, grid_row - 1)
    active = (src_col != snk_col) | (src_row != snk_row)
    if not active.any():
        return empty_i, empty_v, empty_v

    src_col = src_col[active]
    src_row = src_row[active]
    snk_col = snk_col[active]
    snk_row = snk_row[active]
    weights = incremental_scorer.net_weights[nets_2][active].astype(np.float32, copy=False)

    h_parts_idx = []
    h_parts_val = []
    col_lo = np.minimum(src_col, snk_col)
    col_hi = np.maximum(src_col, snk_col)
    for r, lo, hi, weight in zip(src_row, col_lo, col_hi, weights):
        if hi <= lo:
            continue
        cols = np.arange(int(lo), int(hi), dtype=np.int64)
        h_parts_idx.append(int(r) * grid_col + cols)
        h_parts_val.append(np.full(cols.size, float(weight), dtype=np.float32))

    v_parts_idx = []
    v_parts_val = []
    row_lo = np.minimum(src_row, snk_row)
    row_hi = np.maximum(src_row, snk_row)
    for c, lo, hi, weight in zip(snk_col, row_lo, row_hi, weights):
        if hi <= lo:
            continue
        rows = np.arange(int(lo), int(hi), dtype=np.int64)
        v_parts_idx.append(rows * grid_col + int(c))
        v_parts_val.append(np.full(rows.size, float(weight), dtype=np.float32))

    h_idx = np.concatenate(h_parts_idx) if h_parts_idx else empty_i
    h_val = np.concatenate(h_parts_val) if h_parts_val else empty_v
    v_idx = np.concatenate(v_parts_idx) if v_parts_idx else empty_i
    v_val = np.concatenate(v_parts_val) if v_parts_val else empty_v

    if h_idx.size == 0:
        return v_idx, v_val, np.zeros(v_idx.size, dtype=np.float32)
    if v_idx.size == 0:
        return h_idx, np.zeros(h_idx.size, dtype=np.float32), h_val
    flat = np.concatenate([v_idx, h_idx])
    v_out = np.concatenate([v_val, np.zeros(h_idx.size, dtype=np.float32)])
    h_out = np.concatenate([np.zeros(v_idx.size, dtype=np.float32), h_val])
    return flat, v_out, h_out


def _sparse_from_route_strips(
    *,
    grid_col: int,
    h_rows: list[np.ndarray],
    h_los: list[np.ndarray],
    h_his: list[np.ndarray],
    h_ws: list[np.ndarray],
    v_cols: list[np.ndarray],
    v_los: list[np.ndarray],
    v_his: list[np.ndarray],
    v_ws: list[np.ndarray],
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    empty_i = np.empty(0, dtype=np.int64)
    empty_v = np.empty(0, dtype=np.float32)
    h_parts_idx = []
    h_parts_val = []
    if h_rows:
        rows = np.concatenate(h_rows)
        los = np.concatenate(h_los)
        his = np.concatenate(h_his)
        ws_h = np.concatenate(h_ws)
        for r, lo, hi, weight in zip(rows, los, his, ws_h):
            if int(hi) <= int(lo):
                continue
            cols = np.arange(int(lo), int(hi), dtype=np.int64)
            h_parts_idx.append(int(r) * grid_col + cols)
            h_parts_val.append(np.full(cols.size, float(weight), dtype=np.float32))

    v_parts_idx = []
    v_parts_val = []
    if v_cols:
        cols = np.concatenate(v_cols)
        los = np.concatenate(v_los)
        his = np.concatenate(v_his)
        ws_v = np.concatenate(v_ws)
        for c, lo, hi, weight in zip(cols, los, his, ws_v):
            if int(hi) <= int(lo):
                continue
            rows = np.arange(int(lo), int(hi), dtype=np.int64)
            v_parts_idx.append(rows * grid_col + int(c))
            v_parts_val.append(np.full(rows.size, float(weight), dtype=np.float32))

    h_idx = np.concatenate(h_parts_idx) if h_parts_idx else empty_i
    h_val = np.concatenate(h_parts_val) if h_parts_val else empty_v
    v_idx = np.concatenate(v_parts_idx) if v_parts_idx else empty_i
    v_val = np.concatenate(v_parts_val) if v_parts_val else empty_v
    if h_idx.size == 0:
        return v_idx, v_val, np.zeros(v_idx.size, dtype=np.float32)
    if v_idx.size == 0:
        return h_idx, np.zeros(h_idx.size, dtype=np.float32), h_val
    flat = np.concatenate([v_idx, h_idx])
    v_out = np.concatenate([v_val, np.zeros(h_idx.size, dtype=np.float32)])
    h_out = np.concatenate([np.zeros(v_idx.size, dtype=np.float32), h_val])
    return flat, v_out, h_out


def _net_routing_3pin_contrib(
    incremental_scorer,
    module_idx: int,
    cx: float,
    cy: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return 3-pin routing for nets touching one module."""
    empty_i = np.empty(0, dtype=np.int64)
    empty_v = np.empty(0, dtype=np.float32)
    nets = incremental_scorer.macro_to_nets.get(int(module_idx))
    if nets is None or len(nets) == 0:
        return empty_i, empty_v, empty_v

    lengths = incremental_scorer.net_lengths[nets]
    nets_3 = nets[lengths == 3]
    if len(nets_3) == 0:
        return empty_i, empty_v, empty_v

    starts = incremental_scorer.net_starts[nets_3]
    pin_indices = np.concatenate([starts, starts + 1, starts + 2])
    pos_cache = _ensure_pos_cache(incremental_scorer.plc)
    ref_local = incremental_scorer.ref_inv[pin_indices]
    refs = incremental_scorer.unique_ref[ref_local]
    pin_x = pos_cache[refs, 0] + incremental_scorer.x_off[pin_indices]
    pin_y = pos_cache[refs, 1] + incremental_scorer.y_off[pin_indices]
    moved = refs == int(module_idx)
    if moved.any():
        pin_x[moved] = float(cx) + incremental_scorer.x_off[pin_indices[moved]]
        pin_y[moved] = float(cy) + incremental_scorer.y_off[pin_indices[moved]]

    n3 = len(nets_3)
    grid_col = int(incremental_scorer.grid_col)
    grid_row = int(incremental_scorer.grid_row)
    grid_w = float(incremental_scorer.grid_w)
    grid_h = float(incremental_scorer.grid_h)
    cols = np.clip((pin_x / grid_w).astype(np.int64), 0, grid_col - 1)
    rows = np.clip((pin_y / grid_h).astype(np.int64), 0, grid_row - 1)
    g0 = rows[:n3] * grid_col + cols[:n3]
    g1 = rows[n3 : 2 * n3] * grid_col + cols[n3 : 2 * n3]
    g2 = rows[2 * n3 :] * grid_col + cols[2 * n3 :]
    weights = incremental_scorer.net_weights[nets_3].astype(np.float32, copy=False)

    eq01 = g0 == g1
    eq02 = g0 == g2
    eq12 = g1 == g2
    eq_count = eq01.astype(np.int64) + eq02.astype(np.int64) + eq12.astype(np.int64)
    uniq2 = eq_count == 1
    uniq3 = eq_count == 0

    h_rows: list[np.ndarray] = []
    h_los: list[np.ndarray] = []
    h_his: list[np.ndarray] = []
    h_ws: list[np.ndarray] = []
    v_cols: list[np.ndarray] = []
    v_los: list[np.ndarray] = []
    v_his: list[np.ndarray] = []
    v_ws: list[np.ndarray] = []

    if uniq2.any():
        src_2 = g0[uniq2]
        sink_2 = np.where(eq01[uniq2], g2[uniq2], g1[uniq2])
        w2 = weights[uniq2]
        src_row = src_2 // grid_col
        src_col = src_2 % grid_col
        snk_row = sink_2 // grid_col
        snk_col = sink_2 % grid_col
        h_rows.append(src_row)
        h_los.append(np.minimum(src_col, snk_col))
        h_his.append(np.maximum(src_col, snk_col))
        h_ws.append(w2)
        v_cols.append(snk_col)
        v_los.append(np.minimum(src_row, snk_row))
        v_his.append(np.maximum(src_row, snk_row))
        v_ws.append(w2)

    if uniq3.any():
        g0u = g0[uniq3]
        g1u = g1[uniq3]
        g2u = g2[uniq3]
        wu = weights[uniq3]
        y_all = np.stack([g0u // grid_col, g1u // grid_col, g2u // grid_col], axis=1).astype(
            np.int64
        )
        x_all = np.stack([g0u % grid_col, g1u % grid_col, g2u % grid_col], axis=1).astype(np.int64)
        big = int(max(grid_row, grid_col)) + 16
        order = np.argsort(x_all * big + y_all, axis=1, kind="stable")
        y = np.take_along_axis(y_all, order, axis=1)
        x = np.take_along_axis(x_all, order, axis=1)
        y1 = y[:, 0]
        y2 = y[:, 1]
        y3 = y[:, 2]
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        case1 = (x1 < x2) & (x2 < x3) & (np.minimum(y1, y3) < y2) & (np.maximum(y1, y3) > y2)
        case2 = (~case1) & (x2 == x3) & (x1 < x2) & (y1 < np.minimum(y2, y3))
        case3 = (~case1) & (~case2) & (y2 == y3)
        case4 = ~(case1 | case2 | case3)

        if case1.any():
            m = case1
            wm = wu[m]
            h_rows.append(y1[m])
            h_los.append(x1[m])
            h_his.append(x2[m])
            h_ws.append(wm)
            h_rows.append(y2[m])
            h_los.append(x2[m])
            h_his.append(x3[m])
            h_ws.append(wm)
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
            wm = wu[m]
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
            wm = wu[m]
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
            wm = wu[m]
            y_t = y_all[m]
            x_t = x_all[m]
            order_t = np.argsort(y_t * big + x_t, axis=1, kind="stable")
            y_t = np.take_along_axis(y_t, order_t, axis=1)
            x_t = np.take_along_axis(x_t, order_t, axis=1)
            y1t = y_t[:, 0]
            y2t = y_t[:, 1]
            y3t = y_t[:, 2]
            x1t = x_t[:, 0]
            x2t = x_t[:, 1]
            x3t = x_t[:, 2]
            h_rows.append(y2t)
            h_los.append(np.minimum(np.minimum(x1t, x2t), x3t))
            h_his.append(np.maximum(np.maximum(x1t, x2t), x3t))
            h_ws.append(wm)
            v_cols.append(x1t)
            v_los.append(np.minimum(y1t, y2t))
            v_his.append(np.maximum(y1t, y2t))
            v_ws.append(wm)
            v_cols.append(x3t)
            v_los.append(np.minimum(y2t, y3t))
            v_his.append(np.maximum(y2t, y3t))
            v_ws.append(wm)

    return _sparse_from_route_strips(
        grid_col=grid_col,
        h_rows=h_rows,
        h_los=h_los,
        h_his=h_his,
        h_ws=h_ws,
        v_cols=v_cols,
        v_los=v_los,
        v_his=v_his,
        v_ws=v_ws,
    )


def _append_2pin_strips(
    *,
    src_flat: np.ndarray,
    snk_flat: np.ndarray,
    weights: np.ndarray,
    grid_col: int,
    h_rows: list[np.ndarray],
    h_los: list[np.ndarray],
    h_his: list[np.ndarray],
    h_ws: list[np.ndarray],
    v_cols: list[np.ndarray],
    v_los: list[np.ndarray],
    v_his: list[np.ndarray],
    v_ws: list[np.ndarray],
) -> None:
    if src_flat.size == 0:
        return
    src_row = src_flat // grid_col
    src_col = src_flat % grid_col
    snk_row = snk_flat // grid_col
    snk_col = snk_flat % grid_col
    h_rows.append(src_row)
    h_los.append(np.minimum(src_col, snk_col))
    h_his.append(np.maximum(src_col, snk_col))
    h_ws.append(weights)
    v_cols.append(snk_col)
    v_los.append(np.minimum(src_row, snk_row))
    v_his.append(np.maximum(src_row, snk_row))
    v_ws.append(weights)


def _net_routing_highfanout_contrib(
    incremental_scorer,
    module_idx: int,
    cx: float,
    cy: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return high-fanout routing for nets touching one module."""
    empty_i = np.empty(0, dtype=np.int64)
    empty_v = np.empty(0, dtype=np.float32)
    nets = incremental_scorer.macro_to_nets.get(int(module_idx))
    if nets is None or len(nets) == 0:
        return empty_i, empty_v, empty_v

    lengths_all = incremental_scorer.net_lengths[nets]
    nets_big = nets[lengths_all >= 4]
    if len(nets_big) == 0:
        return empty_i, empty_v, empty_v

    pos_cache = _ensure_pos_cache(incremental_scorer.plc)
    grid_col = int(incremental_scorer.grid_col)
    grid_row = int(incremental_scorer.grid_row)
    grid_w = float(incremental_scorer.grid_w)
    grid_h = float(incremental_scorer.grid_h)

    h_rows: list[np.ndarray] = []
    h_los: list[np.ndarray] = []
    h_his: list[np.ndarray] = []
    h_ws: list[np.ndarray] = []
    v_cols: list[np.ndarray] = []
    v_los: list[np.ndarray] = []
    v_his: list[np.ndarray] = []
    v_ws: list[np.ndarray] = []
    g0_3_parts = []
    g1_3_parts = []
    g2_3_parts = []
    w3_parts = []

    for net in nets_big:
        start = int(incremental_scorer.net_starts[net])
        length = int(incremental_scorer.net_lengths[net])
        pin_indices = np.arange(start, start + length, dtype=np.int64)
        ref_local = incremental_scorer.ref_inv[pin_indices]
        refs = incremental_scorer.unique_ref[ref_local]
        pin_x = pos_cache[refs, 0] + incremental_scorer.x_off[pin_indices]
        pin_y = pos_cache[refs, 1] + incremental_scorer.y_off[pin_indices]
        moved = refs == int(module_idx)
        if moved.any():
            pin_x[moved] = float(cx) + incremental_scorer.x_off[pin_indices[moved]]
            pin_y[moved] = float(cy) + incremental_scorer.y_off[pin_indices[moved]]
        pin_col = np.clip((pin_x / grid_w).astype(np.int64), 0, grid_col - 1)
        pin_row = np.clip((pin_y / grid_h).astype(np.int64), 0, grid_row - 1)
        pin_gcell = pin_row * grid_col + pin_col
        src = int(pin_gcell[0])
        sinks = pin_gcell[1:]
        sinks = np.unique(sinks[sinks != src])
        if sinks.size == 0:
            continue
        weight = np.asarray([incremental_scorer.net_weights[net]], dtype=np.float32)
        if sinks.size == 2:
            g0_3_parts.append(np.asarray([src], dtype=np.int64))
            g1_3_parts.append(np.asarray([int(sinks[0])], dtype=np.int64))
            g2_3_parts.append(np.asarray([int(sinks[1])], dtype=np.int64))
            w3_parts.append(weight)
        else:
            src_flat = np.full(sinks.size, src, dtype=np.int64)
            snk_flat = sinks.astype(np.int64, copy=False)
            weights = np.full(sinks.size, float(weight[0]), dtype=np.float32)
            _append_2pin_strips(
                src_flat=src_flat,
                snk_flat=snk_flat,
                weights=weights,
                grid_col=grid_col,
                h_rows=h_rows,
                h_los=h_los,
                h_his=h_his,
                h_ws=h_ws,
                v_cols=v_cols,
                v_los=v_los,
                v_his=v_his,
                v_ws=v_ws,
            )

    if g0_3_parts:
        g0 = np.concatenate(g0_3_parts)
        g1 = np.concatenate(g1_3_parts)
        g2 = np.concatenate(g2_3_parts)
        weights = np.concatenate(w3_parts)
        y_all = np.stack([g0 // grid_col, g1 // grid_col, g2 // grid_col], axis=1).astype(np.int64)
        x_all = np.stack([g0 % grid_col, g1 % grid_col, g2 % grid_col], axis=1).astype(np.int64)
        big = int(max(grid_row, grid_col)) + 16
        order = np.argsort(x_all * big + y_all, axis=1, kind="stable")
        y = np.take_along_axis(y_all, order, axis=1)
        x = np.take_along_axis(x_all, order, axis=1)
        y1 = y[:, 0]
        y2 = y[:, 1]
        y3 = y[:, 2]
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        case1 = (x1 < x2) & (x2 < x3) & (np.minimum(y1, y3) < y2) & (np.maximum(y1, y3) > y2)
        case2 = (~case1) & (x2 == x3) & (x1 < x2) & (y1 < np.minimum(y2, y3))
        case3 = (~case1) & (~case2) & (y2 == y3)
        case4 = ~(case1 | case2 | case3)
        if case1.any():
            m = case1
            wm = weights[m]
            h_rows.append(y1[m])
            h_los.append(x1[m])
            h_his.append(x2[m])
            h_ws.append(wm)
            h_rows.append(y2[m])
            h_los.append(x2[m])
            h_his.append(x3[m])
            h_ws.append(wm)
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
            wm = weights[m]
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
            wm = weights[m]
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
            wm = weights[m]
            y_t = y_all[m]
            x_t = x_all[m]
            order_t = np.argsort(y_t * big + x_t, axis=1, kind="stable")
            y_t = np.take_along_axis(y_t, order_t, axis=1)
            x_t = np.take_along_axis(x_t, order_t, axis=1)
            y1t = y_t[:, 0]
            y2t = y_t[:, 1]
            y3t = y_t[:, 2]
            x1t = x_t[:, 0]
            x2t = x_t[:, 1]
            x3t = x_t[:, 2]
            h_rows.append(y2t)
            h_los.append(np.minimum(np.minimum(x1t, x2t), x3t))
            h_his.append(np.maximum(np.maximum(x1t, x2t), x3t))
            h_ws.append(wm)
            v_cols.append(x1t)
            v_los.append(np.minimum(y1t, y2t))
            v_his.append(np.maximum(y1t, y2t))
            v_ws.append(wm)
            v_cols.append(x3t)
            v_los.append(np.minimum(y2t, y3t))
            v_his.append(np.maximum(y2t, y3t))
            v_ws.append(wm)

    return _sparse_from_route_strips(
        grid_col=grid_col,
        h_rows=h_rows,
        h_los=h_los,
        h_his=h_his,
        h_ws=h_ws,
        v_cols=v_cols,
        v_los=v_los,
        v_his=v_his,
        v_ws=v_ws,
    )


def _smooth_routing_batch_torch(
    grid_flat: torch.Tensor,
    *,
    grid_row: int,
    grid_col: int,
    smooth_range: int,
    axis_h: bool,
) -> torch.Tensor:
    """Torch batch equivalent of `_smooth_routing_cong_vec`."""
    grid_3d = grid_flat.reshape(grid_flat.shape[0], grid_row, grid_col)
    sr = int(smooth_range)
    dev = grid_flat.device
    dtype = grid_flat.dtype
    if sr <= 0:
        return grid_3d.reshape(grid_flat.shape[0], grid_row * grid_col)
    if axis_h:
        rows = torch.arange(grid_row, dtype=torch.long, device=dev)
        cnts = (torch.clamp(rows + sr, max=grid_row - 1) - torch.clamp(rows - sr, min=0) + 1).to(
            dtype
        )
        weighted = grid_3d / cnts.view(1, grid_row, 1)
        zero = torch.zeros(grid_flat.shape[0], 1, grid_col, dtype=dtype, device=dev)
        cs = torch.cumsum(torch.cat([zero, weighted], dim=1), dim=1)
        lo = torch.clamp(rows - sr, min=0)
        hi = torch.clamp(rows + sr + 1, max=grid_row)
        smoothed = cs[:, hi, :] - cs[:, lo, :]
    else:
        cols = torch.arange(grid_col, dtype=torch.long, device=dev)
        cnts = (torch.clamp(cols + sr, max=grid_col - 1) - torch.clamp(cols - sr, min=0) + 1).to(
            dtype
        )
        weighted = grid_3d / cnts.view(1, 1, grid_col)
        zero = torch.zeros(grid_flat.shape[0], grid_row, 1, dtype=dtype, device=dev)
        cs = torch.cumsum(torch.cat([zero, weighted], dim=2), dim=2)
        lo = torch.clamp(cols - sr, min=0)
        hi = torch.clamp(cols + sr + 1, max=grid_col)
        smoothed = cs[:, :, hi] - cs[:, :, lo]
    return smoothed.reshape(grid_flat.shape[0], grid_row * grid_col)


def _is_torch_oom(exc: BaseException) -> bool:
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _cuda_memory_stats() -> dict:
    if _GPU_DEVICE.type != "cuda":
        return {}
    try:
        dev = _GPU_DEVICE
        return {
            "memory_allocated": int(torch.cuda.memory_allocated(dev)),
            "memory_reserved": int(torch.cuda.memory_reserved(dev)),
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(dev)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(dev)),
        }
    except Exception:
        return {}


def _cuda_runtime_status() -> dict:
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    device_count = 0
    if cuda_available:
        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            device_count = 0
    return {
        "configured_backend": _GPU_BACKEND,
        "requested_device": _CUDA_DEVICE_REQUESTED,
        "configured_device_name": _GPU_DEVICE_NAME,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": device_count,
        "torch_cuda_version": torch.version.cuda,
    }


def _cuda_synchronize_if_needed() -> None:
    if _GPU_DEVICE.type != "cuda":
        return
    try:
        torch.cuda.synchronize(_GPU_DEVICE)
    except Exception:
        pass


def _relocation_hpwl_dynamic_bytes_per_proposal(
    incremental_scorer,
    proposals: "list[dict] | None",
) -> int:
    if (
        not proposals
        or incremental_scorer is None
        or not hasattr(incremental_scorer, "hard_indices")
    ):
        return 0
    int64_bytes = np.dtype(np.int64).itemsize
    float32_bytes = np.dtype(np.float32).itemsize
    bool_bytes = np.dtype(np.bool_).itemsize
    total = 0
    module_indices = {
        (
            int(p["module_idx"])
            if "module_idx" in p
            else int(incremental_scorer.hard_indices[int(p["i"])])
        )
        for p in proposals
    }
    hpwl_topology = _build_hpwl_topology_cache(
        incremental_scorer,
        module_indices=module_indices,
    )
    for proposal in proposals:
        if "module_idx" in proposal:
            module_idx = int(proposal["module_idx"])
        else:
            module_idx = int(incremental_scorer.hard_indices[int(proposal["i"])])
        topo = hpwl_topology.get(module_idx)
        if topo is None:
            continue
        nets, _lengths, pin_indices, _total = topo
        n_pins = int(pin_indices.size)
        n_segments = int(nets.size)
        # Static topology tensors are counted elsewhere.
        total += n_pins * (5 * int64_bytes + 2 * float32_bytes + bool_bytes)
        total += n_segments * (2 * int64_bytes + 6 * float32_bytes)
    return int(np.ceil(total / max(len(proposals), 1)))


def _relocation_grid_dynamic_bytes_per_proposal(incremental_scorer) -> int:
    if incremental_scorer is None:
        return 0
    n_cells = int(incremental_scorer.grid_row) * int(incremental_scorer.grid_col)
    # Scratch grids used while scoring one proposal.
    return max(1, n_cells) * 10 * np.dtype(np.float32).itemsize


def _relocation_dynamic_bytes_per_proposal(
    incremental_scorer,
    proposals: "list[dict] | None" = None,
) -> int:
    if incremental_scorer is None:
        return 0
    raw_bytes = _relocation_grid_dynamic_bytes_per_proposal(incremental_scorer)
    raw_bytes += _relocation_hpwl_dynamic_bytes_per_proposal(incremental_scorer, proposals)
    return int(np.ceil(raw_bytes * _relocation_memory_safety_factor()))


def _relocation_dynamic_byte_components(
    incremental_scorer,
    proposals: "list[dict] | None" = None,
) -> dict:
    factor = _relocation_memory_safety_factor()
    grid_raw = _relocation_grid_dynamic_bytes_per_proposal(incremental_scorer)
    hpwl_raw = _relocation_hpwl_dynamic_bytes_per_proposal(incremental_scorer, proposals)
    return {
        "grid_dynamic_bytes_per_proposal": int(np.ceil(grid_raw * factor)),
        "hpwl_dynamic_bytes_per_proposal": int(np.ceil(hpwl_raw * factor)),
    }


def _relocation_memory_safety_factor() -> float:
    try:
        factor = float(
            os.environ.get("RELOC_PROPOSE_MEM_SAFETY", str(const.RELOC_PROPOSE_MEM_SAFETY))
        )
    except ValueError:
        return 1.0
    return max(1.0, factor)


def _relocation_static_tensor_bytes_estimate(
    incremental_scorer,
    proposals: "list[dict] | None" = None,
) -> int:
    if incremental_scorer is None:
        return 0
    n_cells = int(incremental_scorer.grid_row) * int(incremental_scorer.grid_col)
    total = max(1, n_cells) * 5 * np.dtype(np.float32).itemsize
    for name, dtype in (
        ("unique_ref", np.int64),
        ("ref_inv", np.int64),
        ("x_off", np.float32),
        ("y_off", np.float32),
        ("per_net_hpwl", np.float32),
        ("net_weights", np.float32),
    ):
        arr = getattr(incremental_scorer, name, None)
        if arr is not None:
            total += int(np.asarray(arr).size) * np.dtype(dtype).itemsize
    try:
        pos_cache = _ensure_pos_cache(incremental_scorer.plc)
    except Exception:
        pos_cache = None
    if pos_cache is not None:
        total += int(np.asarray(pos_cache).size) * np.dtype(np.float32).itemsize
    if proposals is not None:
        total += len(proposals) * 2 * np.dtype(np.float32).itemsize
    module_indices = None
    if proposals is not None and hasattr(incremental_scorer, "hard_indices"):
        module_indices = {
            (
                int(p["module_idx"])
                if "module_idx" in p
                else int(incremental_scorer.hard_indices[int(p["i"])])
            )
            for p in proposals
        }
    hpwl_topology = _build_hpwl_topology_cache(
        incremental_scorer,
        module_indices=module_indices,
    )
    for nets, lengths, pin_indices, _total in hpwl_topology.values():
        total += int(nets.size) * np.dtype(np.int64).itemsize
        total += int(lengths.size) * np.dtype(np.int64).itemsize
        total += int(pin_indices.size) * np.dtype(np.int64).itemsize
    return int(total)


def _tensor_tree_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_tree_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_bytes(v) for v in value)
    return 0


def _chunk_size_from_memory_budget(
    proposals: list[dict],
    incremental_scorer,
    max_mb_raw: str,
    static_bytes_estimate: int = 0,
) -> "tuple[int | None, float | None, int, int | None, int | None]":
    try:
        max_mb = float(max_mb_raw)
    except (TypeError, ValueError):
        return (
            None,
            None,
            _relocation_dynamic_bytes_per_proposal(incremental_scorer, proposals),
            None,
            None,
        )
    bytes_per = _relocation_dynamic_bytes_per_proposal(incremental_scorer, proposals)
    if bytes_per <= 0:
        return None, max_mb, bytes_per, None, None
    budget = int(max_mb * 1024.0 * 1024.0)
    if budget <= 0:
        return None, max_mb, bytes_per, None, None
    dynamic_budget = max(0, budget - max(0, int(static_bytes_estimate)))
    chunk = max(1, min(len(proposals), dynamic_budget // bytes_per))
    return int(chunk), max_mb, bytes_per, int(dynamic_budget), int(budget)


def _cuda_auto_memory_budget_mb(frac_raw: str) -> "tuple[float, float, int, int] | None":
    if _GPU_DEVICE.type != "cuda":
        return None
    try:
        frac = float(frac_raw)
    except (TypeError, ValueError):
        return None
    if frac <= 0.0:
        return None
    frac = min(frac, 1.0)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(_GPU_DEVICE)
    except Exception:
        return None
    budget = float(free_bytes) * frac / (1024.0 * 1024.0)
    return (budget, frac, int(free_bytes), int(total_bytes)) if budget > 0.0 else None


def _build_hpwl_topology_cache(
    incremental_scorer,
    module_indices: "set[int] | None" = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    macro_to_nets = getattr(incremental_scorer, "macro_to_nets", None)
    if not macro_to_nets:
        return cache
    for module_idx, nets in macro_to_nets.items():
        module_idx = int(module_idx)
        if module_indices is not None and module_idx not in module_indices:
            continue
        if nets is None or len(nets) == 0:
            continue
        starts_np = incremental_scorer.net_starts[nets]
        lengths_np = incremental_scorer.net_lengths[nets]
        total = int(lengths_np.sum())
        if total == 0:
            continue
        local_starts_np = np.concatenate([[0], np.cumsum(lengths_np)[:-1]])
        pin_indices_np = np.repeat(starts_np, lengths_np) + (
            np.arange(total) - np.repeat(local_starts_np, lengths_np)
        )
        cache[module_idx] = (
            nets.astype(np.int64, copy=False),
            lengths_np.astype(np.int64, copy=False),
            pin_indices_np.astype(np.int64, copy=False),
            total,
        )
    return cache


def _build_hpwl_topology_tensor_cache(
    hpwl_topology: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    *,
    dev: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
    for module_idx, (nets, lengths, pin_indices, total) in hpwl_topology.items():
        cache[module_idx] = (
            torch.as_tensor(nets, device=dev, dtype=torch.long),
            torch.as_tensor(lengths, device=dev, dtype=torch.long),
            torch.as_tensor(pin_indices, device=dev, dtype=torch.long),
            int(total),
        )
    return cache


def _net_routing_all_contrib(
    incremental_scorer,
    module_idx: int,
    cx: float,
    cy: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    parts = []
    for helper in (
        _net_routing_2pin_contrib,
        _net_routing_3pin_contrib,
        _net_routing_highfanout_contrib,
    ):
        parts.append(helper(incremental_scorer, module_idx, cx, cy))
    idx_parts = [part[0] for part in parts if part[0].size]
    v_parts = [part[1] for part in parts if part[0].size]
    h_parts = [part[2] for part in parts if part[0].size]
    if idx_parts:
        return (
            np.concatenate(idx_parts),
            np.concatenate(v_parts),
            np.concatenate(h_parts),
        )
    empty_i = np.empty(0, dtype=np.int64)
    empty_v = np.empty(0, dtype=np.float32)
    return empty_i, empty_v, empty_v


def _build_relocation_cuda_static_tensors(
    proposals: list[dict],
    *,
    pos: np.ndarray,
    incremental_scorer,
    dev: torch.device,
) -> dict:
    """Build GPU tensors reused by every proposal chunk."""
    pos_cache_np = _ensure_pos_cache(incremental_scorer.plc)
    proposal_xy_np = np.asarray([p["xy"] for p in proposals], dtype=np.float32)
    proposal_i_np = np.asarray([p["i"] for p in proposals], dtype=np.int64)
    proposal_modules = []
    for p, i in zip(proposals, proposal_i_np):
        if "module_idx" in p:
            proposal_modules.append(int(p["module_idx"]))
        else:
            proposal_modules.append(int(incremental_scorer.hard_indices[int(i)]))
    proposal_module_idx_np = np.asarray(proposal_modules, dtype=np.int64)
    old_density: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    old_macro_route: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    old_net_route: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    module_indices: set[int] = set()
    for p, i_raw, module_idx_raw in zip(proposals, proposal_i_np, proposal_module_idx_np):
        i = int(i_raw)
        module_idx = int(module_idx_raw)
        if i in old_density:
            continue
        module_indices.add(module_idx)
        old_xy = p.get("old_xy")
        if old_xy is None:
            old_x = float(pos[i, 0])
            old_y = float(pos[i, 1])
        else:
            old_x = float(old_xy[0])
            old_y = float(old_xy[1])
        old_density[i] = incremental_scorer._macro_occ(module_idx, old_x, old_y)
        if p.get("macro_route", True):
            old_macro_route[i] = _macro_routing_contrib(
                incremental_scorer,
                module_idx,
                old_x,
                old_y,
            )
        else:
            empty_i = np.empty(0, dtype=np.int64)
            empty_v = np.empty(0, dtype=np.float32)
            old_macro_route[i] = (empty_i, empty_v, empty_v)
        old_net_route[i] = _net_routing_all_contrib(
            incremental_scorer,
            module_idx,
            old_x,
            old_y,
        )
    hpwl_topology = _build_hpwl_topology_cache(
        incremental_scorer,
        module_indices=module_indices,
    )
    return {
        "hpwl_topology": hpwl_topology,
        "hpwl_topology_tensors": _build_hpwl_topology_tensor_cache(
            hpwl_topology,
            dev=dev,
        ),
        "proposal_xy_np": proposal_xy_np,
        "proposal_i_np": proposal_i_np,
        "proposal_module_idx_np": proposal_module_idx_np,
        "proposal_xy": torch.as_tensor(proposal_xy_np, device=dev, dtype=torch.float32),
        "old_density": old_density,
        "old_macro_route": old_macro_route,
        "old_net_route": old_net_route,
        "base_occ": torch.as_tensor(
            incremental_scorer.grid_occupied,
            device=dev,
            dtype=torch.float32,
        ),
        "v_macro_base": torch.as_tensor(
            incremental_scorer.V_macro_flat,
            device=dev,
            dtype=torch.float32,
        ),
        "h_macro_base": torch.as_tensor(
            incremental_scorer.H_macro_flat,
            device=dev,
            dtype=torch.float32,
        ),
        "v_raw_base": torch.as_tensor(
            incremental_scorer.V_flat,
            device=dev,
            dtype=torch.float32,
        ),
        "h_raw_base": torch.as_tensor(
            incremental_scorer.H_flat,
            device=dev,
            dtype=torch.float32,
        ),
        "pos_cache": (
            torch.as_tensor(pos_cache_np, device=dev, dtype=torch.float32)
            if pos_cache_np is not None
            else None
        ),
        "unique_ref": torch.as_tensor(
            incremental_scorer.unique_ref,
            device=dev,
            dtype=torch.long,
        ),
        "ref_inv": torch.as_tensor(
            incremental_scorer.ref_inv,
            device=dev,
            dtype=torch.long,
        ),
        "x_off": torch.as_tensor(
            incremental_scorer.x_off,
            device=dev,
            dtype=torch.float32,
        ),
        "y_off": torch.as_tensor(
            incremental_scorer.y_off,
            device=dev,
            dtype=torch.float32,
        ),
        "per_net_hpwl": torch.as_tensor(
            incremental_scorer.per_net_hpwl,
            device=dev,
            dtype=torch.float32,
        ),
        "net_weights": torch.as_tensor(
            incremental_scorer.net_weights,
            device=dev,
            dtype=torch.float32,
        ),
    }


def _batched_hpwl_delta_torch(
    proposals: list[dict],
    *,
    module_idx_np: np.ndarray,
    nx_t: torch.Tensor,
    ny_t: torch.Tensor,
    incremental_scorer,
    dev: torch.device,
    static_tensors: dict,
) -> "tuple[torch.Tensor, dict]":
    """Return per-proposal HPWL delta via one segmented Torch reduction."""
    n_prop = len(proposals)
    wl_delta = torch.zeros(n_prop, device=dev, dtype=torch.float32)
    pos_cache_t = static_tensors.get("pos_cache")
    if pos_cache_t is None or incremental_scorer.n_nets <= 0:
        return wl_delta, {"hpwl_segments": 0, "hpwl_pins": 0, "hpwl_rows": 0}

    pin_parts: list[torch.Tensor] = []
    seg_parts: list[torch.Tensor] = []
    row_pin_parts: list[torch.Tensor] = []
    moved_module_parts: list[torch.Tensor] = []
    nets_parts: list[torch.Tensor] = []
    row_seg_parts: list[torch.Tensor] = []
    seg_offset = 0
    for row, module_idx_raw in enumerate(module_idx_np):
        module_idx = int(module_idx_raw)
        topo = static_tensors["hpwl_topology_tensors"].get(module_idx)
        if topo is None:
            continue
        nets_t_part, lengths_t, pin_indices_t_part, total = topo
        n_seg = int(nets_t_part.numel())
        seg_template = torch.arange(
            seg_offset,
            seg_offset + n_seg,
            device=dev,
            dtype=torch.long,
        )
        pin_parts.append(pin_indices_t_part)
        seg_parts.append(torch.repeat_interleave(seg_template, lengths_t))
        row_pin_parts.append(torch.full((total,), row, device=dev, dtype=torch.long))
        moved_module_parts.append(torch.full((total,), module_idx, device=dev, dtype=torch.long))
        nets_parts.append(nets_t_part)
        row_seg_parts.append(torch.full((n_seg,), row, device=dev, dtype=torch.long))
        seg_offset += n_seg

    if not pin_parts:
        return wl_delta, {"hpwl_segments": 0, "hpwl_pins": 0, "hpwl_rows": 0}

    pin_indices_t = torch.cat(pin_parts)
    seg_ids_t = torch.cat(seg_parts)
    row_pin_t = torch.cat(row_pin_parts)
    moved_module_t = torch.cat(moved_module_parts)
    nets_t = torch.cat(nets_parts)
    row_seg_t = torch.cat(row_seg_parts)

    unique_ref_t = static_tensors["unique_ref"]
    node_pos_t = pos_cache_t[unique_ref_t]
    ref_inv_t = static_tensors["ref_inv"]
    x_off_t = static_tensors["x_off"]
    y_off_t = static_tensors["y_off"]
    ref_local = ref_inv_t[pin_indices_t]
    refs = unique_ref_t[ref_local]

    pin_x = node_pos_t[ref_local, 0] + x_off_t[pin_indices_t]
    pin_y = node_pos_t[ref_local, 1] + y_off_t[pin_indices_t]
    moved_mask = refs == moved_module_t
    if torch.any(moved_mask):
        pin_x = torch.where(moved_mask, nx_t[row_pin_t] + x_off_t[pin_indices_t], pin_x)
        pin_y = torch.where(moved_mask, ny_t[row_pin_t] + y_off_t[pin_indices_t], pin_y)

    n_seg = int(nets_t.numel())
    min_x = torch.full((n_seg,), float("inf"), device=dev, dtype=torch.float32)
    max_x = torch.full((n_seg,), float("-inf"), device=dev, dtype=torch.float32)
    min_y = torch.full((n_seg,), float("inf"), device=dev, dtype=torch.float32)
    max_y = torch.full((n_seg,), float("-inf"), device=dev, dtype=torch.float32)
    min_x.scatter_reduce_(0, seg_ids_t, pin_x, reduce="amin", include_self=True)
    max_x.scatter_reduce_(0, seg_ids_t, pin_x, reduce="amax", include_self=True)
    min_y.scatter_reduce_(0, seg_ids_t, pin_y, reduce="amin", include_self=True)
    max_y.scatter_reduce_(0, seg_ids_t, pin_y, reduce="amax", include_self=True)

    per_net_hpwl_t = static_tensors["per_net_hpwl"]
    net_weights_t = static_tensors["net_weights"]
    new_hpwl = (max_x - min_x) + (max_y - min_y)
    contrib = (new_hpwl - per_net_hpwl_t[nets_t]) * net_weights_t[nets_t]
    wl_delta.index_add_(0, row_seg_t, contrib)
    return wl_delta / float(incremental_scorer.wl_normalizer), {
        "hpwl_segments": int(nets_t.numel()),
        "hpwl_pins": int(pin_indices_t.numel()),
        "hpwl_rows": len(pin_parts),
    }


def _score_relocation_proposals_cuda_delta_batch(
    proposals: list[dict],
    *,
    proposal_start: int,
    incremental_scorer,
    static_tensors: dict,
) -> dict:
    """Score frozen proposals in one tensor batch."""
    if not proposals:
        return {}
    t0_prep = time.monotonic()
    dev = _GPU_DEVICE
    n_prop = len(proposals)
    proposal_end = proposal_start + n_prop
    xy_np = static_tensors["proposal_xy_np"][proposal_start:proposal_end]
    hard_i_np = static_tensors["proposal_i_np"][proposal_start:proposal_end]
    module_idx_np = static_tensors["proposal_module_idx_np"][proposal_start:proposal_end]

    row_parts = []
    col_parts = []
    val_parts = []
    macro_route_row_parts = []
    macro_route_col_parts = []
    macro_route_v_parts = []
    macro_route_h_parts = []
    net_route_row_parts = []
    net_route_col_parts = []
    net_route_v_parts = []
    net_route_h_parts = []
    for row, (proposal, i_raw, module_idx_raw) in enumerate(
        zip(proposals, hard_i_np, module_idx_np)
    ):
        i = int(i_raw)
        module_idx = int(module_idx_raw)
        nx = float(xy_np[row, 0])
        ny = float(xy_np[row, 1])
        old_idx, old_area = static_tensors["old_density"][i]
        new_idx, new_area = incremental_scorer._macro_occ(
            module_idx,
            nx,
            ny,
        )
        if old_idx.size:
            row_parts.append(np.full(old_idx.size, row, dtype=np.int64))
            col_parts.append(old_idx.astype(np.int64, copy=False))
            val_parts.append(-old_area.astype(np.float32, copy=False))
        if new_idx.size:
            row_parts.append(np.full(new_idx.size, row, dtype=np.int64))
            col_parts.append(new_idx.astype(np.int64, copy=False))
            val_parts.append(new_area.astype(np.float32, copy=False))

        old_route_idx, old_v, old_h = static_tensors["old_macro_route"][i]
        if proposal.get("macro_route", True):
            new_route_idx, new_v, new_h = _macro_routing_contrib(
                incremental_scorer,
                module_idx,
                nx,
                ny,
            )
        else:
            new_route_idx = np.empty(0, dtype=np.int64)
            new_v = np.empty(0, dtype=np.float32)
            new_h = np.empty(0, dtype=np.float32)
        if old_route_idx.size:
            macro_route_row_parts.append(np.full(old_route_idx.size, row, dtype=np.int64))
            macro_route_col_parts.append(old_route_idx)
            macro_route_v_parts.append(-old_v)
            macro_route_h_parts.append(-old_h)
        if new_route_idx.size:
            macro_route_row_parts.append(np.full(new_route_idx.size, row, dtype=np.int64))
            macro_route_col_parts.append(new_route_idx)
            macro_route_v_parts.append(new_v)
            macro_route_h_parts.append(new_h)

        old_net_idx, old_net_v, old_net_h = static_tensors["old_net_route"][i]
        new_net_idx, new_net_v, new_net_h = _net_routing_all_contrib(
            incremental_scorer,
            module_idx,
            nx,
            ny,
        )
        if old_net_idx.size:
            net_route_row_parts.append(np.full(old_net_idx.size, row, dtype=np.int64))
            net_route_col_parts.append(old_net_idx)
            net_route_v_parts.append(-old_net_v)
            net_route_h_parts.append(-old_net_h)
        if new_net_idx.size:
            net_route_row_parts.append(np.full(new_net_idx.size, row, dtype=np.int64))
            net_route_col_parts.append(new_net_idx)
            net_route_v_parts.append(new_net_v)
            net_route_h_parts.append(new_net_h)

    prep_elapsed = time.monotonic() - t0_prep
    with torch.inference_mode():
        occ = static_tensors["base_occ"].unsqueeze(0).repeat(n_prop, 1)
        if row_parts:
            rows = torch.as_tensor(np.concatenate(row_parts), device=dev, dtype=torch.long)
            cols = torch.as_tensor(np.concatenate(col_parts), device=dev, dtype=torch.long)
            vals = torch.as_tensor(np.concatenate(val_parts), device=dev, dtype=torch.float32)
            occ.index_put_((rows, cols), vals, accumulate=True)

        nz = torch.clamp(occ, min=0.0)
        cnt = int(incremental_scorer.dens_density_cnt)
        if incremental_scorer.dens_n_cells < 10:
            density = 0.5 * (nz.sum(dim=1) / torch.clamp((nz != 0.0).sum(dim=1), min=1))
            density = density / float(incremental_scorer.dens_grid_area)
        else:
            k = max(1, min(cnt, int(nz.shape[1])))
            top = torch.topk(nz, k=k, dim=1).values
            density = 0.5 * top.sum(dim=1) / float(incremental_scorer.dens_grid_area) / float(cnt)

        v_macro = static_tensors["v_macro_base"].unsqueeze(0).repeat(n_prop, 1)
        h_macro = static_tensors["h_macro_base"].unsqueeze(0).repeat(n_prop, 1)
        if macro_route_row_parts:
            route_rows = torch.as_tensor(
                np.concatenate(macro_route_row_parts),
                device=dev,
                dtype=torch.long,
            )
            route_cols = torch.as_tensor(
                np.concatenate(macro_route_col_parts),
                device=dev,
                dtype=torch.long,
            )
            route_v = torch.as_tensor(
                np.concatenate(macro_route_v_parts),
                device=dev,
                dtype=torch.float32,
            )
            route_h = torch.as_tensor(
                np.concatenate(macro_route_h_parts),
                device=dev,
                dtype=torch.float32,
            )
            v_macro.index_put_((route_rows, route_cols), route_v, accumulate=True)
            h_macro.index_put_((route_rows, route_cols), route_h, accumulate=True)

        v_raw = static_tensors["v_raw_base"].unsqueeze(0).repeat(n_prop, 1)
        h_raw = static_tensors["h_raw_base"].unsqueeze(0).repeat(n_prop, 1)
        if net_route_row_parts:
            net_rows = torch.as_tensor(
                np.concatenate(net_route_row_parts),
                device=dev,
                dtype=torch.long,
            )
            net_cols = torch.as_tensor(
                np.concatenate(net_route_col_parts),
                device=dev,
                dtype=torch.long,
            )
            net_v = torch.as_tensor(
                np.concatenate(net_route_v_parts),
                device=dev,
                dtype=torch.float32,
            )
            net_h = torch.as_tensor(
                np.concatenate(net_route_h_parts),
                device=dev,
                dtype=torch.float32,
            )
            v_raw.index_put_((net_rows, net_cols), net_v, accumulate=True)
            h_raw.index_put_((net_rows, net_cols), net_h, accumulate=True)

        v_smoothed = _smooth_routing_batch_torch(
            v_raw / float(incremental_scorer.grid_v_routes),
            grid_row=int(incremental_scorer.grid_row),
            grid_col=int(incremental_scorer.grid_col),
            smooth_range=int(incremental_scorer.smooth_range),
            axis_h=False,
        )
        h_smoothed = _smooth_routing_batch_torch(
            h_raw / float(incremental_scorer.grid_h_routes),
            grid_row=int(incremental_scorer.grid_row),
            grid_col=int(incremental_scorer.grid_col),
            smooth_range=int(incremental_scorer.smooth_range),
            axis_h=True,
        )
        v_cong = v_smoothed + v_macro / float(incremental_scorer.grid_v_routes)
        h_cong = h_smoothed + h_macro / float(incremental_scorer.grid_h_routes)
        routing_vals = torch.cat([v_cong, h_cong], dim=1)
        route_cnt = int(routing_vals.shape[1] * 0.05)
        if route_cnt <= 0:
            congestion = torch.max(routing_vals, dim=1).values
        else:
            congestion = torch.topk(routing_vals, k=route_cnt, dim=1).values.mean(dim=1)

        xy_t = static_tensors["proposal_xy"][proposal_start:proposal_end]
        nx_t = xy_t[:, 0]
        ny_t = xy_t[:, 1]

        wl_delta, hpwl_stats = _batched_hpwl_delta_torch(
            proposals,
            module_idx_np=module_idx_np,
            nx_t=nx_t,
            ny_t=ny_t,
            incremental_scorer=incremental_scorer,
            dev=dev,
            static_tensors=static_tensors,
        )

        # Lower is better. This is the same proxy objective used by
        # IncrementalScorer._trial_at, evaluated in a batched tensor path.
        wl_base = float(incremental_scorer.total_wl_raw) / float(incremental_scorer.wl_normalizer)
        score = wl_base + wl_delta + 0.5 * density + 0.5 * congestion
        scores = score.detach().cpu().numpy().astype(np.float64)
    for proposal, score_value in zip(proposals, scores):
        proposal["score"] = float(score_value)
    return {
        "prep_elapsed": prep_elapsed,
        "density_updates": int(sum(part.size for part in row_parts)),
        "macro_route_updates": int(sum(part.size for part in macro_route_row_parts)),
        "net_route_updates": int(sum(part.size for part in net_route_row_parts)),
        "hpwl_segments": int(hpwl_stats.get("hpwl_segments", 0)),
        "hpwl_pins": int(hpwl_stats.get("hpwl_pins", 0)),
        "hpwl_rows": int(hpwl_stats.get("hpwl_rows", 0)),
    }


def _score_relocation_proposals_cuda_delta(
    proposals: list[dict],
    *,
    pos: np.ndarray,
    cw: float,
    ch: float,
    local_cong: np.ndarray,
    tgt_cong: np.ndarray,
    incremental_scorer,
) -> None:
    """Assign CUDA-capable exact-proxy scores, chunked for GPU memory control."""
    if not proposals:
        return

    static_bytes_estimate = _relocation_static_tensor_bytes_estimate(
        incremental_scorer,
        proposals,
    )
    chunk_raw = os.environ.get("RELOC_PROPOSE_CHUNK_SIZE")
    user_chunked = bool(chunk_raw)
    chunk_source = "env" if chunk_raw else ("cuda_default" if _GPU_DEVICE.type == "cuda" else "cpu")
    if chunk_raw:
        try:
            chunk_size = int(chunk_raw)
        except ValueError:
            chunk_size = len(proposals)
    else:
        chunk_size = (
            const.RELOC_PROPOSE_DEFAULT_CUDA_CHUNK_SIZE
            if _GPU_DEVICE.type == "cuda"
            else len(proposals)
        )
        default_chunk_size = chunk_size
        budget_chunk = None
        budget_mb = None
        budget_source = None
        budget_dynamic_bytes = None
        budget_total_bytes = None
        auto_mem_frac = None
        auto_free_bytes = None
        auto_total_bytes = None
        bytes_per_proposal = _relocation_dynamic_bytes_per_proposal(
            incremental_scorer,
            proposals,
        )
        budget_adjusted_after_static = False
        budget_adjustment = "none"
    if _GPU_DEVICE.type == "cuda":
        max_mb_raw = os.environ.get("RELOC_PROPOSE_MAX_MB", "")
        (
            budget_chunk,
            budget_mb,
            bytes_per_proposal,
            budget_dynamic_bytes,
            budget_total_bytes,
        ) = _chunk_size_from_memory_budget(
            proposals,
            incremental_scorer,
            max_mb_raw,
            static_bytes_estimate,
        )
        if budget_chunk is not None:
            budget_source = "max_mb"
        elif not max_mb_raw.strip():
            auto_mem_frac_raw = os.environ.get(
                "RELOC_PROPOSE_AUTO_MEM_FRAC", str(const.RELOC_PROPOSE_AUTO_MEM_FRAC)
            )
            auto_budget = _cuda_auto_memory_budget_mb(auto_mem_frac_raw)
            if auto_budget is not None:
                auto_budget_mb, auto_mem_frac, auto_free_bytes, auto_total_bytes = auto_budget
                (
                    budget_chunk,
                    budget_mb,
                    bytes_per_proposal,
                    budget_dynamic_bytes,
                    budget_total_bytes,
                ) = _chunk_size_from_memory_budget(
                    proposals,
                    incremental_scorer,
                    str(auto_budget_mb),
                    static_bytes_estimate,
                )
                if budget_chunk is not None:
                    budget_source = "auto_mem_frac"
            if budget_chunk is not None:
                chunk_size = min(chunk_size, budget_chunk)
                natural_chunk_size = min(default_chunk_size, len(proposals))
                chunk_source = (
                    "memory_budget" if budget_chunk < natural_chunk_size else "cuda_default"
                )
    if chunk_raw:
        budget_mb = None
        budget_chunk = None
        budget_source = None
        budget_dynamic_bytes = None
        budget_total_bytes = None
        auto_mem_frac = None
        auto_free_bytes = None
        auto_total_bytes = None
        bytes_per_proposal = _relocation_dynamic_bytes_per_proposal(
            incremental_scorer,
            proposals,
        )
        budget_adjusted_after_static = False
        budget_adjustment = "none"
    chunk_size = len(proposals) if chunk_size <= 0 else min(chunk_size, len(proposals))
    initial_chunk_size = chunk_size
    retries = 0
    if _GPU_DEVICE.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(_GPU_DEVICE)
        except Exception:
            pass

    t0_score = time.monotonic()
    try:
        _cuda_synchronize_if_needed()
        t0_static = time.monotonic()
        static_tensors = _build_relocation_cuda_static_tensors(
            proposals,
            pos=pos,
            incremental_scorer=incremental_scorer,
            dev=_GPU_DEVICE,
        )
        _cuda_synchronize_if_needed()
        static_elapsed = time.monotonic() - t0_static
        static_tensor_bytes_actual = _tensor_tree_bytes(static_tensors)
        if budget_total_bytes is not None and not user_chunked:
            actual_dynamic_budget = max(0, int(budget_total_bytes) - static_tensor_bytes_actual)
            budget_dynamic_bytes = actual_dynamic_budget
            budget_static_exceeds = static_tensor_bytes_actual > int(budget_total_bytes)
            if bytes_per_proposal > 0:
                actual_budget_chunk = max(
                    1,
                    min(len(proposals), actual_dynamic_budget // bytes_per_proposal),
                )
                natural_chunk_size = min(default_chunk_size, len(proposals))
                adjusted_chunk = min(int(actual_budget_chunk), natural_chunk_size)
                if adjusted_chunk != chunk_size:
                    previous_chunk_size = chunk_size
                    chunk_size = adjusted_chunk
                    budget_chunk = adjusted_chunk
                    budget_adjusted_after_static = True
                    budget_adjustment = "grow" if adjusted_chunk > previous_chunk_size else "shrink"
                    chunk_source = (
                        "memory_budget" if adjusted_chunk < natural_chunk_size else "cuda_default"
                    )
        else:
            budget_static_exceeds = (
                None
                if budget_total_bytes is None
                else static_tensor_bytes_actual > int(budget_total_bytes)
            )
    except RuntimeError as exc:
        if _GPU_DEVICE.type == "cuda" and _is_torch_oom(exc):
            raise RuntimeError(
                "CUDA OOM while building relocation static tensors; "
                "reducing RELOC_PROPOSE_CHUNK_SIZE will not reduce this allocation "
                "(estimated_static_bytes=%d)." % static_bytes_estimate
            ) from exc
        raise

    while True:
        try:
            batches = 0
            batch_elapsed = 0.0
            prep_elapsed = 0.0
            density_updates = 0
            macro_route_updates = 0
            net_route_updates = 0
            hpwl_segments = 0
            hpwl_pins = 0
            hpwl_rows = 0
            for start in range(0, len(proposals), chunk_size):
                batches += 1
                _cuda_synchronize_if_needed()
                t0_batch = time.monotonic()
                batch_stats = _score_relocation_proposals_cuda_delta_batch(
                    proposals[start : start + chunk_size],
                    proposal_start=start,
                    incremental_scorer=incremental_scorer,
                    static_tensors=static_tensors,
                )
                _cuda_synchronize_if_needed()
                batch_elapsed += time.monotonic() - t0_batch
                if batch_stats:
                    prep_elapsed += float(batch_stats.get("prep_elapsed", 0.0))
                    density_updates += int(batch_stats.get("density_updates", 0))
                    macro_route_updates += int(batch_stats.get("macro_route_updates", 0))
                    net_route_updates += int(batch_stats.get("net_route_updates", 0))
                    hpwl_segments += int(batch_stats.get("hpwl_segments", 0))
                    hpwl_pins += int(batch_stats.get("hpwl_pins", 0))
                    hpwl_rows += int(batch_stats.get("hpwl_rows", 0))
            _cuda_synchronize_if_needed()
            elapsed = time.monotonic() - t0_score
            dynamic_components = _relocation_dynamic_byte_components(
                incremental_scorer,
                proposals,
            )
            _score_relocation_proposals_cuda_delta.last_stats = {
                "device": str(_GPU_DEVICE),
                "backend": _GPU_DEVICE.type,
                "proposals": len(proposals),
                "initial_chunk_size": initial_chunk_size,
                "final_chunk_size": chunk_size,
                "batches": batches,
                "retries": retries,
                "elapsed": elapsed,
                "ms_per_proposal": 1000.0 * elapsed / max(len(proposals), 1),
                "ms_per_batch": 1000.0 * elapsed / max(batches, 1),
                "static_elapsed": static_elapsed,
                "batch_elapsed": batch_elapsed,
                "prep_elapsed": prep_elapsed,
                "tensor_elapsed_estimate": max(0.0, batch_elapsed - prep_elapsed),
                "static_ms": 1000.0 * static_elapsed,
                "batch_ms": 1000.0 * batch_elapsed,
                "prep_ms": 1000.0 * prep_elapsed,
                "tensor_ms_estimate": 1000.0 * max(0.0, batch_elapsed - prep_elapsed),
                "density_updates": density_updates,
                "macro_route_updates": macro_route_updates,
                "net_route_updates": net_route_updates,
                "hpwl_segments": hpwl_segments,
                "hpwl_pins": hpwl_pins,
                "hpwl_rows": hpwl_rows,
                "memory_budget_mb": budget_mb,
                "memory_budget_chunk": budget_chunk,
                "memory_budget_source": budget_source,
                "memory_budget_total_bytes": budget_total_bytes,
                "memory_budget_dynamic_bytes": budget_dynamic_bytes,
                "memory_budget_static_exceeds": budget_static_exceeds,
                "memory_budget_adjusted_after_static": budget_adjusted_after_static,
                "memory_budget_adjustment": budget_adjustment,
                "auto_memory_frac": auto_mem_frac,
                "auto_cuda_free_bytes": auto_free_bytes,
                "auto_cuda_total_bytes": auto_total_bytes,
                "dynamic_bytes_per_proposal": bytes_per_proposal,
                "grid_dynamic_bytes_per_proposal": dynamic_components[
                    "grid_dynamic_bytes_per_proposal"
                ],
                "hpwl_dynamic_bytes_per_proposal": dynamic_components[
                    "hpwl_dynamic_bytes_per_proposal"
                ],
                "memory_safety_factor": _relocation_memory_safety_factor(),
                "static_tensor_bytes_estimate": static_bytes_estimate,
                "static_tensor_bytes_actual": static_tensor_bytes_actual,
                "chunk_source": chunk_source,
            }
            _score_relocation_proposals_cuda_delta.last_stats.update(_cuda_runtime_status())
            _score_relocation_proposals_cuda_delta.last_stats.update(_cuda_memory_stats())
            return
        except RuntimeError as exc:
            if _GPU_DEVICE.type != "cuda" or not _is_torch_oom(exc) or chunk_size <= 1:
                raise
            retries += 1
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            next_chunk = max(1, chunk_size // 2)
            if next_chunk == chunk_size:
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
                source = "user" if user_chunked else "auto"
                print(
                    "  R2 propose-all[cuda_delta]: CUDA OOM at chunk_size=%d "
                    "(%s); retrying chunk_size=%d" % (chunk_size, source, next_chunk),
                    flush=True,
                )
            chunk_size = next_chunk


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
        prep = incremental_scorer._prepare_move(i)
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
                targets.append((nx, ny))
            targets = _dedupe_targets_xy(targets)
            if targets.size:
                scores = incremental_scorer._trial_many_at(prep, targets)
            else:
                scores = np.empty(0, dtype=np.float64)
            for (nx, ny), score in zip(targets, scores):
                if float(score) < best_score - float(min_gain):
                    best_score = float(score)
                    best_xy = (nx, ny)
            if best_xy is not None:
                incremental_scorer._commit_after_prep(prep, best_xy)
                hard_pos[i, 0], hard_pos[i, 1] = best_xy
                accepts += 1
            else:
                incremental_scorer._revert_prep(prep)
        except Exception:
            incremental_scorer._revert_prep(prep)
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
        prep = incremental_scorer._prepare_move_soft(k)
        best_xy = None
        try:
            targets = []
            for dx, dy, _dist in offsets:
                nx = float(np.clip(soft_pos[k, 0] + dx, soft_hw[k], cw - soft_hw[k]))
                ny = float(np.clip(soft_pos[k, 1] + dy, soft_hh[k], ch - soft_hh[k]))
                if soft_region is not None and not point_in_region(soft_region, k, nx, ny):
                    continue
                targets.append((nx, ny))
            targets = _dedupe_targets_xy(targets)
            if targets.size:
                scores = incremental_scorer._trial_many_at_soft(prep, targets)
            else:
                scores = np.empty(0, dtype=np.float64)
            for (nx, ny), score in zip(targets, scores):
                if float(score) < best_score - float(min_gain):
                    best_score = float(score)
                    best_xy = (nx, ny)
            if best_xy is not None:
                incremental_scorer._commit_after_prep_soft(prep, best_xy)
                soft_pos[k, 0], soft_pos[k, 1] = best_xy
                accepts += 1
            else:
                incremental_scorer._revert_prep_soft(prep)
        except Exception:
            incremental_scorer._revert_prep_soft(prep)
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
    scorer_mode = _proposal_scorer_mode()

    for hot_rank, i_raw in enumerate(hot):
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
        d2 = (tgt_x[cand] - pos[i, 0]) ** 2 + (tgt_y[cand] - pos[i, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[i, 0]) ** 2 + (tgt_y[cand] - net_centroid[i, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        if region_bbox is not None and region_bias > 0.0:
            d2 = _region_penalty(d2, tgt_x, tgt_y, cand, region_bbox, i, _span2, region_bias)
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
            legal.append((int(candidate_rank), int(t), nx, ny))
        if not legal:
            continue
        legal_count += len(legal)

        if scorer_mode == "exact":
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
                            "xy": (nx, ny),
                        }
                    )
            finally:
                incremental_scorer._revert_prep(prep)
        else:
            for candidate_rank, target_index, nx, ny in legal:
                proposals.append(
                    {
                        "score": 0.0,
                        "i": i,
                        "hot_rank": int(hot_rank),
                        "candidate_rank": int(candidate_rank),
                        "target_index": int(target_index),
                        "local_field": local_field,
                        "target_field": float(cand_field[target_index]),
                        "xy": (nx, ny),
                    }
                )

    if scorer_mode == "tensor" and proposals:
        _score_relocation_proposals_tensor(
            proposals,
            pos=pos,
            cw=cw,
            ch=ch,
            local_cong=local_cong,
            tgt_cong=tgt_cong,
        )
        frozen_scores = len(proposals)
    elif scorer_mode == "cuda_delta" and proposals:
        _score_relocation_proposals_cuda_delta(
            proposals,
            pos=pos,
            cw=cw,
            ch=ch,
            local_cong=local_cong,
            tgt_cong=tgt_cong,
            incremental_scorer=incremental_scorer,
        )
        frozen_scores = len(proposals)

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
    if getattr(benchmark, "name", ""):
        from placer.local_search.gnn_ranker import reorder_hard_relocation_proposals

        proposals = reorder_hard_relocation_proposals(
            proposals,
            benchmark_name=str(getattr(benchmark, "name", "")),
            field=field,
        )
    if propose_top_m is not None and propose_top_m > 0:
        top_m = int(propose_top_m)
        gnn_rank_on = os.environ.get("HIER_GNN_RANK", "0").strip() not in {
            "0",
            "false",
            "False",
            "no",
            "NO",
            "off",
            "",
        }
        if gnn_rank_on:
            top_m += max(0, int(os.environ.get("HIER_GNN_EXTRA_TOP_K", "0") or "0"))
        proposals = proposals[:top_m]

    log_gnn_event(
        "hier_relocation_candidates",
        benchmark=getattr(benchmark, "name", ""),
        kind="hard_propose_all",
        field=field,
        scorer=scorer_mode,
        initial_proxy=float(initial_score),
        proposal_count=int(len(proposals)),
        legal_count=int(legal_count),
        frozen_scores=int(frozen_scores),
        structural_weight=float(_hierarchy_structural_weight()),
        candidates=_candidate_trace_sample(proposals, gnn_trace_limit()),
    )

    moved = set()
    accepted_trace = []
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
                old_score = float(best_score)
                incremental_scorer._commit_after_prep(prep, (nx, ny))
                pos[i, 0], pos[i, 1] = nx, ny
                best_score = float(score)
                accepts += 1
                moved.add(i)
                accepted_trace.append(
                    {
                        "macro": i,
                        "x": float(nx),
                        "y": float(ny),
                        "old_proxy": old_score,
                        "new_proxy": float(score),
                        "proxy_delta": float(score) - old_score,
                        "target_index": int(p.get("target_index", -1)),
                        "candidate_rank": int(p.get("candidate_rank", -1)),
                        "structural_delta": float(p.get("structural_delta", 0.0)),
                    }
                )
            else:
                incremental_scorer._revert_prep(prep)
        except Exception:
            incremental_scorer._revert_prep(prep)
            raise

    log_gnn_event(
        "hier_relocation_result",
        benchmark=getattr(benchmark, "name", ""),
        kind="hard_propose_all",
        field=field,
        initial_proxy=float(initial_score),
        final_proxy=float(best_score),
        verify_scores=int(verify_scores),
        accepts=int(accepts),
        accepted=accepted_trace[: gnn_trace_limit()],
    )

    if os.environ.get("RELOC_PROPOSE_LOG", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
        "on",
        "ON",
    }:
        scorer_extra = ""
        if scorer_mode == "cuda_delta":
            stats = getattr(_score_relocation_proposals_cuda_delta, "last_stats", None)
            if stats:
                scorer_extra = (
                    " device=%s chunk=%d->%d source=%s batches=%d retries=%d "
                    "score_ms=%.1f static_ms=%.1f batch_ms=%.1f prep_ms=%.1f "
                    "tensor_ms=%.1f updates=%d/%d/%d hpwl=%d/%d/%d ms_per_prop=%.3f "
                    "requested=%s configured=%s torch_cuda=%s/%d"
                    % (
                        stats["device"],
                        stats["initial_chunk_size"],
                        stats["final_chunk_size"],
                        stats.get("chunk_source", ""),
                        stats["batches"],
                        stats["retries"],
                        1000.0 * stats.get("elapsed", 0.0),
                        stats.get("static_ms", 0.0),
                        stats.get("batch_ms", 0.0),
                        stats.get("prep_ms", 0.0),
                        stats.get("tensor_ms_estimate", 0.0),
                        stats.get("density_updates", 0),
                        stats.get("macro_route_updates", 0),
                        stats.get("net_route_updates", 0),
                        stats.get("hpwl_segments", 0),
                        stats.get("hpwl_pins", 0),
                        stats.get("hpwl_rows", 0),
                        stats.get("ms_per_proposal", 0.0),
                        stats.get("requested_device", ""),
                        stats.get("configured_backend", ""),
                        stats.get("torch_cuda_available", False),
                        stats.get("torch_cuda_device_count", 0),
                    )
                )
                if "max_memory_allocated" in stats:
                    scorer_extra += " max_alloc=%.1fMiB max_reserved=%.1fMiB" % (
                        stats["max_memory_allocated"] / (1024.0 * 1024.0),
                        stats.get("max_memory_reserved", 0) / (1024.0 * 1024.0),
                    )
                if stats.get("memory_budget_mb") is not None:
                    scorer_extra += (
                        " budget_mb=%.1f budget_source=%s budget_static_exceeds=%s "
                        "budget_adjusted_after_static=%s budget_adjustment=%s dyn_budget=%.1fMiB "
                        "est_bytes_prop=%d grid_bytes=%d hpwl_bytes=%d safety=%.2f "
                        "static_est=%.1fMiB static_actual=%.1fMiB"
                        % (
                            stats["memory_budget_mb"],
                            stats.get("memory_budget_source", ""),
                            stats.get("memory_budget_static_exceeds", False),
                            stats.get("memory_budget_adjusted_after_static", False),
                            stats.get("memory_budget_adjustment", "none"),
                            (stats.get("memory_budget_dynamic_bytes") or 0) / (1024.0 * 1024.0),
                            stats.get("dynamic_bytes_per_proposal", 0),
                            stats.get("grid_dynamic_bytes_per_proposal", 0),
                            stats.get("hpwl_dynamic_bytes_per_proposal", 0),
                            stats.get("memory_safety_factor", 1.0),
                            stats.get("static_tensor_bytes_estimate", 0) / (1024.0 * 1024.0),
                            stats.get("static_tensor_bytes_actual", 0) / (1024.0 * 1024.0),
                        )
                    )
                    if stats.get("memory_budget_source") == "auto_mem_frac":
                        scorer_extra += " auto_frac=%.3f cuda_free=%.1fMiB cuda_total=%.1fMiB" % (
                            stats.get("auto_memory_frac", 0.0),
                            (stats.get("auto_cuda_free_bytes") or 0) / (1024.0 * 1024.0),
                            (stats.get("auto_cuda_total_bytes") or 0) / (1024.0 * 1024.0),
                        )
        print(
            "  R2 propose-all[%s]: hot=%d legal=%d frozen_scores=%d "
            "selected=%d verify_scores=%d accepts=%d scorer=%s%s elapsed=%.3fs"
            % (
                field,
                len(hot),
                legal_count,
                frozen_scores,
                len(proposals),
                verify_scores,
                accepts,
                scorer_mode,
                scorer_extra,
                time.monotonic() - t0,
            ),
            flush=True,
        )
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
) -> "tuple[np.ndarray, int, float]":
    """Move hot hard macros to colder legal spots.

    When `region_bbox` (per-macro center-feasible box [n,4]) and `region_bias>0`
    are given, out-of-region candidate cells get a ranking penalty so a macro
    strongly prefers staying within its cluster region — a SOFT region lock (it
    can still exit when in-region cold cells run out). Bit-identical when
    `region_bbox is None` (the production pipeline never passes it).
    """
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    trace_field = "combined" if use_combined else ("density" if use_density else "congestion")
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
            else _congestion_field(incremental_scorer, nr, nc)
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
    _thr = np.percentile(flat, 55)
    pool = np.where(flat < _thr)[0]
    if pool.size < max(n_targets, 64):
        pool = np.argsort(flat)[: max(n_targets, 64)]
    tgt_c = (pool % nc).astype(np.float64)
    tgt_r = (pool // nc).astype(np.float64)
    tgt_x = (tgt_c + 0.5) * cell_w
    tgt_y = (tgt_r + 0.5) * cell_h
    tgt_cong = flat[pool]

    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    EPS = 0.05

    # Optional path ranks all proposals before exact checking.
    if propose_all:
        return _relocation_moves_propose_all(
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
        )

    best_score = initial_score
    accepts = 0
    all_idx = np.arange(n)
    _span2 = float(max(cw, ch)) ** 2
    full_pos_for_struct = _full_committed_pos(incremental_scorer)
    sizes_for_struct = _full_macro_sizes(incremental_scorer)
    accepted_trace = []
    for i in hot:
        i = int(i)
        if deadline is not None and time.monotonic() > deadline:
            break
        if not movable[i]:
            continue
        # Try colder targets first, capped per macro.
        cand_field = tgt_cong
        cand = np.where(cand_field < local_cong[i] - 1e-9)[0]
        if cand.size == 0:
            continue
        d2 = (tgt_x[cand] - pos[i, 0]) ** 2 + (tgt_y[cand] - pos[i, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[i, 0]) ** 2 + (tgt_y[cand] - net_centroid[i, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        if region_bbox is not None and region_bias > 0.0:
            d2 = _region_penalty(d2, tgt_x, tgt_y, cand, region_bbox, i, _span2, region_bias)
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
        # Remove the macro once, then try several target cells.
        prep = incremental_scorer._prepare_move(i)
        best_i_xy = None
        score_before_i = float(best_score)
        try:
            targets = []
            for candidate_rank, t in enumerate(cand):
                nx, ny = float(tgt_x[t]), float(tgt_y[t])
                # Keep the whole macro inside the canvas.
                if nx - hw[i] < 0 or nx + hw[i] > cw or ny - hh[i] < 0 or ny + hh[i] > ch:
                    continue
                # Hard macros cannot overlap.
                if ((np.abs(nx - ox) < sxi + EPS) & (np.abs(ny - oy) < syi + EPS)).any():
                    continue
                targets.append((nx, ny))
            if targets:
                scores = incremental_scorer._trial_many_at(prep, np.asarray(targets))
            else:
                scores = np.empty(0, dtype=np.float64)
            for nx, ny, s in [(x, y, score) for (x, y), score in zip(targets, scores)]:
                outside = not point_in_region(region_bbox, i, nx, ny)
                if accepts_region_score(s, best_score, outside, region_escape_min):
                    best_score = s
                    best_i_xy = (nx, ny)
            if best_i_xy is not None:
                incremental_scorer._commit_after_prep(prep, best_i_xy)
                pos[i, 0], pos[i, 1] = best_i_xy
                full_pos_for_struct[i, 0], full_pos_for_struct[i, 1] = best_i_xy
                accepts += 1
                accepted_trace.append(
                    {
                        "macro": i,
                        "x": float(best_i_xy[0]),
                        "y": float(best_i_xy[1]),
                        "old_proxy": score_before_i,
                        "new_proxy": float(best_score),
                        "proxy_delta": float(best_score) - score_before_i,
                    }
                )
            else:
                incremental_scorer._revert_prep(prep)
        except Exception:
            # Restore committed state if a trial failed.
            incremental_scorer._revert_prep(prep)
            raise
    log_gnn_event(
        "hier_relocation_result",
        benchmark=getattr(benchmark, "name", ""),
        kind="hard_sequential",
        field=trace_field,
        initial_proxy=float(initial_score),
        final_proxy=float(best_score),
        hot_count=int(len(hot)),
        accepts=int(accepts),
        structural_weight=float(_hierarchy_structural_weight()),
        accepted=accepted_trace[: gnn_trace_limit()],
    )
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
) -> "tuple[np.ndarray, int, float]":
    """Move hot soft macros to colder cells."""
    num_soft = incremental_scorer.num_soft
    if num_soft == 0:
        return soft_pos, 0, initial_score
    # Skip full scoring when wirelength alone is already too costly.
    _env_wl = os.environ.get("SOFT_RELOC_WL_PREFILTER")
    if _env_wl not in (None, ""):
        wl_prefilter = float(_env_wl)
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    cell_field = (
        _density_field(incremental_scorer, nr, nc)
        if use_density
        else _congestion_field(incremental_scorer, nr, nc)
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
    _thr = np.percentile(flat, 55)
    pool = np.where(flat < _thr)[0]
    if pool.size < max(n_targets, 64):
        pool = np.argsort(flat)[: max(n_targets, 64)]
    tgt_x = ((pool % nc).astype(np.float64) + 0.5) * cell_w
    tgt_y = ((pool // nc).astype(np.float64) + 0.5) * cell_h
    tgt_cong = flat[pool]

    best_score = initial_score
    accepts = 0
    full_pos_for_struct = _full_committed_pos(incremental_scorer)
    sizes_for_struct = _full_macro_sizes(incremental_scorer)
    accepted_trace = []
    for k in hot:
        k = int(k)
        if deadline is not None and time.monotonic() > deadline:
            break
        cand = np.where(tgt_cong < local_cong[k] - 1e-9)[0]
        if cand.size == 0:
            continue
        # Order targets by distance and optional wirelength anchor.
        d2 = (tgt_x[cand] - soft_pos[k, 0]) ** 2 + (tgt_y[cand] - soft_pos[k, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[k, 0]) ** 2 + (tgt_y[cand] - net_centroid[k, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        if region_bbox is not None and region_bias > 0.0:
            _span2 = float(max(cw, ch)) ** 2
            d2 = _region_penalty(d2, tgt_x, tgt_y, cand, region_bbox, k, _span2, region_bias)
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
        # Remove the soft macro once, then try several targets.
        prep = incremental_scorer._prepare_move_soft(k)
        best_k_xy = None
        score_before_k = float(best_score)
        try:
            targets = []
            for t in cand:
                nx = float(np.clip(tgt_x[t], soft_hw[k], cw - soft_hw[k]))
                ny = float(np.clip(tgt_y[t], soft_hh[k], ch - soft_hh[k]))
                # Cheaply skip targets with too much wirelength damage.
                wl_d = 0.0
                if wl_prefilter > 0.0:
                    wl_d = incremental_scorer.wl_delta_move_soft(k, (nx, ny))
                if wl_prefilter > 0.0 and wl_d > wl_prefilter:
                    continue
                targets.append((nx, ny))
            targets = _dedupe_targets_xy(targets)
            if targets.size:
                scores = incremental_scorer._trial_many_at_soft(prep, targets)
            else:
                scores = np.empty(0, dtype=np.float64)
            for nx, ny, s in [
                (float(p[0]), float(p[1]), float(score)) for p, score in zip(targets, scores)
            ]:
                outside = not point_in_region(region_bbox, k, nx, ny)
                min_gain = max(
                    1e-9,
                    float(accept_min_gain),
                    float(region_escape_min) if outside else 0.0,
                )
                if float(s) < float(best_score) - min_gain:
                    best_score = s
                    best_k_xy = (nx, ny)
            if best_k_xy is not None:
                incremental_scorer._commit_after_prep_soft(prep, best_k_xy)
                soft_pos[k, 0], soft_pos[k, 1] = best_k_xy
                full_pos_for_struct[n + k, 0], full_pos_for_struct[n + k, 1] = best_k_xy
                accepts += 1
                accepted_trace.append(
                    {
                        "soft_macro": k,
                        "x": float(best_k_xy[0]),
                        "y": float(best_k_xy[1]),
                        "old_proxy": score_before_k,
                        "new_proxy": float(best_score),
                        "proxy_delta": float(best_score) - score_before_k,
                    }
                )
            else:
                incremental_scorer._revert_prep_soft(prep)
        except Exception:
            incremental_scorer._revert_prep_soft(prep)
            raise
    log_gnn_event(
        "hier_relocation_result",
        benchmark=getattr(benchmark, "name", ""),
        kind="soft_sequential",
        field="density" if use_density else "congestion",
        initial_proxy=float(initial_score),
        final_proxy=float(best_score),
        hot_count=int(len(hot)),
        accepts=int(accepts),
        structural_weight=float(_hierarchy_structural_weight()),
        accepted=accepted_trace[: gnn_trace_limit()],
    )
    return soft_pos, accepts, best_score
