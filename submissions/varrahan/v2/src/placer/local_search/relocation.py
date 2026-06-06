"""Local search move operators."""

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from placer.config import _GPU_DEVICE
from placer.geometry import separation_matrices
from placer.local_search.fields import _congestion_field, _density_field
from placer.ml.data_collection import TraceFields, get_candidate_trace, net_degree_features
from placer.ml.shadow import (
    filter_candidate_indices,
    is_filter_calibration_enabled,
    is_filter_enabled,
    shadow_rank_group,
    update_filter_calibration,
)
from placer.plc.placement import _ensure_pos_cache

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark


def _proposal_scorer_mode() -> str:
    mode = os.environ.get("V2_RELOC_PROPOSE_SCORER", "cuda_delta").strip().lower()
    return mode if mode in {"exact", "tensor", "cuda_delta"} else "cuda_delta"


def _score_relocation_proposals_tensor(
    proposals: list[dict],
    *,
    pos: np.ndarray,
    cw: float,
    ch: float,
    local_cong: np.ndarray,
    tgt_cong: np.ndarray,
) -> None:
    """Assign batched tensor heuristic scores to frozen-base proposals.

    This is a ranking scorer, not the accept gate. Lower score is better to
    match exact proxy sorting. It deliberately uses cheap pre-score quantities:
    source/target field relief, normalized displacement, and candidate rank.
    """
    if not proposals:
        return
    dev = _GPU_DEVICE
    idx = np.asarray([p["i"] for p in proposals], dtype=np.int64)
    target_idx = np.asarray([p["target_index"] for p in proposals], dtype=np.int64)
    candidate_rank = np.asarray([p["candidate_rank"] for p in proposals], dtype=np.float32)
    nx = np.asarray([p["xy"][0] for p in proposals], dtype=np.float32)
    ny = np.asarray([p["xy"][1] for p in proposals], dtype=np.float32)

    with torch.no_grad():
        i_t = torch.as_tensor(idx, device=dev, dtype=torch.long)
        t_t = torch.as_tensor(target_idx, device=dev, dtype=torch.long)
        nx_t = torch.as_tensor(nx, device=dev)
        ny_t = torch.as_tensor(ny, device=dev)
        rank_t = torch.as_tensor(candidate_rank, device=dev)

        pos_t = torch.as_tensor(pos, device=dev, dtype=torch.float32)
        local_t = torch.as_tensor(local_cong, device=dev, dtype=torch.float32)
        tgt_t = torch.as_tensor(tgt_cong, device=dev, dtype=torch.float32)

        field_max = torch.clamp(torch.max(local_t), min=1e-12)
        relief = torch.clamp((local_t[i_t] - tgt_t[t_t]) / field_max, min=0.0)
        dx = (nx_t - pos_t[i_t, 0]) / max(float(cw), 1e-12)
        dy = (ny_t - pos_t[i_t, 1]) / max(float(ch), 1e-12)
        dist = torch.sqrt(dx * dx + dy * dy)
        rank_norm = rank_t / torch.clamp(torch.max(rank_t), min=1.0)

        # Synthetic score: prefer high relief, then shorter moves and earlier
        # heuristic-ranked targets. Serial exact verification still decides.
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
    """Return exact per-cell hard-macro routing blockage contribution.

    Mirrors `routing.apply._apply_macro_routing` for one macro, but returns
    sparse flat indices plus V/H additive values so proposal batches can scatter
    old/new blockage deltas into Torch tensors.
    """
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
    """Return exact 2-pin net-routing contribution for nets touching a module.

    This is an incremental CUDA-scorer slice, not the full net-routing port.
    It mirrors `_apply_2pin_routing` for length-2 touched nets so proposal
    ranking can account for a common exact net-routing delta before the serial
    exact accept gate runs.
    """
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
    """Return exact 3-pin net-routing contribution for nets touching a module."""
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
    g1 = rows[n3:2 * n3] * grid_col + cols[n3:2 * n3]
    g2 = rows[2 * n3:] * grid_col + cols[2 * n3:]
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
        y_all = np.stack([g0u // grid_col, g1u // grid_col, g2u // grid_col], axis=1).astype(np.int64)
        x_all = np.stack([g0u % grid_col, g1u % grid_col, g2u % grid_col], axis=1).astype(np.int64)
        big = int(max(grid_row, grid_col)) + 16
        order = np.argsort(x_all * big + y_all, axis=1, kind="stable")
        y = np.take_along_axis(y_all, order, axis=1)
        x = np.take_along_axis(x_all, order, axis=1)
        y1 = y[:, 0]; y2 = y[:, 1]; y3 = y[:, 2]
        x1 = x[:, 0]; x2 = x[:, 1]; x3 = x[:, 2]

        case1 = (x1 < x2) & (x2 < x3) & (np.minimum(y1, y3) < y2) & (np.maximum(y1, y3) > y2)
        case2 = (~case1) & (x2 == x3) & (x1 < x2) & (y1 < np.minimum(y2, y3))
        case3 = (~case1) & (~case2) & (y2 == y3)
        case4 = ~(case1 | case2 | case3)

        if case1.any():
            m = case1
            wm = wu[m]
            h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
            h_rows.append(y2[m]); h_los.append(x2[m]); h_his.append(x3[m]); h_ws.append(wm)
            v_cols.append(x2[m]); v_los.append(np.minimum(y1[m], y2[m])); v_his.append(np.maximum(y1[m], y2[m])); v_ws.append(wm)
            v_cols.append(x3[m]); v_los.append(np.minimum(y2[m], y3[m])); v_his.append(np.maximum(y2[m], y3[m])); v_ws.append(wm)

        if case2.any():
            m = case2
            wm = wu[m]
            h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
            v_cols.append(x2[m]); v_los.append(y1[m]); v_his.append(np.maximum(y2[m], y3[m])); v_ws.append(wm)

        if case3.any():
            m = case3
            wm = wu[m]
            h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
            h_rows.append(y2[m]); h_los.append(x2[m]); h_his.append(x3[m]); h_ws.append(wm)
            v_cols.append(x2[m]); v_los.append(np.minimum(y2[m], y1[m])); v_his.append(np.maximum(y2[m], y1[m])); v_ws.append(wm)

        if case4.any():
            m = case4
            wm = wu[m]
            y_t = y_all[m]
            x_t = x_all[m]
            order_t = np.argsort(y_t * big + x_t, axis=1, kind="stable")
            y_t = np.take_along_axis(y_t, order_t, axis=1)
            x_t = np.take_along_axis(x_t, order_t, axis=1)
            y1t = y_t[:, 0]; y2t = y_t[:, 1]; y3t = y_t[:, 2]
            x1t = x_t[:, 0]; x2t = x_t[:, 1]; x3t = x_t[:, 2]
            h_rows.append(y2t)
            h_los.append(np.minimum(np.minimum(x1t, x2t), x3t))
            h_his.append(np.maximum(np.maximum(x1t, x2t), x3t))
            h_ws.append(wm)
            v_cols.append(x1t); v_los.append(np.minimum(y1t, y2t)); v_his.append(np.maximum(y1t, y2t)); v_ws.append(wm)
            v_cols.append(x3t); v_los.append(np.minimum(y2t, y3t)); v_his.append(np.maximum(y2t, y3t)); v_ws.append(wm)

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
    """Return exact >=4-pin net-routing contribution for touched nets.

    Mirrors the high-fanout section of `_apply_net_routing_struct`: sinks equal
    to the source gcell are ignored, remaining sinks are deduped per net, two
    unique sinks route as an effective 3-pin net, otherwise source-to-sink
    starlike 2-pin routes are emitted.
    """
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
        y1 = y[:, 0]; y2 = y[:, 1]; y3 = y[:, 2]
        x1 = x[:, 0]; x2 = x[:, 1]; x3 = x[:, 2]
        case1 = (x1 < x2) & (x2 < x3) & (np.minimum(y1, y3) < y2) & (np.maximum(y1, y3) > y2)
        case2 = (~case1) & (x2 == x3) & (x1 < x2) & (y1 < np.minimum(y2, y3))
        case3 = (~case1) & (~case2) & (y2 == y3)
        case4 = ~(case1 | case2 | case3)
        if case1.any():
            m = case1
            wm = weights[m]
            h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
            h_rows.append(y2[m]); h_los.append(x2[m]); h_his.append(x3[m]); h_ws.append(wm)
            v_cols.append(x2[m]); v_los.append(np.minimum(y1[m], y2[m])); v_his.append(np.maximum(y1[m], y2[m])); v_ws.append(wm)
            v_cols.append(x3[m]); v_los.append(np.minimum(y2[m], y3[m])); v_his.append(np.maximum(y2[m], y3[m])); v_ws.append(wm)
        if case2.any():
            m = case2
            wm = weights[m]
            h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
            v_cols.append(x2[m]); v_los.append(y1[m]); v_his.append(np.maximum(y2[m], y3[m])); v_ws.append(wm)
        if case3.any():
            m = case3
            wm = weights[m]
            h_rows.append(y1[m]); h_los.append(x1[m]); h_his.append(x2[m]); h_ws.append(wm)
            h_rows.append(y2[m]); h_los.append(x2[m]); h_his.append(x3[m]); h_ws.append(wm)
            v_cols.append(x2[m]); v_los.append(np.minimum(y2[m], y1[m])); v_his.append(np.maximum(y2[m], y1[m])); v_ws.append(wm)
        if case4.any():
            m = case4
            wm = weights[m]
            y_t = y_all[m]
            x_t = x_all[m]
            order_t = np.argsort(y_t * big + x_t, axis=1, kind="stable")
            y_t = np.take_along_axis(y_t, order_t, axis=1)
            x_t = np.take_along_axis(x_t, order_t, axis=1)
            y1t = y_t[:, 0]; y2t = y_t[:, 1]; y3t = y_t[:, 2]
            x1t = x_t[:, 0]; x2t = x_t[:, 1]; x3t = x_t[:, 2]
            h_rows.append(y2t)
            h_los.append(np.minimum(np.minimum(x1t, x2t), x3t))
            h_his.append(np.maximum(np.maximum(x1t, x2t), x3t))
            h_ws.append(wm)
            v_cols.append(x1t); v_los.append(np.minimum(y1t, y2t)); v_his.append(np.maximum(y1t, y2t)); v_ws.append(wm)
            v_cols.append(x3t); v_los.append(np.minimum(y2t, y3t)); v_his.append(np.maximum(y2t, y3t)); v_ws.append(wm)

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
        cnts = (
            torch.clamp(rows + sr, max=grid_row - 1)
            - torch.clamp(rows - sr, min=0)
            + 1
        ).to(dtype)
        weighted = grid_3d / cnts.view(1, grid_row, 1)
        zero = torch.zeros(grid_flat.shape[0], 1, grid_col, dtype=dtype, device=dev)
        cs = torch.cumsum(torch.cat([zero, weighted], dim=1), dim=1)
        lo = torch.clamp(rows - sr, min=0)
        hi = torch.clamp(rows + sr + 1, max=grid_row)
        smoothed = cs[:, hi, :] - cs[:, lo, :]
    else:
        cols = torch.arange(grid_col, dtype=torch.long, device=dev)
        cnts = (
            torch.clamp(cols + sr, max=grid_col - 1)
            - torch.clamp(cols - sr, min=0)
            + 1
        ).to(dtype)
        weighted = grid_3d / cnts.view(1, 1, grid_col)
        zero = torch.zeros(grid_flat.shape[0], grid_row, 1, dtype=dtype, device=dev)
        cs = torch.cumsum(torch.cat([zero, weighted], dim=2), dim=2)
        lo = torch.clamp(cols - sr, min=0)
        hi = torch.clamp(cols + sr + 1, max=grid_col)
        smoothed = cs[:, :, hi] - cs[:, :, lo]
    return smoothed.reshape(grid_flat.shape[0], grid_row * grid_col)


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
    """Assign CUDA-capable batched delta scores to frozen-base proposals.

    This evaluates proposal density, HPWL delta, and hard-macro blockage
    congestion through Torch batches on `_GPU_DEVICE`. Net routing deltas are
    still represented by a field-relief proxy; the serial exact verify stage
    remains the correctness gate.
    """
    if not proposals:
        return
    dev = _GPU_DEVICE
    n_prop = len(proposals)
    idx = np.asarray([p["i"] for p in proposals], dtype=np.int64)
    target_idx = np.asarray([p["target_index"] for p in proposals], dtype=np.int64)
    candidate_rank = np.asarray([p["candidate_rank"] for p in proposals], dtype=np.float32)
    nx = np.asarray([p["xy"][0] for p in proposals], dtype=np.float32)
    ny = np.asarray([p["xy"][1] for p in proposals], dtype=np.float32)

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
    for row, proposal in enumerate(proposals):
        i = int(proposal["i"])
        module_idx = int(incremental_scorer.hard_indices[i])
        old_idx, old_area = incremental_scorer._macro_occ(
            module_idx,
            float(pos[i, 0]),
            float(pos[i, 1]),
        )
        new_idx, new_area = incremental_scorer._macro_occ(
            module_idx,
            float(proposal["xy"][0]),
            float(proposal["xy"][1]),
        )
        if old_idx.size:
            row_parts.append(np.full(old_idx.size, row, dtype=np.int64))
            col_parts.append(old_idx.astype(np.int64, copy=False))
            val_parts.append(-old_area.astype(np.float32, copy=False))
        if new_idx.size:
            row_parts.append(np.full(new_idx.size, row, dtype=np.int64))
            col_parts.append(new_idx.astype(np.int64, copy=False))
            val_parts.append(new_area.astype(np.float32, copy=False))

        old_route_idx, old_v, old_h = _macro_routing_contrib(
            incremental_scorer,
            module_idx,
            float(pos[i, 0]),
            float(pos[i, 1]),
        )
        new_route_idx, new_v, new_h = _macro_routing_contrib(
            incremental_scorer,
            module_idx,
            float(proposal["xy"][0]),
            float(proposal["xy"][1]),
        )
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

        old_net_idx, old_net_v, old_net_h = _net_routing_2pin_contrib(
            incremental_scorer,
            module_idx,
            float(pos[i, 0]),
            float(pos[i, 1]),
        )
        new_net_idx, new_net_v, new_net_h = _net_routing_2pin_contrib(
            incremental_scorer,
            module_idx,
            float(proposal["xy"][0]),
            float(proposal["xy"][1]),
        )
        old_net3_idx, old_net3_v, old_net3_h = _net_routing_3pin_contrib(
            incremental_scorer,
            module_idx,
            float(pos[i, 0]),
            float(pos[i, 1]),
        )
        new_net3_idx, new_net3_v, new_net3_h = _net_routing_3pin_contrib(
            incremental_scorer,
            module_idx,
            float(proposal["xy"][0]),
            float(proposal["xy"][1]),
        )
        old_netx_idx, old_netx_v, old_netx_h = _net_routing_highfanout_contrib(
            incremental_scorer,
            module_idx,
            float(pos[i, 0]),
            float(pos[i, 1]),
        )
        new_netx_idx, new_netx_v, new_netx_h = _net_routing_highfanout_contrib(
            incremental_scorer,
            module_idx,
            float(proposal["xy"][0]),
            float(proposal["xy"][1]),
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
        if old_net3_idx.size:
            net_route_row_parts.append(np.full(old_net3_idx.size, row, dtype=np.int64))
            net_route_col_parts.append(old_net3_idx)
            net_route_v_parts.append(-old_net3_v)
            net_route_h_parts.append(-old_net3_h)
        if new_net3_idx.size:
            net_route_row_parts.append(np.full(new_net3_idx.size, row, dtype=np.int64))
            net_route_col_parts.append(new_net3_idx)
            net_route_v_parts.append(new_net3_v)
            net_route_h_parts.append(new_net3_h)
        if old_netx_idx.size:
            net_route_row_parts.append(np.full(old_netx_idx.size, row, dtype=np.int64))
            net_route_col_parts.append(old_netx_idx)
            net_route_v_parts.append(-old_netx_v)
            net_route_h_parts.append(-old_netx_h)
        if new_netx_idx.size:
            net_route_row_parts.append(np.full(new_netx_idx.size, row, dtype=np.int64))
            net_route_col_parts.append(new_netx_idx)
            net_route_v_parts.append(new_netx_v)
            net_route_h_parts.append(new_netx_h)

    with torch.no_grad():
        base = torch.as_tensor(
            incremental_scorer.grid_occupied,
            device=dev,
            dtype=torch.float32,
        )
        occ = base.unsqueeze(0).repeat(n_prop, 1)
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

        v_macro = torch.as_tensor(
            incremental_scorer.V_macro_flat,
            device=dev,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(n_prop, 1)
        h_macro = torch.as_tensor(
            incremental_scorer.H_macro_flat,
            device=dev,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(n_prop, 1)
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

        v_raw = torch.as_tensor(
            incremental_scorer.V_flat,
            device=dev,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(n_prop, 1)
        h_raw = torch.as_tensor(
            incremental_scorer.H_flat,
            device=dev,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(n_prop, 1)
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

        i_t = torch.as_tensor(idx, device=dev, dtype=torch.long)
        t_t = torch.as_tensor(target_idx, device=dev, dtype=torch.long)
        nx_t = torch.as_tensor(nx, device=dev)
        ny_t = torch.as_tensor(ny, device=dev)
        rank_t = torch.as_tensor(candidate_rank, device=dev)
        pos_t = torch.as_tensor(pos, device=dev, dtype=torch.float32)
        local_t = torch.as_tensor(local_cong, device=dev, dtype=torch.float32)
        tgt_t = torch.as_tensor(tgt_cong, device=dev, dtype=torch.float32)

        field_max = torch.clamp(torch.max(local_t), min=1e-12)
        relief = torch.clamp((local_t[i_t] - tgt_t[t_t]) / field_max, min=0.0)
        dx = (nx_t - pos_t[i_t, 0]) / max(float(cw), 1e-12)
        dy = (ny_t - pos_t[i_t, 1]) / max(float(ch), 1e-12)
        dist = torch.sqrt(dx * dx + dy * dy)
        rank_norm = rank_t / torch.clamp(torch.max(rank_t), min=1.0)

        wl_delta = torch.zeros(n_prop, device=dev, dtype=torch.float32)
        pos_cache_np = _ensure_pos_cache(incremental_scorer.plc)
        if pos_cache_np is not None and incremental_scorer.n_nets > 0:
            pos_cache_t = torch.as_tensor(pos_cache_np, device=dev, dtype=torch.float32)
            unique_ref_t = torch.as_tensor(incremental_scorer.unique_ref, device=dev, dtype=torch.long)
            node_pos_t = pos_cache_t[unique_ref_t]
            ref_inv_t = torch.as_tensor(incremental_scorer.ref_inv, device=dev, dtype=torch.long)
            x_off_t = torch.as_tensor(incremental_scorer.x_off, device=dev, dtype=torch.float32)
            y_off_t = torch.as_tensor(incremental_scorer.y_off, device=dev, dtype=torch.float32)
            per_net_hpwl_t = torch.as_tensor(
                incremental_scorer.per_net_hpwl,
                device=dev,
                dtype=torch.float32,
            )
            net_weights_t = torch.as_tensor(
                incremental_scorer.net_weights,
                device=dev,
                dtype=torch.float32,
            )
            for row, proposal in enumerate(proposals):
                i = int(proposal["i"])
                module_idx = int(incremental_scorer.hard_indices[i])
                nets = incremental_scorer.macro_to_nets.get(module_idx)
                if nets is None or len(nets) == 0:
                    continue
                starts_t = incremental_scorer.net_starts[nets]
                lengths_t = incremental_scorer.net_lengths[nets]
                total = int(lengths_t.sum())
                if total == 0:
                    continue
                pin_indices_np = np.repeat(starts_t, lengths_t) + (
                    np.arange(total) - np.repeat(
                        np.concatenate([[0], np.cumsum(lengths_t)[:-1]]),
                        lengths_t,
                    )
                )
                pin_indices_t = torch.as_tensor(pin_indices_np, device=dev, dtype=torch.long)
                ref_local = ref_inv_t[pin_indices_t]
                pin_x = node_pos_t[ref_local, 0] + x_off_t[pin_indices_t]
                pin_y = node_pos_t[ref_local, 1] + y_off_t[pin_indices_t]
                moved_mask = unique_ref_t[ref_local] == int(module_idx)
                if torch.any(moved_mask):
                    pin_x = torch.where(moved_mask, nx_t[row] + x_off_t[pin_indices_t], pin_x)
                    pin_y = torch.where(moved_mask, ny_t[row] + y_off_t[pin_indices_t], pin_y)
                sub_starts = np.concatenate([[0], np.cumsum(lengths_t)[:-1]])
                sub_starts_t = torch.as_tensor(sub_starts, device=dev, dtype=torch.long)
                net_hpwl = []
                for start, length in zip(sub_starts_t.tolist(), lengths_t.tolist()):
                    end = int(start) + int(length)
                    if end <= int(start):
                        net_hpwl.append(torch.tensor(0.0, device=dev))
                    else:
                        px = pin_x[int(start):end]
                        py = pin_y[int(start):end]
                        net_hpwl.append((torch.max(px) - torch.min(px)) + (torch.max(py) - torch.min(py)))
                new_hpwl = torch.stack(net_hpwl) if net_hpwl else torch.zeros(0, device=dev)
                nets_t = torch.as_tensor(nets, device=dev, dtype=torch.long)
                delta_raw = torch.sum((new_hpwl - per_net_hpwl_t[nets_t]) * net_weights_t[nets_t])
                wl_delta[row] = delta_raw / float(incremental_scorer.wl_normalizer)

        # Lower is better. WL is a delta (base constant omitted); density and
        # congestion are full post-move costs. Net routing deltas are emitted
        # with the same 2-pin/3-pin/high-fanout reduction as the exact scorer.
        score = wl_delta + 0.5 * density + 0.5 * congestion - 0.12 * relief + 0.02 * dist + 0.005 * rank_norm
        scores = score.detach().cpu().numpy().astype(np.float64)
    for proposal, score_value in zip(proposals, scores):
        proposal["score"] = float(score_value)


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
) -> "tuple[np.ndarray, int, float]":
    """Experimental hard-relocation propose-all path.

    Phase 1 freezes the current state, exact-scores every legal proposal for all
    hot macros, and globally ranks those proposals. Phase 2 walks that ranked
    list, re-checks legality against the current post-commit state, exact-scores
    again, and commits only strict improvements. The exact scorer remains the
    arbiter; this only changes the proposal/ordering policy.
    """
    best_score = initial_score
    accepts = 0
    all_idx = np.arange(n)
    proposals = []
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
        cand = np.where(tgt_cong < local_cong[i] - 1e-9)[0]
        if cand.size == 0:
            continue
        d2 = (tgt_x[cand] - pos[i, 0]) ** 2 + (tgt_y[cand] - pos[i, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[i, 0]) ** 2 + (tgt_y[cand] - net_centroid[i, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        cand = cand[np.argsort(d2)][:n_targets]

        mask = all_idx != i
        sxi = sep_x_mat[i, mask]
        syi = sep_y_mat[i, mask]
        ox = pos[mask, 0]
        oy = pos[mask, 1]
        legal = []
        for candidate_rank, t in enumerate(cand):
            nx, ny = float(tgt_x[t]), float(tgt_y[t])
            if (nx - hw[i] < -eps or nx + hw[i] > cw + eps or
                    ny - hh[i] < -eps or ny + hh[i] > ch + eps):
                continue
            if ((np.abs(nx - ox) < sxi + eps) & (np.abs(ny - oy) < syi + eps)).any():
                continue
            legal.append((int(candidate_rank), int(t), nx, ny))
        if not legal:
            continue
        legal_count += len(legal)

        if scorer_mode == "exact":
            prep = incremental_scorer._prepare_move(i)
            try:
                for candidate_rank, target_index, nx, ny in legal:
                    if deadline is not None and time.monotonic() > deadline:
                        break
                    score = incremental_scorer._trial_at(prep, (nx, ny))
                    frozen_scores += 1
                    proposals.append(
                        {
                            "score": float(score),
                            "i": i,
                            "hot_rank": int(hot_rank),
                            "candidate_rank": int(candidate_rank),
                            "target_index": int(target_index),
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
    if propose_top_m is not None and propose_top_m > 0:
        proposals = proposals[:propose_top_m]

    moved = set()
    for p in proposals:
        if deadline is not None and time.monotonic() > deadline:
            break
        i = int(p["i"])
        if i in moved or not movable[i]:
            continue
        nx, ny = p["xy"]
        if (nx - hw[i] < -eps or nx + hw[i] > cw + eps or
                ny - hh[i] < -eps or ny + hh[i] > ch + eps):
            continue
        mask = all_idx != i
        if ((np.abs(nx - pos[mask, 0]) < sep_x_mat[i, mask] + eps) &
                (np.abs(ny - pos[mask, 1]) < sep_y_mat[i, mask] + eps)).any():
            continue

        prep = incremental_scorer._prepare_move(i)
        try:
            score = incremental_scorer._trial_at(prep, (nx, ny))
            verify_scores += 1
            if score < best_score - 1e-9:
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

    if os.environ.get("V2_RELOC_PROPOSE_LOG", "").strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}:
        print(
            "  R2 propose-all[%s]: hot=%d legal=%d frozen_scores=%d "
            "selected=%d verify_scores=%d accepts=%d scorer=%s elapsed=%.3fs"
            % (
                field,
                len(hot),
                legal_count,
                frozen_scores,
                len(proposals),
                verify_scores,
                accepts,
                scorer_mode,
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
) -> "tuple[np.ndarray, int, float]":
    """Congestion-directed single-macro RELOCATION of hard macros.

    The move 2-opt can't make: it only swaps two macros, never moving a
    routing-heavy macro into an empty gap. For the hottest macros (by the chosen
    field) this trials each into a few of the lowest-field legal cell centers and
    accepts iff the true incremental proxy strictly drops. Legal = in-bounds + no
    overlap with other HARD macros (softs may overlap, ignored).

    use_density: field is grid occupancy (True) vs max(H,V) routing congestion
        (False). use_combined: geometric mean of both (normalized), favouring
        macros moderately hot on both.
    net_centroid / wl_blend: blend distance-to-current with distance-to-WL-anchor
        in target ordering (0 = nearest-to-current).
    Returns (pos, accepts, best_score).
    """
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    trace = get_candidate_trace()
    filter_hard_relocation = is_filter_enabled("hard_relocation")
    calibrate_hard_relocation = is_filter_calibration_enabled("hard_relocation")
    trace_field = "combined" if use_combined else ("density" if use_density else "congestion")
    if use_combined:
        # cong and density each normalized to [0,1], geometric-meaned: a cell
        # ranks hot only if both terms are high.
        cong_field = _congestion_field(plc, nr, nc)
        dens_field = _density_field(incremental_scorer, nr, nc)
        if cong_field is None or dens_field is None:
            return pos, 0, initial_score
        cong_max = max(float(cong_field.max()), 1e-12)
        dens_max = max(float(dens_field.max()), 1e-12)
        cell_cong = np.sqrt((cong_field / cong_max) * (dens_field / dens_max))
    else:
        cell_cong = (_density_field(incremental_scorer, nr, nc) if use_density
                     else _congestion_field(plc, nr, nc))
        if cell_cong is None:
            return pos, 0, initial_score
    tf = None
    if trace is not None or filter_hard_relocation:
        tf = TraceFields(
            cong=_congestion_field(plc, nr, nc),
            dens=_density_field(incremental_scorer, nr, nc),
        )
    cell_w, cell_h = cw / nc, ch / nr
    field_max = max(float(cell_cong.max()), 1e-12)

    # Per-macro local congestion → pick the hottest movable macros to relocate.
    ci_all = np.clip((pos[:n, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri_all = np.clip((pos[:n, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_cong[ri_all, ci_all]
    mov_idx = np.where(movable)[0]
    if mov_idx.size == 0:
        return pos, 0, initial_score
    hot = mov_idx[np.argsort(-local_cong[mov_idx])][:top_hot]

    # Target pool = low-field cell centers. A percentile threshold (not the
    # globally-coldest N) keeps medium-cold cells near each hot macro in play.
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

    # Experimental cross-macro path: score the whole frozen-base proposal pool,
    # then serially re-verify/commit the best proposals against current state.
    # Keep tracing/ML-filter runs on the legacy path for now so diagnostics keep
    # their existing per-group semantics.
    if propose_all and trace is None and not filter_hard_relocation and not calibrate_hard_relocation:
        return _relocation_moves_propose_all(
            pos=pos,
            sizes=sizes,
            hw=hw,
            hh=hh,
            cw=cw,
            ch=ch,
            movable=movable,
            n=n,
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
        )

    best_score = initial_score
    accepts = 0
    all_idx = np.arange(n)
    for i in hot:
        i = int(i)
        if deadline is not None and time.monotonic() > deadline:
            break
        if not movable[i]:
            continue
        # Only consider targets at lower congestion than the macro's current cell
        # (relief), nearest-first among those, capped at n_targets.
        cand = np.where(tgt_cong < local_cong[i] - 1e-9)[0]
        if cand.size == 0:
            continue
        d2 = (tgt_x[cand] - pos[i, 0]) ** 2 + (tgt_y[cand] - pos[i, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[i, 0]) ** 2 + (tgt_y[cand] - net_centroid[i, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        cand = cand[np.argsort(d2)][:n_targets]

        mask = all_idx != i
        sxi = sep_x_mat[i, mask]
        syi = sep_y_mat[i, mask]
        ox = pos[mask, 0]
        oy = pos[mask, 1]
        # Prep i once (subtract its routing+density), trial each candidate, then
        # commit the winner or revert. Saves a routing-apply per trial (~30%/move).
        prep = incremental_scorer._prepare_move(i)
        best_i_xy = None
        state_score = best_score
        group_id = trace.next_group_id("hard_relocation") if trace is not None else None
        rejected_bounds = 0
        rejected_overlap = 0
        scored = 0
        legal_candidates = []
        shadow_candidates = []
        try:
            for candidate_rank, t in enumerate(cand):
                nx, ny = float(tgt_x[t]), float(tgt_y[t])
                if (nx - hw[i] < -EPS or nx + hw[i] > cw + EPS or
                        ny - hh[i] < -EPS or ny + hh[i] > ch + EPS):
                    rejected_bounds += 1
                    continue
                # Overlap vs other HARD macros (vectorized).
                if ((np.abs(nx - ox) < sxi + EPS) & (np.abs(ny - oy) < syi + EPS)).any():
                    rejected_overlap += 1
                    continue
                features = None
                if trace is not None or filter_hard_relocation:
                    target_flat = int(pool[t])
                    features = {
                        **net_degree_features(
                            incremental_scorer, incremental_scorer.hard_indices[i]
                        ),
                        "accepted_in_pass": accepts,
                        "macro_w_norm": float(sizes[i, 0] / cw),
                        "macro_h_norm": float(sizes[i, 1] / ch),
                        "x_norm": float(pos[i, 0] / cw),
                        "y_norm": float(pos[i, 1] / ch),
                        "target_x_norm": float(nx / cw),
                        "target_y_norm": float(ny / ch),
                        "dx_norm": float((nx - pos[i, 0]) / cw),
                        "dy_norm": float((ny - pos[i, 1]) / ch),
                        "source_field_norm": float(local_cong[i] / field_max),
                        "target_field_norm": float(tgt_cong[t] / field_max),
                        "source_congestion_norm": tf.cong_at(ri_all[i], ci_all[i]),
                        "target_congestion_norm": tf.cong_flat(target_flat),
                        "source_density_norm": tf.dens_at(ri_all[i], ci_all[i]),
                        "target_density_norm": tf.dens_flat(target_flat),
                        "source_hot_rank_norm": float(
                            np.where(hot == i)[0][0] / max(len(hot) - 1, 1)
                        ),
                        "target_cold_rank_norm": float(candidate_rank / max(len(cand) - 1, 1)),
                    }
                legal_candidates.append(
                    {
                        "target_index": int(t),
                        "candidate_rank": int(candidate_rank),
                        "nx": nx,
                        "ny": ny,
                        "features": features,
                    }
                )

            candidate_views = [
                {
                    "operator": "hard_relocation",
                    "features": item["features"] or {},
                    "candidate_rank": item["candidate_rank"],
                }
                for item in legal_candidates
            ]
            selected_indices = filter_candidate_indices(
                operator="hard_relocation",
                candidates=candidate_views,
                trace=trace,
                field=trace_field,
                group_id=group_id,
            )
            selected_set = set(selected_indices)

            for legal_index, item in enumerate(legal_candidates):
                if legal_index not in selected_set:
                    continue
                nx = item["nx"]
                ny = item["ny"]
                candidate_rank = item["candidate_rank"]
                features = item["features"]
                s = incremental_scorer._trial_at(prep, (nx, ny))
                scored += 1
                if trace is not None:
                    trace.record(
                        operator="hard_relocation",
                        field=trace_field,
                        group_id=group_id,
                        state_score=state_score,
                        trial_score=s,
                        candidate_rank=candidate_rank,
                        group_size=len(cand),
                        candidate_source="cold_cell",
                        features=features or {},
                    )
                if trace is not None or calibrate_hard_relocation:
                    shadow_candidates.append(
                        {
                            "operator": "hard_relocation",
                            "features": features or {},
                            "candidate_rank": candidate_rank,
                            "score_gain": float(state_score - s),
                        }
                    )
                if s < best_score - 1e-9:
                    best_score = s
                    best_i_xy = (nx, ny)
            if best_i_xy is not None:
                incremental_scorer._commit_after_prep(prep, best_i_xy)
                pos[i, 0], pos[i, 1] = best_i_xy
                accepts += 1
            else:
                incremental_scorer._revert_prep(prep)
            update_filter_calibration(
                operator="hard_relocation",
                candidates=shadow_candidates,
                trace=trace,
                field=trace_field,
                group_id=group_id,
            )
            if trace is not None:
                shadow_rank_group(
                    operator="hard_relocation",
                    candidates=shadow_candidates,
                    trace=trace,
                    field=trace_field,
                    group_id=group_id,
                )
                trace.event(
                    "candidate_group_summary",
                    operator="hard_relocation",
                    field=trace_field,
                    group_id=group_id,
                    generated=int(len(cand)),
                    scored=scored,
                    rejected_bounds=rejected_bounds,
                    rejected_overlap=rejected_overlap,
                    skipped_by_ml=max(0, len(legal_candidates) - scored),
                )
        except Exception:
            # Defensive: restore committed state if anything blew up mid-trial.
            incremental_scorer._revert_prep(prep)
            raise
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
) -> "tuple[np.ndarray, int, float]":
    """Congestion-directed SOFT-macro relocation.

    Relocates the hottest movable soft clusters into low-field cells, accept-on-
    true-proxy via the soft prep/trial path. Softs may overlap, so no conflict
    check - just a half-size clip to keep them in bounds.

    use_density: occupancy field (True) vs max(H,V) congestion (False). Softs are
        the bulk of density and may overlap, so a density pass finds moves the
        cong pass can't.
    net_centroid / wl_blend: blend toward the WL anchor in target ordering
        (0 = nearest-to-current).
    `soft_pos` is [num_soft, 2] canvas coords. Returns (soft_pos, accepts, best_score).
    """
    num_soft = incremental_scorer.num_soft
    if num_soft == 0:
        return soft_pos, 0, initial_score
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    trace = get_candidate_trace()
    trace_field = "density" if use_density else "congestion"
    cell_field = (_density_field(incremental_scorer, nr, nc) if use_density
                  else _congestion_field(plc, nr, nc))
    if cell_field is None:
        return soft_pos, 0, initial_score
    tf = None
    if trace is not None:
        tf = TraceFields(
            cong=_congestion_field(plc, nr, nc),
            dens=_density_field(incremental_scorer, nr, nc),
        )
    cell_w, cell_h = cw / nc, ch / nr
    field_max = max(float(cell_field.max()), 1e-12)

    ci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_field[ri, ci]
    # Only relocate MOVABLE softs - fixed macros must stay put (contract). The
    # IBM benchmarks have 0 fixed softs (no-op here), but NG45/other inputs may.
    order = np.argsort(-local_cong)
    if soft_movable is not None:
        sm = np.asarray(soft_movable, dtype=bool)
        order = order[sm[order]]
    hot = order[:top_hot]

    flat = cell_field.ravel()
    # Target pool = low-field cell centers (percentile threshold, not globally-
    # coldest N, so medium-cold cells near each hot soft stay in play).
    _thr = np.percentile(flat, 55)
    pool = np.where(flat < _thr)[0]
    if pool.size < max(n_targets, 64):
        pool = np.argsort(flat)[: max(n_targets, 64)]
    tgt_x = ((pool % nc).astype(np.float64) + 0.5) * cell_w
    tgt_y = ((pool // nc).astype(np.float64) + 0.5) * cell_h
    tgt_cong = flat[pool]

    best_score = initial_score
    accepts = 0
    for k in hot:
        k = int(k)
        if deadline is not None and time.monotonic() > deadline:
            break
        cand = np.where(tgt_cong < local_cong[k] - 1e-9)[0]
        if cand.size == 0:
            continue
        # Order targets by distance, optionally blended toward the WL anchor.
        d2 = (tgt_x[cand] - soft_pos[k, 0]) ** 2 + (tgt_y[cand] - soft_pos[k, 1]) ** 2
        if wl_blend > 0.0 and net_centroid is not None:
            d2c = (tgt_x[cand] - net_centroid[k, 0]) ** 2 + (tgt_y[cand] - net_centroid[k, 1]) ** 2
            d2 = (1.0 - wl_blend) * d2 + wl_blend * d2c
        cand = cand[np.argsort(d2)][:n_targets]
        # Prep k once, trial each candidate, commit-or-revert (see _relocation_moves).
        prep = incremental_scorer._prepare_move_soft(k)
        best_k_xy = None
        state_score = best_score
        group_id = trace.next_group_id("soft_relocation") if trace is not None else None
        shadow_candidates = []
        try:
            for candidate_rank, t in enumerate(cand):
                nx = float(np.clip(tgt_x[t], soft_hw[k], cw - soft_hw[k]))
                ny = float(np.clip(tgt_y[t], soft_hh[k], ch - soft_hh[k]))
                s = incremental_scorer._trial_at_soft(prep, (nx, ny))
                if trace is not None:
                    target_flat = int(pool[t])
                    features = {
                        **net_degree_features(
                            incremental_scorer, incremental_scorer.soft_indices[k]
                        ),
                        "accepted_in_pass": accepts,
                        "macro_w_norm": float(2.0 * soft_hw[k] / cw),
                        "macro_h_norm": float(2.0 * soft_hh[k] / ch),
                        "x_norm": float(soft_pos[k, 0] / cw),
                        "y_norm": float(soft_pos[k, 1] / ch),
                        "target_x_norm": float(nx / cw),
                        "target_y_norm": float(ny / ch),
                        "dx_norm": float((nx - soft_pos[k, 0]) / cw),
                        "dy_norm": float((ny - soft_pos[k, 1]) / ch),
                        "source_field_norm": float(local_cong[k] / field_max),
                        "target_field_norm": float(tgt_cong[t] / field_max),
                        "source_congestion_norm": tf.cong_at(ri[k], ci[k]),
                        "target_congestion_norm": tf.cong_flat(target_flat),
                        "source_density_norm": tf.dens_at(ri[k], ci[k]),
                        "target_density_norm": tf.dens_flat(target_flat),
                        "source_hot_rank_norm": float(
                            np.where(hot == k)[0][0] / max(len(hot) - 1, 1)
                        ),
                        "target_cold_rank_norm": float(candidate_rank / max(len(cand) - 1, 1)),
                    }
                    trace.record(
                        operator="soft_relocation",
                        field=trace_field,
                        group_id=group_id,
                        state_score=state_score,
                        trial_score=s,
                        candidate_rank=candidate_rank,
                        group_size=len(cand),
                        candidate_source="cold_cell",
                        features=features,
                    )
                    shadow_candidates.append(
                        {
                            "operator": "soft_relocation",
                            "features": features,
                            "candidate_rank": candidate_rank,
                            "score_gain": float(state_score - s),
                        }
                    )
                if s < best_score - 1e-9:
                    best_score = s
                    best_k_xy = (nx, ny)
            if best_k_xy is not None:
                incremental_scorer._commit_after_prep_soft(prep, best_k_xy)
                soft_pos[k, 0], soft_pos[k, 1] = best_k_xy
                accepts += 1
            else:
                incremental_scorer._revert_prep_soft(prep)
            if trace is not None:
                shadow_rank_group(
                    operator="soft_relocation",
                    candidates=shadow_candidates,
                    trace=trace,
                    field=trace_field,
                    group_id=group_id,
                )
        except Exception:
            incremental_scorer._revert_prep_soft(prep)
            raise
    return soft_pos, accepts, best_score
