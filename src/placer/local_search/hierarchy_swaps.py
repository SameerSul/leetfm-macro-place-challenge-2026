"""Region-bounded swap relief for the hierarchy placement path."""

from __future__ import annotations

import time

import numpy as np
import torch

from utils import constants as const
from utils.config import _GPU_BACKEND, _GPU_DEVICE
from placer.shared.geometry import separation_matrices
from placer.local_search.fields import _congestion_field, _density_field, weighted_congestion_field
from placer.local_search.gnn_trace import gnn_trace_limit, log_gnn_event
from placer.local_search.region_rules import accepts_region_score


def _new_stats() -> dict:
    return {
        "hh_scores": 0,
        "hh_accepts": 0,
        "hh_escape_accepts": 0,
        "hs_scores": 0,
        "hs_accepts": 0,
        "hs_escape_accepts": 0,
        "ss_scores": 0,
        "ss_accepts": 0,
        "ss_escape_accepts": 0,
        "proxy_gain": 0.0,
    }


def _accept_swap(score, best_score, outside_region, escape_min, min_gain) -> bool:
    if not accepts_region_score(score, best_score, outside_region, max(escape_min, min_gain)):
        return False
    return float(score) < float(best_score) - float(min_gain)


def _record_accept(stats, kind: str, outside_region: bool, old_score: float, new_score: float):
    stats[f"{kind}_accepts"] += 1
    if outside_region:
        stats[f"{kind}_escape_accepts"] += 1
    stats["proxy_gain"] += max(0.0, float(old_score) - float(new_score))


def _cell_values(pos: np.ndarray, field: np.ndarray, cw: float, ch: float) -> np.ndarray:
    nr, nc = field.shape
    cell_w, cell_h = cw / nc, ch / nr
    ci = np.clip((pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    return field[ri, ci]


def _auto_gpu_enabled(value) -> bool:
    if isinstance(value, str) and value.lower() == "auto":
        return _GPU_BACKEND == "cuda"
    return bool(value)


def _rank_smallest(values: np.ndarray) -> np.ndarray:
    """Return ascending rank order, using CUDA only for large arrays."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if (
        not _auto_gpu_enabled(const.HIER_GPU_RANK_SWAP_CANDIDATES)
        or _GPU_BACKEND != "cuda"
        or n < int(const.HIER_GPU_RANK_MIN_CANDIDATES)
    ):
        return np.argsort(arr)
    try:
        with torch.inference_mode():
            vals = torch.as_tensor(arr, device=_GPU_DEVICE, dtype=torch.float32)
            return torch.argsort(vals, stable=True).cpu().numpy().astype(np.int64, copy=False)
    except Exception:
        return np.argsort(arr)


def _gpu_swap_prescore_order(
    base_rank: np.ndarray,
    *,
    enabled,
    src_x: float,
    src_y: float,
    tgt_x: np.ndarray,
    tgt_y: np.ndarray,
    field_span: float,
    distance_span: float,
) -> np.ndarray:
    """Rank swap candidates with a lightweight CUDA heuristic before exact scoring."""
    base = np.asarray(base_rank, dtype=np.float64).reshape(-1)
    n = int(base.size)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if (
        not _auto_gpu_enabled(enabled)
        or _GPU_BACKEND != "cuda"
        or n < int(const.HIER_GPU_SWAP_PRESCORE_MIN_CANDIDATES)
    ):
        return _rank_smallest(base)
    try:
        with torch.inference_mode():
            tx = torch.as_tensor(tgt_x, device=_GPU_DEVICE, dtype=torch.float32)
            ty = torch.as_tensor(tgt_y, device=_GPU_DEVICE, dtype=torch.float32)
            rank = torch.as_tensor(base, device=_GPU_DEVICE, dtype=torch.float32)
            dx = tx - float(src_x)
            dy = ty - float(src_y)
            span2 = max(float(distance_span) ** 2, 1.0)
            dist = (dx.square() + dy.square()) / span2
            rank = (
                rank
                + float(const.HIER_GPU_SWAP_PRESCORE_DISTANCE_WEIGHT)
                * max(float(field_span), 1e-12)
                * dist
            )
            return torch.argsort(rank, stable=True).cpu().numpy().astype(np.int64, copy=False)
    except Exception:
        return _rank_smallest(base)


def _outside_region_mask(region_bbox, idx, x, y) -> np.ndarray:
    """Vectorized equivalent of `not point_in_region(...)`."""
    idx_arr = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx_arr.size == 0:
        return np.zeros(0, dtype=np.bool_)
    if region_bbox is None:
        return np.zeros(idx_arr.size, dtype=np.bool_)
    rb = region_bbox[idx_arr]
    x_arr = np.broadcast_to(np.asarray(x, dtype=np.float64), idx_arr.shape)
    y_arr = np.broadcast_to(np.asarray(y, dtype=np.float64), idx_arr.shape)
    return (x_arr < rb[:, 0]) | (x_arr > rb[:, 2]) | (y_arr < rb[:, 1]) | (y_arr > rb[:, 3])


def _swap_rejection_reason(
    *,
    legal: bool,
    in_bounds: bool = True,
    outside_region: bool = False,
    accepted_by_region_gate: bool = True,
    scored: bool = True,
) -> str:
    if not in_bounds:
        return "out_of_bounds"
    if not legal:
        return "illegal_overlap"
    if not scored:
        return "not_scored"
    if outside_region and not accepted_by_region_gate:
        return "out_of_hierarchy_region"
    return "exact_proxy_failed"


def _log_swap_candidates(
    benchmark_name: str,
    kind: str,
    field_name: str,
    source: int,
    initial_proxy: float,
    rows: list[dict],
) -> None:
    limit = gnn_trace_limit()
    if limit <= 0 or not rows:
        return
    log_gnn_event(
        "hier_swap_candidates",
        benchmark=benchmark_name,
        operator="region_swaps",
        kind=kind,
        field=field_name,
        source=int(source),
        initial_proxy=float(rows[0].get("old_proxy", initial_proxy)),
        candidate_count=int(len(rows)),
        candidates=rows[:limit],
    )


def _rank_swap_candidates(
    rows: list[dict],
    *,
    benchmark_name: str,
    kind: str,
    field_name: str,
    source: int,
) -> list[dict]:
    if not rows:
        return rows
    from placer.local_search.gnn_ranker import reorder_region_swap_candidates

    return reorder_region_swap_candidates(
        rows,
        benchmark_name=benchmark_name,
        kind=kind,
        field=field_name,
        source=int(source),
    )


def _hierarchy_aware_swap_filter(cand, outside, source_field, target_field, span, *, enabled: bool):
    """Prefer in-region swaps unless outside relief is materially stronger."""
    if not enabled or cand.size <= 1:
        return cand, outside
    outside = np.asarray(outside, dtype=bool)
    inside = ~outside
    if not inside.any():
        return cand, outside
    relief = float(source_field) - np.asarray(target_field, dtype=np.float64)
    best_inside = float(np.max(relief[inside]))
    margin = float(const.HIER_PROPOSAL_OUTSIDE_RELIEF_MARGIN) * max(float(span), 1e-12)
    keep = inside | (relief >= best_inside + margin)
    if not keep.any():
        return cand, outside
    return cand[keep], outside[keep]


def _in_bounds(x: float, y: float, hw: float, hh: float, cw: float, ch: float) -> bool:
    return bool(hw <= x <= cw - hw and hh <= y <= ch - hh)


def _legal_hard_hard_candidates(
    h_pos: np.ndarray,
    sep_x_mat: np.ndarray,
    sep_y_mat: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    i: int,
    cand: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return legality mask for swapping hard macro `i` with each `cand` candidate."""
    cand = np.asarray(cand, dtype=np.int64)
    if cand.size == 0:
        return np.zeros(0, dtype=np.bool_)

    n = h_pos.shape[0]
    all_idx = np.arange(n, dtype=np.int64)
    cand_i_x = h_pos[cand, 0]
    cand_i_y = h_pos[cand, 1]
    cand_j_x = h_pos[i, 0]
    cand_j_y = h_pos[i, 1]
    hw_i = float(hw[i])
    hh_i = float(hh[i])
    hw_j = hw[cand].astype(np.float64, copy=False)
    hh_j = hh[cand].astype(np.float64, copy=False)

    in_bounds_i = (
        (cand_i_x - hw_i >= 0.0)
        & (cand_i_x + hw_i <= cw)
        & (cand_i_y - hh_i >= 0.0)
        & (cand_i_y + hh_i <= ch)
    )
    in_bounds_j = (
        (cand_j_x - hw_j >= 0.0)
        & (cand_j_x + hw_j <= cw)
        & (cand_j_y - hh_j >= 0.0)
        & (cand_j_y + hh_j <= ch)
    )

    # Move i to cand position: check overlap with all existing hards (except i and cand).
    sep_x_i = sep_x_mat[i, :][None, :]
    sep_y_i = sep_y_mat[i, :][None, :]
    x_block = h_pos[:, 0][None, :]
    y_block = h_pos[:, 1][None, :]
    overlap_i = (np.abs(cand_i_x[:, None] - x_block) < (sep_x_i + eps)) & (
        np.abs(cand_i_y[:, None] - y_block) < (sep_y_i + eps)
    )
    overlap_i[:, i] = False
    overlap_i[np.arange(cand.size), cand] = False
    legal_i = ~np.any(overlap_i, axis=1)

    # Move cand to old i position: check overlap with all existing hards (except cand and i).
    sep_x_j = sep_x_mat[cand][:, all_idx]
    sep_y_j = sep_y_mat[cand][:, all_idx]
    overlap_j = (np.abs(cand_j_x - h_pos[:, 0][None, :]) < (sep_x_j + eps)) & (
        np.abs(cand_j_y - h_pos[:, 1][None, :]) < (sep_y_j + eps)
    )
    overlap_j[:, i] = False
    overlap_j[np.arange(cand.size), cand] = False
    legal_j = ~np.any(overlap_j, axis=1)

    return in_bounds_i & in_bounds_j & legal_i & legal_j


def _legal_hard_soft_candidates(
    h_pos: np.ndarray,
    s_pos: np.ndarray,
    sep_x_mat: np.ndarray,
    sep_y_mat: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    i: int,
    cand: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return legality mask for swapping hard `i` with each soft candidate."""
    cand = np.asarray(cand, dtype=np.int64)
    if cand.size == 0:
        return np.zeros(0, dtype=np.bool_)

    x_i = float(h_pos[i, 0])
    y_i = float(h_pos[i, 1])
    k_x = s_pos[cand, 0]
    k_y = s_pos[cand, 1]
    hw_i = float(hw[i])
    hh_i = float(hh[i])
    hw_k = soft_hw[cand].astype(np.float64, copy=False)
    hh_k = soft_hh[cand].astype(np.float64, copy=False)

    in_bounds_i = (
        (k_x - hw_i >= 0.0) & (k_x + hw_i <= cw) & (k_y - hh_i >= 0.0) & (k_y + hh_i <= ch)
    )
    in_bounds_k = (
        (x_i - hw_k >= 0.0) & (x_i + hw_k <= cw) & (y_i - hh_k >= 0.0) & (y_i + hh_k <= ch)
    )

    # Hard move (i -> soft slot): reject overlaps with all other hards except i.
    sep_x_i = sep_x_mat[i, :][None, :]
    sep_y_i = sep_y_mat[i, :][None, :]
    x_block = h_pos[:, 0][None, :]
    y_block = h_pos[:, 1][None, :]
    overlap_i = (np.abs(k_x[:, None] - x_block) < (sep_x_i + eps)) & (
        np.abs(k_y[:, None] - y_block) < (sep_y_i + eps)
    )
    overlap_i[:, i] = False
    legal_i = ~np.any(overlap_i, axis=1)

    return in_bounds_i & in_bounds_k & legal_i


def _hard_at_legal(
    pos: np.ndarray,
    sep_x_mat: np.ndarray,
    sep_y_mat: np.ndarray,
    i: int,
    x: float,
    y: float,
    ignore: "set[int] | None" = None,
) -> bool:
    ignore = ignore or set()
    n = pos.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    for j in ignore:
        mask[int(j)] = False
    if not mask.any():
        return True
    return not (
        (np.abs(x - pos[mask, 0]) < sep_x_mat[i, mask] + 1e-6)
        & (np.abs(y - pos[mask, 1]) < sep_y_mat[i, mask] + 1e-6)
    ).any()


def _hard_swap_legal(
    pos: np.ndarray,
    sep_x_mat: np.ndarray,
    sep_y_mat: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    i: int,
    j: int,
) -> bool:
    ix, iy = float(pos[j, 0]), float(pos[j, 1])
    jx, jy = float(pos[i, 0]), float(pos[i, 1])
    if not _in_bounds(ix, iy, hw[i], hh[i], cw, ch):
        return False
    if not _in_bounds(jx, jy, hw[j], hh[j], cw, ch):
        return False
    if abs(ix - jx) < sep_x_mat[i, j] + 1e-6 and abs(iy - jy) < sep_y_mat[i, j] + 1e-6:
        return False
    return _hard_at_legal(pos, sep_x_mat, sep_y_mat, i, ix, iy, {j}) and _hard_at_legal(
        pos, sep_x_mat, sep_y_mat, j, jx, jy, {i}
    )


def _try_hard_hard(
    h_pos,
    sizes,
    hw,
    hh,
    cw,
    ch,
    movable_h,
    scorer,
    best_score,
    field,
    hard_region,
    k_neighbors,
    region_bias,
    escape_min,
    min_gain,
    min_field_relief,
    stats,
    deadline,
    benchmark_name: str,
    field_name: str,
) -> tuple[int, float]:
    accepts = 0
    n = h_pos.shape[0]
    if n < 2:
        return 0, best_score
    local = _cell_values(h_pos, field, cw, ch)
    movable_idx = np.where(movable_h)[0]
    if movable_idx.size < 2:
        return 0, best_score
    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    span = max(float(field.max()), 1e-12)
    hot = movable_idx[np.argsort(-local[movable_idx])][: min(128, movable_idx.size)]

    for i_raw in hot:
        if deadline is not None and time.monotonic() > deadline:
            break
        i = int(i_raw)
        cand = movable_idx[movable_idx != i]
        cand = cand[(local[i] - local[cand]) > min_field_relief * span]
        if cand.size == 0:
            continue
        legal_mask = _legal_hard_hard_candidates(
            h_pos,
            sep_x_mat,
            sep_y_mat,
            hw,
            hh,
            cw,
            ch,
            i,
            cand,
            1e-6,
        )
        outside = _outside_region_mask(
            hard_region,
            np.full(cand.size, i, dtype=np.int64),
            h_pos[cand, 0],
            h_pos[cand, 1],
        ) | _outside_region_mask(hard_region, cand, h_pos[i, 0], h_pos[i, 1])
        cand_before = cand
        cand, outside = _hierarchy_aware_swap_filter(
            cand,
            outside,
            float(local[i]),
            local[cand],
            span,
            enabled=field_name == "weighted_congestion",
        )
        if cand.size != cand_before.size:
            keep = np.isin(cand_before, cand)
            legal_mask = legal_mask[keep]
        rank = local[cand] + region_bias * span * outside.astype(np.float64)
        ranked_idx = _gpu_swap_prescore_order(
            rank,
            enabled=const.HIER_GPU_SWAP_PRESCORE_HH,
            src_x=float(h_pos[i, 0]),
            src_y=float(h_pos[i, 1]),
            tgt_x=h_pos[cand, 0],
            tgt_y=h_pos[cand, 1],
            field_span=span,
            distance_span=max(float(max(cw, ch)), 1.0),
        )[:k_neighbors]
        ranked = cand[ranked_idx]
        ranked_legal = legal_mask[ranked_idx]
        ranked_outside = outside[ranked_idx]
        trace_rows = []
        trace_by_target = {}
        for candidate_rank, (j_raw, is_legal, outside_move) in enumerate(
            zip(ranked, ranked_legal, ranked_outside)
        ):
            j = int(j_raw)
            in_bounds = _in_bounds(
                float(h_pos[j, 0]), float(h_pos[j, 1]), hw[i], hh[i], cw, ch
            ) and _in_bounds(float(h_pos[i, 0]), float(h_pos[i, 1]), hw[j], hh[j], cw, ch)
            row = {
                "candidate_rank": int(candidate_rank),
                "target": j,
                "source_field": float(local[i]),
                "target_field": float(local[j]),
                "outside_region": bool(outside_move),
                "legal": bool(is_legal),
                "old_proxy": float(best_score),
                "candidate_proxy": None,
                "proxy_delta": None,
                "accepted": False,
                "rejection_reason": _swap_rejection_reason(
                    legal=bool(is_legal),
                    in_bounds=bool(in_bounds),
                    outside_region=bool(outside_move),
                    scored=False,
                ),
            }
            trace_by_target[j] = row
            trace_rows.append(row)
        trace_rows = _rank_swap_candidates(
            trace_rows,
            benchmark_name=benchmark_name,
            kind="hard_hard",
            field_name=field_name,
            source=i,
        )
        scored = []
        for row in trace_rows:
            # ranked_legal is the legality mask in ranked candidate order.
            if deadline is not None and time.monotonic() > deadline:
                break
            j = int(row["target"])
            if not bool(row.get("legal", False)):
                continue
            scored.append((j, bool(row.get("outside_region", False))))
        if scored:
            scores = scorer.score_swap_hard_hard_many(
                i, np.asarray([j for j, _ in scored], dtype=np.int64)
            )
            scored_iter = zip(scored, scores)
        else:
            scored_iter = ()
        for (j, outside_move), score in scored_iter:
            stats["hh_scores"] += 1
            region_ok = accepts_region_score(
                score, best_score, outside_move, max(escape_min, min_gain)
            )
            row = trace_by_target.get(int(j))
            if row is not None:
                row["candidate_proxy"] = float(score)
                row["proxy_delta"] = float(score) - float(best_score)
                row["rejection_reason"] = _swap_rejection_reason(
                    legal=True,
                    outside_region=bool(outside_move),
                    accepted_by_region_gate=bool(region_ok),
                )
            if _accept_swap(score, best_score, outside_move, escape_min, min_gain):
                if row is not None:
                    row["accepted"] = True
                    row["rejection_reason"] = None
                scorer.commit_swap_hard_hard(i, j)
                h_pos[[i, j]] = h_pos[[j, i]]
                _record_accept(stats, "hh", outside_move, best_score, score)
                best_score = float(score)
                accepts += 1
                break
        _log_swap_candidates(benchmark_name, "hard_hard", field_name, i, best_score, trace_rows)
    return accepts, best_score


def _try_soft_soft(
    s_pos,
    soft_hw,
    soft_hh,
    cw,
    ch,
    soft_movable,
    scorer,
    best_score,
    field,
    soft_region,
    k_neighbors,
    region_bias,
    escape_min,
    min_gain,
    soft_barrier_gain,
    min_field_relief,
    stats,
    deadline,
    benchmark_name: str,
    field_name: str,
) -> tuple[int, float]:
    accepts = 0
    n_soft = s_pos.shape[0]
    if n_soft < 2:
        return 0, best_score
    local = _cell_values(s_pos, field, cw, ch)
    movable_idx = np.arange(n_soft)
    if soft_movable is not None:
        sm = np.asarray(soft_movable, dtype=bool)
        movable_idx = movable_idx[sm[movable_idx]]
    if movable_idx.size < 2:
        return 0, best_score
    span = max(float(field.max()), 1e-12)
    hot = movable_idx[np.argsort(-local[movable_idx])][: min(256, movable_idx.size)]

    for a_raw in hot:
        if deadline is not None and time.monotonic() > deadline:
            break
        a = int(a_raw)
        cand = movable_idx[movable_idx != a]
        cand = cand[(local[a] - local[cand]) > min_field_relief * span]
        if cand.size == 0:
            continue
        outside = _outside_region_mask(
            soft_region,
            np.full(cand.size, a, dtype=np.int64),
            s_pos[cand, 0],
            s_pos[cand, 1],
        ) | _outside_region_mask(soft_region, cand, s_pos[a, 0], s_pos[a, 1])
        cand, outside = _hierarchy_aware_swap_filter(
            cand,
            outside,
            float(local[a]),
            local[cand],
            span,
            enabled=field_name == "weighted_congestion",
        )
        rank = local[cand] + region_bias * span * outside.astype(np.float64)
        scored = []
        ranked_idx = _gpu_swap_prescore_order(
            rank,
            enabled=const.HIER_GPU_SWAP_PRESCORE_SS,
            src_x=float(s_pos[a, 0]),
            src_y=float(s_pos[a, 1]),
            tgt_x=s_pos[cand, 0],
            tgt_y=s_pos[cand, 1],
            field_span=span,
            distance_span=max(float(max(cw, ch)), 1.0),
        )[:k_neighbors]
        ranked = cand[ranked_idx]
        ranked_outside = outside[ranked_idx]
        trace_rows = []
        trace_by_target = {}
        for candidate_rank, (b_raw, outside_move) in enumerate(zip(ranked, ranked_outside)):
            b = int(b_raw)
            ax, ay = float(s_pos[b, 0]), float(s_pos[b, 1])
            bx, by = float(s_pos[a, 0]), float(s_pos[a, 1])
            in_bounds = _in_bounds(ax, ay, soft_hw[a], soft_hh[a], cw, ch) and _in_bounds(
                bx, by, soft_hw[b], soft_hh[b], cw, ch
            )
            row = {
                "candidate_rank": int(candidate_rank),
                "target": b,
                "source_field": float(local[a]),
                "target_field": float(local[b]),
                "outside_region": bool(outside_move),
                "legal": bool(in_bounds),
                "old_proxy": float(best_score),
                "candidate_proxy": None,
                "proxy_delta": None,
                "accepted": False,
                "rejection_reason": _swap_rejection_reason(
                    legal=True,
                    in_bounds=bool(in_bounds),
                    outside_region=bool(outside_move),
                    scored=False,
                ),
            }
            trace_by_target[b] = row
            trace_rows.append(row)
        trace_rows = _rank_swap_candidates(
            trace_rows,
            benchmark_name=benchmark_name,
            kind="soft_soft",
            field_name=field_name,
            source=a,
        )
        for row in trace_rows:
            if deadline is not None and time.monotonic() > deadline:
                break
            if not bool(row.get("legal", False)):
                continue
            b = int(row["target"])
            ax, ay = float(s_pos[b, 0]), float(s_pos[b, 1])
            bx, by = float(s_pos[a, 0]), float(s_pos[a, 1])
            if not _in_bounds(ax, ay, soft_hw[a], soft_hh[a], cw, ch):
                continue
            if not _in_bounds(bx, by, soft_hw[b], soft_hh[b], cw, ch):
                continue
            scored.append((b, outside_move))
        if scored:
            scores = scorer.score_swap_soft_soft_many(
                a, np.asarray([b for b, _ in scored], dtype=np.int64)
            )
            scored_iter = zip(scored, scores)
        else:
            scored_iter = ()
        for (b, outside_move), score in scored_iter:
            stats["ss_scores"] += 1
            required_gain = max(min_gain, soft_barrier_gain)
            region_ok = accepts_region_score(
                score, best_score, outside_move, max(escape_min, required_gain)
            )
            row = trace_by_target.get(int(b))
            if row is not None:
                row["candidate_proxy"] = float(score)
                row["proxy_delta"] = float(score) - float(best_score)
                row["soft_barrier_gain"] = float(soft_barrier_gain)
                row["rejection_reason"] = _swap_rejection_reason(
                    legal=True,
                    outside_region=bool(outside_move),
                    accepted_by_region_gate=bool(region_ok),
                )
            if _accept_swap(score, best_score, outside_move, escape_min, required_gain):
                if row is not None:
                    row["accepted"] = True
                    row["rejection_reason"] = None
                scorer.commit_swap_soft_soft(a, b)
                s_pos[[a, b]] = s_pos[[b, a]]
                _record_accept(stats, "ss", outside_move, best_score, score)
                best_score = float(score)
                accepts += 1
                break
        _log_swap_candidates(benchmark_name, "soft_soft", field_name, a, best_score, trace_rows)
    return accepts, best_score


def _try_hard_soft(
    h_pos,
    s_pos,
    sizes,
    hw,
    hh,
    soft_hw,
    soft_hh,
    cw,
    ch,
    movable_h,
    soft_movable,
    scorer,
    best_score,
    field,
    hard_region,
    soft_region,
    k_neighbors,
    region_bias,
    escape_min,
    min_gain,
    soft_barrier_gain,
    min_field_relief,
    stats,
    deadline,
    benchmark_name: str,
    field_name: str,
) -> tuple[int, float]:
    accepts = 0
    if h_pos.shape[0] == 0 or s_pos.shape[0] == 0:
        return 0, best_score
    hard_local = _cell_values(h_pos, field, cw, ch)
    soft_local = _cell_values(s_pos, field, cw, ch)
    hard_idx = np.where(movable_h)[0]
    soft_idx = np.arange(s_pos.shape[0])
    if soft_movable is not None:
        sm = np.asarray(soft_movable, dtype=bool)
        soft_idx = soft_idx[sm[soft_idx]]
    if hard_idx.size == 0 or soft_idx.size == 0:
        return 0, best_score
    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    span = max(float(field.max()), 1e-12)
    hot = hard_idx[np.argsort(-hard_local[hard_idx])][: min(128, hard_idx.size)]

    for i_raw in hot:
        if deadline is not None and time.monotonic() > deadline:
            break
        i = int(i_raw)
        cand = soft_idx[(hard_local[i] - soft_local[soft_idx]) > min_field_relief * span]
        if cand.size == 0:
            continue
        outside = _outside_region_mask(
            hard_region,
            np.full(cand.size, i, dtype=np.int64),
            s_pos[cand, 0],
            s_pos[cand, 1],
        ) | _outside_region_mask(soft_region, cand, h_pos[i, 0], h_pos[i, 1])
        cand, outside = _hierarchy_aware_swap_filter(
            cand,
            outside,
            float(hard_local[i]),
            soft_local[cand],
            span,
            enabled=field_name == "weighted_congestion",
        )
        rank = soft_local[cand] + region_bias * span * outside.astype(np.float64)
        legal_mask = _legal_hard_soft_candidates(
            h_pos,
            s_pos,
            sep_x_mat,
            sep_y_mat,
            hw,
            hh,
            soft_hw,
            soft_hh,
            cw,
            ch,
            i,
            cand,
            1e-6,
        )
        ranked_idx = _gpu_swap_prescore_order(
            rank,
            enabled=const.HIER_GPU_SWAP_PRESCORE_HS,
            src_x=float(h_pos[i, 0]),
            src_y=float(h_pos[i, 1]),
            tgt_x=s_pos[cand, 0],
            tgt_y=s_pos[cand, 1],
            field_span=span,
            distance_span=max(float(max(cw, ch)), 1.0),
        )[:k_neighbors]
        ranked = cand[ranked_idx]
        ranked_outside = outside[ranked_idx]
        trace_rows = []
        trace_by_target = {}
        for candidate_rank, (k_raw, is_legal) in enumerate(zip(ranked, legal_mask[ranked_idx])):
            k = int(k_raw)
            hx, hy = float(s_pos[k, 0]), float(s_pos[k, 1])
            sx, sy = float(h_pos[i, 0]), float(h_pos[i, 1])
            in_bounds = _in_bounds(hx, hy, hw[i], hh[i], cw, ch) and _in_bounds(
                sx, sy, soft_hw[k], soft_hh[k], cw, ch
            )
            outside_move = bool(ranked_outside[candidate_rank])
            row = {
                "candidate_rank": int(candidate_rank),
                "target": k,
                "source_field": float(hard_local[i]),
                "target_field": float(soft_local[k]),
                "outside_region": bool(outside_move),
                "legal": bool(is_legal and in_bounds),
                "old_proxy": float(best_score),
                "candidate_proxy": None,
                "proxy_delta": None,
                "accepted": False,
                "rejection_reason": _swap_rejection_reason(
                    legal=bool(is_legal),
                    in_bounds=bool(in_bounds),
                    outside_region=bool(outside_move),
                    scored=False,
                ),
            }
            trace_by_target[k] = row
            trace_rows.append(row)
        trace_rows = _rank_swap_candidates(
            trace_rows,
            benchmark_name=benchmark_name,
            kind="hard_soft",
            field_name=field_name,
            source=i,
        )
        scored = []
        for row in trace_rows:
            # legal_mask is aligned with `cand`; `ranked_idx` preserves ranking order.
            if deadline is not None and time.monotonic() > deadline:
                break
            if not bool(row.get("legal", False)):
                continue
            k = int(row["target"])
            hx, hy = float(s_pos[k, 0]), float(s_pos[k, 1])
            sx, sy = float(h_pos[i, 0]), float(h_pos[i, 1])
            if not _in_bounds(hx, hy, hw[i], hh[i], cw, ch):
                continue
            if not _in_bounds(sx, sy, soft_hw[k], soft_hh[k], cw, ch):
                continue
            scored.append((k, hx, hy, sx, sy, bool(row.get("outside_region", False))))
        if scored:
            scores = scorer.score_swap_hard_soft_many(
                i, np.asarray([k for k, *_ in scored], dtype=np.int64)
            )
            scored_iter = zip(scored, scores)
        else:
            scored_iter = ()
        for (k, hx, hy, sx, sy, outside_move), score in scored_iter:
            stats["hs_scores"] += 1
            required_gain = max(min_gain, soft_barrier_gain)
            region_ok = accepts_region_score(
                score, best_score, outside_move, max(escape_min, required_gain)
            )
            row = trace_by_target.get(int(k))
            if row is not None:
                row["candidate_proxy"] = float(score)
                row["proxy_delta"] = float(score) - float(best_score)
                row["soft_barrier_gain"] = float(soft_barrier_gain)
                row["rejection_reason"] = _swap_rejection_reason(
                    legal=True,
                    outside_region=bool(outside_move),
                    accepted_by_region_gate=bool(region_ok),
                )
            if _accept_swap(score, best_score, outside_move, escape_min, required_gain):
                if row is not None:
                    row["accepted"] = True
                    row["rejection_reason"] = None
                scorer.commit_swap_hard_soft(i, k)
                h_pos[i, 0], h_pos[i, 1] = hx, hy
                s_pos[k, 0], s_pos[k, 1] = sx, sy
                _record_accept(stats, "hs", outside_move, best_score, score)
                best_score = float(score)
                accepts += 1
                break
        _log_swap_candidates(benchmark_name, "hard_soft", field_name, i, best_score, trace_rows)
    return accepts, best_score


def _region_bounded_swap_relief(
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
    benchmark,
    incremental_scorer,
    initial_score: float,
    hard_region: "np.ndarray | None",
    soft_region: "np.ndarray | None",
    deadline: "float | None" = None,
    rounds: int = 1,
    hard_k: int = 16,
    soft_k: int = 24,
    region_bias: float = 1.0,
    escape_min: float = 0.002,
    min_gain: float = 1e-5,
    soft_barrier_gain: float = 0.0,
    min_field_relief: float = 0.0,
    enable_hh: bool = True,
    enable_hs: bool = True,
    enable_ss: bool = True,
    use_density: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, float, dict]:
    """Run bounded hierarchy-preserving swap relief."""
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)
    best_score = float(initial_score)
    accepts = 0
    rounds = max(1, int(rounds))
    stats = _new_stats()

    for _ in range(rounds):
        if deadline is not None and time.monotonic() > deadline:
            break
        weighted_rank = not use_density
        field = (
            _density_field(incremental_scorer, nr, nc)
            if use_density
            else (
                weighted_congestion_field(incremental_scorer, nr, nc)
                if weighted_rank
                else _congestion_field(incremental_scorer, nr, nc)
            )
        )
        if field is None:
            break
        field_name = (
            "density" if use_density else ("weighted_congestion" if weighted_rank else "congestion")
        )
        if enable_hh:
            got, best_score = _try_hard_hard(
                hard_pos,
                sizes,
                hw,
                hh,
                cw,
                ch,
                movable_h,
                incremental_scorer,
                best_score,
                field,
                hard_region,
                max(1, int(hard_k)),
                region_bias,
                escape_min,
                min_gain,
                min_field_relief,
                stats,
                deadline,
                getattr(benchmark, "name", ""),
                field_name,
            )
            accepts += got
        if enable_hs:
            got, best_score = _try_hard_soft(
                hard_pos,
                soft_pos,
                sizes,
                hw,
                hh,
                soft_hw,
                soft_hh,
                cw,
                ch,
                movable_h,
                soft_movable,
                incremental_scorer,
                best_score,
                field,
                hard_region,
                soft_region,
                max(1, int(soft_k)),
                region_bias,
                escape_min,
                min_gain,
                soft_barrier_gain,
                min_field_relief,
                stats,
                deadline,
                getattr(benchmark, "name", ""),
                field_name,
            )
            accepts += got
        if enable_ss:
            got, best_score = _try_soft_soft(
                soft_pos,
                soft_hw,
                soft_hh,
                cw,
                ch,
                soft_movable,
                incremental_scorer,
                best_score,
                field,
                soft_region,
                max(1, int(soft_k)),
                region_bias,
                escape_min,
                min_gain,
                soft_barrier_gain,
                min_field_relief,
                stats,
                deadline,
                getattr(benchmark, "name", ""),
                field_name,
            )
            accepts += got

    return hard_pos, soft_pos, accepts, best_score, stats
