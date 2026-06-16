"""Region-bounded swap relief for the hierarchy placement path."""

from __future__ import annotations

import time

import numpy as np

from placer.geometry import separation_matrices
from placer.local_search.fields import _congestion_field, _density_field
from placer.local_search.region_rules import accepts_region_score, any_outside_region


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


def _in_bounds(x: float, y: float, hw: float, hh: float, cw: float, ch: float) -> bool:
    return bool(hw <= x <= cw - hw and hh <= y <= ch - hh)


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
        outside = np.array(
            [
                any_outside_region(
                    [
                        (hard_region, i, h_pos[j, 0], h_pos[j, 1]),
                        (hard_region, int(j), h_pos[i, 0], h_pos[i, 1]),
                    ]
                )
                for j in cand
            ],
            dtype=np.float64,
        )
        rank = local[cand] + region_bias * span * outside
        for j_raw in cand[np.argsort(rank)[:k_neighbors]]:
            if deadline is not None and time.monotonic() > deadline:
                break
            j = int(j_raw)
            if not _hard_swap_legal(h_pos, sep_x_mat, sep_y_mat, hw, hh, cw, ch, i, j):
                continue
            outside_move = any_outside_region(
                [
                    (hard_region, i, h_pos[j, 0], h_pos[j, 1]),
                    (hard_region, j, h_pos[i, 0], h_pos[i, 1]),
                ]
            )
            score = scorer.score_swap_hard_hard(i, j)
            stats["hh_scores"] += 1
            if _accept_swap(score, best_score, outside_move, escape_min, min_gain):
                scorer.commit_swap_hard_hard(i, j)
                h_pos[[i, j]] = h_pos[[j, i]]
                _record_accept(stats, "hh", outside_move, best_score, score)
                best_score = float(score)
                accepts += 1
                break
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
    min_field_relief,
    stats,
    deadline,
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
        outside = np.array(
            [
                any_outside_region(
                    [
                        (soft_region, a, s_pos[b, 0], s_pos[b, 1]),
                        (soft_region, int(b), s_pos[a, 0], s_pos[a, 1]),
                    ]
                )
                for b in cand
            ],
            dtype=np.float64,
        )
        rank = local[cand] + region_bias * span * outside
        for b_raw in cand[np.argsort(rank)[:k_neighbors]]:
            if deadline is not None and time.monotonic() > deadline:
                break
            b = int(b_raw)
            ax, ay = float(s_pos[b, 0]), float(s_pos[b, 1])
            bx, by = float(s_pos[a, 0]), float(s_pos[a, 1])
            if not _in_bounds(ax, ay, soft_hw[a], soft_hh[a], cw, ch):
                continue
            if not _in_bounds(bx, by, soft_hw[b], soft_hh[b], cw, ch):
                continue
            outside_move = any_outside_region([(soft_region, a, ax, ay), (soft_region, b, bx, by)])
            score = scorer.score_swap_soft_soft(a, b)
            stats["ss_scores"] += 1
            if _accept_swap(score, best_score, outside_move, escape_min, min_gain):
                scorer.commit_swap_soft_soft(a, b)
                s_pos[[a, b]] = s_pos[[b, a]]
                _record_accept(stats, "ss", outside_move, best_score, score)
                best_score = float(score)
                accepts += 1
                break
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
    min_field_relief,
    stats,
    deadline,
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
        outside = np.array(
            [
                any_outside_region(
                    [
                        (hard_region, i, s_pos[k, 0], s_pos[k, 1]),
                        (soft_region, int(k), h_pos[i, 0], h_pos[i, 1]),
                    ]
                )
                for k in cand
            ],
            dtype=np.float64,
        )
        rank = soft_local[cand] + region_bias * span * outside
        for k_raw in cand[np.argsort(rank)[:k_neighbors]]:
            if deadline is not None and time.monotonic() > deadline:
                break
            k = int(k_raw)
            hx, hy = float(s_pos[k, 0]), float(s_pos[k, 1])
            sx, sy = float(h_pos[i, 0]), float(h_pos[i, 1])
            if not _in_bounds(hx, hy, hw[i], hh[i], cw, ch):
                continue
            if not _in_bounds(sx, sy, soft_hw[k], soft_hh[k], cw, ch):
                continue
            if not _hard_at_legal(h_pos, sep_x_mat, sep_y_mat, i, hx, hy):
                continue
            outside_move = any_outside_region([(hard_region, i, hx, hy), (soft_region, k, sx, sy)])
            score = scorer.score_swap_hard_soft(i, k)
            stats["hs_scores"] += 1
            if _accept_swap(score, best_score, outside_move, escape_min, min_gain):
                scorer.commit_swap_hard_soft(i, k)
                h_pos[i, 0], h_pos[i, 1] = hx, hy
                s_pos[k, 0], s_pos[k, 1] = sx, sy
                _record_accept(stats, "hs", outside_move, best_score, score)
                best_score = float(score)
                accepts += 1
                break
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
        field = (
            _density_field(incremental_scorer, nr, nc)
            if use_density
            else _congestion_field(incremental_scorer.plc, nr, nc)
        )
        if field is None:
            break
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
                min_field_relief,
                stats,
                deadline,
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
                min_field_relief,
                stats,
                deadline,
            )
            accepts += got

    return hard_pos, soft_pos, accepts, best_score, stats
