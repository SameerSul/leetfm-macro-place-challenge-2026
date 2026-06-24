"""Bounded survivor-pool search for hierarchy-preserving placement."""

from __future__ import annotations

import time

import numpy as np
import torch

from utils import constants as const
from utils.config import _GPU_BACKEND, _GPU_DEVICE
from placer.legalize.spiral import _will_legalize
from placer.local_search.cluster_decompress import hierarchy_quality_metric
from placer.local_search.fields import _congestion_field, coldest_window_anchor
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer


def _auto_cuda(value) -> bool:
    if isinstance(value, str) and value.lower() == "auto":
        return _GPU_BACKEND == "cuda"
    return bool(value)


def _hard_valid(hard_xy, hw, hh, cw: float, ch: float) -> bool:
    if hard_xy.shape[0] == 0:
        return True
    if np.any(hard_xy[:, 0] < hw - 1e-6) or np.any(hard_xy[:, 0] > cw - hw + 1e-6):
        return False
    if np.any(hard_xy[:, 1] < hh - 1e-6) or np.any(hard_xy[:, 1] > ch - hh + 1e-6):
        return False
    dx = np.abs(hard_xy[:, None, 0] - hard_xy[None, :, 0])
    dy = np.abs(hard_xy[:, None, 1] - hard_xy[None, :, 1])
    ok = (dx + 1e-6 >= (hw[:, None] + hw[None, :])) | (dy + 1e-6 >= (hh[:, None] + hh[None, :]))
    np.fill_diagonal(ok, True)
    return bool(ok.all())


def _inside_regions(pos: np.ndarray, regions, movable) -> bool:
    if regions is None:
        return True
    movable = np.asarray(movable, dtype=np.bool_)
    idx = np.flatnonzero(movable)
    if idx.size == 0:
        return True
    r = regions[idx]
    p = pos[idx]
    inside = (
        (p[:, 0] >= r[:, 0]) & (p[:, 0] <= r[:, 2]) & (p[:, 1] >= r[:, 1]) & (p[:, 1] <= r[:, 3])
    )
    return bool(np.all(inside))


def _clip_hard(pos, regions, hw, hh, cw: float, ch: float, movable):
    out = pos.copy()
    for i in np.flatnonzero(np.asarray(movable, dtype=np.bool_)):
        if regions is None:
            x0, y0, x1, y1 = hw[i], hh[i], cw - hw[i], ch - hh[i]
        else:
            x0, y0, x1, y1 = regions[i]
        out[i, 0] = np.clip(out[i, 0], x0, x1)
        out[i, 1] = np.clip(out[i, 1], y0, y1)
    return out


def _clip_soft(pos, regions, soft_hw, soft_hh, cw: float, ch: float, movable):
    out = pos.copy()
    for k in np.flatnonzero(np.asarray(movable, dtype=np.bool_)):
        if regions is None:
            x0, y0, x1, y1 = soft_hw[k], soft_hh[k], cw - soft_hw[k], ch - soft_hh[k]
        else:
            x0, y0, x1, y1 = regions[k]
        out[k, 0] = np.clip(out[k, 0], x0, x1)
        out[k, 1] = np.clip(out[k, 1], y0, y1)
    return out


def _field_values(pos: np.ndarray, field: np.ndarray, cw: float, ch: float) -> np.ndarray:
    nr, nc = field.shape
    cell_w, cell_h = cw / max(nc, 1), ch / max(nr, 1)
    ci = np.clip((pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    return field[ri, ci]


def _rank_candidates(candidates: list[dict]) -> list[dict]:
    if len(candidates) <= 1:
        return candidates
    values = np.asarray([float(c["rank"]) for c in candidates], dtype=np.float64)
    if _auto_cuda(const.HIER_SURVIVOR_GPU_RANK) and values.size >= 16:
        try:
            order = torch.argsort(torch.tensor(values, device=_GPU_DEVICE), stable=True)
            order = order.cpu().numpy().astype(np.int64, copy=False)
        except Exception:
            order = np.argsort(values, kind="stable")
    else:
        order = np.argsort(values, kind="stable")
    return [candidates[int(i)] for i in order]


def _cluster_order(hard_xy, clusters, field, cw: float, ch: float, top_k: int):
    rows = []
    vals = _field_values(hard_xy, field, cw, ch)
    for cid, mem in clusters.items():
        idx = np.asarray(mem, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < hard_xy.shape[0])]
        if idx.size:
            rows.append((float(vals[idx].mean()), int(cid), idx))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return rows[: max(1, int(top_k))]


def _cluster_order_for_legalize(clusters: dict[int, np.ndarray], n: int) -> list[int]:
    order = []
    seen = set()
    sorted_clusters = sorted(clusters.values(), key=lambda m: -len(m))
    for mem in sorted_clusters:
        for idx in np.asarray(mem, dtype=np.int64):
            i = int(idx)
            if 0 <= i < n and i not in seen:
                seen.add(i)
                order.append(i)
    order.extend(i for i in range(n) if i not in seen)
    return order


def _move_cluster_candidate(
    hard_xy,
    soft_xy,
    cid: int,
    members: np.ndarray,
    delta: np.ndarray,
    cluster_softs,
    bridge_softs,
    n: int,
    hard_region,
    soft_region,
    hw,
    hh,
    soft_hw,
    soft_hh,
    cw: float,
    ch: float,
    movable_h,
    movable_s,
):
    cand_h = hard_xy.copy()
    cand_s = soft_xy.copy()
    members = members[np.asarray(movable_h, dtype=np.bool_)[members]]
    if members.size:
        cand_h[members] += delta
    if cluster_softs:
        owned = np.asarray(cluster_softs.get(int(cid), []), dtype=np.int64) - int(n)
        owned = owned[(owned >= 0) & (owned < cand_s.shape[0])]
        if owned.size:
            owned = owned[np.asarray(movable_s, dtype=np.bool_)[owned]]
            cand_s[owned] += delta
    if bridge_softs:
        for sk, cids in bridge_softs.items():
            if int(cid) in {int(c) for c in cids}:
                s = int(sk)
                if 0 <= s < cand_s.shape[0] and bool(movable_s[s]):
                    cand_s[s] += 0.5 * delta
    cand_h = _clip_hard(cand_h, hard_region, hw, hh, cw, ch, movable_h)
    cand_s = _clip_soft(cand_s, soft_region, soft_hw, soft_hh, cw, ch, movable_s)
    return cand_h, cand_s


def _score_full(hard_xy, soft_xy, benchmark, plc) -> float:
    full = np.vstack([hard_xy, soft_xy]).astype(np.float32)
    return float(_exact_proxy(torch.tensor(full, dtype=torch.float32), benchmark, plc))


def _parallel_survivor_search(
    hard_xy: np.ndarray,
    soft_xy: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    movable_h,
    movable_s,
    n: int,
    plc,
    benchmark,
    initial_score: float,
    clusters: dict[int, np.ndarray],
    cluster_softs=None,
    bridge_softs=None,
    hard_region=None,
    soft_region=None,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Run a small go-with-the-winners pool over hierarchy-safe candidates."""
    _parallel_survivor_search.last_stats = {
        "candidates": 0,
        "legal": 0,
        "scored": 0,
        "accepts": 0,
        "gpu_rank": bool(_auto_cuda(const.HIER_SURVIVOR_GPU_RANK)),
    }
    if not clusters:
        return hard_xy, soft_xy, 0, float(initial_score)

    movable_h = np.asarray(movable_h, dtype=np.bool_)
    movable_s = np.asarray(movable_s, dtype=np.bool_)
    baseline_quality = hierarchy_quality_metric(hard_xy, clusters)
    quality_budget = max(0.0, float(const.HIER_SURVIVOR_QUALITY_BUDGET))
    min_gain = max(0.0, float(const.HIER_SURVIVOR_MIN_GAIN))
    width = max(1, int(const.HIER_SURVIVOR_WIDTH))
    rounds = max(1, int(const.HIER_SURVIVOR_ROUNDS))
    exact_top_k = max(1, int(const.HIER_SURVIVOR_EXACT_TOP_K))
    cell = min(cw / max(int(benchmark.grid_cols), 1), ch / max(int(benchmark.grid_rows), 1))
    legal_order = _cluster_order_for_legalize(clusters, n)

    survivors = [{"hard": hard_xy.copy(), "soft": soft_xy.copy(), "score": float(initial_score)}]
    best_h = hard_xy.copy()
    best_s = soft_xy.copy()
    best_score = float(initial_score)
    accepts = 0

    for _round in range(rounds):
        if deadline is not None and time.monotonic() >= deadline:
            break
        pool = []
        for state_id, state in enumerate(survivors):
            if deadline is not None and time.monotonic() >= deadline:
                break
            full = np.vstack([state["hard"], state["soft"]]).astype(np.float64)
            scorer = IncrementalScorer(plc, benchmark, full.copy())
            field = _congestion_field(scorer, int(benchmark.grid_rows), int(benchmark.grid_cols))
            if field is None:
                continue
            cold_x, cold_y = coldest_window_anchor(field, *field.shape, cw, ch, win_cells=5)
            for _heat, cid, members in _cluster_order(
                state["hard"],
                clusters,
                field,
                cw,
                ch,
                int(const.HIER_SURVIVOR_HOT_CLUSTERS),
            ):
                centroid = state["hard"][members].mean(axis=0)
                to_cold = np.array([cold_x - centroid[0], cold_y - centroid[1]], dtype=np.float64)
                norm = float(np.linalg.norm(to_cold))
                dirs = []
                if norm > 1e-9:
                    dirs.append(to_cold / norm)
                dirs.extend(
                    [
                        np.array([1.0, 0.0]),
                        np.array([-1.0, 0.0]),
                        np.array([0.0, 1.0]),
                        np.array([0.0, -1.0]),
                    ]
                )
                for step_cells in tuple(float(v) for v in const.HIER_SURVIVOR_STEP_CELLS):
                    for direction in dirs:
                        delta = np.asarray(direction, dtype=np.float64) * (step_cells * cell)
                        cand_h, cand_s = _move_cluster_candidate(
                            state["hard"],
                            state["soft"],
                            cid,
                            members,
                            delta,
                            cluster_softs,
                            bridge_softs,
                            n,
                            hard_region,
                            soft_region,
                            hw,
                            hh,
                            soft_hw,
                            soft_hh,
                            cw,
                            ch,
                            movable_h,
                            movable_s,
                        )
                        rank = float(_field_values(cand_h[members], field, cw, ch).mean())
                        pool.append(
                            {
                                "hard": cand_h,
                                "soft": cand_s,
                                "rank": rank + 1e-6 * abs(float(state_id)),
                                "state_id": int(state_id),
                            }
                        )
        _parallel_survivor_search.last_stats["candidates"] += int(len(pool))
        if not pool:
            break
        ranked = _rank_candidates(pool)[: max(exact_top_k * width, exact_top_k)]
        next_states = list(survivors)
        for cand in ranked:
            if deadline is not None and time.monotonic() >= deadline:
                break
            cand_h = _will_legalize(
                cand["hard"],
                movable_h,
                sizes,
                hw,
                hh,
                cw,
                ch,
                n,
                deadline=deadline,
                order=legal_order,
            )
            if not _hard_valid(cand_h, hw, hh, cw, ch):
                continue
            if not _inside_regions(cand_h, hard_region, movable_h):
                continue
            quality = hierarchy_quality_metric(cand_h, clusters)
            if quality > baseline_quality + quality_budget:
                continue
            _parallel_survivor_search.last_stats["legal"] += 1
            score = _score_full(cand_h, cand["soft"], benchmark, plc)
            _parallel_survivor_search.last_stats["scored"] += 1
            parent_score = float(survivors[int(cand["state_id"])]["score"])
            if score < parent_score - min_gain:
                next_states.append({"hard": cand_h, "soft": cand["soft"].copy(), "score": score})
            if score < best_score - min_gain:
                best_h = cand_h.copy()
                best_s = cand["soft"].copy()
                best_score = float(score)
                accepts = 1
        next_states.sort(key=lambda s: float(s["score"]))
        survivors = next_states[:width]

    _parallel_survivor_search.last_stats["accepts"] = int(accepts)
    return best_h, best_s, accepts, best_score
