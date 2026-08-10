"""Bounded assembly of small hierarchy leaf clusters."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from placer.local_search.subcluster_relocation import (
    _hard_state_is_legal,
    _legalize_group_candidate,
)


def _cluster_centroids(
    hard_pos: np.ndarray,
    clusters: Mapping[int, Sequence[int]],
) -> dict[int, np.ndarray]:
    """Return hard-macro centroids for nonempty leaf clusters."""
    result: dict[int, np.ndarray] = {}
    for cid, raw_members in clusters.items():
        members = np.asarray(raw_members, dtype=np.int64).reshape(-1)
        members = members[(members >= 0) & (members < hard_pos.shape[0])]
        if members.size:
            result[int(cid)] = np.mean(hard_pos[members], axis=0)
    return result


def _neighbor_anchors(
    centroids: Mapping[int, np.ndarray],
    edges,
) -> dict[int, np.ndarray]:
    """Return weighted centroids of structurally adjacent leaf clusters."""
    weighted_sum: dict[int, np.ndarray] = {}
    weight_sum: dict[int, float] = {}
    for edge in edges or ():
        left = int(getattr(edge, "src", -1))
        right = int(getattr(edge, "dst", -1))
        if left not in centroids or right not in centroids:
            continue
        weight = max(0.0, float(getattr(edge, "weight", 0.0)))
        if weight <= 0.0:
            continue
        weighted_sum[left] = weighted_sum.get(left, np.zeros(2)) + weight * centroids[right]
        weighted_sum[right] = weighted_sum.get(right, np.zeros(2)) + weight * centroids[left]
        weight_sum[left] = weight_sum.get(left, 0.0) + weight
        weight_sum[right] = weight_sum.get(right, 0.0) + weight
    return {
        cid: weighted_sum[cid] / weight_sum[cid]
        for cid in weighted_sum
        if weight_sum.get(cid, 0.0) > 0.0
    }


def _small_cluster_consolidation(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    incremental_scorer,
    initial_score: float,
    *,
    clusters: Mapping[int, Sequence[int]],
    cluster_softs: Mapping[int, Sequence[int]],
    edges,
    movable_h: np.ndarray,
    movable_soft: np.ndarray,
    hard_region: np.ndarray,
    soft_region: np.ndarray,
    candidate_allowed: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    structural_score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    deadline: float | None = None,
    min_hard: int = 2,
    max_hard: int = 8,
    max_soft: int = 16,
    top_clusters: int = 8,
    top_slot_clusters: int = 10,
    compact_scales: Sequence[float] = (0.92, 0.84),
    neighbor_shift_fractions: Sequence[float] = (0.0, 0.25),
    soft_scales: Sequence[float] = (0.75, 0.50),
    min_structural_gain: float = 1.0e-6,
    min_proxy_gain: float = 1.0e-5,
    max_scored: int | None = 32,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Compact small leaves and pull owned soft macros into their assembly.

    Candidate generation is structural, but a state commits only when it is
    legal, remains inside its immutable leaf regions, improves the active
    hierarchy score, passes the complete hierarchy contract, and improves the
    exact proxy. This makes the pass safe to compose with later relief.
    """
    score_limit = None if max_scored is None else max(0, int(max_scored))
    stats = {
        "eligible_clusters": 0,
        "selected_clusters": 0,
        "candidates": 0,
        "legal": 0,
        "structural_rejects": 0,
        "hierarchy_rejects": 0,
        "scored": 0,
        "accepts": 0,
        "legalized_candidates": 0,
        "slot_candidates": 0,
        "slot_scored": 0,
        "slot_accepts": 0,
        "best_structural_gain": 0.0,
        "best_proxy_gain": 0.0,
        "score_limit": score_limit,
        "quota_exhausted": bool(score_limit is not None and score_limit <= 0),
    }
    _small_cluster_consolidation.last_stats = stats
    if not clusters or hard_region is None or soft_region is None:
        return hard_pos, soft_pos, 0, float(initial_score)

    movable_h = np.asarray(movable_h, dtype=bool)
    movable_soft = np.asarray(movable_soft, dtype=bool)
    centroids = _cluster_centroids(hard_pos, clusters)
    neighbor_anchor = _neighbor_anchors(centroids, edges)
    canvas_diag = max(float(np.hypot(cw, ch)), 1.0)
    rows = []
    for cid, raw_members in clusters.items():
        members = np.asarray(raw_members, dtype=np.int64).reshape(-1)
        members = members[(members >= 0) & (members < hard_pos.shape[0])]
        if (
            members.size < max(1, int(min_hard))
            or members.size > max(1, int(max_hard))
            or not bool(np.all(movable_h[members]))
        ):
            continue
        soft_indices = np.asarray(cluster_softs.get(int(cid), ()), dtype=np.int64) - int(n)
        soft_indices = soft_indices[(soft_indices >= 0) & (soft_indices < soft_pos.shape[0])]
        soft_indices = soft_indices[movable_soft[soft_indices]]
        if soft_indices.size > max(0, int(max_soft)):
            continue
        center = centroids[int(cid)]
        hard_spread = float(np.mean(np.linalg.norm(hard_pos[members] - center, axis=1)))
        soft_spread = (
            float(np.mean(np.linalg.norm(soft_pos[soft_indices] - center, axis=1)))
            if soft_indices.size
            else 0.0
        )
        edge_pull = (
            float(np.linalg.norm(neighbor_anchor[int(cid)] - center))
            if int(cid) in neighbor_anchor
            else 0.0
        )
        priority = (hard_spread + 0.75 * soft_spread + 0.25 * edge_pull) / canvas_diag
        rows.append((-priority, int(cid), members, soft_indices))
    rows.sort(key=lambda row: (row[0], row[1]))
    all_rows = rows
    stats["eligible_clusters"] = int(len(all_rows))
    rows = all_rows[: max(0, int(top_clusters))]
    stats["selected_clusters"] = int(len(rows))

    scales = sorted(
        {float(value) for value in compact_scales if 0.5 <= float(value) < 1.0},
        reverse=True,
    )
    shifts = sorted(
        {float(value) for value in neighbor_shift_fractions if 0.0 <= float(value) <= 0.5}
    )
    soft_factors = sorted(
        {float(value) for value in soft_scales if 0.0 <= float(value) <= 1.0},
        reverse=True,
    )
    if not scales or not shifts or not soft_factors:
        return hard_pos, soft_pos, 0, float(initial_score)

    current_score = float(initial_score)
    current_structural = (
        float(structural_score_fn(hard_pos, soft_pos))
        if structural_score_fn is not None
        else float("inf")
    )
    accepts = 0
    compact_score_limit = None if score_limit is None else max(1, int(np.ceil(0.5 * score_limit)))
    for _priority, cid, members, soft_indices in rows:
        if compact_score_limit is not None and int(stats["scored"]) >= compact_score_limit:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        center = np.mean(hard_pos[members], axis=0)
        pull = neighbor_anchor.get(cid, center) - center
        best = None
        best_key = None
        for shift_fraction in shifts:
            for compact_scale in scales:
                for soft_scale in soft_factors:
                    if (
                        compact_score_limit is not None
                        and int(stats["scored"]) >= compact_score_limit
                    ):
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    stats["candidates"] += 1
                    shift = float(shift_fraction) * pull
                    targets = center + shift + float(compact_scale) * (hard_pos[members] - center)
                    inside = (
                        (targets[:, 0] >= hard_region[members, 0] - 1.0e-9)
                        & (targets[:, 0] <= hard_region[members, 2] + 1.0e-9)
                        & (targets[:, 1] >= hard_region[members, 1] - 1.0e-9)
                        & (targets[:, 1] <= hard_region[members, 3] + 1.0e-9)
                    )
                    trial_hard = hard_pos.copy()
                    trial_hard[members] = targets
                    if not bool(np.all(inside)) or not _hard_state_is_legal(trial_hard, hw, hh):
                        legalized = _legalize_group_candidate(
                            hard_pos,
                            members,
                            targets,
                            hard_region,
                            hw,
                            hh,
                            cw,
                            ch,
                            deadline,
                        )
                        if legalized is None:
                            continue
                        targets = legalized
                        stats["legalized_candidates"] += 1
                    new_center = np.mean(targets, axis=0)
                    soft_targets = new_center + float(soft_scale) * (
                        soft_pos[soft_indices] - center
                    )
                    if soft_indices.size:
                        soft_inside = (
                            (soft_targets[:, 0] >= soft_region[soft_indices, 0] - 1.0e-9)
                            & (soft_targets[:, 0] <= soft_region[soft_indices, 2] + 1.0e-9)
                            & (soft_targets[:, 1] >= soft_region[soft_indices, 1] - 1.0e-9)
                            & (soft_targets[:, 1] <= soft_region[soft_indices, 3] + 1.0e-9)
                        )
                        if not bool(np.all(soft_inside)):
                            continue
                    stats["legal"] += 1
                    trial_hard = hard_pos.copy()
                    trial_soft = soft_pos.copy()
                    trial_hard[members] = targets
                    trial_soft[soft_indices] = soft_targets
                    structural = (
                        float(structural_score_fn(trial_hard, trial_soft))
                        if structural_score_fn is not None
                        else current_structural - float(min_structural_gain)
                    )
                    structural_gain = current_structural - structural
                    if structural_gain < float(min_structural_gain):
                        stats["structural_rejects"] += 1
                        continue
                    if candidate_allowed is not None and not bool(
                        candidate_allowed(trial_hard, trial_soft)
                    ):
                        stats["hierarchy_rejects"] += 1
                        continue
                    score = float(
                        incremental_scorer.score_move_group(
                            members, targets, soft_indices, soft_targets
                        )
                    )
                    stats["scored"] += 1
                    proxy_gain = current_score - score
                    stats["best_structural_gain"] = max(
                        float(stats["best_structural_gain"]), structural_gain
                    )
                    stats["best_proxy_gain"] = max(float(stats["best_proxy_gain"]), proxy_gain)
                    if proxy_gain < float(min_proxy_gain):
                        continue
                    key = (score, structural, float(compact_scale), float(soft_scale))
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (targets.copy(), soft_targets.copy(), score, structural)
        if best is None:
            continue
        targets, soft_targets, current_score, current_structural = best
        incremental_scorer.commit_move_group(members, targets, soft_indices, soft_targets)
        hard_pos[members] = targets
        soft_pos[soft_indices] = soft_targets
        accepts += 1

    # Compacting a leaf cannot fix a poor global ordering of already assembled
    # leaves. Test a small stable set of whole-leaf slot exchanges as a second
    # lane. The swap uses canvas-bounded affected-only legalization, then the
    # same structural, hierarchy-contract, and exact-proxy gates as compaction.
    slot_rows = all_rows[: max(0, int(top_slot_clusters))]
    slot_pairs = []
    for left_index, left in enumerate(slot_rows):
        for right in slot_rows[left_index + 1 :]:
            left_center = np.mean(hard_pos[left[2]], axis=0)
            right_center = np.mean(hard_pos[right[2]], axis=0)
            distance = float(np.linalg.norm(left_center - right_center)) / canvas_diag
            slot_pairs.append((-distance, int(left[1]), int(right[1]), left, right))
    slot_pairs.sort(key=lambda row: row[:3])
    canvas_region = np.column_stack([hw, hh, float(cw) - hw, float(ch) - hh])
    best_slot = None
    best_slot_key = None
    for _distance, _left_id, _right_id, left, right in slot_pairs:
        if score_limit is not None and int(stats["scored"]) >= score_limit:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        left_members = np.asarray(left[2], dtype=np.int64)
        right_members = np.asarray(right[2], dtype=np.int64)
        left_soft = np.asarray(left[3], dtype=np.int64)
        right_soft = np.asarray(right[3], dtype=np.int64)
        left_center = np.mean(hard_pos[left_members], axis=0)
        right_center = np.mean(hard_pos[right_members], axis=0)
        members = np.concatenate([left_members, right_members])
        soft_indices = np.concatenate([left_soft, right_soft])
        targets = np.vstack(
            [
                hard_pos[left_members] + (right_center - left_center),
                hard_pos[right_members] + (left_center - right_center),
            ]
        )
        stats["candidates"] += 1
        stats["slot_candidates"] += 1
        trial_hard = hard_pos.copy()
        trial_hard[members] = targets
        if not _hard_state_is_legal(trial_hard, hw, hh):
            legalized = _legalize_group_candidate(
                hard_pos,
                members,
                targets,
                canvas_region,
                hw,
                hh,
                cw,
                ch,
                deadline,
            )
            if legalized is None:
                continue
            targets = legalized
            trial_hard[members] = targets
            stats["legalized_candidates"] += 1
        left_shift = np.mean(targets[: left_members.size], axis=0) - left_center
        right_shift = np.mean(targets[left_members.size :], axis=0) - right_center
        soft_targets = np.vstack(
            [soft_pos[left_soft] + left_shift, soft_pos[right_soft] + right_shift]
        )
        if soft_indices.size:
            inside = (
                (soft_targets[:, 0] >= soft_hw[soft_indices] - 1.0e-9)
                & (soft_targets[:, 0] <= float(cw) - soft_hw[soft_indices] + 1.0e-9)
                & (soft_targets[:, 1] >= soft_hh[soft_indices] - 1.0e-9)
                & (soft_targets[:, 1] <= float(ch) - soft_hh[soft_indices] + 1.0e-9)
            )
            if not bool(np.all(inside)):
                continue
        stats["legal"] += 1
        trial_soft = soft_pos.copy()
        trial_soft[soft_indices] = soft_targets
        structural = (
            float(structural_score_fn(trial_hard, trial_soft))
            if structural_score_fn is not None
            else current_structural - float(min_structural_gain)
        )
        structural_gain = current_structural - structural
        if structural_gain < float(min_structural_gain):
            stats["structural_rejects"] += 1
            continue
        if candidate_allowed is not None and not bool(candidate_allowed(trial_hard, trial_soft)):
            stats["hierarchy_rejects"] += 1
            continue
        score = float(
            incremental_scorer.score_move_group(members, targets, soft_indices, soft_targets)
        )
        stats["scored"] += 1
        stats["slot_scored"] += 1
        proxy_gain = current_score - score
        stats["best_structural_gain"] = max(float(stats["best_structural_gain"]), structural_gain)
        stats["best_proxy_gain"] = max(float(stats["best_proxy_gain"]), proxy_gain)
        if proxy_gain < float(min_proxy_gain):
            continue
        key = (score, structural, int(left[1]), int(right[1]))
        if best_slot_key is None or key < best_slot_key:
            best_slot_key = key
            best_slot = (
                members.copy(),
                targets.copy(),
                soft_indices.copy(),
                soft_targets.copy(),
                score,
                structural,
            )
    if best_slot is not None:
        members, targets, soft_indices, soft_targets, current_score, current_structural = best_slot
        incremental_scorer.commit_move_group(members, targets, soft_indices, soft_targets)
        hard_pos[members] = targets
        soft_pos[soft_indices] = soft_targets
        accepts += 1
        stats["slot_accepts"] = 1

    stats["accepts"] = int(accepts)
    stats["quota_exhausted"] = bool(score_limit is not None and int(stats["scored"]) >= score_limit)
    _small_cluster_consolidation.last_stats = stats
    return hard_pos, soft_pos, int(accepts), float(current_score)


_small_cluster_consolidation.last_stats = {}
