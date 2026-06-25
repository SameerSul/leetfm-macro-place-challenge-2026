"""Refinement logic for hierarchy coldspot tightening candidates."""

from __future__ import annotations

import os
from typing import Any, Callable

import time

import numpy as np
import torch

from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer
from placer.local_search.hierarchy_swaps import _region_bounded_swap_relief
from placer.local_search.relocation import _relocation_moves, _soft_relocation_moves

LocalRegionsFn = Callable[[np.ndarray, np.ndarray, int], tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
    np.ndarray,
    np.ndarray,
] | None]
AdditiveFn = Callable[[float | None], bool]


def refine_coldspot_candidate(
    hard_xy: np.ndarray,
    soft_xy: np.ndarray,
    trace: dict,
    *,
    benchmark,
    plc,
    n: int,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    const: Any,
    region_bias: float,
    deadline: float | None,
    local_regions_fn: LocalRegionsFn,
    additive_spare_fn: AdditiveFn,
    hier_soft_barrier_gain: float,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Refine one candidate placement proposal locally and return updated candidate score."""
    score = float(
        _exact_proxy(
            torch.tensor(np.vstack([hard_xy, soft_xy]).astype(np.float32), dtype=torch.float32),
            benchmark,
            plc,
        )
    )
    adaptive_passes = (
        os.environ.get("HIER_ADAPTIVE_PASSES", "1").strip().lower()
        not in {"0", "false", "no", "off", "disable"}
    )
    adaptive_min_gain = float(const.HIER_PLATEAU_PROXY_GAIN)
    def _adaptive_gain(before: float, after: float) -> bool:
        if not adaptive_passes:
            return True
        return float(before) - float(after) > adaptive_min_gain

    if deadline is not None and time.monotonic() >= deadline:
        return hard_xy, soft_xy, score, {}

    local = local_regions_fn(hard_xy, soft_xy, int(trace.get("cluster", -1)))
    if local is None:
        return hard_xy, soft_xy, score, {}

    (
        local_h_region,
        local_s_region,
        local_h_mask,
        local_s_mask,
        local_stats,
        local_target_pool,
        local_region_mask,
    ) = local
    if not local_h_mask.any() and not local_s_mask.any():
        return hard_xy, soft_xy, score, {}

    refined_h = hard_xy.copy()
    refined_s = soft_xy.copy()
    scorer = IncrementalScorer(
        plc,
        benchmark,
        np.vstack([refined_h, refined_s]).astype(np.float64),
    )
    start_score = score
    hard_escape = 1.0e9
    soft_escape = max(0.0, float(const.HIER_COLDSPOT_LOCAL_SOFT_ESCAPE_MIN))
    stats_total = {
        "swap_accepts": 0,
        "hard_reloc_accepts": 0,
        "soft_reloc_accepts": 0,
    }
    local_swap_min_gain = float(const.HIER_SWAP_MIN_GAIN)
    local_reloc_min_gain = float(const.HIER_RELOC_PROPOSE_MIN_GAIN)
    local_soft_min_gain = float(hier_soft_barrier_gain)
    if adaptive_passes:
        local_swap_min_gain = max(local_swap_min_gain, adaptive_min_gain)
        local_reloc_min_gain = max(local_reloc_min_gain, adaptive_min_gain)
        local_soft_min_gain = max(local_soft_min_gain, adaptive_min_gain)

    local_hard_swap_k = max(1, int(const.HIER_COLDSPOT_LOCAL_HARD_SWAP_K))
    local_soft_swap_k = max(1, int(const.HIER_COLDSPOT_LOCAL_SOFT_SWAP_K))
    if additive_spare_fn(deadline):
        extra_k = max(0, int(const.HIER_ADDITIVE_SWAP_EXTRA_K))
        local_hard_swap_k += extra_k
        local_soft_swap_k += extra_k

    hard_reloc_before = float(score)
    for use_density in (False, True):
        swap_before = float(score)
        refined_h, refined_s, got, score, _stats = _region_bounded_swap_relief(
            refined_h,
            refined_s,
            sizes[:n],
            hw,
            hh,
            soft_hw,
            soft_hh,
            cw,
            ch,
            local_h_mask,
            local_s_mask,
            benchmark,
            scorer,
            score,
            local_h_region,
            local_s_region,
            deadline=deadline,
            rounds=max(1, int(const.HIER_COLDSPOT_LOCAL_SWAP_ROUNDS)),
            hard_k=local_hard_swap_k,
            soft_k=local_soft_swap_k,
            region_bias=region_bias,
            escape_min=hard_escape,
            min_gain=float(local_swap_min_gain),
            soft_barrier_gain=hier_soft_barrier_gain,
            min_field_relief=float(const.HIER_SWAP_MIN_FIELD_RELIEF),
            enable_hh=True,
            enable_hs=True,
            enable_ss=False,
            use_density=use_density,
        )
        stats_total["swap_accepts"] += got
        if not _adaptive_gain(swap_before, score):
            return refined_h, refined_s, score, stats_total

        soft_swap_before = float(score)
        refined_h, refined_s, got, score, _stats = _region_bounded_swap_relief(
            refined_h,
            refined_s,
            sizes[:n],
            hw,
            hh,
            soft_hw,
            soft_hh,
            cw,
            ch,
            np.zeros(n, dtype=bool),
            local_s_mask,
            benchmark,
            scorer,
            score,
            local_h_region,
            local_s_region,
            deadline=deadline,
            rounds=max(1, int(const.HIER_COLDSPOT_LOCAL_SWAP_ROUNDS)),
            hard_k=local_hard_swap_k,
            soft_k=local_soft_swap_k,
            region_bias=region_bias,
            escape_min=soft_escape,
            min_gain=float(local_swap_min_gain),
            soft_barrier_gain=hier_soft_barrier_gain,
            min_field_relief=float(const.HIER_SWAP_MIN_FIELD_RELIEF),
            enable_hh=False,
            enable_hs=False,
            enable_ss=True,
            use_density=use_density,
        )
        stats_total["swap_accepts"] += got
        if not _adaptive_gain(soft_swap_before, score):
            return refined_h, refined_s, score, stats_total

        hard_reloc_before = float(score)

    refined_h, got, score = _relocation_moves(
        refined_h,
        sizes[:n],
        hw,
        hh,
        cw,
        ch,
        local_h_mask,
        n,
        plc,
        benchmark,
        scorer,
        score,
        deadline=deadline,
        top_hot=max(1, int(const.HIER_COLDSPOT_LOCAL_HARD_RELOC_TOP_K)),
        n_targets=max(1, int(const.HIER_COLDSPOT_LOCAL_RELOC_TARGETS)),
        use_density=False,
        region_bbox=local_h_region,
        region_bias=region_bias,
        region_escape_min=hard_escape,
        propose_accept_min_gain=float(local_reloc_min_gain),
        target_pool=local_target_pool,
        region_mask=local_region_mask,
    )
    stats_total["hard_reloc_accepts"] += got
    if not _adaptive_gain(hard_reloc_before, score):
        return refined_h, refined_s, score, stats_total

    if local_s_mask.any():
        soft_before = float(score)
        refined_s, got, score = _soft_relocation_moves(
            refined_s,
            soft_hw,
            soft_hh,
            cw,
            ch,
            n,
            plc,
            benchmark,
            scorer,
            score,
            deadline=deadline,
            top_hot=max(1, int(const.HIER_COLDSPOT_LOCAL_SOFT_RELOC_TOP_K)),
            n_targets=max(1, int(const.HIER_COLDSPOT_LOCAL_RELOC_TARGETS)),
            soft_movable=local_s_mask,
            use_density=False,
            region_bbox=local_s_region,
            region_bias=region_bias,
            region_escape_min=soft_escape,
            accept_min_gain=float(local_soft_min_gain),
            target_pool=local_target_pool,
            region_mask=local_region_mask,
        )
        stats_total["soft_reloc_accepts"] += got
        if not _adaptive_gain(soft_before, score):
            return refined_h, refined_s, score, stats_total

    stats_total.update(
        {
            "local_refine_initial_proxy": float(start_score),
            "local_refine_proxy": float(score),
            "local_refine_proxy_delta": float(score) - float(start_score),
            "local_refine_hard_count": int(np.count_nonzero(local_h_mask)),
            "local_refine_soft_count": int(np.count_nonzero(local_s_mask)),
            "local_refine_pad": float(local_stats.get("local_region_pad", 0.0)),
            **local_stats,
        }
    )
    return refined_h, refined_s, score, stats_total
