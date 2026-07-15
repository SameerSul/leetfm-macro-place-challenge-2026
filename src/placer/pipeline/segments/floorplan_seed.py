"""Seed-portfolio helpers for hierarchy floorplan execution."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import os
import torch
import time

from placer.local_search.hierarchy_quality import hierarchy_quality_vector


def select_seed_candidate(
    rows: list[dict[str, object]],
    *,
    hierarchy_first: bool,
    absolute_slack: float,
    relative_slack: float,
) -> dict[str, object]:
    """Select a seed using hierarchy feasibility first and proxy second."""
    if not rows:
        raise ValueError("seed portfolio is empty")
    if not hierarchy_first:
        return min(rows, key=lambda row: (float(row["score"]), str(row["name"])))
    best_quality = min(float(row["hierarchy_composite"]) for row in rows)
    slack = max(float(absolute_slack), abs(best_quality) * float(relative_slack))
    eligible = [row for row in rows if float(row["hierarchy_composite"]) <= best_quality + slack]
    return min(eligible, key=lambda row: (float(row["score"]), str(row["name"])))


def run_seed_portfolio(
    *,
    benchmark,
    plc,
    benchmark_dir,
    n: int,
    n_soft: int,
    clusters,
    order,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    movable: np.ndarray,
    groups: dict | list | tuple | None,
    csofts,
    bridge_softs,
    hierarchy_edges,
    cw: float,
    ch: float,
    const: Any,
    logger: Callable[[str], None],
    run_dreamplace: Callable[..., tuple[np.ndarray, np.ndarray]],
    will_legalize: Callable[..., np.ndarray],
    exact_proxy_fn: Callable[[torch.Tensor, Any, Any], float],
    soft_relocation_fn: Callable[..., tuple[np.ndarray, float]],
    incremental_scorer_cls: type,
    group_weight: int,
    random_seed: int = 1000,
    scratch_root: str = "/tmp/dreamplace_v1_hier",
) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, object]]]:
    """Create and score the seed portfolio used by hierarchy floorplanning.

    The seed portfolio starts from the DREAMPlace seed and several small
    perturbations that preserve hierarchy intent while improving overlap
    survivability before downstream region cleanup.
    """

    def _prepare_dreamplace_candidate(
        *,
        group_weight: int,
        random_seed: int,
        scratch_root: str,
    ):
        raw_hard, raw_soft = run_dreamplace(
            str(benchmark_dir),
            plc=plc,
            scratch_root=scratch_root,
            iterations=300,
            num_threads=2,
            random_seed=random_seed,
            soft_macros_movable=True,
            cluster_groups=(groups or None),
            group_weight=group_weight,
            return_full=True,
        )
        legal_hard = will_legalize(
            raw_hard.copy(),
            movable[:n],
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=time.monotonic() + 120,
            order=order,
        )
        legal_hard = will_legalize(
            legal_hard,
            movable[:n],
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=time.monotonic() + 120,
            order=None,
        )

        cand_pos = np.vstack([legal_hard, raw_soft]).astype(np.float64)
        cand_soft = raw_soft.copy()
        cand_score = float(
            exact_proxy_fn(torch.tensor(cand_pos, dtype=torch.float32), benchmark, plc)
        )
        cand_scorer = incremental_scorer_cls(plc, benchmark, cand_pos.copy())
        soft_mov_local = movable[n : n + n_soft]
        for use_density in (False, True):
            cand_soft, _, cand_score = soft_relocation_fn(
                cand_soft,
                soft_hw,
                soft_hh,
                cw,
                ch,
                n,
                plc,
                benchmark,
                cand_scorer,
                cand_score,
                deadline=time.monotonic() + 30,
                top_hot=1024,
                n_targets=6,
                soft_movable=soft_mov_local,
                use_density=use_density,
            )
        return legal_hard, cand_soft, cand_score

    def _clip_seed(hard_xy: np.ndarray, soft_xy: np.ndarray):
        hard_xy = hard_xy.copy()
        soft_xy = soft_xy.copy()
        hard_mov = movable[:n]
        hard_xy[hard_mov, 0] = np.clip(hard_xy[hard_mov, 0], hw[hard_mov], cw - hw[hard_mov])
        hard_xy[hard_mov, 1] = np.clip(hard_xy[hard_mov, 1], hh[hard_mov], ch - hh[hard_mov])
        if n_soft:
            soft_mov_local = movable[n : n + n_soft]
            soft_xy[soft_mov_local, 0] = np.clip(
                soft_xy[soft_mov_local, 0],
                soft_hw[soft_mov_local],
                cw - soft_hw[soft_mov_local],
            )
            soft_xy[soft_mov_local, 1] = np.clip(
                soft_xy[soft_mov_local, 1],
                soft_hh[soft_mov_local],
                ch - soft_hh[soft_mov_local],
            )
        return hard_xy, soft_xy

    def _legalize_seed(name: str, hard_xy, soft_xy, *, budget_s: float = 60.0):
        hard_xy, soft_xy = _clip_seed(hard_xy, soft_xy)
        seed_deadline = time.monotonic() + float(budget_s)
        legal_hard = will_legalize(
            hard_xy,
            movable[:n],
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=seed_deadline,
            order=order,
        )
        legal_hard = will_legalize(
            legal_hard,
            movable[:n],
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            n,
            deadline=seed_deadline,
            order=None,
        )
        full = np.vstack([legal_hard, soft_xy]).astype(np.float64)
        score = float(exact_proxy_fn(torch.tensor(full, dtype=torch.float32), benchmark, plc))
        return {
            "name": name,
            "hard": legal_hard,
            "soft": soft_xy,
            "score": score,
        }

    def _expanded_seed(base_hard, base_soft):
        hard = base_hard.copy()
        soft = base_soft.copy()
        frac = float(const.HIER_SEED_EXPANSION_FRAC)
        hard_mov = movable[:n]
        if np.any(hard_mov):
            center = np.mean(hard[hard_mov], axis=0)
            hard[hard_mov] = center + (1.0 + frac) * (hard[hard_mov] - center)
        if n_soft and np.any(movable[n : n + n_soft]):
            soft_mov_local = movable[n : n + n_soft]
            center = np.mean(soft[soft_mov_local], axis=0)
            soft[soft_mov_local] = center + (1.0 + frac) * (soft[soft_mov_local] - center)
        return hard, soft

    def _synthetic_clearance_seed(base_hard, base_soft):
        hard = base_hard.copy()
        hard_mov = movable[:n]
        area = sizes[:n, 0] * sizes[:n, 1]
        area_limit = float(np.percentile(area, float(const.HIER_SEED_CLEARANCE_AREA_PCT)))
        eligible = hard_mov & (area <= area_limit)
        if not np.any(eligible):
            return hard, base_soft.copy()
        temp_hw = hw.copy()
        temp_hh = hh.copy()
        temp_hw[eligible] *= 1.0 + float(const.HIER_SEED_CLEARANCE_FRAC)
        temp_hh[eligible] *= 1.0 + float(const.HIER_SEED_CLEARANCE_FRAC)
        iters = max(1, int(const.HIER_SEED_CLEARANCE_ITERS))
        for _ in range(iters):
            delta = np.zeros_like(hard)
            for i in range(n):
                for j in range(i + 1, n):
                    move_i = bool(eligible[i])
                    move_j = bool(eligible[j])
                    if not (move_i or move_j):
                        continue
                    dx = float(hard[i, 0] - hard[j, 0])
                    dy = float(hard[i, 1] - hard[j, 1])
                    ox = float(temp_hw[i] + temp_hw[j] - abs(dx))
                    oy = float(temp_hh[i] + temp_hh[j] - abs(dy))
                    if ox <= 0.0 or oy <= 0.0:
                        continue
                    if ox <= oy:
                        sx = 1.0 if dx >= 0.0 else -1.0
                        push = np.array([0.5 * ox * sx, 0.0], dtype=np.float64)
                    else:
                        sy = 1.0 if dy >= 0.0 else -1.0
                        push = np.array([0.0, 0.5 * oy * sy], dtype=np.float64)
                    if move_i and move_j:
                        delta[i] += push
                        delta[j] -= push
                    elif move_i:
                        delta[i] += 2.0 * push
                    elif move_j:
                        delta[j] -= 2.0 * push
            hard[eligible] += 0.5 * delta[eligible]
            hard, _soft = _clip_seed(hard, base_soft)
        return hard, base_soft.copy()

    def _has_explicit_path_tags() -> bool:
        try:
            hard_b = list(plc.hard_macro_indices[:n])
            tagged = sum(1 for idx in hard_b if "/" in str(plc.modules_w_pins[int(idx)].get_name()))
        except Exception:
            return False
        min_group = max(2, int(const.HIER_TAG_PREFIX_MIN_GROUP))
        return tagged >= max(min_group, int(0.5 * n))

    def _route_channel_seed(base_hard, base_soft):
        hard = base_hard.copy()
        soft = base_soft.copy()
        hard_mov = movable[:n].astype(bool)
        soft_mov = movable[n : n + n_soft].astype(bool)
        min_cluster = max(2, int(const.HIER_SEED_ROUTE_CHANNEL_MIN_CLUSTER))
        lane_frac = max(0.0, float(const.HIER_SEED_ROUTE_CHANNEL_LANE_FRAC))
        push_frac = max(0.0, float(const.HIER_SEED_ROUTE_CHANNEL_PUSH_FRAC))
        max_shift_frac = max(0.0, float(const.HIER_SEED_ROUTE_CHANNEL_MAX_SHIFT_FRAC))
        if lane_frac <= 0.0 or push_frac <= 0.0 or max_shift_frac <= 0.0:
            return hard, soft

        def _channel_delta(xy, local_hw, local_hh, center, span_x, span_y, index_bias):
            dx = xy[:, 0] - center[0]
            dy = xy[:, 1] - center[1]
            sx = np.where(dx >= 0.0, 1.0, -1.0)
            sy = np.where(dy >= 0.0, 1.0, -1.0)
            sx = np.where(np.abs(dx) > 1.0e-9, sx, np.where(index_bias % 2 == 0, 1.0, -1.0))
            sy = np.where(np.abs(dy) > 1.0e-9, sy, np.where(index_bias % 3 == 0, 1.0, -1.0))
            lane_x = max(float(np.median(local_hw)) * 0.75, float(span_x) * lane_frac)
            lane_y = max(float(np.median(local_hh)) * 0.75, float(span_y) * lane_frac)
            max_x = float(span_x) * max_shift_frac
            max_y = float(span_y) * max_shift_frac
            push_x = np.maximum(0.0, lane_x - np.abs(dx)) * push_frac
            push_y = np.maximum(0.0, lane_y - np.abs(dy)) * push_frac
            out = np.zeros_like(xy)
            out[:, 0] = sx * np.minimum(push_x, max_x)
            out[:, 1] = sy * np.minimum(push_y, max_y)
            return out

        for cid, mem in clusters.items():
            mem = np.asarray(mem, dtype=np.int64)
            if mem.size < min_cluster:
                continue
            active = mem[hard_mov[mem]]
            if active.size == 0:
                continue
            left = float(np.min(hard[mem, 0] - hw[mem]))
            right = float(np.max(hard[mem, 0] + hw[mem]))
            bottom = float(np.min(hard[mem, 1] - hh[mem]))
            top = float(np.max(hard[mem, 1] + hh[mem]))
            span_x = max(1.0, right - left)
            span_y = max(1.0, top - bottom)
            center = np.array([(left + right) * 0.5, (bottom + top) * 0.5], dtype=np.float64)
            hard[active] += _channel_delta(
                hard[active],
                hw[active],
                hh[active],
                center,
                span_x,
                span_y,
                active,
            )
            if n_soft:
                owned = np.asarray(csofts.get(int(cid), []), dtype=np.int64) - n
                owned = owned[(owned >= 0) & (owned < n_soft)]
                owned = owned[soft_mov[owned]]
                if owned.size:
                    soft[owned] += _channel_delta(
                        soft[owned],
                        soft_hw[owned],
                        soft_hh[owned],
                        center,
                        span_x,
                        span_y,
                        owned + mem.size,
                    )
        return _clip_seed(hard, soft)

    def _select_seed_portfolio(dp_hard, dp_soft, dp_score):
        initial = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
        init_hard = initial[:n].copy()
        init_soft = initial[n : n + n_soft].copy()
        rows = [
            {
                "name": "dreamplace",
                "hard": dp_hard,
                "soft": dp_soft,
                "score": float(dp_score),
            }
        ]
        raw_candidates = []
        raw_candidates.append(("initial", init_hard, init_soft))
        for alpha in tuple(float(a) for a in const.HIER_SEED_BLEND_ALPHAS):
            hard = (1.0 - alpha) * dp_hard + alpha * init_hard
            soft = (1.0 - alpha) * dp_soft + alpha * init_soft
            raw_candidates.append((f"blend_{alpha:.2f}", hard, soft))
        raw_candidates.append(("expand", *_expanded_seed(dp_hard, dp_soft)))
        raw_candidates.append(("synthetic_clearance", *_synthetic_clearance_seed(dp_hard, dp_soft)))
        if _has_explicit_path_tags():
            raw_candidates.append(("route_channel", *_route_channel_seed(dp_hard, dp_soft)))
        for name, cand_h, cand_s in raw_candidates:
            try:
                rows.append(_legalize_seed(name, cand_h, cand_s, budget_s=45.0))
            except Exception as exc:
                logger(f"  [hier] seed {name} failed prescore: {type(exc).__name__}: {exc}")
        for row in rows:
            vector = hierarchy_quality_vector(
                np.asarray(row["hard"], dtype=np.float64),
                np.asarray(row["soft"], dtype=np.float64),
                clusters,
                csofts,
                bridge_softs,
                hierarchy_edges,
                cw,
                ch,
            )
            row["hierarchy_vector"] = vector
            row["hierarchy_composite"] = float(vector["composite"])
        hierarchy_first = os.environ.get(
            "HIER_SEED_HIERARCHY_SELECT",
            "1" if bool(const.HIER_SEED_HIERARCHY_SELECT) else "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        selected = select_seed_candidate(
            rows,
            hierarchy_first=hierarchy_first,
            absolute_slack=float(const.HIER_SEED_HIERARCHY_ABS_SLACK),
            relative_slack=float(const.HIER_SEED_HIERARCHY_REL_SLACK),
        )
        rows.sort(
            key=lambda row: (
                row is not selected,
                float(row["score"]),
                str(row["name"]),
            )
        )
        for row in rows:
            row["selected"] = row is selected
        summary = ", ".join(
            f"{r['name']}={float(r['score']):.4f}/hq={float(r['hierarchy_composite']):.5f}"
            for r in rows
        )
        logger(
            f"  [hier] seed portfolio prescore: {summary}; selected={selected['name']}; "
            f"hierarchy_first={int(hierarchy_first)}"
        )
        return selected["hard"], selected["soft"], float(selected["score"]), rows

    hard, soft, s_score = _prepare_dreamplace_candidate(
        group_weight=group_weight,
        random_seed=random_seed,
        scratch_root=scratch_root,
    )
    return _select_seed_portfolio(hard, soft, s_score)
