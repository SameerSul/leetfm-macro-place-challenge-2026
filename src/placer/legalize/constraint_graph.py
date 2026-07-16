"""Deterministic overlap projection with horizontal and vertical constraint DAGs."""

from __future__ import annotations

import time

import numpy as np


def _axis_order(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stable coordinate order and its inverse rank."""
    indices = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((indices, values)).astype(np.int64, copy=False)
    rank = np.empty(values.size, dtype=np.int64)
    rank[order] = np.arange(values.size, dtype=np.int64)
    return order, rank


def _solve_axis(
    target: np.ndarray,
    half_size: np.ndarray,
    canvas: float,
    movable: np.ndarray,
    order: np.ndarray,
    edges: dict[tuple[int, int], float],
) -> np.ndarray | None:
    """Project targets onto one acyclic set of separation constraints."""
    n = target.size
    lower = half_size.astype(np.float64, copy=True)
    upper = float(canvas) - half_size
    fixed = np.logical_not(movable)
    lower[fixed] = target[fixed]
    upper[fixed] = target[fixed]

    outgoing: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    incoming: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (u, v), weight in edges.items():
        outgoing[int(u)].append((int(v), float(weight)))
        incoming[int(v)].append((int(u), float(weight)))

    earliest = lower.copy()
    tolerance = 1e-8
    for u_raw in order:
        u = int(u_raw)
        if earliest[u] > upper[u] + tolerance:
            return None
        for v, weight in outgoing[u]:
            value = earliest[u] + weight
            if value > earliest[v]:
                earliest[v] = value

    latest = upper.copy()
    for u_raw in order[::-1]:
        u = int(u_raw)
        for v, weight in outgoing[u]:
            value = latest[v] - weight
            if value < latest[u]:
                latest[u] = value
        if latest[u] < earliest[u] - tolerance:
            return None

    result = np.empty(n, dtype=np.float64)
    for v_raw in order:
        v = int(v_raw)
        required = earliest[v]
        for u, weight in incoming[v]:
            required = max(required, result[u] + weight)
        if required > latest[v] + tolerance:
            return None
        if fixed[v]:
            value = target[v]
            if value < required - tolerance or value > latest[v] + tolerance:
                return None
            result[v] = value
        else:
            result[v] = min(max(float(target[v]), required), latest[v])
    return result


def _overlap_pairs(
    pos: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    eps: float,
) -> list[tuple[int, int]]:
    """Return strict hard-macro overlaps in stable pair order."""
    pairs: list[tuple[int, int]] = []
    n = pos.shape[0]
    for i in range(n):
        dx = np.abs(pos[i + 1 :, 0] - pos[i, 0])
        dy = np.abs(pos[i + 1 :, 1] - pos[i, 1])
        bad = (dx < hw[i] + hw[i + 1 :] + eps) & (dy < hh[i] + hh[i + 1 :] + eps)
        for offset in np.flatnonzero(bad):
            pairs.append((i, i + 1 + int(offset)))
    return pairs


def _will_legalize_constraint_graph(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    deadline: float | None = None,
    max_rounds: int = 6,
    eps: float = 0.05,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Resolve overlaps by projecting onto deterministic separation DAGs.

    Each conflicting pair is assigned to the axis requiring the smaller
    fraction of its available boundary slack. Edge orientation follows the
    original coordinate order, so both graphs stay acyclic. The closest
    feasible coordinate is then selected inside longest-path earliest/latest
    bounds. A caller should retain its ordinary legalizer as a final safety
    pass because a dense newly generated graph can be infeasible.
    """
    t0 = time.monotonic()
    n = min(int(n), int(pos.shape[0]))
    target = np.asarray(pos[:n], dtype=np.float64)
    movable = np.asarray(movable[:n], dtype=bool)
    hw = np.asarray(hw[:n], dtype=np.float64)
    hh = np.asarray(hh[:n], dtype=np.float64)
    legal = target.copy()
    legal[movable, 0] = np.clip(legal[movable, 0], hw[movable], float(cw) - hw[movable])
    legal[movable, 1] = np.clip(legal[movable, 1], hh[movable], float(ch) - hh[movable])

    order_x, rank_x = _axis_order(target[:, 0])
    order_y, rank_y = _axis_order(target[:, 1])
    edges_x: dict[tuple[int, int], float] = {}
    edges_y: dict[tuple[int, int], float] = {}
    initial_pairs = _overlap_pairs(legal, hw, hh, float(eps))
    unresolved = initial_pairs
    infeasible = False
    rounds_done = 0

    for round_index in range(max(1, int(max_rounds))):
        if deadline is not None and time.monotonic() > deadline:
            break
        unresolved = _overlap_pairs(legal, hw, hh, float(eps))
        if not unresolved:
            break
        added = 0
        for i, j in unresolved:
            # A tiny guard keeps the realized float64 distance on the legal
            # side of the strict `< separation + eps` audit comparison.
            sep_x = float(hw[i] + hw[j] + eps + 1e-6)
            sep_y = float(hh[i] + hh[j] + eps + 1e-6)
            if rank_x[i] < rank_x[j]:
                ux, vx = i, j
            else:
                ux, vx = j, i
            if rank_y[i] < rank_y[j]:
                uy, vy = i, j
            else:
                uy, vy = j, i

            need_x = max(0.0, sep_x - abs(float(legal[i, 0] - legal[j, 0])))
            need_y = max(0.0, sep_y - abs(float(legal[i, 1] - legal[j, 1])))
            slack_x = max(
                1e-9,
                float(legal[ux, 0] - hw[ux]) + float(cw - hw[vx] - legal[vx, 0]),
            )
            slack_y = max(
                1e-9,
                float(legal[uy, 1] - hh[uy]) + float(ch - hh[vy] - legal[vy, 1]),
            )
            prefer_x = need_x / slack_x <= need_y / slack_y
            if prefer_x:
                key = (int(ux), int(vx))
                old = edges_x.get(key, 0.0)
                edges_x[key] = max(old, sep_x)
                added += int(edges_x[key] > old)
            else:
                key = (int(uy), int(vy))
                old = edges_y.get(key, 0.0)
                edges_y[key] = max(old, sep_y)
                added += int(edges_y[key] > old)

        if added == 0:
            break
        solved_x = _solve_axis(target[:, 0], hw, cw, movable, order_x, edges_x)
        solved_y = _solve_axis(target[:, 1], hh, ch, movable, order_y, edges_y)
        if solved_x is None or solved_y is None:
            infeasible = True
            break
        legal[:, 0] = solved_x
        legal[:, 1] = solved_y
        rounds_done = round_index + 1

    final_pairs = _overlap_pairs(legal, hw, hh, float(eps))
    displacement = np.linalg.norm(legal[movable] - target[movable], axis=1)
    stats: dict[str, float | int | bool] = {
        "initial_overlaps": int(len(initial_pairs)),
        "final_overlaps": int(len(final_pairs)),
        "x_constraints": int(len(edges_x)),
        "y_constraints": int(len(edges_y)),
        "rounds": int(rounds_done),
        "infeasible": bool(infeasible),
        "deadline_hit": bool(deadline is not None and time.monotonic() > deadline),
        "max_displacement": float(displacement.max()) if displacement.size else 0.0,
        "elapsed_s": float(time.monotonic() - t0),
    }
    _will_legalize_constraint_graph.last_stats = stats
    return legal.astype(pos.dtype, copy=False), stats


_will_legalize_constraint_graph.last_stats = {}
