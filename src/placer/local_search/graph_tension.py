"""Hierarchy graph pressure signals for candidate ordering."""

from __future__ import annotations

import numpy as np


def _cluster_centroids(hard_xy: np.ndarray, clusters: dict) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for cid, raw_members in clusters.items():
        members = np.asarray(raw_members, dtype=np.int64)
        members = members[(members >= 0) & (members < hard_xy.shape[0])]
        if members.size:
            out[int(cid)] = np.asarray(hard_xy[members].mean(axis=0), dtype=np.float64)
    return out


def _corridor_field_mean(
    field: np.ndarray | None,
    a: np.ndarray,
    b: np.ndarray,
    cw: float,
    ch: float,
    samples: int,
) -> float:
    if field is None or field.size == 0:
        return 1.0
    nr, nc = field.shape
    cell_w = float(cw) / max(1, nc)
    cell_h = float(ch) / max(1, nr)
    steps = max(2, int(samples))
    xs = np.linspace(float(a[0]), float(b[0]), steps)
    ys = np.linspace(float(a[1]), float(b[1]), steps)
    cols = np.clip((xs / cell_w).astype(np.int64), 0, nc - 1)
    rows = np.clip((ys / cell_h).astype(np.int64), 0, nr - 1)
    avg = float(np.mean(field[rows, cols]))
    base = max(float(np.mean(field)), 1e-12)
    return max(0.25, avg / base)


def cluster_graph_tension(
    hard_xy: np.ndarray,
    clusters: dict,
    edges,
    *,
    cw: float,
    ch: float,
    field: np.ndarray | None = None,
    seed_hard_xy: np.ndarray | None = None,
    confidence: dict[int, float] | None = None,
    samples: int = 9,
) -> dict[int, float]:
    """Return normalized per-cluster pressure from stretched graph edges.

    The signal is advisory only. It measures weighted inter-cluster edges whose
    current centroid relation is longer than the selected hierarchy seed, then
    boosts edges crossing hot congestion corridors. Returned scores are in
    ``[0, 1]`` so callers can blend them into existing heat-based ordering.
    """
    if not clusters or not edges:
        return {}
    cur_centroids = _cluster_centroids(hard_xy, clusters)
    if not cur_centroids:
        return {}
    seed_centroids = (
        _cluster_centroids(seed_hard_xy, clusters)
        if seed_hard_xy is not None and seed_hard_xy.shape == hard_xy.shape
        else cur_centroids
    )
    diag = max(float(np.hypot(cw, ch)), 1.0)
    raw = {int(cid): 0.0 for cid in clusters}
    for edge in edges:
        a = int(getattr(edge, "src", -1))
        b = int(getattr(edge, "dst", -1))
        if a not in cur_centroids or b not in cur_centroids:
            continue
        ca = cur_centroids[a]
        cb = cur_centroids[b]
        cur_dist = float(np.linalg.norm(ca - cb))
        seed_a = seed_centroids.get(a, ca)
        seed_b = seed_centroids.get(b, cb)
        seed_dist = max(float(np.linalg.norm(seed_a - seed_b)), diag * 0.01)
        stretch = max(0.0, cur_dist / seed_dist - 1.0)
        absolute = cur_dist / diag
        corridor = _corridor_field_mean(field, ca, cb, cw, ch, samples)
        weight = max(0.0, float(getattr(edge, "weight", 1.0)))
        conf_a = 1.0 if confidence is None else float(confidence.get(a, 1.0))
        conf_b = 1.0 if confidence is None else float(confidence.get(b, 1.0))
        conf_boost = 1.0 + 0.25 * max(0.0, 1.0 - min(conf_a, conf_b))
        pressure = weight * (stretch + 0.15 * absolute) * corridor * conf_boost
        raw[a] = raw.get(a, 0.0) + pressure
        raw[b] = raw.get(b, 0.0) + pressure
    max_score = max(raw.values()) if raw else 0.0
    if max_score <= 1e-12:
        return {}
    return {int(cid): float(score / max_score) for cid, score in raw.items() if score > 0.0}


def hard_tension_from_labels(labels: np.ndarray, cluster_tension: dict[int, float], n: int) -> np.ndarray:
    """Map cluster tension scores onto hard macro indices."""
    out = np.zeros(int(n), dtype=np.float64)
    if not cluster_tension:
        return out
    labels = np.asarray(labels, dtype=np.int64)
    limit = min(int(n), labels.size)
    for i in range(limit):
        out[i] = float(cluster_tension.get(int(labels[i]), 0.0))
    return out
