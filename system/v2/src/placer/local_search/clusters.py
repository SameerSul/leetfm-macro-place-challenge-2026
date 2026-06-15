"""Derive macro communities ("hierarchy") from the flat netlist.

The ICCAD04 benchmarks ship no module hierarchy, so subsystems are inferred
from connectivity: union-find over low-fanout nets, treating each such net as
evidence that its hard-macro pins belong to the same logical group. High-fanout
nets (clocks, buses) connect everything and are skipped — they carry no grouping
signal. The result is a partition of movable hard macros into clusters, cached
on the plc and consumed by the LSMC cluster-coherent kick.

This only *labels* groups; it never forces them together. Clustering connected
macros tightly is anti-correlated with the congestion-dominated proxy (see
PROGRESS.md). The labels are used solely to translate a subsystem as a unit
during exploration, where the exact proxy gate still decides every accept.
"""

from __future__ import annotations

import os

import numpy as np

from placer.scoring.wirelength import _build_wl_cache


def _union_find_parents(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


def _find(parents: np.ndarray, i: int) -> int:
    root = i
    while parents[root] != root:
        root = parents[root]
    # Path compression.
    while parents[i] != root:
        parents[i], i = root, parents[i]
    return root


def derive_hard_clusters(plc, n: int, n_soft: int = 0, max_fanout: int = 8,
                         min_edge: int = 2):
    """Partition movable hard macros [0, n) into connectivity clusters.

    Returns (labels, clusters):
      - labels: int array [n]; cluster id per hard macro, -1 if unclustered.
      - clusters: dict cluster_id -> np.ndarray of member hard-macro indices
        (placement space A; only clusters with >= 2 members are kept).

    **Method.** Build a weighted hard-macro graph: each low-fanout net
    (2..max_fanout pins) contributes a clique among its hard-macro pins, edge
    weight = number of such shared nets. Union-find merges only edges with
    weight >= min_edge, then connected components of >= 2 macros are clusters.
    The threshold is essential: at min_edge=1 the graph is nearly one blob
    (a few shared nets chain everything); min_edge>=2 yields separable
    subsystems (e.g. ibm01: 9 clusters of 21..53 macros).

    **Index spaces.** `_build_wl_cache` ref indices are `modules_w_pins` indices
    (space B: ports 0.., then hard, then soft — NOT placement order). Hard pins
    are identified by membership in `plc.hard_macro_indices` (space B), and the
    resulting components are mapped back to placement space A
    (`i ∈ [0, n)` = `hard_macro_indices[i]`), which is what the kick and
    relocation use. Cached on plc keyed by (n, n_soft, max_fanout, min_edge).
    """
    key = (int(n), int(n_soft), int(max_fanout), int(min_edge))
    cached = getattr(plc, "_hard_clusters", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]

    cache = _build_wl_cache(plc)
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]

    b_to_a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}

    # Accumulate hard-hard edge weights (space A) from low-fanout net cliques.
    edge_w: "dict[tuple[int, int], int]" = {}
    for net_i in range(len(net_starts)):
        length = int(net_lengths[net_i])
        if length < 2 or length > max_fanout:
            continue
        start = int(net_starts[net_i])
        pin_refs = ref_idx[start:start + length]
        hard_a = sorted({b_to_a[int(r)] for r in pin_refs if int(r) in b_to_a})
        if len(hard_a) < 2:
            continue
        for i in range(len(hard_a)):
            for j in range(i + 1, len(hard_a)):
                e = (hard_a[i], hard_a[j])
                edge_w[e] = edge_w.get(e, 0) + 1

    parents = _union_find_parents(n)
    for (a, b), w in edge_w.items():
        if w >= min_edge:
            parents[_find(parents, a)] = _find(parents, b)

    roots: "dict[int, list[int]]" = {}
    for i in range(n):
        roots.setdefault(_find(parents, i), []).append(i)

    labels = np.full(n, -1, dtype=np.int64)
    clusters: "dict[int, np.ndarray]" = {}
    next_id = 0
    for members_a in roots.values():
        if len(members_a) < 2:
            continue
        arr = np.array(sorted(members_a), dtype=np.int64)
        labels[arr] = next_id
        clusters[next_id] = arr
        next_id += 1

    plc._hard_clusters = (key, labels, clusters)
    return labels, clusters


def derive_cluster_softs(plc, n: int, n_soft: int, labels: np.ndarray,
                         max_fanout: int = 8):
    """Map each hard cluster to the soft macros it drives.

    For every low-fanout net, the soft pins are attributed to the cluster(s) of
    the net's hard pins; each soft is then assigned to the single cluster it
    shares the most such nets with. Returns dict cluster_id -> np.ndarray of
    soft indices in PLACEMENT space (n + soft_order), so the kick can co-move a
    subsystem's softs with its hard macros. Cached on plc.

    Index spaces: hard pins via `hard_macro_indices` (space B) -> space A label;
    soft pins via `soft_macro_indices` (space B) -> placement index n+order.
    """
    key = (int(n), int(n_soft), int(max_fanout), id(labels))
    cached = getattr(plc, "_cluster_softs", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    cache = _build_wl_cache(plc)
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]
    hb2a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}
    sb2p = {int(b): n + a for a, b in enumerate(plc.soft_macro_indices)}

    # Count (soft placement idx, cluster id) co-occurrences over low-fanout nets.
    counts: "dict[tuple[int, int], int]" = {}
    for net_i in range(len(net_starts)):
        length = int(net_lengths[net_i])
        if length < 2 or length > max_fanout:
            continue
        start = int(net_starts[net_i])
        refs = [int(r) for r in ref_idx[start:start + length]]
        cids = {int(labels[hb2a[r]]) for r in refs
                if r in hb2a and labels[hb2a[r]] >= 0}
        if not cids:
            continue
        softs = [sb2p[r] for r in refs if r in sb2p]
        for s in softs:
            for cid in cids:
                counts[(s, cid)] = counts.get((s, cid), 0) + 1

    best: "dict[int, tuple[int, int]]" = {}  # soft -> (best_count, cid)
    for (s, cid), c in counts.items():
        if s not in best or c > best[s][0]:
            best[s] = (c, cid)

    cluster_softs: "dict[int, list[int]]" = {}
    for s, (_c, cid) in best.items():
        cluster_softs.setdefault(cid, []).append(s)
    out = {cid: np.array(sorted(v), dtype=np.int64) for cid, v in cluster_softs.items()}

    plc._cluster_softs = (key, out)
    return out


def cluster_max_fanout() -> int:
    """Net pin-count ceiling for cluster unioning (V2_CLUSTER_MAX_FANOUT)."""
    try:
        return max(2, int(os.environ.get("V2_CLUSTER_MAX_FANOUT", "8")))
    except ValueError:
        return 8


def cluster_min_edge() -> int:
    """Min shared-net count to merge two hard macros (V2_CLUSTER_MIN_EDGE)."""
    try:
        return max(1, int(os.environ.get("V2_CLUSTER_MIN_EDGE", "2")))
    except ValueError:
        return 2
