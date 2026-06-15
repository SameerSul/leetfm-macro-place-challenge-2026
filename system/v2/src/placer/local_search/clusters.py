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


def derive_hard_clusters(plc, n: int, n_soft: int = 0, max_fanout: int = 8):
    """Partition movable hard macros [0, n) into connectivity clusters.

    Returns (labels, clusters):
      - labels: int array [n]; cluster id per hard macro, -1 if singleton.
      - clusters: dict cluster_id -> np.ndarray of member hard-macro indices
        (only clusters with >= 2 members are kept).

    In these flat netlists hard macros almost never share a net directly — they
    talk to standard cells (soft macros), which then talk to other hard macros.
    So union-find runs over BOTH hard and soft nodes on low-fanout nets, and the
    hard macros that land in the same component (possibly connected only through
    shared softs) form a cluster. High-fanout nets (> max_fanout, e.g. clocks /
    buses) connect everything and carry no grouping signal, so they are skipped.
    Cached on plc keyed by (n, n_soft, max_fanout).
    """
    key = (int(n), int(n_soft), int(max_fanout))
    cached = getattr(plc, "_hard_clusters", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]

    cache = _build_wl_cache(plc)
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]

    # Macro nodes occupy ref indices [0, n_nodes); ports use larger pin indices.
    n_nodes = n + int(n_soft)
    parents = _union_find_parents(n_nodes)
    for net_i in range(len(net_starts)):
        length = int(net_lengths[net_i])
        if length < 2 or length > max_fanout:
            continue
        start = int(net_starts[net_i])
        pin_refs = ref_idx[start:start + length]
        nodes = pin_refs[(pin_refs >= 0) & (pin_refs < n_nodes)]
        if nodes.size < 2:
            continue
        anchor = _find(parents, int(nodes[0]))
        for r in nodes[1:]:
            parents[_find(parents, int(r))] = anchor

    # Read off components restricted to hard macros [0, n).
    roots = np.array([_find(parents, i) for i in range(n)], dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    clusters: "dict[int, np.ndarray]" = {}
    next_id = 0
    for root in np.unique(roots):
        members = np.flatnonzero(roots == root)
        if members.size < 2:
            continue
        labels[members] = next_id
        clusters[next_id] = members
        next_id += 1

    plc._hard_clusters = (key, labels, clusters)
    return labels, clusters


def cluster_max_fanout() -> int:
    """Net pin-count ceiling for cluster unioning (V2_CLUSTER_MAX_FANOUT)."""
    try:
        return max(2, int(os.environ.get("V2_CLUSTER_MAX_FANOUT", "8")))
    except ValueError:
        return 8
