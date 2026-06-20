"""Derive macro communities ("hierarchy") from the flat netlist.

The ICCAD04 benchmarks ship no module hierarchy, so subsystems are inferred
from connectivity: union-find over low-fanout nets, treating each such net as
evidence that its hard-macro pins belong to the same logical group. High-fanout
nets (clocks, buses) connect everything and are skipped — they carry no grouping
signal. The result is a partition of movable hard macros into clusters, cached
on the plc and consumed by the hierarchy floorplan, region-relief, swap, and
coldspot-tightening passes.

The labels intentionally preserve connected subsystems in the current
hierarchy-only production path. Exact proxy gates still decide local relief
moves, but the selected system no longer optimizes for the lowest proxy at the
expense of hierarchy.
"""

from __future__ import annotations

import numpy as np

from utils import constants as const
from placer.scoring.wirelength import _build_wl_cache


def _heat_thresholds(cluster_heat, hot_percentile: float):
    if not cluster_heat:
        return None, None
    vals = np.asarray([float(v) for v in cluster_heat.values()], dtype=np.float64)
    if vals.size == 0:
        return None, None
    return float(np.percentile(vals, hot_percentile)), max(float(vals.max()), 1e-12)


def _heat_room_factor(
    cid: int,
    cluster_heat,
    heat_threshold,
    heat_max,
    heat_expand_frac: float,
    heat_escape_min: float,
) -> float:
    if not cluster_heat or heat_threshold is None or heat_expand_frac <= 0.0:
        return 1.0
    heat = float(cluster_heat.get(int(cid), 0.0))
    if heat < heat_threshold:
        return 1.0
    denom = max(float(heat_max) - float(heat_threshold), 1e-12)
    scale = np.clip((heat - float(heat_threshold)) / denom, 0.0, 1.0)
    scale = max(float(scale), float(heat_escape_min))
    return 1.0 + float(heat_expand_frac) * scale


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


def derive_hard_clusters(plc, n: int, n_soft: int = 0, max_fanout: int = 8, min_edge: int = 2):
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
        pin_refs = ref_idx[start : start + length]
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


def derive_cluster_softs(plc, n: int, n_soft: int, labels: np.ndarray, max_fanout: int = 8):
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
        refs = [int(r) for r in ref_idx[start : start + length]]
        cids = {int(labels[hb2a[r]]) for r in refs if r in hb2a and labels[hb2a[r]] >= 0}
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


def derive_soft_cluster_roles(
    plc,
    n: int,
    n_soft: int,
    labels: np.ndarray,
    max_fanout: int = 8,
    bridge_ratio: float = 0.6,
):
    """Split soft macros into owned and bridge roles.

    Owned softs have one dominant cluster affinity. Bridge softs connect two or
    more clusters with comparable strength and should live in a corridor between
    those clusters instead of being pulled fully inside one cluster.
    """
    key = (int(n), int(n_soft), int(max_fanout), id(labels), float(bridge_ratio))
    cached = getattr(plc, "_soft_cluster_roles", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]

    cache = _build_wl_cache(plc)
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]
    hb2a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}
    sb2s = {int(b): a for a, b in enumerate(plc.soft_macro_indices)}

    counts: "dict[tuple[int, int], int]" = {}
    for net_i in range(len(net_starts)):
        length = int(net_lengths[net_i])
        if length < 2 or length > max_fanout:
            continue
        start = int(net_starts[net_i])
        refs = [int(r) for r in ref_idx[start : start + length]]
        cids = {int(labels[hb2a[r]]) for r in refs if r in hb2a and labels[hb2a[r]] >= 0}
        if not cids:
            continue
        softs = [sb2s[r] for r in refs if r in sb2s]
        for s in softs:
            for cid in cids:
                counts[(s, cid)] = counts.get((s, cid), 0) + 1

    by_soft: "dict[int, list[tuple[int, int]]]" = {}
    for (s, cid), c in counts.items():
        by_soft.setdefault(int(s), []).append((int(cid), int(c)))

    owned: "dict[int, list[int]]" = {}
    bridge: "dict[int, np.ndarray]" = {}
    for s, vals in by_soft.items():
        vals.sort(key=lambda x: (-x[1], x[0]))
        best_cid, best_count = vals[0]
        tied = [cid for cid, c in vals if c >= max(1.0, bridge_ratio * best_count)]
        if len(tied) >= 2:
            bridge[s] = np.array(tied, dtype=np.int64)
        else:
            owned.setdefault(best_cid, []).append(n + s)

    owned_out = {cid: np.array(sorted(v), dtype=np.int64) for cid, v in owned.items()}
    plc._soft_cluster_roles = (key, owned_out, bridge)
    return owned_out, bridge


def cluster_max_fanout() -> int:
    """Net pin-count ceiling for cluster unioning."""
    return max(2, int(const.CLUSTER_MAX_FANOUT))


def cluster_min_edge() -> int:
    """Min shared-net count to merge two hard macros."""
    return max(1, int(const.CLUSTER_MIN_EDGE))


def compute_region_bbox(
    hard_xy,
    sizes,
    hw,
    hh,
    cw,
    ch,
    n,
    labels,
    clusters,
    target_density: float = 0.5,
    margin: float = 0.0,
    singleton_window: float = 0.05,
    cluster_heat=None,
    heat_expand_frac: float = 0.0,
    heat_hot_percentile: float = 70.0,
    heat_escape_min: float = 0.25,
) -> np.ndarray:
    """Per-macro CENTER-feasible region box [n,4] = (xlo, ylo, xhi, yhi).

    A macro's center must stay within its box to keep its whole footprint inside
    the cluster region (the box is pre-inset by the macro's half-extents, so a
    plain `xlo <= cx <= xhi` test region-locks it). Clustered macros share a box
    sized to give the cluster breathing room for congestion relief:
    `region_area = member_area / target_density`, at the cluster's current aspect
    ratio, never smaller than the current member footprint (so macros aren't
    trapped), centered on the cluster centroid, clipped to canvas by shifting.
    `margin>0` uses the simpler footprint+margin sizing instead. Singletons get a
    local window (`singleton_window`; 0 => pinned at their current spot).
    """
    region = np.empty((n, 4), dtype=np.float64)
    big = max(float(cw), float(ch))
    heat_threshold, heat_max = _heat_thresholds(cluster_heat, heat_hot_percentile)

    for cid, mem in clusters.items():
        mem = np.asarray(mem, dtype=np.int64)
        xs, ys = hard_xy[mem, 0], hard_xy[mem, 1]
        # Center on the footprint MIDPOINT (not the mean) so a region sized to
        # the footprint width always contains every member even when the
        # centroid is skewed.
        cx = float(xs.min() + xs.max()) / 2.0
        cy = float(ys.min() + ys.max()) / 2.0
        mhw, mhh = float(hw[mem].max()), float(hh[mem].max())
        bw0 = max(float(xs.max() - xs.min()) + 2.0 * mhw, 1e-6)
        bh0 = max(float(ys.max() - ys.min()) + 2.0 * mhh, 1e-6)
        if margin > 0.0:
            rw, rh = bw0 + 2.0 * margin * big, bh0 + 2.0 * margin * big
        else:
            member_area = float(np.sum(sizes[mem, 0] * sizes[mem, 1]))
            region_area = member_area / max(target_density, 1e-3)
            ar = bw0 / bh0
            rh = float(np.sqrt(region_area / ar))
            rw = ar * rh
            rw, rh = max(rw, bw0), max(rh, bh0)  # never below current footprint
        room_factor = _heat_room_factor(
            int(cid),
            cluster_heat,
            heat_threshold,
            heat_max,
            heat_expand_frac,
            heat_escape_min,
        )
        rw *= room_factor
        rh *= room_factor
        x0, x1 = cx - rw / 2.0, cx + rw / 2.0
        y0, y1 = cy - rh / 2.0, cy + rh / 2.0
        if x0 < 0.0:
            x1 -= x0
            x0 = 0.0
        if x1 > cw:
            x0 = max(x0 - (x1 - cw), 0.0)
            x1 = cw
        if y0 < 0.0:
            y1 -= y0
            y0 = 0.0
        if y1 > ch:
            y0 = max(y0 - (y1 - ch), 0.0)
            y1 = ch
        # Inset to center-feasible per member half-extent.
        region[mem, 0] = x0 + hw[mem]
        region[mem, 1] = y0 + hh[mem]
        region[mem, 2] = x1 - hw[mem]
        region[mem, 3] = y1 - hh[mem]

    # Singletons (unclustered): local window, or pinned when window is 0.
    sing = np.flatnonzero(labels < 0)
    if sing.size:
        w = singleton_window * big
        region[sing, 0] = np.clip(hard_xy[sing, 0] - w, hw[sing], cw - hw[sing])
        region[sing, 2] = np.clip(hard_xy[sing, 0] + w, hw[sing], cw - hw[sing])
        region[sing, 1] = np.clip(hard_xy[sing, 1] - w, hh[sing], ch - hh[sing])
        region[sing, 3] = np.clip(hard_xy[sing, 1] + w, hh[sing], ch - hh[sing])

    # Always contain the macro's current center, so a no-op move is in-region
    # (the macro is never forced out of its own box). This also subsumes the
    # degenerate-box case (footprint wider than region).
    region[:, 0] = np.minimum(region[:, 0], hard_xy[:, 0])
    region[:, 2] = np.maximum(region[:, 2], hard_xy[:, 0])
    region[:, 1] = np.minimum(region[:, 1], hard_xy[:, 1])
    region[:, 3] = np.maximum(region[:, 3], hard_xy[:, 1])
    return region


def _cluster_outer_region(
    hard_xy,
    sizes,
    hw,
    hh,
    cw,
    ch,
    mem,
    target_density: float,
    margin: float,
    cid: int | None = None,
    cluster_heat=None,
    heat_threshold=None,
    heat_max=None,
    heat_expand_frac: float = 0.0,
    heat_escape_min: float = 0.25,
) -> tuple[float, float, float, float]:
    """Return the unclipped-footprint cluster region as an outer canvas box."""
    mem = np.asarray(mem, dtype=np.int64)
    xs, ys = hard_xy[mem, 0], hard_xy[mem, 1]
    cx = float(xs.min() + xs.max()) / 2.0
    cy = float(ys.min() + ys.max()) / 2.0
    mhw, mhh = float(hw[mem].max()), float(hh[mem].max())
    bw0 = max(float(xs.max() - xs.min()) + 2.0 * mhw, 1e-6)
    bh0 = max(float(ys.max() - ys.min()) + 2.0 * mhh, 1e-6)
    big = max(float(cw), float(ch))
    if margin > 0.0:
        rw, rh = bw0 + 2.0 * margin * big, bh0 + 2.0 * margin * big
    else:
        member_area = float(np.sum(sizes[mem, 0] * sizes[mem, 1]))
        region_area = member_area / max(target_density, 1e-3)
        ar = bw0 / bh0
        rh = float(np.sqrt(region_area / ar))
        rw = ar * rh
        rw, rh = max(rw, bw0), max(rh, bh0)
    if cid is not None:
        room_factor = _heat_room_factor(
            int(cid),
            cluster_heat,
            heat_threshold,
            heat_max,
            heat_expand_frac,
            heat_escape_min,
        )
        rw *= room_factor
        rh *= room_factor
    x0, x1 = cx - rw / 2.0, cx + rw / 2.0
    y0, y1 = cy - rh / 2.0, cy + rh / 2.0
    if x0 < 0.0:
        x1 -= x0
        x0 = 0.0
    if x1 > cw:
        x0 = max(x0 - (x1 - cw), 0.0)
        x1 = cw
    if y0 < 0.0:
        y1 -= y0
        y0 = 0.0
    if y1 > ch:
        y0 = max(y0 - (y1 - ch), 0.0)
        y1 = ch
    return x0, y0, x1, y1


def compute_soft_region_bbox(
    hard_xy,
    soft_xy,
    hard_sizes,
    hard_hw,
    hard_hh,
    soft_hw,
    soft_hh,
    cw,
    ch,
    n,
    clusters,
    cluster_softs,
    bridge_softs=None,
    target_density: float = 0.5,
    margin: float = 0.0,
    singleton_window: float = 0.05,
    cluster_heat=None,
    heat_expand_frac: float = 0.0,
    heat_hot_percentile: float = 70.0,
    heat_escape_min: float = 0.25,
) -> np.ndarray:
    """Per-soft center-feasible region box [num_soft,4]."""
    num_soft = int(soft_xy.shape[0])
    region = np.empty((num_soft, 4), dtype=np.float64)
    assigned = np.zeros(num_soft, dtype=bool)
    heat_threshold, heat_max = _heat_thresholds(cluster_heat, heat_hot_percentile)

    for cid, soft_pidx in cluster_softs.items():
        mem = clusters.get(int(cid))
        if mem is None or len(mem) == 0:
            continue
        x0, y0, x1, y1 = _cluster_outer_region(
            hard_xy,
            hard_sizes,
            hard_hw,
            hard_hh,
            cw,
            ch,
            mem,
            target_density,
            margin,
            cid=int(cid),
            cluster_heat=cluster_heat,
            heat_threshold=heat_threshold,
            heat_max=heat_max,
            heat_expand_frac=heat_expand_frac,
            heat_escape_min=heat_escape_min,
        )
        for p in np.asarray(soft_pidx, dtype=np.int64):
            k = int(p) - int(n)
            if k < 0 or k >= num_soft:
                continue
            region[k, 0] = x0 + soft_hw[k]
            region[k, 1] = y0 + soft_hh[k]
            region[k, 2] = x1 - soft_hw[k]
            region[k, 3] = y1 - soft_hh[k]
            assigned[k] = True

    if bridge_softs:
        cluster_boxes = {}
        for cid, mem in clusters.items():
            cluster_boxes[int(cid)] = _cluster_outer_region(
                hard_xy,
                hard_sizes,
                hard_hw,
                hard_hh,
                cw,
                ch,
                mem,
                target_density,
                margin,
                cid=int(cid),
                cluster_heat=cluster_heat,
                heat_threshold=heat_threshold,
                heat_max=heat_max,
                heat_expand_frac=heat_expand_frac,
                heat_escape_min=heat_escape_min,
            )
        big = max(float(cw), float(ch))
        pad = max(singleton_window * big, 0.02 * big)
        for k, cids in bridge_softs.items():
            k = int(k)
            if k < 0 or k >= num_soft:
                continue
            boxes = [cluster_boxes[int(cid)] for cid in cids if int(cid) in cluster_boxes]
            if not boxes:
                continue
            x0 = max(0.0, min(b[0] for b in boxes) - pad)
            y0 = max(0.0, min(b[1] for b in boxes) - pad)
            x1 = min(float(cw), max(b[2] for b in boxes) + pad)
            y1 = min(float(ch), max(b[3] for b in boxes) + pad)
            region[k, 0] = x0 + soft_hw[k]
            region[k, 1] = y0 + soft_hh[k]
            region[k, 2] = x1 - soft_hw[k]
            region[k, 3] = y1 - soft_hh[k]
            assigned[k] = True

    unassigned = np.flatnonzero(~assigned)
    if unassigned.size:
        big = max(float(cw), float(ch))
        w = singleton_window * big
        region[unassigned, 0] = np.clip(
            soft_xy[unassigned, 0] - w, soft_hw[unassigned], cw - soft_hw[unassigned]
        )
        region[unassigned, 2] = np.clip(
            soft_xy[unassigned, 0] + w, soft_hw[unassigned], cw - soft_hw[unassigned]
        )
        region[unassigned, 1] = np.clip(
            soft_xy[unassigned, 1] - w, soft_hh[unassigned], ch - soft_hh[unassigned]
        )
        region[unassigned, 3] = np.clip(
            soft_xy[unassigned, 1] + w, soft_hh[unassigned], ch - soft_hh[unassigned]
        )

    region[:, 0] = np.minimum(region[:, 0], soft_xy[:, 0])
    region[:, 2] = np.maximum(region[:, 2], soft_xy[:, 0])
    region[:, 1] = np.minimum(region[:, 1], soft_xy[:, 1])
    region[:, 3] = np.maximum(region[:, 3], soft_xy[:, 1])
    return region


def hier_region_density() -> float:
    """Target packing density for area-based region sizing.

    Higher = tighter regions (more region-locked, less relief); lower = bigger
    regions (more congestion relief, looser hierarchy). 0.65 leans toward keeping
    macros locked while still relieving congestion.
    """
    return float(const.HIER_REGION_DENSITY)


def hier_region_margin() -> float:
    """Fractional-margin fallback region sizing."""
    return float(const.HIER_REGION_MARGIN)


def hier_region_singleton() -> float:
    """Local-window half-width fraction for unclustered macros."""
    return float(const.HIER_REGION_SINGLETON)
