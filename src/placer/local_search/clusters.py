"""Derive macro communities ("hierarchy") from the flat netlist.

The ICCAD04 benchmarks ship no module hierarchy, so subsystems are inferred
from low-fanout connectivity. The optional one-level refinement also requires
the connected macros to form a compact physical group in the initial placement,
using shared-soft affinity, local macro-area density, and placed wire demand as
confidence evidence. High-fanout nets (clocks, buses) connect everything and
are skipped. The result is consumed by the hierarchy floorplan, region-relief,
swap, and coldspot-tightening passes.

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


def derive_path_tag_hard_clusters(plc, n: int):
    """Partition hard macros by explicit slash-separated instance-path tags.

    NG45 macro names carry real RTL hierarchy paths. When those paths produce
    enough nontrivial groups, they are a stronger hierarchy signal than the
    flat netlist's sparse low-fanout macro graph. Flat-name benchmarks return
    ``None`` so connectivity-derived clustering remains the default.
    """
    max_depth = max(1, int(const.HIER_TAG_PREFIX_MAX_DEPTH))
    min_group = max(2, int(const.HIER_TAG_PREFIX_MIN_GROUP))
    min_coverage = float(const.HIER_TAG_PREFIX_MIN_COVERAGE)
    key = (int(n), max_depth, min_group, min_coverage, "path_tags")
    cached = getattr(plc, "_hard_clusters_path_tags", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    try:
        hard_b = list(plc.hard_macro_indices[:n])
        names = [plc.modules_w_pins[int(i)].get_name() for i in hard_b]
    except Exception:
        return None
    if sum(1 for name in names if "/" in str(name)) < max(min_group, int(0.5 * n)):
        return None

    def _prefix(name: str, depth: int) -> str:
        parts = [p for p in str(name).split("/") if p]
        if len(parts) < depth:
            return str(name)
        return "/".join(parts[:depth])

    best_groups = None
    best_depth = None
    best_score = (-1, -1, 0)
    for depth in range(1, max_depth + 1):
        buckets: dict[str, list[int]] = {}
        for i, name in enumerate(names):
            buckets.setdefault(_prefix(name, depth), []).append(i)
        groups = [
            np.asarray(sorted(v), dtype=np.int64)
            for v in buckets.values()
            if min_group <= len(v) < n
        ]
        if not groups:
            continue
        covered = int(sum(len(g) for g in groups))
        coverage = covered / max(1, n)
        if coverage < min_coverage:
            continue
        score = (len(groups), covered, -max(int(len(g)) for g in groups))
        if score > best_score:
            best_groups = groups
            best_depth = int(depth)
            best_score = score
    if not best_groups:
        return None

    labels = np.full(n, -1, dtype=np.int64)
    clusters: dict[int, np.ndarray] = {}
    for cid, members in enumerate(best_groups):
        labels[members] = int(cid)
        clusters[int(cid)] = members

    # Retain exactly one useful ancestor level above the selected leaf
    # partition. The production partition remains unchanged; this metadata is
    # used only by the bounded child-group search.
    parent_clusters: dict[int, np.ndarray] = {}
    parent_children: dict[int, tuple[int, ...]] = {}
    child_parent: dict[int, int] = {}
    parent_depth = None
    for depth in range(max(1, int(best_depth or 1) - 1), 0, -1):
        children_by_prefix: dict[str, list[int]] = {}
        for cid, members in clusters.items():
            prefix = _prefix(names[int(members[0])], depth)
            children_by_prefix.setdefault(prefix, []).append(int(cid))
        useful = [
            (prefix, tuple(sorted(child_ids)))
            for prefix, child_ids in children_by_prefix.items()
            if len(child_ids) >= 2
        ]
        if not useful:
            continue
        for parent_id, (_prefix_name, child_ids) in enumerate(sorted(useful)):
            members = np.unique(
                np.concatenate([clusters[int(child_id)] for child_id in child_ids])
            ).astype(np.int64)
            parent_clusters[int(parent_id)] = members
            parent_children[int(parent_id)] = child_ids
            for child_id in child_ids:
                child_parent[int(child_id)] = int(parent_id)
        parent_depth = int(depth)
        break

    plc._hard_clusters_path_tag_hierarchy = (
        key,
        parent_clusters,
        parent_children,
        child_parent,
        parent_depth,
    )
    plc._hard_clusters_path_tags = (key, labels, clusters)
    return labels, clusters


def derive_oversized_hard_clusters(
    plc,
    n: int,
    n_soft: int = 0,
    max_fanout: int = 8,
    min_edge: int = 2,
    hard_sizes: "np.ndarray | None" = None,
):
    """Split only flat clusters that dominate the hard-macro population.

    This is a conservative middle ground between flat inferred clusters and the
    old absolute-size recursive clustering. A flat component must exceed
    `HIER_OVERSIZE_CLUSTER_START_FRAC` of hard macros before bisection is even
    considered; accepted splits then recurse until leaves are below
    `HIER_OVERSIZE_CLUSTER_TARGET_FRAC` of hard macros.

    A single flat component cannot produce bridge-soft evidence because every
    soft has only one possible owner. When that component covers nearly all
    hard macros, shared hard-to-soft affinity may recover subgroups; a stricter
    graph cut can retain a partial split when affinity is inconclusive.
    Multi-component designs continue to require component-local bridge softs.
    """
    start_frac = float(const.HIER_OVERSIZE_CLUSTER_START_FRAC)
    target_frac = float(const.HIER_OVERSIZE_CLUSTER_TARGET_FRAC)
    key = (
        int(n),
        int(n_soft),
        int(max_fanout),
        int(min_edge),
        float(start_frac),
        float(target_frac),
        float(const.HIER_OVERSIZE_CLUSTER_TARGET_TOL),
        int(const.HIER_OVERSIZE_CLUSTER_MIN_BRIDGE_SOFTS),
        int(const.HIER_OVERSIZE_CLUSTER_MIN_SIZE),
        float(const.HIER_OVERSIZE_CLUSTER_MAX_CUT_RATIO),
        float(const.HIER_SINGLE_COMPONENT_SPLIT_MIN_COVERAGE),
        int(const.HIER_SINGLE_COMPONENT_SPLIT_MIN_SIZE),
        float(const.HIER_SINGLE_COMPONENT_SPLIT_MAX_CUT_RATIO),
        float(const.HIER_SINGLE_COMPONENT_SOFT_COSINE),
        "oversized",
    )
    cached = getattr(plc, "_hard_clusters_oversized", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]

    flat_labels, flat_clusters = derive_hard_clusters(
        plc,
        n,
        n_soft=n_soft,
        max_fanout=max_fanout,
        min_edge=min_edge,
    )
    _edge_count, edge_weight = _hard_edge_maps(plc, n, max_fanout)
    areas = _cluster_partition_areas(n, hard_sizes)
    start_size = max(2, int(np.floor(max(0.0, start_frac) * float(n))))
    target_size = max(2, int(np.floor(max(0.0, target_frac) * float(n))))
    target_accept = max(
        target_size, int(np.ceil(target_size * float(const.HIER_OVERSIZE_CLUSTER_TARGET_TOL)))
    )
    min_size = max(2, int(const.HIER_OVERSIZE_CLUSTER_MIN_SIZE))
    max_cut_ratio = float(const.HIER_OVERSIZE_CLUSTER_MAX_CUT_RATIO)
    single_component_min_coverage = float(const.HIER_SINGLE_COMPONENT_SPLIT_MIN_COVERAGE)
    single_component_min_size = max(2, int(const.HIER_SINGLE_COMPONENT_SPLIT_MIN_SIZE))
    single_component_max_cut_ratio = float(const.HIER_SINGLE_COMPONENT_SPLIT_MAX_CUT_RATIO)
    single_component_soft_cosine = float(const.HIER_SINGLE_COMPONENT_SOFT_COSINE)
    min_bridge_softs = max(0, int(const.HIER_OVERSIZE_CLUSTER_MIN_BRIDGE_SOFTS))
    if min_bridge_softs > 0:
        _owned_flat, bridge_flat = derive_soft_cluster_roles(
            plc,
            n,
            n_soft,
            flat_labels,
            max_fanout=max_fanout,
            bridge_ratio=float(const.HIER_BRIDGE_SOFT_RATIO),
        )
        comp_bridge_soft_members: dict[int, set[int]] = {}
        for soft_idx, comp_ids in bridge_flat.items():
            for comp_id in np.asarray(comp_ids, dtype=np.int64).reshape(-1):
                comp_bridge_soft_members.setdefault(int(comp_id), set()).add(int(soft_idx))
        comp_bridge_soft_counts = {
            int(comp_id): len(members) for comp_id, members in comp_bridge_soft_members.items()
        }
    else:
        comp_bridge_soft_counts = {}

    leaves: list[np.ndarray] = []
    plc._hard_clusters_oversized_source = "hierarchy_oversized_connectivity"
    for comp_id, comp in flat_clusters.items():
        comp = np.asarray(comp, dtype=np.int64)
        if comp.size < 2:
            continue
        comp_bridge_softs = int(comp_bridge_soft_counts.get(int(comp_id), 0))
        bridge_supported = comp_bridge_softs >= min_bridge_softs
        single_component_supported = bool(
            len(flat_clusters) == 1
            and float(comp.size) / max(float(n), 1.0) >= single_component_min_coverage
        )
        if (not bridge_supported and not single_component_supported) or comp.size <= start_size:
            leaves.append(comp)
            continue
        if single_component_supported and not bridge_supported:
            soft_affinity_leaves = _single_component_soft_affinity_split(
                plc,
                n,
                n_soft,
                max_fanout=max_fanout,
                min_size=single_component_min_size,
                cosine_threshold=single_component_soft_cosine,
            )
            if soft_affinity_leaves is not None:
                leaves.extend(soft_affinity_leaves)
                plc._hard_clusters_oversized_source = "hierarchy_single_component_soft_affinity"
                continue
        split_cut_ratio = max_cut_ratio if bridge_supported else single_component_max_cut_ratio
        split_min_size = min_size if bridge_supported else single_component_min_size
        split_leaves = _recursive_bisect_component(
            comp,
            edge_weight,
            areas,
            max_size=target_size,
            min_size=split_min_size,
            max_cut_ratio=split_cut_ratio,
        )
        reached_target = max(int(len(leaf)) for leaf in split_leaves) <= target_accept
        if len(split_leaves) <= 1 or (bridge_supported and not reached_target):
            leaves.append(comp)
        else:
            leaves.extend(split_leaves)

    labels = np.full(n, -1, dtype=np.int64)
    clusters: "dict[int, np.ndarray]" = {}
    next_id = 0
    for leaf in leaves:
        leaf = np.asarray(sorted(int(x) for x in leaf), dtype=np.int64)
        if leaf.size < 2:
            continue
        labels[leaf] = next_id
        clusters[next_id] = leaf
        next_id += 1

    plc._hard_clusters_oversized = (key, labels, clusters)
    return labels, clusters


def _hard_edge_maps(plc, n: int, max_fanout: int):
    cache = _build_wl_cache(plc)
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]
    net_weights = cache["net_weights"]
    b_to_a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}
    edge_count: "dict[tuple[int, int], int]" = {}
    edge_weight: "dict[tuple[int, int], float]" = {}
    for net_i in range(len(net_starts)):
        length = int(net_lengths[net_i])
        if length < 2 or length > max_fanout:
            continue
        start = int(net_starts[net_i])
        pin_refs = ref_idx[start : start + length]
        hard_a = sorted({b_to_a[int(r)] for r in pin_refs if int(r) in b_to_a})
        if len(hard_a) < 2:
            continue
        weight = float(net_weights[net_i])
        for i in range(len(hard_a)):
            for j in range(i + 1, len(hard_a)):
                e = (hard_a[i], hard_a[j])
                edge_count[e] = edge_count.get(e, 0) + 1
                edge_weight[e] = edge_weight.get(e, 0.0) + weight
    return edge_count, edge_weight


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    """Scale a nonnegative signal by its robust upper tail."""
    arr = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    positive = arr[arr > 0.0]
    if positive.size == 0:
        return np.zeros_like(arr)
    scale = max(float(np.percentile(positive, 90.0)), 1.0e-12)
    return np.clip(arr / scale, 0.0, 1.0)


def _spatial_structural_context(
    plc,
    n: int,
    n_soft: int,
    max_fanout: int,
    hard_sizes,
    direct_edges: dict[tuple[int, int], float],
    *,
    shared_soft_weight: float,
    proximity_weight: float,
    pressure_weight: float,
    spatial_neighbors: int,
    max_soft_degree: int,
):
    """Build connectivity affinities reinforced by initial physical evidence."""
    try:
        hard_refs = list(plc.hard_macro_indices[:n])
        soft_refs = list(plc.soft_macro_indices[:n_soft])
        hard_pos = np.asarray(
            [plc.modules_w_pins[int(ref)].get_pos() for ref in hard_refs],
            dtype=np.float64,
        )
        soft_pos = np.asarray(
            [plc.modules_w_pins[int(ref)].get_pos() for ref in soft_refs],
            dtype=np.float64,
        ).reshape(-1, 2)
        if hard_pos.shape != (int(n), 2) or not np.all(np.isfinite(hard_pos)):
            return None
        if hard_sizes is None:
            hard_wh = np.asarray(
                [
                    (
                        plc.modules_w_pins[int(ref)].get_width(),
                        plc.modules_w_pins[int(ref)].get_height(),
                    )
                    for ref in hard_refs
                ],
                dtype=np.float64,
            )
        else:
            hard_wh = np.asarray(hard_sizes[:n], dtype=np.float64).reshape(n, 2)
        soft_wh = np.asarray(
            [
                (
                    plc.modules_w_pins[int(ref)].get_width(),
                    plc.modules_w_pins[int(ref)].get_height(),
                )
                for ref in soft_refs
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        canvas_w, canvas_h = plc.get_canvas_width_height()
    except Exception:
        return None

    hard_area = np.maximum(hard_wh[:, 0] * hard_wh[:, 1], 1.0e-12)
    soft_area = (
        np.maximum(soft_wh[:, 0] * soft_wh[:, 1], 1.0e-12)
        if soft_wh.size
        else np.zeros(0, dtype=np.float64)
    )
    canvas_diag = max(float(np.hypot(canvas_w, canvas_h)), 1.0e-12)
    hard_dist2 = np.sum((hard_pos[:, None, :] - hard_pos[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(hard_dist2, np.inf)
    neighbor = min(max(1, int(spatial_neighbors)), max(int(n) - 1, 1))
    kth = np.partition(hard_dist2, neighbor - 1, axis=1)[:, neighbor - 1]
    finite_kth = kth[np.isfinite(kth)]
    radius = float(np.sqrt(np.median(finite_kth))) if finite_kth.size else 0.05 * canvas_diag
    radius = max(
        radius,
        2.0 * float(np.sqrt(np.median(hard_area))),
        0.01 * canvas_diag,
        1.0e-9,
    )

    all_pos = hard_pos if not soft_pos.size else np.vstack([hard_pos, soft_pos])
    all_area = hard_area if not soft_area.size else np.concatenate([hard_area, soft_area])
    local_dist2 = np.sum((hard_pos[:, None, :] - all_pos[None, :, :]) ** 2, axis=2)
    density_signal = np.exp(-local_dist2 / max(radius * radius, 1.0e-12)) @ all_area
    density_signal /= max(np.pi * radius * radius, 1.0e-12)

    cache = _build_wl_cache(plc)
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]
    net_weights = cache["net_weights"]
    hard_by_ref = {int(ref): index for index, ref in enumerate(hard_refs)}
    soft_by_ref = {int(ref): index for index, ref in enumerate(soft_refs)}
    by_soft: dict[int, dict[int, float]] = {}
    wire_signal = np.zeros(n, dtype=np.float64)
    for net_index, start_raw in enumerate(net_starts):
        length = int(net_lengths[net_index])
        if length < 2 or length > int(max_fanout):
            continue
        start = int(start_raw)
        refs = {int(ref) for ref in ref_idx[start : start + length]}
        hard = [hard_by_ref[ref] for ref in refs if ref in hard_by_ref]
        soft = [soft_by_ref[ref] for ref in refs if ref in soft_by_ref]
        if not hard:
            continue
        weight = float(net_weights[net_index])
        endpoint_pos = [hard_pos[int(index)] for index in hard]
        endpoint_pos.extend(soft_pos[int(index)] for index in soft)
        if len(endpoint_pos) >= 2:
            endpoints = np.asarray(endpoint_pos, dtype=np.float64)
            placed_span = float(np.sum(np.ptp(endpoints, axis=0))) / radius
        else:
            placed_span = 0.0
        route_demand = 1.0 + min(max(placed_span, 0.0), 4.0)
        for hard_index in hard:
            wire_signal[int(hard_index)] += weight * max(length - 1, 1) * route_demand
            for soft_index in soft:
                rows = by_soft.setdefault(int(soft_index), {})
                rows[int(hard_index)] = rows.get(int(hard_index), 0.0) + weight

    shared_raw: dict[tuple[int, int], float] = {}
    shared_spatial: dict[tuple[int, int], float] = {}
    for soft_index, hard_rows in by_soft.items():
        rows = sorted(hard_rows.items())
        if len(rows) < 2 or len(rows) > max(2, int(max_soft_degree)):
            continue
        soft_xy = soft_pos[int(soft_index)]
        for left_index, (left, left_weight) in enumerate(rows):
            for right, right_weight in rows[left_index + 1 :]:
                key = (min(int(left), int(right)), max(int(left), int(right)))
                support = min(float(left_weight), float(right_weight))
                mean_distance = 0.5 * (
                    float(np.linalg.norm(hard_pos[int(left)] - soft_xy))
                    + float(np.linalg.norm(hard_pos[int(right)] - soft_xy))
                )
                proximity = float(np.exp(-((mean_distance / radius) ** 2)))
                shared_raw[key] = shared_raw.get(key, 0.0) + support
                shared_spatial[key] = shared_spatial.get(key, 0.0) + support * proximity

    structural = {key: float(value) for key, value in direct_edges.items() if value > 0.0}
    for key, value in shared_raw.items():
        structural[key] = structural.get(key, 0.0) + float(shared_soft_weight) * float(value)
    if not structural:
        return None

    for (left, right), weight in structural.items():
        wire_signal[int(left)] += float(weight)
        wire_signal[int(right)] += float(weight)
    density_norm = _robust_unit_interval(density_signal)
    wire_norm = _robust_unit_interval(wire_signal)
    pressure = 0.5 * density_norm + 0.5 * wire_norm

    affinity: dict[tuple[int, int], float] = {}
    for key, raw_weight in structural.items():
        left, right = key
        distance = float(np.sqrt(hard_dist2[int(left), int(right)]))
        hard_proximity = float(np.exp(-((distance / radius) ** 2)))
        pair_pressure = float(np.sqrt(pressure[int(left)] * pressure[int(right)]))
        direct = float(direct_edges.get(key, 0.0))
        soft = float(shared_spatial.get(key, 0.0)) * float(shared_soft_weight)
        placed_weight = direct * (1.0 + float(proximity_weight) * hard_proximity) + soft
        placed_weight *= 1.0 + float(pressure_weight) * hard_proximity * pair_pressure
        if placed_weight > 0.0:
            affinity[key] = placed_weight
    if not affinity:
        return None
    return {
        "hard_pos": hard_pos,
        "hard_wh": hard_wh,
        "hard_area": hard_area,
        "structural": structural,
        "affinity": affinity,
        "pressure": pressure,
        "radius": float(radius),
    }


def _bbox_utilization(
    hard_pos: np.ndarray,
    hard_wh: np.ndarray,
    hard_area: np.ndarray,
    members: np.ndarray,
) -> float:
    """Return footprint utilization of one hard-macro set's enclosing box."""
    members = np.asarray(members, dtype=np.int64)
    if members.size == 0:
        return 0.0
    lo = np.min(hard_pos[members] - 0.5 * hard_wh[members], axis=0)
    hi = np.max(hard_pos[members] + 0.5 * hard_wh[members], axis=0)
    bbox_area = max(float(np.prod(np.maximum(hi - lo, 1.0e-12))), 1.0e-12)
    return float(np.sum(hard_area[members]) / bbox_area)


def _spatial_split_evidence(
    members: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    context: dict[str, object],
) -> dict[str, float | str]:
    """Score whether a topological split also looks like two placed subsystems."""
    hard_pos = np.asarray(context["hard_pos"], dtype=np.float64)
    hard_wh = np.asarray(context["hard_wh"], dtype=np.float64)
    hard_area = np.asarray(context["hard_area"], dtype=np.float64)
    pressure = np.asarray(context["pressure"], dtype=np.float64)
    structural = context["structural"]
    member_set = {int(index) for index in np.asarray(members, dtype=np.int64)}
    left_set = {int(index) for index in np.asarray(left, dtype=np.int64)}
    right_set = {int(index) for index in np.asarray(right, dtype=np.int64)}

    total_weight = 0.0
    cut_weight = 0.0
    internal_weight = 0.0
    for (a, b), weight_raw in structural.items():
        if int(a) not in member_set or int(b) not in member_set:
            continue
        weight = float(weight_raw)
        total_weight += weight
        if (int(a) in left_set and int(b) in right_set) or (
            int(a) in right_set and int(b) in left_set
        ):
            cut_weight += weight
        else:
            internal_weight += weight
    if total_weight <= 0.0:
        return {
            "source": "placement_spatial_structural",
            "confidence": 0.0,
            "cut_ratio": 1.0,
            "compactness_gain": 0.0,
            "density_gain": 0.0,
            "wire_gain": 0.0,
            "pressure_support": 0.0,
        }

    parent_points = hard_pos[np.asarray(members, dtype=np.int64)]
    parent_dist = np.linalg.norm(parent_points[:, None, :] - parent_points[None, :, :], axis=2)
    parent_upper = parent_dist[np.triu_indices(parent_points.shape[0], 1)]
    parent_mean = float(np.mean(parent_upper)) if parent_upper.size else 0.0
    within_sum = 0.0
    within_count = 0
    for child in (left, right):
        child_points = hard_pos[np.asarray(child, dtype=np.int64)]
        child_dist = np.linalg.norm(
            child_points[:, None, :] - child_points[None, :, :],
            axis=2,
        )
        child_upper = child_dist[np.triu_indices(child_points.shape[0], 1)]
        within_sum += float(np.sum(child_upper))
        within_count += int(child_upper.size)
    within_mean = within_sum / max(float(within_count), 1.0)
    compactness_gain = float(np.clip(1.0 - within_mean / max(parent_mean, 1.0e-12), 0.0, 1.0))

    parent_util = _bbox_utilization(hard_pos, hard_wh, hard_area, members)
    child_util = 0.5 * (
        _bbox_utilization(hard_pos, hard_wh, hard_area, left)
        + _bbox_utilization(hard_pos, hard_wh, hard_area, right)
    )
    density_gain = float(np.clip(child_util / max(parent_util, 1.0e-12) - 1.0, 0.0, 1.0))
    parent_span = np.ptp(parent_points, axis=0)
    parent_bbox_area = max(float(np.prod(np.maximum(parent_span, 1.0e-12))), 1.0e-12)
    child_bbox_area = 0.0
    for child in (left, right):
        span = np.ptp(hard_pos[np.asarray(child, dtype=np.int64)], axis=0)
        child_bbox_area += max(float(np.prod(np.maximum(span, 1.0e-12))), 1.0e-12)
    parent_wire_density = total_weight / parent_bbox_area
    child_wire_density = internal_weight / max(child_bbox_area, 1.0e-12)
    wire_gain = float(
        np.clip(child_wire_density / max(parent_wire_density, 1.0e-12) - 1.0, 0.0, 1.0)
    )
    cut_ratio = float(cut_weight / total_weight)
    connectivity = float(np.clip(1.0 - cut_ratio, 0.0, 1.0))
    pressure_support = float(np.mean(pressure[np.asarray(members, dtype=np.int64)]))
    confidence = (
        0.50 * connectivity
        + 0.25 * compactness_gain
        + 0.10 * density_gain
        + 0.10 * wire_gain
        + 0.05 * pressure_support
    )
    return {
        "source": "placement_spatial_structural",
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "cut_ratio": cut_ratio,
        "compactness_gain": compactness_gain,
        "density_gain": density_gain,
        "wire_gain": wire_gain,
        "pressure_support": pressure_support,
    }


def derive_one_level_hard_subclusters(
    plc,
    n: int,
    clusters: dict[int, np.ndarray],
    *,
    max_fanout: int,
    n_soft: int = 0,
    hard_sizes=None,
    min_parent_size: int,
    min_child_size: int,
    max_cut_ratio: float,
    shared_soft_weight: float = 0.75,
    proximity_weight: float = 1.0,
    pressure_weight: float = 0.50,
    spatial_neighbors: int = 8,
    max_soft_degree: int = 24,
    min_compactness_gain: float = 0.10,
    min_confidence: float = 0.54,
):
    """Infer one child level from connectivity reinforced by placed evidence."""
    _edge_count, edge_weight = _hard_edge_maps(plc, n, max_fanout)
    spatial = _spatial_structural_context(
        plc,
        n,
        n_soft,
        max_fanout,
        hard_sizes,
        edge_weight,
        shared_soft_weight=float(shared_soft_weight),
        proximity_weight=float(proximity_weight),
        pressure_weight=float(pressure_weight),
        spatial_neighbors=max(1, int(spatial_neighbors)),
        max_soft_degree=max(2, int(max_soft_degree)),
    )
    if spatial is None:
        return {}, {}, {}, {}, {}
    areas = _cluster_partition_areas(n, hard_sizes)
    min_child = max(2, int(min_child_size))
    min_parent = max(2 * min_child, int(min_parent_size))
    retained_parents: dict[int, np.ndarray] = {}
    subclusters: dict[int, np.ndarray] = {}
    parent_children: dict[int, tuple[int, ...]] = {}
    child_parent: dict[int, int] = {}
    parent_evidence: dict[int, dict[str, float | str]] = {}
    next_child = 0
    for parent_id, members_raw in sorted(clusters.items()):
        members = np.asarray(members_raw, dtype=np.int64)
        if members.size < min_parent:
            continue
        split = _balanced_graph_split(
            members,
            spatial["affinity"],
            areas,
            min_size=min_child,
        )
        if split is None:
            continue
        left, right, _affinity_cut_ratio = split
        evidence = _spatial_split_evidence(members, left, right, spatial)
        if (
            float(evidence["cut_ratio"]) > float(max_cut_ratio)
            or float(evidence["compactness_gain"]) < float(min_compactness_gain)
            or float(evidence["confidence"]) < float(min_confidence)
        ):
            continue
        left_id, right_id = int(next_child), int(next_child + 1)
        next_child += 2
        retained_parents[int(parent_id)] = members.copy()
        subclusters[left_id] = np.asarray(left, dtype=np.int64)
        subclusters[right_id] = np.asarray(right, dtype=np.int64)
        parent_children[int(parent_id)] = (left_id, right_id)
        child_parent[left_id] = int(parent_id)
        child_parent[right_id] = int(parent_id)
        parent_evidence[int(parent_id)] = evidence
    return retained_parents, subclusters, parent_children, child_parent, parent_evidence


def _cosine_affinity_components(
    affinity: np.ndarray,
    *,
    cosine_threshold: float,
    min_size: int,
) -> list[np.ndarray] | None:
    """Partition rows by strong cosine affinity and merge tiny fragments."""
    affinity = np.asarray(affinity, dtype=np.float64)
    if affinity.ndim != 2 or affinity.shape[0] < 2 or affinity.shape[1] == 0:
        return None
    norms = np.linalg.norm(affinity, axis=1)
    similarity = affinity @ affinity.T
    similarity /= np.maximum(norms[:, None] * norms[None, :], 1.0e-12)

    parents = _union_find_parents(affinity.shape[0])
    rows, cols = np.where(np.triu(similarity >= float(cosine_threshold), 1))
    for left, right in zip(rows, cols):
        parents[_find(parents, int(left))] = _find(parents, int(right))

    by_root: dict[int, list[int]] = {}
    for index in range(affinity.shape[0]):
        by_root.setdefault(_find(parents, index), []).append(index)
    groups = [np.asarray(values, dtype=np.int64) for values in by_root.values()]
    if len(groups) <= 1:
        return None

    minimum = max(2, int(min_size))
    large = [group for group in groups if group.size >= minimum]
    small = [group for group in groups if group.size < minimum]
    if len(large) < 2:
        return None
    large.sort(key=lambda group: int(group[0]))
    small.sort(key=lambda group: (int(group.size), int(group[0])))
    for fragment in small:
        scores = np.asarray(
            [float(np.mean(similarity[np.ix_(fragment, group)])) for group in large],
            dtype=np.float64,
        )
        best = int(np.argmax(scores))
        if float(scores[best]) <= 0.0:
            return None
        large[best] = np.sort(np.concatenate([large[best], fragment])).astype(np.int64)

    large.sort(key=lambda group: int(group[0]))
    return large


def _single_component_soft_affinity_split(
    plc,
    n: int,
    n_soft: int,
    *,
    max_fanout: int,
    min_size: int,
    cosine_threshold: float,
) -> list[np.ndarray] | None:
    """Infer hard subgroups from shared low-fanout soft-macro affinity."""
    if n < 2 or n_soft <= 0:
        return None
    cache = _build_wl_cache(plc)
    ref_idx = cache["ref_idx"]
    net_starts = cache["net_starts"]
    net_lengths = cache["net_lengths"]
    hard_by_ref = {int(ref): index for index, ref in enumerate(plc.hard_macro_indices[:n])}
    soft_by_ref = {int(ref): index for index, ref in enumerate(plc.soft_macro_indices[:n_soft])}
    affinity = np.zeros((int(n), int(n_soft)), dtype=np.float64)
    for net_index, start_raw in enumerate(net_starts):
        length = int(net_lengths[net_index])
        if length < 2 or length > int(max_fanout):
            continue
        start = int(start_raw)
        refs = {int(ref) for ref in ref_idx[start : start + length]}
        hard = [hard_by_ref[ref] for ref in refs if ref in hard_by_ref]
        soft = [soft_by_ref[ref] for ref in refs if ref in soft_by_ref]
        if not hard or not soft:
            continue
        for hard_index in hard:
            affinity[hard_index, soft] += 1.0
    return _cosine_affinity_components(
        affinity,
        cosine_threshold=cosine_threshold,
        min_size=min_size,
    )


def _cluster_partition_areas(n: int, hard_sizes) -> np.ndarray:
    if hard_sizes is None:
        return np.ones(n, dtype=np.float64)
    sizes = np.asarray(hard_sizes, dtype=np.float64)
    if sizes.ndim != 2 or sizes.shape[0] < n or sizes.shape[1] < 2:
        return np.ones(n, dtype=np.float64)
    area = sizes[:n, 0] * sizes[:n, 1]
    area = np.asarray(area, dtype=np.float64)
    area[~np.isfinite(area) | (area <= 0.0)] = 1.0
    return area


def _edge_lookup(edge_weight: dict[tuple[int, int], float], a: int, b: int) -> float:
    if a > b:
        a, b = b, a
    return float(edge_weight.get((int(a), int(b)), 0.0))


def _recursive_bisect_component(
    members: np.ndarray,
    edge_weight: dict[tuple[int, int], float],
    areas: np.ndarray,
    *,
    max_size: int,
    min_size: int,
    max_cut_ratio: float,
) -> list[np.ndarray]:
    members = np.asarray(members, dtype=np.int64)
    if members.size <= max_size:
        return [members]
    split = _balanced_graph_split(members, edge_weight, areas, min_size=min_size)
    if split is None:
        return [members]
    left, right, cut_ratio = split
    if cut_ratio > max_cut_ratio:
        return [members]
    out: list[np.ndarray] = []
    out.extend(
        _recursive_bisect_component(
            left,
            edge_weight,
            areas,
            max_size=max_size,
            min_size=min_size,
            max_cut_ratio=max_cut_ratio,
        )
    )
    out.extend(
        _recursive_bisect_component(
            right,
            edge_weight,
            areas,
            max_size=max_size,
            min_size=min_size,
            max_cut_ratio=max_cut_ratio,
        )
    )
    return out


def _balanced_graph_split(
    members: np.ndarray,
    edge_weight: dict[tuple[int, int], float],
    areas: np.ndarray,
    *,
    min_size: int,
):
    member_set = {int(x) for x in members}
    degree = {}
    total_weight = 0.0
    for i_raw in members:
        i = int(i_raw)
        for j_raw in members:
            j = int(j_raw)
            if j <= i:
                continue
            w = _edge_lookup(edge_weight, i, j)
            if w <= 0.0:
                continue
            degree[i] = degree.get(i, 0.0) + w
            degree[j] = degree.get(j, 0.0) + w
            total_weight += w
    if total_weight <= 0.0 or members.size < 2 * min_size:
        return None

    total_area = float(np.sum(areas[members]))
    target = 0.5 * total_area
    seed = int(max(members, key=lambda x: (degree.get(int(x), 0.0), areas[int(x)], -int(x))))
    left = {seed}
    right = set(member_set)
    right.remove(seed)
    left_area = float(areas[seed])

    while len(left) < members.size - min_size and left_area < target:
        best = None
        for node in sorted(right):
            to_left = sum(_edge_lookup(edge_weight, node, other) for other in left)
            to_right = sum(
                _edge_lookup(edge_weight, node, other) for other in right if other != node
            )
            balance_penalty = abs((left_area + float(areas[node])) - target) / max(target, 1.0)
            score = to_left - 0.35 * to_right - 0.05 * balance_penalty
            row = (score, degree.get(node, 0.0), -balance_penalty, -node, node)
            if best is None or row > best:
                best = row
        if best is None:
            break
        node = int(best[-1])
        left.add(node)
        right.remove(node)
        left_area += float(areas[node])

    if len(left) < min_size or len(right) < min_size:
        return None

    left_arr = np.array(sorted(left), dtype=np.int64)
    right_arr = np.array(sorted(right), dtype=np.int64)
    cut = 0.0
    for a in left_arr:
        for b in right_arr:
            cut += _edge_lookup(edge_weight, int(a), int(b))
    cut_ratio = float(cut / max(total_weight, 1e-12))
    return left_arr, right_arr, cut_ratio


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


def derive_cluster_softs(
    plc,
    n: int,
    n_soft: int,
    labels: np.ndarray,
    max_fanout: int = 8,
    bridge_ratio: float = 0.6,
) -> dict[int, np.ndarray]:
    """Backward-compatible alias for legacy verification utilities."""
    owned, _bridge = derive_soft_cluster_roles(
        plc,
        n=n,
        n_soft=n_soft,
        labels=labels,
        max_fanout=max_fanout,
        bridge_ratio=bridge_ratio,
    )
    return owned


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
    cluster_margins=None,
) -> np.ndarray:
    """Per-macro CENTER-feasible region box [n,4] = (xlo, ylo, xhi, yhi).

    A macro's center must stay within its box to keep its whole footprint inside
    the cluster region (the box is pre-inset by the macro's half-extents, so a
    plain `xlo <= cx <= xhi` test region-locks it). Clustered macros share a box
    sized to give the cluster breathing room for congestion relief:
    `region_area = member_area / target_density`, at the cluster's current aspect
    ratio, never smaller than the current member footprint (so macros aren't
    trapped), centered on the cluster centroid, clipped to canvas by shifting.
    `margin>0` uses the simpler footprint+margin sizing instead. A
    `cluster_margins` mapping overrides that margin per cluster, which lets a
    deeper hierarchy layer grant room from graph/field pressure without
    changing the boxes of unrelated children. Singletons get a local window
    (`singleton_window`; 0 => pinned at their current spot).
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
        local_margin = (
            float(cluster_margins.get(int(cid), margin))
            if cluster_margins is not None
            else float(margin)
        )
        if local_margin > 0.0:
            rw, rh = bw0 + 2.0 * local_margin * big, bh0 + 2.0 * local_margin * big
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
    cluster_margins=None,
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
    local_margin = (
        float(cluster_margins.get(int(cid), margin))
        if cluster_margins is not None and cid is not None
        else float(margin)
    )
    if local_margin > 0.0:
        rw, rh = bw0 + 2.0 * local_margin * big, bh0 + 2.0 * local_margin * big
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
    cluster_margins=None,
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
            cluster_margins=cluster_margins,
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
                cluster_margins=cluster_margins,
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
