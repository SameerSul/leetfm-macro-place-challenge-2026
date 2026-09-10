"""Bounded joint rearrangement of immutable leaves using exact tile attribution."""

from itertools import permutations
import time

import numpy as np
import torch

from placer.local_search.fields import cold_connected_component_target_pool
from placer.local_search.subcluster_relocation import _hard_group_is_legal, _hard_state_is_legal
from placer.plc.placement import clamp_in_bounds
from placer.routing.apply import (
    _apply_macro_routing_subset, _apply_net_routing_struct, _build_net_routing_struct,
)
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer


def _tail_view(scorer):
    """Directional tail weights, including the transpose of evaluator smoothing."""
    nr, nc = scorer.grid_row, scorer.grid_col
    baseline = scorer._swap_tail_baseline()
    count = max(1, int(2 * nr * nc * 0.05))
    weights = np.zeros(2 * nr * nc)
    weights[baseline["congestion_order"][:count]] = 0.5 / count
    vertical, horizontal = weights.reshape(2, nr, nc)

    def unsmooth(tail, axis):
        # A source distributes to its clipped neighborhood, so its derivative
        # is the neighborhood sum divided by that source's neighborhood size.
        a = np.moveaxis(tail, axis, 0)
        indices = np.arange(len(a))
        lo = np.maximum(0, indices - scorer.smooth_range)
        hi = np.minimum(len(a), indices + scorer.smooth_range + 1)
        cumulative = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
        return np.moveaxis((cumulative[hi] - cumulative[lo]) / (hi - lo)[:, None], 0, axis)

    density = np.zeros(nr * nc)
    occupied = baseline["density_nonzero"]
    if occupied:
        count = max(1, int(nr * nc * 0.1))
        if nr * nc < 10:
            count = occupied
        density[baseline["density_order"][:min(count, occupied)]] = (
            0.25 / (count * scorer.dens_grid_area)
        )
    return dict(
        h=horizontal.ravel() / scorer.grid_h_routes,
        v=vertical.ravel() / scorer.grid_v_routes,
        route_h=unsmooth(horizontal, 0).ravel() / scorer.grid_h_routes,
        route_v=unsmooth(vertical, 1).ravel() / scorer.grid_v_routes,
        density=density,
        field=(baseline["congestion"].reshape(2, nr, nc).sum(axis=0)
               + (baseline["density"] / scorer.dens_grid_area).reshape(nr, nc)),
    )


def _unit_pressure(scorer, indices, positions, view):
    """Attribute complete incident routes and footprints without position/grid writes."""
    modules = np.asarray(scorer.hard_indices + scorer.soft_indices)[indices]
    nets = scorer._touched_nets_many(modules)
    horizontal = np.zeros(scorer.grid_row * scorer.grid_col)
    vertical = np.zeros_like(horizontal)
    struct = _build_net_routing_struct(scorer.plc, nets)
    _apply_net_routing_struct(scorer.plc, struct, 1.0, horizontal, vertical)
    pressure = float(horizontal @ view["route_h"] + vertical @ view["route_v"])
    horizontal.fill(0.0)
    vertical.fill(0.0)
    hard = indices[indices < scorer.n_hard]
    _apply_macro_routing_subset(
        scorer.plc, scorer._hard_slot_array(*hard), 1.0, vertical, horizontal,
    )
    pressure += float(horizontal @ view["h"] + vertical @ view["v"])
    for index, module in zip(indices, modules):
        cells, areas = scorer._macro_occ(int(module), *positions[index])
        pressure += float(areas @ view["density"][cells])
    return pressure


def _leaf_units(hierarchy, movable, n):
    """Keep bundles intact and never combine different leaf/child memberships."""
    graph = hierarchy.location_graph
    if graph is None:
        return {}
    owner = np.full(len(movable), -1, dtype=np.int64)
    child = np.full(len(movable), -1, dtype=np.int64)
    owner[:n] = hierarchy.labels
    child[:n] = hierarchy.subcluster_labels
    for cid, members in hierarchy.cluster_softs.items():
        owner[np.asarray(members, dtype=np.int64)] = cid
    for cid, members in hierarchy.subcluster_softs.items():
        child[np.asarray(members, dtype=np.int64)] = cid
    allowed = np.asarray(movable, dtype=bool).copy()
    for roles in (hierarchy.bridge_softs, hierarchy.subcluster_bridge_softs,
                  hierarchy.parent_bridge_softs):
        allowed[n + np.asarray(list(roles), dtype=np.int64)] = False

    bundles = {tuple(sorted(int(k) + n for k in bundle.members))
               for rows in (hierarchy.soft_bundles, hierarchy.soft_connectivity_bundles,
                            hierarchy.soft_bundle_evidence, hierarchy.active_soft_bundles,
                            hierarchy.soft_only_bundles)
               for bundle in rows if len(bundle.members) > 1}
    explicit = {tuple(sorted(int(k) + n for k in bundle.members))
                for bundle in hierarchy.active_soft_bundles if bundle.source == "path"}
    grouped = set().union(*map(set, bundles)) if bundles else set()
    units = [np.asarray([i]) for i in np.flatnonzero(allowed & (owner >= 0)) if i not in grouped]
    for bundle in sorted(bundles):
        members = np.asarray(bundle, dtype=np.int64)
        if (bundle in explicit and len(bundle) <= 32 and np.all(allowed[members])
                and len(set(owner[members])) == 1 and owner[members[0]] >= 0
                and len(set(child[members])) == 1
                and not any(set(bundle) & set(other) for other in bundles if other != bundle)):
            units.append(members)
    result = {}
    for members in units:
        key = (int(owner[members[0]]), int(child[members[0]]))
        result.setdefault(key, []).append(members)
    return {key: rows for key, rows in sorted(result.items()) if len(rows) >= 2}


def _patch_assignments(units, positions, regions, scorer, graph, field):
    """Enumerate small slot assignments; unary costs only order exact trials."""
    centers = np.array([positions[u].mean(axis=0) for u in units])
    nr, nc = field.shape
    cold = cold_connected_component_target_pool(
        field, cold_percentile=35.0, max_components=4, min_cells=1, size_weight=0.0,
    )["indices"]
    cells = np.column_stack(((cold % nc + 0.5) * scorer.grid_w,
                             (cold // nc + 0.5) * scorer.grid_h))
    members = np.concatenate(units)
    if cells.size:
        lo, hi = regions[members, :2].min(axis=0), regions[members, 2:].max(axis=0)
        cells = cells[np.all((cells >= lo) & (cells <= hi), axis=1)]
    if cells.size:
        order = np.argsort(np.sum((cells - centers.mean(axis=0)) ** 2, axis=1), kind="stable")
        cells = cells[order[:2]]
    slots = np.vstack([centers, cells.reshape(-1, 2)])
    targets, costs = {}, {}
    for i, unit in enumerate(units):
        for j, slot in enumerate(slots):
            xy = (positions[unit] + slot - centers[i]).astype(np.float32).astype(np.float64)
            if not np.all((xy >= regions[unit, :2] - 1e-7) & (xy <= regions[unit, 2:] + 1e-7)):
                continue
            cost = 0.0
            for index, target in zip(unit, xy):
                module = (scorer.hard_indices + scorer.soft_indices)[index]
                occupied, areas = scorer._macro_occ(module, *target)
                cost += float(areas @ field.ravel()[occupied]) / scorer.dens_grid_area
                for neighbor, weight in graph.neighbors(int(index)).items():
                    cost += float(weight) * np.abs(target - positions[neighbor]).sum() / scorer.wl_normalizer
            targets[i, j], costs[i, j] = xy, cost
    ranked = []
    # ponytail: at most 6P4=360 assignments; use a solver only for larger patches.
    for assignment in permutations(range(len(slots)), len(units)):
        if all(j == i for i, j in enumerate(assignment)):
            continue
        if all((i, j) in targets for i, j in enumerate(assignment)):
            ranked.append((sum(costs[i, j] for i, j in enumerate(assignment)), assignment))
    ranked.sort()
    for _, assignment in ranked:
        yield members, np.vstack([targets[i, j] for i, j in enumerate(assignment)])


def final_cluster_tile_relief(
    hard, soft, benchmark, plc, hierarchy, candidate_allowed, *,
    hard_region, soft_region, deadline=None,
):
    """Keep only fresh exact gains on the same float32 coordinates the API returns."""
    stats = dict(patches=0, candidates=0, scored=0, accepts=0, hierarchy_rejects=0,
                 full_exact_scored=0)
    if deadline is not None and time.monotonic() >= deadline:
        return hard, soft, None, None, stats
    baseline = clamp_in_bounds(torch.from_numpy(np.vstack([hard, soft]).astype(np.float32)),
                               benchmark).numpy()
    positions = baseline.astype(np.float64)
    n = len(hard)
    units_by_leaf = _leaf_units(hierarchy, ~benchmark.macro_fixed.numpy(), n)
    if not units_by_leaf or (deadline is not None and time.monotonic() >= deadline):
        return hard, soft, None, None, stats
    scorer = IncrementalScorer(plc, benchmark, baseline)
    before = float(_exact_proxy(torch.from_numpy(baseline), benchmark, plc))
    stats["full_exact_scored"] += 1
    score = before
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    hw, hh = sizes[:n, 0] / 2, sizes[:n, 1] / 2
    canvas = np.array([benchmark.canvas_width, benchmark.canvas_height])
    regions = np.vstack([hard_region, soft_region]).copy()
    regions[:, :2] = np.maximum(regions[:, :2], sizes / 2)
    regions[:, 2:] = np.minimum(regions[:, 2:], canvas - sizes / 2)
    graph = hierarchy.location_graph
    attempted = set()
    while stats["patches"] < 4 and stats["scored"] < 64:
        if deadline is not None and time.monotonic() >= deadline:
            break
        view = _tail_view(scorer)
        leaves = []
        for key, units in units_by_leaf.items():
            if key in attempted:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                break
            pressure = _unit_pressure(scorer, np.concatenate(units), positions, view)
            if pressure > 0:
                leaves.append((-pressure, key))
        if not leaves:
            break
        _, key = min(leaves)
        attempted.add(key)
        units = units_by_leaf[key]
        pressures = []
        for unit in units:
            if deadline is not None and time.monotonic() >= deadline:
                break
            pressures.append(_unit_pressure(scorer, unit, positions, view))
        if len(pressures) != len(units):
            break
        selected = [int(np.argmax(pressures))]
        remaining = set(range(len(units))) - set(selected)
        while remaining and len(selected) < 4:
            connected = set(np.concatenate([units[i] for i in selected]).tolist())
            neighbors = []
            for i in remaining:
                weight = sum(w for member in units[i] for node, w in graph.neighbors(int(member)).items()
                             if node in connected)
                if weight > 0 and sum(len(units[j]) for j in selected) + len(units[i]) <= 32:
                    neighbors.append((-weight, -pressures[i], i))
            if not neighbors:
                break
            i = min(neighbors)[2]
            selected.append(i)
            remaining.remove(i)
        if len(selected) < 2:
            continue
        stats["patches"] += 1
        patch_scores = 0
        best = None
        for members, targets in _patch_assignments(
            [units[i] for i in selected], positions, regions, scorer, graph, view["field"],
        ):
            if patch_scores >= 16 or (deadline is not None and time.monotonic() >= deadline):
                break
            stats["candidates"] += 1
            mask = members < n
            hidx, sidx = members[mask], members[~mask] - n
            hxy, sxy = targets[mask], targets[~mask]
            if hidx.size and (not _hard_group_is_legal(positions[:n], hidx, hxy, hw, hh)
                              or not _hard_state_is_legal(hxy, hw[hidx], hh[hidx])):
                continue
            trial_score = float(scorer.score_move_group(hidx, hxy, sidx, sxy))
            stats["scored"] += 1
            patch_scores += 1
            if trial_score >= (score if best is None else best[0]) - 1e-6:
                continue
            trial = positions.copy()
            trial[members] = targets
            if not candidate_allowed(trial[:n], trial[n:]):
                stats["hierarchy_rejects"] += 1
                continue
            best = (trial_score, hidx, hxy.copy(), sidx, sxy.copy())
        if best is not None:
            score, hidx, hxy, sidx, sxy = best
            scorer.commit_move_group(hidx, hxy, sidx, sxy)
            positions[hidx], positions[n + sidx] = hxy, sxy
            stats["accepts"] += 1
    if stats["accepts"]:
        candidate = clamp_in_bounds(torch.from_numpy(positions.astype(np.float32)), benchmark).numpy()
        IncrementalScorer(plc, benchmark, candidate)
        after = float(_exact_proxy(torch.from_numpy(candidate), benchmark, plc))
        stats["full_exact_scored"] += 1
        moved = np.flatnonzero(np.any(candidate[:n] != baseline[:n], axis=1))
        final_hard = candidate[:n].astype(np.float64)
        legal = (not moved.size or (
            _hard_group_is_legal(baseline[:n].astype(np.float64), moved, final_hard[moved], hw, hh)
            and _hard_state_is_legal(final_hard[moved], hw[moved], hh[moved])
        ))
        if legal and after < before - 1e-6 and candidate_allowed(candidate[:n], candidate[n:]):
            hierarchy.location_graph.synchronize(candidate[:n], candidate[n:])
            return candidate[:n].astype(np.float64), candidate[n:].astype(np.float64), before, after, stats
    stats["accepts"] = 0
    IncrementalScorer(plc, benchmark, baseline)
    return hard, soft, before, before, stats
