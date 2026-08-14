"""Adjacent-cluster ownership transfers followed by intra-cluster repair."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from placer.local_search.fields import weighted_congestion_field
from placer.local_search.subcluster_relocation import _hard_group_is_legal
from placer.scoring.exact import exact_proxy_components
from placer.scoring.incremental import IncrementalScorer


def _cluster_parent(graph, cluster_id: int) -> int:
    values = {
        graph.macros[index].parent_id
        for index in graph.clusters[int(cluster_id)].hard_members
        if graph.macros[index].parent_id >= 0
    }
    return int(next(iter(values))) if len(values) == 1 else -1


def _cluster_subcluster(graph, cluster_id: int) -> int:
    values = {
        graph.macros[index].subcluster_id
        for index in graph.clusters[int(cluster_id)].hard_members
        if graph.macros[index].subcluster_id >= 0
    }
    return int(next(iter(values))) if len(values) == 1 else -1


def _snapshot_hierarchy(hierarchy) -> dict:
    return {
        "labels": hierarchy.labels.copy(),
        "clusters": {key: value.copy() for key, value in hierarchy.clusters.items()},
        "cluster_softs": {key: value.copy() for key, value in hierarchy.cluster_softs.items()},
        "subcluster_labels": hierarchy.subcluster_labels.copy(),
        "subclusters": {key: value.copy() for key, value in hierarchy.subclusters.items()},
        "edges": list(hierarchy.edges),
        "graph_roles": {
            index: (
                node.cluster_id,
                node.subcluster_id,
                node.parent_id,
                node.bridge_clusters,
            )
            for index, node in hierarchy.location_graph.macros.items()
        },
    }


def _restore_hierarchy(hierarchy, snapshot: dict, hard_pos, soft_pos, changed_indices=()) -> None:
    hierarchy.labels[:] = snapshot["labels"]
    hierarchy.clusters.clear()
    hierarchy.clusters.update({key: value.copy() for key, value in snapshot["clusters"].items()})
    hierarchy.cluster_softs.clear()
    hierarchy.cluster_softs.update(
        {key: value.copy() for key, value in snapshot["cluster_softs"].items()}
    )
    hierarchy.subcluster_labels[:] = snapshot["subcluster_labels"]
    hierarchy.subclusters.clear()
    hierarchy.subclusters.update(
        {key: value.copy() for key, value in snapshot["subclusters"].items()}
    )
    graph = hierarchy.location_graph
    changed = tuple(int(index) for index in changed_indices)
    assignments = {
        index: snapshot["graph_roles"][index][0]
        for index in changed
        if snapshot["graph_roles"][index][0] >= 0
    }
    subclusters = {index: snapshot["graph_roles"][index][1] for index in changed}
    parents = {index: snapshot["graph_roles"][index][2] for index in changed}
    if assignments:
        graph.reassign_macros(
            assignments,
            subclusters=subclusters,
            parents=parents,
            rebuild_edges=False,
        )
    for index in changed:
        values = snapshot["graph_roles"][index]
        node = graph.macros[index]
        node.cluster_id = int(values[0])
        node.subcluster_id = int(values[1])
        node.parent_id = int(values[2])
        node.bridge_clusters = tuple(values[3])
    hierarchy.edges[:] = snapshot["edges"]
    graph.synchronize(hard_pos, soft_pos)


def _transfer_ownership(hierarchy, assignments: dict[int, int], n_hard: int) -> None:
    """Apply active-leaf ownership changes to hierarchy arrays and graph nodes."""
    graph = hierarchy.location_graph
    destination_subclusters = {}
    destination_parents = {}
    for index, destination in assignments.items():
        destination_subclusters[int(index)] = _cluster_subcluster(graph, int(destination))
        destination_parents[int(index)] = _cluster_parent(graph, int(destination))

    for index_raw, destination_raw in assignments.items():
        index, destination = int(index_raw), int(destination_raw)
        old_cluster = int(graph.macros[index].cluster_id)
        if old_cluster == destination:
            continue
        if index < n_hard:
            hierarchy.clusters[old_cluster] = hierarchy.clusters[old_cluster][
                hierarchy.clusters[old_cluster] != index
            ]
            hierarchy.clusters[destination] = np.unique(
                np.append(hierarchy.clusters[destination], index)
            ).astype(np.int64)
            hierarchy.labels[index] = destination
            old_child = int(hierarchy.subcluster_labels[index])
            new_child = int(destination_subclusters[index])
            if old_child >= 0 and old_child in hierarchy.subclusters:
                hierarchy.subclusters[old_child] = hierarchy.subclusters[old_child][
                    hierarchy.subclusters[old_child] != index
                ]
            if new_child >= 0 and new_child in hierarchy.subclusters:
                hierarchy.subclusters[new_child] = np.unique(
                    np.append(hierarchy.subclusters[new_child], index)
                ).astype(np.int64)
            hierarchy.subcluster_labels[index] = new_child
        else:
            for cluster_id, members in list(hierarchy.cluster_softs.items()):
                hierarchy.cluster_softs[cluster_id] = members[members != index]
            hierarchy.cluster_softs[destination] = np.unique(
                np.append(
                    hierarchy.cluster_softs.get(destination, np.zeros(0, dtype=np.int64)), index
                )
            ).astype(np.int64)

    graph.reassign_macros(
        assignments,
        subclusters=destination_subclusters,
        parents=destination_parents,
    )


def _field_values(scorer, positions, *, hard: bool) -> np.ndarray:
    nr, nc = int(scorer.grid_row), int(scorer.grid_col)
    field = np.asarray(weighted_congestion_field(scorer, nr, nc), dtype=np.float64)
    density = np.asarray(scorer.grid_occupied, dtype=np.float64).reshape(nr, nc)
    density /= max(float(scorer.dens_grid_area), 1.0e-12)
    combined = field + density
    cell_w, cell_h = float(scorer.plc.width) / nc, float(scorer.plc.height) / nr
    ci = np.clip((positions[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((positions[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    return combined[ri, ci]


def _cold_targets(cluster, graph, field, nr, nc, cw, ch, *, count: int) -> list[np.ndarray]:
    x0, y0, x1, y1 = map(float, cluster.bbox)
    if x1 <= x0 or y1 <= y0:
        return []
    cell_w, cell_h = cw / nc, ch / nr
    c0 = max(0, int(np.floor(x0 / cell_w)))
    c1 = min(nc - 1, int(np.floor(x1 / cell_w)))
    r0 = max(0, int(np.floor(y0 / cell_h)))
    r1 = min(nr - 1, int(np.floor(y1 / cell_h)))
    rows = []
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            rows.append((float(field[row, col]), row, col))
    rows.sort()
    return [
        np.asarray([(col + 0.5) * cell_w, (row + 0.5) * cell_h], dtype=np.float64)
        for _value, row, col in rows[: max(0, int(count))]
    ]


def _candidate_layout(proposal, hard_pos, soft_pos, n_hard):
    trial_hard, trial_soft = hard_pos.copy(), soft_pos.copy()
    for index, target in proposal["moves"]:
        if index < n_hard:
            trial_hard[index] = target
        else:
            trial_soft[index - n_hard] = target
    return trial_hard, trial_soft


def _legal_proposal(proposal, hard_pos, soft_pos, hw, hh, soft_hw, soft_hh, cw, ch, n_hard):
    hard_indices = np.asarray(
        [index for index, _target in proposal["moves"] if index < n_hard], dtype=np.int64
    )
    hard_targets = np.asarray(
        [target for index, target in proposal["moves"] if index < n_hard], dtype=np.float64
    ).reshape((-1, 2))
    if hard_indices.size and not _hard_group_is_legal(hard_pos, hard_indices, hard_targets, hw, hh):
        return False
    for index, target in proposal["moves"]:
        if index < n_hard:
            if not (
                hw[index] <= target[0] <= cw - hw[index]
                and hh[index] <= target[1] <= ch - hh[index]
            ):
                return False
        else:
            local = index - n_hard
            if not (
                soft_hw[local] <= target[0] <= cw - soft_hw[local]
                and soft_hh[local] <= target[1] <= ch - soft_hh[local]
            ):
                return False
    return True


def _generate_intercluster_proposals(
    graph,
    hard_pos,
    soft_pos,
    hard_heat,
    soft_heat,
    movable_h,
    movable_s,
    field,
    nr,
    nc,
    cw,
    ch,
    *,
    top_per_role: int,
    targets_per_cluster: int,
) -> list[dict]:
    n_hard = hard_pos.shape[0]
    proposals = []
    pairs = [
        (left, right)
        for left, cluster in sorted(graph.clusters.items())
        for right in sorted(graph.cluster_neighbors(left))
        if left < right and _cluster_parent(graph, left) == _cluster_parent(graph, right)
    ]
    for left, right in pairs:
        left_node, right_node = graph.clusters[left], graph.clusters[right]
        left_frontier = {row["index"]: row for row in graph.frontier_records(left, right)}
        right_frontier = {row["index"]: row for row in graph.frontier_records(right, left)}
        left_h = sorted(
            (index for index in left_node.hard_members if movable_h[index]),
            key=lambda index: (-left_frontier[index]["score"], index),
        )[:top_per_role]
        right_h = sorted(
            (index for index in right_node.hard_members if movable_h[index]),
            key=lambda index: (-right_frontier[index]["score"], index),
        )[:top_per_role]
        left_s = sorted(
            (index for index in left_node.soft_members if movable_s[index - n_hard]),
            key=lambda index: (-left_frontier[index]["score"], index),
        )[:top_per_role]
        right_s = sorted(
            (index for index in right_node.soft_members if movable_s[index - n_hard]),
            key=lambda index: (-right_frontier[index]["score"], index),
        )[:top_per_role]
        for source_rows, target_rows, kind in (
            (left_h, right_h, "hard_hard_swap"),
            (left_s, right_s, "soft_soft_swap"),
            (left_h, right_s, "hard_soft_swap"),
            (right_h, left_s, "hard_soft_swap"),
        ):
            for source in source_rows:
                for target in target_rows:
                    if (
                        kind == "hard_soft_swap"
                        and source < n_hard
                        and len(graph.clusters[graph.cluster_of(source)].hard_members) <= 1
                    ):
                        continue
                    source_xy = graph.macros[source].position.copy()
                    target_xy = graph.macros[target].position.copy()
                    proposals.append(
                        {
                            "kind": kind,
                            "moves": ((source, target_xy), (target, source_xy)),
                            "assignments": {
                                source: graph.cluster_of(target),
                                target: graph.cluster_of(source),
                            },
                            "rank": -float(
                                (
                                    left_frontier
                                    if graph.cluster_of(source) == left
                                    else right_frontier
                                )[source]["score"]
                                + (
                                    right_frontier
                                    if graph.cluster_of(target) == right
                                    else left_frontier
                                )[target]["score"]
                            ),
                        }
                    )
        for source_cluster, target_cluster, hard_rows, soft_rows in (
            (left, right, left_h, left_s),
            (right, left, right_h, right_s),
        ):
            targets = _cold_targets(
                graph.clusters[target_cluster],
                graph,
                field,
                nr,
                nc,
                cw,
                ch,
                count=targets_per_cluster,
            )
            for source in [*hard_rows, *soft_rows]:
                if source < n_hard and len(graph.clusters[source_cluster].hard_members) <= 1:
                    continue
                source_heat = hard_heat[source] if source < n_hard else soft_heat[source - n_hard]
                for target in targets:
                    frontier = left_frontier if source_cluster == left else right_frontier
                    if frontier[source]["capacity_ratio"] < 1.05:
                        continue
                    proposals.append(
                        {
                            "kind": "hard_relocation" if source < n_hard else "soft_relocation",
                            "moves": ((source, target),),
                            "assignments": {source: target_cluster},
                            "rank": -float(frontier[source]["score"] + source_heat),
                        }
                    )
            frontier = left_frontier if source_cluster == left else right_frontier
            seed_rows = sorted([*hard_rows, *soft_rows], key=lambda i: (-frontier[i]["score"], i))
            for seed in seed_rows[:2]:
                bundle = graph.connected_frontier_bundle(seed, target_cluster, max_members=4)
                if len(bundle) < 2:
                    continue
                hard_count = sum(index < n_hard for index in bundle)
                if hard_count >= len(graph.clusters[source_cluster].hard_members):
                    continue
                area = sum(float(np.prod(graph.macros[index].size)) for index in bundle)
                if area * 1.15 > graph.clusters[target_cluster].metrics.get(
                    "available_capacity", 0.0
                ):
                    continue
                bundle_center = np.mean([graph.macros[index].position for index in bundle], axis=0)
                for target in targets[:1]:
                    moves = tuple(
                        (index, target + graph.macros[index].position - bundle_center)
                        for index in bundle
                    )
                    proposals.append(
                        {
                            "kind": "connected_frontier_relocation",
                            "moves": moves,
                            "assignments": {index: target_cluster for index in bundle},
                            "rank": -float(sum(frontier[index]["score"] for index in bundle)),
                        }
                    )
    proposals.sort(key=lambda row: (row["rank"], row["kind"], tuple(row["assignments"])))
    return proposals


def _generate_intracluster_proposals(
    graph,
    hard_pos,
    soft_pos,
    hard_heat,
    soft_heat,
    movable_h,
    movable_s,
    field,
    nr,
    nc,
    cw,
    ch,
):
    n_hard = hard_pos.shape[0]
    proposals = []
    for cluster_id, cluster in sorted(graph.clusters.items()):
        hard = sorted(
            (index for index in cluster.hard_members if movable_h[index]),
            key=lambda index: (-hard_heat[index], index),
        )[:4]
        soft = sorted(
            (index for index in cluster.soft_members if movable_s[index - n_hard]),
            key=lambda index: (-soft_heat[index - n_hard], index),
        )[:4]
        for rows, kind in ((hard, "hard_hard_swap"), (soft, "soft_soft_swap")):
            for left_pos, left in enumerate(rows):
                for right in rows[left_pos + 1 :]:
                    proposals.append(
                        {
                            "kind": f"intra_{kind}",
                            "moves": (
                                (left, graph.macros[right].position.copy()),
                                (right, graph.macros[left].position.copy()),
                            ),
                            "assignments": {},
                            "rank": -max(
                                hard_heat[left] if left < n_hard else soft_heat[left - n_hard],
                                hard_heat[right] if right < n_hard else soft_heat[right - n_hard],
                            ),
                        }
                    )
        targets = _cold_targets(cluster, graph, field, nr, nc, cw, ch, count=3)
        for source in [*hard[:2], *soft[:2]]:
            source_heat = hard_heat[source] if source < n_hard else soft_heat[source - n_hard]
            for target in targets:
                proposals.append(
                    {
                        "kind": (
                            "intra_hard_relocation" if source < n_hard else "intra_soft_relocation"
                        ),
                        "moves": ((source, target),),
                        "assignments": {},
                        "rank": -float(source_heat),
                    }
                )
    proposals.sort(key=lambda row: (row["rank"], row["kind"]))
    return proposals


def run_adjacent_cluster_transfer(
    hard_pos,
    soft_pos,
    hw,
    hh,
    soft_hw,
    soft_hh,
    cw,
    ch,
    movable_h,
    movable_s,
    hierarchy,
    plc,
    benchmark,
    initial_score: float,
    *,
    candidate_allowed: Callable[[np.ndarray, np.ndarray], bool] | None,
    deadline: float | None,
    min_proxy_gain: float,
    min_density_congestion_gain: float,
    max_inter_scored: int,
    max_intra_scored: int,
    max_inter_accepts: int,
) -> tuple[np.ndarray, np.ndarray, float, IncrementalScorer, dict]:
    """Transfer ownership across adjacent leaves, then repair inside new leaves."""
    hard_pos = np.asarray(hard_pos, dtype=np.float64)
    soft_pos = np.asarray(soft_pos, dtype=np.float64)
    graph = hierarchy.location_graph
    graph.synchronize(hard_pos, soft_pos)
    scorer = IncrementalScorer(plc, benchmark, np.vstack([hard_pos, soft_pos]))
    stats = {
        "inter_candidates": 0,
        "inter_legal": 0,
        "inter_scored": 0,
        "inter_accepts": 0,
        "intra_candidates": 0,
        "intra_legal": 0,
        "intra_scored": 0,
        "intra_accepts": 0,
        "hierarchy_rejects": 0,
        "component_rejects": 0,
        "attempts": 0,
        "accepted_kind": "none",
        "proxy_gain": 0.0,
        "density_congestion_gain": 0.0,
        "best_candidate_proxy_gain": float("-inf"),
        "best_candidate_density_congestion_gain": float("-inf"),
    }

    def run_phase(intercluster: bool, max_scored: int):
        nonlocal hard_pos, soft_pos, scorer, initial_score
        if deadline is not None and time.monotonic() >= deadline:
            return False
        hard_heat = _field_values(scorer, hard_pos, hard=True)
        soft_heat = _field_values(scorer, soft_pos, hard=False)
        nr, nc = int(scorer.grid_row), int(scorer.grid_col)
        field = np.asarray(weighted_congestion_field(scorer, nr, nc), dtype=np.float64)
        density = np.asarray(scorer.grid_occupied, dtype=np.float64).reshape(nr, nc)
        field = field + density / max(float(scorer.dens_grid_area), 1.0e-12)
        graph.analyze_state(field, cw, ch)
        proposals = (
            _generate_intercluster_proposals(
                graph,
                hard_pos,
                soft_pos,
                hard_heat,
                soft_heat,
                np.asarray(movable_h, dtype=bool),
                np.asarray(movable_s, dtype=bool),
                field,
                nr,
                nc,
                cw,
                ch,
                top_per_role=2,
                targets_per_cluster=3,
            )
            if intercluster
            else _generate_intracluster_proposals(
                graph,
                hard_pos,
                soft_pos,
                hard_heat,
                soft_heat,
                np.asarray(movable_h, dtype=bool),
                np.asarray(movable_s, dtype=bool),
                field,
                nr,
                nc,
                cw,
                ch,
            )
        )
        prefix = "inter" if intercluster else "intra"
        stats[f"{prefix}_candidates"] += len(proposals)
        baseline_full = np.vstack([hard_pos, soft_pos])
        baseline = exact_proxy_components(baseline_full, benchmark, plc)
        best = None
        phase_attempts = 0
        for proposal in proposals:
            if stats[f"{prefix}_scored"] >= max(0, int(max_scored)):
                break
            if phase_attempts >= max(1, 2 * int(max_scored)):
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            stats["attempts"] += 1
            phase_attempts += 1
            if not _legal_proposal(
                proposal, hard_pos, soft_pos, hw, hh, soft_hw, soft_hh, cw, ch, hard_pos.shape[0]
            ):
                continue
            stats[f"{prefix}_legal"] += 1
            trial_hard, trial_soft = _candidate_layout(
                proposal, hard_pos, soft_pos, hard_pos.shape[0]
            )
            snapshot = _snapshot_hierarchy(hierarchy)
            if proposal["assignments"]:
                _transfer_ownership(hierarchy, proposal["assignments"], hard_pos.shape[0])
            graph.synchronize(trial_hard, trial_soft)
            allowed = candidate_allowed is None or bool(candidate_allowed(trial_hard, trial_soft))
            if not allowed:
                stats["hierarchy_rejects"] += 1
                _restore_hierarchy(hierarchy, snapshot, hard_pos, soft_pos, proposal["assignments"])
                continue
            components = exact_proxy_components(np.vstack([trial_hard, trial_soft]), benchmark, plc)
            stats[f"{prefix}_scored"] += 1
            dc_gain = 0.5 * (
                baseline["density"]
                + baseline["congestion"]
                - components["density"]
                - components["congestion"]
            )
            proxy_gain = baseline["proxy"] - components["proxy"]
            stats["best_candidate_proxy_gain"] = max(
                float(stats["best_candidate_proxy_gain"]), float(proxy_gain)
            )
            stats["best_candidate_density_congestion_gain"] = max(
                float(stats["best_candidate_density_congestion_gain"]), float(dc_gain)
            )
            _restore_hierarchy(hierarchy, snapshot, hard_pos, soft_pos, proposal["assignments"])
            if proxy_gain < float(min_proxy_gain) or dc_gain < float(min_density_congestion_gain):
                stats["component_rejects"] += 1
                continue
            key = (-proxy_gain, -dc_gain, proposal["kind"], tuple(proposal["assignments"]))
            if best is None or key < best[0]:
                best = (key, proposal, trial_hard, trial_soft, components, proxy_gain, dc_gain)
        exact_proxy_components(baseline_full, benchmark, plc)
        if best is None:
            graph.synchronize(hard_pos, soft_pos)
            scorer = IncrementalScorer(plc, benchmark, baseline_full)
            return False
        _key, proposal, hard_pos, soft_pos, components, proxy_gain, dc_gain = best
        if proposal["assignments"]:
            _transfer_ownership(hierarchy, proposal["assignments"], hard_pos.shape[0])
        graph.synchronize(hard_pos, soft_pos)
        graph.analyze_state(field, cw, ch)
        graph.record_transfer(proposal, proxy_gain, dc_gain)
        graph.checkpoint("accepted_adjacent_transfer", proxy=components)
        scorer = IncrementalScorer(plc, benchmark, np.vstack([hard_pos, soft_pos]))
        initial_score = float(components["proxy"])
        stats[f"{prefix}_accepts"] += 1
        stats["accepted_kind"] = str(proposal["kind"])
        stats["proxy_gain"] += float(proxy_gain)
        stats["density_congestion_gain"] += float(dc_gain)
        return True

    transferred = False
    for _round in range(max(0, int(max_inter_accepts))):
        if not run_phase(True, max_inter_scored):
            break
        transferred = True
    if transferred:
        run_phase(False, max_intra_scored)
    run_adjacent_cluster_transfer.last_stats = stats
    return hard_pos, soft_pos, float(initial_score), scorer, stats


run_adjacent_cluster_transfer.last_stats = {}
