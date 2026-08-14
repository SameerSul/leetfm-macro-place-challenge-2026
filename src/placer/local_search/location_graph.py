"""Persistent location-aware macro and hierarchy graph."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from placer.scoring.wirelength import _build_wl_cache


@dataclass
class MacroLocationNode:
    """One placement macro with stable hierarchy and topology metadata."""

    index: int
    module_index: int
    name: str
    kind: str
    size: np.ndarray
    position: np.ndarray
    cluster_id: int = -1
    subcluster_id: int = -1
    parent_id: int = -1
    bridge_clusters: tuple[int, ...] = ()
    neighbors: dict[int, float] = field(default_factory=dict)
    heat: float = 0.0


@dataclass
class ClusterLocationNode:
    """One hierarchy leaf with live geometry and inter-leaf connectivity."""

    cluster_id: int
    hard_members: tuple[int, ...]
    soft_members: tuple[int, ...]
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    metrics: dict[str, float] = field(default_factory=dict)
    side_capacity: dict[str, float] = field(default_factory=dict)

    @property
    def members(self) -> tuple[int, ...]:
        return self.hard_members + self.soft_members


class LocationAwareGraph:
    """Persistent topology whose node geometry mirrors the committed placement.

    Placement and scoring continue to use dense NumPy arrays. This graph owns
    stable macro/cluster nodes and copies committed coordinates into them at
    synchronization points, making hierarchy and spatial queries available
    without rebuilding net topology.
    """

    def __init__(
        self,
        macros,
        clusters,
        active_edges,
        wl_cache,
        module_to_placement,
        max_fanout,
        n_hard,
    ):
        self.macros = macros
        self.clusters = clusters
        # This is the exact list owned by HierarchyModel.edges, not a copy.
        self.active_edges = active_edges
        # The immutable net topology remains canonical on the PLC; keep references only.
        self.wl_cache = wl_cache
        self.module_to_placement = module_to_placement
        self.max_fanout = max(2, int(max_fanout))
        self.n_hard = int(n_hard)
        self.revision = 0
        self.analysis_revision = -1
        self.routing_attribution: dict[str, Any] = {}
        self.transfer_history: deque[dict[str, Any]] = deque(maxlen=32)
        self._checkpoints: deque[dict[str, Any]] = deque(maxlen=8)

    @classmethod
    def build(
        cls,
        plc,
        hierarchy: Any,
        positions: np.ndarray,
        sizes: np.ndarray,
        *,
        max_fanout: int = 16,
    ) -> "LocationAwareGraph":
        """Build static topology and hierarchy metadata, then load positions."""
        positions = np.asarray(positions, dtype=np.float64)
        sizes = np.asarray(sizes, dtype=np.float64)
        n_hard = len(plc.hard_macro_indices)
        n_soft = len(plc.soft_macro_indices)
        count = n_hard + n_soft
        if positions.shape != (count, 2) or sizes.shape != (count, 2):
            raise ValueError("location graph positions and sizes must match all placement macros")

        soft_owner: dict[int, int] = {}
        for cluster_id, members in hierarchy.cluster_softs.items():
            for index in np.asarray(members, dtype=np.int64):
                if n_hard <= int(index) < count:
                    soft_owner[int(index) - n_hard] = int(cluster_id)

        macros: dict[int, MacroLocationNode] = {}
        module_to_placement: dict[int, int] = {}
        for index in range(count):
            hard = index < n_hard
            local = index if hard else index - n_hard
            module_index = int(
                plc.hard_macro_indices[local] if hard else plc.soft_macro_indices[local]
            )
            module_to_placement[module_index] = index
            try:
                name = str(plc.modules_w_pins[module_index].get_name())
            except Exception:
                name = f"{'hard' if hard else 'soft'}_{local}"
            cluster_id = int(hierarchy.labels[index]) if hard else soft_owner.get(local, -1)
            subcluster_id = int(hierarchy.subcluster_labels[index]) if hard else -1
            parent_id = int(hierarchy.parent_labels[index]) if hard else -1
            if not hard and cluster_id >= 0:
                hard_members = np.asarray(hierarchy.clusters.get(cluster_id, ()), dtype=np.int64)
                if hard_members.size:
                    child_values = np.unique(hierarchy.subcluster_labels[hard_members])
                    parent_values = np.unique(hierarchy.parent_labels[hard_members])
                    if child_values.size == 1:
                        subcluster_id = int(child_values[0])
                    if parent_values.size == 1:
                        parent_id = int(parent_values[0])
            bridge_clusters = (
                tuple(int(value) for value in hierarchy.bridge_softs.get(local, ()))
                if not hard
                else ()
            )
            macros[index] = MacroLocationNode(
                index=index,
                module_index=module_index,
                name=name,
                kind="hard" if hard else "soft",
                size=sizes[index].copy(),
                position=positions[index].copy(),
                cluster_id=cluster_id,
                subcluster_id=subcluster_id,
                parent_id=parent_id,
                bridge_clusters=bridge_clusters,
            )

        cache = _build_wl_cache(plc)
        for net_index, start_raw in enumerate(cache["net_starts"]):
            length = int(cache["net_lengths"][net_index])
            if length < 2 or length > max(2, int(max_fanout)):
                continue
            start = int(start_raw)
            endpoints = sorted(
                {
                    module_to_placement[int(ref)]
                    for ref in cache["ref_idx"][start : start + length]
                    if int(ref) in module_to_placement
                }
            )
            if len(endpoints) < 2:
                continue
            net_weight = max(0.0, float(cache["net_weights"][net_index]))
            weight = net_weight / max(len(endpoints) - 1, 1)
            for left_pos, left in enumerate(endpoints):
                for right in endpoints[left_pos + 1 :]:
                    macros[left].neighbors[right] = macros[left].neighbors.get(right, 0.0) + weight
                    macros[right].neighbors[left] = macros[right].neighbors.get(left, 0.0) + weight

        clusters = {}
        for cluster_id, hard_members_raw in hierarchy.clusters.items():
            hard_members = tuple(
                int(index) for index in np.asarray(hard_members_raw, dtype=np.int64)
            )
            soft_members = tuple(
                int(index)
                for index in np.asarray(
                    hierarchy.cluster_softs.get(int(cluster_id), ()), dtype=np.int64
                )
            )
            clusters[int(cluster_id)] = ClusterLocationNode(
                cluster_id=int(cluster_id),
                hard_members=hard_members,
                soft_members=soft_members,
            )

        graph = cls(
            macros,
            clusters,
            hierarchy.edges,
            cache,
            module_to_placement,
            max_fanout,
            n_hard,
        )
        graph.synchronize(positions[:n_hard], positions[n_hard:])
        return graph

    def synchronize(self, hard_positions: np.ndarray, soft_positions: np.ndarray) -> None:
        """Copy one committed placement into macro and cluster location nodes."""
        hard_positions = np.asarray(hard_positions, dtype=np.float64)
        soft_positions = np.asarray(soft_positions, dtype=np.float64)
        n_hard = sum(node.kind == "hard" for node in self.macros.values())
        if hard_positions.shape != (n_hard, 2):
            raise ValueError("hard position shape does not match location graph")
        if soft_positions.shape != (len(self.macros) - n_hard, 2):
            raise ValueError("soft position shape does not match location graph")
        for index in range(n_hard):
            self.macros[index].position[:] = hard_positions[index]
        for local in range(soft_positions.shape[0]):
            self.macros[n_hard + local].position[:] = soft_positions[local]
        self._refresh_clusters(self.clusters)
        self.revision += 1

    def update_macros(self, indices, positions) -> None:
        """Update accepted macro moves without rescanning unaffected clusters."""
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        positions = np.asarray(positions, dtype=np.float64).reshape((-1, 2))
        if positions.shape != (indices.size, 2) or np.unique(indices).size != indices.size:
            raise ValueError("location graph updates require unique indices and [N, 2] positions")
        affected = set()
        for index, position in zip(indices, positions):
            node = self.macros[int(index)]
            node.position[:] = position
            if node.cluster_id >= 0:
                affected.add(int(node.cluster_id))
        self._refresh_clusters(affected)
        self.revision += 1

    def reassign_macros(
        self,
        assignments: dict[int, int],
        *,
        subclusters: dict[int, int] | None = None,
        parents: dict[int, int] | None = None,
        rebuild_edges: bool = True,
    ) -> None:
        """Transfer macro ownership and rebuild affected cluster membership."""
        affected = set()
        for index_raw, cluster_raw in assignments.items():
            index, cluster_id = int(index_raw), int(cluster_raw)
            node = self.macros[index]
            if node.cluster_id >= 0:
                affected.add(int(node.cluster_id))
            if cluster_id not in self.clusters:
                raise KeyError(f"unknown destination cluster {cluster_id}")
            node.cluster_id = cluster_id
            node.bridge_clusters = ()
            if subclusters is not None and index in subclusters:
                node.subcluster_id = int(subclusters[index])
            if parents is not None and index in parents:
                node.parent_id = int(parents[index])
            affected.add(cluster_id)
        for cluster_id in affected:
            hard_members = tuple(
                index
                for index, node in sorted(self.macros.items())
                if node.kind == "hard" and node.cluster_id == cluster_id
            )
            soft_members = tuple(
                index
                for index, node in sorted(self.macros.items())
                if node.kind == "soft" and node.cluster_id == cluster_id
            )
            cluster = self.clusters[cluster_id]
            cluster.hard_members = hard_members
            cluster.soft_members = soft_members
        if rebuild_edges and any(self.macros[int(index)].kind == "hard" for index in assignments):
            self._rebuild_cluster_adjacency()
        self._refresh_clusters(affected)
        self.revision += 1

    def _rebuild_cluster_adjacency(self) -> None:
        """Rebuild the canonical active graph with hierarchy-model semantics.

        Each eligible net contributes its full weight once to every pair of
        distinct hard-macro owners. Soft ownership therefore cannot silently
        alter the hard hierarchy graph.
        """
        pair_weight: dict[tuple[int, int], float] = {}
        for net_index, start_raw in enumerate(self.wl_cache["net_starts"]):
            length = int(self.wl_cache["net_lengths"][net_index])
            if length < 2 or length > self.max_fanout:
                continue
            start = int(start_raw)
            hard_endpoints = {
                self.module_to_placement[int(ref)]
                for ref in self.wl_cache["ref_idx"][start : start + length]
                if int(ref) in self.module_to_placement
                and self.module_to_placement[int(ref)] < self.n_hard
            }
            weight = max(0.0, float(self.wl_cache["net_weights"][net_index]))
            owners = sorted(
                {
                    self.macros[index].cluster_id
                    for index in hard_endpoints
                    if self.macros[index].cluster_id >= 0
                }
            )
            for left_pos, left in enumerate(owners):
                for right in owners[left_pos + 1 :]:
                    pair = (int(left), int(right))
                    pair_weight[pair] = pair_weight.get(pair, 0.0) + float(weight)
        from placer.local_search.hierarchy_model import HierarchyEdge

        self.active_edges[:] = [
            HierarchyEdge(int(left), int(right), float(weight))
            for (left, right), weight in sorted(pair_weight.items())
        ]

    def cluster_neighbors(self, cluster_id: int) -> dict[int, float]:
        """Project the canonical active edge list around one hierarchy leaf."""
        cluster_id = int(cluster_id)
        neighbors = {}
        for edge in self.active_edges:
            if int(edge.src) == cluster_id:
                neighbors[int(edge.dst)] = float(edge.weight)
            elif int(edge.dst) == cluster_id:
                neighbors[int(edge.src)] = float(edge.weight)
        return neighbors

    def _refresh_clusters(self, cluster_ids) -> None:
        """Recompute live geometry for selected hierarchy leaves."""
        for cluster_id in cluster_ids:
            cluster = self.clusters[int(cluster_id)]
            members = cluster.members
            if not members:
                cluster.centroid[:] = 0.0
                cluster.bbox[:] = 0.0
                continue
            points = np.asarray([self.macros[index].position for index in members])
            half_sizes = 0.5 * np.asarray([self.macros[index].size for index in members])
            cluster.centroid[:] = np.mean(points, axis=0)
            lo = np.min(points - half_sizes, axis=0)
            hi = np.max(points + half_sizes, axis=0)
            cluster.bbox[:] = [lo[0], lo[1], hi[0], hi[1]]

    def cluster_of(self, macro_index: int) -> int:
        """Return the active leaf owner of one macro, or -1 when unowned."""
        return int(self.macros[int(macro_index)].cluster_id)

    def summary(self) -> dict[str, int]:
        """Return compact topology and hierarchy counts for diagnostics."""
        hard = sum(node.kind == "hard" for node in self.macros.values())
        macro_edges = sum(len(node.neighbors) for node in self.macros.values()) // 2
        cluster_edges = len(self.active_edges)
        return {
            "revision": int(self.revision),
            "macros": int(len(self.macros)),
            "hard_macros": int(hard),
            "soft_macros": int(len(self.macros) - hard),
            "owned_macros": int(sum(node.cluster_id >= 0 for node in self.macros.values())),
            "bridge_softs": int(sum(bool(node.bridge_clusters) for node in self.macros.values())),
            "macro_edges": int(macro_edges),
            "clusters": int(len(self.clusters)),
            "cluster_edges": int(cluster_edges),
        }

    def neighbors(self, macro_index: int) -> dict[int, float]:
        """Return a copy of one macro's weighted low-fanout adjacency."""
        return dict(self.macros[int(macro_index)].neighbors)

    def macros_in_box(
        self,
        rect: tuple[float, float, float, float] | np.ndarray,
        *,
        cluster_id: int | None = None,
    ) -> tuple[int, ...]:
        """Return macros whose centers lie in a rectangle, optionally by leaf."""
        x0, y0, x1, y1 = map(float, rect)
        return tuple(
            index
            for index, node in sorted(self.macros.items())
            if (cluster_id is None or node.cluster_id == int(cluster_id))
            and x0 <= float(node.position[0]) <= x1
            and y0 <= float(node.position[1]) <= y1
        )

    def graph_hop_profile(
        self,
        boundary_members: np.ndarray,
        allowed_members: np.ndarray,
        *,
        max_hops: int,
        decay: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return allowed nodes reached from a boundary and per-hop decay."""
        allowed = {int(index) for index in np.asarray(allowed_members, dtype=np.int64)}
        depth = {
            int(index): 0
            for index in np.asarray(boundary_members, dtype=np.int64)
            if int(index) in allowed
        }
        frontier = sorted(depth)
        for hop in range(max(0, int(max_hops))):
            next_frontier = []
            for index in frontier:
                for neighbor in sorted(self.macros[index].neighbors):
                    if neighbor in allowed and neighbor not in depth:
                        depth[neighbor] = hop + 1
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        members = np.asarray(sorted(depth), dtype=np.int64)
        scales = np.asarray(
            [np.clip(float(decay), 0.0, 1.0) ** depth[int(index)] for index in members],
            dtype=np.float64,
        )
        return members, scales

    def analyze_state(
        self,
        field: np.ndarray,
        canvas_width: float,
        canvas_height: float,
        *,
        contract_headroom: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        """Refresh revision-scoped heat, capacity, demand, and routing attribution."""
        field = np.asarray(field, dtype=np.float64)
        if field.ndim != 2 or not field.size:
            raise ValueError("location graph field must be a non-empty 2-D array")
        rows, cols = field.shape
        cell_w = float(canvas_width) / cols
        cell_h = float(canvas_height) / rows
        for node in self.macros.values():
            col = int(np.clip(node.position[0] / cell_w, 0, cols - 1))
            row = int(np.clip(node.position[1] / cell_h, 0, rows - 1))
            node.heat = float(field[row, col])

        routing = {
            "internal_cluster": 0.0,
            "cross_cluster": 0.0,
            "bridge_soft": 0.0,
            "unowned_soft": 0.0,
            "boundary": {},
        }
        internal = {cluster_id: 0.0 for cluster_id in self.clusters}
        external = {cluster_id: 0.0 for cluster_id in self.clusters}
        for left, node in self.macros.items():
            for right, weight_raw in node.neighbors.items():
                if right <= left:
                    continue
                weight = float(weight_raw)
                other = self.macros[right]
                if node.bridge_clusters or other.bridge_clusters:
                    routing["bridge_soft"] += weight
                elif (
                    node.kind == "soft"
                    and node.cluster_id < 0
                    or (other.kind == "soft" and other.cluster_id < 0)
                ):
                    routing["unowned_soft"] += weight
                elif node.cluster_id >= 0 and node.cluster_id == other.cluster_id:
                    routing["internal_cluster"] += weight
                    internal[node.cluster_id] += weight
                elif node.cluster_id >= 0 and other.cluster_id >= 0:
                    routing["cross_cluster"] += weight
                    external[node.cluster_id] += weight
                    external[other.cluster_id] += weight
                    pair = tuple(sorted((node.cluster_id, other.cluster_id)))
                    boundary = routing["boundary"]
                    boundary[pair] = boundary.get(pair, 0.0) + weight

        for cluster_id, cluster in self.clusters.items():
            members = cluster.members
            hard_area = sum(
                float(np.prod(self.macros[index].size)) for index in cluster.hard_members
            )
            soft_area = sum(
                float(np.prod(self.macros[index].size)) for index in cluster.soft_members
            )
            width = max(0.0, float(cluster.bbox[2] - cluster.bbox[0]))
            height = max(0.0, float(cluster.bbox[3] - cluster.bbox[1]))
            bbox_area = width * height
            total_area = hard_area + soft_area
            heat = [self.macros[index].heat for index in members]
            cluster.metrics = {
                "hard_area": hard_area,
                "soft_area": soft_area,
                "total_area": total_area,
                "bbox_area": bbox_area,
                "utilization": total_area / max(bbox_area, 1.0e-12),
                "free_capacity": max(0.0, bbox_area - total_area),
                "mean_heat": float(np.mean(heat)) if heat else 0.0,
                "max_heat": float(np.max(heat)) if heat else 0.0,
                "internal_demand": internal[cluster_id],
                "external_demand": external[cluster_id],
                "contract_headroom": float((contract_headroom or {}).get(cluster_id, 0.0)),
            }
            cluster.side_capacity = {
                "left": max(0.0, float(cluster.bbox[0])) * height,
                "right": max(0.0, float(canvas_width - cluster.bbox[2])) * height,
                "bottom": max(0.0, float(cluster.bbox[1])) * width,
                "top": max(0.0, float(canvas_height - cluster.bbox[3])) * width,
            }
            cluster.metrics["available_capacity"] = cluster.metrics["free_capacity"] + 0.25 * max(
                cluster.side_capacity.values(), default=0.0
            )
        routing["boundary"] = {
            f"{left}:{right}": float(weight)
            for (left, right), weight in sorted(routing["boundary"].items())
        }
        self.routing_attribution = routing
        self.analysis_revision = self.revision
        return routing

    def frontier_records(self, source: int, destination: int) -> list[dict[str, Any]]:
        """Describe macros on the geometric/topological frontier of two leaves."""
        source_node = self.clusters[int(source)]
        destination_node = self.clusters[int(destination)]
        vector = destination_node.centroid - source_node.centroid
        norm = max(float(np.linalg.norm(vector)), 1.0e-12)
        direction = vector / norm
        projections = {
            index: float(np.dot(self.macros[index].position - source_node.centroid, direction))
            for index in source_node.members
        }
        scale = max(
            max(projections.values(), default=0.0) - min(projections.values(), default=0.0), 1.0
        )
        rows = []
        destination_capacity = destination_node.metrics.get("available_capacity", 0.0)
        destination_heat = destination_node.metrics.get("mean_heat", 0.0)
        for index in source_node.members:
            node = self.macros[index]
            internal_weight = 0.0
            destination_weight = 0.0
            external_weight = 0.0
            for neighbor, weight in node.neighbors.items():
                owner = self.macros[neighbor].cluster_id
                if owner == source:
                    internal_weight += float(weight)
                else:
                    external_weight += float(weight)
                    if owner == destination:
                        destination_weight += float(weight)
            cut_gain = destination_weight - internal_weight
            facing = (projections[index] - min(projections.values(), default=0.0)) / scale
            area = max(float(np.prod(node.size)), 1.0e-12)
            capacity_ratio = destination_capacity / area
            pressure_relief = node.heat - destination_heat
            score = (
                2.0 * max(0.0, cut_gain)
                + external_weight
                + max(0.0, pressure_relief)
                + facing
                + min(capacity_ratio, 4.0) * 0.25
            )
            rows.append(
                {
                    "index": int(index),
                    "source": int(source),
                    "destination": int(destination),
                    "kind": node.kind,
                    "facing": float(facing),
                    "internal_weight": float(internal_weight),
                    "external_weight": float(external_weight),
                    "destination_weight": float(destination_weight),
                    "cut_gain": float(cut_gain),
                    "capacity_ratio": float(capacity_ratio),
                    "pressure_relief": float(pressure_relief),
                    "score": float(score),
                }
            )
        return sorted(rows, key=lambda row: (-row["score"], -row["facing"], row["index"]))

    def connected_frontier_bundle(
        self,
        seed: int,
        destination: int,
        *,
        max_members: int = 4,
    ) -> tuple[int, ...]:
        """Grow one bounded, connected source-frontier migration bundle."""
        source = self.cluster_of(seed)
        if source < 0 or source == int(destination):
            return ()
        ranked = {row["index"]: row for row in self.frontier_records(source, int(destination))}
        chosen = [int(seed)]
        seen = {int(seed)}
        queue = [int(seed)]
        while queue and len(chosen) < max(1, int(max_members)):
            current = queue.pop(0)
            candidates = [
                neighbor
                for neighbor in self.macros[current].neighbors
                if neighbor not in seen and self.cluster_of(neighbor) == source
            ]
            candidates.sort(key=lambda index: (-ranked[index]["score"], index))
            for neighbor in candidates:
                seen.add(neighbor)
                row = ranked[neighbor]
                if row["facing"] < 0.35 and row["destination_weight"] <= 0.0:
                    continue
                chosen.append(int(neighbor))
                queue.append(int(neighbor))
                if len(chosen) >= int(max_members):
                    break
        return tuple(chosen)

    def directional_graph_profile(
        self,
        boundary_members: np.ndarray,
        allowed_members: np.ndarray,
        *,
        max_hops: int,
        decay: float,
        min_edge_fraction: float = 0.08,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate a decompression wave with degree damping and hot-node boost."""
        allowed = {int(index) for index in np.asarray(allowed_members, dtype=np.int64)}
        depth = {int(index): 0 for index in boundary_members if int(index) in allowed}
        frontier = sorted(depth)
        for hop in range(max(0, int(max_hops))):
            next_frontier = []
            for index in frontier:
                eligible = [
                    (neighbor, float(weight))
                    for neighbor, weight in self.macros[index].neighbors.items()
                    if neighbor in allowed and neighbor not in depth
                ]
                strongest = max((weight for _neighbor, weight in eligible), default=0.0)
                for neighbor, weight in sorted(eligible):
                    if strongest > 0.0 and weight < strongest * float(min_edge_fraction):
                        continue
                    depth[neighbor] = hop + 1
                    next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        members = np.asarray(sorted(depth), dtype=np.int64)
        heat_values = np.asarray([self.macros[index].heat for index in members], dtype=np.float64)
        heat_scale = max(float(np.mean(heat_values)) if heat_values.size else 0.0, 1.0e-12)
        scales = []
        for index in members:
            internal_degree = sum(
                weight
                for neighbor, weight in self.macros[int(index)].neighbors.items()
                if neighbor in allowed
            )
            damping = 1.0 / (1.0 + 0.08 * float(internal_degree))
            heat_boost = np.clip(self.macros[int(index)].heat / heat_scale, 0.75, 1.35)
            scales.append((np.clip(decay, 0.0, 1.0) ** depth[int(index)]) * damping * heat_boost)
        return members, np.asarray(scales, dtype=np.float64)

    def dynamic_cluster_bounds(
        self, cluster_id: int, canvas_width: float, canvas_height: float
    ) -> np.ndarray:
        """Return a capacity-directed live region for one reassigned cluster."""
        cluster = self.clusters[int(cluster_id)]
        x0, y0, x1, y1 = map(float, cluster.bbox)
        width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        margin = 0.12 * max(width, height)
        capacity = cluster.side_capacity
        best_side = max(capacity, key=capacity.get) if capacity else "right"
        expansion = {
            "left": [margin, 0, 0, 0],
            "right": [0, 0, margin, 0],
            "bottom": [0, margin, 0, 0],
            "top": [0, 0, 0, margin],
        }[best_side]
        return np.asarray(
            [
                max(0.0, x0 - margin - expansion[0]),
                max(0.0, y0 - margin - expansion[1]),
                min(float(canvas_width), x1 + margin + expansion[2]),
                min(float(canvas_height), y1 + margin + expansion[3]),
            ]
        )

    def legalization_order(self, indices) -> np.ndarray:
        """Order hard macros boundary-first, then graph followers and interiors."""
        requested = [int(index) for index in np.asarray(indices, dtype=np.int64)]
        allowed = set(requested)
        ordered = []
        for cluster_id, cluster in sorted(self.clusters.items()):
            local = [index for index in cluster.hard_members if index in allowed]
            if not local:
                continue

            def boundary_key(index):
                node = self.macros[index]
                external = sum(
                    weight
                    for neighbor, weight in node.neighbors.items()
                    if self.cluster_of(neighbor) != cluster_id
                )
                x0, y0, x1, y1 = cluster.bbox
                edge_distance = min(
                    node.position[0] - x0,
                    x1 - node.position[0],
                    node.position[1] - y0,
                    y1 - node.position[1],
                )
                return (-external, edge_distance, index)

            seeds = sorted(local, key=boundary_key)
            seen = set()
            queue = deque(seeds)
            while queue:
                index = queue.popleft()
                if index in seen:
                    continue
                seen.add(index)
                ordered.append(index)
                followers = [
                    n for n in self.macros[index].neighbors if n in local and n not in seen
                ]
                followers.sort(key=boundary_key)
                queue.extend(followers)
        ordered.extend(index for index in requested if index not in set(ordered))
        return np.asarray(ordered, dtype=np.int64)

    def checkpoint(self, reason: str, *, proxy: dict[str, float] | None = None) -> int:
        """Save a bounded graph-state checkpoint for diagnostics and rollback."""
        self._checkpoints.append(
            {
                "revision": int(self.revision),
                "reason": str(reason),
                "positions": np.asarray([self.macros[i].position for i in sorted(self.macros)]),
                "roles": tuple(
                    (
                        self.macros[i].cluster_id,
                        self.macros[i].subcluster_id,
                        self.macros[i].parent_id,
                        self.macros[i].bridge_clusters,
                    )
                    for i in sorted(self.macros)
                ),
                "proxy": dict(proxy or {}),
                "cluster_geometry": {
                    int(cluster_id): {
                        "centroid": cluster.centroid.copy(),
                        "bbox": cluster.bbox.copy(),
                        "metrics": dict(cluster.metrics),
                        "side_capacity": dict(cluster.side_capacity),
                    }
                    for cluster_id, cluster in self.clusters.items()
                },
            }
        )
        return int(self.revision)

    def rollback_graph(self) -> bool:
        """Restore the most recent graph-only checkpoint."""
        if not self._checkpoints:
            return False
        state = self._checkpoints.pop()
        for index, position in enumerate(state["positions"]):
            node = self.macros[index]
            node.position[:] = position
            role = state["roles"][index]
            node.cluster_id, node.subcluster_id, node.parent_id = map(int, role[:3])
            node.bridge_clusters = tuple(role[3])
        for cluster_id, cluster in self.clusters.items():
            cluster.hard_members = tuple(
                i for i, n in self.macros.items() if n.kind == "hard" and n.cluster_id == cluster_id
            )
            cluster.soft_members = tuple(
                i for i, n in self.macros.items() if n.kind == "soft" and n.cluster_id == cluster_id
            )
        self._rebuild_cluster_adjacency()
        self._refresh_clusters(self.clusters)
        self.revision += 1
        return True

    def record_transfer(self, proposal: dict[str, Any], proxy_gain: float, dc_gain: float) -> None:
        """Append one accepted ownership/movement transaction to bounded history."""
        self.transfer_history.append(
            {
                "revision": int(self.revision),
                "kind": str(proposal.get("kind", "unknown")),
                "indices": [int(index) for index, _target in proposal.get("moves", ())],
                "assignments": {str(k): int(v) for k, v in proposal.get("assignments", {}).items()},
                "proxy_gain": float(proxy_gain),
                "density_congestion_gain": float(dc_gain),
            }
        )

    def to_visualizer_payload(self, *, max_frontiers: int = 96) -> dict[str, Any]:
        """Serialize bounded hierarchy pressure, capacity, frontier, and history state."""
        frontier = []
        for source, cluster in sorted(self.clusters.items()):
            for destination in sorted(self.cluster_neighbors(source)):
                if source < destination:
                    frontier.extend(self.frontier_records(source, destination)[:2])
                    frontier.extend(self.frontier_records(destination, source)[:2])
        return {
            "summary": self.summary(),
            "analysis_revision": int(self.analysis_revision),
            "clusters": {
                str(cluster_id): {
                    "centroid": cluster.centroid.tolist(),
                    "bbox": cluster.bbox.tolist(),
                    "neighbors": {
                        str(k): float(v) for k, v in self.cluster_neighbors(cluster_id).items()
                    },
                    "metrics": {k: float(v) for k, v in cluster.metrics.items()},
                    "side_capacity": {k: float(v) for k, v in cluster.side_capacity.items()},
                }
                for cluster_id, cluster in sorted(self.clusters.items())
            },
            "routing_attribution": self.routing_attribution,
            "frontiers": frontier[: max(0, int(max_frontiers))],
            "transfers": list(self.transfer_history),
        }
