"""Deterministic hierarchy colours and wiring models for the dashboard."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Any, Mapping

def hierarchy_color(cluster: int, *, parent: bool = False) -> tuple[int, int, int, int]:
    """Return a stable related colour for a hierarchy identifier."""
    if cluster < 0:
        return (135, 145, 155, 150)
    hue = (int(cluster) * 0.6180339887498949) % 1.0
    saturation = 0.42 if parent else 0.68
    value = 0.72 if parent else 0.92
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (round(255 * r), round(255 * g), round(255 * b), 85 if parent else 220)


@dataclass(frozen=True)
class Endpoint:
    macro: int | None
    port: int | None
    offset: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class Hyperedge:
    index: int
    weight: float
    endpoints: tuple[Endpoint, ...]


def extract_real_nets(metadata: Mapping[str, Any]) -> list[Hyperedge]:
    """Extract macro/port topology while preserving stable net order."""
    macro_count = len(metadata.get("macro_names", ()))
    ports = metadata.get("port_positions", ())
    offsets = metadata.get("macro_pin_offsets", ())
    pin_cursor = [0] * macro_count
    rows = []
    weights = metadata.get("net_weights", ())
    for net_index, nodes in enumerate(metadata.get("net_nodes", ())):
        endpoints = []
        for raw_node in nodes:
            node = int(raw_node)
            if 0 <= node < macro_count:
                available = offsets[node] if node < len(offsets) else ()
                cursor = pin_cursor[node]
                offset = available[cursor] if cursor < len(available) else (0.0, 0.0)
                pin_cursor[node] += 1
                endpoints.append(Endpoint(node, None, (float(offset[0]), float(offset[1]))))
            else:
                port = node - macro_count
                if 0 <= port < len(ports):
                    endpoints.append(Endpoint(None, port))
        if len(endpoints) >= 2:
            weight = float(weights[net_index]) if net_index < len(weights) else 1.0
            rows.append(Hyperedge(net_index, weight, tuple(endpoints)))
    return rows


def filter_real_nets(
    nets: list[Hyperedge], limit: int, selected_macro: int | None = None
) -> list[Hyperedge]:
    """Stable-rank low-fanout/high-weight nets plus selected-macro incidence."""
    ranked = sorted(nets, key=lambda net: (len(net.endpoints), -net.weight, net.index))
    chosen = {net.index for net in ranked[: max(0, int(limit))]}
    if selected_macro is not None:
        chosen.update(
            net.index
            for net in nets
            if any(endpoint.macro == selected_macro for endpoint in net.endpoints)
        )
    return [net for net in nets if net.index in chosen]


def collapse_synthetic_groups(hierarchy: Mapping[str, Any], group_weight: int) -> list[dict]:
    """Represent repeated grouping nets as one centroid-and-spoke hyperedge."""
    return [
        {
            "cluster": int(cluster),
            "members": tuple(int(member) for member in members),
            "weight": int(group_weight),
        }
        for cluster, members in sorted(
            hierarchy.get("leaf_clusters", {}).items(), key=lambda row: int(row[0])
        )
        if len(members) >= 2
    ]


def hierarchy_centroid_edges(hierarchy: Mapping[str, Any]) -> list[tuple[int, int, float]]:
    rows = []
    for edge in hierarchy.get("edges", ()):
        if len(edge) >= 2:
            rows.append((int(edge[0]), int(edge[1]), float(edge[2]) if len(edge) > 2 else 1.0))
    return rows
