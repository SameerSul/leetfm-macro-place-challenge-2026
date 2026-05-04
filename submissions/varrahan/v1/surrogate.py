"""
surrogate.py
============

Fast surrogate scorer for benchmarks where exact PlacementCost evaluation is
too slow for multi-restart selection (n > 340 or grid_cells > 2000 in placer.py).

Background
----------
The default fallback in placer.py for these benchmarks is to return the baseline
legalization unmodified, because the previous `_density_score` (sum-of-squares
macro occupancy) was empirically anti-correlated with proxy cost: it rewards
spread placements, but spread placements have longer wires and higher routing
congestion, which dominates the real proxy (PROGRESS.md v2/v3 regression --
density-selected ibm11 result was 11.5% WORSE than baseline).

This surrogate combines three components that mirror the proxy formula:

    proxy = 1.0 * wirelength + 0.5 * density + 0.5 * congestion

The wire-density term is the key addition over the old fallback. Each net's
bounding box is rasterized onto a grid (uniform per-cell weight), so when
macros spread out, net bboxes grow, cover more cells, and overlap more --
which raises the score and correctly penalizes the over-spreading that hurt
the actual proxy on ibm10-18.

Each component is normalized so the proxy formula's weights apply directly;
absolute scale is uncalibrated and only meaningful for ranking within one
benchmark.

Usage in placer.py
------------------
    from submissions.varrahan.v1.surrogate import surrogate_score
    score = surrogate_score(legalized_pos, benchmark, n, cw, ch)
"""

import numpy as np
from macro_place.benchmark import Benchmark

_BENCHMARK_CACHE = {}

def _get_cached_nets(benchmark: Benchmark):
    """
    Dynamically caches NumPy conversions on the benchmark instance.
    """
    b_id = id(benchmark)
    
    if b_id not in _BENCHMARK_CACHE:
        # Perform translation
        net_nodes_np = [nodes.numpy() for nodes in benchmark.net_nodes]
        net_weights_np = benchmark.net_weights.numpy()
        _BENCHMARK_CACHE[b_id] = (net_nodes_np, net_weights_np)
        
    # Return the translated arrays from our dictionary
    return _BENCHMARK_CACHE[b_id]

def _all_node_pos(pos: np.ndarray, benchmark: Benchmark, n: int) -> np.ndarray:
    """Concatenate macro positions (hard from `pos`, soft from benchmark) and
    port positions into one array indexable by net node indices.
    """
    macros = benchmark.macro_positions.numpy().astype(np.float64).copy()
    macros[:n] = pos[:n]
    if benchmark.port_positions.numel() > 0:
        ports = benchmark.port_positions.numpy().astype(np.float64)
        return np.concatenate([macros, ports], axis=0)
    return macros


def _hpwl(pos: np.ndarray, benchmark: Benchmark) -> float:
    """Weighted half-perimeter wirelength over all nets.

    Includes hard macros, soft macros, and ports as net endpoints -- omitting
    any of them would let the score ignore real wire span growth, which is the
    failure mode of the old macro-only density fallback.
    """
    n_total = pos.shape[0]
    total = 0.0
    net_nodes_np, net_weights_np = _get_cached_nets(benchmark)
    
    for net_idx, nodes_np in enumerate(net_nodes_np):
        # Drop any out-of-range indices
        nodes_np = nodes_np[nodes_np < n_total]
        if len(nodes_np) < 2:
            continue
        net_pos = pos[nodes_np]
        x_span = float(net_pos[:, 0].max() - net_pos[:, 0].min())
        y_span = float(net_pos[:, 1].max() - net_pos[:, 1].min())
        total += float(net_weights_np[net_idx]) * (x_span + y_span)
    return total


def _macro_density_sos(pos: np.ndarray, n: int, cw: float, ch: float, G: int = 20) -> float:
    """Sum-of-squares macro count on a G x G grid (matches the old _density_score).

    Kept as the density-component proxy because the proxy's density term itself
    rewards spread; the anti-correlation problem only appears when this is the
    SOLE signal. Combined with wire-congestion below, it contributes correctly.
    """
    cw_g = cw / G
    ch_g = ch / G
    
    # Vectorized calculation of row and column indice
    c = np.clip((pos[:n, 0] / cw_g).astype(int), 0, G - 1)
    r = np.clip((pos[:n, 1] / ch_g).astype(int), 0, G - 1)
    
    grid = np.zeros((G, G), dtype=np.float64)
    np.add.at(grid, (r, c), 1.0)
    
    return float((grid ** 2).sum())


def _wire_congestion(
    pos: np.ndarray, benchmark: Benchmark, cw: float, ch: float, G: int = 20
) -> float:
    """Routing-demand proxy: rasterize each net's bbox onto a G x G grid with
    uniform density (net_weight / bbox_area_cells). Sum-of-squares of the grid
    penalizes overlap of net bboxes -- zones where many nets must route.

    This is what the old fallback was missing: when macros spread out, peak
    macro density drops (good for the old score) but net bboxes grow and
    overlap more (bad for real congestion). This term restores the right sign.
    """
    grid = np.zeros((G, G), dtype=np.float64)
    cw_g = cw / G
    ch_g = ch / G
    n_total = pos.shape[0]

    net_nodes_np, net_weights_np = _get_cached_nets(benchmark)

    for net_idx, nodes_np in enumerate(net_nodes_np):
        nodes_np = nodes_np[nodes_np < n_total]
        if len(nodes_np) < 2:
            continue
        weight = float(net_weights_np[net_idx])
        net_pos = pos[nodes_np]
        
        c_lo = max(0, min(G - 1, int(net_pos[:, 0].min() / cw_g)))
        c_hi = max(0, min(G - 1, int(net_pos[:, 0].max() / cw_g)))
        r_lo = max(0, min(G - 1, int(net_pos[:, 1].min() / ch_g)))
        r_hi = max(0, min(G - 1, int(net_pos[:, 1].max() / ch_g)))
        
        n_cells = (c_hi - c_lo + 1) * (r_hi - r_lo + 1)
        grid[r_lo:r_hi + 1, c_lo:c_hi + 1] += weight / n_cells
        
    return float((grid ** 2).sum())


def surrogate_score(
    pos: np.ndarray,
    benchmark: Benchmark,
    n: int,
    cw: float,
    ch: float,
    G: int = 20,
) -> float:
    """Composite proxy-cost surrogate: 1*WL + 0.5*density + 0.5*congestion.

    Each component is normalized by a placement-independent quantity so the
    proxy formula's weights are meaningful and the result is O(1):

      - WL:  divide by total_net_weight * canvas_perimeter
             -> unitless "average normalized net span"
      - density: divide by n^2 / G^2 (= mean grid value squared, summed)
             -> 1.0 when macros are perfectly uniform across the grid
      - congestion: divide by (total_net_weight / G)^2
             -> O(1) regardless of net count

    Lower is better, like proxy_cost. Absolute value is NOT calibrated against
    the real proxy and should never be compared across benchmarks -- only used
    to rank candidate placements within a single benchmark.
    """
    perim = 2.0 * (cw + ch)
    total_w = float(benchmark.net_weights.sum().item())

    all_pos = _all_node_pos(pos, benchmark, n)
    
    wl_raw = _hpwl(all_pos, benchmark)
    md_raw = _macro_density_sos(all_pos, n, cw, ch, G)
    wc_raw = _wire_congestion(all_pos, benchmark, cw, ch, G)

    wl = wl_raw / max(total_w * perim, 1e-9)
    md = md_raw / max(n * n / (G * G), 1e-9)
    wc = wc_raw / max((total_w / G) ** 2, 1e-9)

    return 1.0 * wl + 0.5 * md + 0.5 * wc


def surrogate_components(
    pos: np.ndarray,
    benchmark: Benchmark,
    n: int,
    cw: float,
    ch: float,
    G: int = 20,
) -> dict:
    """Same as surrogate_score but returns the three normalized components
    separately. Useful for diagnostics when tuning.
    """
    perim = 2.0 * (cw + ch)
    total_w = float(benchmark.net_weights.sum().item())

    all_pos = _all_node_pos(pos, benchmark, n)
    
    wl_raw = _hpwl(all_pos, benchmark, n)
    md_raw = _macro_density_sos(all_pos, n, cw, ch, G)
    wc_raw = _wire_congestion(all_pos, benchmark, n, cw, ch, G)

    return {
        "wl": wl_raw / max(total_w * perim, 1e-9),
        "density": md_raw / max(n * n / (G * G), 1e-9),
        "congestion": wc_raw / max((total_w / G) ** 2, 1e-9),
        "score": (
            wl_raw / max(total_w * perim, 1e-9)
            + 0.5 * md_raw / max(n * n / (G * G), 1e-9)
            + 0.5 * wc_raw / max((total_w / G) ** 2, 1e-9)
        ),
    }
