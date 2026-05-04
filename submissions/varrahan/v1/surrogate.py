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

Caching
-------
Per-benchmark constants (numpy net arrays, pre-filtered indices, pre-allocated
position buffer) are cached on the benchmark instance via a single attribute
`_surrogate_cache`. This avoids the unsafe `id(benchmark)` global-dict pattern
(Python reuses memory addresses after GC) and ties the cache lifetime to the
benchmark itself.

Net topology source
-------------------
The shared `macro_place/loader.py` leaves `benchmark.net_nodes = []` and
`benchmark.net_weights = zeros(num_nets)` (loader.py:118-120) -- the net
topology is never extracted. We can't modify the loader (hackathon rule:
read-only outside the submission), so when `surrogate_score` is called with
the `plc` object, we extract net topology from `plc.nets` ourselves and
populate the cache. Without `plc`, the surrogate degenerates to macro-density
only and `surrogate_usable()` returns False.

Usage in placer.py
------------------
    from submissions.varrahan.v1.surrogate import surrogate_score
    score = surrogate_score(legalized_pos, benchmark, n, cw, ch, plc=plc)
"""

import numpy as np
from macro_place.benchmark import Benchmark


_CACHE_ATTR = "_surrogate_cache"


def _extract_nets_from_plc(benchmark: Benchmark, plc, n_total: int):
    """Build (nets, weights) from `plc.nets` since the loader leaves them empty.

    `plc.nets` is a dict mapping net name -> list of pin names like 'a7419/IP1'.
    We map each pin name back to a benchmark tensor index using plc's
    hard/soft/port index lists, dedupe per-net (multiple pins on same macro
    contribute one node), and drop nets with < 2 distinct macro endpoints.

    Returns (nets, weights) where nets is a list of np.ndarray and weights is
    a parallel list of floats. plc doesn't expose per-net weight, so all
    weights are 1.0 -- consistent with what the actual loader stores when nets
    *are* eventually plumbed through.
    """
    name_to_idx = {}
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros

    for i, plc_idx in enumerate(plc.hard_macro_indices):
        name_to_idx[plc.modules_w_pins[plc_idx].get_name()] = i
    for i, plc_idx in enumerate(plc.soft_macro_indices):
        name_to_idx[plc.modules_w_pins[plc_idx].get_name()] = n_hard + i
    for i, plc_idx in enumerate(plc.port_indices):
        name_to_idx[plc.modules_w_pins[plc_idx].get_name()] = n_macros + i

    nets = []
    weights = []
    for pin_list in plc.nets.values():
        idxs = set()
        for pin_name in pin_list:
            macro_name = pin_name.rsplit("/", 1)[0] if "/" in pin_name else pin_name
            idx = name_to_idx.get(macro_name)
            if idx is not None and idx < n_total:
                idxs.add(idx)
        if len(idxs) >= 2:
            nets.append(np.asarray(sorted(idxs), dtype=np.int64))
            weights.append(1.0)
    return nets, weights


def _get_cache(benchmark: Benchmark, plc=None) -> dict:
    """Build (once) and return per-benchmark caches.

    Cached fields:
      all_pos_buffer : [n_total, 2] float64 array. Soft-macro and port positions
        are pre-filled; the hard-macro slice is overwritten per call.
      n_hard, n_total: ints (num hard macros, num macros + num ports)
      nets    : list of pre-filtered index arrays (only nodes < n_total kept,
                only nets with >= 2 such nodes retained)
      weights : np.ndarray parallel to `nets`
      total_w_raw : float, sum of all net weights used in `nets`
      nets_source : 'benchmark' if loader populated net_nodes,
                    'plc' if we extracted them from plc.nets ourselves,
                    'empty' if neither path produced any nets
      n_nets_total, n_nets_filtered: ints, for diagnostics

    `plc` is consulted only on the FIRST call. Once the cache exists, later
    calls reuse it regardless of whether plc is passed.
    """
    cache = getattr(benchmark, _CACHE_ATTR, None)
    if cache is not None:
        return cache

    n_hard = benchmark.num_hard_macros
    macros = benchmark.macro_positions.numpy().astype(np.float64).copy()
    if benchmark.port_positions.numel() > 0:
        ports = benchmark.port_positions.numpy().astype(np.float64)
        all_pos_buffer = np.concatenate([macros, ports], axis=0)
    else:
        all_pos_buffer = macros
    n_total = all_pos_buffer.shape[0]

    nets = []
    weights = []
    nets_source = "empty"
    n_nets_total = 0

    if len(benchmark.net_nodes) > 0:
        nets_source = "benchmark"
        n_nets_total = len(benchmark.net_nodes)
        for nodes_t, w in zip(benchmark.net_nodes, benchmark.net_weights):
            nodes_np = nodes_t.numpy()
            valid = nodes_np[nodes_np < n_total]
            if len(valid) < 2:
                continue
            nets.append(valid)
            weights.append(float(w))
    elif plc is not None:
        nets, weights = _extract_nets_from_plc(benchmark, plc, n_total)
        nets_source = "plc"
        n_nets_total = len(plc.nets)

    weights_np = np.asarray(weights, dtype=np.float64)
    cache = {
        "all_pos_buffer": all_pos_buffer,
        "n_hard": n_hard,
        "n_total": n_total,
        "nets": nets,
        "weights": weights_np,
        "total_w_raw": float(weights_np.sum()) if len(weights_np) else 0.0,
        "nets_source": nets_source,
        "n_nets_total": n_nets_total,
        "n_nets_filtered": len(nets),
    }
    setattr(benchmark, _CACHE_ATTR, cache)
    return cache


def _fill_pos(pos: np.ndarray, cache: dict) -> np.ndarray:
    """Overwrite the hard-macro slice of the cached buffer with `pos[:n_hard]`.
    Returns the buffer (mutated). Not thread-safe; fine for sequential ranking.
    """
    buf = cache["all_pos_buffer"]
    n = cache["n_hard"]
    buf[:n] = pos[:n]
    return buf


def _hpwl(all_pos: np.ndarray, cache: dict) -> float:
    """Weighted half-perimeter wirelength over pre-filtered nets."""
    total = 0.0
    weights = cache["weights"]
    for i, valid in enumerate(cache["nets"]):
        net_pos = all_pos[valid]
        x_span = float(net_pos[:, 0].max() - net_pos[:, 0].min())
        y_span = float(net_pos[:, 1].max() - net_pos[:, 1].min())
        total += float(weights[i]) * (x_span + y_span)
    return total


def _macro_density_sos(pos: np.ndarray, n: int, cw: float, ch: float, G: int = 20) -> float:
    """Sum-of-squares macro count on a G x G grid (matches the old _density_score).

    Kept as the density-component proxy because the proxy's density term itself
    rewards spread; the anti-correlation problem only appears when this is the
    SOLE signal. Combined with wire-congestion below, it contributes correctly.
    """
    cw_g = cw / G
    ch_g = ch / G
    c = np.clip((pos[:n, 0] / cw_g).astype(int), 0, G - 1)
    r = np.clip((pos[:n, 1] / ch_g).astype(int), 0, G - 1)
    grid = np.zeros((G, G), dtype=np.float64)
    np.add.at(grid, (r, c), 1.0)
    return float((grid ** 2).sum())


def _wire_congestion(
    all_pos: np.ndarray, cache: dict, cw: float, ch: float, G: int = 20
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
    weights = cache["weights"]

    for i, valid in enumerate(cache["nets"]):
        net_pos = all_pos[valid]
        c_lo = max(0, min(G - 1, int(net_pos[:, 0].min() / cw_g)))
        c_hi = max(0, min(G - 1, int(net_pos[:, 0].max() / cw_g)))
        r_lo = max(0, min(G - 1, int(net_pos[:, 1].min() / ch_g)))
        r_hi = max(0, min(G - 1, int(net_pos[:, 1].max() / ch_g)))
        n_cells = (c_hi - c_lo + 1) * (r_hi - r_lo + 1)
        grid[r_lo:r_hi + 1, c_lo:c_hi + 1] += float(weights[i]) / n_cells

    return float((grid ** 2).sum())


def surrogate_usable(benchmark: Benchmark, plc=None) -> bool:
    """Returns True iff this benchmark has the net topology needed for the WL
    and congestion components to be meaningful. When False, `surrogate_score`
    degenerates to macro-density-only -- which is the original anti-correlated
    fallback that placer.py deliberately avoids. Caller should fall back to
    returning the baseline placement instead of using the surrogate.

    Pass `plc` so we can extract nets from `plc.nets` when the shared loader
    has left `benchmark.net_nodes` empty (which is currently always the case).
    """
    return _get_cache(benchmark, plc)["n_nets_filtered"] > 0


def surrogate_score(
    pos: np.ndarray,
    benchmark: Benchmark,
    n: int,
    cw: float,
    ch: float,
    plc=None,
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
    cache = _get_cache(benchmark, plc)
    all_pos = _fill_pos(pos, cache)

    perim = 2.0 * (cw + ch)
    total_w = cache["total_w_raw"]

    wl_raw = _hpwl(all_pos, cache)
    md_raw = _macro_density_sos(all_pos, n, cw, ch, G)
    wc_raw = _wire_congestion(all_pos, cache, cw, ch, G)

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
    plc=None,
    G: int = 20,
) -> dict:
    """Same as surrogate_score but returns the three normalized components
    separately (and the raw values). Useful for diagnostics when tuning.
    """
    cache = _get_cache(benchmark, plc)
    all_pos = _fill_pos(pos, cache)

    perim = 2.0 * (cw + ch)
    total_w = cache["total_w_raw"]

    wl_raw = _hpwl(all_pos, cache)
    md_raw = _macro_density_sos(all_pos, n, cw, ch, G)
    wc_raw = _wire_congestion(all_pos, cache, cw, ch, G)

    wl = wl_raw / max(total_w * perim, 1e-9)
    md = md_raw / max(n * n / (G * G), 1e-9)
    wc = wc_raw / max((total_w / G) ** 2, 1e-9)

    return {
        "wl": wl,
        "density": md,
        "congestion": wc,
        "wl_raw": wl_raw,
        "density_raw": md_raw,
        "congestion_raw": wc_raw,
        "score": 1.0 * wl + 0.5 * md + 0.5 * wc,
        "n_nets_total": cache["n_nets_total"],
        "n_nets_filtered": cache["n_nets_filtered"],
    }
