"""
surrogate.py
============

Fast surrogate ranker for benchmarks where exact PlacementCost evaluation is
too slow for multi-restart selection (n > 340 or grid_cells > 2000 in placer.py).

Status (after calibration)
--------------------------
The DEFAULT surrogate is now WL-only -- weighted half-perimeter wirelength
over the netlist, normalized by total_net_weight * canvas_perimeter.

Calibration test (`_calibration_test.py`) on 10 legalized restarts each from
ibm01 and ibm10 measured Spearman rank correlation between candidate variants
and the real proxy_cost:

    | variant            | ibm01   | ibm10   |
    | WL only            | +0.479  | +0.539  |  <-- default
    | WL + 0.5*C         | +0.661  | +0.079  |
    | WL + 0.5*D + 0.5*C | -0.515  | +0.055  |

WL-only is the only weighting that stays positive on both small and large
benchmarks. The composite formulas were dropped because:

  1. Macro-density SoS is anti-correlated with real density on ibm01
     (-0.71). Penalizing spread is the wrong sign on benchmarks where
     real density is dominated by stdcell mass, not macro distribution.

  2. Wire-bbox congestion (`weight/n_cells` per-net) makes each net
     contribute a constant grid sum -- only concentration varies, while
     bbox area does not. On ibm10 sur_c is 3.34-3.40 (range 0.06) and
     selects the WORST real placement (k=5: 1.7045 vs baseline 1.4037).

Caching
-------
Per-benchmark constants (numpy net arrays, pre-filtered indices, pre-allocated
position buffer) are cached on the benchmark instance via a single attribute
`_surrogate_cache`. Avoids the unsafe `id(benchmark)` global-dict pattern --
Python reuses memory addresses after GC -- and ties cache lifetime to the
benchmark itself.

Net topology source
-------------------
The shared `macro_place/loader.py` leaves `benchmark.net_nodes = []` and
`benchmark.net_weights = zeros(num_nets)` (loader.py:118-120) -- the net
topology is never extracted. We can't modify the loader (hackathon rule:
read-only outside the submission), so when `surrogate_score` is called with
the `plc` object, we extract net topology from `plc.nets` ourselves and
populate the cache. Without `plc`, no nets are available and
`surrogate_usable()` returns False -- caller should fall back to baseline.

Usage in placer.py
------------------
    from submissions.varrahan.v1.surrogate import surrogate_score, surrogate_usable
    if surrogate_usable(benchmark, plc=plc):
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
    """Returns True iff this benchmark has net topology available -- required
    for the WL ranker. When False, `surrogate_score` would only have access to
    the macro-density term (anti-correlated with proxy on ibm01), so caller
    should fall back to returning the baseline placement instead.

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
    w_wl: float = 1.0,
    w_density: float = 0.0,
    w_congestion: float = 0.0,
) -> float:
    """Surrogate ranker. Default is WL-only based on calibration evidence:

        | variant            | Spearman vs real proxy on ibm01 | on ibm10 |
        | WL only            |             +0.479               | +0.539   |
        | WL + 0.5*C         |             +0.661               | +0.079   |
        | WL + 0.5*D + 0.5*C |             -0.515               | +0.055   |

    WL-only is the only weighting that stays positive on both small (ibm01)
    and large (ibm10) benchmarks. The density-SoS term is anti-correlated on
    ibm01; the wire-congestion term degenerates on large benchmarks because
    its per-net `weight/n_cells` normalization makes each net contribute a
    constant grid sum -- only concentration jitters, while bbox area does
    not -- and the resulting near-constant term drowns out the WL signal.

    Pass non-default weights for diagnostic comparisons; for actual restart
    selection use the default.

    Each component is normalized by a placement-independent quantity so the
    weights are meaningful and the result is O(1):

      - WL:  divide by total_net_weight * canvas_perimeter
             -> unitless "average normalized net span"
      - density: divide by n^2 / G^2  (1.0 when macros are perfectly uniform)
      - congestion: divide by (total_net_weight / G)^2

    Lower is better. Absolute value is NOT calibrated against real proxy and
    must never be compared across benchmarks -- only used to rank candidate
    placements within one benchmark.
    """
    cache = _get_cache(benchmark, plc)
    all_pos = _fill_pos(pos, cache)

    perim = 2.0 * (cw + ch)
    total_w = cache["total_w_raw"]

    wl_raw = _hpwl(all_pos, cache) if w_wl != 0.0 else 0.0
    wc_raw = _wire_congestion(all_pos, cache, cw, ch, G) if w_congestion != 0.0 else 0.0
    md_raw = _macro_density_sos(all_pos, n, cw, ch, G) if w_density != 0.0 else 0.0

    wl = wl_raw / max(total_w * perim, 1e-9)
    wc = wc_raw / max((total_w / G) ** 2, 1e-9)
    md = md_raw / max(n * n / (G * G), 1e-9)

    return w_wl * wl + w_density * md + w_congestion * wc


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
        "score_default": wl,
        "score_legacy": 1.0 * wl + 0.5 * md + 0.5 * wc,
        "n_nets_total": cache["n_nets_total"],
        "n_nets_filtered": cache["n_nets_filtered"],
    }
