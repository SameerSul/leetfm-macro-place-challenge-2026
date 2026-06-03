"""Vectorized congestion cost and PLC patching."""

import numpy as np
from macro_place.benchmark import Benchmark

from placer.routing.apply import _build_cong_cache, _vectorized_get_routing
from placer.scoring.wirelength import _build_wl_cache

def _vectorized_get_congestion_cost(plc) -> float:
    """Numpy-fast replacement for `PlacementCost.get_congestion_cost`.

    The reference sorts V+H routing congestion and returns mean(top 5%). We
    get the top-cnt elements via np.partition (unordered, but the mean is
    order-independent) - same result at O(n) instead of an O(n log n) sort.
    """
    if plc.FLAG_UPDATE_CONGESTION:
        plc.get_routing()  # patched to _vectorized_get_routing
    v = plc.V_routing_cong
    h = plc.H_routing_cong
    # Concat. plc may still hold the legacy lists on the very first call
    # (before our get_routing patched-write executes); handle gracefully.
    if isinstance(v, list):
        v = np.asarray(v, dtype=np.float64)
    if isinstance(h, list):
        h = np.asarray(h, dtype=np.float64)
    xx = np.concatenate([v, h])
    n = xx.size
    cnt = int(n * 0.05)  # floor (positive value)
    if cnt == 0:
        return float(xx.max())
    # Top-cnt values via partition (unordered, but mean is order-independent).
    top = np.partition(xx, n - cnt)[n - cnt:]
    return float(top.sum() / cnt)


def _patch_plc_congestion(plc, benchmark: Benchmark) -> None:
    """Install vectorized congestion (get_routing + get_congestion_cost) on this plc."""
    if getattr(plc, "_cong_vec_installed", False):
        return
    _build_wl_cache(plc)
    _build_cong_cache(plc, benchmark)
    plc.get_routing = lambda _plc=plc: _vectorized_get_routing(_plc)
    plc.get_congestion_cost = lambda _plc=plc: _vectorized_get_congestion_cost(_plc)
    plc._cong_vec_installed = True


def _ensure_congestion_arrays(plc) -> None:
    """Mirror objective._ensure_congestion_arrays without re-importing."""
    expected_size = plc.grid_col * plc.grid_row
    if len(plc.H_routing_cong) != expected_size:
        # numpy arrays (not lists) to match _vectorized_get_routing's output.
        plc.V_routing_cong = np.zeros(expected_size, dtype=np.float64)
        plc.H_routing_cong = np.zeros(expected_size, dtype=np.float64)
        plc.V_macro_routing_cong = np.zeros(expected_size, dtype=np.float64)
        plc.H_macro_routing_cong = np.zeros(expected_size, dtype=np.float64)
