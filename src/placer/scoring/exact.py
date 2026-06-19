"""Exact proxy scoring wrappers over patched PLC methods."""

import atexit
import time

import torch
from macro_place.benchmark import Benchmark

from placer import constants as const
from placer.plc.placement import _fast_set_placement
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.density import _patch_plc_density
from placer.scoring.wirelength import _patch_plc_wirelength

# Optional profiling: count + time full _exact_proxy calls.
_PROFILE_EXACT = const.PROFILE_EXACT
_exact_stats = {"calls": 0, "total_s": 0.0}
if _PROFILE_EXACT:
    atexit.register(
        lambda: print(
            f"[PROFILE_EXACT] _exact_proxy: {_exact_stats['calls']} calls, "
            f"{_exact_stats['total_s']:.2f}s total"
        )
    )


def _exact_proxy(placement: torch.Tensor, benchmark: Benchmark, plc) -> float:
    """Score placement with the fast wirelength, density, and congestion paths."""
    _t0 = time.perf_counter() if _PROFILE_EXACT else 0.0
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, benchmark)
    _patch_plc_density(plc, benchmark)
    placement_np = placement.cpu().numpy()
    # The placement tensor carries both hard and soft macro positions.
    _fast_set_placement(plc, placement_np, benchmark)
    wl = plc.get_cost()
    dens = plc.get_density_cost()
    cong = plc.get_congestion_cost()
    if _PROFILE_EXACT:
        _exact_stats["calls"] += 1
        _exact_stats["total_s"] += time.perf_counter() - _t0
    return float(wl + 0.5 * dens + 0.5 * cong)
