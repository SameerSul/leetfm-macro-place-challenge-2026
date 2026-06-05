"""Exact proxy scoring wrappers over patched PLC methods."""

import torch
from macro_place.benchmark import Benchmark

from placer.plc.placement import _fast_set_placement
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.density import _patch_plc_density
from placer.scoring.wirelength import _patch_plc_wirelength

def _exact_proxy(placement: torch.Tensor, benchmark: Benchmark, plc) -> float:
    """Fast proxy cost: skips overlap metrics, skips unchanged macro updates,
    and uses the vectorized wirelength patch installed on plc.

    Bypasses macro_place.objective.compute_proxy_cost entirely. We never
    consume overlap metrics here; the placer only reads proxy_cost. Saves
    O(n_hard²) pure-Python pair iterations per scoring call (e.g. ~289k on
    ibm17) plus the redundant per-pin set_pos overhead on unchanged macros.
    """
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, benchmark)
    _patch_plc_density(plc, benchmark)
    placement_np = placement.cpu().numpy()
    # The placement tensor carries both hard and soft macro positions.
    _fast_set_placement(plc, placement_np, benchmark)
    wl = plc.get_cost()
    dens = plc.get_density_cost()
    cong = plc.get_congestion_cost()
    return float(wl + 0.5 * dens + 0.5 * cong)


def _proxy_decomp(placement: torch.Tensor, benchmark: Benchmark, plc):
    """(proxy, wl, 0.5*den, 0.5*cong) - the WEIGHTED proxy split. Re-scores the
    placement (mutates plc state), so use only in diagnostic contexts."""
    p = _exact_proxy(placement, benchmark, plc)
    wl = float(plc.get_cost())
    den = 0.5 * float(plc.get_density_cost())
    cong = 0.5 * float(plc.get_congestion_cost())
    return p, wl, den, cong
