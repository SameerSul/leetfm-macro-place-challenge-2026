"""Placement position-cache helpers."""

import numpy as np
from macro_place.benchmark import Benchmark

def _ensure_pos_cache(plc) -> np.ndarray:
    """Maintain a per-module (x, y) position cache for vectorized scoring.

    Lets scoring read positions via fancy indexing instead of per-call
    `mods[idx].get_pos()` loops. Built once (O(n_modules) get_pos calls),
    updated in-place by `_fast_set_placement`. Returns a (n_modules, 2)
    float64 array indexed by `plc.modules_w_pins` index.
    """
    cache = getattr(plc, "_global_pos_cache", None)
    if cache is None:
        mods = plc.modules_w_pins
        cache = np.empty((len(mods), 2), dtype=np.float64)
        for k, m in enumerate(mods):
            x, y = m.get_pos()
            cache[k, 0] = x
            cache[k, 1] = y
        plc._global_pos_cache = cache
    return cache



def _fast_set_placement(plc, placement_np: np.ndarray, benchmark: Benchmark) -> None:
    """Faster drop-in for objective._set_placement.

    Three wins vs the reference:
      1. Cache last-applied positions per macro on plc and SKIP set_pos
         when the value matches. Soft macros almost never move after the
         baseline restoration; this collapses thousands of no-op calls per
         score into a single equality check per macro.
      2. Skip pin.set_pos entirely. Verified that every cost path in
         plc_client_os recomputes pin coordinates via __get_pin_position
         (ref_node.get_pos() + pin.get_offset()) - nothing reads pin.x/.y.
         The pin.set_pos calls were dead code defending against a non-issue.
      3. Skip the overlap-metric computation downstream (we never read it).
    """
    n_hard = benchmark.num_hard_macros
    hard_indices = benchmark.hard_macro_indices
    soft_indices = benchmark.soft_macro_indices

    last = getattr(plc, "_last_pos_cache", None)
    if last is None or last.shape != placement_np.shape:
        last = np.full(placement_np.shape, np.nan, dtype=np.float64)
        plc._last_pos_cache = last

    # Keep the global pos cache synchronized with each set_pos call.
    pos_cache = _ensure_pos_cache(plc)

    any_changed = False

    # Hard macros
    for i, macro_idx in enumerate(hard_indices):
        x = float(placement_np[i, 0])
        y = float(placement_np[i, 1])
        if last[i, 0] == x and last[i, 1] == y:
            continue
        any_changed = True
        last[i, 0] = x
        last[i, 1] = y
        plc.modules_w_pins[macro_idx].set_pos(x, y)
        pos_cache[macro_idx, 0] = x
        pos_cache[macro_idx, 1] = y

    # Soft macros - usually unchanged after baseline; the equality check
    # short-circuits the per-macro work for the common no-op case.
    for i, macro_idx in enumerate(soft_indices):
        row = n_hard + i
        x = float(placement_np[row, 0])
        y = float(placement_np[row, 1])
        if last[row, 0] == x and last[row, 1] == y:
            continue
        any_changed = True
        last[row, 0] = x
        last[row, 1] = y
        plc.modules_w_pins[macro_idx].set_pos(x, y)
        pos_cache[macro_idx, 0] = x
        pos_cache[macro_idx, 1] = y

    from placer.scoring.congestion import _ensure_congestion_arrays

    _ensure_congestion_arrays(plc)
    # Only invalidate cached costs if something actually moved. If nothing
    # changed, plc's dirty flags can stay False and the cached values are
    # returned for free.
    if any_changed:
        plc.FLAG_UPDATE_WIRELENGTH = True
        plc.FLAG_UPDATE_DENSITY = True
        plc.FLAG_UPDATE_CONGESTION = True
