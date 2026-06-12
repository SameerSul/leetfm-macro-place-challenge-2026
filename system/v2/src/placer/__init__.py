"""Varrahan v2 macro placer package.

Besides `MacroPlacer`, this re-exports the internal scoring / routing /
legalization helpers at the top level so diagnostics and verifiers can use the
flat `from placer import _exact_proxy, ...` namespace the pre-refactor monolith
exposed.
"""

# Import the pipeline first: it pulls in every leaf module in a known-good order
# and fully initializes them, so the flat re-exports below never hit a
# partially-initialized module (the routing<->scoring imports are circular if
# entered in the wrong order).
from .pipeline.macro_placer import MacroPlacer

from .legalize.spiral import _ring_offsets, _will_legalize
from .plc.loader import _load_plc
from .plc.placement import _ensure_pos_cache, _fast_set_placement
from .routing.apply import (
    _apply_2pin_routing,
    _apply_3pin_routing_vec,
    _apply_macro_routing,
    _apply_macro_routing_subset,
    _apply_net_routing_struct,
    _apply_net_routing_subset,
    _build_cong_cache,
    _build_net_routing_struct,
    _smooth_routing_cong_vec,
    _vectorized_get_routing,
)
from .scoring.congestion import (
    _ensure_congestion_arrays,
    _patch_plc_congestion,
    _vectorized_get_congestion_cost,
)
from .scoring.density import _patch_plc_density, _vectorized_get_grid_cells_density
from .scoring.exact import _exact_proxy
from .scoring.incremental import IncrementalScorer
from .scoring.wirelength import _build_wl_cache, _patch_plc_wirelength

__all__ = [
    "MacroPlacer",
    "IncrementalScorer",
    "_apply_2pin_routing",
    "_apply_3pin_routing_vec",
    "_apply_macro_routing",
    "_apply_macro_routing_subset",
    "_apply_net_routing_struct",
    "_apply_net_routing_subset",
    "_build_cong_cache",
    "_build_net_routing_struct",
    "_build_wl_cache",
    "_ensure_congestion_arrays",
    "_ensure_pos_cache",
    "_exact_proxy",
    "_fast_set_placement",
    "_load_plc",
    "_patch_plc_congestion",
    "_patch_plc_density",
    "_patch_plc_wirelength",
    "_ring_offsets",
    "_smooth_routing_cong_vec",
    "_vectorized_get_congestion_cost",
    "_vectorized_get_grid_cells_density",
    "_vectorized_get_routing",
    "_will_legalize",
]
