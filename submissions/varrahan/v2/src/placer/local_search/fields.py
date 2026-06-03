"""Hot/cold cell fields shared by the relocation and swap move passes."""

import numpy as np

def _congestion_field(plc, nr: int, nc: int):
    """max(H, V) routing congestion as an (nr, nc) grid, or None if unavailable."""
    try:
        h_arr = np.asarray(plc.get_horizontal_routing_congestion(), dtype=np.float64)
        v_arr = np.asarray(plc.get_vertical_routing_congestion(), dtype=np.float64)
    except Exception:
        return None
    if h_arr.size != nr * nc or v_arr.size != nr * nc:
        return None
    return np.maximum(h_arr.reshape(nr, nc), v_arr.reshape(nr, nc))


def _density_field(incremental_scorer, nr: int, nc: int):
    """Occupancy density as an (nr, nc) grid, or None if unavailable."""
    go = getattr(incremental_scorer, "grid_occupied", None)
    if go is None or go.size != nr * nc:
        return None
    return (go / incremental_scorer.dens_grid_area).reshape(nr, nc)
