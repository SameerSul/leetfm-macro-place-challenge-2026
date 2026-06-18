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


def coldest_window_anchor(
    field, nr: int, nc: int, cw: float, ch: float, win_cells: int
) -> "tuple[float, float]":
    """Center (in microns) of the lowest-average `win_cells`-square window.

    Finds the grid window with the least congestion/density (the coldspot with
    routing headroom) via a 2D integral image, and returns its center cell mapped
    to microns. Used to anchor a coldspot-aware cluster kick. Falls back to the
    globally coldest cell when the window is larger than the grid.
    """
    w = int(max(1, min(win_cells, nr, nc)))
    R, C = nr - w + 1, nc - w + 1
    if R <= 0 or C <= 0:
        r, c = divmod(int(np.argmin(field)), nc)
    else:
        integ = np.zeros((nr + 1, nc + 1), dtype=np.float64)
        integ[1:, 1:] = np.cumsum(np.cumsum(field, axis=0), axis=1)
        win_sum = (
            integ[w : w + R, w : w + C]
            - integ[0:R, w : w + C]
            - integ[w : w + R, 0:C]
            + integ[0:R, 0:C]
        )
        r0, c0 = divmod(int(np.argmin(win_sum)), C)
        r, c = r0 + w // 2, c0 + w // 2
    return (c + 0.5) * (cw / nc), (r + 0.5) * (ch / nr)
