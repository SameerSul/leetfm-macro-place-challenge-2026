"""Hot/cold cell fields shared by the relocation and swap move passes."""

import numpy as np

from utils import constants as const


def _congestion_field(source, nr: int, nc: int):
    """max(H, V) routing congestion as an (nr, nc) grid, or None if unavailable."""
    if const.USE_SCORER_CONGESTION_FIELD:
        scorer_field = getattr(source, "congestion_field", None)
        if scorer_field is not None:
            try:
                field = scorer_field()
            except Exception:
                field = None
            if field is not None and field.shape == (nr, nc):
                return field
    else:
        scorer_field = None
    if scorer_field is not None:
        try:
            field = scorer_field()
        except Exception:
            field = None
        if field is not None and field.shape == (nr, nc):
            return field
    plc = getattr(source, "plc", source)
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


def weighted_congestion_field(source, nr: int, nc: int):
    """Congestion-heavy proposal field used only for candidate ranking."""
    cong = _congestion_field(source, nr, nc)
    if cong is None:
        return None
    dens = _density_field(source, nr, nc)
    cong = np.asarray(cong, dtype=np.float64)
    cong_norm = cong / max(float(np.max(cong)), 1e-12)
    if dens is None:
        dens_norm = np.zeros_like(cong_norm)
    else:
        dens = np.asarray(dens, dtype=np.float64)
        dens_norm = dens / max(float(np.max(dens)), 1e-12)
    return (
        float(const.HIER_PROPOSAL_CONGESTION_WEIGHT) * cong_norm
        + float(const.HIER_PROPOSAL_DENSITY_WEIGHT) * dens_norm
    )


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
