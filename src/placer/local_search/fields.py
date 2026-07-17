"""Hot/cold cell fields shared by the relocation and swap move passes."""

import numpy as np

from utils import constants as const


def _congestion_field(source, nr: int, nc: int):
    """max(H, V) routing congestion as an (nr, nc) grid, or None if unavailable."""
    scorer_field = getattr(source, "congestion_field", None)
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


def cold_connected_component_target_pool(
    field,
    *,
    cold_percentile: float,
    max_components: int,
    min_cells: int,
    size_weight: float,
) -> dict[str, np.ndarray]:
    """Return cold connected-component cells plus per-cell component penalties."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return {"indices": np.zeros(0, dtype=np.int64), "penalty": np.zeros(0, dtype=np.float64)}
    min_cells = max(1, int(min_cells))
    components = cold_connected_components(
        arr,
        cold_percentile=float(cold_percentile),
        min_cells=min_cells,
    )
    if not components:
        idx = np.argsort(arr.ravel())[: max(min_cells, 1)].astype(np.int64, copy=False)
        return {"indices": idx, "penalty": np.zeros(idx.size, dtype=np.float64)}

    max_size = max(int(comp["size"]) for comp in components)
    f_min = float(np.min(arr))
    f_span = max(float(np.max(arr)) - f_min, 1e-12)
    rows = []
    for comp in components:
        avg = float(comp["avg"])
        size = int(comp["size"])
        idx = np.asarray(comp["indices"], dtype=np.int64)
        field_norm = (float(avg) - f_min) / f_span
        size_norm = float(size) / max(float(max_size), 1.0)
        penalty = (1.0 - float(size_weight)) * field_norm + float(size_weight) * (1.0 - size_norm)
        rows.append((float(penalty), -int(size), float(avg), idx))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    rows = rows[: max(1, int(max_components))]
    indices = np.concatenate([row[3] for row in rows]).astype(np.int64, copy=False)
    penalties = np.concatenate(
        [np.full(row[3].size, float(row[0]), dtype=np.float64) for row in rows]
    )
    order = np.argsort(penalties + 1e-9 * arr.ravel()[indices], kind="stable")
    return {"indices": indices[order], "penalty": penalties[order]}


def cold_connected_components(
    field,
    *,
    cold_percentile: float,
    min_cells: int,
) -> list[dict[str, object]]:
    """Return 4-neighbor connected components below a field percentile."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return []
    nr, nc = arr.shape
    threshold = float(np.percentile(arr, float(cold_percentile)))
    cold = arr <= threshold
    seen = np.zeros_like(cold, dtype=bool)
    out: list[dict[str, object]] = []
    min_cells = max(1, int(min_cells))
    flat = arr.ravel()
    for r0 in range(nr):
        for c0 in range(nc):
            if seen[r0, c0] or not cold[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            rows = []
            cols = []
            while stack:
                r, c = stack.pop()
                rows.append(r)
                cols.append(c)
                cells.append(r * nc + c)
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                        continue
                    if seen[rr, cc] or not cold[rr, cc]:
                        continue
                    seen[rr, cc] = True
                    stack.append((rr, cc))
            if len(cells) < min_cells:
                continue
            idx = np.asarray(cells, dtype=np.int64)
            row_arr = np.asarray(rows, dtype=np.int64)
            col_arr = np.asarray(cols, dtype=np.int64)
            out.append(
                {
                    "indices": idx,
                    "rows": row_arr,
                    "cols": col_arr,
                    "size": int(idx.size),
                    "avg": float(np.mean(flat[idx])),
                    "r0": int(np.min(row_arr)),
                    "r1": int(np.max(row_arr)),
                    "c0": int(np.min(col_arr)),
                    "c1": int(np.max(col_arr)),
                    "centroid_r": float(np.mean(row_arr)),
                    "centroid_c": float(np.mean(col_arr)),
                }
            )
    out.sort(key=lambda comp: (float(comp["avg"]), -int(comp["size"])))
    return out
