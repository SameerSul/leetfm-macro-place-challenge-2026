"""Congestion-gradient placement perturbations."""

import numpy as np

def _routing_congestion_perturb(
    pos: np.ndarray,
    plc,
    benchmark: Benchmark,
    n: int,
    cw: float,
    ch: float,
    hw: np.ndarray,
    hh: np.ndarray,
    movable: np.ndarray,
    frac: float = 0.04,
    rng: np.random.RandomState | None = None,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """Step macros away from high routing-congestion cells.

    Uses the real H/V routing congestion map plc holds after the last
    get_congestion_cost() call. For each macro in a congested cell, step against
    the finite-difference gradient of the congestion field (toward lower
    congestion) plus a small random component to break symmetry. A private rng
    leaves the global numpy state untouched so noise restarts stay reproducible.

    top_k: if set, perturb only the K movable macros with the highest local
    congestion among those above the threshold (None = all qualifying).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    nr, nc_grid = benchmark.grid_rows, benchmark.grid_cols
    try:
        h_list = list(plc.get_horizontal_routing_congestion())
        v_list = list(plc.get_vertical_routing_congestion())
    except Exception:
        return pos.copy()
    if len(h_list) != nr * nc_grid:
        return pos.copy()

    # Per-cell congestion field: max(H,V) (beat both H+V and per-axis in A/B -
    # gradient sharpness matters more than field semantics).
    cell_cong = np.maximum(
        np.asarray(h_list).reshape(nr, nc_grid),
        np.asarray(v_list).reshape(nr, nc_grid),
    )

    cell_w = cw / nc_grid
    cell_h = ch / nr
    scale = frac * min(cw, ch)
    cong_threshold = 0.5

    # Per-macro cell indices and local congestion (vectorized over all n macros)
    c_idx_all = np.minimum((pos[:n, 0] / cell_w).astype(np.int64), nc_grid - 1)
    r_idx_all = np.minimum((pos[:n, 1] / cell_h).astype(np.int64), nr - 1)
    local_cong_all = cell_cong[r_idx_all, c_idx_all]

    # Qualifying macros: movable AND in a congested cell. The boolean mask over
    # 0..n-1 preserves per-i order so the rng draw sequence is reproducible.
    mask = movable.astype(bool) & (local_cong_all >= cong_threshold)
    perturbed = pos.copy()
    if not mask.any():
        return perturbed

    # Optional TOP-K filter: keep only the K qualifying macros with the highest
    # local congestion (used by Phase 8; top_k=None keeps every qualifying macro).
    if top_k is not None and int(mask.sum()) > top_k:
        qual_indices = np.where(mask)[0]
        qual_cong = local_cong_all[qual_indices]
        # Negative for argpartition: pick the top_k LARGEST values.
        top_pos_in_qual = np.argpartition(-qual_cong, top_k - 1)[:top_k]
        focused_mask = np.zeros_like(mask)
        focused_mask[qual_indices[top_pos_in_qual]] = True
        mask = focused_mask

    r_idx = r_idx_all[mask]
    c_idx = c_idx_all[mask]
    local_cong = local_cong_all[mask]

    # Bounds-safe neighbor lookups for the finite-difference gradient
    c_left = np.maximum(c_idx - 1, 0)
    c_right = np.minimum(c_idx + 1, nc_grid - 1)
    r_down = np.maximum(r_idx - 1, 0)
    r_up = np.minimum(r_idx + 1, nr - 1)

    grad_x = (cell_cong[r_idx, c_right] - cell_cong[r_idx, c_left]) / 2.0
    grad_y = (cell_cong[r_up, c_idx] - cell_cong[r_down, c_idx]) / 2.0
    grad_len = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-10

    # move_scale is linear in local_cong (no cap - capping strangled motion on
    # the cong>2 hotspots descent most wants to clear). noise draws in C order
    # to match the original per-macro (dx, dy) sequence, keeping rng reproducible.
    move_scale = scale * local_cong
    noise = rng.normal(0.0, scale * 0.1, size=(int(mask.sum()), 2))
    dx = -(grad_x / grad_len) * move_scale + noise[:, 0]
    dy = -(grad_y / grad_len) * move_scale + noise[:, 1]

    perturbed[mask, 0] = np.clip(pos[mask, 0] + dx, hw[mask], cw - hw[mask])
    perturbed[mask, 1] = np.clip(pos[mask, 1] + dy, hh[mask], ch - hh[mask])

    return perturbed


