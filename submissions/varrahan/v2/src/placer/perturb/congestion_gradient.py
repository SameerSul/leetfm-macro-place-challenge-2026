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
    """
    Move macros away from high routing-congestion cells using the ACTUAL
    routing congestion map stored in plc after the last get_congestion_cost() call.

    Unlike _density_gradient_perturb (which uses a macro-count occupancy proxy),
    this uses the real H/V routing congestion computed by PlacementCost.

    Gradient: for each macro in a congested cell, compute finite-difference
    gradient of congestion w.r.t. position, then move the macro AGAINST the
    gradient (toward lower congestion). A small random component breaks symmetry.

    Uses a separate rng (not np.random) so the main random state is unchanged
    and subsequent noise restarts get identical draws to before.

    top_k (A6 attack #1, 2026-05-23): if set, restrict perturbation to the
    K movable macros with HIGHEST local congestion (out of all those above
    `cong_threshold`). Original behavior (top_k=None) moves every qualifying
    macro. Rationale per A3 diagnostic: DP loses uniformly on congestion by
    ~+0.08 on average; our cong-grad currently spreads motion across all
    congested macros, possibly blunting the gradient. TOP-K focuses motion
    on the hottest cells where it should matter most.
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

    # Per-cell congestion field for the gradient. A/B tested 2026-05-21:
    #   - H+V (objective-aligned): ibm03 −0.021 regression, ibm07 −0.004
    #   - per-axis (V→x grad, H→y grad): ibm03/ibm04 +0.011 regressions
    # max(H,V) wins both A/Bs. What seems to matter is gradient SHARPNESS,
    # not field semantics: H and V are spatially correlated (macros in dense
    # regions cause both), so decoupling them or summing them blunts the
    # per-cell "hot cell" peak that drives effective motion.
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

    # Macros that qualify for perturbation: movable AND in a congested cell.
    # RNG draws below are sequenced in qualifying-macro order (mask is a boolean
    # over 0..n-1 so np.where preserves the original per-i traversal order that
    # the prior scalar loop used).
    mask = movable.astype(bool) & (local_cong_all >= cong_threshold)
    perturbed = pos.copy()
    if not mask.any():
        return perturbed

    # A6 attack #1 (2026-05-23): TOP-K filter. Of all qualifying macros,
    # keep only the K with highest local congestion. Tested on top of the
    # full-mask baseline as a NEW candidate (Phase 8); the default
    # `top_k=None` preserves existing Phase 1/2/3/5b/5c/7 behavior.
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

    # rng.normal(size=(k, 2)) draws in C order: noise[0,0], noise[0,1], noise[1,0], ...
    # This matches the scalar loop's interleaved (dx-noise, dy-noise) pair-per-macro
    # draw order exactly, so rng_cong advances identically and downstream Phase 3
    # / noise restarts see the same RNG state as the pre-vectorization version.
    # Linear no-cap (was np.minimum(local_cong, 2.0)). A/B over 6 benchmarks:
    #   ibm04 −0.0080, ibm03 −0.0013, ibm10 +0.0010, ibm01/07/09 tied.
    # Net −0.0083; ibm10 regression is within run-to-run noise (smaller than
    # the smallest other win). Cap was strangling motion on cong>2 cells —
    # exactly the hotspots gradient descent wants to clear. Canvas-bounds
    # clip on the next two lines keeps motion in-bounds.
    move_scale = scale * local_cong
    noise = rng.normal(0.0, scale * 0.1, size=(int(mask.sum()), 2))
    dx = -(grad_x / grad_len) * move_scale + noise[:, 0]
    dy = -(grad_y / grad_len) * move_scale + noise[:, 1]

    perturbed[mask, 0] = np.clip(pos[mask, 0] + dx, hw[mask], cw - hw[mask])
    perturbed[mask, 1] = np.clip(pos[mask, 1] + dy, hh[mask], ch - hh[mask])

    return perturbed


