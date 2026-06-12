"""Local displacement-reducing hard-macro swap pass."""

import time

import numpy as np

from placer.geometry import separation_matrices


def _two_opt_swap(
    legal_pos: np.ndarray,
    init_pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    n: int,
    k_neighbors: int = 5,
    max_iters: int = 3,
    deadline: float | None = None,
) -> "tuple[np.ndarray, int]":
    """Post-legalize 2-opt swap pass.

    `_will_legalize` is greedy and cannot backtrack: once macro A is placed,
    it cannot move to give macro B a closer slot. This 2-opt pass examines
    pairs of nearby movable macros and tries swapping their positions. A swap
    is accepted iff:
        (1) Both macros remain in canvas bounds at their new positions.
        (2) Neither macro conflicts with any OTHER placed macro at its new
            position (and they don't conflict with each other).
        (3) Total per-pair displacement from init_pos strictly decreases.

    Spatial scope: for each macro i, we consider only its k_neighbors nearest
    placed macros (by current legal position). Distant swaps would increase
    total displacement anyway, so this restriction is essentially free.

    Iterates until no improvement or max_iters reached. Each iter is O(n²·k)
    in vectorized numpy (k_neighbors=5, max_iters=3 → ~1-3s for n=760).

    Returns (new_pos, swap_count).
    """
    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    EPS = 0.05

    pos = legal_pos.copy()
    # Per-macro squared displacement from initial. We use squared (not L2) so
    # the strict-improvement check `d_new < d_old - 1e-9` is exact in float64.
    disp_sq = (pos[:, 0] - init_pos[:, 0]) ** 2 + (pos[:, 1] - init_pos[:, 1]) ** 2

    swap_count = 0
    for it in range(max_iters):
        if deadline is not None and time.monotonic() > deadline:
            break
        improved_any = False

        # For each macro i, try swapping with its K nearest movable peers. kNN is
        # re-derived per outer iter since positions (and the neighborhood) change.
        # Pairwise sq distances: O(n²) memory, fine for n<=800.
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        d_pair = dx * dx + dy * dy
        np.fill_diagonal(d_pair, np.inf)
        # Mask non-movable rows/cols to inf so they're never selected as neighbors.
        non_movable = ~movable
        d_pair[non_movable, :] = np.inf
        d_pair[:, non_movable] = np.inf
        # kNN per row: indices of K smallest entries.
        # argpartition is O(n) per row, faster than argsort.
        k_eff = min(k_neighbors, n - 1)
        if k_eff <= 0:
            break
        neighbors = np.argpartition(d_pair, k_eff, axis=1)[:, :k_eff]

        for i in range(n):
            if not movable[i]:
                continue
            if deadline is not None and time.monotonic() > deadline:
                break
            for j in neighbors[i]:
                if not movable[j] or i == j:
                    continue
                # Tentative swap: i moves to pos[j], j moves to pos[i].
                new_ix, new_iy = pos[j, 0], pos[j, 1]
                new_jx, new_jy = pos[i, 0], pos[i, 1]

                # Bounds check (strict: the evaluator has zero overhang tolerance).
                if (new_ix - hw[i] < 0 or new_ix + hw[i] > cw or
                        new_iy - hh[i] < 0 or new_iy + hh[i] > ch):
                    continue
                if (new_jx - hw[j] < 0 or new_jx + hw[j] > cw or
                        new_jy - hh[j] < 0 or new_jy + hh[j] > ch):
                    continue

                # Displacement check - strict improvement only.
                d_i_new = (new_ix - init_pos[i, 0]) ** 2 + (new_iy - init_pos[i, 1]) ** 2
                d_j_new = (new_jx - init_pos[j, 0]) ** 2 + (new_jy - init_pos[j, 1]) ** 2
                if d_i_new + d_j_new >= disp_sq[i] + disp_sq[j] - 1e-9:
                    continue

                # Conflict check: i at new pos vs all macros except i,j.
                # Build a mask excluding i and j.
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                ox = pos[mask, 0]
                oy = pos[mask, 1]
                sxi = sep_x_mat[i, mask]
                syi = sep_y_mat[i, mask]
                conf_i = ((np.abs(new_ix - ox) < sxi + EPS) &
                          (np.abs(new_iy - oy) < syi + EPS)).any()
                if conf_i:
                    continue
                sxj = sep_x_mat[j, mask]
                syj = sep_y_mat[j, mask]
                conf_j = ((np.abs(new_jx - ox) < sxj + EPS) &
                          (np.abs(new_jy - oy) < syj + EPS)).any()
                if conf_j:
                    continue
                # i vs j: symmetric to the original legal separation, but verify
                # defensively.
                if (abs(new_ix - new_jx) < sep_x_mat[i, j] + EPS and
                        abs(new_iy - new_jy) < sep_y_mat[i, j] + EPS):
                    continue

                # Accept swap.
                pos[i, 0], pos[i, 1] = new_ix, new_iy
                pos[j, 0], pos[j, 1] = new_jx, new_jy
                disp_sq[i] = d_i_new
                disp_sq[j] = d_j_new
                improved_any = True
                swap_count += 1
                break  # move to next i (positions changed; further j checks stale)

        if not improved_any:
            break

    return pos, swap_count


