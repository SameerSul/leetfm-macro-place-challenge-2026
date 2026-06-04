"""Vectorized greedy spiral legalizer."""

import time

import numpy as np

from placer.geometry import separation_matrices


def _ring_offsets(r: int) -> np.ndarray:
    """Offsets (ddx, ddy) on the spiral ring at radius r, in the same lex
    order as the original nested-loop traversal: for ddx in -r..r, for ddy in
    -r..r if (|ddx|=r or |ddy|=r). Returns a [K, 2] int64 array (K = 8r for
    r>=1, K=1 for r=0).

    Lex order matters: `np.argmin` returns the first-occurrence index of the
    minimum, so on ties this matches the original `if d < best_d` strict
    less-than semantics that kept the lex-first candidate.
    """
    if r == 0:
        return np.array([[0, 0]], dtype=np.int64)
    # Left edge: ddx = -r, ddy in [-r, r]
    e1_ddx = np.full(2 * r + 1, -r, dtype=np.int64)
    e1_ddy = np.arange(-r, r + 1, dtype=np.int64)
    # Middle columns: ddx in (-r, r), ddy in {-r, +r} interleaved per ddx
    mid_range = np.arange(-r + 1, r, dtype=np.int64)  # length 2r-1
    mid_ddx = np.repeat(mid_range, 2)
    mid_ddy = np.tile(np.array([-r, r], dtype=np.int64), len(mid_range))
    # Right edge: ddx = +r, ddy in [-r, r]
    e2_ddx = np.full(2 * r + 1, r, dtype=np.int64)
    e2_ddy = np.arange(-r, r + 1, dtype=np.int64)
    return np.stack(
        [
            np.concatenate([e1_ddx, mid_ddx, e2_ddx]),
            np.concatenate([e1_ddy, mid_ddy, e2_ddy]),
        ],
        axis=1,
    )


def _will_legalize(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
    deadline: float | None = None,
    order: list | None = None,
) -> np.ndarray:
    """Min-displacement legalization with configurable placement order.

    Places macros one by one at the nearest overlap-free position to their
    target via expanding spiral search; non-movable macros are fixed first.
    Per ring, all K candidates are tested against placed macros in one [K, P]
    conflict matrix. With _ring_offsets' lex order + np.argmin first-occurrence,
    the output is bit-equivalent to the original nested-loop version.

    order: placement sequence (None = largest-area-first); different orders
    explore different legal arrangements.
    deadline: optional time.monotonic() cutoff; remaining macros keep pos[].
    """
    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    if order is None:
        order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    MAX_R = 200
    EPS = 0.05  # separation tolerance, mirrors the original `+ 0.05` constant

    for idx in order:
        if deadline is not None and time.monotonic() > deadline:
            break
        if not movable[idx]:
            placed[idx] = True
            continue

        sep_x_idx = sep_x_mat[idx]
        sep_y_idx = sep_y_mat[idx]

        # Current-position conflict check (only over actually-placed macros).
        # When no macros are placed yet, fall through to spiral search to match
        # the prior behavior of always moving the first movable macro by 1 step.
        if placed.any():
            cdx = np.abs(legal[idx, 0] - legal[placed, 0])
            cdy = np.abs(legal[idx, 1] - legal[placed, 1])
            if not (
                (cdx < sep_x_idx[placed] + EPS) & (cdy < sep_y_idx[placed] + EPS)
            ).any():
                placed[idx] = True
                continue

        # Spiral search
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        px = float(pos[idx, 0])
        py = float(pos[idx, 1])
        hw_idx = float(hw[idx])
        hh_idx = float(hh[idx])
        placed_x = legal[placed, 0]
        placed_y = legal[placed, 1]
        sep_xp = sep_x_idx[placed]
        sep_yp = sep_y_idx[placed]
        best = legal[idx].copy()

        for r in range(1, MAX_R):
            ring = _ring_offsets(r)
            cand_x = np.clip(px + ring[:, 0] * step, hw_idx, cw - hw_idx)
            cand_y = np.clip(py + ring[:, 1] * step, hh_idx, ch - hh_idx)
            if placed_x.size > 0:
                # [K, P] overlap test in one numpy op
                dx_mat = np.abs(cand_x[:, None] - placed_x[None, :])
                dy_mat = np.abs(cand_y[:, None] - placed_y[None, :])
                bad = (
                    (dx_mat < sep_xp[None, :] + EPS)
                    & (dy_mat < sep_yp[None, :] + EPS)
                ).any(axis=1)
                valid = ~bad
            else:
                valid = np.ones(len(cand_x), dtype=bool)
            if not valid.any():
                continue
            # argmin first-occurrence → "first improvement wins". CRITICAL:
            # compute d² in pos.dtype, not float64 - on float32 pos the lower
            # precision breaks ties between symmetric candidates differently, and
            # matching it is required for bit-equivalence with sameer_v1.
            diff_x = cand_x.astype(pos.dtype, copy=False) - pos[idx, 0]
            diff_y = cand_y.astype(pos.dtype, copy=False) - pos[idx, 1]
            d2 = diff_x * diff_x + diff_y * diff_y
            best_local = int(np.argmin(np.where(valid, d2, np.inf)))
            best = np.array([cand_x[best_local], cand_y[best_local]])
            break

        legal[idx] = best
        placed[idx] = True
    return legal


