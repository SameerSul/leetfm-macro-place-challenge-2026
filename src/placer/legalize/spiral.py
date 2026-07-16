"""Vectorized greedy spiral legalizer."""

from functools import lru_cache
import time

import numpy as np

from placer.shared.geometry import separation_matrices
from utils.config import HAS_NUMBA, _numba_njit

if HAS_NUMBA:

    @_numba_njit(cache=True, fastmath=False)
    def _legalize_one_jit(pos, legal, placed, sizes, hw, hh, cw, ch, idx):
        """Find one macro's lex-first minimum-displacement legal position."""
        eps = 0.05
        placed_any = False
        in_bounds = (
            legal[idx, 0] - hw[idx] >= 0.0
            and legal[idx, 0] + hw[idx] <= cw
            and legal[idx, 1] - hh[idx] >= 0.0
            and legal[idx, 1] + hh[idx] <= ch
        )
        if in_bounds:
            conflict = False
            for other in range(placed.shape[0]):
                if not placed[other]:
                    continue
                placed_any = True
                sep_x = (sizes[idx, 0] + sizes[other, 0]) / 2.0
                sep_y = (sizes[idx, 1] + sizes[other, 1]) / 2.0
                if (
                    abs(legal[idx, 0] - legal[other, 0]) < sep_x + eps
                    and abs(legal[idx, 1] - legal[other, 1]) < sep_y + eps
                ):
                    conflict = True
                    break
            if placed_any and not conflict:
                return legal[idx, 0], legal[idx, 1]

        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        px = pos[idx, 0]
        py = pos[idx, 1]
        hw_idx = hw[idx]
        hh_idx = hh[idx]
        best_x = min(max(legal[idx, 0], hw_idx), cw - hw_idx)
        best_y = min(max(legal[idx, 1], hh_idx), ch - hh_idx)

        for radius in range(1, 200):
            found = False
            best_d2 = np.inf
            ring_x = best_x
            ring_y = best_y
            for ddx in range(-radius, radius + 1):
                for ddy in range(-radius, radius + 1):
                    if abs(ddx) != radius and abs(ddy) != radius:
                        continue
                    cand_x = min(max(px + ddx * step, hw_idx), cw - hw_idx)
                    cand_y = min(max(py + ddy * step, hh_idx), ch - hh_idx)
                    conflict = False
                    for other in range(placed.shape[0]):
                        if not placed[other]:
                            continue
                        sep_x = (sizes[idx, 0] + sizes[other, 0]) / 2.0
                        sep_y = (sizes[idx, 1] + sizes[other, 1]) / 2.0
                        if (
                            abs(cand_x - legal[other, 0]) < sep_x + eps
                            and abs(cand_y - legal[other, 1]) < sep_y + eps
                        ):
                            conflict = True
                            break
                    if conflict:
                        continue
                    diff_x = cand_x - pos[idx, 0]
                    diff_y = cand_y - pos[idx, 1]
                    d2 = diff_x * diff_x + diff_y * diff_y
                    if d2 < best_d2:
                        best_d2 = d2
                        ring_x = cand_x
                        ring_y = cand_y
                        found = True
            if found:
                return ring_x, ring_y
        return best_x, best_y


@lru_cache(maxsize=256)
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
        out = np.array([[0, 0]], dtype=np.int64)
        out.setflags(write=False)
        return out
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
    out = np.stack(
        [
            np.concatenate([e1_ddx, mid_ddx, e2_ddx]),
            np.concatenate([e1_ddy, mid_ddy, e2_ddy]),
        ],
        axis=1,
    )
    out.setflags(write=False)
    return out


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
    if order is None:
        order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()

    if HAS_NUMBA:
        for idx in order:
            if deadline is not None and time.monotonic() > deadline:
                break
            if not movable[idx]:
                placed[idx] = True
                continue
            legal[idx] = _legalize_one_jit(pos, legal, placed, sizes, hw, hh, cw, ch, idx)
            placed[idx] = True
        return legal

    sep_x_mat, sep_y_mat = separation_matrices(sizes)
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

        # In-place accept requires the macro be BOTH non-conflicting AND in
        # bounds. A far-OOB seed macro sitting in empty space has no conflict, so
        # without the bounds gate it was accepted at its illegal position and the
        # spiral search (which clips to canvas) was never reached — leaving OOB
        # seed macros uncorrected. Forcing OOB macros into the spiral pulls them
        # inside; in-bounds macros are unaffected, so the normal output is unchanged.
        in_bounds = (
            legal[idx, 0] - hw[idx] >= 0.0
            and legal[idx, 0] + hw[idx] <= cw
            and legal[idx, 1] - hh[idx] >= 0.0
            and legal[idx, 1] + hh[idx] <= ch
        )
        # Current-position conflict check (only over actually-placed macros).
        # When no macros are placed yet, fall through to spiral search to match
        # the prior behavior of always moving the first movable macro by 1 step.
        if in_bounds and placed.any():
            cdx = np.abs(legal[idx, 0] - legal[placed, 0])
            cdy = np.abs(legal[idx, 1] - legal[placed, 1])
            if not ((cdx < sep_x_idx[placed] + EPS) & (cdy < sep_y_idx[placed] + EPS)).any():
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
        # Fallback (used only if all rings fail) clamped in-bounds, so a macro the
        # spiral can't seat never lands outside the canvas.
        best = legal[idx].copy()
        best[0] = min(max(float(best[0]), hw_idx), cw - hw_idx)
        best[1] = min(max(float(best[1]), hh_idx), ch - hh_idx)

        for r in range(1, MAX_R):
            ring = _ring_offsets(r)
            cand_x = np.clip(px + ring[:, 0] * step, hw_idx, cw - hw_idx)
            cand_y = np.clip(py + ring[:, 1] * step, hh_idx, ch - hh_idx)
            if placed_x.size > 0:
                # [K, P] overlap test in one numpy op
                dx_mat = np.abs(cand_x[:, None] - placed_x[None, :])
                dy_mat = np.abs(cand_y[:, None] - placed_y[None, :])
                bad = ((dx_mat < sep_xp[None, :] + EPS) & (dy_mat < sep_yp[None, :] + EPS)).any(
                    axis=1
                )
                valid = ~bad
            else:
                valid = np.ones(len(cand_x), dtype=bool)
            if not valid.any():
                continue
            # Keep tie behavior stable by scoring in the input dtype.
            diff_x = cand_x.astype(pos.dtype, copy=False) - pos[idx, 0]
            diff_y = cand_y.astype(pos.dtype, copy=False) - pos[idx, 1]
            d2 = diff_x * diff_x + diff_y * diff_y
            best_local = int(np.argmin(np.where(valid, d2, np.inf)))
            best = np.array([cand_x[best_local], cand_y[best_local]])
            break

        legal[idx] = best
        placed[idx] = True
    return legal
