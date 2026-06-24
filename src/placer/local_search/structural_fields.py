"""Deterministic structural placement metrics.

These helpers are scale-normalized diagnostics and ranking signals. They do not
check legality and do not mutate placements.
"""

from __future__ import annotations

import numpy as np

from utils import constants as const
from utils.config import HAS_NUMBA, _numba_njit


def _notch_numba_enabled(n_idx: int, n_all: int) -> bool:
    if n_all <= 1:
        return False
    min_pairs = int(const.HIER_STRUCTURAL_NOTCH_NUMBA_MIN_PAIRS)
    return int(n_idx * n_all) >= max(min_pairs, 1)


def _notch_gpu_enabled(n_idx: int, n_all: int) -> bool:
    if n_all <= 1 or n_idx <= 0:
        return False
    if not const.HIER_STRUCTURAL_NOTCH_GPU:
        return False
    min_n = int(const.HIER_STRUCTURAL_NOTCH_GPU_MIN_N)
    return n_idx >= min_n and n_all >= min_n


if HAS_NUMBA:

    @_numba_njit(cache=True, fastmath=False)
    def _notch_penalty_numba(
        left: np.ndarray,
        right: np.ndarray,
        bottom: np.ndarray,
        top: np.ndarray,
        wx: np.ndarray,
        wy: np.ndarray,
        idx: np.ndarray,
        window: float,
    ) -> tuple[float, int]:
        m = idx.shape[0]
        n = left.shape[0]
        total = 0.0
        count = 0
        if m == 0 or n == 0:
            return total, count

        for a in range(m):
            i = int(idx[a])
            hw_i = wx[i] * 0.5
            hh_i = wy[i] * 0.5
            x_i = (left[i] + right[i]) * 0.5
            y_i = (bottom[i] + top[i]) * 0.5

            for j in range(n):
                if j == i:
                    continue
                x_gap1 = left[j] - (x_i + hw_i)
                x_gap2 = (x_i - hw_i) - right[j]
                x_gap = x_gap1 if x_gap1 > x_gap2 else x_gap2
                y_overlap = (top[i] if top[i] < top[j] else top[j]) - (
                    bottom[i] if bottom[i] > bottom[j] else bottom[j]
                )
                if x_gap > 0.0 and x_gap < window and y_overlap > 0.0:
                    min_h = wy[i] if wy[i] < wy[j] else wy[j]
                    denom = min_h if min_h > 1e-9 else 1e-9
                    val = ((window - x_gap) / window) ** 2 * (y_overlap / denom)
                    if val > 1.0:
                        val = ((window - x_gap) / window) ** 2
                    total += val
                    count += 1

                y_gap1 = bottom[j] - (y_i + hh_i)
                y_gap2 = (y_i - hh_i) - top[j]
                y_gap = y_gap1 if y_gap1 > y_gap2 else y_gap2
                if y_gap <= 0.0 or y_gap >= window:
                    continue
                x_overlap = (right[i] if right[i] < right[j] else right[j]) - (
                    left[i] if left[i] > left[j] else left[j]
                )
                if x_overlap > 0.0:
                    min_w = wx[i] if wx[i] < wx[j] else wx[j]
                    denom = min_w if min_w > 1e-9 else 1e-9
                    val = ((window - y_gap) / window) ** 2 * (x_overlap / denom)
                    if val > 1.0:
                        val = ((window - y_gap) / window) ** 2
                    total += val
                    count += 1
        return float(total), int(count)


def _notch_penalty_cpu_vectorized(
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    idx: np.ndarray,
    window: float,
) -> tuple[float, int]:
    idx = np.asarray(idx, dtype=np.int64)
    m = idx.size
    if m == 0:
        return 0.0, 0
    n = left.shape[0]

    left_i = left[idx][:, None]
    right_i = right[idx][:, None]
    bottom_i = bottom[idx][:, None]
    top_i = top[idx][:, None]
    widths_i = widths[idx][:, None]
    heights_i = heights[idx][:, None]

    left_all = left[None, :]
    right_all = right[None, :]
    bottom_all = bottom[None, :]
    top_all = top[None, :]
    widths_all = widths[None, :]
    heights_all = heights[None, :]

    x_gap = np.maximum.reduce(
        [
            left_all - right_i,
            left_i - right_all,
            np.zeros((m, n), dtype=np.float64),
        ]
    )
    y_gap = np.maximum.reduce(
        [
            bottom_all - top_i,
            bottom_i - top_all,
            np.zeros((m, n), dtype=np.float64),
        ]
    )
    x_overlap = np.minimum(right_i, right_all) - np.maximum(left_i, left_all)
    y_overlap = np.minimum(top_i, top_all) - np.maximum(bottom_i, bottom_all)

    valid_x = (x_gap > 0.0) & (x_gap < window) & (y_overlap > 0.0)
    valid_y = (y_gap > 0.0) & (y_gap < window) & (x_overlap > 0.0)
    if m == n:
        diag = np.arange(m)
        valid_x[diag, diag] = False
        valid_y[diag, diag] = False

    min_h = np.minimum(heights_i, heights_all)
    min_w = np.minimum(widths_i, widths_all)

    denom_h = np.maximum(min_h, 1e-9)
    denom_w = np.maximum(min_w, 1e-9)

    terms_x = np.zeros((m, n), dtype=np.float64)
    terms_y = np.zeros((m, n), dtype=np.float64)
    if valid_x.any():
        cover = np.minimum(1.0, y_overlap[valid_x] / denom_h[valid_x])
        terms_x[valid_x] = ((window - x_gap[valid_x]) / window) ** 2 * cover
    if valid_y.any():
        cover = np.minimum(1.0, x_overlap[valid_y] / denom_w[valid_y])
        terms_y[valid_y] = ((window - y_gap[valid_y]) / window) ** 2 * cover

    totals = terms_x + terms_y
    return float(np.sum(totals)), int(np.count_nonzero(totals > 0.0))


def _notch_penalty_torch(
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    idx: np.ndarray,
    window: float,
) -> tuple[float, int] | None:
    import torch

    from utils.config import _GPU_DEVICE

    if _GPU_DEVICE.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        dev = _GPU_DEVICE
        left_t = torch.as_tensor(left, dtype=torch.float64, device=dev)
        right_t = torch.as_tensor(right, dtype=torch.float64, device=dev)
        bottom_t = torch.as_tensor(bottom, dtype=torch.float64, device=dev)
        top_t = torch.as_tensor(top, dtype=torch.float64, device=dev)
        widths_t = torch.as_tensor(widths, dtype=torch.float64, device=dev)
        heights_t = torch.as_tensor(heights, dtype=torch.float64, device=dev)
        idx_t = torch.as_tensor(idx, dtype=torch.int64, device=dev)

        left_i = left_t[idx_t][:, None]
        right_i = right_t[idx_t][:, None]
        bottom_i = bottom_t[idx_t][:, None]
        top_i = top_t[idx_t][:, None]

        left_all = left_t[None, :]
        right_all = right_t[None, :]
        bottom_all = bottom_t[None, :]
        top_all = top_t[None, :]
        x_gap = torch.maximum(
            torch.maximum(left_all - right_i, left_i - right_all),
            torch.zeros((idx_t.shape[0], left_t.shape[0]), dtype=torch.float64, device=dev),
        )
        y_gap = torch.maximum(
            torch.maximum(bottom_all - top_i, bottom_i - top_all),
            torch.zeros((idx_t.shape[0], left_t.shape[0]), dtype=torch.float64, device=dev),
        )
        x_overlap = torch.minimum(right_i, right_all) - torch.maximum(left_i, left_all)
        y_overlap = torch.minimum(top_i, top_all) - torch.maximum(bottom_i, bottom_all)
        win = float(window)
        window_t = torch.tensor(win, dtype=torch.float64, device=dev)

        valid_x = (x_gap > 0.0) & (x_gap < window_t) & (y_overlap > 0.0)
        valid_y = (y_gap > 0.0) & (y_gap < window_t) & (x_overlap > 0.0)
        if idx_t.shape[0] == left_t.shape[0]:
            diag = torch.arange(idx_t.shape[0], device=dev)
            valid_x[diag, diag] = False
            valid_y[diag, diag] = False

        min_h = torch.minimum(heights_t[idx_t][:, None], heights_t[None, :])
        min_w = torch.minimum(widths_t[idx_t][:, None], widths_t[None, :])
        denom_h = torch.maximum(min_h, torch.tensor(1e-9, dtype=torch.float64, device=dev))
        denom_w = torch.maximum(min_w, torch.tensor(1e-9, dtype=torch.float64, device=dev))

        terms_x = torch.zeros_like(x_gap)
        terms_y = torch.zeros_like(y_gap)
        if torch.any(valid_x):
            cover = torch.minimum(
                torch.tensor(1.0, dtype=torch.float64, device=dev),
                y_overlap[valid_x] / denom_h[valid_x],
            )
            terms_x[valid_x] = ((window_t - x_gap[valid_x]) / window_t) ** 2 * cover
        if torch.any(valid_y):
            cover = torch.minimum(
                torch.tensor(1.0, dtype=torch.float64, device=dev),
                x_overlap[valid_y] / denom_w[valid_y],
            )
            terms_y[valid_y] = ((window_t - y_gap[valid_y]) / window_t) ** 2 * cover

        totals = terms_x + terms_y
        count = int(torch.count_nonzero(totals > 0.0).item())
        return float(torch.sum(totals).item()), count
    except Exception:
        return None


def _structural_notch_penalty_totals(
    positions: np.ndarray,
    sizes: np.ndarray,
    window: float,
    idx: np.ndarray,
) -> tuple[float, int]:
    n = positions.shape[0]
    idx = np.asarray(idx, dtype=np.int64)
    if n == 0 or idx.size == 0:
        return 0.0, 0

    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0
    left = positions[:, 0] - hw
    right = positions[:, 0] + hw
    bottom = positions[:, 1] - hh
    top = positions[:, 1] + hh
    if _notch_gpu_enabled(idx.size, n):
        out = _notch_penalty_torch(left, right, bottom, top, sizes[:, 0], sizes[:, 1], idx, window)
        if out is not None:
            return out

    if HAS_NUMBA and _notch_numba_enabled(idx.size, n):
        return _notch_penalty_numba(left, right, bottom, top, sizes[:, 0], sizes[:, 1], idx, window)

    return _notch_penalty_cpu_vectorized(
        left, right, bottom, top, sizes[:, 0], sizes[:, 1], idx, window
    )


def _as_pos_size(
    positions: np.ndarray,
    sizes: np.ndarray,
    indices: "np.ndarray | list[int] | None",
) -> "tuple[np.ndarray, np.ndarray]":
    pos = np.asarray(positions, dtype=np.float64)
    sz = np.asarray(sizes, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must have shape [N, 2]")
    if sz.ndim != 2 or sz.shape[1] != 2 or sz.shape[0] != pos.shape[0]:
        raise ValueError("sizes must have shape [N, 2] matching positions")
    if indices is None:
        return pos, sz
    idx = np.asarray(indices, dtype=np.int64)
    return pos[idx], sz[idx]


def _default_keepout(sizes: np.ndarray, canvas_width: float, canvas_height: float) -> float:
    if sizes.size == 0:
        return max(min(float(canvas_width), float(canvas_height)) * 0.02, 1e-9)
    macro_scale = float(np.median(np.minimum(sizes[:, 0], sizes[:, 1])))
    canvas_scale = min(float(canvas_width), float(canvas_height)) * 0.02
    return max(min(macro_scale, canvas_scale), 1e-9)


def edge_keepout_penalty(
    positions: np.ndarray,
    sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    *,
    keepout: "float | None" = None,
    indices: "np.ndarray | list[int] | None" = None,
) -> float:
    """Penalize movable blocks that sit inside a soft keepout band near edges."""
    pos, sz = _as_pos_size(positions, sizes, indices)
    if pos.shape[0] == 0:
        return 0.0
    cw, ch = float(canvas_width), float(canvas_height)
    ko = float(keepout) if keepout is not None else _default_keepout(sz, cw, ch)
    ko = max(ko, 1e-9)
    half = sz / 2.0
    clear = np.minimum.reduce(
        [
            pos[:, 0] - half[:, 0],
            cw - (pos[:, 0] + half[:, 0]),
            pos[:, 1] - half[:, 1],
            ch - (pos[:, 1] + half[:, 1]),
        ]
    )
    penalty = np.clip((ko - clear) / ko, 0.0, None)
    return float(np.mean(penalty * penalty))


def grid_alignment_penalty(
    positions: np.ndarray,
    sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    *,
    grid_cols: "int | None" = None,
    grid_rows: "int | None" = None,
    pitch_x: "float | None" = None,
    pitch_y: "float | None" = None,
    indices: "np.ndarray | list[int] | None" = None,
) -> float:
    """Penalize centers that are far from the nearest routing-grid cell center."""
    pos, sz = _as_pos_size(positions, sizes, indices)
    if pos.shape[0] == 0:
        return 0.0
    del sz
    if pitch_x is None:
        pitch_x = float(canvas_width) / max(int(grid_cols or 0), 1) if grid_cols else None
    if pitch_y is None:
        pitch_y = float(canvas_height) / max(int(grid_rows or 0), 1) if grid_rows else None
    if pitch_x is None or pitch_y is None:
        scale = max(min(float(canvas_width), float(canvas_height)) * 0.02, 1e-9)
        pitch_x = pitch_y = scale
    px, py = max(float(pitch_x), 1e-9), max(float(pitch_y), 1e-9)

    gx = (np.floor(pos[:, 0] / px) + 0.5) * px
    gy = (np.floor(pos[:, 1] / py) + 0.5) * py
    dx = np.abs(pos[:, 0] - gx) / (0.5 * px)
    dy = np.abs(pos[:, 1] - gy) / (0.5 * py)
    penalty = 0.5 * (np.clip(dx, 0.0, 1.0) ** 2 + np.clip(dy, 0.0, 1.0) ** 2)
    return float(np.mean(penalty))


def notch_penalty(
    positions: np.ndarray,
    sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    *,
    notch_window: "float | None" = None,
    indices: "np.ndarray | list[int] | None" = None,
) -> float:
    """Penalize narrow channels between overlapping macro projections."""
    pos_all = np.asarray(positions, dtype=np.float64)
    sz_all = np.asarray(sizes, dtype=np.float64)
    _as_pos_size(pos_all, sz_all, None)
    if pos_all.shape[0] < 2:
        return 0.0
    idx = np.arange(pos_all.shape[0], dtype=np.int64) if indices is None else np.asarray(indices)
    if idx.size == 0:
        return 0.0
    window = (
        float(notch_window)
        if notch_window is not None
        else _default_keepout(sz_all, float(canvas_width), float(canvas_height))
    )
    window = max(window, 1e-9)
    total, count = _structural_notch_penalty_totals(pos_all, sz_all, window, idx)
    return float(total / max(count, 1))


def structural_penalty_components(
    positions: np.ndarray,
    sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    *,
    grid_cols: "int | None" = None,
    grid_rows: "int | None" = None,
    keepout: "float | None" = None,
    notch_window: "float | None" = None,
    indices: "np.ndarray | list[int] | None" = None,
) -> dict[str, float]:
    """Return individual structural metric components."""
    return {
        "edge_keepout": edge_keepout_penalty(
            positions, sizes, canvas_width, canvas_height, keepout=keepout, indices=indices
        ),
        "grid_alignment": grid_alignment_penalty(
            positions,
            sizes,
            canvas_width,
            canvas_height,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            indices=indices,
        ),
        "notch": notch_penalty(
            positions,
            sizes,
            canvas_width,
            canvas_height,
            notch_window=notch_window,
            indices=indices,
        ),
    }


def combined_structural_penalty(
    positions: np.ndarray,
    sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    *,
    grid_cols: "int | None" = None,
    grid_rows: "int | None" = None,
    keepout_weight: float = 0.2,
    grid_align_weight: float = 0.2,
    notch_weight: float = 0.6,
    keepout: "float | None" = None,
    notch_window: "float | None" = None,
    indices: "np.ndarray | list[int] | None" = None,
) -> float:
    """Weighted structural penalty; lower is more regular."""
    comp = structural_penalty_components(
        positions,
        sizes,
        canvas_width,
        canvas_height,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        keepout=keepout,
        notch_window=notch_window,
        indices=indices,
    )
    return float(
        float(keepout_weight) * comp["edge_keepout"]
        + float(grid_align_weight) * comp["grid_alignment"]
        + float(notch_weight) * comp["notch"]
    )
