"""Shared region-gating rules for hierarchy-preserving relief moves."""

from __future__ import annotations


def point_in_region(region_bbox, idx: int, x: float, y: float) -> bool:
    """Return whether a center point is inside its assigned region box."""
    if region_bbox is None:
        return True
    rb = region_bbox[int(idx)]
    return bool(rb[0] <= x <= rb[2] and rb[1] <= y <= rb[3])


def accepts_region_score(
    score: float,
    best_score: float,
    outside_region: bool,
    escape_min: float,
) -> bool:
    """Exact-proxy accept rule for in-region and region-escaping moves."""
    min_gain = max(1e-9, float(escape_min)) if outside_region else 1e-9
    return float(score) < float(best_score) - min_gain


def any_outside_region(rows) -> bool:
    """Return whether any ``(region, idx, x, y)`` tuple falls outside its region."""
    for region_bbox, idx, x, y in rows:
        if not point_in_region(region_bbox, int(idx), float(x), float(y)):
            return True
    return False
