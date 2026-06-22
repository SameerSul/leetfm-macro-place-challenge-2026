"""Verify NG45 hierarchy tag locality after placement.

NG45 macro names carry RTL instance paths. This check derives hierarchy tags
from slash-separated path prefixes and verifies that hard macros sharing those
tags remain locally coherent after the hierarchy placer runs.

Usage:
    uv run python test/verification/_verify_ng45_hierarchy_tags.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np

from macro_place.evaluate import NG45_BENCHMARKS, evaluate_benchmark
from src.main import MacroPlacer


MAX_RADIUS_GROWTH = 1.25
MAX_FINAL_RADIUS = 0.05
MAX_PURITY_DROP = 0.08


def _prefix(name: str, depth: int) -> str:
    parts = [p for p in str(name).split("/") if p]
    if len(parts) < depth:
        return str(name)
    return "/".join(parts[:depth])


def _tag_groups(names: list[str], depth: int) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        buckets.setdefault(_prefix(name, depth), []).append(idx)
    return {
        tag: np.asarray(indices, dtype=np.int64)
        for tag, indices in buckets.items()
        if len(indices) >= 2
    }


def _mean_group_radius(pos: np.ndarray, groups: dict[str, np.ndarray], diag: float) -> float:
    values = []
    for members in groups.values():
        xy = pos[members]
        centroid = np.mean(xy, axis=0)
        values.append(float(np.mean(np.linalg.norm(xy - centroid, axis=1))) / diag)
    return float(np.mean(values)) if values else 0.0


def _nearest_purity(pos: np.ndarray, groups: dict[str, np.ndarray], depth_tags: list[str]) -> float:
    covered = np.zeros(len(depth_tags), dtype=bool)
    for members in groups.values():
        covered[members] = True
    indices = np.where(covered)[0]
    if indices.size == 0:
        return 1.0
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    hits = 0
    total = 0
    tags = np.asarray(depth_tags, dtype=object)
    for i in indices:
        group_size = int(np.sum(tags == tags[i]))
        k = max(1, min(group_size - 1, 4))
        nearest = np.argsort(d2[i])[:k]
        hits += int(np.sum(tags[nearest] == tags[i]))
        total += k
    return float(hits) / float(max(total, 1))


def _best_depth(names: list[str]) -> int:
    best = 1
    best_score = (-1, -1)
    total = len(names)
    for depth in range(1, 6):
        groups = _tag_groups(names, depth)
        if not groups:
            continue
        covered = sum(len(v) for v in groups.values())
        nontrivial = sum(1 for v in groups.values() if 2 <= len(v) < total)
        score = (nontrivial, covered)
        if score > best_score:
            best = depth
            best_score = score
    return best


def _hard_overlap_count(pos: np.ndarray, sizes: np.ndarray) -> int:
    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0
    sep_x = np.abs(pos[:, None, 0] - pos[None, :, 0]) >= (hw[:, None] + hw[None, :]) - 1e-6
    sep_y = np.abs(pos[:, None, 1] - pos[None, :, 1]) >= (hh[:, None] + hh[None, :]) - 1e-6
    ok = sep_x | sep_y
    np.fill_diagonal(ok, True)
    return int((~ok).sum() // 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "designs",
        nargs="*",
        default=list(NG45_BENCHMARKS.keys()),
        help="Optional NG45 design names to verify.",
    )
    args = parser.parse_args()
    placer = MacroPlacer()
    rows = []
    failures = []
    for design in args.designs:
        ng45_dir = NG45_BENCHMARKS[design]
        result = evaluate_benchmark(placer, design, "", ng45_dir=ng45_dir)
        benchmark = result["benchmark"]
        n = int(benchmark.num_hard_macros)
        names = list(benchmark.macro_names[:n])
        initial = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)[:n]
        final = result["placement"].detach().cpu().numpy().astype(np.float64)[:n]
        sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)[:n]
        depth = _best_depth(names)
        tags = [_prefix(name, depth) for name in names]
        groups = _tag_groups(names, depth)
        diag = float(np.hypot(float(benchmark.canvas_width), float(benchmark.canvas_height)))
        initial_radius = _mean_group_radius(initial, groups, diag)
        final_radius = _mean_group_radius(final, groups, diag)
        radius_growth = final_radius / max(initial_radius, 1e-12)
        initial_purity = _nearest_purity(initial, groups, tags)
        final_purity = _nearest_purity(final, groups, tags)
        purity_drop = initial_purity - final_purity
        overlaps = _hard_overlap_count(final, sizes)
        relative_ok = radius_growth <= MAX_RADIUS_GROWTH and purity_drop <= MAX_PURITY_DROP
        absolute_ok = final_radius <= MAX_FINAL_RADIUS
        ok = overlaps == 0 and (relative_ok or absolute_ok)
        rows.append(
            {
                "design": design,
                "proxy": float(result["proxy_cost"]),
                "valid": bool(result["valid"]),
                "overlaps": overlaps,
                "depth": depth,
                "groups": len(groups),
                "initial_radius": initial_radius,
                "final_radius": final_radius,
                "radius_growth": radius_growth,
                "initial_purity": initial_purity,
                "final_purity": final_purity,
                "ok": ok,
            }
        )
        if not ok:
            failures.append(design)

    print(
        "design,proxy,valid,overlaps,prefix_depth,groups,"
        "initial_radius,final_radius,radius_growth,initial_purity,final_purity,status"
    )
    for row in rows:
        print(
            f"{row['design']},{row['proxy']:.4f},{int(row['valid'])},"
            f"{row['overlaps']},{row['depth']},{row['groups']},"
            f"{row['initial_radius']:.4f},{row['final_radius']:.4f},"
            f"{row['radius_growth']:.3f},{row['initial_purity']:.3f},"
            f"{row['final_purity']:.3f},{'PASS' if row['ok'] else 'FAIL'}"
        )
    if failures:
        raise SystemExit(f"NG45 hierarchy tag verification failed: {failures}")
    print("NG45 HIERARCHY TAG VERIFICATION PASSED")


if __name__ == "__main__":
    main()
