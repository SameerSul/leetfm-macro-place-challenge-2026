"""Explain why IBM soft macros do not receive hierarchy roles."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from macro_place.evaluate import IBM_BENCHMARKS  # noqa: E402
from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from placer.local_search.hierarchy_model import HierarchyModel  # noqa: E402
from placer.local_search.soft_hierarchy import (  # noqa: E402
    select_stable_residual_soft_bundles,
)
from placer.scoring.wirelength import _build_wl_cache  # noqa: E402
from utils import constants as const  # noqa: E402


def analyze_design(name: str) -> dict:
    benchmark_dir = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name
    benchmark, plc = load_benchmark_from_dir(str(benchmark_dir))
    n = int(benchmark.num_hard_macros)
    n_soft = int(benchmark.num_soft_macros)
    hierarchy = HierarchyModel.build(
        plc,
        n,
        n_soft,
        hard_sizes=benchmark.macro_sizes[:n].numpy(),
    )
    cache = _build_wl_cache(plc)
    hard_by_module = {
        int(module): output for output, module in enumerate(plc.hard_macro_indices[:n])
    }
    soft_by_module = {
        int(module): soft for soft, module in enumerate(plc.soft_macro_indices[:n_soft])
    }
    has_any_net = np.zeros(n_soft, dtype=bool)
    has_any_hard = np.zeros(n_soft, dtype=bool)
    has_clustered_hard = np.zeros(n_soft, dtype=bool)
    has_low_fanout_clustered_hard = np.zeros(n_soft, dtype=bool)
    fanout_limit = max(int(hierarchy.max_fanout), int(const.HIER_SOFT_ROLE_MAX_FANOUT))
    for net_index, start_raw in enumerate(cache["net_starts"]):
        length = int(cache["net_lengths"][net_index])
        if length < 2:
            continue
        start = int(start_raw)
        modules = {int(module) for module in cache["ref_idx"][start : start + length]}
        softs = {soft_by_module[module] for module in modules if module in soft_by_module}
        if not softs:
            continue
        hard_outputs = [hard_by_module[module] for module in modules if module in hard_by_module]
        clustered = any(int(hierarchy.labels[output]) >= 0 for output in hard_outputs)
        for soft in softs:
            has_any_net[soft] = True
            has_any_hard[soft] |= bool(hard_outputs)
            has_clustered_hard[soft] |= bool(clustered)
            has_low_fanout_clustered_hard[soft] |= bool(clustered and length <= fanout_limit)

    assigned = np.zeros(n_soft, dtype=bool)
    for outputs in hierarchy.cluster_softs.values():
        for output in np.asarray(outputs, dtype=np.int64):
            soft = int(output) - n
            if 0 <= soft < n_soft:
                assigned[soft] = True
    for soft in hierarchy.bridge_softs:
        if 0 <= int(soft) < n_soft:
            assigned[int(soft)] = True

    source_counts = Counter(
        str(row.get("source", "unknown")) for row in hierarchy.soft_role_evidence.values()
    )

    reasons: Counter[str] = Counter()
    for soft in range(n_soft):
        if assigned[soft]:
            reasons["assigned"] += 1
        elif has_low_fanout_clustered_hard[soft]:
            reasons["qualifying_but_unassigned"] += 1
        elif has_clustered_hard[soft]:
            reasons["clustered_hard_only_high_fanout"] += 1
        elif has_any_hard[soft]:
            reasons["only_unclustered_hard"] += 1
        elif has_any_net[soft]:
            reasons["soft_or_io_only"] += 1
        else:
            reasons["no_multi_pin_net"] += 1
    return {
        "benchmark": name,
        "soft_total": n_soft,
        "hard_cluster_coverage": float(np.count_nonzero(hierarchy.labels >= 0) / max(n, 1)),
        "fanout_limit": fanout_limit,
        "role_sources": dict(sorted(source_counts.items())),
        "soft_only_groups": int(len(hierarchy.soft_only_bundles)),
        "soft_only_members": int(
            sum(len(bundle.members) for bundle in hierarchy.soft_only_bundles)
        ),
        "soft_only_selection": dict(select_stable_residual_soft_bundles.last_stats),
        "reasons": dict(sorted(reasons.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [analyze_design(name) for name in IBM_BENCHMARKS]
    totals: Counter[str] = Counter()
    role_sources: Counter[str] = Counter()
    soft_only_selection: Counter[str] = Counter()
    for row in rows:
        totals.update(row["reasons"])
        role_sources.update(row["role_sources"])
        soft_only_selection.update(row["soft_only_selection"])
    result = {
        "soft_total": int(sum(row["soft_total"] for row in rows)),
        "totals": dict(sorted(totals.items())),
        "role_sources": dict(sorted(role_sources.items())),
        "soft_only_groups": int(sum(row["soft_only_groups"] for row in rows)),
        "soft_only_members": int(sum(row["soft_only_members"] for row in rows)),
        "soft_only_selection": dict(sorted(soft_only_selection.items())),
        "designs": rows,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
