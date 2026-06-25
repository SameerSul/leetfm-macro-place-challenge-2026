"""Verify region boxes and region-escape acceptance rules.

uv run python test/verification/_verify_region_escape_gate.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from placer.local_search.clusters import hier_region_density, hier_region_singleton  # noqa: E402
from placer.local_search.hierarchy_model import HierarchyModel  # noqa: E402
from placer.local_search.region_rules import (  # noqa: E402
    accepts_region_score,
    any_outside_region,
    point_in_region,
)
from placer.plc.loader import _load_plc  # noqa: E402


def run_one(bench_name):
    bench, _ = load_benchmark_from_dir(
        str(ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench_name)
    )
    plc = _load_plc(bench_name, bench)
    n = bench.num_hard_macros
    ns = bench.num_soft_macros
    pos = bench.macro_positions.cpu().numpy().astype(np.float64)
    sizes = bench.macro_sizes.cpu().numpy().astype(np.float64)
    hard = pos[:n]
    soft = pos[n : n + ns]
    hw, hh = sizes[:n, 0] / 2.0, sizes[:n, 1] / 2.0
    soft_hw = sizes[n : n + ns, 0] / 2.0
    soft_hh = sizes[n : n + ns, 1] / 2.0
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)

    hierarchy = HierarchyModel.build(plc, n, ns, hard_sizes=sizes[:n])
    clusters = hierarchy.clusters
    csofts = hierarchy.cluster_softs
    hard_region = hierarchy.hard_regions(
        hard,
        sizes[:n],
        hw,
        hh,
        cw,
        ch,
        n,
    )
    soft_region = hierarchy.soft_regions(
        hard,
        soft,
        sizes[:n],
        hw,
        hh,
        soft_hw,
        soft_hh,
        cw,
        ch,
        n,
    )

    assert np.all([point_in_region(hard_region, i, hard[i, 0], hard[i, 1]) for i in range(n)])
    assert np.all([point_in_region(soft_region, k, soft[k, 0], soft[k, 1]) for k in range(ns)])

    assert accepts_region_score(0.999999, 1.0, False, 0.002)
    assert not accepts_region_score(0.999999, 1.0, True, 0.002)
    assert accepts_region_score(0.9979, 1.0, True, 0.002)

    if n:
        outside = any_outside_region([(hard_region, 0, hard_region[0, 2] + 1.0, hard[0, 1])])
        assert outside

    print(
        f"{bench_name}: region boxes OK hard={n} soft={ns} "
        f"clusters={len(clusters)} assigned_softs={sum(len(v) for v in csofts.values())} "
        f"density={hier_region_density():.2f} singleton={hier_region_singleton():.2f}"
    )


def main():
    for bench_name in sys.argv[1:] or ["ibm01", "ibm04", "ibm10"]:
        run_one(bench_name)
    print("REGION-ESCAPE VERIFICATION PASSED")


if __name__ == "__main__":
    main()
