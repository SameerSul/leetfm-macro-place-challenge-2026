"""Inspect derived connectivity clusters: count, sizes, and current spread.

Spread = bbox diagonal of each cluster's current macro centers, as a fraction
of the canvas diagonal. Large spread means the inferred subsystem is currently
scattered across the chip — exactly the case a cluster-coherent kick targets.

    uv run python system/v2/test/diagnostic/_cluster_stats.py [ibm10] [max_fanout]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "system" / "v2" / "src"))

import numpy as np

from macro_place.loader import load_benchmark_from_dir
from placer.local_search.clusters import derive_hard_clusters

BENCH = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
MAX_FANOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SRC = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / BENCH
benchmark, plc = load_benchmark_from_dir(str(SRC))

n = benchmark.num_hard_macros
n_soft = benchmark.num_soft_macros
pos = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)[:n]
cw = float(plc.get_canvas_width_height()[0])
ch = float(plc.get_canvas_width_height()[1])
canvas_diag = float(np.hypot(cw, ch))

labels, clusters = derive_hard_clusters(plc, n, n_soft=n_soft, max_fanout=MAX_FANOUT)
clustered = int((labels >= 0).sum())

sizes = np.array([m.size for m in clusters.values()]) if clusters else np.array([0])
spreads = []
for members in clusters.values():
    p = pos[members]
    diag = float(np.hypot(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
    spreads.append(diag / canvas_diag)
spreads = np.array(spreads) if spreads else np.array([0.0])

print(f"{BENCH}: n_hard={n}  max_fanout={MAX_FANOUT}")
print(f"  clusters(>=2): {len(clusters)}  macros clustered: {clustered}/{n} "
      f"({100*clustered/max(1,n):.0f}%)")
print(f"  cluster size: min={sizes.min()} median={int(np.median(sizes))} "
      f"max={sizes.max()} mean={sizes.mean():.1f}")
print(f"  spread (bbox-diag / canvas-diag): min={spreads.min():.2f} "
      f"median={np.median(spreads):.2f} max={spreads.max():.2f}")
print(f"  clusters >50% canvas-diag spread: {int((spreads > 0.5).sum())}/{len(clusters)}")
