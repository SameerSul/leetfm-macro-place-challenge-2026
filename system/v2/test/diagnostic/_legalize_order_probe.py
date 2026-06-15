"""Does cluster-consecutive legalization order preserve DP grouping?

The spiral legalizer places macros largest-area-first globally, so a cluster's
members compete with other macros for cells near their centroid and scatter.
This probes whether ordering each cluster's members consecutively (so they grab
their region before other clusters invade) keeps the grouping tight after
legalization. Reuses the grouped DP output cached by _hier_tradeoff (/tmp/hier_w8).

    uv run python system/v2/test/diagnostic/_legalize_order_probe.py [ibm10]
"""
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "system" / "v2" / "src"))

import numpy as np

from macro_place.loader import load_benchmark_from_dir
from placer.local_search.clusters import derive_hard_clusters
from placer.legalize.spiral import _will_legalize
from dreamplace_bridge.bookshelf_to_pb import read_dreamplace_positions_full

bench = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
src = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
benchmark, plc = load_benchmark_from_dir(str(src))
n, n_soft = benchmark.num_hard_macros, benchmark.num_soft_macros
cw, ch = (float(v) for v in plc.get_canvas_width_height())
diag = float(np.hypot(cw, ch))
sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
hw, hh = sizes[:n, 0] / 2.0, sizes[:n, 1] / 2.0
movable = benchmark.get_movable_mask().detach().cpu().numpy()

labels, clusters = derive_hard_clusters(plc, n, n_soft=n_soft, min_edge=2)
hb2a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}
from placer.scoring.wirelength import _build_wl_cache
c = _build_wl_cache(plc)
ref, ns, nl = c["ref_idx"], c["net_starts"], c["net_lengths"]
hh_pairs = {}
for i in range(len(ns)):
    L = int(nl[i])
    if L < 2 or L > 8:
        continue
    s = int(ns[i]); hs = sorted({hb2a[x] for x in (int(v) for v in ref[s:s + L]) if x in hb2a})
    for a, b in combinations(hs, 2):
        hh_pairs[(a, b)] = hh_pairs.get((a, b), 0) + 1
hh_pairs = [k for k, w in hh_pairs.items() if w >= 2]
intra = []
for mem in clusters.values():
    intra += list(combinations(sorted(int(x) for x in mem), 2))


def dist(pos, pairs):
    p = np.array(pairs)
    return float(np.hypot(pos[p[:, 0], 0] - pos[p[:, 1], 0],
                          pos[p[:, 0], 1] - pos[p[:, 1], 1]).mean()) / diag


hard, _soft = read_dreamplace_positions_full(plc, f"/tmp/hier_w8/{bench}", bench)
print(f"{bench}: grouped DP raw  hh={dist(hard, hh_pairs):.4f} intra={dist(hard, intra):.4f}")

# Default order (largest-area-first).
d = time.monotonic() + 60
leg_def = _will_legalize(hard.copy(), movable[:n], sizes[:n], hw, hh, cw, ch, n, deadline=d)
print(f"  default-order legalize    hh={dist(leg_def, hh_pairs):.4f} intra={dist(leg_def, intra):.4f}")

# Cluster-consecutive order: members of each cluster back-to-back (largest first
# within cluster), clusters ordered by size desc, then non-clustered largest-first.
order = []
for mem in sorted(clusters.values(), key=lambda m: -m.size):
    order += sorted((int(x) for x in mem), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
rest = [i for i in range(n) if labels[i] < 0]
order += sorted(rest, key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
d = time.monotonic() + 60
leg_cons = _will_legalize(hard.copy(), movable[:n], sizes[:n], hw, hh, cw, ch, n,
                          deadline=d, order=order)
print(f"  cluster-consec legalize   hh={dist(leg_cons, hh_pairs):.4f} intra={dist(leg_cons, intra):.4f}")

# Cluster-consec order + centroid-anchored spiral: members search around their
# shared cluster centroid (in the grouped DP placement), not their own pos.
anchor = hard.copy()
for mem in clusters.values():
    cen = hard[mem].mean(axis=0)
    anchor[mem] = cen
d = time.monotonic() + 60
leg_anc = _will_legalize(hard.copy(), movable[:n], sizes[:n], hw, hh, cw, ch, n,
                         deadline=d, order=order, anchor=anchor)
print(f"  consec+centroid-anchor    hh={dist(leg_anc, hh_pairs):.4f} intra={dist(leg_anc, intra):.4f}")
