"""Measure whether DREAMPlace soft-grouping keeps connected macros closer.

Runs DREAMPlace global placement at several group weights (0 = ungrouped) and
reports connectivity-weighted distances on the RAW DP output (before any
refinement/gating), normalized by the canvas diagonal. This isolates the
structural effect of the grouping nets, separate from the proxy accept gate.

    uv run python test/diagnostic/_dp_group_closeness.py [ibm10] [0,4,8]
"""
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "system" / "v2" / "src"))

import numpy as np

from macro_place.loader import load_benchmark_from_dir
from placer.local_search.clusters import derive_hard_clusters
from dreamplace_bridge.run_bridge import run_dreamplace
from dreamplace_bridge.bookshelf_to_pb import read_dreamplace_positions_full


def _connectivity(plc, n, n_soft, max_fanout=8):
    from placer.scoring.wirelength import _build_wl_cache
    c = _build_wl_cache(plc)
    ref, nstarts, nl = c["ref_idx"], c["net_starts"], c["net_lengths"]
    hb2a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}
    sb2p = {int(b): n + a for a, b in enumerate(plc.soft_macro_indices)}
    hh, hs = {}, []
    for i in range(len(nstarts)):
        L = int(nl[i])
        if L < 2 or L > max_fanout:
            continue
        s = int(nstarts[i])
        refs = [int(x) for x in ref[s:s + L]]
        hards = sorted({hb2a[x] for x in refs if x in hb2a})
        softs = sorted({sb2p[x] for x in refs if x in sb2p})
        for a, b in combinations(hards, 2):
            hh[(a, b)] = hh.get((a, b), 0) + 1
        for h in hards:
            for sft in softs:
                hs.append((h, sft))
    return [k for k, w in hh.items() if w >= 2], hs


def _dist(pos, pairs):
    if not pairs:
        return float("nan")
    p = np.array(pairs)
    return float(np.hypot(pos[p[:, 0], 0] - pos[p[:, 1], 0],
                          pos[p[:, 0], 1] - pos[p[:, 1], 1]).mean())


def _measure(bench, weights):
    src = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(src))
    n, n_soft = benchmark.num_hard_macros, benchmark.num_soft_macros
    cw, ch = (float(v) for v in plc.get_canvas_width_height())
    diag = float(np.hypot(cw, ch))
    hh_pairs, hs_pairs = _connectivity(plc, n, n_soft)
    _, clusters = derive_hard_clusters(plc, n, n_soft=n_soft, min_edge=2)
    intra = []
    for mem in clusters.values():
        intra += list(combinations(sorted(int(x) for x in mem), 2))
    hmi = plc.hard_macro_indices
    groups = [[plc.modules_w_pins[hmi[int(a)]].get_name() for a in mem]
              for mem in clusters.values()]

    print(f"\n{bench}: {len(clusters)} clusters; distances / canvas diag (lower=closer)")
    print(f"  {'weight':>7s} {'hard-hard':>10s} {'intra':>8s} {'hard-soft':>10s}")
    for w in weights:
        scratch = f"/tmp/dp_grp_close_w{w}"
        run_dreamplace(str(src), plc=plc, scratch_root=scratch, iterations=300,
                       num_threads=2, soft_macros_movable=True,
                       cluster_groups=(groups if w > 0 else None), group_weight=w)
        hard, soft = read_dreamplace_positions_full(plc, f"{scratch}/{bench}", bench)
        pos = np.vstack([hard, soft])
        print(f"  {w:>7d} {_dist(pos, hh_pairs)/diag:>10.4f} "
              f"{_dist(pos, intra)/diag:>8.4f} {_dist(pos, hs_pairs)/diag:>10.4f}")


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    weights = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "4", "8", "16"])]
    _measure(bench, weights)
