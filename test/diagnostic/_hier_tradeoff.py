"""Hierarchy-preserving tradeoff: grouped DP seed + limited refinement.

Measures the proxy cost of KEEPING hard-macro hierarchy. Grouped DREAMPlace
(clusters incl. their connected softs) gives a hierarchical global placement;
we then apply only refinement that does NOT move hard macros (legalize, then
soft-only relocation), so hard hierarchy is preserved by construction. Reports
connectivity closeness + exact proxy at each level, vs an ungrouped baseline and
the full-pipeline proxy (which spreads hierarchy out for a lower proxy).

    uv run python test/diagnostic/_hier_tradeoff.py [ibm10] [weight]
"""
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "system" / "v2" / "src"))

import numpy as np
import torch

from macro_place.loader import load_benchmark_from_dir
from placer.local_search.clusters import derive_cluster_softs, derive_hard_clusters
from placer.local_search.relocation import _soft_relocation_moves
from placer.legalize.spiral import _will_legalize
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer
from dreamplace_bridge.run_bridge import run_dreamplace
from dreamplace_bridge.bookshelf_to_pb import read_dreamplace_positions_full


def _conn(plc, n, n_soft, mf=8):
    from placer.scoring.wirelength import _build_wl_cache
    c = _build_wl_cache(plc)
    ref, ns, nl = c["ref_idx"], c["net_starts"], c["net_lengths"]
    hb2a = {int(b): a for a, b in enumerate(plc.hard_macro_indices)}
    sb2p = {int(b): n + a for a, b in enumerate(plc.soft_macro_indices)}
    hh, hs = {}, []
    for i in range(len(ns)):
        L = int(nl[i])
        if L < 2 or L > mf:
            continue
        s = int(ns[i]); refs = [int(x) for x in ref[s:s + L]]
        hards = sorted({hb2a[x] for x in refs if x in hb2a})
        softs = sorted({sb2p[x] for x in refs if x in sb2p})
        for a, b in combinations(hards, 2):
            hh[(a, b)] = hh.get((a, b), 0) + 1
        for h in hards:
            for sft in softs:
                hs.append((h, sft))
    return [k for k, w in hh.items() if w >= 2], hs


def _dist(pos, pairs, diag):
    if not pairs:
        return float("nan")
    p = np.array(pairs)
    return float(np.hypot(pos[p[:, 0], 0] - pos[p[:, 1], 0],
                          pos[p[:, 0], 1] - pos[p[:, 1], 1]).mean()) / diag


def _report(tag, pos, hh, hs, intra, diag, proxy):
    print(f"  {tag:24s} hh={_dist(pos,hh,diag):.4f} intra={_dist(pos,intra,diag):.4f} "
          f"hs={_dist(pos,hs,diag):.4f}  proxy={proxy:.4f}")


def main(bench, weight):
    src = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(src))
    n, n_soft = benchmark.num_hard_macros, benchmark.num_soft_macros
    cw, ch = (float(v) for v in plc.get_canvas_width_height())
    diag = float(np.hypot(cw, ch))
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
    hw, hh_ = sizes[:n, 0] / 2.0, sizes[:n, 1] / 2.0
    soft_hw = sizes[n:n + n_soft, 0] / 2.0
    soft_hh = sizes[n:n + n_soft, 1] / 2.0
    movable = benchmark.get_movable_mask().detach().cpu().numpy()
    hh_pairs, hs_pairs = _conn(plc, n, n_soft)
    labels, clusters = derive_hard_clusters(plc, n, n_soft=n_soft, min_edge=2)
    intra = []
    for mem in clusters.values():
        intra += list(combinations(sorted(int(x) for x in mem), 2))

    # Groups incl. connected softs so DP pulls softs toward their hard cluster.
    csofts = derive_cluster_softs(plc, n, n_soft, labels)
    hmi, smi = plc.hard_macro_indices, plc.soft_macro_indices
    groups = []
    for cid, mem in clusters.items():
        names = [plc.modules_w_pins[hmi[int(a)]].get_name() for a in mem]
        for p in csofts.get(cid, []):
            names.append(plc.modules_w_pins[smi[int(p) - n]].get_name())
        groups.append(names)

    def score(pos):
        pl = torch.tensor(pos, dtype=torch.float32)
        return float(_exact_proxy(pl, benchmark, plc))

    print(f"\n{bench}: {len(clusters)} clusters, weight={weight}")
    for w, gname in ((0, "ungrouped"), (weight, "grouped+softs")):
        scratch = f"/tmp/hier_w{w}"
        run_dreamplace(str(src), plc=plc, scratch_root=scratch, iterations=300,
                       num_threads=2, soft_macros_movable=True,
                       cluster_groups=(groups if w > 0 else None), group_weight=w)
        hard, soft = read_dreamplace_positions_full(plc, f"{scratch}/{bench}", bench)
        # L0: legalize hard only.
        legal = _will_legalize(hard.copy(), movable[:n], sizes[:n], hw, hh_, cw, ch, n,
                               deadline=time.monotonic() + 60)
        pos = np.vstack([legal, soft])
        _report(f"[{gname}] legalize", pos, hh_pairs, hs_pairs, intra, diag, score(pos))
        # L1: soft-only relocation (hard untouched -> hierarchy preserved).
        scorer = IncrementalScorer(plc, benchmark, pos.copy())
        s_pos = soft.copy()
        s_score = score(pos)
        soft_mov = movable[n:n + n_soft]
        for ud in (False, True):
            s_pos, _, s_score = _soft_relocation_moves(
                s_pos, soft_hw, soft_hh, cw, ch, n, plc, benchmark, scorer, s_score,
                deadline=time.monotonic() + 30, top_hot=1024, n_targets=6,
                soft_movable=soft_mov, use_density=ud)
        pos2 = np.vstack([legal, s_pos])
        _report(f"[{gname}] +soft-refine", pos2, hh_pairs, hs_pairs, intra, diag, score(pos2))


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    weight = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(bench, weight)
