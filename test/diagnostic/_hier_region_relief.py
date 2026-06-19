"""Region-locked congestion relief: proxy + closeness, relief OFF vs ON.

Runs hierarchy placement with the region-relief constant forced OFF then ON, and
reports final-placement connectivity closeness plus exact proxy. Expectation:
proxy DROPS while closeness stays near the OFF values.

    uv run python test/diagnostic/_hier_region_relief.py [ibm01 ibm10 ...]
"""

import importlib
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


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
        s = int(ns[i])
        refs = [int(x) for x in ref[s : s + L]]
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
    return (
        float(np.hypot(pos[p[:, 0], 0] - pos[p[:, 1], 0], pos[p[:, 0], 1] - pos[p[:, 1], 1]).mean())
        / diag
    )


def _run(bench, relief):
    from macro_place.loader import load_benchmark_from_dir
    from placer import constants as const
    import placer.pipeline.macro_placer as mp

    const.HIER_REGION_RELIEF = bool(relief)
    importlib.reload(mp)
    from placer.scoring.exact import _exact_proxy
    import torch

    src = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(src))
    n, n_soft = benchmark.num_hard_macros, benchmark.num_soft_macros
    pos = mp.MacroPlacer().place(benchmark).detach().cpu().numpy().astype(np.float64)
    proxy = float(_exact_proxy(torch.tensor(pos, dtype=torch.float32), benchmark, plc))
    cw, ch = (float(v) for v in plc.get_canvas_width_height())
    diag = float(np.hypot(cw, ch))
    from placer.local_search.clusters import derive_hard_clusters

    _, clusters = derive_hard_clusters(plc, n, n_soft=n_soft, min_edge=2)
    hh, hs = _conn(plc, n, n_soft)
    intra = []
    for mem in clusters.values():
        intra += list(combinations(sorted(int(x) for x in mem), 2))
    return {
        "proxy": proxy,
        "hh": _dist(pos, hh, diag),
        "intra": _dist(pos, intra, diag),
        "hs": _dist(pos, hs, diag),
    }


if __name__ == "__main__":
    benches = sys.argv[1:] or ["ibm01", "ibm10", "ibm17"]
    for b in benches:
        off = _run(b, False)
        on = _run(b, True)
        print(f"\n{b}:")
        print(f"  {'':12s} {'proxy':>8s} {'hh':>8s} {'intra':>8s} {'hs':>8s}")
        print(
            f"  relief OFF   {off['proxy']:8.4f} {off['hh']:8.4f} {off['intra']:8.4f} {off['hs']:8.4f}"
        )
        print(
            f"  relief ON    {on['proxy']:8.4f} {on['hh']:8.4f} {on['intra']:8.4f} {on['hs']:8.4f}"
        )
        print(
            f"  delta        {on['proxy']-off['proxy']:+8.4f} {on['hh']-off['hh']:+8.4f} "
            f"{on['intra']-off['intra']:+8.4f} {on['hs']-off['hs']:+8.4f}"
        )
