"""Verify hierarchy-floorplan + region-locked relief yields a VALID placement.

Runs MacroPlacer with V2_HIER_FLOORPLAN=1 and V2_HIER_REGION_RELIEF=1 and asserts
the hard placement is overlap-free and in-bounds (the harness's hard requirements).
The region lock is SOFT (macros may exit on a large proxy win by design), so this
does not assert a hard region invariant — hierarchy retention is measured
quantitatively by test/diagnostic/_hier_region_relief.py. Skips when DREAMPlace is
unavailable (the mode falls back to the normal pipeline).

    uv run python test/verification/_verify_region_relief.py [ibm01]
"""
import os
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from macro_place.loader import load_benchmark_from_dir
from dreamplace_bridge.run_bridge import is_available

TOL = 0.05


def main(bench):
    if not is_available():
        print("DREAMPlace unavailable; hier mode falls back to normal place(). SKIP.")
        return
    os.environ["V2_HIER_FLOORPLAN"] = "1"
    os.environ["V2_HIER_REGION_RELIEF"] = "1"
    import importlib
    import placer.pipeline.macro_placer as mp
    importlib.reload(mp)

    src = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(src))
    n = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
    hw, hh = sizes[:n, 0] / 2.0, sizes[:n, 1] / 2.0
    cw, ch = (float(v) for v in plc.get_canvas_width_height())

    pos = mp.MacroPlacer().place(benchmark).detach().cpu().numpy().astype(np.float64)[:n]

    # In bounds.
    assert np.all(pos[:, 0] >= hw - TOL) and np.all(pos[:, 0] <= cw - hw + TOL), "x oob"
    assert np.all(pos[:, 1] >= hh - TOL) and np.all(pos[:, 1] <= ch - hh + TOL), "y oob"

    # No hard-macro overlap.
    sep_x = np.abs(pos[:, None, 0] - pos[None, :, 0]) + TOL >= (hw[:, None] + hw[None, :])
    sep_y = np.abs(pos[:, None, 1] - pos[None, :, 1]) + TOL >= (hh[:, None] + hh[None, :])
    ok = sep_x | sep_y
    np.fill_diagonal(ok, True)
    n_overlap = int((~ok).sum() // 2)
    assert n_overlap == 0, f"{n_overlap} hard-macro overlaps"

    # Report intra-cluster spread for eyeballing hierarchy retention.
    from placer.local_search.clusters import derive_hard_clusters
    _, clusters = derive_hard_clusters(plc, n, n_soft=n_soft, min_edge=2)
    diag = float(np.hypot(cw, ch))
    intra = []
    for mem in clusters.values():
        intra += list(combinations(sorted(int(x) for x in mem), 2))
    if intra:
        p = np.array(intra)
        spread = float(np.hypot(pos[p[:, 0], 0] - pos[p[:, 1], 0],
                                pos[p[:, 0], 1] - pos[p[:, 1], 1]).mean()) / diag
    else:
        spread = float("nan")
    print(f"{bench}: region-relief OK  hard={n}  0 overlaps, in-bounds, "
          f"intra-cluster spread={spread:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ibm01")
    print("REGION-RELIEF VERIFICATION PASSED")
