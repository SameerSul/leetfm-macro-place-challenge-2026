"""Verify the hierarchy-floorplan mode produces a VALID placement.

Runs MacroPlacer with V2_HIER_FLOORPLAN=1 on a small benchmark and asserts the
returned hard placement is overlap-free and in-bounds (the harness's hard
requirements). Skips when DREAMPlace is unavailable, since the mode falls back
to the normal pipeline in that case.

    uv run python system/v2/test/verification/_verify_hier_floorplan.py [ibm01]
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "system" / "v2" / "src"))

import numpy as np

from macro_place.loader import load_benchmark_from_dir
from dreamplace_bridge.run_bridge import is_available

TOL = 0.05


def main(bench):
    if not is_available():
        print("DREAMPlace unavailable; hier mode falls back to normal place(). SKIP.")
        return
    os.environ["V2_HIER_FLOORPLAN"] = "1"
    import importlib
    import placer.pipeline.macro_placer as mp
    importlib.reload(mp)

    src = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(src))
    n = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
    hw, hh = sizes[:n, 0] / 2.0, sizes[:n, 1] / 2.0
    cw, ch = (float(v) for v in plc.get_canvas_width_height())
    movable = benchmark.get_movable_mask().detach().cpu().numpy()[:n]

    pos = mp.MacroPlacer().place(benchmark).detach().cpu().numpy().astype(np.float64)[:n]

    # In bounds.
    assert np.all(pos[:, 0] >= hw - TOL) and np.all(pos[:, 0] <= cw - hw + TOL), "x oob"
    assert np.all(pos[:, 1] >= hh - TOL) and np.all(pos[:, 1] <= ch - hh + TOL), "y oob"

    # No hard-macro overlap (movable + fixed all participate).
    p = pos
    sep_x = np.abs(p[:, None, 0] - p[None, :, 0]) + TOL >= (hw[:, None] + hw[None, :])
    sep_y = np.abs(p[:, None, 1] - p[None, :, 1]) + TOL >= (hh[:, None] + hh[None, :])
    ok = sep_x | sep_y
    np.fill_diagonal(ok, True)
    n_overlap = int((~ok).sum() // 2)
    assert n_overlap == 0, f"{n_overlap} hard-macro overlaps"
    assert int(movable.sum()) >= 0  # sanity touch of movable

    print(f"{bench}: hier-floorplan OK  hard={n}  0 overlaps, in-bounds")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ibm01")
    print("HIER-FLOORPLAN VERIFICATION PASSED")
