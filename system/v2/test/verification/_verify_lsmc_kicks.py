"""Verify LSMC kick operators produce legal hard-macro placements.

Usage:
  PYTHONPATH=system/v2/src \
  uv run python system/v2/test/verification/_verify_lsmc_kicks.py
"""

from __future__ import annotations

import time

import numpy as np
from macro_place.loader import load_benchmark_from_dir

from placer.local_search.lsmc_explore import _select_kick
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.exact import _exact_proxy


def _assert_legal_hard(pos, sizes, hw, hh, cw, ch, movable, n, init_pos) -> None:
    if np.any(pos[:, 0] - hw < -1e-6) or np.any(pos[:, 0] + hw > cw + 1e-6):
        raise AssertionError("hard macro x coordinate out of bounds")
    if np.any(pos[:, 1] - hh < -1e-6) or np.any(pos[:, 1] + hh > ch + 1e-6):
        raise AssertionError("hard macro y coordinate out of bounds")
    if not np.allclose(pos[~movable], init_pos[~movable]):
        raise AssertionError("fixed hard macro moved")

    for i in range(n):
        for j in range(i + 1, n):
            overlap_x = abs(pos[i, 0] - pos[j, 0]) < (hw[i] + hw[j] - 1e-6)
            overlap_y = abs(pos[i, 1] - pos[j, 1]) < (hh[i] + hh[j] - 1e-6)
            if overlap_x and overlap_y:
                raise AssertionError(f"hard overlap after kick: {i}, {j}")


def _check(name: str) -> None:
    bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    _patch_plc_congestion(plc, bm)
    _exact_proxy(bm.macro_positions, bm, plc)

    n = bm.num_hard_macros
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0
    cw = float(bm.canvas_width)
    ch = float(bm.canvas_height)
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    init_pos = bm.macro_positions[:n].numpy().astype(np.float64)
    rng = np.random.default_rng(123)

    for mode in ("random", "congestion", "mixed"):
        kicked = _select_kick(
            init_pos,
            sizes,
            hw,
            hh,
            cw,
            ch,
            movable,
            n,
            0.02,
            rng,
            time.monotonic() + 15.0,
            plc,
            bm,
            mode,
            iteration=0,
        )
        _assert_legal_hard(kicked, sizes, hw, hh, cw, ch, movable, n, init_pos)
        print(f"{name}: {mode} kick legal")


def main() -> int:
    _check("ibm01")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
