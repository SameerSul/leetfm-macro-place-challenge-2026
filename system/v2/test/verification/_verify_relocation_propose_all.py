"""Verify experimental hard-relocation propose-all state consistency.

This does not assert equivalence with the legacy sequential greedy policy; the
propose-all path intentionally changes proposal ordering. It verifies the core
invariant needed before A/B testing: after serial exact verify/commit, the
incremental scorer's returned proxy matches a fresh exact proxy recompute.

Usage:
  PYTHONPATH=system/v2/src \
  uv run python system/v2/test/verification/_verify_relocation_propose_all.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from macro_place.loader import load_benchmark_from_dir

from placer.local_search.relocation import _relocation_moves
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer


def _check(name: str, top_hot: int = 6, n_targets: int = 6) -> None:
    bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    _patch_plc_congestion(plc, bm)

    pl = bm.macro_positions.numpy().astype(np.float64)
    n = bm.num_hard_macros
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0
    movable = bm.get_movable_mask().numpy()[:n]
    base = float(_exact_proxy(bm.macro_positions, bm, plc))

    scorer = IncrementalScorer(plc, bm, pl)
    pos, accepts, score = _relocation_moves(
        pl[:n].copy(),
        sizes,
        hw,
        hh,
        float(bm.canvas_width),
        float(bm.canvas_height),
        movable,
        n,
        plc,
        bm,
        scorer,
        base,
        deadline=time.monotonic() + 10.0,
        top_hot=top_hot,
        n_targets=n_targets,
        propose_all=True,
    )

    full = pl.copy()
    full[:n] = pos
    exact = float(_exact_proxy(torch.tensor(full, dtype=torch.float32), bm, plc))
    delta = abs(score - exact)
    print(
        f"{name}: base={base:.6f} accepts={accepts} "
        f"scorer={score:.6f} exact={exact:.6f} delta={delta:.3e}"
    )
    if delta > 1e-8:
        raise AssertionError(f"{name}: scorer/exact mismatch {delta:.3e}")


def main() -> int:
    for name in ("ibm01", "ibm04"):
        _check(name)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
