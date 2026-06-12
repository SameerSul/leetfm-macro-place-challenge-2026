"""Verify CUDA proposal-routing helper parity.

The hard-relocation `cuda_delta` proposal scorer builds sparse routing deltas
without mutating PLC state. This verifier checks those sparse helpers against
the authoritative `_apply_net_routing_subset` implementation for touched nets.

Usage:
  PYTHONPATH=system/v2/src \
  uv run python system/v2/test/verification/_verify_relocation_cuda_routing_helpers.py
"""

from __future__ import annotations

import numpy as np
from macro_place.loader import load_benchmark_from_dir

from placer.local_search.relocation import (
    _net_routing_2pin_contrib,
    _net_routing_3pin_contrib,
    _net_routing_highfanout_contrib,
)
from placer.routing.apply import _apply_net_routing_subset
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.incremental import IncrementalScorer


def _check(name: str, max_macros: int = 40, tol: float = 1e-6) -> None:
    bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    _patch_plc_congestion(plc, bm)

    pl = bm.macro_positions.numpy().astype(np.float64)
    scorer = IncrementalScorer(plc, bm, pl)
    max_v = 0.0
    max_h = 0.0
    checked = 0

    for i_hard in range(min(max_macros, scorer.n_hard)):
        module = int(scorer.hard_indices[i_hard])
        nets = scorer.macro_to_nets.get(module)
        if nets is None or len(nets) == 0:
            continue

        v_ref = np.zeros_like(scorer.V_flat)
        h_ref = np.zeros_like(scorer.H_flat)
        _apply_net_routing_subset(plc, nets, 1.0, h_ref, v_ref)

        v_new = np.zeros_like(scorer.V_flat)
        h_new = np.zeros_like(scorer.H_flat)
        for helper in (
            _net_routing_2pin_contrib,
            _net_routing_3pin_contrib,
            _net_routing_highfanout_contrib,
        ):
            flat, v_val, h_val = helper(scorer, module, pl[i_hard, 0], pl[i_hard, 1])
            if flat.size:
                np.add.at(v_new, flat, v_val.astype(np.float64))
                np.add.at(h_new, flat, h_val.astype(np.float64))

        max_v = max(max_v, float(np.max(np.abs(v_ref - v_new))))
        max_h = max(max_h, float(np.max(np.abs(h_ref - h_new))))
        checked += 1

    print(f"{name}: checked={checked} max_v={max_v:.3e} max_h={max_h:.3e}")
    if checked == 0:
        raise AssertionError(f"{name}: no touched-net routing helpers checked")
    if max_v > tol or max_h > tol:
        raise AssertionError(f"{name}: routing helper mismatch max_v={max_v:.3e} max_h={max_h:.3e}")


def main() -> int:
    for name in ("ibm01", "ibm04"):
        _check(name)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
