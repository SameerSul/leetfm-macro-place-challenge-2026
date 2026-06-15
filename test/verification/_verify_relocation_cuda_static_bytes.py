"""Verify relocation cuda_delta static tensor byte accounting.

This checks the real benchmark static cache, not only monkeypatched wrapper
paths, so future cache changes must keep byte telemetry honest.

Usage:
  PYTHONPATH=src \
  uv run python test/verification/_verify_relocation_cuda_static_bytes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from macro_place.loader import load_benchmark_from_dir

from placer.local_search.relocation import (
    _GPU_DEVICE,
    _build_relocation_cuda_static_tensors,
    _relocation_static_tensor_bytes_estimate,
    _tensor_tree_bytes,
)
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.incremental import IncrementalScorer

_VERIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_VERIFY_DIR))
from _verify_relocation_cuda_delta_scores import _collect_proposals  # noqa: E402


def _check(name: str) -> None:
    bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    _patch_plc_congestion(plc, bm)

    pl = bm.macro_positions.numpy().astype(np.float64)
    n = bm.num_hard_macros
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0
    movable = bm.get_movable_mask().numpy()[:n]
    scorer = IncrementalScorer(plc, bm, pl)

    proposals, _local_cong, _tgt_cong = _collect_proposals(
        pos=pl[:n].copy(),
        sizes=sizes,
        hw=hw,
        hh=hh,
        cw=float(bm.canvas_width),
        ch=float(bm.canvas_height),
        movable=movable,
        plc=plc,
        benchmark=bm,
        incremental_scorer=scorer,
    )
    if not proposals:
        raise AssertionError(f"{name}: no legal proposals collected")

    static_tensors = _build_relocation_cuda_static_tensors(
        proposals,
        pos=pl[:n].copy(),
        incremental_scorer=scorer,
        dev=_GPU_DEVICE,
    )
    estimated = _relocation_static_tensor_bytes_estimate(scorer, proposals)
    actual = _tensor_tree_bytes(static_tensors)
    print(f"{name}: estimated={estimated} actual={actual}")
    if estimated != actual:
        raise AssertionError(f"{name}: static byte mismatch estimated={estimated} actual={actual}")


def main() -> int:
    for name in ("ibm01", "ibm04"):
        _check(name)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
