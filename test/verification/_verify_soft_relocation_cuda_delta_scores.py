"""Verify CUDA soft-relocation proposal scores against exact trial scores.

Usage:
  PYTHONPATH=src \
  uv run python test/verification/_verify_soft_relocation_cuda_delta_scores.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from macro_place.loader import load_benchmark_from_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.local_search.fields import _congestion_field
from placer.local_search.relocation import _score_relocation_proposals_cuda_delta
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer


def _collect_soft_proposals(
    *,
    soft_pos: np.ndarray,
    soft_hw: np.ndarray,
    soft_hh: np.ndarray,
    cw: float,
    ch: float,
    soft_movable: np.ndarray,
    plc,
    benchmark,
    incremental_scorer,
    top_hot: int = 6,
    n_targets: int = 5,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    cell_cong = _congestion_field(plc, nr, nc)
    if cell_cong is None:
        return [], np.empty(0), np.empty(0)

    cell_w = cw / nc
    cell_h = ch / nr
    ci = np.clip((soft_pos[:, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri = np.clip((soft_pos[:, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_cong[ri, ci]
    movable_idx = np.where(soft_movable)[0]
    hot = movable_idx[np.argsort(-local_cong[movable_idx])][:top_hot]

    flat = cell_cong.ravel()
    threshold = np.percentile(flat, 55)
    pool = np.where(flat < threshold)[0]
    if pool.size < max(n_targets, 64):
        pool = np.argsort(flat)[: max(n_targets, 64)]
    tgt_x = ((pool % nc).astype(np.float64) + 0.5) * cell_w
    tgt_y = ((pool // nc).astype(np.float64) + 0.5) * cell_h
    tgt_cong = flat[pool]

    proposals: list[dict] = []
    for hot_rank, k_raw in enumerate(hot):
        k = int(k_raw)
        cand = np.where(tgt_cong < local_cong[k] - 1e-9)[0]
        if cand.size == 0:
            continue
        d2 = (tgt_x[cand] - soft_pos[k, 0]) ** 2 + (tgt_y[cand] - soft_pos[k, 1]) ** 2
        cand = cand[np.argsort(d2)][:n_targets]
        for candidate_rank, target_index in enumerate(cand):
            nx = float(np.clip(tgt_x[target_index], soft_hw[k], cw - soft_hw[k]))
            ny = float(np.clip(tgt_y[target_index], soft_hh[k], ch - soft_hh[k]))
            proposals.append(
                {
                    "score": 0.0,
                    "i": k,
                    "module_idx": int(incremental_scorer.soft_indices[k]),
                    "old_xy": (float(soft_pos[k, 0]), float(soft_pos[k, 1])),
                    "macro_route": False,
                    "hot_rank": int(hot_rank),
                    "candidate_rank": int(candidate_rank),
                    "target_index": int(target_index),
                    "xy": (nx, ny),
                }
            )
    return proposals, local_cong, tgt_cong


def _check(name: str, tol: float = 5e-4) -> None:
    bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    _patch_plc_congestion(plc, bm)

    pl = bm.macro_positions.numpy().astype(np.float64)
    n = bm.num_hard_macros
    ns = bm.num_soft_macros
    if ns == 0:
        raise AssertionError(f"{name}: benchmark has no soft macros")

    soft_sizes = bm.macro_sizes[n : n + ns].numpy().astype(np.float64)
    soft_hw = soft_sizes[:, 0] / 2.0
    soft_hh = soft_sizes[:, 1] / 2.0
    soft_movable = bm.get_movable_mask().numpy()[n : n + ns]
    base = float(_exact_proxy(bm.macro_positions, bm, plc))
    scorer = IncrementalScorer(plc, bm, pl)

    soft_pos = pl[n : n + ns].copy()
    proposals, local_cong, tgt_cong = _collect_soft_proposals(
        soft_pos=soft_pos,
        soft_hw=soft_hw,
        soft_hh=soft_hh,
        cw=float(bm.canvas_width),
        ch=float(bm.canvas_height),
        soft_movable=soft_movable,
        plc=plc,
        benchmark=bm,
        incremental_scorer=scorer,
    )
    if not proposals:
        raise AssertionError(f"{name}: no soft proposals collected")

    _score_relocation_proposals_cuda_delta(
        proposals,
        pos=soft_pos,
        cw=float(bm.canvas_width),
        ch=float(bm.canvas_height),
        local_cong=local_cong,
        tgt_cong=tgt_cong,
        incremental_scorer=scorer,
    )
    stats = getattr(_score_relocation_proposals_cuda_delta, "last_stats", {})
    if stats.get("macro_route_updates", -1) != 0:
        raise AssertionError(f"{name}: soft proposals should not update macro routes {stats}")
    if stats.get("density_updates", 0) <= 0 or stats.get("net_route_updates", 0) <= 0:
        raise AssertionError(f"{name}: expected density and net-route work {stats}")
    if stats.get("hpwl_segments", 0) <= 0 or stats.get("hpwl_pins", 0) <= 0:
        raise AssertionError(f"{name}: expected HPWL workload {stats}")

    max_delta = 0.0
    for proposal in proposals:
        k = int(proposal["i"])
        prep = scorer._prepare_move_soft(k)
        try:
            exact = float(scorer._trial_at_soft(prep, proposal["xy"]))
        finally:
            scorer._revert_prep_soft(prep)
        max_delta = max(max_delta, abs(float(proposal["score"]) - exact))

    print(f"{name}: base={base:.6f} proposals={len(proposals)} max_delta={max_delta:.3e}")
    if max_delta > tol:
        raise AssertionError(f"{name}: cuda_delta soft score mismatch {max_delta:.3e}")


def main() -> int:
    for name in ("ibm01", "ibm04"):
        _check(name)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
