"""Verify CUDA relocation proposal scores against exact trial scores.

The `cuda_delta` scorer should now evaluate the same proxy objective as
`IncrementalScorer._trial_at`, but batched through Torch tensors. This verifier
builds legal hard-relocation proposals, scores them with the CUDA-capable path,
and compares those values to exact per-proposal trials.

Usage:
  PYTHONPATH=src \
  uv run python test/verification/_verify_relocation_cuda_delta_scores.py
"""

from __future__ import annotations

import numpy as np
from macro_place.loader import load_benchmark_from_dir

from placer.geometry import separation_matrices
from placer.local_search.fields import _congestion_field
from placer.local_search.relocation import _score_relocation_proposals_cuda_delta
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer


def _collect_proposals(
    *,
    pos: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    plc,
    benchmark,
    incremental_scorer,
    top_hot: int = 5,
    n_targets: int = 5,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    nr, nc = benchmark.grid_rows, benchmark.grid_cols
    cell_cong = _congestion_field(plc, nr, nc)
    if cell_cong is None:
        return [], np.empty(0), np.empty(0)

    n = benchmark.num_hard_macros
    cell_w = cw / nc
    cell_h = ch / nr
    ci_all = np.clip((pos[:n, 0] / cell_w).astype(np.int64), 0, nc - 1)
    ri_all = np.clip((pos[:n, 1] / cell_h).astype(np.int64), 0, nr - 1)
    local_cong = cell_cong[ri_all, ci_all]
    mov_idx = np.where(movable)[0]
    hot = mov_idx[np.argsort(-local_cong[mov_idx])][:top_hot]

    flat = cell_cong.ravel()
    threshold = np.percentile(flat, 55)
    pool = np.where(flat < threshold)[0]
    if pool.size < max(n_targets, 64):
        pool = np.argsort(flat)[: max(n_targets, 64)]
    tgt_c = (pool % nc).astype(np.float64)
    tgt_r = (pool // nc).astype(np.float64)
    tgt_x = (tgt_c + 0.5) * cell_w
    tgt_y = (tgt_r + 0.5) * cell_h
    tgt_cong = flat[pool]

    sep_x_mat, sep_y_mat = separation_matrices(sizes)
    all_idx = np.arange(n)
    proposals: list[dict] = []
    for hot_rank, i_raw in enumerate(hot):
        i = int(i_raw)
        cand = np.where(tgt_cong < local_cong[i] - 1e-9)[0]
        if cand.size == 0:
            continue
        d2 = (tgt_x[cand] - pos[i, 0]) ** 2 + (tgt_y[cand] - pos[i, 1]) ** 2
        cand = cand[np.argsort(d2)][:n_targets]
        mask = all_idx != i
        sxi = sep_x_mat[i, mask]
        syi = sep_y_mat[i, mask]
        ox = pos[mask, 0]
        oy = pos[mask, 1]
        for candidate_rank, target_index in enumerate(cand):
            nx = float(tgt_x[target_index])
            ny = float(tgt_y[target_index])
            if nx - hw[i] < -0.05 or nx + hw[i] > cw + 0.05:
                continue
            if ny - hh[i] < -0.05 or ny + hh[i] > ch + 0.05:
                continue
            if ((np.abs(nx - ox) < sxi + 0.05) & (np.abs(ny - oy) < syi + 0.05)).any():
                continue
            proposals.append(
                {
                    "score": 0.0,
                    "i": i,
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
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0
    movable = bm.get_movable_mask().numpy()[:n]
    base = float(_exact_proxy(bm.macro_positions, bm, plc))
    scorer = IncrementalScorer(plc, bm, pl)

    proposals, local_cong, tgt_cong = _collect_proposals(
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

    _score_relocation_proposals_cuda_delta(
        proposals,
        pos=pl[:n].copy(),
        cw=float(bm.canvas_width),
        ch=float(bm.canvas_height),
        local_cong=local_cong,
        tgt_cong=tgt_cong,
        incremental_scorer=scorer,
    )
    stats = getattr(_score_relocation_proposals_cuda_delta, "last_stats", {})
    for key in (
        "prep_elapsed",
        "tensor_elapsed_estimate",
        "density_updates",
        "macro_route_updates",
        "net_route_updates",
        "hpwl_segments",
        "hpwl_pins",
        "hpwl_rows",
        "grid_dynamic_bytes_per_proposal",
        "hpwl_dynamic_bytes_per_proposal",
    ):
        if key not in stats:
            raise AssertionError(f"{name}: missing scorer stat {key!r}")
    if stats["prep_elapsed"] < 0.0 or stats["tensor_elapsed_estimate"] < 0.0:
        raise AssertionError(f"{name}: negative timing stats {stats}")
    if stats["density_updates"] <= 0 or stats["macro_route_updates"] <= 0:
        raise AssertionError(f"{name}: expected nonzero scatter update stats {stats}")
    if stats["hpwl_segments"] <= 0 or stats["hpwl_pins"] <= 0 or stats["hpwl_rows"] <= 0:
        raise AssertionError(f"{name}: expected nonzero HPWL workload stats {stats}")
    grid_only_dynamic = scorer.grid_row * scorer.grid_col * 10 * np.dtype(np.float32).itemsize
    if stats["dynamic_bytes_per_proposal"] <= grid_only_dynamic:
        raise AssertionError(f"{name}: expected HPWL-aware dynamic byte estimate {stats}")
    if stats["grid_dynamic_bytes_per_proposal"] != grid_only_dynamic:
        raise AssertionError(f"{name}: unexpected grid dynamic byte estimate {stats}")
    if stats["hpwl_dynamic_bytes_per_proposal"] <= 0:
        raise AssertionError(f"{name}: expected HPWL dynamic byte component {stats}")

    max_delta = 0.0
    for proposal in proposals:
        i = int(proposal["i"])
        prep = scorer._prepare_move(i)
        try:
            exact = float(scorer._trial_at(prep, proposal["xy"]))
        finally:
            scorer._revert_prep(prep)
        max_delta = max(max_delta, abs(float(proposal["score"]) - exact))

    print(f"{name}: base={base:.6f} proposals={len(proposals)} max_delta={max_delta:.3e}")
    if max_delta > tol:
        raise AssertionError(f"{name}: cuda_delta score mismatch {max_delta:.3e}")


def main() -> int:
    for name in ("ibm01", "ibm04"):
        _check(name)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
