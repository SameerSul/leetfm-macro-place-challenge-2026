"""Multiprocessing workers for local-search phases."""

import time

import numpy as np
import torch

from placer.local_search.fields import _congestion_field
from placer.local_search.two_opt import _two_opt_proxy_swap
from placer.plc.loader import _load_plc
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer


def _force_worker_cpu() -> None:
    """Disable every GPU-gated path in this (forked) process.

    Each module reads `_USE_GPU` as its own module global, so flipping config
    alone wouldn't help — patch each holder. Idempotent.
    """
    import placer.config as _cfg
    import placer.routing.apply as _apply
    import placer.local_search.two_opt as _two_opt

    _cfg._USE_GPU = False
    _apply._USE_GPU = False
    _two_opt._USE_GPU = False


def _multiseed_2opt_worker(
    name: str,
    iccad_path: str,
    seed_pl_full_np: np.ndarray,
    seed_score: float,
    seed_tag: str,
    n: int,
    cw: float,
    ch: float,
    sizes_np: np.ndarray,
    hw_np: np.ndarray,
    hh_np: np.ndarray,
    movable_np: np.ndarray,
    deadline_sec: float = 15.0,
    k_neighbors: int = 20,
    max_iters: int = 6,
) -> dict:
    """Run one seed of multi-seed 2-opt in an independent subprocess.

    Each subprocess loads its own benchmark + plc (the C++ object isn't
    picklable, so it can't be shared) and runs the full 2-opt path. Returns a
    picklable result dict. Running the DP seeds in parallel with the main-thread
    "best" seed turns ~60s of sequential 2-opt into ~18s.

    The pool is forked and CUDA contexts don't survive fork, so the worker forces
    itself CPU-only (`_force_worker_cpu`) — else it crashes when the 2-opt kNN or
    routing smoothing touches the GPU. The GPU was only doing the kNN anyway.
    """
    _force_worker_cpu()

    # Lazy import inside the worker (avoid top-level circulars on parallel boot).
    from macro_place.loader import load_benchmark_from_dir

    bm, _ = load_benchmark_from_dir(iccad_path)
    plc = _load_plc(name, bm)

    # Reconstruct seed placement as a torch tensor matching bm.macro_positions.
    pl_full = bm.macro_positions.clone()
    pl_full[:, 0] = torch.tensor(seed_pl_full_np[:, 0], dtype=torch.float32)
    pl_full[:, 1] = torch.tensor(seed_pl_full_np[:, 1], dtype=torch.float32)

    # Establish plc state at the seed placement (also caches routing map).
    _ = _exact_proxy(pl_full, bm, plc)

    # Build the incremental scorer from the seed.
    try:
        scorer = IncrementalScorer(plc, bm, pl_full.cpu().numpy().astype(np.float64))
    except Exception:
        scorer = None

    # S9 per-macro local congestion snapshot (same as the inline main-thread code).
    macro_cong = None
    try:
        nr_g, nc_g = bm.grid_rows, bm.grid_cols
        cell_cong = _congestion_field(plc, nr_g, nc_g)
        if cell_cong is not None:
            cwc, chc = cw / nc_g, ch / nr_g
            hard_xy0 = seed_pl_full_np[:n]
            ci = np.clip((hard_xy0[:, 0] / cwc).astype(np.int64), 0, nc_g - 1)
            ri = np.clip((hard_xy0[:, 1] / chc).astype(np.int64), 0, nr_g - 1)
            macro_cong = cell_cong[ri, ci]
    except Exception:
        macro_cong = None

    # _exact_proxy fallback closure for 2-opt (uses this worker's plc).
    scratch = pl_full.clone()

    def _score_fn(pos_arr, _scr=scratch, _bm=bm, _plc=plc):
        pos32 = torch.from_numpy(np.ascontiguousarray(pos_arr)).float()
        _scr[:n, 0] = pos32[:, 0]
        _scr[:n, 1] = pos32[:, 1]
        return float(_exact_proxy(_scr, _bm, _plc))

    # Run 2-opt (same parameters as the inline path).
    pass_deadline = time.monotonic() + deadline_sec
    work_hard = seed_pl_full_np[:n].copy()
    opt_pos, ac, _, sc = _two_opt_proxy_swap(
        work_hard, sizes_np, hw_np, hh_np, cw, ch, movable_np, n,
        score_fn=_score_fn, initial_score=seed_score,
        k_neighbors=k_neighbors, max_iters=max_iters, deadline=pass_deadline,
        incremental_scorer=scorer, macro_cong=macro_cong,
    )

    # True final rescore on this worker's plc.
    cand_pl = pl_full.clone()
    cand_pl[:n, 0] = torch.tensor(opt_pos[:, 0], dtype=torch.float32)
    cand_pl[:n, 1] = torch.tensor(opt_pos[:, 1], dtype=torch.float32)
    true_final = float(_exact_proxy(cand_pl, bm, plc))

    return {
        "tag": seed_tag,
        "true_final": true_final,
        "opt_pos_full": cand_pl.cpu().numpy(),  # full (num_macros, 2)
        "accept_count": int(ac),
        "score_calls": int(sc),
    }
