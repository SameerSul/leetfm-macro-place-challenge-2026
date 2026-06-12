"""LSMC exploration phase (GPU-ops.md Stage 2a).

Single-chain Large-Step Markov Chain: kick a fraction of movable hard macros
to random spots, legalize, descend with GPU-scored propose-all relocation,
then accept on the bit-exact proxy of the descended state (zero-temperature,
strict improvement). Runs between seed selection and R2.

Stage 2b (batched multi-chain Torch, island model) builds on this once the
multi-GPU hardware is available; this version keeps acceptance exact, so no
approximate-cost handoff is needed.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from placer.legalize.spiral import _will_legalize
from placer.local_search.relocation import _relocation_moves
from placer.scoring.incremental import IncrementalScorer


def _explore_enabled(gpu_backend: str) -> bool:
    """V2_GPU_EXPLORE: '1' forces on, 'auto' requires the CUDA backend, default off."""
    raw = os.environ.get("V2_GPU_EXPLORE", "").strip().lower()
    if raw in {"1", "true", "on"}:
        return True
    if raw in {"auto", "cuda", "gpu"}:
        return gpu_backend == "cuda"
    return False


def _kick(
    hard_xy: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    n: int,
    kick_ratio: float,
    rng: np.random.Generator,
    deadline: float,
) -> np.ndarray:
    """Relocate a random kick_ratio subset of movable hard macros, then legalize."""
    movable_idx = np.flatnonzero(movable[:n])
    n_kick = max(1, math.ceil(kick_ratio * movable_idx.size))
    picks = rng.choice(movable_idx, size=min(n_kick, movable_idx.size), replace=False)

    kicked = hard_xy.copy()
    kicked[picks, 0] = rng.uniform(hw[picks], cw - hw[picks])
    kicked[picks, 1] = rng.uniform(hh[picks], ch - hh[picks])
    return _will_legalize(kicked, movable, sizes, hw, hh, cw, ch, n, deadline=deadline)


def _lsmc_explore(
    best_pl: torch.Tensor,
    best_score: float,
    benchmark,
    plc,
    exact_proxy,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    n: int,
    time_budget_s: float,
    log=None,
) -> tuple[torch.Tensor, float, int, int]:
    """Run the kick/descent/accept loop. Returns (pl, score, iters, accepts).

    Each iteration: kick from the incumbent best, build a fresh
    IncrementalScorer on the kicked state (never patch the pre-kick scorer),
    descend with one congestion + one density propose-all relocation pass,
    then accept only on strict true-proxy improvement. Early exit after
    V2_GPU_EXPLORE_FAILS consecutive non-improving iterations.
    """
    kick_ratio = float(os.environ.get("V2_GPU_EXPLORE_KICK", "0.10"))
    max_fails = int(os.environ.get("V2_GPU_EXPLORE_FAILS", "5"))
    seed = int(os.environ.get("V2_GPU_EXPLORE_SEED", "0"))
    rng = np.random.default_rng(seed)

    wall = time.monotonic() + time_budget_s
    iters = 0
    accepts = 0
    fails = 0

    while time.monotonic() < wall and fails < max_fails:
        iters += 1
        hard_xy = best_pl[:n].detach().cpu().numpy().astype(np.float64)
        kicked = _kick(hard_xy, sizes, hw, hh, cw, ch, movable, n,
                       kick_ratio, rng, deadline=wall)

        cand = best_pl.clone()
        cand[:n, 0] = torch.tensor(kicked[:, 0], dtype=torch.float32)
        cand[:n, 1] = torch.tensor(kicked[:, 1], dtype=torch.float32)
        kick_score = float(exact_proxy(cand, benchmark, plc))

        cand_np = cand.detach().cpu().numpy().astype(np.float64)
        scorer = IncrementalScorer(plc, benchmark, cand_np)

        # Descend to a local optimum: alternate cong/density propose-all
        # passes until a full round stops improving or the wall is hit.
        desc_xy = kicked
        desc_score = kick_score
        while time.monotonic() < wall:
            round_start = desc_score
            for use_density in (False, True):
                if time.monotonic() >= wall:
                    break
                desc_xy, _, desc_score = _relocation_moves(
                    desc_xy, sizes, hw, hh, cw, ch, movable, n, plc,
                    benchmark, scorer, desc_score,
                    deadline=wall,
                    use_density=use_density,
                    propose_all=True,
                )
            if round_start - desc_score < 1e-4:
                break

        cand[:n, 0] = torch.tensor(desc_xy[:, 0], dtype=torch.float32)
        cand[:n, 1] = torch.tensor(desc_xy[:, 1], dtype=torch.float32)
        true_score = float(exact_proxy(cand, benchmark, plc))

        if true_score < best_score - 1e-6:
            best_pl = cand
            best_score = true_score
            accepts += 1
            fails = 0
            if log:
                log(f"  LSMC iter {iters}: kick={kick_score:.4f} "
                    f"descended={true_score:.4f} ACCEPT")
        else:
            fails += 1
            if log:
                log(f"  LSMC iter {iters}: kick={kick_score:.4f} "
                    f"descended={true_score:.4f} reject ({fails}/{max_fails})")

    return best_pl, best_score, iters, accepts
