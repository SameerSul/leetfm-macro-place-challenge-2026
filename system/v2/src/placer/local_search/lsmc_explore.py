"""LSMC exploration phase (GPU-ops.md Stage 2a).

Single-chain Large-Step Markov Chain: kick a fraction of movable hard macros
to random spots, legalize, descend with GPU-scored propose-all relocation,
then accept on the bit-exact proxy of the descended state (zero-temperature,
strict improvement). Runs as the final quality phase after R2.

Stage 2c (batched multi-chain Torch on one GPU) can build on this. This
version keeps acceptance exact, so no approximate-cost handoff is needed.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from placer.legalize.spiral import _will_legalize
from placer.local_search.fields import _congestion_field
from placer.local_search.relocation import _relocation_moves, _soft_relocation_moves
from placer.scoring.incremental import IncrementalScorer


def _explore_enabled(gpu_backend: str) -> bool:
    """V2_GPU_EXPLORE: '1' forces on, '0' disables, default/auto requires CUDA.

    Default-on under CUDA since the 2026-06-12 paired gate (2/2 seeds,
    mean -0.0042; see ISSUES.md S17).
    """
    raw = os.environ.get("V2_GPU_EXPLORE", "auto").strip().lower()
    if raw in {"1", "true", "on"}:
        return True
    if raw in {"0", "false", "off"}:
        return False
    return gpu_backend == "cuda"


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


def _congestion_kick(
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
    plc,
    benchmark,
) -> np.ndarray:
    """Move a hot-congestion subset of movable hard macros toward cold cells."""
    field = _congestion_field(plc, benchmark.grid_rows, benchmark.grid_cols)
    if field is None:
        return _kick(hard_xy, sizes, hw, hh, cw, ch, movable, n, kick_ratio, rng, deadline)

    movable_idx = np.flatnonzero(movable[:n])
    if movable_idx.size == 0:
        return hard_xy.copy()

    n_kick = max(1, math.ceil(kick_ratio * movable_idx.size))
    n_kick = min(n_kick, movable_idx.size)
    cell_w = cw / benchmark.grid_cols
    cell_h = ch / benchmark.grid_rows
    ci = np.clip((hard_xy[:n, 0] / cell_w).astype(np.int64), 0, benchmark.grid_cols - 1)
    ri = np.clip((hard_xy[:n, 1] / cell_h).astype(np.int64), 0, benchmark.grid_rows - 1)
    local = field[ri, ci]

    # Sample from the hot half, weighted by local congestion. This keeps diversity
    # while biasing kicks toward macros that are actually sitting in routed hot spots.
    hot_pool_size = min(movable_idx.size, max(n_kick * 4, n_kick))
    hot_pool = movable_idx[np.argsort(-local[movable_idx])[:hot_pool_size]]
    weights = local[hot_pool] - float(local[hot_pool].min()) + 1e-9
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = None
    else:
        weights = weights / float(weights.sum())
    picks = rng.choice(hot_pool, size=n_kick, replace=False, p=weights)

    flat = field.ravel()
    cold_count = min(flat.size, max(64, n_kick * 16))
    cold = np.argsort(flat)[:cold_count]
    tgt_c = (cold % benchmark.grid_cols).astype(np.float64)
    tgt_r = (cold // benchmark.grid_cols).astype(np.float64)
    tgt_x = (tgt_c + 0.5) * cell_w
    tgt_y = (tgt_r + 0.5) * cell_h

    kicked = hard_xy.copy()
    for i in picks:
        # Try several cold cells so large macros do not collapse to clipped edges.
        for _attempt in range(8):
            t = int(rng.integers(0, cold_count))
            x = float(np.clip(tgt_x[t], hw[i], cw - hw[i]))
            y = float(np.clip(tgt_y[t], hh[i], ch - hh[i]))
            if np.isfinite(x) and np.isfinite(y):
                kicked[i, 0] = x
                kicked[i, 1] = y
                break

    return _will_legalize(kicked, movable, sizes, hw, hh, cw, ch, n, deadline=deadline)


def _select_kick(
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
    plc,
    benchmark,
    mode: str,
    iteration: int,
) -> np.ndarray:
    """Dispatch the LSMC large step."""
    if mode == "random":
        return _kick(hard_xy, sizes, hw, hh, cw, ch, movable, n, kick_ratio, rng, deadline)
    if mode == "congestion":
        return _congestion_kick(
            hard_xy, sizes, hw, hh, cw, ch, movable, n, kick_ratio,
            rng, deadline, plc, benchmark,
        )
    # Mixed mode preserves some basin diversity while spending every other kick
    # on congestion escape, the dominant term in this objective.
    if iteration % 2 == 0:
        return _congestion_kick(
            hard_xy, sizes, hw, hh, cw, ch, movable, n, kick_ratio,
            rng, deadline, plc, benchmark,
        )
    return _kick(hard_xy, sizes, hw, hh, cw, ch, movable, n, kick_ratio, rng, deadline)


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
    soft_hw: np.ndarray | None = None,
    soft_hh: np.ndarray | None = None,
    soft_movable: np.ndarray | None = None,
    n_soft: int = 0,
) -> tuple[torch.Tensor, float, int, int]:
    """Run the kick/descent/accept loop. Returns (pl, score, iters, accepts).

    Each iteration: kick from the incumbent best, build a fresh
    IncrementalScorer on the kicked state (never patch the pre-kick scorer),
    descend with one congestion + one density propose-all relocation pass,
    then accept only on strict true-proxy improvement. Early exit after
    V2_GPU_EXPLORE_FAILS consecutive non-improving iterations.
    """
    kick_ratio = float(os.environ.get("V2_GPU_EXPLORE_KICK", "0.02"))
    max_fails = int(os.environ.get("V2_GPU_EXPLORE_FAILS", "5"))
    seed = int(os.environ.get("V2_GPU_EXPLORE_SEED", "0"))
    kick_mode = os.environ.get("V2_GPU_EXPLORE_KICK_MODE", "mixed").strip().lower()
    if kick_mode not in {"random", "congestion", "mixed"}:
        kick_mode = "mixed"
    rng = np.random.default_rng(seed)

    # Stage 2b kick pre-screen: score a batch of kicks and descend only the
    # best one (descent dominates iteration cost; kick scoring is cheap).
    # Batch size adapts down when scoring one kick is slow on large grids.
    prescreen = max(1, int(os.environ.get("V2_GPU_EXPLORE_PRESCREEN", "8")))

    # Stage 2c multi-chain (single GPU): run CHAINS independent kick/descent
    # trajectories from the SAME incumbent, each with its own RNG and an equal
    # share of the budget, and keep the best. CHAINS=1 is the shipped 2b path.
    # This single-process keep-best form tests whether cross-chain diversity
    # beats one deeper chain before the GPU-batched-descent refactor.
    chains = max(1, int(os.environ.get("V2_GPU_EXPLORE_CHAINS", "1")))
    per_chain_s = time_budget_s / chains

    start_pl, start_score = best_pl, best_score
    iters = 0
    accepts = 0

    for chain in range(chains):
        rng = np.random.default_rng(seed + 1009 * chain)
        wall = time.monotonic() + per_chain_s
        c_best_pl, c_best_score = start_pl, start_score
        fails = 0
        if kick_mode != "random":
            c_best_score = float(exact_proxy(c_best_pl, benchmark, plc))

        while time.monotonic() < wall and fails < max_fails:
            iters += 1
            hard_xy = c_best_pl[:n].detach().cpu().numpy().astype(np.float64)

            kicked = None
            kick_score = float("inf")
            cand = c_best_pl.clone()
            for _b in range(prescreen):
                trial = _select_kick(
                    hard_xy, sizes, hw, hh, cw, ch, movable, n,
                    kick_ratio, rng, deadline=wall, plc=plc, benchmark=benchmark,
                    mode=kick_mode, iteration=iters + _b,
                )
                t_ks = time.monotonic()
                cand[:n, 0] = torch.tensor(trial[:, 0], dtype=torch.float32)
                cand[:n, 1] = torch.tensor(trial[:, 1], dtype=torch.float32)
                trial_score = float(exact_proxy(cand, benchmark, plc))
                t_ks = time.monotonic() - t_ks
                if trial_score < kick_score:
                    kicked, kick_score = trial, trial_score
                # Keep the pre-screen a small fraction of the slice.
                if time.monotonic() + t_ks > wall - t_ks or t_ks > per_chain_s / (2 * prescreen):
                    break
            if kicked is None:
                break
            cand[:n, 0] = torch.tensor(kicked[:, 0], dtype=torch.float32)
            cand[:n, 1] = torch.tensor(kicked[:, 1], dtype=torch.float32)

            cand_np = cand.detach().cpu().numpy().astype(np.float64)
            scorer = IncrementalScorer(plc, benchmark, cand_np)

            # Descend to a local optimum: alternate hard cong/density propose-all
            # passes, then soft relocation passes (most of the congestion-dominated
            # proxy lives in soft placement), until a full round stops improving.
            desc_xy = kicked
            desc_soft = None
            if n_soft > 0:
                desc_soft = cand[n:n + n_soft].detach().cpu().numpy().astype(np.float64)
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
                if desc_soft is not None:
                    for use_density in (False, True):
                        if time.monotonic() >= wall:
                            break
                        desc_soft, _, desc_score = _soft_relocation_moves(
                            desc_soft, soft_hw, soft_hh, cw, ch, n, plc,
                            benchmark, scorer, desc_score,
                            deadline=wall,
                            top_hot=1024, n_targets=4,
                            soft_movable=soft_movable,
                            use_density=use_density,
                        )
                if round_start - desc_score < 1e-4:
                    break

            cand[:n, 0] = torch.tensor(desc_xy[:, 0], dtype=torch.float32)
            cand[:n, 1] = torch.tensor(desc_xy[:, 1], dtype=torch.float32)
            if desc_soft is not None:
                cand[n:n + n_soft, 0] = torch.tensor(desc_soft[:, 0], dtype=torch.float32)
                cand[n:n + n_soft, 1] = torch.tensor(desc_soft[:, 1], dtype=torch.float32)
            true_score = float(exact_proxy(cand, benchmark, plc))

            if true_score < c_best_score - 1e-6:
                c_best_pl = cand
                c_best_score = true_score
                accepts += 1
                fails = 0
                if log:
                    log(f"  LSMC chain {chain + 1}/{chains} iter {iters}: "
                        f"kick={kick_score:.4f} descended={true_score:.4f} ACCEPT")
            else:
                fails += 1
                if kick_mode != "random":
                    float(exact_proxy(c_best_pl, benchmark, plc))
                if log:
                    log(f"  LSMC chain {chain + 1}/{chains} iter {iters}: "
                        f"kick={kick_score:.4f} descended={true_score:.4f} "
                        f"reject ({fails}/{max_fails})")

        # Merge this chain's best into the global best (keep-best across chains).
        if c_best_score < best_score - 1e-6:
            best_pl, best_score = c_best_pl, c_best_score

    return best_pl, best_score, iters, accepts
