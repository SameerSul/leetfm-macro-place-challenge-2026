"""LSMC exploration phase (GPU-ops.md Stage 2a).

Large-Step Markov Chain: kick hard macros, legalize, build a fresh
IncrementalScorer, descend with hard/soft relocation, then accept on the
bit-exact proxy of the descended state (zero-temperature, strict improvement).
Runs as the final quality phase after R2.

The shipped loop is serial and exact-gated. `V2_GPU_EXPLORE=auto` uses CUDA
availability as the default enable condition, while the optional batched
multi-chain GPU rewrite remains a dormant future path.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from placer.legalize.spiral import _will_legalize
from placer.local_search.clusters import (
    cluster_max_fanout,
    cluster_min_edge,
    derive_cluster_softs,
    derive_hard_clusters,
)
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


def _cluster_kick(
    hard_xy: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    movable: np.ndarray,
    n: int,
    clusters: "dict[int, np.ndarray]",
    rng: np.random.Generator,
    deadline: float,
    mode: str = "gather",
    max_size: int = 32,
    soft_xy: "np.ndarray | None" = None,
    soft_hw: "np.ndarray | None" = None,
    soft_hh: "np.ndarray | None" = None,
    cluster_softs: "dict[int, np.ndarray] | None" = None,
) -> "tuple[np.ndarray, np.ndarray | None] | None":
    """Kick one connectivity cluster as a unit, then legalize.

    Picks a random cluster of >= 2 movable members and either:
      - "gather": seed all members at one random in-bounds anchor (with tiny
        jitter); the legalizer's nearest-free-cell search then packs them into a
        compact blob — directly testing "keep connected macros together".
      - "translate": apply one feasible rigid (dx, dy) so the subsystem's current
        internal arrangement is preserved and relocated as a whole.
      - "both": pick gather or translate at random for this kick.
    When `cluster_softs`/`soft_xy` are given, the cluster's connected soft macros
    receive the SAME transform (translate offset or gather anchor), so hard↔soft
    proximity is preserved; softs may overlap so they only need clipping, no
    legalization. Returns (hard_legalized, soft_new) — soft_new is None when no
    softs are co-moved — or None when no usable cluster exists (caller falls back
    to the random kick). The LSMC exact post-descent gate decides acceptance.
    """
    if not clusters:
        return None
    if mode == "both":
        mode = "translate" if rng.random() < 0.5 else "gather"
    co_soft = cluster_softs is not None and soft_xy is not None
    cluster_ids = list(clusters.keys())
    rng.shuffle(cluster_ids)
    for cid in cluster_ids:
        members = clusters[cid]
        members = members[movable[members]]
        if members.size < 2 or members.size > max_size:
            continue
        x = hard_xy[members, 0]
        y = hard_xy[members, 1]
        kicked = hard_xy.copy()
        # Soft indices to co-move (placement space -> local soft index).
        s_local = None
        if co_soft:
            s_arr = cluster_softs.get(cid)
            if s_arr is not None and s_arr.size:
                s_local = s_arr - n
        soft_new = soft_xy.copy() if co_soft else None
        if mode == "translate":
            # Feasible rigid translation: keep every member within [hw, cw-hw].
            tx_lo = float(np.max(hw[members] - x))
            tx_hi = float(np.min((cw - hw[members]) - x))
            ty_lo = float(np.max(hh[members] - y))
            ty_hi = float(np.min((ch - hh[members]) - y))
            if tx_hi < tx_lo or ty_hi < ty_lo:
                continue
            tx = rng.uniform(tx_lo, tx_hi)
            ty = rng.uniform(ty_lo, ty_hi)
            kicked[members, 0] = x + tx
            kicked[members, 1] = y + ty
            if s_local is not None:
                soft_new[s_local, 0] = np.clip(soft_xy[s_local, 0] + tx,
                                               soft_hw[s_local], cw - soft_hw[s_local])
                soft_new[s_local, 1] = np.clip(soft_xy[s_local, 1] + ty,
                                               soft_hh[s_local], ch - soft_hh[s_local])
        else:
            # Gather: seed all members at one anchor; legalize packs them tight.
            mhw = float(hw[members].max())
            mhh = float(hh[members].max())
            ax = rng.uniform(mhw, cw - mhw)
            ay = rng.uniform(mhh, ch - mhh)
            kicked[members, 0] = np.clip(ax + rng.normal(0.0, mhw, members.size),
                                         hw[members], cw - hw[members])
            kicked[members, 1] = np.clip(ay + rng.normal(0.0, mhh, members.size),
                                         hh[members], ch - hh[members])
            if s_local is not None:
                soft_new[s_local, 0] = np.clip(
                    ax + rng.normal(0.0, mhw, s_local.size),
                    soft_hw[s_local], cw - soft_hw[s_local])
                soft_new[s_local, 1] = np.clip(
                    ay + rng.normal(0.0, mhh, s_local.size),
                    soft_hh[s_local], ch - soft_hh[s_local])
        legal_hard = _will_legalize(kicked, movable, sizes, hw, hh, cw, ch, n,
                                    deadline=deadline)
        return legal_hard, soft_new
    return None


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

    # Cluster-coherent kicks: with probability cluster_p, kick a whole derived
    # connectivity cluster as a rigid unit instead of scattering random macros.
    # Default off (prototype, env-gated). The exact post-descent gate is
    # unchanged, so this only changes which basins are explored.
    cluster_p = float(os.environ.get("V2_GPU_EXPLORE_CLUSTER_P", "0.0"))
    cluster_mode = os.environ.get("V2_GPU_EXPLORE_CLUSTER_MODE", "gather").strip().lower()
    cluster_maxsz = max(2, int(os.environ.get("V2_GPU_EXPLORE_CLUSTER_MAXSZ", "32")))
    co_soft = os.environ.get("V2_GPU_EXPLORE_CLUSTER_SOFT", "1").strip().lower() \
        not in {"0", "false", "off"}
    clusters: "dict[int, np.ndarray]" = {}
    cluster_softs: "dict[int, np.ndarray]" = {}
    if cluster_p > 0.0:
        try:
            labels, clusters = derive_hard_clusters(
                plc, n, n_soft=n_soft, max_fanout=cluster_max_fanout(),
                min_edge=cluster_min_edge(),
            )
            if co_soft and clusters and n_soft > 0 and soft_hw is not None:
                cluster_softs = derive_cluster_softs(
                    plc, n, n_soft, labels, max_fanout=cluster_max_fanout(),
                )
        except Exception:
            clusters = {}
            cluster_softs = {}
        if log:
            log(f"  LSMC cluster kicks: p={cluster_p} clusters={len(clusters)} "
                f"co_soft={'on' if cluster_softs else 'off'}")

    start_pl, start_score = best_pl, best_score
    iters = 0
    accepts = 0

    for chain in range(chains):
        rng = np.random.default_rng(seed + 1009 * chain)
        wall = time.monotonic() + per_chain_s
        c_best_pl, c_best_score = start_pl, start_score
        fails = 0

        while time.monotonic() < wall and fails < max_fails:
            iters += 1
            hard_xy = c_best_pl[:n].detach().cpu().numpy().astype(np.float64)

            kicked = None
            kicked_soft = None
            kick_score = float("inf")
            cand = c_best_pl.clone()
            soft_inc = (c_best_pl[n:n + n_soft].detach().cpu().numpy().astype(np.float64)
                        if n_soft > 0 else None)
            for _b in range(prescreen):
                trial = None
                trial_soft = None
                if clusters and rng.random() < cluster_p:
                    res = _cluster_kick(hard_xy, sizes, hw, hh, cw, ch,
                                        movable, n, clusters, rng, deadline=wall,
                                        mode=cluster_mode, max_size=cluster_maxsz,
                                        soft_xy=soft_inc, soft_hw=soft_hw,
                                        soft_hh=soft_hh, cluster_softs=cluster_softs)
                    if res is not None:
                        trial, trial_soft = res
                if trial is None:
                    trial = _kick(hard_xy, sizes, hw, hh, cw, ch, movable, n,
                                  kick_ratio, rng, deadline=wall)
                t_ks = time.monotonic()
                cand[:n, 0] = torch.tensor(trial[:, 0], dtype=torch.float32)
                cand[:n, 1] = torch.tensor(trial[:, 1], dtype=torch.float32)
                # Softs: co-moved set for a cluster kick, else reset to incumbent
                # (cand is reused across prescreen trials).
                if n_soft > 0:
                    s_src = trial_soft if trial_soft is not None else soft_inc
                    cand[n:n + n_soft, 0] = torch.tensor(s_src[:, 0], dtype=torch.float32)
                    cand[n:n + n_soft, 1] = torch.tensor(s_src[:, 1], dtype=torch.float32)
                trial_score = float(exact_proxy(cand, benchmark, plc))
                t_ks = time.monotonic() - t_ks
                if trial_score < kick_score:
                    kicked, kicked_soft, kick_score = trial, trial_soft, trial_score
                # Keep the pre-screen a small fraction of the slice.
                if time.monotonic() + t_ks > wall - t_ks or t_ks > per_chain_s / (2 * prescreen):
                    break
            if kicked is None:
                break
            cand[:n, 0] = torch.tensor(kicked[:, 0], dtype=torch.float32)
            cand[:n, 1] = torch.tensor(kicked[:, 1], dtype=torch.float32)
            if n_soft > 0:
                s_src = kicked_soft if kicked_soft is not None else soft_inc
                cand[n:n + n_soft, 0] = torch.tensor(s_src[:, 0], dtype=torch.float32)
                cand[n:n + n_soft, 1] = torch.tensor(s_src[:, 1], dtype=torch.float32)

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
                if log:
                    log(f"  LSMC chain {chain + 1}/{chains} iter {iters}: "
                        f"kick={kick_score:.4f} descended={true_score:.4f} "
                        f"reject ({fails}/{max_fails})")

        # Merge this chain's best into the global best (keep-best across chains).
        if c_best_score < best_score - 1e-6:
            best_pl, best_score = c_best_pl, c_best_score

    return best_pl, best_score, iters, accepts
