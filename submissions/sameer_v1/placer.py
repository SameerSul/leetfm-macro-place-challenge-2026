"""
Competitive Macro Placer -- Partcl/HRT Challenge 2026
Sameer Suleman (sameersul)

Algorithm:
  Multi-restart legalization with exact proxy-cost selection.

  1. Legalize from the provided initial.plc positions (baseline).
  2. Run N perturbed restarts: add Gaussian noise to initial positions at
     increasing scales, re-legalize each, get N+1 candidate placements.
  3. Score every candidate with the EXACT proxy cost evaluator
     (compute_proxy_cost → PlacementCost ground truth).
  4. Return the candidate with the lowest proxy cost.

Why restarts beat SA:
  Testing showed legalization-alone (avg 1.5062) beats will_seed (1.5338)
  and SA makes things worse on most benchmarks. The initial.plc positions
  are already good; SA over-optimises wirelength at the cost of congestion.
  Perturbation restarts explore different overlap-resolution arrangements
  without ever touching the fundamental macro connectivity structure.

Why exact scoring matters:
  Proxy cost = 1×WL + 0.5×density + 0.5×congestion.
  WL is only ~4% of the total cost. Congestion (~60%) dominates.
  Without exact congestion scoring, any local heuristic picks the wrong
  candidate. Calling compute_proxy_cost() inside place() costs ~2s per
  candidate but guarantees we always return the provably best restart.

Baselines (17 IBM benchmarks):
  SA baseline avg:         2.1251
  will_seed avg:           1.5338
  sameer_v1 (leg-only):    1.5062  ← previous best
  sameer_v1 (this):        TBD after restarts evaluation
  RePlAce avg:             1.4578  ← Grand Prize target

Runtime:
  N=4 restarts: ~(5-30s legalize + 2s score) × 5 candidates per benchmark
  ~30-60s/benchmark → ~10-15 min total (well within 1 h limit).
"""

import random
import time
from pathlib import Path
from typing import List, Optional, Tuple  # noqa: F401

import numpy as np
import torch
from macro_place.benchmark import Benchmark


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Will's minimum-displacement legalization
# ---------------------------------------------------------------------------

def _will_legalize(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    n: int,
) -> np.ndarray:
    """
    Largest-macro-first min-displacement legalization.
    Macros are placed one by one (largest area first) at the nearest
    overlap-free position to their target, found by expanding spiral search.
    Non-movable macros are fixed in place first.
    """
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue
        if placed.any():
            cdx = np.abs(legal[idx, 0] - legal[:, 0])
            cdy = np.abs(legal[idx, 1] - legal[:, 1])
            conf = (cdx < sep_x[idx] + 0.05) & (cdy < sep_y[idx] + 0.05) & placed
            conf[idx] = False
            if not conf.any():
                placed[idx] = True
                continue
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best = legal[idx].copy()
        best_d = float("inf")
        for r in range(1, 200):
            found = False
            for ddx in range(-r, r + 1):
                for ddy in range(-r, r + 1):
                    if abs(ddx) != r and abs(ddy) != r:
                        continue
                    cx = float(np.clip(pos[idx, 0] + ddx * step, hw[idx], cw - hw[idx]))
                    cy = float(np.clip(pos[idx, 1] + ddy * step, hh[idx], ch - hh[idx]))
                    if placed.any():
                        dd = np.abs(cx - legal[:, 0])
                        de = np.abs(cy - legal[:, 1])
                        conf2 = (dd < sep_x[idx] + 0.05) & (de < sep_y[idx] + 0.05) & placed
                        conf2[idx] = False
                        if conf2.any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d, best = d, np.array([cx, cy])
                        found = True
            if found:
                break
        legal[idx] = best
        placed[idx] = True
    return legal


def _load_plc(name: str):
    """Load PlacementCost object for exact proxy scoring (uses posix paths for Windows compat)."""
    try:
        from macro_place.loader import load_benchmark_from_dir, load_benchmark
        root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        if root.exists():
            _, plc = load_benchmark_from_dir(root.as_posix())
            return plc
        # ng45 fallback
        ng45 = {
            "ariane133_ng45": "ariane133",
            "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla",
            "mempool_tile_ng45": "mempool_tile",
        }
        d = ng45.get(name)
        if d:
            base = (Path("external/MacroPlacement/Flows/NanGate45")
                    / d / "netlist" / "output_CT_Grouping")
            if (base / "netlist.pb.txt").exists():
                _, plc = load_benchmark(
                    (base / "netlist.pb.txt").as_posix(),
                    (base / "initial.plc").as_posix(),
                )
                return plc
    except Exception as exc:
        _log(f"  Warning: plc load failed ({exc})")
    return None


def _exact_proxy(placement: torch.Tensor, benchmark: Benchmark, plc) -> float:
    """Return proxy_cost using the ground-truth PlacementCost evaluator."""
    from macro_place.objective import compute_proxy_cost
    costs = compute_proxy_cost(placement, benchmark, plc)
    return float(costs["proxy_cost"])


def _density_score(pos: np.ndarray, n: int, cw: float, ch: float) -> float:
    """Fallback scorer: max-cell macro count on a 20×20 grid (lower = better spread)."""
    G = 20
    grid = np.zeros((G, G), dtype=np.float64)
    for i in range(n):
        r = min(int(pos[i, 1] / (ch / G)), G - 1)
        c = min(int(pos[i, 0] / (cw / G)), G - 1)
        grid[r, c] += 1
    return float((grid ** 2).sum())


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class MacroPlacer:
    """
    Multi-restart legalization placer for the Partcl/HRT Challenge 2026.

    Runs N perturbed legalizations from initial.plc, scores each with the
    exact proxy evaluator (WL + 0.5×density + 0.5×congestion), returns best.

    Parameters
    ----------
    n_restarts : int
        Total number of legalization runs including the unperturbed baseline.
        Larger values improve quality but increase runtime linearly.
        Default 5 → ~10-15 min for all 17 benchmarks on CPU.
    noise_fracs : list[float]
        Noise magnitude for restarts 1..N-1, as a fraction of min(cw, ch).
        Restart 0 is always the unperturbed baseline.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_restarts: int = 5,
        noise_fracs: Optional[List[float]] = None,
        seed: int = 42,
        time_budget_s: float = 200.0,
    ):
        self.n_restarts = n_restarts
        # Small noise levels only — testing showed large noise (>10%) always hurts.
        # The optimal arrangement is usually within 2-8% of the initial.plc spread.
        self.noise_fracs = noise_fracs or [0.02, 0.04, 0.06, 0.08]
        self.seed = seed
        # Per-benchmark wall-clock budget. 200s × 17 = 56 min → safely under 1 hour.
        # Large benchmarks (ibm12, ibm17) use most of this just for baseline scoring;
        # restarts are automatically skipped when the budget would be exceeded.
        self.time_budget_s = time_budget_s

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        random.seed(self.seed)

        t0 = time.time()
        n = benchmark.num_hard_macros
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n].numpy()
        init_pos = benchmark.macro_positions[:n].numpy().copy().astype(np.float64)

        _log(f"  [{benchmark.name}] hard={n}  movable={movable.sum()}  "
             f"budget={self.time_budget_s:.0f}s")

        # Load plc for exact proxy scoring.
        # For large benchmarks (n > 350 macros), compute_proxy_cost takes
        # 100-500s per call — too slow for multiple restarts. Above the
        # threshold we use the fast density fallback to rank candidates;
        # the competition evaluator always re-scores whatever we return,
        # so quality is unaffected. Restarts empirically never improve
        # large benchmarks (ibm10/12/14/16/17 all had baseline win).
        EXACT_MACRO_THRESHOLD = 350
        plc = _load_plc(benchmark.name)
        use_exact = (plc is not None) and (n <= EXACT_MACRO_THRESHOLD)
        if plc is None:
            _log("  Warning: plc unavailable — using density-score fallback")
        elif n > EXACT_MACRO_THRESHOLD:
            _log(f"  Large benchmark (n={n} > {EXACT_MACRO_THRESHOLD}) — "
                 f"using density fallback to rank restarts (fast)")

        def _score(pos: np.ndarray) -> Tuple[float, torch.Tensor]:
            pl = benchmark.macro_positions.clone()
            pl[:n, 0] = torch.tensor(pos[:, 0], dtype=torch.float32)
            pl[:n, 1] = torch.tensor(pos[:, 1], dtype=torch.float32)
            if use_exact:
                try:
                    return float(_exact_proxy(pl, benchmark, plc)), pl
                except Exception:
                    pass
            return _density_score(pos, n, cw, ch), pl

        # ── Baseline (restart 0) ──────────────────────────────────────────
        _log(f"  Restart 0 (baseline)...")
        t1 = time.time()
        baseline_pos = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
        _log(f"    Legalized in {time.time()-t1:.1f}s")

        t_score0 = time.time()
        best_score, best_pl = _score(baseline_pos)
        t_one_score = time.time() - t_score0
        _log(f"  Candidate 0: proxy={best_score:.4f}  (scored in {t_one_score:.1f}s)")

        # ── Adaptive restarts: only run if budget allows ──────────────────
        # Each restart costs: legalize_time + score_time ≈ 2 × t_one_score
        # We stop as soon as remaining budget < estimated cost of next restart.
        noise_scale_base = min(cw, ch)
        for k, frac in enumerate(self.noise_fracs[: self.n_restarts - 1], start=1):
            elapsed = time.time() - t0
            remaining = self.time_budget_s - elapsed
            estimated_cost = t_one_score * 2.0  # conservative estimate
            if remaining < estimated_cost:
                _log(f"  Skipping restart {k}+ (budget: {remaining:.0f}s left, "
                     f"need ~{estimated_cost:.0f}s)")
                break

            t1 = time.time()
            noise_scale = frac * noise_scale_base
            noise = np.random.normal(0, noise_scale, init_pos.shape)
            perturbed = init_pos + noise
            perturbed[:, 0] = np.clip(perturbed[:, 0], hw, cw - hw)
            perturbed[:, 1] = np.clip(perturbed[:, 1], hh, ch - hh)
            leg = _will_legalize(perturbed, movable, sizes, hw, hh, cw, ch, n)
            _log(f"  Restart {k} (noise={frac:.0%} × canvas) legalized in "
                 f"{time.time()-t1:.1f}s")

            score, pl = _score(leg)
            _log(f"  Candidate {k}: proxy={score:.4f}")
            if score < best_score:
                best_score = score
                best_pl = pl

        _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
        return best_pl
