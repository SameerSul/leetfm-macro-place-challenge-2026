"""
Competitive Macro Placer -- Partcl/HRT Challenge 2026
Sameer Suleman (sameersul)

Algorithm:
  Multi-restart legalization with DIRECTED perturbations guided by research papers.

  1. Legalize from initial.plc positions (baseline).
  2. Run directed restarts (paper-derived) using the baseline as a guide:
       a. Density-gradient perturbation (MaskRegulate, NeurIPS 2024):
          Build a macro-occupancy heatmap, push macros from crowded zones
          toward empty zones before re-legalizing.
       b. Wire-pull perturbation (WireMask-BBO, NeurIPS 2023):
          For each macro, compute which direction reduces total HPWL of its
          connected nets (net centroid pull), move in that direction before
          re-legalizing.
  3. Fill remaining time budget with random Gaussian restarts.
  4. Score all candidates with the EXACT proxy evaluator; return best.

Paper contributions implemented:
  - WireMask-BBO (Gu et al., LAMDA/NeurIPS 2023): wire-pull vector field
    → _compute_wire_pull() + _wire_pull_perturb()
  - MaskRegulate (Chen et al., LAMDA/NeurIPS 2024): occupancy density gradient
    → _congestion_heatmap() + _density_gradient_perturb()
  - Random restarts (baseline): unchanged from sameer_v1 original

Why restarts beat SA:
  SA over-optimises WL at the cost of congestion. Restarts explore
  different legalization arrangements without destroying the good spread
  already present in initial.plc.

Proxy cost breakdown (why congestion is the target):
  Proxy = 1×WL + 0.5×density + 0.5×congestion
  WL ≈ 0.06 (tiny), congestion ≈ 2.0 (dominant — 30× larger).
  All directed perturbations target congestion, not WL.

Baselines:
  will_seed avg:           1.5338
  sameer_v1 (leg-only):    1.5062
  sameer_v1 (restarts):    ~1.49 (confirmed on ibm01/03/08)
  RePlAce avg:             1.4578  ← target to beat
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
# Will's minimum-displacement legalization (unchanged)
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


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

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
    """Fallback scorer: sum-of-squares macro count on a 20×20 grid (lower = better spread)."""
    G = 20
    grid = np.zeros((G, G), dtype=np.float64)
    for i in range(n):
        r = min(int(pos[i, 1] / (ch / G)), G - 1)
        c = min(int(pos[i, 0] / (cw / G)), G - 1)
        grid[r, c] += 1
    return float((grid ** 2).sum())


# ---------------------------------------------------------------------------
# Directed perturbation: WireMask-BBO (NeurIPS 2023)
# ---------------------------------------------------------------------------

def _compute_wire_pull(pos: np.ndarray, benchmark: Benchmark, n: int) -> np.ndarray:
    """
    Wire-pull vector field — adapted from WireMask-BBO (Gu et al., NeurIPS 2023).

    WireMask-BBO's greedy evaluator places each macro at the grid cell that
    minimises the HPWL increase over all its connected nets (the 'wire mask').
    We approximate this as a continuous pull: for each macro i, sum the
    displacement vectors from i to the centroid of each net it belongs to,
    weighted by net weight. The result is the direction that most reduces
    total HPWL if macro i moves there.

    Only hard macros (indices < n) are used as net participants — ports are
    not perturbed, so omitting them makes the pull slightly conservative but
    keeps the code benchmark-agnostic.

    Returns
    -------
    pull : np.ndarray [n, 2]
        Unnormalized pull displacement per hard macro.
        Magnitude encodes how 'important' the move is; used in _wire_pull_perturb.
    """
    pull = np.zeros((n, 2), dtype=np.float64)

    for net_idx, nodes in enumerate(benchmark.net_nodes):
        weight = float(benchmark.net_weights[net_idx])
        nodes_np = nodes.numpy()
        # Hard macros only (indices 0..n-1)
        hard_nodes = [int(nd) for nd in nodes_np if nd < n]
        if len(hard_nodes) < 2:
            continue

        net_pos = pos[hard_nodes]           # [k, 2]
        centroid = net_pos.mean(axis=0)     # [2] — HPWL midpoint approximation

        for nd in hard_nodes:
            pull[nd] += weight * (centroid - pos[nd])

    return pull


def _wire_pull_perturb(
    init_pos: np.ndarray,
    leg_pos: np.ndarray,
    benchmark: Benchmark,
    movable: np.ndarray,
    n: int,
    cw: float,
    ch: float,
    hw: np.ndarray,
    hh: np.ndarray,
    frac: float,
) -> np.ndarray:
    """
    Wire-pull directed perturbation (WireMask-BBO inspired).

    Computes per-macro wire-pull vectors from the baseline legalization,
    then applies them (clamped to `frac × min(canvas)`) to init_pos.
    Re-legalizing from these perturbed positions tends to produce placements
    where connected macros are closer, reducing HPWL at the cost of slight
    density increase.  In practice this is useful when WL dominates over
    density (small benchmarks with dense netlists).

    Parameters
    ----------
    init_pos : initial.plc positions (unlegalized) — we perturb these.
    leg_pos  : baseline legalized positions — used only to compute pull vectors.
    frac     : displacement cap as fraction of min(cw, ch).
    """
    magnitude = frac * min(cw, ch)
    pull = _compute_wire_pull(leg_pos, benchmark, n)

    perturbed = init_pos.copy()
    for i in range(n):
        if not movable[i]:
            continue
        dx, dy = pull[i, 0], pull[i, 1]
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-10
        # Clamp: never displace more than `magnitude`
        scale = min(magnitude / norm, 1.0)
        perturbed[i, 0] = np.clip(init_pos[i, 0] + scale * dx, hw[i], cw - hw[i])
        perturbed[i, 1] = np.clip(init_pos[i, 1] + scale * dy, hh[i], ch - hh[i])

    return perturbed


# ---------------------------------------------------------------------------
# Directed perturbation: MaskRegulate regularity (NeurIPS 2024)
# ---------------------------------------------------------------------------

def _congestion_heatmap(pos: np.ndarray, n: int, cw: float, ch: float, G: int = 20) -> np.ndarray:
    """
    Macro occupancy density grid — basis of MaskRegulate's regularity mask.

    MaskRegulate (Chen et al., NeurIPS 2024) defines a 'regularity' metric
    that penalises macros placed far from a balanced/centered distribution.
    Their get_regular_mask() returns per-cell regularity cost, guiding the RL
    policy away from overcrowded zones.

    We build the same signal without RL: a G×G grid where each cell stores
    the number of macro centers inside it.  High cell counts = congestion risk.

    Returns
    -------
    grid : np.ndarray [G, G]
    """
    grid = np.zeros((G, G), dtype=np.float64)
    cw_g = cw / G
    ch_g = ch / G
    for i in range(n):
        c = min(int(pos[i, 0] / cw_g), G - 1)
        r = min(int(pos[i, 1] / ch_g), G - 1)
        grid[r, c] += 1.0
    return grid


def _box_blur(grid: np.ndarray, G: int) -> np.ndarray:
    """
    3×3 box blur using pure NumPy (no scipy dependency).
    Equivalent to one step of Gaussian blur with σ≈0.85.
    Running this 3× approximates σ≈1.5 Gaussian.
    """
    padded = np.pad(grid, 1, mode="edge")
    result = np.zeros_like(grid)
    for dr in range(3):
        for dc in range(3):
            result += padded[dr: dr + G, dc: dc + G]
    return result / 9.0


def _density_gradient_perturb(
    init_pos: np.ndarray,
    leg_pos: np.ndarray,
    movable: np.ndarray,
    n: int,
    cw: float,
    ch: float,
    hw: np.ndarray,
    hh: np.ndarray,
    frac: float,
    G: int = 20,
) -> np.ndarray:
    """
    Congestion-aware directed perturbation (MaskRegulate regularity inspired).

    MaskRegulate's get_regular_mask() penalises each candidate cell by how
    overcrowded it is relative to the chip mean.  Their RL policy learns to
    avoid those cells.  We replicate this WITHOUT RL:

      1. Build G×G macro occupancy grid from the baseline legalization.
      2. Apply 3 passes of box blur to smooth out quantization artifacts.
      3. Compute the negative density gradient (finite differences) —
         the direction pointing from each cell toward its lowest-density
         neighbor.
      4. For each movable hard macro, look up its cell's gradient and
         apply a displacement of `frac × min(cw,ch)` in that direction
         to init_pos.
      5. Re-legalizing from these displaced positions tends to spread macros
         more evenly, reducing both density and congestion components.

    Parameters
    ----------
    init_pos : initial.plc positions — we perturb these for re-legalization.
    leg_pos  : baseline legalized positions — used only to build the heatmap.
    frac     : displacement magnitude as fraction of min(cw, ch).
    G        : grid resolution (default 20 × 20).
    """
    magnitude = frac * min(cw, ch)
    cell_w = cw / G
    cell_h = ch / G

    # Build and smooth the occupancy grid
    grid = _congestion_heatmap(leg_pos, n, cw, ch, G)
    smooth = grid.copy()
    for _ in range(3):
        smooth = _box_blur(smooth, G)

    # Negative density gradient: points toward lower density
    grad_x = np.zeros((G, G))
    grad_y = np.zeros((G, G))
    for r in range(G):
        for c in range(G):
            left  = smooth[r, c - 1] if c > 0    else smooth[r, c]
            right = smooth[r, c + 1] if c < G-1  else smooth[r, c]
            grad_x[r, c] = -(right - left) / 2.0   # negative = toward lower density

            down = smooth[r - 1, c] if r > 0    else smooth[r, c]
            up   = smooth[r + 1, c] if r < G-1  else smooth[r, c]
            grad_y[r, c] = -(up - down) / 2.0

    perturbed = init_pos.copy()
    for i in range(n):
        if not movable[i]:
            continue
        c_idx = min(int(leg_pos[i, 0] / cell_w), G - 1)
        r_idx = min(int(leg_pos[i, 1] / cell_h), G - 1)
        dx = grad_x[r_idx, c_idx]
        dy = grad_y[r_idx, c_idx]
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-10
        perturbed[i, 0] = np.clip(init_pos[i, 0] + magnitude * dx / norm, hw[i], cw - hw[i])
        perturbed[i, 1] = np.clip(init_pos[i, 1] + magnitude * dy / norm, hh[i], ch - hh[i])

    return perturbed


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class MacroPlacer:
    """
    Multi-restart legalization placer with directed perturbations.

    Restart schedule (subject to adaptive time budget):
      0  Baseline          — unperturbed initial.plc legalization
      1  Density-grad      — MaskRegulate regularity: push from crowded zones
      2  Wire-pull         — WireMask-BBO: pull toward net centroids
      3+ Random Gaussian   — stochastic restarts (original strategy)

    Parameters
    ----------
    n_restarts : int
        Total candidates including baseline (restarts = n_restarts - 1).
    noise_fracs : list[float]
        Magnitudes for random restarts (fraction of min canvas dimension).
    seed : int
        Random seed for reproducibility.
    time_budget_s : float
        Per-benchmark wall-clock budget; restarts skipped when insufficient.
    """

    def __init__(
        self,
        n_restarts: int = 5,
        noise_fracs: Optional[List[float]] = None,
        seed: int = 42,
        time_budget_s: float = 200.0,
    ):
        self.n_restarts = n_restarts
        self.noise_fracs = noise_fracs or [0.02, 0.04, 0.06, 0.08]
        self.seed = seed
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

        # Large benchmarks use density fallback: exact proxy takes 100-500s/call
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

        # ── Restart 0: Baseline ───────────────────────────────────────────
        _log(f"  Restart 0 (baseline)...")
        t1 = time.time()
        baseline_pos = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
        _log(f"    Legalized in {time.time()-t1:.1f}s")

        t_score0 = time.time()
        best_score, best_pl = _score(baseline_pos)
        t_one_score = time.time() - t_score0
        _log(f"  Candidate 0: proxy={best_score:.4f}  (scored in {t_one_score:.1f}s)")

        def _try_restart(label: str, perturbed_init: np.ndarray, k: int) -> bool:
            """Legalize + score one candidate. Returns True if best score updated."""
            nonlocal best_score, best_pl
            elapsed = time.time() - t0
            remaining = self.time_budget_s - elapsed
            estimated_cost = t_one_score * 2.0
            if remaining < estimated_cost:
                _log(f"  Skipping restart {k}+ (budget: {remaining:.0f}s left, "
                     f"need ~{estimated_cost:.0f}s)")
                return False  # signal: stop further restarts

            t1 = time.time()
            leg = _will_legalize(perturbed_init, movable, sizes, hw, hh, cw, ch, n)
            _log(f"  Restart {k} ({label}) legalized in {time.time()-t1:.1f}s")

            score, pl = _score(leg)
            _log(f"  Candidate {k}: proxy={score:.4f}")
            if score < best_score:
                best_score = score
                best_pl = pl
                return True  # improvement found
            return True  # no improvement but keep going

        # ── Restart 1: Density-gradient (MaskRegulate) ───────────────────
        density_perturbed = _density_gradient_perturb(
            init_pos, baseline_pos, movable, n, cw, ch, hw, hh, frac=0.04
        )
        if not _try_restart("density-grad frac=4%", density_perturbed, k=1):
            _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
            return best_pl

        # ── Restart 2: Wire-pull (WireMask-BBO) ──────────────────────────
        wire_perturbed = _wire_pull_perturb(
            init_pos, baseline_pos, benchmark, movable, n, cw, ch, hw, hh, frac=0.06
        )
        if not _try_restart("wire-pull frac=6%", wire_perturbed, k=2):
            _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
            return best_pl

        # ── Restarts 3+: Random Gaussian ─────────────────────────────────
        noise_scale_base = min(cw, ch)
        for k, frac in enumerate(self.noise_fracs[: self.n_restarts - 3], start=3):
            noise = np.random.normal(0, frac * noise_scale_base, init_pos.shape)
            perturbed = np.clip(
                init_pos + noise,
                np.stack([hw, hh], axis=1),
                np.stack([cw - hw, ch - hh], axis=1),
            )
            if not _try_restart(f"random noise={frac:.0%}", perturbed, k=k):
                break

        _log(f"  Best proxy={best_score:.4f}  total={time.time()-t0:.1f}s")
        return best_pl
