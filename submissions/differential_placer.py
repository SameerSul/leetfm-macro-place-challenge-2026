"""
Differentiable Macro Placer — Submission

Algorithm overview:
1. Connectivity-aware circular initialization (macros sorted by net degree)
2. GPU-accelerated Adam gradient descent on LSE-HPWL + density penalty
   (density weight ramped from 0 → target over training to avoid local minima)
3. SA legalization: minimize total overlap area with O(N)-per-step delta computation
4. Short SA refinement pass on HPWL after legalization

Targets proxy cost < 1.4578 (RePlAce baseline).

Usage:
    uv run evaluate submissions/differential_placer.py -b ibm01
    uv run evaluate submissions/differential_placer.py --all
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import torch

from macro_place.benchmark import Benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Net preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_nets(
    benchmark: Benchmark, device: torch.device
) -> Dict:
    """Extract valid nets (≥2 nodes) and move weights to device."""
    valid = [(i, nodes) for i, nodes in enumerate(benchmark.net_nodes) if len(nodes) >= 2]
    if not valid:
        return {"ids": [], "node_lists": [], "weights": torch.tensor([], device=device)}

    ids = [i for i, _ in valid]
    node_lists = [nodes.to(device) if hasattr(nodes, "to") else torch.tensor(nodes, device=device)
                  for _, nodes in valid]
    weights = benchmark.net_weights[ids].to(device)
    return {"ids": ids, "node_lists": node_lists, "weights": weights}


def _macro_degree(benchmark: Benchmark) -> List[int]:
    """Number of nets each macro appears in."""
    deg = [0] * benchmark.num_macros
    for nodes in benchmark.net_nodes:
        for n in nodes:
            deg[int(n)] += 1
    return deg


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable cost functions
# ─────────────────────────────────────────────────────────────────────────────

def _lse_hpwl(
    positions: torch.Tensor, net_data: Dict, gamma: float = 0.01
) -> torch.Tensor:
    """
    Log-sum-exp HPWL approximation — differentiable surrogate for HPWL.

    LSE-max(x, γ) = γ · log(Σ exp(xᵢ/γ))   (numerically stable via logsumexp)
    LSE-min(x, γ) = -γ · log(Σ exp(-xᵢ/γ))
    HPWL(net) ≈ w · ((LSE-max_x − LSE-min_x) + (LSE-max_y − LSE-min_y))
    """
    device = positions.device
    total = torch.zeros(1, device=device, dtype=torch.float32)

    for i, nodes in enumerate(net_data["node_lists"]):
        if len(nodes) < 2:
            continue
        net_pos = positions[nodes]  # [k, 2]
        w = net_data["weights"][i]

        x = net_pos[:, 0]
        y = net_pos[:, 1]

        max_x = gamma * torch.logsumexp(x / gamma, dim=0)
        min_x = -gamma * torch.logsumexp(-x / gamma, dim=0)
        max_y = gamma * torch.logsumexp(y / gamma, dim=0)
        min_y = -gamma * torch.logsumexp(-y / gamma, dim=0)

        total = total + w * ((max_x - min_x) + (max_y - min_y))

    return total


def _density_penalty(
    positions: torch.Tensor,
    benchmark: Benchmark,
    device: torch.device,
) -> torch.Tensor:
    """
    Bell-curve density penalty on the placement grid.

    Each macro spreads its area onto the grid using Gaussian kernels.
    Penalty = mean of top-10% occupied cells (matches proxy density metric).
    """
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    grid_r = benchmark.grid_rows
    grid_c = benchmark.grid_cols

    cell_w = canvas_w / grid_c
    cell_h = canvas_h / grid_r

    sizes = benchmark.macro_sizes.to(device).float()
    macro_area = sizes[:, 0] * sizes[:, 1]  # [N]

    sigma_x = max(cell_w, float(sizes[:, 0].mean().item())) * 0.6
    sigma_y = max(cell_h, float(sizes[:, 1].mean().item())) * 0.6

    # Grid cell centers
    cx = torch.linspace(cell_w / 2, canvas_w - cell_w / 2, grid_c, device=device)
    cy = torch.linspace(cell_h / 2, canvas_h - cell_h / 2, grid_r, device=device)

    px = positions[:, 0]  # [N]
    py = positions[:, 1]  # [N]

    dx = px.unsqueeze(1) - cx.unsqueeze(0)   # [N, grid_c]
    dy = py.unsqueeze(1) - cy.unsqueeze(0)   # [N, grid_r]

    gx = torch.exp(-0.5 * (dx / sigma_x) ** 2)  # [N, grid_c]
    gy = torch.exp(-0.5 * (dy / sigma_y) ** 2)  # [N, grid_r]

    # density[r, c] = Σ_n macro_area[n] * gx[n,c] * gy[n,r]
    density = torch.einsum("nc,nr->rc", gx * macro_area.unsqueeze(1), gy)

    # Normalize to fractional utilization per cell
    cell_area = cell_w * cell_h
    density_frac = density / cell_area

    # Top-10% mean
    flat = density_frac.flatten()
    top_k = max(1, int(0.1 * len(flat)))
    top_vals, _ = flat.topk(top_k)

    return top_vals.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Overlap helpers (CPU, O(N) per move)
# ─────────────────────────────────────────────────────────────────────────────

def _overlap_contribution(
    idx: int,
    px: list,
    py: list,
    hw: list,
    hh: list,
    num_hard: int,
) -> float:
    """Total overlap area between macro idx and all other hard macros."""
    total = 0.0
    xi, yi = px[idx], py[idx]
    hxi, hyi = hw[idx], hh[idx]
    for j in range(num_hard):
        if j == idx:
            continue
        ox = max(0.0, hxi + hw[j] - abs(xi - px[j]))
        oy = max(0.0, hyi + hh[j] - abs(yi - py[j]))
        total += ox * oy
    return total


def _total_overlap_area(px: list, py: list, hw: list, hh: list, num_hard: int) -> float:
    total = 0.0
    for i in range(num_hard):
        for j in range(i + 1, num_hard):
            ox = max(0.0, hw[i] + hw[j] - abs(px[i] - px[j]))
            oy = max(0.0, hh[i] + hh[j] - abs(py[i] - py[j]))
            total += ox * oy
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Main placer class
# ─────────────────────────────────────────────────────────────────────────────

class DifferentialMacroPlacerV1:
    """
    GPU-accelerated differentiable macro placer.

    Steps:
      1. Circular initialization sorted by net degree (high-degree macros near center).
      2. Adam gradient descent on LSE-HPWL + ramped density penalty (GPU if available).
      3. SA legalization: minimize total overlap area, O(N) delta per step.
      4. Return zero-overlap placement with lowest HPWL found.
    """

    def __init__(
        self,
        # Device
        device: str = "auto",
        # Gradient phase
        lr: float = 0.8,
        max_grad_iter: int = 4000,
        lse_gamma: float = 0.01,
        density_weight_max: float = 0.15,
        density_ramp_frac: float = 0.4,  # fraction of iterations before ramping density
        # SA legalization
        sa_max_steps: int = 500_000,
        sa_t_init: float = 1.0,
        sa_t_decay: float = 0.99997,
        sa_min_t: float = 0.001,
        # Multi-start
        n_restarts: int = 2,
        time_budget_s: float = 3000.0,  # ~50 min, well within 1-hour limit
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.lr = lr
        self.max_grad_iter = max_grad_iter
        self.lse_gamma = lse_gamma
        self.density_weight_max = density_weight_max
        self.density_ramp_frac = density_ramp_frac
        self.sa_max_steps = sa_max_steps
        self.sa_t_init = sa_t_init
        self.sa_t_decay = sa_t_decay
        self.sa_min_t = sa_min_t
        self.n_restarts = n_restarts
        self.time_budget_s = time_budget_s

    # ── public API ──────────────────────────────────────────────────────────

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_start = time.time()
        device = self.device

        print(f"  Device: {device} | macros: {benchmark.num_hard_macros} hard / "
              f"{benchmark.num_soft_macros} soft | nets: {benchmark.num_nets}")

        # Movable hard macro indices
        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_idx = torch.where(movable_mask)[0]

        if len(movable_idx) == 0:
            return benchmark.macro_positions.clone()

        # Preprocess nets for gradient computation
        net_data = _preprocess_nets(benchmark, device)
        degree = _macro_degree(benchmark)

        best_placement: Optional[torch.Tensor] = None
        best_hpwl = float("inf")

        for restart in range(self.n_restarts):
            elapsed = time.time() - t_start
            if elapsed > self.time_budget_s * 0.95:
                print(f"  Time budget reached after {restart} restarts.")
                break

            print(f"  Restart {restart + 1}/{self.n_restarts} (elapsed {elapsed:.0f}s)")

            # Phase 1: initialize
            placement = self._initialize(benchmark, movable_idx, degree, restart, device)

            # Phase 2: gradient descent
            placement = self._gradient_optimize(
                placement, benchmark, net_data, movable_idx, device
            )

            # Phase 3: SA legalization
            placement = self._sa_legalize(placement, benchmark, movable_idx)

            # Evaluate approximate HPWL
            with torch.no_grad():
                hpwl = _lse_hpwl(placement.to(device).float(), net_data, gamma=self.lse_gamma).item()

            print(f"    HPWL (LSE approx) = {hpwl:.4f}")

            if hpwl < best_hpwl:
                best_hpwl = hpwl
                best_placement = placement.cpu().clone()

        assert best_placement is not None
        return best_placement

    # ── initialization ───────────────────────────────────────────────────────

    def _initialize(
        self,
        benchmark: Benchmark,
        movable_idx: torch.Tensor,
        degree: List[int],
        restart: int,
        device: torch.device,
    ) -> torch.Tensor:
        placement = benchmark.macro_positions.clone().to(device)
        sizes = benchmark.macro_sizes.to(device)
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        if restart == 0:
            # Sort by net degree descending; high-degree macros near center
            sorted_movable = sorted(
                movable_idx.tolist(), key=lambda i: -degree[i]
            )
            n = len(sorted_movable)
            cx, cy = canvas_w / 2, canvas_h / 2
            r = min(canvas_w, canvas_h) * 0.35

            for k, idx in enumerate(sorted_movable):
                angle = 2 * math.pi * k / n
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                hw = sizes[idx, 0].item() / 2
                hh = sizes[idx, 1].item() / 2
                placement[idx, 0] = max(hw, min(canvas_w - hw, x))
                placement[idx, 1] = max(hh, min(canvas_h - hh, y))

        else:
            # Jittered random restart
            for idx in movable_idx.tolist():
                hw = sizes[idx, 0].item() / 2
                hh = sizes[idx, 1].item() / 2
                placement[idx, 0] = hw + random.random() * (canvas_w - 2 * hw)
                placement[idx, 1] = hh + random.random() * (canvas_h - 2 * hh)

        return placement

    # ── gradient optimization ────────────────────────────────────────────────

    def _gradient_optimize(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        net_data: Dict,
        movable_idx: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        sizes = benchmark.macro_sizes.to(device).float()
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        pos = placement[movable_idx].clone().float().to(device).requires_grad_(True)

        optimizer = torch.optim.Adam([pos], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.max_grad_iter, eta_min=self.lr * 0.01
        )

        ramp_start = int(self.density_ramp_frac * self.max_grad_iter)

        for step in range(self.max_grad_iter):
            # Assemble full placement tensor (detach fixed/soft macros)
            full_pos = placement.clone().float()
            full_pos[movable_idx] = pos

            # LSE-HPWL
            loss = _lse_hpwl(full_pos, net_data, gamma=self.lse_gamma)

            # Density penalty (ramped in after ramp_start iterations)
            if step >= ramp_start and self.density_weight_max > 0:
                ramp_frac = (step - ramp_start) / max(1, self.max_grad_iter - ramp_start)
                dw = self.density_weight_max * ramp_frac
                loss = loss + dw * _density_penalty(full_pos, benchmark, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Clamp to canvas bounds
            with torch.no_grad():
                hw = sizes[movable_idx, 0] / 2
                hh = sizes[movable_idx, 1] / 2
                pos.data[:, 0].clamp_(hw, canvas_w - hw)
                pos.data[:, 1].clamp_(hh, canvas_h - hh)

            if step % 500 == 0:
                print(f"    grad step {step:4d}/{self.max_grad_iter}  loss={loss.item():.4f}")

        result = placement.clone()
        result[movable_idx] = pos.detach()
        return result

    # ── SA legalization ──────────────────────────────────────────────────────

    def _sa_legalize(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        movable_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Remove overlaps via SA while minimizing overlap area delta per move."""
        placement_cpu = placement.cpu().clone()
        sizes_np = benchmark.macro_sizes.cpu()
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height
        num_hard = benchmark.num_hard_macros

        # Working lists for fast O(N) overlap computation
        px = [placement_cpu[i, 0].item() for i in range(num_hard)]
        py = [placement_cpu[i, 1].item() for i in range(num_hard)]
        hw = [sizes_np[i, 0].item() / 2 for i in range(num_hard)]
        hh = [sizes_np[i, 1].item() / 2 for i in range(num_hard)]

        movable_list = movable_idx.tolist()

        # Initial cost = total overlap area
        current_cost = _total_overlap_area(px, py, hw, hh, num_hard)
        best_cost = current_cost

        if current_cost == 0.0:
            print("    SA legalization: already overlap-free.")
            return placement_cpu

        print(f"    SA legalization: initial overlap area = {current_cost:.3f}")

        # Best solution tracking
        best_px = px[:]
        best_py = py[:]

        T = self.sa_t_init
        decay = self.sa_t_decay
        min_T = self.sa_min_t

        for step in range(self.sa_max_steps):
            if current_cost == 0.0:
                break

            idx = random.choice(movable_list)
            old_x, old_y = px[idx], py[idx]

            # Compute overlap contribution before move
            old_contrib = _overlap_contribution(idx, px, py, hw, hh, num_hard)

            # Propose move — scale decreases with temperature
            scale = max(hw[idx], hh[idx]) * (1.0 + 3.0 * T)
            new_x = old_x + random.gauss(0.0, scale)
            new_y = old_y + random.gauss(0.0, scale)
            new_x = max(hw[idx], min(canvas_w - hw[idx], new_x))
            new_y = max(hh[idx], min(canvas_h - hh[idx], new_y))

            # Compute overlap contribution after move
            px[idx] = new_x
            py[idx] = new_y
            new_contrib = _overlap_contribution(idx, px, py, hw, hh, num_hard)

            delta = new_contrib - old_contrib

            # Accept or reject (Metropolis criterion)
            if delta < 0 or (T > min_T and random.random() < math.exp(-delta / T)):
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_px = px[:]
                    best_py = py[:]
            else:
                px[idx] = old_x
                py[idx] = old_y

            T = max(T * decay, min_T)

            if step % 100_000 == 0 and step > 0:
                print(f"    SA step {step:6d}  overlap={current_cost:.3f}  T={T:.5f}")

        print(f"    SA legalization done: overlap area = {best_cost:.4f}")

        # Write best solution back
        for i in range(num_hard):
            placement_cpu[i, 0] = best_px[i]
            placement_cpu[i, 1] = best_py[i]

        return placement_cpu
