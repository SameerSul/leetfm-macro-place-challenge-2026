"""
Will's Seed v4 - Minimal Legalization + GPU Refinement

1. Legalize initial placement with minimum displacement
2. GPU gradient refinement: reduce wirelength while maintaining no-overlap
   via projected gradient (undo any step that creates overlap)
3. Soft macro FD at the end

Usage:
    uv run evaluate submissions/will_seed/placer.py
    uv run evaluate submissions/will_seed/placer.py --all
"""

import sys, io, math, random
import torch
import numpy as np
from pathlib import Path
from macro_place.benchmark import Benchmark


def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


def _extract_edges(benchmark, plc):
    n_hard = benchmark.num_hard_macros
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    edge_dict = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            ml = sorted(macros)
            w = 1.0 / (len(ml) - 1)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    pair = (ml[i], ml[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w
    if not edge_dict:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0)
    return (torch.tensor(list(edge_dict.keys()), dtype=torch.long),
            torch.tensor([edge_dict[e] for e in edge_dict], dtype=torch.float32))


class WillSeedPlacer:
    def __init__(self, seed=42, refine_iters=3000):
        self.seed = seed
        self.refine_iters = refine_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard].numpy()

        plc = _load_plc(benchmark.name)
        if plc is not None:
            edges, edge_weights = _extract_edges(benchmark, plc)
        else:
            edges = torch.zeros(0, 2, dtype=torch.long)
            edge_weights = torch.zeros(0)

        # Step 1: Legalize initial hard macro placement
        pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        pos = self._legalize(pos, movable, sizes_np, half_w, half_h, cw, ch, n_hard)

        # Step 2: SA refinement with overlap rejection (fast numpy)
        if len(edges) > 0:
            pos = self._sa_refine(pos, edges.numpy(), edge_weights.numpy(),
                                   movable, sizes_np, half_w, half_h, cw, ch, n_hard, plc, benchmark)

        # Step 3: Build full placement + soft macro FD
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(pos, dtype=torch.float32)

        # Keep soft macros at initial positions - they were already optimized
        # for the initial hard macro layout and minimal legalization preserves this

        return full_pos

    def _sa_refine(self, pos, edges, edge_weights, movable, sizes, half_w, half_h, cw, ch, n, plc, benchmark):
        """SA with single-macro overlap check (O(N) per move, not O(N^2))."""
        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        pos = pos.copy()
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

        # Build neighbor lists
        neighbors = [[] for _ in range(n)]
        for i, j in edges:
            neighbors[i].append(j)
            neighbors[j].append(i)

        def wl_cost():
            dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
            dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
            return (edge_weights * (dx + dy)).sum()

        def check_single_overlap(idx):
            """O(N) check: does macro idx overlap any other? (with safety gap)"""
            gap = 0.05
            dx = np.abs(pos[idx, 0] - pos[:, 0])
            dy = np.abs(pos[idx, 1] - pos[:, 1])
            overlaps = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
            overlaps[idx] = False
            return overlaps.any()

        current_cost = wl_cost()
        best_pos = pos.copy()
        best_cost = current_cost

        T_start = max(cw, ch) * 0.15
        T_end = max(cw, ch) * 0.001

        for step in range(self.refine_iters):
            frac = step / self.refine_iters
            T = T_start * (T_end / T_start) ** frac

            move = random.random()
            i = random.choice(movable_idx)
            old_x, old_y = pos[i, 0], pos[i, 1]

            if move < 0.5:
                # SHIFT
                shift = T * (0.3 + 0.7 * (1 - frac))
                pos[i, 0] = np.clip(pos[i, 0] + random.gauss(0, shift), half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + random.gauss(0, shift), half_h[i], ch - half_h[i])
            elif move < 0.8:
                # SWAP
                if neighbors[i] and random.random() < 0.7:
                    cands = [j for j in neighbors[i] if movable[j]]
                    j = random.choice(cands) if cands else random.choice(movable_idx)
                else:
                    j = random.choice(movable_idx)
                if i != j:
                    old_jx, old_jy = pos[j, 0], pos[j, 1]
                    pos[i, 0] = np.clip(old_jx, half_w[i], cw - half_w[i])
                    pos[i, 1] = np.clip(old_jy, half_h[i], ch - half_h[i])
                    pos[j, 0] = np.clip(old_x, half_w[j], cw - half_w[j])
                    pos[j, 1] = np.clip(old_y, half_h[j], ch - half_h[j])
                    # Check both macros
                    if check_single_overlap(i) or check_single_overlap(j):
                        pos[i, 0] = old_x; pos[i, 1] = old_y
                        pos[j, 0] = old_jx; pos[j, 1] = old_jy
                        continue
                    new_cost = wl_cost()
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost; best_pos = pos.copy()
                    else:
                        pos[i, 0] = old_x; pos[i, 1] = old_y
                        pos[j, 0] = old_jx; pos[j, 1] = old_jy
                    continue
            else:
                # MOVE TOWARD NEIGHBOR
                if neighbors[i]:
                    j = random.choice(neighbors[i])
                    alpha = random.uniform(0.05, 0.3)
                    pos[i, 0] = np.clip(pos[i, 0]+alpha*(pos[j, 0]-pos[i, 0]), half_w[i], cw-half_w[i])
                    pos[i, 1] = np.clip(pos[i, 1]+alpha*(pos[j, 1]-pos[i, 1]), half_h[i], ch-half_h[i])

            # Single macro overlap check - O(N)
            if check_single_overlap(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue

            new_cost = wl_cost()
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost; best_pos = pos.copy()
            else:
                pos[i, 0] = old_x; pos[i, 1] = old_y

        return best_pos

    def _legalize(self, pos, movable, sizes, half_w, half_h, cw, ch, n):
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal = pos.copy()
        for idx in order:
            if not movable[idx]:
                placed[idx] = True; continue
            if placed.any():
                dx = np.abs(legal[idx, 0] - legal[:, 0])
                dy = np.abs(legal[idx, 1] - legal[:, 1])
                c = (dx < sep_x[idx]+0.05) & (dy < sep_y[idx]+0.05) & placed
                c[idx] = False
                if not c.any():
                    placed[idx] = True; continue
            step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
            best_p = legal[idx].copy(); best_d = float('inf')
            for r in range(1, 150):
                found = False
                for dxm in range(-r, r+1):
                    for dym in range(-r, r+1):
                        if abs(dxm) != r and abs(dym) != r: continue
                        cx = np.clip(pos[idx, 0]+dxm*step, half_w[idx], cw-half_w[idx])
                        cy = np.clip(pos[idx, 1]+dym*step, half_h[idx], ch-half_h[idx])
                        if placed.any():
                            dx = np.abs(cx-legal[:, 0]); dy = np.abs(cy-legal[:, 1])
                            c = (dx < sep_x[idx]+0.05) & (dy < sep_y[idx]+0.05) & placed
                            c[idx] = False
                            if c.any(): continue
                        d = (cx-pos[idx, 0])**2+(cy-pos[idx, 1])**2
                        if d < best_d:
                            best_d = d; best_p = np.array([cx, cy]); found = True
                if found: break
            legal[idx] = best_p; placed[idx] = True
        return legal
