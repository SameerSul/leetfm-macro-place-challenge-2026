"""Legalization-only test placer (no SA)."""
import numpy as np
import torch
from macro_place.benchmark import Benchmark
from submissions.sameer_v1.placer import _will_legalize


class MacroPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        n = benchmark.num_hard_macros
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        s = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        mv = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n].numpy()
        pos = benchmark.macro_positions[:n].numpy().copy().astype(np.float64)
        pos = _will_legalize(pos, mv, s, s[:, 0]/2, s[:, 1]/2, cw, ch, n)
        pl = benchmark.macro_positions.clone()
        pl[:n, 0] = torch.tensor(pos[:, 0], dtype=torch.float32)
        pl[:n, 1] = torch.tensor(pos[:, 1], dtype=torch.float32)
        return pl
