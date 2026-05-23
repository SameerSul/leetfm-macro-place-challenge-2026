"""Profile a single score() call to find the next hot path."""
import cProfile
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import importlib.util

from macro_place.loader import load_benchmark_from_dir

V2 = ROOT / "submissions" / "varrahan" / "v2" / "placer.py"
spec = importlib.util.spec_from_file_location("v2_placer", V2)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

BENCH = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
SRC = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / BENCH
benchmark, plc = load_benchmark_from_dir(str(SRC))

# Patch
import torch
import numpy as np

placement = torch.tensor(
    benchmark.macro_positions.detach().cpu().numpy(), dtype=torch.float32
)

# Warm up — install all patches + build caches
v2._exact_proxy(placement, benchmark, plc)

# Run a few iters with small jitter to force recompute
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5

def run():
    rng = np.random.RandomState(0)
    base = placement.numpy().copy()
    for _ in range(N):
        jitter = rng.normal(0, 1e-6, size=base.shape).astype(np.float32)
        p = torch.from_numpy(base + jitter)
        v2._exact_proxy(p, benchmark, plc)

prof = cProfile.Profile()
prof.enable()
run()
prof.disable()

stats = pstats.Stats(prof)
stats.sort_stats("cumulative")
stats.print_stats(35)
