"""
v16 ibm01 Phase 4 validation: tests whether macro-swap from best_pl (1.1854)
can escape the local minimum that 300 noise restarts cannot.

Expected behaviour:
  - Noise loop: ~300 restarts in ~1800s. Best should be 1.1854 (restart 4, 6% frac).
  - Phase 4: ~250s left → ~35 swap iterations from best_pl=1.1854.
  - If any swap finds < 1.1854: IMPROVEMENT confirmed.

Run from project root: python scripts/test_v16_ibm01_swap.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"

print("[ibm01 v16-swap] Loading benchmark...", flush=True)
benchmark, plc = load_benchmark_from_dir(benchmark_dir)
print(f"  n={benchmark.num_hard_macros}, grid={benchmark.grid_rows}x{benchmark.grid_cols}",
      flush=True)

from submissions.sameer_v1.placer import MacroPlacer

# Use full 3300s budget: noise phase runs ~301 restarts (85% = 2805s) then
# Phase 4 swap runs ~53 iterations (15% = 495s remaining).
placer = MacroPlacer()
print(f"  budget={placer.time_budget_s:.0f}s (noise ~85% + swap ~15%)", flush=True)
print("[ibm01 v16-swap] Running placer.place()...", flush=True)

t0 = time.time()
placement = placer.place(benchmark)
t_place = time.time() - t0
print(f"[ibm01 v16-swap] place() done in {t_place:.1f}s", flush=True)

costs = compute_proxy_cost(placement, benchmark, plc)
result = costs['proxy_cost']
print(f"[ibm01 v16-swap] Final proxy={result:.4f}", flush=True)
print(f"  wl={costs['wirelength_cost']:.3f} den={costs['density_cost']:.3f} "
      f"cong={costs['congestion_cost']:.3f}", flush=True)
print(f"[ibm01 v16-swap] Total: {time.time()-t0:.1f}s", flush=True)

v15_best = 1.1854
replace = 0.9976
print(f"\n  v15_best=1.1854  RePlAce=0.9976  result={result:.4f}", flush=True)
if result < v15_best - 0.001:
    print(f"  SWAP IMPROVEMENT: {v15_best:.4f} -> {result:.4f} (+{v15_best-result:.4f})", flush=True)
elif abs(result - v15_best) < 0.001:
    print(f"  SAME as v15 best (Phase 4 swaps could not improve on 1.1854)", flush=True)
else:
    print(f"  REGRESSION: {v15_best:.4f} -> {result:.4f} — check Phase 4 code!", flush=True)
