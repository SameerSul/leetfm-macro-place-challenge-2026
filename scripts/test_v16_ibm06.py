"""
v16 ibm06 test: n=178, grid=31x28=868 cells, t_score=19-20s.
v14 best=1.6797 (vs RePlAce=1.6187, -3.8%). 3300s -> ~110+ noise restarts.
7 cong-grad iters at adaptive frac=0.01 wins in v14; v15 adds many more noise restarts.

Run from project root: python scripts/test_v16_ibm06.py
"""
import sys, time
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm06"
print("[ibm06 v16] Loading benchmark...", flush=True)
benchmark, plc = load_benchmark_from_dir(benchmark_dir)
n = benchmark.num_hard_macros
grid_cells = benchmark.grid_rows * benchmark.grid_cols
print(f"  n={n}, grid={benchmark.grid_rows}x{benchmark.grid_cols} ({grid_cells} cells)", flush=True)

from submissions.sameer_v1.placer import MacroPlacer
placer = MacroPlacer()
print(f"  budget={placer.time_budget_s:.0f}s  n_restarts={placer.n_restarts}", flush=True)
print("[ibm06 v16] Running placer.place()...", flush=True)

t0 = time.time()
placement = placer.place(benchmark)
t_place = time.time() - t0
print(f"[ibm06 v16] place() done in {t_place:.1f}s", flush=True)

costs = compute_proxy_cost(placement, benchmark, plc)
result = costs['proxy_cost']
print(f"[ibm06 v16] Final proxy={result:.4f}", flush=True)
print(f"  wl={costs['wirelength_cost']:.3f} den={costs['density_cost']:.3f} "
      f"cong={costs['congestion_cost']:.3f}", flush=True)
print(f"[ibm06 v16] Total: {time.time()-t0:.1f}s", flush=True)

v14_best = 1.6797
replace = 1.6187
print(f"\n  v14_best={v14_best:.4f}  RePlAce={replace:.4f}  result={result:.4f}", flush=True)
if result < v14_best - 0.001:
    print(f"  IMPROVEMENT: {v14_best:.4f} -> {result:.4f} (+{v14_best-result:.4f})", flush=True)
elif abs(result - v14_best) < 0.001:
    print(f"  SAME as v14 best", flush=True)
else:
    print(f"  REGRESSION: {v14_best:.4f} -> {result:.4f} — check code!", flush=True)
if result < replace:
    print(f"  BEATS RePlAce ({replace:.4f})!", flush=True)
