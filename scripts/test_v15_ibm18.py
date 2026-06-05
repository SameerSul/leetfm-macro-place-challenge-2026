"""
v15/v16 ibm18 test: n=285, grid=55x39=2145 cells.
Grid limit raised 2000→2200 in v15, so ibm18 NOW gets optimization.
t_score ~220s → ~14 restarts with 3300s budget.

v14: returned baseline 1.7941 (excluded by EXACT_GRID_CELL_LIMIT=2000).
v15/v16: first ever optimization of ibm18.
RePlAce: 1.7722

Run from project root: python scripts/test_v15_ibm18.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm18"

print("[ibm18 v15/v16] Loading benchmark...", flush=True)
benchmark, plc = load_benchmark_from_dir(benchmark_dir)
n = benchmark.num_hard_macros
grid_cells = benchmark.grid_rows * benchmark.grid_cols
print(f"  n={n}, grid={benchmark.grid_rows}x{benchmark.grid_cols} ({grid_cells} cells)",
      flush=True)
print(f"  EXACT_GRID_CELL_LIMIT=2200 -> {'INCLUDED' if grid_cells <= 2200 else 'EXCLUDED'}",
      flush=True)

from submissions.sameer_v1.placer import MacroPlacer
placer = MacroPlacer()
print(f"  budget={placer.time_budget_s:.0f}s  n_restarts={placer.n_restarts}", flush=True)
print("[ibm18 v15/v16] Running placer.place()...", flush=True)

t0 = time.time()
placement = placer.place(benchmark)
t_place = time.time() - t0
print(f"[ibm18 v15/v16] place() done in {t_place:.1f}s", flush=True)

costs = compute_proxy_cost(placement, benchmark, plc)
print(f"[ibm18 v15/v16] Final proxy={costs['proxy_cost']:.4f}", flush=True)
print(f"  wl={costs['wirelength_cost']:.3f} den={costs['density_cost']:.3f} "
      f"cong={costs['congestion_cost']:.3f}", flush=True)
print(f"[ibm18 v15/v16] Total: {time.time()-t0:.1f}s", flush=True)

baseline = 1.7941
replace = 1.7722
result = costs['proxy_cost']
print(f"\n  v14 baseline=1.7941  RePlAce=1.7722  result={result:.4f}", flush=True)
if result < baseline - 0.001:
    print(f"  IMPROVEMENT: {baseline:.4f} -> {result:.4f} (+{baseline-result:.4f})", flush=True)
elif abs(result - baseline) < 0.001:
    print(f"  SAME as baseline (all restarts worse, or returned baseline)", flush=True)
else:
    print(f"  REGRESSION: {baseline:.4f} -> {result:.4f}", flush=True)
if result < replace:
    print(f"  BEATS RePlAce ({replace:.4f})!", flush=True)
