"""
v15 ibm13 test: n=466 macros (EXACT_MACRO_THRESHOLD=430 → EXCLUDED by default).
Wait — actually ibm13 n=466 exceeds EXACT_MACRO_THRESHOLD=430, so it returns baseline.
But ibm13 was in SKIP_EXACT in v14 with n<=430? Let me check: ibm13 has n=??

Actually from v14 memory: ibm13 was in SKIP_EXACT = {"ibm11", "ibm13"}.
In v14, SKIP_EXACT returned baseline for ibm11(n=373) and ibm13 (n=?).
ibm13 baseline=1.4011 (SKIP_EXACT).

This script tests ibm13 with v15's full 3300s budget (SKIP_EXACT now empty).
If ibm13 n>430, it returns baseline immediately — this script will confirm that.
Run from project root: python scripts/test_v15_ibm13.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm13"

print("[ibm13 v15] Loading benchmark...", flush=True)
benchmark, plc = load_benchmark_from_dir(benchmark_dir)
n = benchmark.num_hard_macros
grid_cells = benchmark.grid_rows * benchmark.grid_cols
print(f"  n={n}, grid={benchmark.grid_rows}x{benchmark.grid_cols} ({grid_cells} cells)",
      flush=True)
print(f"  EXACT_MACRO_THRESHOLD=430 -> {'INCLUDED' if n <= 430 else 'EXCLUDED (returns baseline)'}",
      flush=True)

from submissions.sameer_v1.placer import MacroPlacer
placer = MacroPlacer()
print(f"  budget={placer.time_budget_s:.0f}s  n_restarts={placer.n_restarts}", flush=True)
print("[ibm13 v15] Running placer.place()...", flush=True)

t0 = time.time()
placement = placer.place(benchmark)
t_place = time.time() - t0
print(f"[ibm13 v15] place() done in {t_place:.1f}s", flush=True)

costs = compute_proxy_cost(placement, benchmark, plc)
print(f"[ibm13 v15] Final proxy={costs['proxy_cost']:.4f}", flush=True)
print(f"  wl={costs['wirelength_cost']:.3f} den={costs['density_cost']:.3f} "
      f"cong={costs['congestion_cost']:.3f}", flush=True)
print(f"[ibm13 v15] Total: {time.time()-t0:.1f}s", flush=True)

baseline = 1.4011
replace = 1.3355
result = costs['proxy_cost']
print(f"\n  v14 baseline=1.4011  RePlAce=1.3355  result={result:.4f}", flush=True)
if result < baseline - 0.001:
    print(f"  IMPROVEMENT: {baseline:.4f} -> {result:.4f} (+{baseline-result:.4f})", flush=True)
elif abs(result - baseline) < 0.001:
    print(f"  SAME as baseline (n={n} likely > threshold or SKIP_EXACT)", flush=True)
else:
    print(f"  REGRESSION: {baseline:.4f} -> {result:.4f}", flush=True)
if result < replace:
    print(f"  BEATS RePlAce ({replace:.4f})!", flush=True)
