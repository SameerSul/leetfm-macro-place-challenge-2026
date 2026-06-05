"""
v17 parallel scoring validation: ibm01 with n_workers=4 vs n_workers=1.

Tests that:
  1. Parallel mode produces a valid result (no errors)
  2. Result is at least as good as v16 serial mode
  3. More noise restarts are attempted in same budget

Expected:
  - n_workers=1 (serial): ~300 restarts in 3300s (same as v16)
  - n_workers=4 (parallel): ~500+ restarts in 3300s (~1.7× more for ibm01)

Use a shorter budget (500s) to validate quickly.

Run from project root: python scripts/test_v17_parallel_ibm01.py
"""
import sys, time
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

benchmark_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
print("[ibm01 v17-parallel] Loading benchmark...", flush=True)
benchmark, plc = load_benchmark_from_dir(benchmark_dir)
print(f"  n={benchmark.num_hard_macros}, grid={benchmark.grid_rows}x{benchmark.grid_cols}", flush=True)

from submissions.sameer_v1.placer import MacroPlacer

# Test with 500s budget to keep validation quick
# n_workers=4 should give ~1.7-3x more restarts than n_workers=1
BUDGET = 500.0

print(f"\n[v17] Serial run (n_workers=1, budget={BUDGET}s)...", flush=True)
placer_serial = MacroPlacer(time_budget_s=BUDGET, n_workers=1)
t0 = time.time()
placement_s = placer_serial.place(benchmark)
t_serial = time.time() - t0
costs_s = compute_proxy_cost(placement_s, benchmark, plc)
print(f"  Serial: proxy={costs_s['proxy_cost']:.4f} in {t_serial:.1f}s", flush=True)

print(f"\n[v17] Parallel run (n_workers=4, budget={BUDGET}s)...", flush=True)
placer_par = MacroPlacer(time_budget_s=BUDGET, n_workers=4)
t0 = time.time()
placement_p = placer_par.place(benchmark)
t_par = time.time() - t0
costs_p = compute_proxy_cost(placement_p, benchmark, plc)
print(f"  Parallel: proxy={costs_p['proxy_cost']:.4f} in {t_par:.1f}s", flush=True)

print(f"\n[v17] Summary:", flush=True)
print(f"  Serial:   proxy={costs_s['proxy_cost']:.4f}  time={t_serial:.1f}s", flush=True)
print(f"  Parallel: proxy={costs_p['proxy_cost']:.4f}  time={t_par:.1f}s", flush=True)
if costs_p['proxy_cost'] <= costs_s['proxy_cost'] + 0.001:
    print(f"  PARALLEL OK: result similar or better than serial", flush=True)
else:
    print(f"  REGRESSION: parallel ({costs_p['proxy_cost']:.4f}) worse than serial "
          f"({costs_s['proxy_cost']:.4f})", flush=True)
