"""Verify the vectorized get_routing matches the scalar reference bit-close.

Runs on a small benchmark to keep the scalar baseline fast. Compares H + V
routing-congestion arrays after a get_routing() call.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from macro_place.loader import load_benchmark_from_dir

import importlib.util

V2 = ROOT / "system" / "v2" / "src" / "main.py"
spec = importlib.util.spec_from_file_location("v2_placer", V2)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


IBM_SOURCE_FMT = "external/MacroPlacement/Testcases/ICCAD04/{name}"


def _load(bench_name: str):
    src = ROOT / IBM_SOURCE_FMT.format(name=bench_name)
    benchmark, plc = load_benchmark_from_dir(str(src))
    return benchmark, plc


def run(bench_name: str):
    print(f"=== {bench_name} ===")
    benchmark, plc_scalar = _load(bench_name)
    # Use whatever positions the loader gave us (from initial.plc).
    plc_scalar.FLAG_UPDATE_CONGESTION = True
    plc_scalar.get_routing()
    H_scalar = np.asarray(plc_scalar.H_routing_cong, dtype=np.float64)
    V_scalar = np.asarray(plc_scalar.V_routing_cong, dtype=np.float64)

    # Build vectorized plc fresh (same initial placement).
    _, plc_vec = _load(bench_name)
    v2._patch_plc_congestion(plc_vec, benchmark)
    plc_vec.FLAG_UPDATE_CONGESTION = True
    plc_vec.get_routing()
    H_vec = np.asarray(plc_vec.H_routing_cong, dtype=np.float64)
    V_vec = np.asarray(plc_vec.V_routing_cong, dtype=np.float64)

    h_diff = np.abs(H_vec - H_scalar)
    v_diff = np.abs(V_vec - V_scalar)
    print(f"  H: max={h_diff.max():.3e} mean={h_diff.mean():.3e} sum_scalar={H_scalar.sum():.3f}")
    print(f"  V: max={v_diff.max():.3e} mean={v_diff.mean():.3e} sum_scalar={V_scalar.sum():.3f}")
    ok = (h_diff.max() < 1e-9) and (v_diff.max() < 1e-9)
    print(f"  {'OK' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
    ok = run(bench)
    sys.exit(0 if ok else 1)
