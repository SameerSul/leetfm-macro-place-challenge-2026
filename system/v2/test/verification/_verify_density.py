"""Verify the vectorized get_grid_cells_density matches the scalar reference."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
from macro_place.loader import load_benchmark_from_dir

import importlib.util
V2 = ROOT / "system" / "v2" / "src" / "main.py"
spec = importlib.util.spec_from_file_location("v2_placer", V2)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

IBM_SOURCE_FMT = "external/MacroPlacement/Testcases/ICCAD04/{name}"


def _load(name):
    src = ROOT / IBM_SOURCE_FMT.format(name=name)
    return load_benchmark_from_dir(str(src))


def run(name):
    print(f"=== {name} ===")
    benchmark, plc_s = _load(name)
    plc_s.FLAG_UPDATE_DENSITY = True
    gd_scalar = np.asarray(plc_s.get_grid_cells_density(), dtype=np.float64)
    cost_scalar = float(plc_s.get_density_cost())

    _, plc_v = _load(name)
    v2._patch_plc_density(plc_v, benchmark)
    plc_v.FLAG_UPDATE_DENSITY = True
    gd_vec = np.asarray(plc_v.get_grid_cells_density(), dtype=np.float64)
    cost_vec = float(plc_v.get_density_cost())

    d = np.abs(gd_vec - gd_scalar)
    print(f"  grid_cells: max={d.max():.3e} mean={d.mean():.3e} sum_scalar={gd_scalar.sum():.6f}")
    print(f"  density_cost scalar={cost_scalar:.6f} vec={cost_vec:.6f} diff={abs(cost_scalar-cost_vec):.3e}")
    ok = (d.max() < 1e-10) and (abs(cost_scalar - cost_vec) < 1e-10)
    print(f"  {'OK' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
    ok = run(bench)
    sys.exit(0 if ok else 1)
