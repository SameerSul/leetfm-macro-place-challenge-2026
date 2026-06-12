"""Stress-test vectorized congestion + density against scalar across MANY
randomly perturbed placements. Catches edge cases in PARTIAL_OVERLAP and
3-pin classification that initial.plc positions might not exercise.
"""
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


def perturb_movable_positions(plc, rng, scale_frac=0.05):
    """Move movable hard macros by random offsets up to scale_frac × canvas."""
    cw, ch = plc.get_canvas_width_height()
    for i in plc.hard_macro_indices + plc.soft_macro_indices:
        m = plc.modules_w_pins[i]
        x, y = m.get_pos()
        dx = rng.normal(0, scale_frac * cw)
        dy = rng.normal(0, scale_frac * ch)
        # Keep within canvas
        nx = max(m.get_width() / 2, min(cw - m.get_width() / 2, x + dx))
        ny = max(m.get_height() / 2, min(ch - m.get_height() / 2, y + dy))
        m.set_pos(float(nx), float(ny))


def run(name, n_trials=5, seed=0):
    print(f"=== {name} (n_trials={n_trials}) ===")
    benchmark, _ = _load(name)
    max_h = 0.0
    max_v = 0.0
    max_dens_cell = 0.0
    max_dens_cost = 0.0
    for trial in range(n_trials):
        _, plc_s = _load(name)
        _, plc_v = _load(name)
        # Same perturbation for both
        rng_s = np.random.RandomState(seed + trial)
        rng_v = np.random.RandomState(seed + trial)
        perturb_movable_positions(plc_s, rng_s)
        perturb_movable_positions(plc_v, rng_v)
        # Scalar
        plc_s.FLAG_UPDATE_CONGESTION = True
        plc_s.FLAG_UPDATE_DENSITY = True
        plc_s.get_routing()
        H_s = np.asarray(plc_s.H_routing_cong, dtype=np.float64)
        V_s = np.asarray(plc_s.V_routing_cong, dtype=np.float64)
        d_s = np.asarray(plc_s.get_grid_cells_density(), dtype=np.float64)
        cost_s = float(plc_s.get_density_cost())
        # Vec
        v2._patch_plc_congestion(plc_v, benchmark)
        v2._patch_plc_density(plc_v, benchmark)
        plc_v.FLAG_UPDATE_CONGESTION = True
        plc_v.FLAG_UPDATE_DENSITY = True
        plc_v.get_routing()
        H_v = np.asarray(plc_v.H_routing_cong, dtype=np.float64)
        V_v = np.asarray(plc_v.V_routing_cong, dtype=np.float64)
        d_v = np.asarray(plc_v.get_grid_cells_density(), dtype=np.float64)
        cost_v = float(plc_v.get_density_cost())
        h_diff = np.abs(H_v - H_s).max()
        v_diff = np.abs(V_v - V_s).max()
        d_diff = np.abs(d_v - d_s).max()
        cc_diff = abs(cost_v - cost_s)
        max_h = max(max_h, h_diff)
        max_v = max(max_v, v_diff)
        max_dens_cell = max(max_dens_cell, d_diff)
        max_dens_cost = max(max_dens_cost, cc_diff)
    print(f"  Hcong max: {max_h:.3e}")
    print(f"  Vcong max: {max_v:.3e}")
    print(f"  dens_cell max: {max_dens_cell:.3e}")
    print(f"  dens_cost max: {max_dens_cost:.3e}")
    ok = max(max_h, max_v, max_dens_cell) < 1e-8 and max_dens_cost < 1e-8
    print(f"  {'OK' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm04"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    ok = run(bench, n)
    sys.exit(0 if ok else 1)
