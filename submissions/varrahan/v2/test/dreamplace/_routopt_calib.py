"""DP1 calibration sweep (2026-05-27): with route_num_bins matched to the proxy
grid, sweep DREAMPlace's unit routing capacity both directions and measure the
proxy decomposition. RUDY utilization = demand/(bin_area*unit_cap), so larger
capacity → gentler area inflation. Physical capacity = routes_per_micron/scale
(scale=1000) is LOWER than DREAMPlace's default (more aggressive). This maps the
full response: does ANY capacity make routopt lower the proxy cong term?

    uv run python submissions/varrahan/v2/test/dreamplace/_routopt_calib.py ibm10
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
V2_DIR = THIS.parents[2]
REPO_ROOT = THIS.parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2_DIR / "src"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2_placer", str(V2_DIR / "src" / "submit.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_will_legalize = _v2._will_legalize
_proxy_decomp = _v2._proxy_decomp
_load_plc = _v2._load_plc

from dreamplace_bridge.run_bridge import launch_dreamplace_async, is_available  # noqa: E402

ICCAD_DIR = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
SCALE = 1000.0


def _decomp(dp_hard, dp_soft, bm, plc):
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    clip = dp_hard.copy()
    clip[:, 0] = np.clip(clip[:, 0], hw, cw - hw)
    clip[:, 1] = np.clip(clip[:, 1], hh, ch - hh)
    leg = _will_legalize(clip, movable, sizes, hw, hh, cw, ch, n)
    pl = bm.macro_positions.clone()
    pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)
    ns = int(min(dp_soft.shape[0], bm.num_soft_macros))
    if ns > 0:
        pl[n:n + ns, 0] = torch.tensor(dp_soft[:ns, 0], dtype=torch.float32)
        pl[n:n + ns, 1] = torch.tensor(dp_soft[:ns, 1], dtype=torch.float32)
    return _proxy_decomp(pl, bm, plc)


def _run(name, **kw):
    bm, _ = load_benchmark_from_dir(str(ICCAD_DIR / name))
    plc = _load_plc(name, bm)
    h = launch_dreamplace_async(
        str(ICCAD_DIR / name), plc=plc,
        scratch_root="/tmp/dp_routopt_calib", timeout_s=300.0,
        iterations=300, num_threads=4, soft_macros_movable=False,
        target_density=0.85, **kw,
    )
    full = h.wait_for_result_full(max_wait_s=300.0)
    if full is None:
        h.kill()
        return None
    return _decomp(full[0], full[1], bm, plc)


def run(name):
    print(f"\n=== {name} ===")
    bm, _ = load_benchmark_from_dir(str(ICCAD_DIR / name))
    plc = _load_plc(name, bm)
    gc, gr = int(plc.grid_col), int(plc.grid_row)
    uh_phys = plc.hroutes_per_micron / SCALE
    uv_phys = plc.vroutes_per_micron / SCALE
    print(f"  grid={gc}x{gr}  physical unit_cap h={uh_phys:.5f} v={uv_phys:.5f} "
          f"(default 1.5625/1.45)")

    # Baseline: routopt OFF.
    d = _run(name, routability_opt=False)
    print(f"  routopt OFF                          : proxy={d[0]:.4f} "
          f"wl={d[1]:.4f} den={d[2]:.4f} cong={d[3]:.4f}")

    # Sweep: route_num_bins matched to grid; unit_cap from physical → gentler.
    # mult scales physical capacity up (gentler inflation). mult=1 = physical.
    for mult in (1.0, 4.0, 16.0, 64.0):
        d = _run(name, routability_opt=True,
                 route_num_bins_x=gc, route_num_bins_y=gr,
                 unit_h_cap=uh_phys * mult, unit_v_cap=uv_phys * mult)
        if d is None:
            print(f"  routopt ON bins={gc}x{gr} cap×{mult:<5g}: DID NOT FINISH")
            continue
        print(f"  routopt ON bins={gc}x{gr} cap×{mult:<5g}: proxy={d[0]:.4f} "
              f"wl={d[1]:.4f} den={d[2]:.4f} cong={d[3]:.4f}")


if __name__ == "__main__":
    if not is_available():
        print("DREAMPlace not installed.")
        sys.exit(1)
    for nm in (sys.argv[1:] or ["ibm10"]):
        try:
            run(nm)
        except Exception as e:
            import traceback
            print(f"{nm}: ERROR {e}")
            traceback.print_exc()
