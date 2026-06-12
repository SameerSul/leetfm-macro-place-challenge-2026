"""Decompose proxy into weighted WL / density / congestion contributions.

Runs on the baseline legalization (fast, no full pipeline) so we can see which
term dominates the proxy and how much it varies across benchmarks. The point is
strategic: where is the mass, and where is the spread (= movable headroom).

    uv run python system/v2/test/diagnostic/_term_breakdown.py ibm01 ibm04 ibm10 ibm16
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
V2_DIR = THIS.parents[2]
REPO_ROOT = THIS.parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2_DIR / "src"))

from macro_place.loader import load_benchmark_from_dir  # type: ignore  # noqa: E402

# Load v2 placer under a distinct name to dodge the sameer_v1 module collision.
_spec = importlib.util.spec_from_file_location("v2_placer", str(V2_DIR / "src" / "main.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_exact_proxy = _v2._exact_proxy
_will_legalize = _v2._will_legalize

ICCAD_DIR = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


def run(name: str):
    bm, plc = load_benchmark_from_dir(str(ICCAD_DIR / name))
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    init_pos = bm.macro_positions[:n].numpy().copy().astype(np.float64)

    leg = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
    pl = bm.macro_positions.clone()
    pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)

    p = _exact_proxy(pl, bm, plc)
    wl = float(plc.get_cost())
    den = float(plc.get_density_cost())
    cong = float(plc.get_congestion_cost())

    wl_c, den_c, cong_c = 1.0 * wl, 0.5 * den, 0.5 * cong
    tot = wl_c + den_c + cong_c
    print(f"{name:7s} proxy={p:.4f} (chk {tot:.4f})  "
          f"WL {wl_c:.4f} ({100*wl_c/tot:4.1f}%)  "
          f"DEN {den_c:.4f} ({100*den_c/tot:4.1f}%)  "
          f"CONG {cong_c:.4f} ({100*cong_c/tot:4.1f}%)")


if __name__ == "__main__":
    names = sys.argv[1:] or ["ibm01", "ibm04", "ibm10", "ibm16"]
    print("Weighted proxy decomposition on BASELINE legalization:")
    print("  proxy = 1.0*WL + 0.5*DEN + 0.5*CONG\n")
    for nm in names:
        try:
            run(nm)
        except Exception as e:
            print(f"{nm:7s} ERROR: {e}")
