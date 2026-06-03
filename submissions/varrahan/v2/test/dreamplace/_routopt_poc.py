"""Proof-of-concept (DP1, 2026-05-27): does DREAMPlace's built-in routability
optimization (routability_opt_flag + adjust_rudy_area_flag) reduce the TILOS
PROXY congestion term?

DP_DIAG established our DREAMPlace candidates lose to 'best' purely on congestion.
DREAMPlace can place congestion-aware (RUDY map → inflate node areas in hotspots),
but it optimizes ITS routing estimate, not the proxy's. This test runs the same
DP config (hi-fix: td=0.85, soft_movable=False) with routopt OFF vs ON, legalizes
the hard macros exactly as the placer does, and decomposes the proxy. If routopt
drops the proxy cong term while keeping DP's wl/den edge, the lever is real.

    uv run python submissions/varrahan/v2/test/dreamplace/_routopt_poc.py ibm10 ibm12
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

from dreamplace_bridge.run_bridge import (  # noqa: E402
    launch_dreamplace_async, is_available,
)

ICCAD_DIR = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


def _dp_to_placement(dp_hard, dp_soft, bm, plc):
    """Replicate the placer: clip + legalize hard, place DP softs, decompose."""
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


def run(name):
    print(f"\n=== {name} ===")
    bm, _ = load_benchmark_from_dir(str(ICCAD_DIR / name))
    plc = _load_plc(name, bm)
    iccad = str(ICCAD_DIR / name)
    for ro in (False, True):
        h = launch_dreamplace_async(
            iccad, plc=plc,
            scratch_root=f"/tmp/dp_routopt_poc_{int(ro)}",
            timeout_s=300.0, iterations=300, num_threads=4,
            soft_macros_movable=False, target_density=0.85,
            routability_opt=ro,
        )
        full = h.wait_for_result_full(max_wait_s=300.0)
        if full is None:
            print(f"  routopt={ro}: DREAMPlace did not finish; killing")
            h.kill()
            continue
        dp_hard, dp_soft = full
        p, w, d, c = _dp_to_placement(dp_hard, dp_soft, bm, plc)
        print(f"  routopt={str(ro):5s}: proxy={p:.4f}  wl={w:.4f} "
              f"den={d:.4f} cong={c:.4f}  ({h.time_elapsed():.0f}s)")


if __name__ == "__main__":
    if not is_available():
        print("DREAMPlace not installed; cannot run.")
        sys.exit(1)
    for nm in (sys.argv[1:] or ["ibm10", "ibm12"]):
        try:
            run(nm)
        except Exception as e:
            import traceback
            print(f"{nm}: ERROR {e}")
            traceback.print_exc()
