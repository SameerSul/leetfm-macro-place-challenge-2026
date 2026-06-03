"""cProfile the internals of score_move_soft to attribute the ~78% of per-move
time that P5 (_profile_move.py) left unbroken-down (cong=20%, density=0.7%).
Tells us whether the snapshot copies, the touched-net routing apply
(_apply_net_routing_subset), _apply_pos, or the per-net HPWL dominate — i.e.
what a per-move speedup should target.

    uv run python submissions/varrahan/v2/test/diagnostic/_profile_move_internals.py ibm10
"""
import sys
import cProfile
import pstats
import io
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
_exact_proxy = _v2._exact_proxy
_load_plc = _v2._load_plc
IncrementalScorer = _v2.IncrementalScorer


def run(name, n_moves=4000):
    bm, _ = load_benchmark_from_dir(
        str(REPO_ROOT / f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    )
    plc = _load_plc(name, bm)
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    leg = _will_legalize(bm.macro_positions[:n].numpy().astype(np.float64),
                         movable, sizes, hw, hh, cw, ch, n)
    pl = bm.macro_positions.clone()
    pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)
    _exact_proxy(pl, bm, plc)
    sc = IncrementalScorer(plc, bm, pl.cpu().numpy().astype(np.float64))
    ns = bm.num_soft_macros
    rng = np.random.RandomState(0)
    coords = [(int(rng.randint(0, ns)), float(rng.uniform(0, cw)), float(rng.uniform(0, ch)))
              for _ in range(n_moves)]

    def work():
        for k, x, y in coords:
            sc.score_move_soft(k, (x, y))

    work()  # warm
    pr = cProfile.Profile()
    pr.enable()
    work()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(18)
    print(f"=== {name}: {n_moves} score_move_soft calls (grid={bm.grid_rows*bm.grid_cols}, soft={ns}) ===")
    print(s.getvalue())


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["ibm10"]):
        run(nm)
