"""Verify _score_candidates_hard (batched) == sequential _trial_at loop.

The batched evaluator selects which candidate to commit; correctness here means
its per-candidate scores match the sequential path closely enough that selection
is identical (WL is exact; cong/density use the base+delta decomposition, so a
few ULP of drift is expected and harmless — the R2 round re-gates on exact proxy).

    uv run python submissions/varrahan/v2/test/verification/_verify_batch_eval.py [ibmNN]
"""
import sys
from pathlib import Path

_V2_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_V2_DIR / "src"))

import numpy as np  # noqa: E402

from placer import _exact_proxy, IncrementalScorer  # noqa: E402
from placer.scoring.batch_eval import (  # noqa: E402
    _score_candidates_hard, _score_candidates_hard_gpu,
)
from placer.config import _USE_GPU  # noqa: E402
from macro_place.loader import load_benchmark_from_dir  # noqa: E402


def _check(name, n_macros=10, n_cands=20, tol=1e-9):
    src = _V2_DIR.parents[2] / "external/MacroPlacement/Testcases/ICCAD04" / name
    bm, plc = load_benchmark_from_dir(str(src))
    pl = bm.macro_positions.cpu().numpy().astype(np.float64)
    _exact_proxy(bm.macro_positions, bm, plc)
    sc = IncrementalScorer(plc, bm, pl)
    cw, ch = bm.canvas_width, bm.canvas_height
    rng = np.random.RandomState(1)
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[: bm.num_hard_macros].numpy()
    hot = [i for i in range(bm.num_hard_macros) if movable[i]]
    rng.shuffle(hot)

    worst = 0.0
    sel_mismatch = 0
    checked = 0
    for i_hard in hot[:n_macros]:
        prep = sc._prepare_move(i_hard)
        if prep["struct"] is None:
            sc._revert_prep(prep)
            continue
        cands = np.stack([rng.uniform(0, cw, n_cands), rng.uniform(0, ch, n_cands)], axis=1)

        batched = (_score_candidates_hard_gpu(sc, prep, cands) if _USE_GPU
                   else _score_candidates_hard(sc, prep, cands))
        seq = np.array([sc._trial_at(prep, (cands[k, 0], cands[k, 1])) for k in range(n_cands)])

        worst = max(worst, float(np.max(np.abs(batched - seq))))
        if int(np.argmin(batched)) != int(np.argmin(seq)):
            sel_mismatch += 1
        checked += n_cands
        sc._revert_prep(prep)

    label = "GPU" if _USE_GPU else "CPU-ref"
    status = "PASS" if worst < tol else "FAIL"
    print(f"  {name} [{label}]: {status}  ({checked} scores, max |Δ| = {worst:.2e}, "
          f"argmin mismatches = {sel_mismatch}/{n_macros})")
    return worst < tol


if __name__ == "__main__":
    benches = sys.argv[1:] or ["ibm01", "ibm10"]
    ok = all(_check(b) for b in benches)
    print("=== Summary:", "PASS ===" if ok else "FAIL ===")
    sys.exit(0 if ok else 1)
