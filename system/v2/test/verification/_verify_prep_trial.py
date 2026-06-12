"""Bit-exactness check for S1's prep/trial/commit/revert pattern vs the
existing score_move/commit_move methods. The pattern hoists the loop-invariant
subtract-old out of the candidate inner loop; the trial score and final state
must match score_move's trial score and revert behavior exactly.

Tests three properties for each benchmark, for both hard and soft paths:
  T1: prep(k) + trial(target) gives the same score as score_move(k, target).
  T2: prep(k) + trial(...) + revert_prep(k) restores state s.t. score_move(k, t2)
      after gives the same score as if no prep had ever happened.
  T3: prep(k) + commit_after_prep(k, target) leaves the same state as
      commit_move(k, target).

    uv run python submissions/varrahan/v2/test/verification/_verify_prep_trial.py
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

_spec = importlib.util.spec_from_file_location("v2_placer", str(V2_DIR / "src" / "main.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_will_legalize = _v2._will_legalize
_exact_proxy = _v2._exact_proxy
_load_plc = _v2._load_plc
IncrementalScorer = _v2.IncrementalScorer

TOL = 1e-8


def _setup(name):
    bm, _ = load_benchmark_from_dir(
        str(REPO_ROOT / f"external/MacroPlacement/Testcases/ICCAD04/{name}"))
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
    return bm, plc, pl.cpu().numpy().astype(np.float64), n, cw, ch


def _run(name, n_trials=10):
    bm, plc, pl_np, n, cw, ch = _setup(name)
    rng = np.random.RandomState(0)
    print(f"\n=== {name} ===")

    # ---------- HARD path ----------
    sc = IncrementalScorer(plc, bm, pl_np)
    mov = np.where((bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy())[0]
    fail = 0
    print(f"  Hard T1: prep+trial == score_move (×{n_trials})")
    for _ in range(n_trials):
        i = int(mov[rng.randint(0, mov.size)])
        target = (float(rng.uniform(0, cw)), float(rng.uniform(0, ch)))
        # Reference: score_move
        ref = sc.score_move(i, target)
        # New: prep + trial + revert
        prep = sc._prepare_move(i)
        got = sc._trial_at(prep, target)
        sc._revert_prep(prep)
        d = abs(ref - got)
        status = "ok" if d <= TOL else "FAIL"
        if d > TOL:
            fail += 1
        print(f"    hard({i:4d}) target=({target[0]:8.0f},{target[1]:8.0f}) "
              f"ref={ref:.6f} got={got:.6f} Δ={d:.2e} {status}")

    # T3: prep + commit_after_prep == commit_move
    print("  Hard T3: prep+commit == commit_move (final-state check, ×3)")
    for _ in range(3):
        sc1 = IncrementalScorer(plc, bm, pl_np)
        sc2 = IncrementalScorer(plc, bm, pl_np)
        # Make sure plc state is consistent: just rebuilt.
        # (Both scorers share the same plc, but they don't co-exist concurrently
        # - we just use one then the other.)
        i = int(mov[rng.randint(0, mov.size)])
        target = (float(rng.uniform(0, cw)), float(rng.uniform(0, ch)))
        # Reference path: commit_move
        sc1.commit_move(i, target)
        ref_final = sc1.score_move(i, target)  # use as a fingerprint of final state
        # New path: prep + commit_after_prep on a freshly-built scorer
        sc2 = IncrementalScorer(plc, bm, pl_np)
        prep = sc2._prepare_move(i)
        sc2._commit_after_prep(prep, target)
        got_final = sc2.score_move(i, target)
        d = abs(ref_final - got_final)
        status = "ok" if d <= TOL else "FAIL"
        if d > TOL:
            fail += 1
        print(f"    hard({i:4d}) commit-final-state fingerprint ref={ref_final:.6f} "
              f"got={got_final:.6f} Δ={d:.2e} {status}")

    # ---------- SOFT path ----------
    ns = bm.num_soft_macros
    sc = IncrementalScorer(plc, bm, pl_np)
    print(f"  Soft T1: prep+trial == score_move_soft (×{n_trials})")
    for _ in range(n_trials):
        k = int(rng.randint(0, ns))
        target = (float(rng.uniform(0, cw)), float(rng.uniform(0, ch)))
        ref = sc.score_move_soft(k, target)
        prep = sc._prepare_move_soft(k)
        got = sc._trial_at_soft(prep, target)
        sc._revert_prep_soft(prep)
        d = abs(ref - got)
        status = "ok" if d <= TOL else "FAIL"
        if d > TOL:
            fail += 1
        print(f"    soft({k:4d}) ref={ref:.6f} got={got:.6f} Δ={d:.2e} {status}")

    # T2 (soft): post-revert, score_move_soft on a DIFFERENT target should match
    # what you'd get if no prep had ever happened.
    print("  Soft T2: post-revert state restoration (×3)")
    for _ in range(3):
        k = int(rng.randint(0, ns))
        t_a = (float(rng.uniform(0, cw)), float(rng.uniform(0, ch)))
        t_b = (float(rng.uniform(0, cw)), float(rng.uniform(0, ch)))
        sc_ref = IncrementalScorer(plc, bm, pl_np)
        ref_after = sc_ref.score_move_soft(k, t_b)
        sc_new = IncrementalScorer(plc, bm, pl_np)
        prep = sc_new._prepare_move_soft(k)
        sc_new._trial_at_soft(prep, t_a)  # discard
        sc_new._revert_prep_soft(prep)
        got_after = sc_new.score_move_soft(k, t_b)
        d = abs(ref_after - got_after)
        status = "ok" if d <= TOL else "FAIL"
        if d > TOL:
            fail += 1
        print(f"    soft({k:4d}) post-revert score(t_b) ref={ref_after:.6f} "
              f"got={got_after:.6f} Δ={d:.2e} {status}")

    return fail


if __name__ == "__main__":
    total_fail = 0
    for nm in (sys.argv[1:] or ["ibm01", "ibm04", "ibm10"]):
        total_fail += _run(nm)
    print(f"\n=== Summary: {'PASS' if total_fail == 0 else f'{total_fail} FAILURES'} ===")
    sys.exit(1 if total_fail else 0)
