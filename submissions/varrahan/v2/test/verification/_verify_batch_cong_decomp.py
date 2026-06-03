"""Keystone correctness check for the S2 GPU-batched candidate evaluator.

Proves the decomposition the batch relies on (box-filter smoothing is linear):

    cong(base + candidate_delta)
        == top5%( [ base_V_smoothed + smooth(netV_delta/routes) + (base_Vm + macroV_delta),
                    base_H_smoothed + smooth(netH_delta/routes) + (base_Hm + macroH_delta) ] )

i.e. a candidate's congestion = a SHARED base (the macro removed) plus a
localized, batchable delta. If this holds bit-exact against the live
apply+resmooth path that `_trial_at` uses, the batched cong evaluator is sound.

    uv run python submissions/varrahan/v2/test/verification/_verify_batch_cong_decomp.py [ibmNN]
"""
import sys
from pathlib import Path

_V2_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_V2_DIR / "src"))

import numpy as np  # noqa: E402

from placer import _exact_proxy, IncrementalScorer  # noqa: E402
from placer.routing.apply import (  # noqa: E402
    _apply_net_routing_struct, _apply_macro_routing_subset, _smooth_routing_cong_vec,
)
from macro_place.loader import load_benchmark_from_dir  # noqa: E402


def _cong_from_state(sc, H_sm, V_sm, Hm_flat, Vm_flat):
    """Replicate IncrementalScorer._compute_cong_cost from explicit state."""
    Hm = Hm_flat / sc.grid_h_routes
    Vm = Vm_flat / sc.grid_v_routes
    xx = np.concatenate([V_sm.ravel() + Vm, H_sm.ravel() + Hm])
    n = xx.size
    cnt = int(n * 0.05)
    if cnt == 0:
        return float(xx.max())
    top = np.partition(xx, n - cnt)[n - cnt:]
    return float(top.sum() / cnt)


def _check(name, n_macros=8, n_cands=6):
    src = _V2_DIR.parents[2] / "external/MacroPlacement/Testcases/ICCAD04" / name
    bm, plc = load_benchmark_from_dir(str(src))
    pl = bm.macro_positions.cpu().numpy().astype(np.float64)
    _exact_proxy(bm.macro_positions, bm, plc)
    sc = IncrementalScorer(plc, bm, pl)
    cw, ch = bm.canvas_width, bm.canvas_height
    rng = np.random.RandomState(0)

    worst = 0.0
    checked = 0
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[: bm.num_hard_macros].numpy()
    hot = [i for i in range(bm.num_hard_macros) if movable[i]]
    rng.shuffle(hot)

    for i_hard in hot[:n_macros]:
        prep = sc._prepare_move(i_hard)              # base = macro removed
        struct = prep["struct"]
        macro_subset = prep["macro_subset"]
        if struct is None:
            sc._revert_prep(prep)
            continue
        # Snapshot the shared base (macro-removed) state.
        baseH_sm = sc.H_smoothed.copy()
        baseV_sm = sc.V_smoothed.copy()
        baseHm = sc.H_macro_flat.copy()
        baseVm = sc.V_macro_flat.copy()
        baseH = sc.H_flat.copy()
        baseV = sc.V_flat.copy()

        for _ in range(n_cands):
            nx = float(rng.uniform(0, cw))
            ny = float(rng.uniform(0, ch))

            # --- DIRECT: live apply + resmooth, then _compute_cong_cost (the _trial_at path).
            sc._apply_pos(prep["i_module"], nx, ny)
            bb = _apply_net_routing_struct(sc.plc, struct, +1.0, sc.H_flat, sc.V_flat)
            if macro_subset.size:
                _apply_macro_routing_subset(sc.plc, macro_subset, +1.0,
                                            sc.V_macro_flat, sc.H_macro_flat)
            if bb is not None:
                sc._resmooth_bbox(*bb)
            cong_direct = sc._compute_cong_cost()
            # revert to base
            sc.H_flat[:] = baseH; sc.V_flat[:] = baseV
            sc.H_macro_flat[:] = baseHm; sc.V_macro_flat[:] = baseVm
            sc.H_smoothed[:] = baseH_sm; sc.V_smoothed[:] = baseV_sm
            sc._apply_pos(prep["i_module"], prep["old_ix"], prep["old_iy"])

            # --- DECOMP: deltas on zero grids + base + smooth(delta), linearity.
            sc._apply_pos(prep["i_module"], nx, ny)
            Hd = np.zeros_like(sc.H_flat); Vd = np.zeros_like(sc.V_flat)
            _apply_net_routing_struct(sc.plc, struct, +1.0, Hd, Vd)
            Hmd = np.zeros_like(sc.H_macro_flat); Vmd = np.zeros_like(sc.V_macro_flat)
            if macro_subset.size:
                _apply_macro_routing_subset(sc.plc, macro_subset, +1.0, Vmd, Hmd)
            sc._apply_pos(prep["i_module"], prep["old_ix"], prep["old_iy"])

            sm_Hd = _smooth_routing_cong_vec(
                Hd / sc.grid_h_routes, sc.grid_row, sc.grid_col, sc.smooth_range, axis_h=True
            ).reshape(sc.grid_row, sc.grid_col)
            sm_Vd = _smooth_routing_cong_vec(
                Vd / sc.grid_v_routes, sc.grid_row, sc.grid_col, sc.smooth_range, axis_h=False
            ).reshape(sc.grid_row, sc.grid_col)
            H_sm = baseH_sm + sm_Hd
            V_sm = baseV_sm + sm_Vd
            cong_decomp = _cong_from_state(sc, H_sm, V_sm, baseHm + Hmd, baseVm + Vmd)

            worst = max(worst, abs(cong_direct - cong_decomp))
            checked += 1

        sc._revert_prep(prep)

    status = "PASS" if worst < 1e-9 else "FAIL"
    print(f"  {name}: {status}  ({checked} candidate checks, max |Δcong| = {worst:.2e})")
    return worst < 1e-9


if __name__ == "__main__":
    benches = sys.argv[1:] or ["ibm01", "ibm10"]
    ok = all(_check(b) for b in benches)
    print("=== Summary:", "PASS ===" if ok else "FAIL ===")
    sys.exit(0 if ok else 1)
