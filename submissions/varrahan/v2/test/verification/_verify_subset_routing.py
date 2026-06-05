"""Verify _apply_net_routing_subset + _apply_macro_routing_subset (B3 phase 4).

Test 1: subset = ALL nets/macros should produce bit-equivalent H_flat /
        V_flat / H_macro_flat / V_macro_flat to the existing full
        `_vectorized_get_routing`.

Test 2: subset of a few nets at weight=+1 then -1 should net to zero
        modification of the flat arrays (subtract-then-add roundtrip).

Usage:
  uv run python submissions/varrahan/v2/test/verification/_verify_subset_routing.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[2]
REPO_ROOT = HERE.parents[5]
for p in (str(REPO_ROOT), str(V2_DIR / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from placer import (  # noqa: E402
    _patch_plc_wirelength,
    _patch_plc_congestion,
    _patch_plc_density,
    _fast_set_placement,
    _apply_net_routing_subset,
    _apply_macro_routing_subset,
    _apply_macro_routing,
    _ensure_pos_cache,
)
from macro_place.loader import load_benchmark_from_dir  # noqa: E402


def _build_full_routing(plc):
    """Replicate the routing-build part of _vectorized_get_routing.

    Returns (H_flat, V_flat, H_macro_flat, V_macro_flat) pre-smooth, pre-
    normalization. Used as the reference oracle for subset verification.
    """
    from placer import _vectorized_get_routing
    # Trigger a fresh compute, then capture the per-call internal state.
    # The function sets plc.V_routing_cong etc. AFTER smoothing+normalization,
    # so we re-run the routing math here without those final transforms.
    cache = plc._cong_cache
    wl = plc._wl_vec_cache

    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)

    n_cells = grid_row * grid_col
    H_flat = np.zeros(n_cells, dtype=np.float64)
    V_flat = np.zeros(n_cells, dtype=np.float64)
    H_macro_flat = np.zeros(n_cells, dtype=np.float64)
    V_macro_flat = np.zeros(n_cells, dtype=np.float64)

    n_nets = wl["n_nets"]
    if n_nets > 0:
        all_nets = np.arange(n_nets, dtype=np.int64)
        _apply_net_routing_subset(plc, all_nets, +1.0, H_flat, V_flat)

    n_hard = cache["n_hard"]
    if n_hard > 0:
        all_macros = np.arange(n_hard, dtype=np.int64)
        _apply_macro_routing_subset(plc, all_macros, +1.0, V_macro_flat, H_macro_flat)

    return H_flat, V_flat, H_macro_flat, V_macro_flat


def _build_full_routing_reference(plc):
    """Reference: build flats by duplicating _vectorized_get_routing's logic
    INLINE (without our new subset helper). Used as the oracle."""
    cache = plc._cong_cache
    wl = plc._wl_vec_cache
    from placer import (
        _apply_2pin_routing, _apply_3pin_routing_vec, _apply_macro_routing
    )

    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    grid_w = float(plc.width / grid_col)
    grid_h = float(plc.height / grid_row)

    n_cells = grid_row * grid_col
    H_flat = np.zeros(n_cells, dtype=np.float64)
    V_flat = np.zeros(n_cells, dtype=np.float64)
    H_macro_flat = np.zeros(n_cells, dtype=np.float64)
    V_macro_flat = np.zeros(n_cells, dtype=np.float64)

    n_nets = wl["n_nets"]
    if n_nets > 0:
        unique_ref = wl["unique_ref"]
        pos_cache = _ensure_pos_cache(plc)
        node_x = pos_cache[unique_ref, 0]
        node_y = pos_cache[unique_ref, 1]
        inv = wl["ref_inv"]
        pin_x = node_x[inv] + wl["x_off"]
        pin_y = node_y[inv] + wl["y_off"]
        pin_col = np.clip((pin_x / grid_w).astype(np.int64), 0, grid_col - 1)
        pin_row = np.clip((pin_y / grid_h).astype(np.int64), 0, grid_row - 1)
        pin_gcell = pin_row * grid_col + pin_col

        starts = cache["starts"]
        lengths = cache["lengths"]
        net_weights = wl["net_weights"]

        bucket_2_src = []
        bucket_2_snk = []
        bucket_2_w = []
        bucket_3_g0 = []
        bucket_3_g1 = []
        bucket_3_g2 = []
        bucket_3_w = []

        idx2 = cache["idx2"]
        if idx2.size > 0:
            src2 = pin_gcell[cache["s2"]]
            snk2 = pin_gcell[cache["s2p1"]]
            mask = src2 != snk2
            if mask.any():
                bucket_2_src.append(src2[mask])
                bucket_2_snk.append(snk2[mask])
                bucket_2_w.append(net_weights[idx2][mask])

        idx3 = cache["idx3"]
        if idx3.size > 0:
            g0 = pin_gcell[cache["s3"]]
            g1 = pin_gcell[cache["s3p1"]]
            g2 = pin_gcell[cache["s3p2"]]
            eq01 = g0 == g1
            eq02 = g0 == g2
            eq12 = g1 == g2
            eq_count = eq01.astype(np.int64) + eq02.astype(np.int64) + eq12.astype(np.int64)
            uniq2 = eq_count == 1
            uniq3 = eq_count == 0
            mask2 = uniq2
            if mask2.any():
                src_2 = g0[mask2]
                sink_2 = np.where(eq01[mask2], g2[mask2], g1[mask2])
                bucket_2_src.append(src_2)
                bucket_2_snk.append(sink_2)
                bucket_2_w.append(net_weights[idx3][mask2])
            if uniq3.any():
                idx3_uniq3 = idx3[uniq3]
                bucket_3_g0.append(g0[uniq3])
                bucket_3_g1.append(g1[uniq3])
                bucket_3_g2.append(g2[uniq3])
                bucket_3_w.append(net_weights[idx3_uniq3])

        idx_big = cache["idx_big"]
        if idx_big.size > 0:
            starts_big = cache["starts_big"]
            sink_total = cache["sink_total"]
            src_gcells_big = pin_gcell[starts_big]
            if sink_total > 0:
                B = cache["B_big"]
                net_local_ids = cache["net_local_ids"]
                global_pin_idx = cache["global_pin_idx"]
                sink_gcells = pin_gcell[global_pin_idx]
                mask_not_src = sink_gcells != src_gcells_big[net_local_ids]
                if mask_not_src.any():
                    nli_ns = net_local_ids[mask_not_src]
                    sg_ns = sink_gcells[mask_not_src]
                    order = np.lexsort((sg_ns, nli_ns))
                    nli_sorted = nli_ns[order]
                    sg_sorted = sg_ns[order]
                    keep = np.empty(sg_sorted.size, dtype=bool)
                    keep[0] = True
                    if sg_sorted.size > 1:
                        keep[1:] = (
                            (nli_sorted[1:] != nli_sorted[:-1])
                            | (sg_sorted[1:] != sg_sorted[:-1])
                        )
                    nli_uniq = nli_sorted[keep]
                    sg_uniq = sg_sorted[keep]
                    uniq_sink_counts = np.bincount(nli_uniq, minlength=B)
                    n_uniq_total = 1 + uniq_sink_counts
                    net_is_3 = n_uniq_total == 3
                    net_is_starlike = ~net_is_3
                    mask_starlike = net_is_starlike[nli_uniq]
                    if mask_starlike.any():
                        nli_emit = nli_uniq[mask_starlike]
                        bucket_2_src.append(src_gcells_big[nli_emit])
                        bucket_2_snk.append(sg_uniq[mask_starlike])
                        bucket_2_w.append(net_weights[idx_big[nli_emit]])
                    if net_is_3.any():
                        cum_counts = np.cumsum(uniq_sink_counts)
                        net3_ids = np.where(net_is_3)[0]
                        ends = cum_counts[net3_ids]
                        bucket_3_g0.append(src_gcells_big[net3_ids])
                        bucket_3_g1.append(sg_uniq[ends - 2])
                        bucket_3_g2.append(sg_uniq[ends - 1])
                        bucket_3_w.append(net_weights[idx_big[net3_ids]])

        if bucket_2_src:
            src_flat = np.concatenate(bucket_2_src)
            snk_flat = np.concatenate(bucket_2_snk)
            w_arr = np.concatenate(bucket_2_w)
            _apply_2pin_routing(
                H_flat, V_flat,
                src_flat // grid_col, src_flat % grid_col,
                snk_flat // grid_col, snk_flat % grid_col,
                w_arr, grid_row, grid_col,
            )
        if bucket_3_g0:
            g0_arr = np.concatenate(bucket_3_g0)
            g1_arr = np.concatenate(bucket_3_g1)
            g2_arr = np.concatenate(bucket_3_g2)
            w_arr3 = np.concatenate(bucket_3_w)
            _apply_3pin_routing_vec(H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col)

    n_hard = cache["n_hard"]
    if n_hard > 0:
        hard_indices = cache["hard_indices"]
        hard_indices_arr = cache.get("hard_indices_arr")
        if hard_indices_arr is None:
            hard_indices_arr = np.asarray(hard_indices, dtype=np.int64)
            cache["hard_indices_arr"] = hard_indices_arr
        pos_cache = _ensure_pos_cache(plc)
        hard_x = pos_cache[hard_indices_arr, 0]
        hard_y = pos_cache[hard_indices_arr, 1]
        _apply_macro_routing(
            V_macro_flat, H_macro_flat, hard_x, hard_y,
            cache["hard_half_w"], cache["hard_half_h"],
            grid_w, grid_h, grid_row, grid_col,
            float(plc.vrouting_alloc), float(plc.hrouting_alloc),
        )

    return H_flat, V_flat, H_macro_flat, V_macro_flat


def run_bench(bname):
    print(f"\n=== {bname} ===", flush=True)
    bench, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{bname}")
    _patch_plc_wirelength(plc)
    _patch_plc_congestion(plc, bench)
    _patch_plc_density(plc, bench)
    placement_np = bench.macro_positions.cpu().numpy().astype(np.float64)
    _fast_set_placement(plc, placement_np, bench)
    plc.get_congestion_cost()  # warm caches

    # Test 1: subset = ALL nets/macros → should match the inline-replicated full path.
    Hr, Vr, Hmr, Vmr = _build_full_routing_reference(plc)
    Hs, Vs, Hms, Vms = _build_full_routing(plc)

    dH = np.abs(Hr - Hs).max()
    dV = np.abs(Vr - Vs).max()
    dHm = np.abs(Hmr - Hms).max()
    dVm = np.abs(Vmr - Vms).max()
    ok1 = max(dH, dV, dHm, dVm) < 1e-9
    print(f"  Test 1 (subset=all): max|ΔH|={dH:.2e}, |ΔV|={dV:.2e}, |ΔHm|={dHm:.2e}, |ΔVm|={dVm:.2e}  {'PASS' if ok1 else 'FAIL'}",
          flush=True)

    # Test 2: roundtrip - apply subset of N nets at +1, then -1, should net to zero.
    n_nets = plc._wl_vec_cache["n_nets"]
    sample_nets = np.array(sorted(np.random.RandomState(7).choice(n_nets, size=min(200, n_nets), replace=False)),
                            dtype=np.int64)
    H_test = Hr.copy()
    V_test = Vr.copy()
    _apply_net_routing_subset(plc, sample_nets, +1.0, H_test, V_test)
    _apply_net_routing_subset(plc, sample_nets, -1.0, H_test, V_test)
    dH2 = np.abs(H_test - Hr).max()
    dV2 = np.abs(V_test - Vr).max()
    ok2 = max(dH2, dV2) < 1e-9
    print(f"  Test 2 (subset ± roundtrip): max|ΔH|={dH2:.2e}, |ΔV|={dV2:.2e}  {'PASS' if ok2 else 'FAIL'}",
          flush=True)

    # Test 3: ALL = sum of SUBSETS. Build flats by accumulating two disjoint subsets.
    half = n_nets // 2
    s1 = np.arange(half, dtype=np.int64)
    s2 = np.arange(half, n_nets, dtype=np.int64)
    H_acc = np.zeros_like(Hr)
    V_acc = np.zeros_like(Vr)
    _apply_net_routing_subset(plc, s1, +1.0, H_acc, V_acc)
    _apply_net_routing_subset(plc, s2, +1.0, H_acc, V_acc)
    dH3 = np.abs(H_acc - Hr).max()
    dV3 = np.abs(V_acc - Vr).max()
    ok3 = max(dH3, dV3) < 1e-9
    print(f"  Test 3 (disjoint-subset additivity): max|ΔH|={dH3:.2e}, |ΔV|={dV3:.2e}  {'PASS' if ok3 else 'FAIL'}",
          flush=True)

    return ok1 and ok2 and ok3


def main():
    results = {}
    for bname in ("ibm01", "ibm04", "ibm10"):
        results[bname] = run_bench(bname)
    print("\n=== Summary ===")
    for bname, ok in results.items():
        print(f"  {bname}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
