"""Profile _vectorized_get_routing to find post-B3p3 hot spots (B4).

Per-stage time-stamp instrumentation. Runs the function N times with
forced dirty flag so each call does the full recompute. Reports per-
stage averages.

Usage:
  uv run python submissions/varrahan/v2/test/diagnostic/_profile_get_routing.py
"""
from __future__ import annotations

import sys
import time
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
    _ensure_pos_cache,
    _apply_2pin_routing,
    _apply_3pin_routing_vec,
    _smooth_routing_cong_vec,
    _apply_macro_routing,
)
from macro_place.loader import load_benchmark_from_dir  # noqa: E402


def profile_get_routing(plc, n_iters: int = 50) -> dict:
    """Replicate _vectorized_get_routing with per-stage timing.

    Returns a dict mapping stage name → average ms per call.
    """
    cache = plc._cong_cache
    wl = plc._wl_vec_cache
    timings: dict[str, list[float]] = {}

    def record(name, dt_ms):
        timings.setdefault(name, []).append(dt_ms)

    for _ in range(n_iters):
        # Force a full recompute by mutating one position (invalidates plc dirty flag)
        plc.modules_w_pins[plc.hard_macro_indices[0]].set_pos(
            float(plc.modules_w_pins[plc.hard_macro_indices[0]].get_pos()[0] + 0.01),
            float(plc.modules_w_pins[plc.hard_macro_indices[0]].get_pos()[1]),
        )
        # Sync pos_cache
        pc = _ensure_pos_cache(plc)
        pc[plc.hard_macro_indices[0], 0] += 0.01

        t0 = time.perf_counter()

        grid_col = int(plc.grid_col)
        grid_row = int(plc.grid_row)
        grid_w = float(plc.width / grid_col)
        grid_h = float(plc.height / grid_row)
        plc.grid_width = grid_w
        plc.grid_height = grid_h
        grid_v_routes = grid_w * plc.vroutes_per_micron
        grid_h_routes = grid_h * plc.hroutes_per_micron
        plc.grid_v_routes = grid_v_routes
        plc.grid_h_routes = grid_h_routes

        n_cells = grid_row * grid_col
        H_flat = np.zeros(n_cells, dtype=np.float64)
        V_flat = np.zeros(n_cells, dtype=np.float64)
        H_macro_flat = np.zeros(n_cells, dtype=np.float64)
        V_macro_flat = np.zeros(n_cells, dtype=np.float64)

        t1 = time.perf_counter()
        record("0_alloc_flats", (t1 - t0) * 1000)

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

            t2 = time.perf_counter()
            record("1_pin_gcell_compute", (t2 - t1) * 1000)

            starts = cache["starts"]
            lengths = cache["lengths"]
            net_weights = wl["net_weights"]

            bucket_2_src_flat = []
            bucket_2_snk_flat = []
            bucket_2_w_arrs = []
            bucket_3_g0 = []
            bucket_3_g1 = []
            bucket_3_g2 = []
            bucket_3_w_arrs = []

            idx2 = np.where(lengths == 2)[0]
            if idx2.size > 0:
                s2 = starts[idx2]
                src2 = pin_gcell[s2]
                snk2 = pin_gcell[s2 + 1]
                mask = src2 != snk2
                if mask.any():
                    bucket_2_src_flat.append(src2[mask])
                    bucket_2_snk_flat.append(snk2[mask])
                    bucket_2_w_arrs.append(net_weights[idx2][mask])

            t3 = time.perf_counter()
            record("2_dispatch_len2", (t3 - t2) * 1000)

            idx3 = np.where(lengths == 3)[0]
            if idx3.size > 0:
                s3 = starts[idx3]
                g0 = pin_gcell[s3]
                g1 = pin_gcell[s3 + 1]
                g2 = pin_gcell[s3 + 2]
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
                    bucket_2_src_flat.append(src_2)
                    bucket_2_snk_flat.append(sink_2)
                    bucket_2_w_arrs.append(net_weights[idx3][mask2])
                if uniq3.any():
                    idx3_uniq3 = idx3[uniq3]
                    bucket_3_g0.append(g0[uniq3])
                    bucket_3_g1.append(g1[uniq3])
                    bucket_3_g2.append(g2[uniq3])
                    bucket_3_w_arrs.append(net_weights[idx3_uniq3])

            t4 = time.perf_counter()
            record("3_dispatch_len3", (t4 - t3) * 1000)

            idx_big = np.where(lengths >= 4)[0]
            if idx_big.size > 0:
                starts_big = starts[idx_big]
                lengths_big = lengths[idx_big]
                src_gcells_big = pin_gcell[starts_big]
                sink_lens = lengths_big - 1
                sink_total = int(sink_lens.sum())
                if sink_total > 0:
                    B = idx_big.size
                    net_local_ids = np.repeat(np.arange(B, dtype=np.int64), sink_lens)
                    cum_sink_starts = np.zeros(B + 1, dtype=np.int64)
                    np.cumsum(sink_lens, out=cum_sink_starts[1:])
                    offset_in_sinks = (
                        np.arange(sink_total, dtype=np.int64)
                        - np.repeat(cum_sink_starts[:-1], sink_lens)
                    )
                    global_pin_idx = (starts_big + 1)[net_local_ids] + offset_in_sinks
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
                            bucket_2_src_flat.append(src_gcells_big[nli_emit])
                            bucket_2_snk_flat.append(sg_uniq[mask_starlike])
                            bucket_2_w_arrs.append(net_weights[idx_big[nli_emit]])
                        if net_is_3.any():
                            cum_counts = np.cumsum(uniq_sink_counts)
                            net3_ids = np.where(net_is_3)[0]
                            ends = cum_counts[net3_ids]
                            bucket_3_g0.append(src_gcells_big[net3_ids])
                            bucket_3_g1.append(sg_uniq[ends - 2])
                            bucket_3_g2.append(sg_uniq[ends - 1])
                            bucket_3_w_arrs.append(net_weights[idx_big[net3_ids]])

            t5 = time.perf_counter()
            record("4_dispatch_len_big", (t5 - t4) * 1000)

            if bucket_2_src_flat:
                src_flat = np.concatenate(bucket_2_src_flat)
                snk_flat = np.concatenate(bucket_2_snk_flat)
                w_arr = np.concatenate(bucket_2_w_arrs)
                _apply_2pin_routing(
                    H_flat, V_flat,
                    src_flat // grid_col, src_flat % grid_col,
                    snk_flat // grid_col, snk_flat % grid_col,
                    w_arr, grid_row, grid_col,
                )

            t6 = time.perf_counter()
            record("5_apply_2pin", (t6 - t5) * 1000)

            if bucket_3_g0:
                g0_arr = np.concatenate(bucket_3_g0)
                g1_arr = np.concatenate(bucket_3_g1)
                g2_arr = np.concatenate(bucket_3_g2)
                w_arr3 = np.concatenate(bucket_3_w_arrs)
                _apply_3pin_routing_vec(H_flat, V_flat, g0_arr, g1_arr, g2_arr, w_arr3, grid_row, grid_col)

            t7 = time.perf_counter()
            record("6_apply_3pin", (t7 - t6) * 1000)

        n_hard = cache["n_hard"]
        if n_hard > 0:
            hard_indices_arr = cache.get("hard_indices_arr")
            if hard_indices_arr is None:
                hard_indices_arr = np.asarray(cache["hard_indices"], dtype=np.int64)
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

        t8 = time.perf_counter()
        record("7_apply_macro", (t8 - t7) * 1000)

        H_flat /= grid_h_routes
        V_flat /= grid_v_routes
        H_macro_flat /= grid_h_routes
        V_macro_flat /= grid_v_routes

        t9 = time.perf_counter()
        record("8_normalize", (t9 - t8) * 1000)

        smooth_range = int(plc.smooth_range)
        if smooth_range > 0:
            V_flat = _smooth_routing_cong_vec(V_flat, grid_row, grid_col, smooth_range, axis_h=False)
            H_flat = _smooth_routing_cong_vec(H_flat, grid_row, grid_col, smooth_range, axis_h=True)

        t10 = time.perf_counter()
        record("9_smooth", (t10 - t9) * 1000)

    return {k: np.mean(v) for k, v in timings.items()}


def main():
    for bname in ("ibm01", "ibm10", "ibm17"):
        print(f"\n=== {bname} ===", flush=True)
        bdir = Path("external/MacroPlacement/Testcases/ICCAD04") / bname
        bench, plc = load_benchmark_from_dir(bdir.as_posix())
        _patch_plc_wirelength(plc)
        _patch_plc_congestion(plc, bench)
        _patch_plc_density(plc, bench)
        # Initial set_placement to populate caches
        placement_np = bench.macro_positions.cpu().numpy().astype(np.float64)
        _fast_set_placement(plc, placement_np, bench)
        # Warmup: trigger initial routing build
        plc.get_congestion_cost()

        N = 30
        avg = profile_get_routing(plc, n_iters=N)
        total = sum(avg.values())
        print(f"  ({N} iterations, sorted by cost):", flush=True)
        for stage, ms in sorted(avg.items(), key=lambda x: -x[1]):
            pct = 100 * ms / total
            print(f"    {stage:25s}  {ms:6.3f} ms  ({pct:4.1f}%)", flush=True)
        print(f"    {'TOTAL':25s}  {total:6.3f} ms", flush=True)


if __name__ == "__main__":
    main()
