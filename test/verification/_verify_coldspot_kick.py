"""Verify the coldspot cluster kick produces VALID (overlap-free, in-bounds) output.

Builds a real congestion field, then calls `_coldspot_cluster_kick` across many
seeds and asserts the legalized hard placement has zero hard-macro overlaps and is
in-bounds, and that co-moved softs stay in-bounds.

    uv run python test/verification/_verify_coldspot_kick.py [ibm04 ibm10]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from macro_place.loader import load_benchmark_from_dir
from placer.local_search.clusters import derive_cluster_softs, derive_hard_clusters
from placer.local_search.fields import _congestion_field
from placer.local_search.lsmc_explore import (
    _coldspot_cluster_kick,
    _coldspot_cluster_kick_candidates,
)
from placer.scoring.exact import _exact_proxy
from utils import constants as const

TOL = 0.05


def _check(bench):
    src = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(src))
    n, n_soft = benchmark.num_hard_macros, benchmark.num_soft_macros
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    hw, hh = sizes[:n, 0] / 2, sizes[:n, 1] / 2
    soft_hw, soft_hh = sizes[n : n + n_soft, 0] / 2, sizes[n : n + n_soft, 1] / 2
    movable = benchmark.get_movable_mask().numpy()
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    nr, nc = int(benchmark.grid_rows), int(benchmark.grid_cols)

    pos = benchmark.macro_positions.numpy().astype(np.float64)
    float(_exact_proxy(torch.tensor(pos, dtype=torch.float32), benchmark, plc))
    field = _congestion_field(plc, nr, nc)
    assert field is not None, "no congestion field"

    labels, clusters = derive_hard_clusters(plc, n, n_soft=n_soft, min_edge=2)
    csofts = derive_cluster_softs(plc, n, n_soft, labels)
    hard_xy = pos[:n]
    soft_xy = pos[n : n + n_soft]
    mv = np.flatnonzero(movable[:n])

    kicks = 0
    pools = 0
    partial_pools = 0
    for s in range(25):
        rng = np.random.default_rng(s)
        for pk in ("hot", "random"):
            res = _coldspot_cluster_kick(
                hard_xy,
                sizes[:n],
                hw,
                hh,
                cw,
                ch,
                movable[:n],
                n,
                clusters,
                csofts,
                soft_xy,
                soft_hw,
                soft_hh,
                movable[n : n + n_soft],
                field,
                nr,
                nc,
                rng,
                deadline=float("inf"),
                pick=pk,
            )
            if res is None:
                continue
            kicks += 1
            out, out_soft = res
            assert np.all(out[:, 0] >= hw - TOL) and np.all(
                out[:, 0] <= cw - hw + TOL
            ), "hard x oob"
            assert np.all(out[:, 1] >= hh - TOL) and np.all(
                out[:, 1] <= ch - hh + TOL
            ), "hard y oob"
            p = out[mv]
            sx = np.abs(p[:, None, 0] - p[None, :, 0]) + TOL >= (hw[mv][:, None] + hw[mv][None, :])
            sy = np.abs(p[:, None, 1] - p[None, :, 1]) + TOL >= (hh[mv][:, None] + hh[mv][None, :])
            ok = sx | sy
            np.fill_diagonal(ok, True)
            assert ok.all(), f"{pk} seed{s}: hard overlap"
            if out_soft is not None:
                moved = np.any(np.abs(out_soft - soft_xy) > TOL, axis=1)
                m = out_soft[moved]
                assert np.all(m[:, 0] >= soft_hw[moved] - TOL) and np.all(
                    m[:, 0] <= cw - soft_hw[moved] + TOL
                ), "soft x oob"
                assert np.all(m[:, 1] >= soft_hh[moved] - TOL) and np.all(
                    m[:, 1] <= ch - soft_hh[moved] + TOL
                ), "soft y oob"

    for s, pk in ((0, "hot"), (1, "random")):
        pool = _coldspot_cluster_kick_candidates(
            hard_xy,
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            movable[:n],
            n,
            clusters,
            csofts,
            soft_xy,
            soft_hw,
            soft_hh,
            movable[n : n + n_soft],
            field,
            nr,
            nc,
            np.random.default_rng(s),
            deadline=float("inf"),
            pick=pk,
            kick_count=2,
        )
        if not pool:
            continue
        pools += 1
        cluster_ids = {int(trace["cluster"]) for _, _, trace in pool}
        anchors = {
            (round(float(trace["anchor_x"]), 6), round(float(trace["anchor_y"]), 6))
            for _, _, trace in pool
        }
        assert len(cluster_ids) == 1, "candidate pool changed selected cluster"
        assert len(anchors) == 1, "candidate pool changed selected coldspot anchor"
        assert len(pool) <= 2, "candidate pool exceeded requested kick count"
        for out, out_soft, trace in pool:
            assert "cluster_bbox_before" in trace and "cluster_bbox_after" in trace
            assert "hard_disp_mean" in trace and "target_field" in trace
            assert np.all(out[:, 0] >= hw - TOL) and np.all(
                out[:, 0] <= cw - hw + TOL
            ), "pool hard x oob"
            assert np.all(out[:, 1] >= hh - TOL) and np.all(
                out[:, 1] <= ch - hh + TOL
            ), "pool hard y oob"
            p = out[mv]
            sx = np.abs(p[:, None, 0] - p[None, :, 0]) + TOL >= (hw[mv][:, None] + hw[mv][None, :])
            sy = np.abs(p[:, None, 1] - p[None, :, 1]) + TOL >= (hh[mv][:, None] + hh[mv][None, :])
            ok = sx | sy
            np.fill_diagonal(ok, True)
            assert ok.all(), f"{pk} seed{s}: pool hard overlap"
            if out_soft is not None:
                moved = np.any(np.abs(out_soft - soft_xy) > TOL, axis=1)
                m = out_soft[moved]
                assert np.all(m[:, 0] >= soft_hw[moved] - TOL) and np.all(
                    m[:, 0] <= cw - soft_hw[moved] + TOL
                ), "pool soft x oob"
                assert np.all(m[:, 1] >= soft_hh[moved] - TOL) and np.all(
                    m[:, 1] <= ch - soft_hh[moved] + TOL
                ), "pool soft y oob"

    old_partial = bool(const.HIER_COLDSPOT_PARTIAL_FRONTIER)
    old_min_cluster = int(const.HIER_COLDSPOT_PARTIAL_MIN_CLUSTER_HARD)
    old_min_remaining = int(const.HIER_COLDSPOT_PARTIAL_MIN_REMAINING_HARD)
    old_max_member_frac = float(const.HIER_COLDSPOT_PARTIAL_MAX_MEMBER_FRAC)
    old_require_connected = bool(const.HIER_COLDSPOT_PARTIAL_REQUIRE_CONNECTED)
    old_radius_ratio = float(const.HIER_COLDSPOT_PARTIAL_MAX_RADIUS_RATIO)
    old_bbox_ratio = float(const.HIER_COLDSPOT_PARTIAL_MAX_BBOX_RATIO)
    old_sep_ratio = float(const.HIER_COLDSPOT_PARTIAL_MAX_SEPARATION_RATIO)
    const.HIER_COLDSPOT_PARTIAL_FRONTIER = True
    const.HIER_COLDSPOT_PARTIAL_MIN_CLUSTER_HARD = 2
    const.HIER_COLDSPOT_PARTIAL_MIN_REMAINING_HARD = 1
    const.HIER_COLDSPOT_PARTIAL_MAX_MEMBER_FRAC = 1.0
    const.HIER_COLDSPOT_PARTIAL_REQUIRE_CONNECTED = False
    const.HIER_COLDSPOT_PARTIAL_MAX_RADIUS_RATIO = 1.0e9
    const.HIER_COLDSPOT_PARTIAL_MAX_BBOX_RATIO = 1.0e9
    const.HIER_COLDSPOT_PARTIAL_MAX_SEPARATION_RATIO = 1.0e9
    try:
        partial_pool = _coldspot_cluster_kick_candidates(
            hard_xy,
            sizes[:n],
            hw,
            hh,
            cw,
            ch,
            movable[:n],
            n,
            clusters,
            csofts,
            soft_xy,
            soft_hw,
            soft_hh,
            movable[n : n + n_soft],
            field,
            nr,
            nc,
            np.random.default_rng(2),
            deadline=float("inf"),
            pick="random",
            kick_count=2,
            plc=plc,
        )
    finally:
        const.HIER_COLDSPOT_PARTIAL_FRONTIER = old_partial
        const.HIER_COLDSPOT_PARTIAL_MIN_CLUSTER_HARD = old_min_cluster
        const.HIER_COLDSPOT_PARTIAL_MIN_REMAINING_HARD = old_min_remaining
        const.HIER_COLDSPOT_PARTIAL_MAX_MEMBER_FRAC = old_max_member_frac
        const.HIER_COLDSPOT_PARTIAL_REQUIRE_CONNECTED = old_require_connected
        const.HIER_COLDSPOT_PARTIAL_MAX_RADIUS_RATIO = old_radius_ratio
        const.HIER_COLDSPOT_PARTIAL_MAX_BBOX_RATIO = old_bbox_ratio
        const.HIER_COLDSPOT_PARTIAL_MAX_SEPARATION_RATIO = old_sep_ratio
    partial = [
        (out, out_soft, trace)
        for out, out_soft, trace in partial_pool
        if trace.get("partial_frontier")
    ]
    if partial:
        partial_pools += 1
        assert (
            len(partial_pool) <= 3
        ), "partial pool exceeded whole candidates plus partial candidate"
        for out, out_soft, trace in partial:
            assert trace["partial_moved_hard"] >= const.HIER_COLDSPOT_PARTIAL_MIN_HARD
            assert trace["partial_moved_hard"] < trace["member_count"]
            assert trace["partial_capacity"] > 0.0
            assert trace["partial_pred_radius_ratio"] > 0.0
            assert trace["partial_pred_bbox_ratio"] > 0.0
            assert trace["partial_pred_separation_ratio"] >= 0.0
            assert np.all(out[:, 0] >= hw - TOL) and np.all(
                out[:, 0] <= cw - hw + TOL
            ), "partial hard x oob"
            assert np.all(out[:, 1] >= hh - TOL) and np.all(
                out[:, 1] <= ch - hh + TOL
            ), "partial hard y oob"
            p = out[mv]
            sx = np.abs(p[:, None, 0] - p[None, :, 0]) + TOL >= (hw[mv][:, None] + hw[mv][None, :])
            sy = np.abs(p[:, None, 1] - p[None, :, 1]) + TOL >= (hh[mv][:, None] + hh[mv][None, :])
            ok = sx | sy
            np.fill_diagonal(ok, True)
            assert ok.all(), "partial hard overlap"
            if out_soft is not None:
                moved = np.any(np.abs(out_soft - soft_xy) > TOL, axis=1)
                m = out_soft[moved]
                assert np.all(m[:, 0] >= soft_hw[moved] - TOL) and np.all(
                    m[:, 0] <= cw - soft_hw[moved] + TOL
                ), "partial soft x oob"
                assert np.all(m[:, 1] >= soft_hh[moved] - TOL) and np.all(
                    m[:, 1] <= ch - soft_hh[moved] + TOL
                ), "partial soft y oob"
    assert kicks > 0, "no kicks fired (no usable cluster?)"
    assert pools > 0, "no candidate pools fired"
    assert partial_pools > 0, "partial frontier candidate did not fire"
    print(
        f"{bench}: coldspot-kick OK  {len(clusters)} clusters, "
        f"{kicks} kicks, {pools} pools, {partial_pools} partial pools, "
        "0 overlaps, in-bounds"
    )


if __name__ == "__main__":
    for b in sys.argv[1:] or ["ibm04", "ibm10"]:
        _check(b)
    print("COLDSPOT-KICK VERIFICATION PASSED")
