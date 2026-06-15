"""Verify cluster derivation determinism + cluster-kick legality.

Checks, on a few IBM benchmarks:
  1. derive_hard_clusters is deterministic and cached (same object on re-call).
  2. members are valid hard-macro indices, clusters have >= 2 members, disjoint.
  3. _cluster_kick (gather/translate/both) returns an in-bounds, hard-overlap-free
     placement (legalized), for many RNG seeds.

    uv run python test/verification/_verify_cluster_kick.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "system" / "v2" / "src"))

import numpy as np

from macro_place.loader import load_benchmark_from_dir
from placer.local_search.clusters import derive_cluster_softs, derive_hard_clusters
from placer.local_search.lsmc_explore import _cluster_kick

BENCHES = ["ibm01", "ibm04", "ibm10"]
TOL = 1e-6


def _check(bench):
    src = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(src))
    n = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)[:n]
    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0
    cw, ch = (float(v) for v in plc.get_canvas_width_height())
    movable = benchmark.get_movable_mask().detach().cpu().numpy()[:n]
    hard_xy = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)[:n]

    # 1. determinism + cache.
    labels_a, clusters_a = derive_hard_clusters(plc, n, n_soft=n_soft)
    labels_b, clusters_b = derive_hard_clusters(plc, n, n_soft=n_soft)
    assert clusters_a is clusters_b, "not cached"
    assert np.array_equal(labels_a, labels_b), "labels not deterministic"

    # 2. structural validity + disjointness.
    seen = set()
    for cid, members in clusters_a.items():
        assert members.size >= 2, f"cluster {cid} too small"
        assert members.min() >= 0 and members.max() < n, "member out of range"
        ms = set(members.tolist())
        assert not (ms & seen), "clusters overlap"
        seen |= ms

    # Soft co-move inputs.
    soft_sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)[n:n + n_soft]
    soft_hw = soft_sizes[:, 0] / 2.0
    soft_hh = soft_sizes[:, 1] / 2.0
    soft_xy = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)[n:n + n_soft]
    cluster_softs = derive_cluster_softs(plc, n, n_soft, labels_a)

    # 3. kick legality across modes + seeds (with soft co-move).
    for mode in ("gather", "translate", "both"):
        for s in range(20):
            rng = np.random.default_rng(s)
            res = _cluster_kick(hard_xy, sizes, hw, hh, cw, ch, movable, n,
                                clusters_a, rng, deadline=float("inf"), mode=mode,
                                soft_xy=soft_xy, soft_hw=soft_hw, soft_hh=soft_hh,
                                cluster_softs=cluster_softs)
            if res is None:
                continue
            out, out_soft = res
            assert np.all(out[:, 0] >= hw - TOL) and np.all(out[:, 0] <= cw - hw + TOL), \
                f"{mode} x out of bounds"
            assert np.all(out[:, 1] >= hh - TOL) and np.all(out[:, 1] <= ch - hh + TOL), \
                f"{mode} y out of bounds"
            # No hard-macro overlap (legalized): pairwise separation on movable.
            mv = np.flatnonzero(movable)
            p = out[mv]
            sep_x = np.abs(p[:, None, 0] - p[None, :, 0]) + TOL >= (hw[mv][:, None] + hw[mv][None, :])
            sep_y = np.abs(p[:, None, 1] - p[None, :, 1]) + TOL >= (hh[mv][:, None] + hh[mv][None, :])
            ok = sep_x | sep_y
            np.fill_diagonal(ok, True)
            assert ok.all(), f"{mode} produced overlapping hard macros (seed {s})"
            if out_soft is not None:
                # Only co-moved softs (changed vs incumbent) must be in bounds;
                # untouched softs keep their incumbent centers verbatim.
                moved = np.any(np.abs(out_soft - soft_xy) > TOL, axis=1)
                m = out_soft[moved]
                assert np.all(m[:, 0] >= soft_hw[moved] - TOL) and \
                    np.all(m[:, 0] <= cw - soft_hw[moved] + TOL), f"{mode} soft x oob"
                assert np.all(m[:, 1] >= soft_hh[moved] - TOL) and \
                    np.all(m[:, 1] <= ch - soft_hh[moved] + TOL), f"{mode} soft y oob"
                assert np.allclose(out_soft[~moved], soft_xy[~moved]), \
                    f"{mode} mutated untouched softs"

    print(f"{bench}: OK  clusters={len(clusters_a)} clustered={int((labels_a >= 0).sum())}/{n} "
          f"cluster_softs={sum(len(v) for v in cluster_softs.values())}")


if __name__ == "__main__":
    for b in BENCHES:
        _check(b)
    print("ALL CLUSTER-KICK CHECKS PASSED")
