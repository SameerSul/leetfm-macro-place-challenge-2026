"""Diagnostic: does DREAMPlace's OWN legalizer (legalize_flag=1) produce a
competitive placement on ibm04?

Runs DREAMPlace with several legalize-enabled config variants, then scores
DP's output DIRECTLY — no greedy spiral, no other Phase 1/2/3 perturbation.
This tells us whether the path to Fix 3 (DP as primary) is viable: if DP's
own legalized placement scores below baseline 1.4101 on its own merits, then
restructuring placer.py to use DP as primary makes sense. If even with its
own legalizer DP scores ~1.77, the restructure won't help.

Note: DREAMPlace writes the legalized result to `<design>.lg.pl` when
legalize_flag=1, not the unlegalized `<design>.gp.pl`. We read whichever
exists.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
V1_DIR = HERE.parent
for p in (str(REPO_ROOT), str(V1_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from macro_place import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from dreamplace_bridge import run_bridge  # noqa: E402
from dreamplace_bridge.run_bridge import run_dreamplace  # noqa: E402
from dreamplace_bridge.bookshelf_to_pb import read_dreamplace_positions_full  # noqa: E402

from placer import _will_legalize  # noqa: E402


def _make_factory(overrides: dict):
    original = run_bridge._default_dreamplace_config

    def patched(*args, **kwargs):
        cfg = original(*args, **kwargs)
        for k, v in overrides.items():
            cfg[k] = v
        return cfg

    return patched


def _score(benchmark, plc, work_dir, design, sizes, hw, hh, cw, ch, movable, n,
           apply_greedy_legalize: bool, lg_pl_first: bool):
    """Read DP positions and score. Returns (proxy, what_was_read)."""
    # If legalize_flag=1, DREAMPlace writes <design>.lg.pl in results/
    # Try lg.pl first, fall back to gp.pl
    results_dir = work_dir / "results" / design
    lg_pl = results_dir / f"{design}.lg.pl"
    gp_pl = results_dir / f"{design}.gp.pl"
    output_pl = None
    read_label = ""
    if lg_pl_first and lg_pl.exists():
        output_pl = str(lg_pl)
        read_label = "lg.pl"
    elif gp_pl.exists():
        output_pl = str(gp_pl)
        read_label = "gp.pl"
    else:
        return float("nan"), "(no output)"

    hard_pos, soft_pos = read_dreamplace_positions_full(
        plc, str(work_dir), design, output_pl=output_pl
    )
    if apply_greedy_legalize:
        hard_pos = _will_legalize(hard_pos, movable, sizes, hw, hh, cw, ch, n)
        read_label += "+greedy"

    dp_pl = benchmark.macro_positions.clone()
    dp_pl[:n, 0] = torch.tensor(hard_pos[:, 0], dtype=torch.float32)
    dp_pl[:n, 1] = torch.tensor(hard_pos[:, 1], dtype=torch.float32)
    n_soft = int(min(soft_pos.shape[0], benchmark.num_soft_macros))
    if n_soft > 0:
        dp_pl[n:n + n_soft, 0] = torch.tensor(soft_pos[:n_soft, 0], dtype=torch.float32)
        dp_pl[n:n + n_soft, 1] = torch.tensor(soft_pos[:n_soft, 1], dtype=torch.float32)

    costs = compute_proxy_cost(dp_pl, benchmark, plc)
    return float(costs["proxy_cost"]), read_label


def main():
    bench_name = "ibm04"
    bench_dir = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench_name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    n = benchmark.num_hard_macros
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n].numpy()

    print(f"\nbenchmark={bench_name}, n={n}, canvas={cw:.1f}x{ch:.1f}")
    print(f"baseline ibm04 = 1.4101, Phase 3 win = 1.3316\n")

    variants = [
        ("baseline-Fix2 (current, gp.pl, greedy legalize)",
         {}, True, False),
        ("Fix3-A: legalize_flag=1, read lg.pl directly (no greedy)",
         {"legalize_flag": 1}, False, True),
        ("Fix3-B: legalize_flag=1 + enable_fillers=1, lg.pl direct",
         {"legalize_flag": 1, "enable_fillers": 1}, False, True),
        ("Fix3-C: legalize_flag=1 + iterations=300, lg.pl direct",
         {"legalize_flag": 1,
          "global_place_stages": [{
              "num_bins_x": 64, "num_bins_y": 64,
              "iteration": 300, "learning_rate": 0.01,
              "wirelength": "weighted_average", "optimizer": "nesterov",
              "Llambda_density_weight_iteration": 1, "Lsub_iteration": 1,
          }]}, False, True),
    ]

    original = run_bridge._default_dreamplace_config
    results = []

    try:
        for label, overrides, apply_greedy, lg_first in variants:
            print(f"--- {label} ---")
            run_bridge._default_dreamplace_config = _make_factory(overrides)
            scratch = Path(f"/tmp/dp_probe/{label.split(':')[0].replace(' ', '_')}")
            if scratch.exists():
                import shutil
                shutil.rmtree(scratch)
            t = time.time()
            try:
                run_dreamplace(
                    str(bench_dir), plc=plc,
                    scratch_root=str(scratch),
                    timeout_s=180.0,
                    iterations=150,  # overridden by global_place_stages if present
                    num_threads=4,
                    soft_macros_movable=True,
                )
                work_dir = scratch / bench_name
                proxy, read = _score(
                    benchmark, plc, work_dir, bench_name,
                    sizes, hw, hh, cw, ch, movable, n,
                    apply_greedy_legalize=apply_greedy,
                    lg_pl_first=lg_first,
                )
            except Exception as e:
                proxy = float("nan")
                read = f"FAIL: {type(e).__name__}: {e}"
            secs = time.time() - t
            results.append((label, proxy, secs, read))
            print(f"  proxy={proxy:.4f}  ({secs:.1f}s, read={read})\n")
    finally:
        run_bridge._default_dreamplace_config = original

    print("\n=== Summary (lower proxy is better) ===")
    print(f"  ibm04 baseline = 1.4101  |  Phase 3 win = 1.3316")
    print(f"  {'variant':<60} {'proxy':>8} {'sec':>6}")
    for label, proxy, secs, _ in results:
        print(f"  {label:<60} {proxy:8.4f} {secs:6.1f}")


if __name__ == "__main__":
    main()
