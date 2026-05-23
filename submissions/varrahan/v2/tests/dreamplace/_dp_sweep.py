"""DREAMPlace density-parameter sweep on a single benchmark.

Loops over (target_density, density_weight) combos, runs DREAMPlace standalone
(no surrounding Phase 1/2/3 pipeline), legalizes hard macros, scores with the
exact proxy. Outputs a sorted table of (params -> proxy).

Usage:
    uv run python submissions/varrahan/v1/_dp_sweep.py --benchmark ibm04

Lives under submissions/varrahan/v1/ because that's the only writable area;
delete after the sweep is done.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[5]
V2_DIR = HERE.parents[2]

for p in (str(REPO_ROOT), str(V2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from macro_place import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from dreamplace_bridge import run_bridge  # noqa: E402
from dreamplace_bridge.run_bridge import run_dreamplace  # noqa: E402
from dreamplace_bridge.bookshelf_to_pb import read_dreamplace_positions_full  # noqa: E402

# Reuse the production placer's legalizer
from placer import _will_legalize  # noqa: E402


def _make_config_factory(target_density: float, density_weight: float):
    """Return a drop-in replacement for _default_dreamplace_config that
    overrides target_density / density_weight while preserving every other
    setting."""
    original = run_bridge._default_dreamplace_config

    def patched(*args, **kwargs):
        cfg = original(*args, **kwargs)
        cfg["target_density"] = target_density
        cfg["density_weight"] = density_weight
        return cfg

    return patched


def _score_dreamplace_output(
    benchmark, plc, work_dir: Path, design: str,
    sizes: np.ndarray, hw: np.ndarray, hh: np.ndarray,
    cw: float, ch: float, movable: np.ndarray, n: int,
) -> float:
    """Read DREAMPlace's .gp.pl output, legalize hard, score exact proxy."""
    hard_pos, soft_pos = read_dreamplace_positions_full(plc, str(work_dir), design)
    hard_leg = _will_legalize(hard_pos, movable, sizes, hw, hh, cw, ch, n)

    dp_pl = benchmark.macro_positions.clone()
    dp_pl[:n, 0] = torch.tensor(hard_leg[:, 0], dtype=torch.float32)
    dp_pl[:n, 1] = torch.tensor(hard_leg[:, 1], dtype=torch.float32)
    n_soft = int(min(soft_pos.shape[0], benchmark.num_soft_macros))
    if n_soft > 0:
        dp_pl[n:n + n_soft, 0] = torch.tensor(soft_pos[:n_soft, 0], dtype=torch.float32)
        dp_pl[n:n + n_soft, 1] = torch.tensor(soft_pos[:n_soft, 1], dtype=torch.float32)

    costs = compute_proxy_cost(dp_pl, benchmark, plc)
    return float(costs["proxy_cost"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="ibm04")
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--scratch", default="/tmp/dp_sweep")
    args = ap.parse_args()

    bench_dir = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / args.benchmark
    if not bench_dir.exists():
        raise SystemExit(f"benchmark dir not found: {bench_dir}")

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    n = benchmark.num_hard_macros
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n].numpy()

    print(f"\nbenchmark={args.benchmark}, n={n}, canvas={cw:.1f}x{ch:.1f}")
    print(f"iterations={args.iterations}, threads={args.threads}\n")

    # The 30-combo grid from DREAMPLACE_FIXES.md
    target_densities = [0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
    density_weights = [8e-5, 5e-4, 1e-3, 5e-3, 1e-2]

    results = []  # list of (target_density, density_weight, proxy, secs)
    original_default = run_bridge._default_dreamplace_config

    t_sweep_start = time.time()
    total = len(target_densities) * len(density_weights)
    combo_idx = 0

    try:
        for td in target_densities:
            for dw in density_weights:
                combo_idx += 1
                t_combo = time.time()
                scratch = Path(args.scratch) / f"td{td}_dw{dw}"
                # Clear per-combo scratch so DREAMPlace writes fresh files
                if scratch.exists():
                    import shutil
                    shutil.rmtree(scratch)

                # Monkey-patch the default config builder for this combo.
                run_bridge._default_dreamplace_config = _make_config_factory(td, dw)
                try:
                    run_dreamplace(
                        str(bench_dir),
                        plc=plc,
                        scratch_root=str(scratch),
                        timeout_s=120.0,
                        iterations=args.iterations,
                        num_threads=args.threads,
                        soft_macros_movable=True,
                    )
                    work_dir = scratch / args.benchmark
                    proxy = _score_dreamplace_output(
                        benchmark, plc, work_dir, args.benchmark,
                        sizes, hw, hh, cw, ch, movable, n,
                    )
                except Exception as e:
                    proxy = float("nan")
                    print(f"  [{combo_idx}/{total}] td={td:.2f} dw={dw:.0e}: FAILED ({type(e).__name__}: {e})")
                    continue
                finally:
                    run_bridge._default_dreamplace_config = original_default

                secs = time.time() - t_combo
                results.append((td, dw, proxy, secs))
                print(f"  [{combo_idx}/{total}] td={td:.2f} dw={dw:.0e}: proxy={proxy:.4f}  ({secs:.1f}s)")
    finally:
        run_bridge._default_dreamplace_config = original_default

    print(f"\nTotal sweep time: {time.time() - t_sweep_start:.1f}s")
    print(f"\nSorted by proxy (best first):")
    print(f"  {'td':>6}  {'dw':>10}  {'proxy':>8}  {'sec':>6}")
    for td, dw, proxy, secs in sorted(results, key=lambda r: r[2]):
        print(f"  {td:6.2f}  {dw:10.0e}  {proxy:8.4f}  {secs:6.1f}")


if __name__ == "__main__":
    main()
