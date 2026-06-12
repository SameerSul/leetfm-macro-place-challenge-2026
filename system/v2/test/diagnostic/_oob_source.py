"""Diagnose out-of-bounds macros in the placer's output: hard vs soft, edge, overhang.

Usage: uv run python system/v2/test/diagnostic/_oob_source.py ibm10 ibm09
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from macro_place.evaluate import _load_placer  # noqa: E402
from macro_place.loader import load_benchmark  # noqa: E402


def main():
    names = sys.argv[1:] or ["ibm10"]
    placer = _load_placer(ROOT / "system/v2/src/main.py")
    for name in names:
        d = ROOT / "external/MacroPlacement/Testcases/ICCAD04" / name
        benchmark, plc = load_benchmark(str(d / "netlist.pb.txt"), str(d / "initial.plc"))
        benchmark._cached_plc = plc
        pl = placer.place(benchmark)
        n_hard = benchmark.num_hard_macros
        half = benchmark.macro_sizes / 2
        lo = pl - half
        hi = pl + half
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        over = torch.zeros(pl.shape[0])
        over = torch.maximum(over, -lo[:, 0])
        over = torch.maximum(over, -lo[:, 1])
        over = torch.maximum(over, hi[:, 0] - cw)
        over = torch.maximum(over, hi[:, 1] - ch)
        oob = torch.where(over > 1e-6)[0].tolist()
        print(f"\n=== {name}: {len(oob)} OOB (canvas {cw:.1f}x{ch:.1f}, n_hard={n_hard}) ===")
        n_hard_oob = sum(1 for i in oob if i < n_hard)
        print(f"  hard OOB: {n_hard_oob}   soft OOB: {len(oob) - n_hard_oob}")
        init = benchmark.macro_positions
        mov = benchmark.get_movable_mask() if hasattr(benchmark, "get_movable_mask") else None
        fix = benchmark.macro_fixed if hasattr(benchmark, "macro_fixed") else None
        for i in sorted(oob, key=lambda k: -over[k])[:8]:
            kind = "HARD" if i < n_hard else "soft"
            edges = []
            if lo[i, 0] < -1e-6: edges.append(f"L{-lo[i,0]:.3f}")
            if lo[i, 1] < -1e-6: edges.append(f"B{-lo[i,1]:.3f}")
            if hi[i, 0] - cw > 1e-6: edges.append(f"R{hi[i,0]-cw:.3f}")
            if hi[i, 1] - ch > 1e-6: edges.append(f"T{hi[i,1]-ch:.3f}")
            init_oob = (init[i, 0] - half[i, 0] < -1e-6 or init[i, 1] - half[i, 1] < -1e-6
                        or init[i, 0] + half[i, 0] > cw + 1e-6 or init[i, 1] + half[i, 1] > ch + 1e-6)
            mflag = bool(mov[i]) if mov is not None else "?"
            fflag = bool(fix[i]) if fix is not None else "?"
            print(f"  [{kind} #{i}] out=({pl[i,0]:.2f},{pl[i,1]:.2f}) "
                  f"init=({init[i,0]:.2f},{init[i,1]:.2f}) init_oob={init_oob} "
                  f"size=({benchmark.macro_sizes[i,0]:.2f}x{benchmark.macro_sizes[i,1]:.2f}) "
                  f"overhang={over[i]:.3f} edges={edges} movable={mflag} fixed={fflag}")


if __name__ == "__main__":
    main()
