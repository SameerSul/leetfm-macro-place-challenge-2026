"""PlacementCost loader helpers."""

from pathlib import Path
from typing import Optional

from macro_place.benchmark import Benchmark

from placer.config import _log

def _load_plc(name: str, benchmark: Optional[Benchmark] = None):
    """Load PlacementCost for exact proxy scoring (posix paths for Windows compat).

    Caches the loaded plc on the benchmark object as `_cached_plc` so repeated
    place() calls on the same benchmark in dev iteration skip the ~1-3s load.
    """
    if benchmark is not None:
        cached = getattr(benchmark, "_cached_plc", None)
        if cached is not None:
            return cached
    try:
        from macro_place.loader import load_benchmark_from_dir, load_benchmark
        root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        plc = None
        if root.exists():
            _, plc = load_benchmark_from_dir(root.as_posix())
        else:
            # NG45 designs all share the leaf-directory name
            # "output_CT_Grouping" → benchmark.name doesn't disambiguate
            # them. Two-step lookup:
            #   1. Try the legacy "<design>_ng45" alias (kept for
            #      backward compat with older harnesses).
            #   2. If name is "output_CT_Grouping" or otherwise unmatched,
            #      iterate the 4 NG45 designs and pick the one whose
            #      plc matches `benchmark`'s canvas dimensions.
            ng45_aliases = {
                "ariane133_ng45": "ariane133",
                "ariane136_ng45": "ariane136",
                "nvdla_ng45": "nvdla",
                "mempool_tile_ng45": "mempool_tile",
            }
            ng45_base = Path("external/MacroPlacement/Flows/NanGate45")
            d = ng45_aliases.get(name)
            if d:
                base = ng45_base / d / "netlist" / "output_CT_Grouping"
                if (base / "netlist.pb.txt").exists():
                    _, plc = load_benchmark(
                        (base / "netlist.pb.txt").as_posix(),
                        (base / "initial.plc").as_posix(),
                    )
            elif benchmark is not None and name in (
                "output_CT_Grouping",
            ) and ng45_base.exists():
                # Disambiguate by canvas dimensions.
                bench_cw, bench_ch = benchmark.canvas_width, benchmark.canvas_height
                for design in ("ariane133", "ariane136", "nvdla", "mempool_tile"):
                    base = ng45_base / design / "netlist" / "output_CT_Grouping"
                    if not (base / "netlist.pb.txt").exists():
                        continue
                    try:
                        cand_bench, cand_plc = load_benchmark(
                            (base / "netlist.pb.txt").as_posix(),
                            (base / "initial.plc").as_posix(),
                        )
                        if (
                            abs(cand_bench.canvas_width - bench_cw) < 1e-6
                            and abs(cand_bench.canvas_height - bench_ch) < 1e-6
                        ):
                            plc = cand_plc
                            _log(f"  NG45 design matched: {design}")
                            break
                    except Exception:
                        continue
        if plc is not None and benchmark is not None:
            setattr(benchmark, "_cached_plc", plc)
        return plc
    except Exception as exc:
        _log(f"  Warning: plc load failed ({exc})")
    return None


