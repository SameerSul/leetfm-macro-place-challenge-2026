"""PlacementCost loader helpers."""

from pathlib import Path
from typing import Optional

from macro_place.benchmark import Benchmark

from utils.config import _log


def _trace_benchmark_name(name: str, benchmark_dir: Optional[Path]) -> str:
    """Return a stable design name for telemetry and calibration reports."""
    raw = str(name)
    if raw != "output_CT_Grouping" or benchmark_dir is None:
        return raw
    path = Path(benchmark_dir)
    if path.name == "output_CT_Grouping" and path.parent.name == "netlist":
        return path.parent.parent.name
    return raw


def _resolve_benchmark_dir(name: str, benchmark: Optional[Benchmark] = None) -> Optional[Path]:
    """Resolve the benchmark source directory for DREAMPlace/PlacementCost loading.

    Supported layouts:
    - legacy ICCAD04: ``external/MacroPlacement/Testcases/ICCAD04/<name>``
    - generated EDA/challenge inputs attached as ``benchmark._source_dir``
    - NG45 aliases: ``<design>_ng45`` (e.g., ``ariane133_ng45``)
    - generated/loaded NG45 output directories via exact canvas-size match
    """
    iccad = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if iccad.exists():
        return iccad

    if benchmark is not None:
        source_dir = getattr(benchmark, "_source_dir", None)
        if source_dir is not None:
            source_dir = Path(source_dir)
            if (source_dir / "netlist.pb.txt").exists():
                return source_dir

    ng45_base = Path("external/MacroPlacement/Flows/NanGate45")
    if not ng45_base.exists():
        return None

    ng45_aliases = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
        "bp_quad_ng45": "bp_quad",
    }
    if name in ng45_aliases:
        base = ng45_base / ng45_aliases[name] / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            return base

    if "_ng45" in name:
        base_name = name.split("_ng45", 1)[0]
        base = ng45_base / base_name / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            return base

    if benchmark is not None and name in ("output_CT_Grouping",):
        bench_cw, bench_ch = benchmark.canvas_width, benchmark.canvas_height
        bench_n = int(benchmark.num_macros) if hasattr(benchmark, "num_macros") else None
        for design in ("ariane133", "ariane136", "nvdla", "mempool_tile", "bp_quad"):
            base = ng45_base / design / "netlist" / "output_CT_Grouping"
            if not (base / "netlist.pb.txt").exists():
                continue
            try:
                from macro_place.loader import load_benchmark  # local import to avoid init loops

                cand_bench, cand_plc = load_benchmark(
                    (base / "netlist.pb.txt").as_posix(),
                    (base / "initial.plc").as_posix(),
                )
                if (
                    abs(cand_bench.canvas_width - bench_cw) < 1e-6
                    and abs(cand_bench.canvas_height - bench_ch) < 1e-6
                    and (bench_n is None or int(cand_bench.num_macros) == bench_n)
                ):
                    if hasattr(cand_plc, "modules_w_pins"):
                        _log(f"  NG45 design matched by size: {design}")
                    return base
            except Exception:
                continue

    return None


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

        root = _resolve_benchmark_dir(name, benchmark)
        plc = None
        if root is not None:
            if root == (Path("external/MacroPlacement/Testcases/ICCAD04") / name):
                _, plc = load_benchmark_from_dir(root.as_posix())
            else:
                plc = load_benchmark(
                    (root / "netlist.pb.txt").as_posix(),
                    (
                        (root / "initial.plc").as_posix()
                        if (root / "initial.plc").exists()
                        else None
                    ),
                )[1]
        if plc is not None and benchmark is not None:
            setattr(benchmark, "_cached_plc", plc)
        return plc
    except Exception as exc:
        _log(f"  Warning: plc load failed ({exc})")
    return None
