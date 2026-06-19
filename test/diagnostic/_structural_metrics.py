"""Report deterministic structural metrics for a benchmark placement.

Usage:
    uv run python test/diagnostic/_structural_metrics.py ibm10
    RUN_PLACER=0 uv run python test/diagnostic/_structural_metrics.py ibm10
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from placer.local_search.structural_fields import (  # noqa: E402
    combined_structural_penalty,
    structural_penalty_components,
)
from placer.pipeline.macro_placer import MacroPlacer  # noqa: E402
from placer.scoring.exact import _exact_proxy  # noqa: E402


def _placement_for(benchmark):
    if os.environ.get("RUN_PLACER", "1").strip() in {"0", "false", "False", "no", "NO"}:
        return benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
    placer = MacroPlacer()
    with torch.inference_mode():
        return placer.place(benchmark).detach().cpu().numpy().astype(np.float64)


def main() -> None:
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    src = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(src))
    placement = _placement_for(benchmark)
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)

    comp = structural_penalty_components(
        placement,
        sizes,
        cw,
        ch,
        grid_cols=int(benchmark.grid_cols),
        grid_rows=int(benchmark.grid_rows),
    )
    total = combined_structural_penalty(
        placement,
        sizes,
        cw,
        ch,
        grid_cols=int(benchmark.grid_cols),
        grid_rows=int(benchmark.grid_rows),
    )
    proxy = float(_exact_proxy(torch.tensor(placement, dtype=torch.float32), benchmark, plc))

    print(f"{bench}: proxy={proxy:.4f} structural={total:.6f}")
    for key in ("edge_keepout", "grid_alignment", "notch"):
        print(f"  {key:16s} {comp[key]:.6f}")


if __name__ == "__main__":
    main()
