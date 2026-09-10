"""Record full-precision evaluator results and placements for isolated trials.

Example: HIER_DIAGNOSTIC_NO_DEADLINES=1 uv run python
test/diagnostic/run_proxy_comparison.py --out ml_data/trial ibm04 ibm09
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from macro_place.evaluate import IBM_BENCHMARKS, NG45_BENCHMARKS, evaluate_benchmark
from main import MacroPlacer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", type=Path, help="stop at the first proxy regression")
    parser.add_argument("names", nargs="+", choices=[*IBM_BENCHMARKS, *NG45_BENCHMARKS])
    args = parser.parse_args()
    control = (
        {row["benchmark"]: row for row in json.loads(args.control.read_text())}
        if args.control else {}
    )
    if args.control:
        assert set(args.names) <= control.keys(), "control is missing a requested benchmark"
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["HIER_PLATEAU_TRACE_PATH"] = str(args.out / "trace.jsonl")
    placer = MacroPlacer()
    rows = []
    for name in args.names:
        result = evaluate_benchmark(
            placer, name, str(ROOT / "external/MacroPlacement/Testcases/ICCAD04"),
            ng45_dir=NG45_BENCHMARKS.get(name),
        )
        row = {"benchmark": name}
        for key in ("proxy_cost", "wirelength", "density", "congestion", "overlaps", "runtime"):
            if key in result:
                row[key] = float(result[key])
        row["valid"] = bool(result["valid"])
        rows.append(row)
        np.save(args.out / f"{name}.npy", result["placement"].detach().cpu().numpy())
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("RESULT " + json.dumps(row), flush=True)
        assert row["valid"], name
        if name in control and row["proxy_cost"] > control[name]["proxy_cost"] + 1.0e-7:
            print(f"REJECT {name}: proxy increased by "
                  f"{row['proxy_cost'] - control[name]['proxy_cost']:.9f}", flush=True)
            raise SystemExit(1)
    print(f"MEAN {np.mean([r['proxy_cost'] for r in rows]):.9f}", flush=True)


if __name__ == "__main__":
    main()
