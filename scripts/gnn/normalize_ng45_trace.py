#!/usr/bin/env python3
"""Normalize NG45 trace benchmark names.

The NG45 evaluator loads each design from a directory named output_CT_Grouping,
so trace rows can all arrive as benchmark=output_CT_Grouping. This script
splits one NG45 trace by hier_final boundaries in evaluator order and rewrites
benchmark names to stable aliases such as ariane133_ng45.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ORDER = ["ariane133_ng45", "ariane136_ng45", "mempool_tile_ng45", "nvdla_ng45"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path, required=True)
    parser.add_argument("--out", dest="out", type=Path, required=True)
    parser.add_argument("--design", action="append", help="Design aliases in trace order")
    args = parser.parse_args()

    order = args.design or DEFAULT_ORDER
    design_i = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.src.open("r", encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as out:
        for line_no, line in enumerate(src, 1):
            raw = line.strip()
            if not raw:
                continue
            row: dict[str, Any] = json.loads(raw)
            if row.get("benchmark") == "output_CT_Grouping":
                if design_i >= len(order):
                    raise ValueError(f"{args.src}:{line_no}: more NG45 segments than design names")
                row["benchmark"] = order[design_i]
            out.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            if row.get("event") == "hier_final":
                design_i += 1
    if design_i != len(order):
        raise ValueError(f"expected {len(order)} NG45 segments, found {design_i}")


if __name__ == "__main__":
    main()
