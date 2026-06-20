#!/usr/bin/env python3
"""Print train-benchmark CLI args from a GNN dataset.

By default this excludes the IBM validation and holdout benchmarks used by the
full IBM+NG45 split, so all NG45 benchmarks present in the dataset become
training benchmarks automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

DEFAULT_EXCLUDE = {"ibm10", "ibm12", "ibm16", "ibm17", "ibm18"}


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--include-default-excludes", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print names only, one per line")
    args = parser.parse_args()

    dataset = _load(args.dataset)
    examples = dataset.get("examples", {})
    names = sorted({str(name) for name in examples.get("benchmark", [])})
    excluded = set(args.exclude)
    if not args.include_default_excludes:
        excluded |= DEFAULT_EXCLUDE
    train_names = [name for name in names if name and name not in excluded]

    if args.list:
        print("\n".join(train_names))
    else:
        print(" ".join(f"--train-benchmark {name}" for name in train_names))


if __name__ == "__main__":
    main()
