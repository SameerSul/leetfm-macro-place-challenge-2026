"""Run the current region-swap diagnostic.

Examples:
    uv run python test/diagnostic/_sweep_region_swaps.py --bench ibm10 ibm17
    uv run python test/diagnostic/_sweep_region_swaps.py --quick --cuda-status
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROXY_RE = re.compile(r"proxy=([0-9.]+).*VALID")


VARIANTS = [("current", {})]


def _run_eval(bench: str, variant: str, env_updates: dict[str, str]) -> tuple[float | None, str]:
    env = os.environ.copy()
    env.update(env_updates)
    cmd = ["uv", "run", "evaluate", "src/main.py", "-b", bench]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    proxy = None
    for line in proc.stdout.splitlines():
        match = PROXY_RE.search(line)
        if match:
            proxy = float(match.group(1))
    status = "PASS" if proc.returncode == 0 and proxy is not None else f"FAIL({proc.returncode})"
    print(
        f"{bench:>6} {variant:<16} {status:<8} proxy={proxy if proxy is not None else 'NA'}",
        flush=True,
    )
    return proxy, proc.stdout


def _cuda_status() -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "test/diagnostic/_cuda_relocation_status.py",
        "--benchmark",
        "ibm01",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print("\n=== CUDA relocation diagnostic ===", flush=True)
    print(proc.stdout)
    print(
        "NOTE: hierarchy region swaps are still sequential incremental-scored moves; "
        "this diagnostic covers the dormant hard-relocation cuda_delta scorer."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench",
        nargs="+",
        default=["ibm02", "ibm07", "ibm10", "ibm11", "ibm16", "ibm17"],
        help="Benchmarks to sweep.",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run only current, swaps_off, and density_off."
    )
    parser.add_argument(
        "--cuda-status", action="store_true", help="Print CUDA relocation diagnostic too."
    )
    args = parser.parse_args()

    variants = VARIANTS
    if args.quick:
        keep = {"current", "swaps_off", "density_off"}
        variants = [v for v in VARIANTS if v[0] in keep]

    print("bench  variant          status   proxy", flush=True)
    print("----------------------------------------", flush=True)
    for bench in args.bench:
        for name, env_updates in variants:
            _run_eval(bench, name, env_updates)

    if args.cuda_status:
        _cuda_status()


if __name__ == "__main__":
    sys.exit(main())
