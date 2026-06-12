"""Verify the CUDA relocation diagnostic can require real CUDA execution.

On CUDA-visible hosts, --require-cuda should pass. On CPU-backed or GPU-blocked
hosts, it should fail after printing enough runtime detail to explain why.

Usage:
  PYTHONPATH=submissions/varrahan/v2/src \
  uv run python submissions/varrahan/v2/test/verification/_verify_cuda_diagnostic_require_cuda.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in [
            "submissions/varrahan/v2/src",
            env.get("PYTHONPATH", ""),
        ]
        if part
    )
    proc = subprocess.run(
        [
            sys.executable,
            "submissions/varrahan/v2/test/diagnostic/_cuda_relocation_status.py",
            "--benchmark",
            "ibm01",
            "--exact-limit",
            "0",
            "--require-cuda",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if torch.cuda.is_available():
        if proc.returncode != 0:
            raise AssertionError(proc.stdout + proc.stderr)
        if "scorer_stats={'device': 'cuda" not in proc.stdout and '"device": "cuda' not in proc.stdout:
            raise AssertionError(f"missing CUDA scorer stats in stdout={proc.stdout!r}")
        if "cuda_allocation_status=ok" not in proc.stdout:
            raise AssertionError(f"missing successful CUDA allocation smoke check in stdout={proc.stdout!r}")
        print("PASS require-cuda diagnostic passed on CUDA-visible runtime")
        return 0

    if proc.returncode == 0:
        raise AssertionError("expected --require-cuda diagnostic failure on CUDA-unavailable runtime")
    combined = proc.stdout + proc.stderr
    for needle in (
        "require-cuda failed",
        "torch_cuda_available=False",
        "scorer_device=cpu",
        "cuda_allocation_status=skipped",
        "cuda_allocation_output=",
        "nvidia_smi_status=",
    ):
        if needle not in combined:
            raise AssertionError(f"missing {needle!r} in diagnostic output={combined!r}")
    print("PASS require-cuda diagnostic failed with runtime details")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
