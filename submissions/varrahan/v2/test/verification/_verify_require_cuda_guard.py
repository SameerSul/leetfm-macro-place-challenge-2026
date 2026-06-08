"""Verify V2_REQUIRE_CUDA fails fast when CUDA is unavailable.

This intentionally only asserts behavior on CPU-backed runtimes. On machines
where CUDA is visible, V2_REQUIRE_CUDA should allow import and this verifier
reports that state.

Usage:
  PYTHONPATH=submissions/varrahan/v2/src \
  uv run python submissions/varrahan/v2/test/verification/_verify_require_cuda_guard.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch


def main() -> int:
    env = os.environ.copy()
    env["V2_REQUIRE_CUDA"] = "1"
    requested_device = "cuda:0" if torch.cuda.is_available() else "cuda:7"
    env["V2_CUDA_DEVICE"] = requested_device
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [
            "submissions/varrahan/v2/src",
            env.get("PYTHONPATH", ""),
        ]
        if part
    )
    proc = subprocess.run(
        [sys.executable, "-c", "import placer.config; print(placer.config._GPU_BACKEND)"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if torch.cuda.is_available():
        if proc.returncode != 0:
            raise AssertionError(proc.stderr)
        if "cuda" not in proc.stdout:
            raise AssertionError(f"expected cuda backend, got stdout={proc.stdout!r}")
        print("PASS cuda-visible require guard allowed import")
        return 0

    if proc.returncode == 0:
        raise AssertionError("expected V2_REQUIRE_CUDA import failure on CUDA-unavailable runtime")
    if "V2_REQUIRE_CUDA=1" not in proc.stderr:
        raise AssertionError(f"missing require-cuda error in stderr={proc.stderr!r}")
    if requested_device not in proc.stderr:
        raise AssertionError(f"missing requested-device detail in stderr={proc.stderr!r}")
    print("PASS cuda-unavailable require guard failed fast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
