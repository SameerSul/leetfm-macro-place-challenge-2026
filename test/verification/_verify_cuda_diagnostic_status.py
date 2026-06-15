"""Verify CUDA diagnostic nvidia-smi status normalization.

The real diagnostic should be useful on both CUDA-visible and CUDA-blocked
hosts. This verifier patches subprocess behavior in-process so it does not
depend on the current machine's GPU access state.

Usage:
  PYTHONPATH=src \
  uv run python test/verification/_verify_cuda_diagnostic_status.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

_DIAG_DIR = Path(__file__).resolve().parents[1] / "diagnostic"
sys.path.insert(0, str(_DIAG_DIR))
import _cuda_relocation_status as diag  # noqa: E402


def _patch_run(fn):
    original = diag.subprocess.run
    diag.subprocess.run = fn
    return original


def main() -> int:
    original = diag.subprocess.run
    try:
        _patch_run(
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout="NVIDIA Test GPU, 555.55, 24576 MiB\n",
                stderr="",
            )
        )
        ok = diag._nvidia_smi_status()
        if ok["status"] != "ok" or ok["returncode"] != 0 or "NVIDIA Test GPU" not in ok["output"]:
            raise AssertionError(f"unexpected ok status: {ok}")

        _patch_run(
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=255,
                stdout="",
                stderr="Failed to initialize NVML\n",
            )
        )
        err = diag._nvidia_smi_status()
        if err["status"] != "error" or err["returncode"] != 255 or "NVML" not in err["output"]:
            raise AssertionError(f"unexpected error status: {err}")

        def missing(*_args, **_kwargs):
            raise FileNotFoundError()

        _patch_run(missing)
        miss = diag._nvidia_smi_status()
        if miss["status"] != "missing" or miss["returncode"] != 127:
            raise AssertionError(f"unexpected missing status: {miss}")

        def timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

        _patch_run(timeout)
        tout = diag._nvidia_smi_status()
        if tout["status"] != "timeout" or tout["returncode"] != 124:
            raise AssertionError(f"unexpected timeout status: {tout}")
    finally:
        diag.subprocess.run = original

    print("PASS cuda_diagnostic_status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
