"""Shared runtime configuration and logging."""

import os

import torch

_TRUE_ENV = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_CUDA_DEVICE_REQUESTED = os.environ.get("V2_CUDA_DEVICE", "cuda:0").strip() or "cuda:0"


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip() in _TRUE_ENV


if torch.cuda.is_available():
    _USE_GPU = True
    _GPU_DEVICE = torch.device(_CUDA_DEVICE_REQUESTED)
    if _GPU_DEVICE.type != "cuda":
        raise RuntimeError(
            f"V2_CUDA_DEVICE must name a CUDA device, got {_CUDA_DEVICE_REQUESTED!r}."
        )
    _cuda_index = _GPU_DEVICE.index
    if _cuda_index is None:
        _cuda_index = torch.cuda.current_device()
    if _cuda_index < 0 or _cuda_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"V2_CUDA_DEVICE={_CUDA_DEVICE_REQUESTED!r} is outside visible CUDA "
            f"device_count={torch.cuda.device_count()}."
        )
    _GPU_BACKEND = "cuda"
    _GPU_DEVICE_NAME = torch.cuda.get_device_name(_cuda_index)
else:
    if _env_enabled("V2_REQUIRE_CUDA"):
        raise RuntimeError(
            "V2_REQUIRE_CUDA=1 but PyTorch cannot see a CUDA device "
            f"(torch_cuda_available={torch.cuda.is_available()}, "
            f"torch_cuda_version={torch.version.cuda}, "
            f"requested_device={_CUDA_DEVICE_REQUESTED!r})."
        )
    _USE_GPU = False
    _GPU_DEVICE = torch.device("cpu")
    _GPU_BACKEND = "cpu"
    _GPU_DEVICE_NAME = "CPU"

# Use numba for routing hot loops. Production runs should fail visibly if the
# JIT path is missing because the numpy fallback can miss deadline-bound rounds.
try:
    from numba import njit as _numba_njit

    HAS_NUMBA = True
except ImportError:
    _numba_njit = None
    HAS_NUMBA = False
    if not _env_enabled("V2_ALLOW_NUMBA_FALLBACK"):
        raise RuntimeError(
            "numba is required for production hierarchy placement. Install the "
            "project dependencies (`uv sync` or `uv pip install -r requirements.txt`), "
            "or set V2_ALLOW_NUMBA_FALLBACK=1 for slow diagnostic-only runs."
        )
    import warnings as _warnings

    _warnings.warn(
        "numba not installed; using the slow numpy fallback because " "V2_ALLOW_NUMBA_FALLBACK=1.",
        stacklevel=2,
    )


def _log(msg: str) -> None:
    print(msg, flush=True)
