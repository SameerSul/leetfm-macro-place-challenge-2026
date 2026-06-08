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

# Use numba for routing hot loops when available; otherwise use numpy fallbacks.
# numba is REQUIRED for full speed — it JITs the routing-apply (~half the runtime).
# Without it the placer still runs (numpy fallbacks) but ~25% slower: --all goes
# from ~43 min to ~58 min (near the 1 h cap) and the deadline-bound benchmarks
# lose rounds (avg 1.1380 -> 1.1403). numba is in v2/requirements.txt but NOT
# pyproject.toml, so `uv sync` alone does not install it — install requirements.txt.
try:
    from numba import njit as _numba_njit
    HAS_NUMBA = True
except ImportError:
    import warnings as _warnings
    _warnings.warn(
        "numba not installed — routing-apply runs the slow numpy fallback "
        "(~25% slower, --all ~58min near the 1h cap). Install "
        "submissions/varrahan/v2/requirements.txt for the JIT path.",
        stacklevel=2,
    )
    _numba_njit = None
    HAS_NUMBA = False


def _log(msg: str) -> None:
    print(msg, flush=True)
