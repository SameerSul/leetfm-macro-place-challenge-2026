"""Shared runtime configuration and logging."""

import torch

if torch.cuda.is_available():
    _USE_GPU = True
    _GPU_DEVICE = torch.device("cuda:0")
    _GPU_BACKEND = "cuda"
    _GPU_DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    _USE_GPU = False
    _GPU_DEVICE = torch.device("cpu")
    _GPU_BACKEND = "cpu"
    _GPU_DEVICE_NAME = "CPU"

# Use numba for routing hot loops when available; otherwise use numpy fallbacks.
try:
    from numba import njit as _numba_njit
    HAS_NUMBA = True
except ImportError:
    _numba_njit = None
    HAS_NUMBA = False


def _log(msg: str) -> None:
    print(msg, flush=True)
