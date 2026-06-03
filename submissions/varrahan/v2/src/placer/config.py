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

# Speedup #34 (2026-05-31): numba JIT for the routing-apply hot loops
# (_apply_h_strips_batch / _apply_v_strips_batch). Profile on ibm15 attributed
# 10% of per-move time to these two strip-fillers alone, plus another 38%
# inside _apply_3pin_routing_vec → strips. Numba-jitted explicit-loop
# variants are ~3-5× the numpy version. Soft-import so the placer still
# works in eval environments without numba installed (falls back to the
# original numpy path).
try:
    from numba import njit as _numba_njit
    HAS_NUMBA = True
except ImportError:
    _numba_njit = None
    HAS_NUMBA = False


def _log(msg: str) -> None:
    print(msg, flush=True)

