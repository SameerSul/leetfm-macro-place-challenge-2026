"""Verify cuda_delta does not chunk-retry static-cache CUDA OOMs.

Batch OOMs can be fixed by reducing the proposal chunk size. Static-cache OOMs
come from pass-level tensors and should fail fast with a clear message.

Usage:
  PYTHONPATH=src \
  uv run python test/verification/_verify_relocation_cuda_static_oom.py
"""

from __future__ import annotations

import os

import torch

import placer.local_search.relocation as relocation


def main() -> int:
    proposals = [
        {"score": 0.0, "i": i, "target_index": i, "candidate_rank": i, "xy": (0.0, 0.0)}
        for i in range(5)
    ]
    calls = 0
    original_device = relocation._GPU_DEVICE
    original_static = relocation._build_relocation_cuda_static_tensors
    original_batch = relocation._score_relocation_proposals_cuda_delta_batch
    old_chunk = os.environ.get("RELOC_PROPOSE_CHUNK_SIZE")

    def fake_static(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("CUDA out of memory while allocating static tensor")

    def fake_batch(*_args, **_kwargs):
        raise AssertionError("batch scorer should not be called after static OOM")

    try:
        relocation._GPU_DEVICE = torch.device("cuda:0")
        relocation._build_relocation_cuda_static_tensors = fake_static
        relocation._score_relocation_proposals_cuda_delta_batch = fake_batch
        os.environ["RELOC_PROPOSE_CHUNK_SIZE"] = "4"
        try:
            relocation._score_relocation_proposals_cuda_delta(
                proposals,
                pos=None,
                cw=0.0,
                ch=0.0,
                local_cong=None,
                tgt_cong=None,
                incremental_scorer=None,
            )
        except RuntimeError as exc:
            if "static tensors" not in str(exc):
                raise AssertionError(f"unexpected error: {exc}") from exc
            if "estimated_static_bytes=0" not in str(exc):
                raise AssertionError(f"missing static estimate in error: {exc}") from exc
        else:
            raise AssertionError("expected static-cache OOM failure")
    finally:
        relocation._GPU_DEVICE = original_device
        relocation._build_relocation_cuda_static_tensors = original_static
        relocation._score_relocation_proposals_cuda_delta_batch = original_batch
        if old_chunk is None:
            os.environ.pop("RELOC_PROPOSE_CHUNK_SIZE", None)
        else:
            os.environ["RELOC_PROPOSE_CHUNK_SIZE"] = old_chunk

    if calls != 1:
        raise AssertionError(f"static cache should be attempted once, got {calls}")
    print("PASS static_oom_failed_fast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
