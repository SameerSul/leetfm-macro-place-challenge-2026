"""Verify cuda_delta chunk retry behavior without requiring a visible GPU.

The real retry path only activates when `_GPU_DEVICE.type == "cuda"` and a
Torch OOM is raised. This verifier monkeypatches the relocation module in
process: the first oversized batch raises a CUDA-OOM-like RuntimeError, then
the wrapper should retry with smaller chunks and score every proposal.

Usage:
  PYTHONPATH=src \
  uv run python test/verification/_verify_relocation_cuda_chunk_retry.py
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
    calls: list[int] = []
    static_calls = 0
    original_device = relocation._GPU_DEVICE
    original_batch = relocation._score_relocation_proposals_cuda_delta_batch
    original_static = relocation._build_relocation_cuda_static_tensors
    old_chunk = os.environ.get("RELOC_PROPOSE_CHUNK_SIZE")
    old_log = os.environ.get("RELOC_PROPOSE_LOG")

    def fake_batch(batch, **_kwargs):
        calls.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory while allocating test tensor")
        for proposal in batch:
            proposal["score"] = 10.0 + float(proposal["i"])

    def fake_static(*_args, **_kwargs):
        nonlocal static_calls
        static_calls += 1
        return {}

    try:
        relocation._GPU_DEVICE = torch.device("cuda:0")
        relocation._score_relocation_proposals_cuda_delta_batch = fake_batch
        relocation._build_relocation_cuda_static_tensors = fake_static
        os.environ["RELOC_PROPOSE_CHUNK_SIZE"] = "4"
        os.environ["RELOC_PROPOSE_LOG"] = "1"

        relocation._score_relocation_proposals_cuda_delta(
            proposals,
            pos=None,
            cw=0.0,
            ch=0.0,
            local_cong=None,
            tgt_cong=None,
            incremental_scorer=None,
        )
    finally:
        relocation._GPU_DEVICE = original_device
        relocation._score_relocation_proposals_cuda_delta_batch = original_batch
        relocation._build_relocation_cuda_static_tensors = original_static
        if old_chunk is None:
            os.environ.pop("RELOC_PROPOSE_CHUNK_SIZE", None)
        else:
            os.environ["RELOC_PROPOSE_CHUNK_SIZE"] = old_chunk
        if old_log is None:
            os.environ.pop("RELOC_PROPOSE_LOG", None)
        else:
            os.environ["RELOC_PROPOSE_LOG"] = old_log

    if calls != [4, 2, 2, 1]:
        raise AssertionError(f"unexpected retry chunk calls: {calls}")
    if static_calls != 1:
        raise AssertionError(f"static tensors should be built once, got {static_calls}")
    scores = [p["score"] for p in proposals]
    if scores != [10.0, 11.0, 12.0, 13.0, 14.0]:
        raise AssertionError(f"unexpected proposal scores: {scores}")

    print(f"PASS calls={calls} scores={scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
