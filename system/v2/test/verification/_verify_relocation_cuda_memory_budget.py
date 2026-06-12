"""Verify V2_RELOC_PROPOSE_MAX_MB caps automatic cuda_delta chunking.

This monkeypatches the relocation module to exercise CUDA chunk selection
without requiring a visible GPU.

Usage:
  PYTHONPATH=system/v2/src \
  uv run python system/v2/test/verification/_verify_relocation_cuda_memory_budget.py
"""

from __future__ import annotations

import os

import torch

import placer.local_search.relocation as relocation


class _Scorer:
    grid_row = 1
    grid_col = 1


def main() -> int:
    proposals = [
        {"score": 0.0, "i": i, "target_index": i, "candidate_rank": i, "xy": (0.0, 0.0)}
        for i in range(7)
    ]
    original_device = relocation._GPU_DEVICE
    original_static = relocation._build_relocation_cuda_static_tensors
    original_batch = relocation._score_relocation_proposals_cuda_delta_batch
    original_mem_get_info = relocation.torch.cuda.mem_get_info
    old_chunk = os.environ.get("V2_RELOC_PROPOSE_CHUNK_SIZE")
    old_budget = os.environ.get("V2_RELOC_PROPOSE_MAX_MB")
    old_auto_frac = os.environ.get("V2_RELOC_PROPOSE_AUTO_MEM_FRAC")
    old_safety = os.environ.get("V2_RELOC_PROPOSE_MEM_SAFETY")

    try:
        relocation._GPU_DEVICE = torch.device("cuda:0")
        relocation._build_relocation_cuda_static_tensors = lambda *_args, **_kwargs: {}

        calls: list[int] = []

        def fake_batch(batch, **_kwargs):
            calls.append(len(batch))
            for proposal in batch:
                proposal["score"] = float(proposal["i"])

        relocation._score_relocation_proposals_cuda_delta_batch = fake_batch
        os.environ.pop("V2_RELOC_PROPOSE_CHUNK_SIZE", None)
        os.environ.pop("V2_RELOC_PROPOSE_MEM_SAFETY", None)
        os.environ["V2_RELOC_PROPOSE_MAX_MB"] = "0.00008"
        relocation._score_relocation_proposals_cuda_delta(
            proposals,
            pos=None,
            cw=0.0,
            ch=0.0,
            local_cong=None,
            tgt_cong=None,
            incremental_scorer=_Scorer(),
        )
        if calls != [2, 2, 2, 1]:
            raise AssertionError(f"unexpected budget chunk calls: {calls}")
        stats = getattr(relocation._score_relocation_proposals_cuda_delta, "last_stats", {})
        if (
            stats.get("initial_chunk_size") != 1
            or stats.get("final_chunk_size") != 2
            or stats.get("memory_budget_chunk") != 2
            or stats.get("chunk_source") != "memory_budget"
            or stats.get("memory_budget_total_bytes") != 83
            or stats.get("memory_budget_dynamic_bytes") != 83
            or stats.get("memory_budget_static_exceeds") is not False
            or stats.get("memory_budget_adjusted_after_static") is not True
            or stats.get("memory_budget_adjustment") != "grow"
            or stats.get("static_tensor_bytes_estimate") != 76
            or stats.get("static_tensor_bytes_actual") != 0
            or stats.get("grid_dynamic_bytes_per_proposal") != 40
            or stats.get("hpwl_dynamic_bytes_per_proposal") != 0
            or stats.get("memory_safety_factor") != 1.0
        ):
            raise AssertionError(f"unexpected budget stats: {stats}")

        calls.clear()
        os.environ.pop("V2_RELOC_PROPOSE_CHUNK_SIZE", None)
        os.environ["V2_RELOC_PROPOSE_MAX_MB"] = "0.00008"
        os.environ["V2_RELOC_PROPOSE_MEM_SAFETY"] = "2"
        relocation._score_relocation_proposals_cuda_delta(
            proposals,
            pos=None,
            cw=0.0,
            ch=0.0,
            local_cong=None,
            tgt_cong=None,
            incremental_scorer=_Scorer(),
        )
        if calls != [1, 1, 1, 1, 1, 1, 1]:
            raise AssertionError(f"unexpected safety-budget calls: {calls}")
        stats = getattr(relocation._score_relocation_proposals_cuda_delta, "last_stats", {})
        if (
            stats.get("memory_budget_chunk") != 1
            or stats.get("dynamic_bytes_per_proposal") != 80
            or stats.get("grid_dynamic_bytes_per_proposal") != 80
            or stats.get("hpwl_dynamic_bytes_per_proposal") != 0
            or stats.get("memory_budget_total_bytes") != 83
            or stats.get("memory_budget_dynamic_bytes") != 83
            or stats.get("memory_budget_static_exceeds") is not False
            or stats.get("memory_budget_adjusted_after_static") is not False
            or stats.get("memory_budget_adjustment") != "none"
            or stats.get("memory_safety_factor") != 2.0
        ):
            raise AssertionError(f"unexpected safety-budget stats: {stats}")

        calls.clear()
        os.environ.pop("V2_RELOC_PROPOSE_MEM_SAFETY", None)
        os.environ["V2_RELOC_PROPOSE_CHUNK_SIZE"] = "4"
        relocation._score_relocation_proposals_cuda_delta(
            proposals,
            pos=None,
            cw=0.0,
            ch=0.0,
            local_cong=None,
            tgt_cong=None,
            incremental_scorer=_Scorer(),
        )
        if calls != [4, 3]:
            raise AssertionError(f"unexpected explicit chunk calls: {calls}")
        stats = getattr(relocation._score_relocation_proposals_cuda_delta, "last_stats", {})
        if (
            stats.get("initial_chunk_size") != 4
            or stats.get("memory_budget_chunk") is not None
            or stats.get("chunk_source") != "env"
        ):
            raise AssertionError(f"unexpected explicit stats: {stats}")

        calls.clear()
        os.environ.pop("V2_RELOC_PROPOSE_CHUNK_SIZE", None)
        os.environ["V2_RELOC_PROPOSE_MAX_MB"] = "1000"
        relocation._score_relocation_proposals_cuda_delta(
            proposals,
            pos=None,
            cw=0.0,
            ch=0.0,
            local_cong=None,
            tgt_cong=None,
            incremental_scorer=_Scorer(),
        )
        if calls != [7]:
            raise AssertionError(f"unexpected high-budget calls: {calls}")
        stats = getattr(relocation._score_relocation_proposals_cuda_delta, "last_stats", {})
        if stats.get("chunk_source") != "cuda_default":
            raise AssertionError(f"unexpected high-budget stats: {stats}")

        calls.clear()
        os.environ.pop("V2_RELOC_PROPOSE_CHUNK_SIZE", None)
        os.environ.pop("V2_RELOC_PROPOSE_MAX_MB", None)
        os.environ.pop("V2_RELOC_PROPOSE_MEM_SAFETY", None)
        os.environ["V2_RELOC_PROPOSE_AUTO_MEM_FRAC"] = "0.3"
        relocation.torch.cuda.mem_get_info = lambda *_args, **_kwargs: (400, 1000)
        relocation._score_relocation_proposals_cuda_delta(
            proposals,
            pos=None,
            cw=0.0,
            ch=0.0,
            local_cong=None,
            tgt_cong=None,
            incremental_scorer=_Scorer(),
        )
        if calls != [3, 3, 1]:
            raise AssertionError(f"unexpected auto-budget calls: {calls}")
        stats = getattr(relocation._score_relocation_proposals_cuda_delta, "last_stats", {})
        if (
            stats.get("memory_budget_chunk") != 3
            or stats.get("memory_budget_source") != "auto_mem_frac"
            or stats.get("chunk_source") != "memory_budget"
            or stats.get("memory_budget_total_bytes") != 120
            or stats.get("memory_budget_dynamic_bytes") != 120
            or stats.get("memory_budget_static_exceeds") is not False
            or stats.get("memory_budget_adjusted_after_static") is not True
            or stats.get("memory_budget_adjustment") != "grow"
            or stats.get("auto_memory_frac") != 0.3
            or stats.get("auto_cuda_free_bytes") != 400
            or stats.get("auto_cuda_total_bytes") != 1000
        ):
            raise AssertionError(f"unexpected auto-budget stats: {stats}")

        calls.clear()
        os.environ.pop("V2_RELOC_PROPOSE_CHUNK_SIZE", None)
        os.environ.pop("V2_RELOC_PROPOSE_AUTO_MEM_FRAC", None)
        os.environ["V2_RELOC_PROPOSE_MAX_MB"] = "0.00004"
        relocation._build_relocation_cuda_static_tensors = lambda *_args, **_kwargs: {
            "oversized_static": torch.empty(100, dtype=torch.uint8)
        }
        relocation._score_relocation_proposals_cuda_delta(
            proposals,
            pos=None,
            cw=0.0,
            ch=0.0,
            local_cong=None,
            tgt_cong=None,
            incremental_scorer=_Scorer(),
        )
        if calls != [1, 1, 1, 1, 1, 1, 1]:
            raise AssertionError(f"unexpected static-exceeds calls: {calls}")
        stats = getattr(relocation._score_relocation_proposals_cuda_delta, "last_stats", {})
        if (
            stats.get("memory_budget_total_bytes") != 41
            or stats.get("memory_budget_dynamic_bytes") != 0
            or stats.get("memory_budget_static_exceeds") is not True
            or stats.get("memory_budget_adjusted_after_static") is not False
            or stats.get("memory_budget_adjustment") != "none"
            or stats.get("static_tensor_bytes_actual") != 100
            or stats.get("memory_budget_chunk") != 1
        ):
            raise AssertionError(f"unexpected static-exceeds stats: {stats}")

        calls.clear()
        os.environ["V2_RELOC_PROPOSE_MAX_MB"] = "0.00018"
        relocation._build_relocation_cuda_static_tensors = lambda *_args, **_kwargs: {
            "larger_static": torch.empty(120, dtype=torch.uint8)
        }
        relocation._score_relocation_proposals_cuda_delta(
            proposals,
            pos=None,
            cw=0.0,
            ch=0.0,
            local_cong=None,
            tgt_cong=None,
            incremental_scorer=_Scorer(),
        )
        if calls != [1, 1, 1, 1, 1, 1, 1]:
            raise AssertionError(f"unexpected actual-static-shrink calls: {calls}")
        stats = getattr(relocation._score_relocation_proposals_cuda_delta, "last_stats", {})
        if (
            stats.get("initial_chunk_size") != 2
            or stats.get("final_chunk_size") != 1
            or stats.get("memory_budget_chunk") != 1
            or stats.get("memory_budget_total_bytes") != 188
            or stats.get("memory_budget_dynamic_bytes") != 68
            or stats.get("memory_budget_adjusted_after_static") is not True
            or stats.get("memory_budget_adjustment") != "shrink"
            or stats.get("static_tensor_bytes_estimate") != 76
            or stats.get("static_tensor_bytes_actual") != 120
            or stats.get("memory_budget_static_exceeds") is not False
        ):
            raise AssertionError(f"unexpected actual-static-shrink stats: {stats}")
    finally:
        relocation._GPU_DEVICE = original_device
        relocation._build_relocation_cuda_static_tensors = original_static
        relocation._score_relocation_proposals_cuda_delta_batch = original_batch
        relocation.torch.cuda.mem_get_info = original_mem_get_info
        if old_chunk is None:
            os.environ.pop("V2_RELOC_PROPOSE_CHUNK_SIZE", None)
        else:
            os.environ["V2_RELOC_PROPOSE_CHUNK_SIZE"] = old_chunk
        if old_budget is None:
            os.environ.pop("V2_RELOC_PROPOSE_MAX_MB", None)
        else:
            os.environ["V2_RELOC_PROPOSE_MAX_MB"] = old_budget
        if old_auto_frac is None:
            os.environ.pop("V2_RELOC_PROPOSE_AUTO_MEM_FRAC", None)
        else:
            os.environ["V2_RELOC_PROPOSE_AUTO_MEM_FRAC"] = old_auto_frac
        if old_safety is None:
            os.environ.pop("V2_RELOC_PROPOSE_MEM_SAFETY", None)
        else:
            os.environ["V2_RELOC_PROPOSE_MEM_SAFETY"] = old_safety

    print("PASS memory_budget_and_env_override")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
