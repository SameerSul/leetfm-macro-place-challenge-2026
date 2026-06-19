"""Diagnose the hard-relocation CUDA proposal scorer.

This is a lightweight runtime check for real GPU sessions. It reports PyTorch
CUDA visibility, the configured placer device, effective chunking stats, and a
small exact-score parity check for the `cuda_delta` proposal scorer.

Usage:
  PYTHONPATH=src \
  uv run python test/diagnostic/_cuda_relocation_status.py --benchmark ibm01

  # Fail unless the runtime and scorer actually use CUDA:
  PYTHONPATH=src \
  uv run python test/diagnostic/_cuda_relocation_status.py --require-cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from macro_place.loader import load_benchmark_from_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.config import _CUDA_DEVICE_REQUESTED, _GPU_BACKEND, _GPU_DEVICE, _GPU_DEVICE_NAME
from placer.local_search.relocation import _score_relocation_proposals_cuda_delta
from placer.scoring.congestion import _patch_plc_congestion
from placer.scoring.incremental import IncrementalScorer

_VERIFY_DIR = Path(__file__).resolve().parents[1] / "verification"
sys.path.insert(0, str(_VERIFY_DIR))
from _verify_relocation_cuda_delta_scores import _collect_proposals  # noqa: E402


def _torch_cuda_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _nvidia_smi_status() -> dict[str, str | int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except FileNotFoundError:
        return {"status": "missing", "returncode": 127, "output": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": 124, "output": "nvidia-smi timed out"}
    output = "\n".join(
        line for line in (result.stdout + result.stderr).splitlines() if line.strip()
    )
    status = "ok" if result.returncode == 0 else "error"
    return {"status": status, "returncode": int(result.returncode), "output": output[:500]}


def _cuda_allocation_status() -> dict[str, str]:
    if not torch.cuda.is_available() or _GPU_DEVICE.type != "cuda":
        return {"status": "skipped", "output": "cuda runtime or configured device unavailable"}
    try:
        _ = torch.empty((1,), device=_GPU_DEVICE)
        torch.cuda.synchronize(_GPU_DEVICE)
    except Exception as exc:
        return {"status": "error", "output": str(exc)[:500]}
    return {"status": "ok", "output": ""}


def _score_exact_subset(scorer: IncrementalScorer, proposals: list[dict], limit: int) -> float:
    max_delta = 0.0
    for proposal in proposals[:limit]:
        prep = scorer._prepare_move(int(proposal["i"]))
        try:
            exact = float(scorer._trial_at(prep, proposal["xy"]))
        finally:
            scorer._revert_prep(prep)
        max_delta = max(max_delta, abs(float(proposal["score"]) - exact))
    return max_delta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="ibm01")
    parser.add_argument("--top-hot", type=int, default=5)
    parser.add_argument("--n-targets", type=int, default=5)
    parser.add_argument("--exact-limit", type=int, default=16)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="exit nonzero unless PyTorch and the relocation scorer are using CUDA",
    )
    args = parser.parse_args()

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_device_count={_torch_cuda_device_count()}")
    smi = _nvidia_smi_status()
    print(f"nvidia_smi_status={smi['status']}")
    print(f"nvidia_smi_returncode={smi['returncode']}")
    print(f"nvidia_smi_output={smi['output']}")
    alloc = _cuda_allocation_status()
    print(f"cuda_allocation_status={alloc['status']}")
    print(f"cuda_allocation_output={alloc['output']}")
    print(f"requested_device={_CUDA_DEVICE_REQUESTED}")
    print(f"placer_backend={_GPU_BACKEND}")
    print(f"placer_device={_GPU_DEVICE}")
    print(f"placer_device_name={_GPU_DEVICE_NAME}")
    print(f"V2_REQUIRE_CUDA={os.environ.get('V2_REQUIRE_CUDA', '')}")
    print(f"V2_RELOC_PROPOSE_CHUNK_SIZE={os.environ.get('V2_RELOC_PROPOSE_CHUNK_SIZE', '')}")
    print(f"V2_RELOC_PROPOSE_MAX_MB={os.environ.get('V2_RELOC_PROPOSE_MAX_MB', '')}")
    print(f"V2_RELOC_PROPOSE_AUTO_MEM_FRAC={os.environ.get('V2_RELOC_PROPOSE_AUTO_MEM_FRAC', '')}")
    print(f"V2_RELOC_PROPOSE_MEM_SAFETY={os.environ.get('V2_RELOC_PROPOSE_MEM_SAFETY', '')}")

    bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{args.benchmark}")
    _patch_plc_congestion(plc, bm)
    pl = bm.macro_positions.numpy().astype(np.float64)
    n = bm.num_hard_macros
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0
    movable = bm.get_movable_mask().numpy()[:n]
    scorer = IncrementalScorer(plc, bm, pl)

    proposals, local_cong, tgt_cong = _collect_proposals(
        pos=pl[:n].copy(),
        sizes=sizes,
        hw=hw,
        hh=hh,
        cw=float(bm.canvas_width),
        ch=float(bm.canvas_height),
        movable=movable,
        plc=plc,
        benchmark=bm,
        incremental_scorer=scorer,
        top_hot=args.top_hot,
        n_targets=args.n_targets,
    )
    if not proposals:
        raise SystemExit("no legal hard-relocation proposals collected")

    _score_relocation_proposals_cuda_delta(
        proposals,
        pos=pl[:n].copy(),
        cw=float(bm.canvas_width),
        ch=float(bm.canvas_height),
        local_cong=local_cong,
        tgt_cong=tgt_cong,
        incremental_scorer=scorer,
    )
    stats = getattr(_score_relocation_proposals_cuda_delta, "last_stats", {})
    print(f"benchmark={args.benchmark}")
    print(f"proposals={len(proposals)}")
    print(f"scorer_stats={stats}")
    if args.require_cuda:
        cuda_ok = (
            torch.cuda.is_available()
            and _GPU_DEVICE.type == "cuda"
            and alloc["status"] == "ok"
            and stats.get("backend") == "cuda"
            and str(stats.get("device", "")).startswith("cuda")
        )
        if not cuda_ok:
            raise SystemExit(
                "require-cuda failed: "
                f"torch_cuda_available={torch.cuda.is_available()} "
                f"torch_device_count={_torch_cuda_device_count()} "
                f"placer_device={_GPU_DEVICE} scorer_device={stats.get('device')} "
                f"cuda_allocation_status={alloc['status']} cuda_allocation_output={alloc['output']} "
                f"nvidia_smi_status={smi['status']} nvidia_smi_output={smi['output']}"
            )
    exact_limit = max(0, min(args.exact_limit, len(proposals)))
    if exact_limit:
        print(f"exact_checked={exact_limit}")
        print(f"max_score_delta={_score_exact_subset(scorer, proposals, exact_limit):.3e}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
