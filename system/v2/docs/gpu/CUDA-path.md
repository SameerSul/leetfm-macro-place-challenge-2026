# CUDA path

## What this integration provides

The CUDA path adds an opt-in, GPU-backed scorer for hard-macro relocation in R2.
It does not replace the placer, the legality checks, or the final exact accept
gate. Its job is narrower: score a frozen pool of hard-relocation proposals in a
batched Torch path, rank them, then let the existing incremental exact scorer
verify and commit candidates serially.

The useful change is the restructuring around hard relocation:

1. Freeze the current placement state.
2. Generate legal relocation candidates across all selected hot hard macros.
3. Score the whole proposal pool with `cuda_delta`.
4. Sort proposals globally.
5. Re-check legality and exact-score proposals serially against the current
   post-commit state.
6. Commit only strict true-proxy improvements.

That gives us the cross-macro "evaluate-all-then-commit" shape needed to make GPU
work worthwhile. Earlier per-macro GPU batching was too small to beat the CPU
path on IBM-scale grids; this path makes the batch span many macros and many
targets.

## Where it lives

- `src/placer/local_search/relocation.py`
  - `_relocation_moves_propose_all(...)` implements the hard-relocation
    propose-all flow.
  - `_score_relocation_proposals_cuda_delta(...)` chunks and runs the CUDA-capable
    proposal scorer.
  - `_score_relocation_proposals_cuda_delta_batch(...)` evaluates HPWL, density,
    macro blockage, and touched-net routing deltas through Torch tensors.
  - Static tensors and memory accounting are built once per proposal pool, then
    reused across chunks.

- `src/placer/pipeline/macro_placer.py`
  - R2 passes `propose_all=...` into hard relocation.
  - `V2_RELOC_PROPOSE_ALL=auto` enables propose-all only when the configured
    backend is CUDA.

- `test/diagnostic/_cuda_relocation_status.py`
  - Reports CUDA visibility, `nvidia-smi`, configured device, allocation status,
    scorer stats, memory stats, and exact-score parity.
  - `--require-cuda` fails unless PyTorch can see CUDA, a real allocation works,
    and the relocation scorer actually ran on CUDA.

## How to use it

Default production behavior remains conservative. The CUDA hard-relocation path is
not enabled by default.

Enable it only on CUDA-visible runs:

```bash
V2_RELOC_PROPOSE_ALL=auto \
V2_RELOC_PROPOSE_LOG=1 \
PYTHONPATH=system/v2/src \
uv run evaluate system/v2/src/main.py -b ibm01
```

Force it regardless of backend:

```bash
V2_RELOC_PROPOSE_ALL=1
```

Keep it disabled:

```bash
V2_RELOC_PROPOSE_ALL=0
```

Limit exact verification to the top ranked proposals:

```bash
V2_RELOC_PROPOSE_TOP_M=32
```

Control CUDA chunking:

```bash
V2_RELOC_PROPOSE_CHUNK_SIZE=128
V2_RELOC_PROPOSE_MAX_MB=512
V2_RELOC_PROPOSE_AUTO_MEM_FRAC=0.25
V2_RELOC_PROPOSE_MEM_SAFETY=1.5
```

`V2_RELOC_PROPOSE_CHUNK_SIZE` is an explicit override. `V2_RELOC_PROPOSE_MAX_MB`
uses a fixed memory budget. `V2_RELOC_PROPOSE_AUTO_MEM_FRAC` derives the budget
from free CUDA memory. Static tensors are accounted separately, then the dynamic
chunk size is adjusted after the actual static allocation is known.

## What the GPU actually does

For each proposal chunk, Torch tensors compute the same proxy components used by
the exact relocation trial:

- HPWL delta for nets touched by the moved macro.
- Density delta from old and new macro occupancy.
- Hard-macro routing blockage delta.
- Touched-net routing delta.
- Smoothed routing congestion and top-k congestion reduction.

The scorer writes a proposal score equivalent to:

```text
wl_base + wl_delta + 0.5 * density + 0.5 * congestion
```

That is the same objective shape used by `IncrementalScorer._trial_at`. The CUDA
score is still only a ranking score. Before any placement is accepted, the serial
verify stage calls the exact incremental scorer on the current state and commits
only if the true proxy strictly improves.

## What this does not change

- It does not make soft relocation GPU-backed.
- It does not remove the CPU sequential relocation path.
- It does not bypass legality checks.
- It does not bypass exact scoring for accepted moves.
- It does not guarantee a score win on IBM benchmarks by itself.

This is intentionally an opt-in algorithmic variant. It can choose a different
proposal order than the legacy greedy loop, so it must be benchmarked as a
different search policy, not treated as an equivalent acceleration.

## Verification commands

Check that this machine can run the CUDA path:

```bash
PYTHONPATH=system/v2/src \
uv run python system/v2/test/diagnostic/_cuda_relocation_status.py \
  --benchmark ibm01 --exact-limit 9 --require-cuda
```

Expected CUDA-visible signs:

```text
torch_cuda_available=True
torch_device_count=1
cuda_allocation_status=ok
placer_backend=cuda
scorer_stats={'device': 'cuda:0', 'backend': 'cuda', ...}
```

Focused verifiers:

```bash
PYTHONPATH=system/v2/src \
uv run python system/v2/test/verification/_verify_relocation_propose_all_auto.py

PYTHONPATH=system/v2/src \
V2_RELOC_PROPOSE_ALL=auto V2_RELOC_PROPOSE_LOG=1 \
uv run python system/v2/test/verification/_verify_relocation_propose_all.py

PYTHONPATH=system/v2/src \
uv run python system/v2/test/verification/_verify_relocation_cuda_delta_scores.py
```

Recent local evidence on this workstation:

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, driver 610.47, 6141 MiB.
- `--require-cuda` diagnostic passed with `cuda_allocation_status=ok`.
- `cuda_delta` ran on `cuda:0`.
- Proposal-score parity passed on `ibm01` and `ibm04` with max deltas around
  `1e-7`.
- A pipeline smoke with `V2_RELOC_PROPOSE_ALL=auto` reached R2 and logged
  `scorer=cuda_delta device=cuda:0`.

## Practical interpretation

This integration gets the CUDA pull path into the real placement loop in a
controlled way. It gives us a working GPU-backed hard-relocation proposal scorer,
runtime diagnostics that prove real CUDA execution, and an `auto` switch that
uses the path only when CUDA is available.

The remaining question is not "can the system use the GPU?" It can. The remaining
question is whether this search policy beats the CPU default under full benchmark
budgets, especially because the CPU default still has strong sequential
prefilters and soft-relocation improvements that this hard-only CUDA path does
not replace.
