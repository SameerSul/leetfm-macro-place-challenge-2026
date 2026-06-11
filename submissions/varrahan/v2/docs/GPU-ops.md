# GPU Operations Status

This note tracks GPU-related architecture ideas for CongFlow v2. As of
2026-06-11, there is **no production Phase 9b GPU global-exploration stage** and
no live `USE_GPU_EXPLORATION` entry point in `src/placer/pipeline/macro_placer.py`.

The implemented GPU path is narrower and documented in
[`CUDA-path.md`](CUDA-path.md): an opt-in CUDA-backed hard-relocation
`propose_all` ranking path for R2. It generates a frozen pool of hard-relocation
proposals, ranks them with a batched Torch scorer, then re-checks legality and
exact-scores candidates serially before any commit. The strict true-proxy accept
gate remains the authority.

## Current Implemented GPU Path

- Runtime backend detection lives in `src/placer/config.py`.
- `V2_CUDA_DEVICE` selects the CUDA device when PyTorch can see CUDA.
- `V2_REQUIRE_CUDA=1` turns missing CUDA into a hard error.
- `V2_RELOC_PROPOSE_ALL=auto` enables hard-relocation propose-all only when the
  configured backend is CUDA.
- `V2_RELOC_PROPOSE_ALL=1` forces the propose-all search variant.
- `V2_RELOC_PROPOSE_TOP_M`, `V2_RELOC_PROPOSE_CHUNK_SIZE`,
  `V2_RELOC_PROPOSE_MAX_MB`, `V2_RELOC_PROPOSE_AUTO_MEM_FRAC`, and
  `V2_RELOC_PROPOSE_MEM_SAFETY` tune verification depth and memory/chunking.

See `CUDA-path.md` for diagnostics and verification commands.

## Retired / Not Implemented

Earlier per-macro GPU batching of relocation candidate evaluation was
implemented, verified, measured, and removed because it did not beat the CPU +
numba path on IBM-scale grids. The bottleneck was CPU-side routing-strip
generation and kernel-launch granularity, not GPU reduction throughput.

The old "Phase 9b GPU Global Exploration" concept remains a possible design
direction, but it is not integrated. If revived, it should be treated as a new
search policy with these gates:

- Add an explicit pipeline phase after Phase 9 and before multi-seed 2-opt.
- Use a fresh `_exact_proxy` / `IncrementalScorer` initialization for returned
  candidate layouts; do not patch a pre-existing scorer through a massive GPU
  delta.
- Keep exact CPU scoring as the commit gate.
- Add focused verification and an `--all` A/B against the current CPU default,
  not just a CUDA smoke test.

Until that work exists in code, references to Phase 9b should be read as a design
proposal only.
