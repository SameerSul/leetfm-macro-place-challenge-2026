# v1 — Varrahan's Submission

Active placer for the Partcl/HRT Macro Placement Challenge. Built on top of `sameer_v1` with targeted changes documented in `PROGRESS.md`.

## What's different from `sameer_v1`

Pipeline structure is preserved: legalize from `initial.plc` → cong-grad Phase 1/2/3 → noise tail. Concrete code changes:

| Change | sameer_v1 | v1 | Effect |
|---|---|---|---|
| `EXACT_MACRO_THRESHOLD` | 340 | **400** | Re-includes ibm11 (n=373), ibm15 (n=393) in exact pipeline |
| `EXACT_GRID_CELL_LIMIT` | 2000 | **2200** | Re-includes ibm15 (grid=2166), ibm18 (grid=2145) |
| `BUDGET_OVERRUN_S` | (n/a) | **60.0s** | Allows directed-restart phases (cong-grad Phase 1/2/3) up to 60s overrun |
| `_will_legalize` | scalar spiral search | **vectorized** | 7-12× speedup; bit-equivalent to scalar (Tier 3, 2026-05-19) |
| 2-opt swap post-pass on baseline | (none) | **on baseline-only branch** | Tiny +0.0001 to avg via local refinement on n>400 benchmarks |
| `t_one_score` budget guard | fixed at baseline | **running max** | Adapts to CPU contention under --all (re-add v11 logic) |
| `_routing_congestion_perturb` | scalar | **vectorized** | Tier 2, 2026-05-19. Same RNG order, deterministic |
| `pl_scratch` scoring buffer | clone-per-call | **shared, in-place** | Saves N-1 clones per benchmark |

Plus minor pattern change: cong-grad call sites switched from `if not _try_restart(...): return best_pl` to `if not _try_restart(...): break` so a budget-failure in Phase 1 doesn't kill Phase 2/3.

## Tier 3 vectorize `_will_legalize` (added 2026-05-19)

The greedy spiral search is now fully numpy-vectorized:
- Per-ring candidate generation: all 8r positions at once via `_ring_offsets`.
- Per-ring conflict check: single `[K, P]` matrix op instead of nested loops.
- 7-12× faster on ibm04 (3.2s → 0.27s); proportionally faster on larger benchmarks.

**Critical float32 precision fix:** the original scalar code computes `d² = (cx - pos[idx, 0])² + ...` where `cx` is a Python float and `pos[idx, 0]` is a numpy float32; numpy demotes Python float to float32 in the subtraction, so d² is computed at float32 precision. This causes symmetric ring candidates like (-1, 0) and (0, -1) to break ties at float32 noise instead of being truly equal. The vectorized version mirrors this by casting `cand_x` to `pos.dtype` before the subtraction. Without this, ibm04's cong-grad iter-2 trajectory diverges and ibm04 lands at 1.3364 instead of 1.3316. See placer.py `_will_legalize` for the gory details.

## Async DREAMPlace bridge (in progress, 2026-05-20)

The deleted `dreamplace_bridge/` module was restored from commit 111f315 and extended with:
- `AsyncDreamplaceHandle` — polling-friendly subprocess handle.
- `launch_dreamplace_async(...)` — non-blocking `subprocess.Popen` launch.

The placer launches DREAMPlace at `place()` entry (subprocess runs in parallel with our scoring) and checks after Phase 3 as one additive candidate ("dreamplace global"). If DREAMPlace's placement legalizes-and-scores well, it becomes a new best. A second additive ("cong-grad from-dreamplace f=0.04") runs cong-grad from DREAMPlace's legalized position to exploit the plc-state mutation effect that PROGRESS.md noted as the source of v13's real wins.

This addresses the v13 (sync) failure: 10-15s subprocess overhead displaced productive restarts on 7/17 benchmarks. Async hides that cost behind scoring time. DREAMPlace build status: see `dreamplace_build/`.

## Today's experiments (2026-05-19 to 2026-05-20) — all reverted

| Experiment | Result | Why reverted |
|---|---|---|
| Multi-order baseline (Phase 1-disrupting) | sporadic | --all CPU contention amplified overhead; ibm03/04/09 regressed |
| Displacement-ranked multi-order on baseline-only | catastrophic | Displacement-sum NOT correlated with proxy. ibm10 +0.162; ibm12 INVALID (overlaps) |
| 2-opt-everywhere (every legalize) | sporadic | Disrupts cong-grad trajectory: ibm04 −0.0115 ✓ but ibm06 +0.0087 ✗, ibm02 +0.0015 ✗ |
| Multi-frac Phase 3 (fracs 0.02/0.04/0.06) | neutral | f=0.04 already optimal on tested benchmarks |
| WireMask-BBO + congestion penalty α=30 | sporadic | Helps sparse (ibm01 −0.029) but hurts dense (ibm04 +0.097, ibm06 +0.169). Constructive placer abandons initial.plc's good seed |
| `plc.optimize_stdcells` post-pass | dead | 130s on smallest benchmark (ibm01) AND +0.13 regression with default FD params |

Cleanup performed: removed `surrogate.py`, `_calibration_test.py`, `_path3_incremental_test.py` (all rejected experiments); removed `_wiremask_place`, `_build_wm_net_cache`, `_density_gradient_perturb`, `_congestion_heatmap`, `_box_blur` from `placer.py` (all dead code on IBM benchmarks). placer.py: 1159 → 894 lines.

## Result vs `sameer_v1` / v11

| Benchmark | v11 (PROGRESS.md) | v12/v14 (--all stable) | Δ | Notes |
|---|---|---|---|---|
| ibm01 | 1.1854 | 1.1860 | +0.0006 | v11 was lucky outlier |
| ibm02 | 1.5823 | 1.5923 | +0.0100 | v11 was lucky outlier (stale-plc lottery) |
| ibm03 | 1.3547 | 1.3603 | +0.0056 | v11 was lucky outlier |
| **ibm04** | 1.3390 | **1.3316** | **−0.0074** | clean rediscovery + budget fix makes it stable |
| **ibm06** | 1.6797 | **1.6684** | **−0.0113** | clean CPU rediscovery |
| **ibm07** | 1.4950 | **1.4924** | **−0.0026** | 1% noise restart wins |
| ibm08 | 1.5251 | 1.5251 | 0.0000 | |
| ibm09 | 1.1304 | 1.1304 | 0.0000 | |
| ibm10–17 large/baseline-only | unchanged | unchanged | ~0.0000 | 2-opt nudges most by −0.0001 to −0.0006 |
| **ibm18** | 1.7941 | **1.7896** | **−0.0045** | threshold change re-includes; cong-grad iter=1 wins |

**17-benchmark --all average:**

| Run | Avg | Notes |
|---|---|---|
| PROGRESS.md v11 estimate | 1.4860 | composite, never actually --all'd |
| PROGRESS.md v10b actual --all | 1.4877 | 2026-04-30 historical |
| **v12 --all stable (with budget fix)** | **1.4854** | reproducible across runs |
| v14 with Tier 3 + 2-opt + running-max (clean, no DREAMPlace) | ≈1.4854 | speed-only improvements; gap to RePlAce unchanged |

**Gap to RePlAce (1.4578):** −2.2% / 0.028. Not closeable by further tuning of this restart pipeline — confirmed by today's exhaustive search across multi-order/2-opt-everywhere/WireMask/multi-frac-Phase-3. Closing it requires a structural change. DREAMPlace async (in progress) is the only known path with proven sub-1.45 results on ICCAD04.

## File index

| File | Purpose | Carry to v2? |
|---|---|---|
| `placer.py` | **The submission.** ~894 lines: Tier 3 vectorized legalize, cong-grad Phase 1/2/3, noise tail, 2-opt post-pass on baseline-only, async DREAMPlace integration (Phase 5). | **Yes** — v2 builds on this. |
| `_ibm15_timing_test.py` | Scoring-time harness. Measures legalize + 3 scoring calls for any benchmark. Used to discover stale PROGRESS.md estimates that motivated v12's threshold change. | **Yes** — useful any time we re-verify scoring assumptions. |
| `PROGRESS.md` | Local copy of team's `/PROGRESS.md`, extended with v12-v14 entries. Source of truth for "what works"; consult before re-running rejected experiments. | **Yes** — keep updating. |
| `dreamplace_bridge/` | Working pb.txt ↔ Bookshelf conversion + DREAMPlace subprocess runner. Restored 2026-05-20 from commit 111f315 with new async wrapper (`AsyncDreamplaceHandle`, `launch_dreamplace_async`). Used by placer.py for the Phase 5 candidate. | **Yes** — central to the async DREAMPlace path. |
| `dreamplace_build/` | Local DREAMPlace install (~500MB build artifacts, gitignored). Rebuild via: `sudo apt install -y flex bison libboost-all-dev`; clone DREAMPlace; `cmake .. -DCMAKE_CXX_ABI=1 -DPython_EXECUTABLE=$(which python)`; `make -j2 install` (NOT `-j$(nproc)` — OOM); `sed -i 's/np\.string_/np.bytes_/g' install/dreamplace/PlaceDB.py` for NumPy 2.0 compat. | **No** — gitignored. Reproduce via steps above. |

### Removed 2026-05-20

These files used to live here but were deleted as part of the DREAMPlace prep cleanup:

| File | Purpose | Reason removed |
|---|---|---|
| `surrogate.py` | WL-only surrogate ranker for large benchmarks. Spearman +0.83/+0.94 vs real proxy on ibm11/ibm15 but ties near optimum broke wrong. | Never wired in; net zero/negative. Reachable via git if needed. |
| `_calibration_test.py` | Drove surrogate's weighting decision. Documents *why* the surrogate was rejected. | Documents a dead path. |
| `_path3_incremental_test.py` | Tested `plc.set_use_incremental_cost(True)`. Found incremental mode freezes density+congestion, only refreshes WL (anti-correlated). | One-off experiment, dead path. |

## v2 starting point

When starting v2:

1. **Copy** `placer.py`, `_ibm15_timing_test.py`, `PROGRESS.md`, `dreamplace_bridge/`. Carry the README forward as a template.
2. **Critical config to preserve**: `EXACT_MACRO_THRESHOLD=400`, `EXACT_GRID_CELL_LIMIT=2200`, `BUDGET_OVERRUN_S=60.0`. Don't drop without re-validating the wins they enable (especially ibm04 1.3316 + ibm18 1.7896).
3. **Critical code to preserve**: Tier 3 vectorized `_will_legalize` *and its float32 precision fix* (without that fix, ibm04 lands at 1.3364 instead of 1.3316). 2-opt post-pass on baseline-only (banks ~0.0001 to avg). Running-max `t_one_score` (defends against --all CPU contention).
4. **Highest-leverage open work** (per PROGRESS.md / today's session):
   - Finish async DREAMPlace integration (build + test). Only proven path to sub-1.45.
   - If async fails to overlap (depends on plc GIL release), pivot to:
     - WireMask outer BBO loop (the actual paper algorithm; we only tried greedy).
     - Custom force-directed soft macro re-placement in vectorized numpy (the academic `optimize_stdcells` is too slow at 130s/call).
5. **Do NOT retry without specific reason**: multi-order legalize (displacement uncorrelated with proxy), pure WireMask greedy (clusters), `optimize_stdcells` (too slow + regresses), continuous wire-pull approximation (always loses to cong-grad).
