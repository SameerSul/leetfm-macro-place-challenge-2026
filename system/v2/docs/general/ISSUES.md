# Open issues — v2 placer (last revised 2026-06-12)

This is a **clean rewrite**. All issues that have been resolved or
rejected have been removed; their findings are captured in commit
messages and in PROGRESS.md. This file now tracks **only what's
open**: known gaps in the current placer, speculative score ideas that
haven't been tried, and follow-up work that's been queued but not
started.

---

## Current state (headline)

| Metric | Value |
|---|---|
| Best `--all` avg | **1.1252** (2026-06-11 — S10 ML hard-relocation ranker connected as production default; 17/17 VALID, 0 overlaps, **2337s ~39min**). |
| Prior `--all` avg | 1.1272 (S16, DP basins restored) → 1.1379 (S14, **DP-OFF** — hand-JIT) → 1.1380 (S13) → 1.1403 (S12) → 1.1423 (S11) → 1.1500 (refactor) |
| RePlAce target | 1.4578 |
| **Gap to RePlAce** | **−22.8% (beat by 0.333 — beats on every benchmark)** |
| DREAMPlace leaderboard | 1.4076 (UT Austin) |
| **Gap to leaderboard** | **−20.1% (BEATS by 0.282)** |
| NG45 (Tier 2) avg | 0.7830 |
| `--all` wall-clock | 2337s (~39 min) in the 2026-06-11 re-baseline |

All 17 IBM benchmarks improved vs v12 baseline. The **relocation family** is the
dominant lever (R1 −0.0096, R2 −0.0083, R2b −0.0027, R3 −0.0452, **R5 −0.0965**
→ 1.2799), and a **bit-exact scoring-speedup stack** layered on top buys another
−0.0044: incremental congestion cost (1.2799→1.2767, isolated), idea #1
subset-cumsum, idea #2 topology-struct cache, floor-reservation allocator, and
the A+C round-3 cong cap + density `top_hot` boost (1.2767→1.2755 combined). The
throughline of the relocation family: softs are the bulk of BOTH the congestion
and density terms and were frozen at initial.plc by every prior placer;
relocating them (cong-targeted then density-targeted, interleaved with hard
reloc + 2-opt) is where the win lives.

**Budget margin (CLOSED 2026-05-29).** The `--all` 2026-05-29 #1+#2 run starved
ibm18 (cumulative monotonic reached 3228s of the 3300 internal cap by benchmark
16; the old guard's blunt `cumulative > 95%·3300` test returned baseline for
ibm18, costing ~0.23 on that benchmark / ~+0.013 on the average). Fix:
**floor-reservation allocator** — when allocating a benchmark, reserve
`(PER_BENCH_FLOOR_S=110 + BUDGET_OVERRUN_S=60) · (remaining_benchmarks − 1)` for
the others plus 60s of own-overrun slack, so the last benchmark always gets
≥110s; clamp the result to the 3540s hard-cap headroom; the old guard reduces to
`eff < 45s → baseline` (only fires when the headroom genuinely runs out).
Worst-case simulation (every benchmark overruns by 60s): all 17 get ≥110s,
cumulative lands at exactly 3300. Combined-stack `--all` confirmed ibm18 =
1.5787 (vs the starved 1.7941). The 3600s hard cap is structurally protected.

---

## Open issues

### S18. Cluster-coherent LSMC kicks — macro-hierarchy awareness (SHIPPED 2026-06-14)

**What:** the LSMC kick now optionally moves a derived connectivity *cluster*
as a unit instead of scattering random hard macros. Communities are inferred
from the netlist because the user asked to keep connected subsystems together.

**Key finding:** these flat ICCAD04 netlists have almost no hard-to-hard nets
(ibm01: **0** nets with ≥2 hard pins; ibm10: 4) — hard macros talk to standard
cells, which talk to other hard macros. So clusters are derived by union-find
over **low-fanout nets through the bipartite hard↔soft graph**
(`local_search/clusters.py`, cached on plc). Coverage is sparse: ~8–22% of hard
macros cluster, groups of 2–9, many spread >50% of the canvas diagonal.

**Kick modes** (`_cluster_kick` in `lsmc_explore.py`): `gather` (seed all
members at one anchor, legalizer packs them — directly tests "keep them
together"), `translate` (rigid relocate preserving arrangement), `both`
(per-kick random pick). The exact post-descent accept gate is unchanged, so a
cluster kick can never *commit* a worse placement — it only changes which basins
are explored. Kicks fall back to random when no cluster is available.

**Why it's safe vs the standing "clustering hurts congestion" warning:** the
disproven experiments (WireMask greedy, optimize_stdcells, net-centroid hard
bias) *forced* clustering. This only *proposes* it behind the exact gate; the
kept moves actually *reduce* congestion (it's the term that drops in every win).

**Evidence:** phase-isolation harness (`V2_LSMC_ISOLATE=1`, same incumbent /
seed / budget) — cluster kicks beat random **6/6** benchmarks, −0.0053 avg.
Paired multi-seed `--all` ON (p=1.0, both) vs OFF — **3/3 seeds**, mean
1.1206→1.1183 (−0.0023), 0 regressions, all 17/17 VALID. Seed 0 was a favorable
draw (−0.0054); steady-state ~−0.0008/seed. Shipped as default in `src/main.py`
(`_enable_cluster_kick_defaults`), overridable via `V2_GPU_EXPLORE_CLUSTER_P`
(0 disables), `V2_GPU_EXPLORE_CLUSTER_MODE`, `V2_CLUSTER_MAX_FANOUT`. Verified:
`test/verification/_verify_cluster_kick.py`.

**Open follow-ups (higher ceiling, not started):** cluster-outlier *relocation*
move (pull a macro far from its cluster centroid toward a cold region, gated by
proxy); feed derived clusters into the DREAMPlace seed as soft grouping. Both
are larger efforts; the marginal `--all` gain from kicks alone is small.

### S17. GPU / LSMC staged rollout — current work is generic multi-incumbent LSMC

**Current state (2026-06-14):** cong-grad phases have been deleted from the
active pipeline, and LSMC is the remaining GPU-backed final exploration layer.
The active LSMC expansion is intentionally generic: it seeds from legalized
baseline, random-noise restarts, random-order legalize trials, pre-R2 best, and
post-R2 best. It does **not** use DREAMPlace/bridge-specific placements as LSMC
seeds and does **not** use cong-grad-derived kicks or seed sources. The exact
post-descent accept gate remains the safety invariant.

**Next LSMC improvement methods:** update and gate one at a time:
- seed-pool calibration (`V2_GPU_EXPLORE_MAX_SEEDS`, `SEED_MARGIN`, and per-seed
  time allocation);
- generic geometry-only kick families (area-weighted picks, displacement-window
  kicks, edge-biased targets, group kicks);
- soft-aware hard-kick recovery using existing soft relocation;
- adaptive kick pre-screen based on score time and seed type;
- a small R2-as-descent experiment where acceptance still happens after the
  final exact score;
- chain-local non-greedy acceptance variants that remain final-gated against the
  global incumbent.

Lesson from the earlier pruning gates: judge changes on full 17-benchmark paired
runs, not prefix smokes. The final LSMC gate cannot itself accept a worse score,
but time displaced from R2/post-soft phases can still regress the final result.

### S17-prev. Stages 2a+2b (best --all 1.1176)

**Stage 2b (2026-06-13):** kick pre-screen `V2_GPU_EXPLORE_PRESCREEN` (default
8) — score a batch of kicks, descend the best (cuGenOpt evaluate-reduce at the
kick level). Paired gate 2/2: seed1 1.1198→1.1176, seed2 1.1237→1.1219, mean
−0.0020; accepts ~doubled; B8 slightly faster than B1. `PRESCREEN=1` = prior
2a behavior. Shipped (default already 8 in code).

**Stage 2c (multi-chain — PROBED, REFACTOR REJECTED 2026-06-13).** Hardware is
single-GPU-always, so "multi-chain" means batched chains on one device, not
islands. Built `V2_GPU_EXPLORE_CHAINS` scaffolding (single-process keep-best,
CHAINS=1 = verified no-op; commit fd6ceee, **merged as a dormant knob**, default
off). Diversity-vs-depth probe at matched 90s compute (3 chains vs 1) on 6
benchmarks: ibm12 −0.0095 (real), ibm04/09/11/15/16 between 0 and −0.0009
(noise floor). **The entire signal is one benchmark.** Extra chains add accepts
(1→3) but to equivalent basins. Conclusion: the GPU-batched-descent refactor
(batch the relocation scorer across a chain dim + per-chain commits, a multi-day
rewrite) is NOT justified by a ~1/17-benchmark payoff at 1.1176 with shrinking
increments. Budget-split multi-chain is also unshippable (10s/chain too shallow
at the 30s cap; 90s breaks the 1h `--all` cap). Knob left dormant for possible
revisit. NOT a candidate by itself: annealed acceptance (LAHC disproven). **Next
lever:** LSMC-only seed/kick/descent improvements from the current generic pool.

**Stage 2a verdict (2026-06-12 evening):** post-R2 LSMC kick/descent/accept
(`lsmc_explore.py`) shipped default-on under CUDA, kick=0.02, 30s slice.
Full-stack paired gate: seed1 −0.0051 (on-arm 1.1194 = NEW BEST), seed2
−0.0033. Design invariants discovered: the accept gate must be the final
quality phase (earlier hooks accepted states that lost after later
refinement), and worktree-pinned runs need gitignored assets symlinked in
(DP/ML silently off otherwise — invalidated the first gate attempt).
Remaining in this entry: Stage 0 hardware half and possible LSMC-only
experiments. The old island/multi-GPU framing is retired; target hardware is one
GPU, and extra chains mean either serial budget splits or a future one-device
batch dimension.

Plan of record: `docs/gpu/GPU-ops.md` (cuda_delta-based LSMC exploration on one
GPU, generic multi-incumbent scheduling, evidence-gated LSMC-only changes).

**Stage 0 (done 2026-06-11):** re-baseline avg **1.1243**, 17/17 VALID, 2679s
(noise-equivalent to the 1.1252 record). CUDA diagnostic PASS (parity 1.541e-07);
DREAMPlace dpenv healthy but **sm_89-only**; numba present. Open half: the new
multi-GPU machines aren't reachable from this box — on access, run GPU
inventory, rebuild DP for their arch, re-baseline, and re-run the Stage 1
winner with raised `V2_RELOC_PROPOSE_{MAX_MB,TOP_M}` / pool sizes.

**Stage 1 (done 2026-06-12):** paired multi-seed A/B of
`V2_RELOC_PROPOSE_ALL=auto` vs off, seeds 1/2/3
(`ml_data/compare/all_20260612_propall_*`): +0.0090 / +0.0047 / −0.0076
cumulative — mean +0.0020, 2/3 seeds worse → **stays opt-in** (S10 ship bar is
3/3 wins). No `--all` wall-time win: budget allocator reabsorbs per-benchmark
speedups. Divergences vs the CPU policy are deterministic (ibm18 seed1 +0.0188
replays bit-exact) but seed-dependent in sign — the GPU policy finds different
basins, not better ones, when single-candidate.

**Open → Stage 2 (`V2_GPU_EXPLORE`):** build the chain engine per GPU-ops.md
§2 (kick ≈0.10 → cuda_delta descent → post-descent zero-temp accept, batched
chains) + §3 handoff (spiral legalize → fresh `IncrementalScorer` → adaptive K
from `t_one_score`). The Stage 1 result sharpens the thesis: single-candidate
GPU policy is a wash, so the win must come from *many* candidates + the exact
gate harvesting the ±0.02 per-benchmark spread. Verification checklist in
GPU-ops.md §6 before any score run; gate vs Stage 1 via paired multi-seed.

### S16. Silent DREAMPlace ABI break — DP was dead since S13 (SHIPPED 2026-06-10 — 1.1379 → 1.1272)

The DP bridge (`src/dreamplace_bridge/run_bridge.py`) launched DREAMPlace with
`REPO_ROOT/.venv/bin/python`. The repo `.venv` was upgraded to **Python 3.14** for
numba (S13), but every DP compiled extension under `dreamplace_build/install` is
ABI-tagged **cpython-310** (built against `dreamplace_build/dpenv`, Python 3.10). So
`import dreamplace.ops.place_io.place_io_cpp` died with `ModuleNotFoundError` ~4s
after launch — and the harvest masked it as a benign **"not ready (elapsed=4.4s);
killing subprocess"** (the result-wait can't tell "still computing" from "already
exited non-zero"). Net: **DREAMPlace produced ZERO seed basins on every benchmark
from S13 onward**; the multi-seed 2-opt ran single-basin, and **the 1.1379 @2117s
(S14) headline was a DP-OFF run.**

**Fix (one spot, graceful):** `VENV_PYTHON` now prefers the DP build env's
interpreter (`dreamplace_build/dpenv/bin/python`, 3.10), falling back to `.venv`
only when dpenv is absent (e.g. a machine where DP was built in-place). The DP
subprocess already sets its own `PYTHONPATH`, so the parent stays on 3.14 and only
the DP child uses 3.10 — the documented "isolated envs" design.

**Result:** `--all` **1.1379 → 1.1272 (−0.0107)**, all 17 VALID / 0 overlaps,
**51/51 DP launches ready / 0 failures**, DP basins used (not pruned) in the 2-opt
on all 17. Runtime 2117 → 2645s (the +528s is DP candidate-scoring + DP-basin
2-opt; well under the 3300s soft cap). Confirms basin diversity is a real lever in
aggregate — but **only resolves above noise at the 17-benchmark average**: the
single-benchmark spot-check (ibm12 −0.006, ibm17 +0.004, ibm18 −0.006) read as
neutral/noise, one even regressing. **Follow-up:** DP basins are still mostly pruned
or lose the 2-opt selection (DP's raw proxy is congestion-blind, 1.7–3.0 vs the
cong-grad best ~1.65–1.79); the gain comes from the minority of benchmarks where
DP's WL/density basin 2-opts below best. More/different DP configs (S15's basin-
diversity idea) may still have headroom now that DP actually runs.

**LAHC (disproven, reverted 2026-06-10).** Tested Late-Acceptance Hill Climbing on
the 2-opt-on-winner to break the strict-greedy accept gate. Strictly worse on
ibm12/17/18 (ibm17 2-opt 1.7299→1.7401 at L=1000, →1.7328 at L=50 — tighter history
only recovers greedy, never beats it; ~85% accept rate = plateau random-walk). The
deadline-bound 2-opt converges fast to a strong basin min, leaving no headroom for
non-monotonic exploration (matches the S1 basin-hopping disproof). Reverted in full.

### R1. Congestion-directed relocation moves (SHIPPED 2026-05-27 — 1.4422 → 1.4326)

The single biggest lever of the session. The 2-opt search only EXCHANGES two
macros' positions — it can never relocate a routing-heavy macro into an empty
low-congestion gap (a swap would dump some other macro into the vacated hot
spot). R1 adds that missing move: a post-2-opt pass (`_relocation_moves`) that,
for the hottest macros (by live `max(H,V)` congestion), tries moving each into
the nearest lower-congestion legal cell centers, accepting only on a strict
true-proxy drop via the incremental scorer's new `score_move` (single-macro
analogue of `score_swap`; verified bit-exact ≤6e-9, no drift, in
`_verify_score_move.py`). Legality = in-bounds + no overlap with other HARD
macros (softs may overlap). The proxy gate filters far moves that spike WL.

**Result:** --all 1.4422 → **1.4326** (−0.0096), **ALL 17 improved** (ibm04
−0.034, ibm02 −0.026, ibm01 −0.018, ibm15 −0.016, ibm10/13 −0.011), gain in the
congestion term as designed, at ~0.1–0.2s/benchmark (~288 incremental score_move
calls). Strictly non-regressing by construction (best_pl only updates on a true
re-score improvement).

**Why it worked where DP1 didn't:** R1 relieves congestion with a DIRECT,
proxy-gated move on the placement we already have, rather than trying to fix
DREAMPlace's congestion-blind global placement (which trades away its wl/den edge,
DP1) or refine via swaps only (2-opt).

### R5. Soft DENSITY relocation (SHIPPED 2026-05-29 — 1.3764 → 1.2799, the dominant lever)

R3 relocated softs by the **congestion** field. R5 adds a second soft pass per
interleave round that relocates by the **density** field (softs in the densest
cells → low-density cells). Softs are the bulk of BOTH terms, and — since softs
may overlap — the cong pass can pile them into low-cong cells without relieving
density. The headroom measurement showed that on the (cong-converged) best_pl
the cong field finds **0** more moves but the density field finds **22–68**, for
−0.011 to −0.020, all in the density term. Implemented by adding `use_density` to
`_soft_relocation_moves` (build the hot/cold field from the scorer's occupancy
grid instead of the routing map) + `score_move_soft` already handles it; the R2
soft pass became a two-field loop (`cong` then `density`).

**Result:** --all 1.3764 → **1.2799** (−0.0965), ALL 17 improved (ibm13/02/08
each −0.122, ibm06 −0.120, ibm18 −0.214), all VALID / 0 overlaps, 2639s. The
interleave compounds it (single density pass −0.011/−0.020 on best_pl → −0.03 in
the loop → −0.097 across the full pipeline). Also folds in **R3b** (soft top_hot
48→128). Beats RePlAce by 12.2%, leaderboard by 9.1%.

**Open follow-up — budget margin / speedup** (see headline note + P5): fits at
2639s clean but ibm09 = 307s; the queued speedup is the **shared scorer** across
interleave passes (the `_profile_move.py` profile retired the earlier
"incremental `_compute_cong_cost`" plan — smoothing is only ~20% of a move).

### R4. WL-aware HARD relocation targeting (DISPROVEN 2026-05-29)

Probe of biasing hard-relocation targets toward each macro's net centroid (`wl_blend`
of distance-to-current vs distance-to-centroid) so cong relief costs less WL.
Post-hoc on best_pl was a no-op (hard relocation already converged → 0–2 moves);
the in-loop production A/B (`WL_AWARE=0.5`) was **slightly worse** (ibm03 +0.0015,
ibm07 +0.0025) — the centroid bias steers the greedy interleave to a worse local
min, no upside. Reverted the production gate and removed the probe scaffolding.
(Consistent
with O3's finding that the WL-centroid blend on *softs* gave ~0 — things sit near
their centroids already.)

### R3. Soft-macro relocation (SHIPPED 2026-05-28 — 1.4216 → 1.3764, BEATS leaderboard)

**The dominant lever of the whole effort — −0.0452, all 17 improved, and it put
v2 below the UT Austin DREAMPlace leaderboard (1.3764 < 1.4076).** Soft macros
(std-cell-cluster stand-ins) are the **bulk of the routing demand**, and every
prior placer froze them at `initial.plc`. R3 applies the R1 relocation move to
SOFT macros: relocate the hottest soft clusters (by live `max(H,V)`) into low-
congestion cells, accept-on-true-proxy via the scorer's new `score_move_soft`
(softs touch WL + net-routing congestion + density, NOT macro blockage — no
legality check since softs may overlap; verified bit-exact in
`_verify_score_move_soft.py`). Wired as a **third move type in the R2 interleave
loop** (hard reloc ⇄ soft reloc ⇄ 2-opt), so it compounds round-over-round.

**Result:** --all 1.4216 → **1.3764** (−0.0452), ALL 17 improved, gain in the
congestion term: ibm06 −0.102, ibm07 −0.080, ibm03 −0.067, ibm12/14 −0.062,
ibm17 −0.061. All VALID / 0 overlaps (softs are movable — 0 fixed softs on IBM,
confirmed; a `soft_movable` guard defends NG45/other inputs). 2350s. The
interleave makes the gain 2–4× a single soft pass (each soft move opens new
hard/2-opt moves).

**This corrects O3.** O3 closed soft-repositioning, but only tested *bulk* moves
(WL-centroid blends, gradient spreads). Discrete, proxy-gated, R1-style soft
relocation is a different operator and it is the biggest win we found.

**Follow-ups:** confirmation re-run (the −0.0452 jump is large; all 17 improved
far above the noise floor so it's robust, but a second --all is cheap insurance);
tune soft `top_hot`/`n_targets`; soft 2-opt swaps (exchange two soft clusters).

### R2. Interleaved relocation ⇄ 2-opt (SHIPPED 2026-05-27 — 1.4326 → 1.4243 → 1.4216)

R1 ran relocation once, after 2-opt. R2 ALTERNATES a relocation pass and a 2-opt
cleanup pass (≤6 rounds, budget-gated, break on no-improvement): each relocation
opens new swap opportunities and vice versa, so they compound. Both accept-on-
true-proxy → strictly non-regressing. Relocation runs first each round (the
multi-seed block already 2-opt-converged best_pl). --all 1.4326 → **1.4243**
(−0.0083), ALL 17 improved (ibm04 −0.043, ibm10 −0.022, ibm02 −0.015, ibm12
−0.011); per-benchmark the interleave roughly doubled R1's single-pass gain on
the high-leverage benchmarks (ibm04 walked 1.27→1.19 over 4–6 rounds, monotonic).

**R2b — widened relocation candidate set (top_hot 24→48, n_targets 12→16).**
"Squeeze R2" follow-up: raising the round cap (6→10) was tapped (rounds 7+ below
the noise floor), but the binding limit on large benchmarks was top_hot per
round — at 24 it covered only ~3% of ibm10's 786 macros/round. Widening to 48/16
relieves more hot macros per round (and converges in fewer rounds, so it's also
*faster* on the large benchmarks). --all 1.4243 → **1.4216** (−0.0027), all
improved-or-flat; broader than expected (ibm12 −0.008, ibm16 −0.0065, ibm11
−0.006, not just the largest). 1518s.

**Leverage analysis (`_reloc_leverage.py`):** gain correlates with hard-macro
utilization (canvas fraction occupied by hard macros) gated by congestion
headroom — NOT with macro dominance or open space (both hypotheses refuted:
ibm15 has the most-dominant macro but small gain; ibm18 has the most open space
but the smallest gain). Big gainers (ibm04/10/02/12) have util 0.42–0.60 + cong
above the floor; the two lowest-congestion benchmarks (ibm01/09) and the lowest-
hard-util ones (ibm17/18) barely move.

**Follow-ups:** soft-macro relocation (for the soft/net-dominated low-hard-util
benchmarks ibm17/18 — see option 2 in the alternatives menu); more rounds /
larger 2-opt slice on benchmarks that hit the round cap with budget to spare.

### DP1. Congestion-aware DREAMPlace — the leaderboard gap is pure congestion (CLOSED 2026-05-27 — routopt can't move the proxy)

**Diagnosis.** Our DREAMPlace (DP)
candidates lose to the cong-grad "best" seed 15/17. Decomposing why, on the
congestion-heavy benchmarks, shows the loss is **entirely congestion** —
DREAMPlace is *better* on wirelength and density (it optimizes those) and only
loses on the term it can't see:

| | wl | den | cong | proxy |
|---|----|----|------|-------|
| ibm10 raw dp[hi-fix] | 0.0574 | 0.3774 | **0.9543** | 1.3891 |
| ibm10 final best | 0.0636 | 0.3804 | **0.8904** | 1.3344 |
| Δ (dp − best) | −0.006 | −0.003 | **+0.064** | +0.055 |
| ibm12 raw dp[hi-fix] | 0.0626 | 0.3968 | **1.2497** | 1.7090 |
| ibm12 final best | 0.0608 | 0.4017 | **1.1749** | 1.6375 |
| Δ (dp − best) | +0.002 | −0.005 | **+0.075** | +0.071 |

**Post-hoc repair ruled out (mostly).** A ceiling test ran a generous ungated
cong-grad descent + 2-opt on the raw DP basin. ibm10 *did*
recover below best (1.3279 vs 1.3337) — but the production realization (Phase 7b)
was REVERTED: the descent is budget-hungry (~30s/bench), high-variance, and not
even reproducible at fixed seed (plc-state-dependent on pipeline position — seed
777 gave 1.3639 post-pipeline vs 1.3730 mid-pipeline). Captured zero net gain
in-pipeline. Relieving DP's congestion by moving macros *afterward* trades away
its wl/den edge as fast as it gains — the trade-off must be resolved *inside* the
global placement, not after.

**The lever: enable DREAMPlace's built-in routability optimization.** DREAMPlace
has `routability_opt_flag` + `adjust_rudy_area_flag` (params.json) — it computes
a RUDY/RISA routing-congestion map mid-placement and inflates node areas in
congested regions (≤`max_num_area_adjust`=3 times), so the density penalty
spreads cells out of routing hotspots. This is congestion *in the global
objective*. The current bridge no longer exposes this retired mode.

**Result: routopt CANNOT move the TILOS proxy congestion — CLOSED.** Enabling
DREAMPlace's `routability_opt_flag` required two fixes first: a config bug where
the routability keys were not applied, and a crash in
`PlaceObj.build_nctugr_congestion_map` (it needs per-layer
`unit_horizontal_capacities`, which are None for Bookshelf inputs — patched both
`dreamplace_src` and `dreamplace_build/install` to build the NCTUgr map only when
`adjust_nctugr_area_flag` is set; RUDY is used otherwise, so safe). With routopt
genuinely firing, on ibm10:

| config | proxy | cong |
|---|---|---|
| routopt OFF | 1.3891 | 0.9543 |
| ON, bins=64, default caps | 1.4109 | 0.9658 (worse) |
| ON, bins=grid(55×41), caps physical×{1,4,16,64} | 1.3891 (all) | 0.9543 (no effect) |

Across a 64× capacity sweep (both directions) + grid-matched route bins, routopt
is either a **no-op or a regression** — it never lowers the proxy congestion. Why:
routopt spreads *movable* cells out of RUDY hotspots, but with
`soft_macros_movable=False` the only movable objects are the hard macros (few,
large, density-dominated) so area inflation barely moves them; and when it does
engage (bins=64) RUDY relieves cells that aren't the TILOS proxy's hotspots
(RUDY ≠ TILOS congestion), with a density headwind. The 0.064 congestion gap to
best is **not closable** via the built-in routability opt.

**Removed as stale:** the bridge `routability_opt` knob, its calibration
parameters, and the one-off diagnostics. The negative result remains here so the
experiment is not repeated. v2 stayed at **1.4422**.

**Not pursued (low EV / big build):** `soft_macros_movable=True` + routopt (the
`hi-mov` base is already 1.92, far above best); a custom congestion penalty map
fed from the *TILOS* field rather than RUDY (higher ceiling, substantial
DREAMPlace-source build with a per-iteration feedback loop).


### O1. ibm09 / ibm13 small regressions vs the v2-combined baseline (RESOLVED 2026-05-25 — kept 3-DP)

**Status: 3-DP shipped.** `--all` avg 1.4475 → 1.4471 (−0.0004).
Adding `hi-fix` as a 3rd DP (target_density=0.85, soft_movable=False)
recovered the ibm09/ibm13 regressions:

| Bench | 2-DP | 3-DP | Δ |
|---|---|---|---|
| ibm09 | 1.1116 | **1.1035** | **−0.0081** ✓ |
| ibm13 | 1.3890 | **1.3828** | **−0.0062** ✓ |
| ibm08 | 1.5076 | 1.5019 | −0.0057 ✓ (bonus) |
| ibm17 | 1.7372 | 1.7359 | −0.0013 ✓ |
| ibm04 | 1.2797 | 1.2899 | **+0.0102** ⚠ (see O2) |
| ibm10 | 1.3378 | 1.3416 | +0.0038 |
| ibm15/ibm16 | (same) | +0.0007 each | small |

Net cumulative across 17: −0.0070, avg delta −0.0004. Wall-clock
+102s (526s → 628s) for the third DP. The +0.010 ibm04 regression
is a path-dependency issue tracked separately in O2.

Phase 7 RNG isolation (in commit adaf693) was a prerequisite for this
fix — the original 3-DP attempt 2026-05-24 had to be reverted because
adding a third Phase 7 chain caused rng_cong drift that regressed
ibm10 +0.036. Now with isolation, ibm10 only sees +0.004.

### O2. ibm04 path-dependency under multi-DP (RESOLVED 2026-05-25 — candidate #2 shipped)

**Status: multi-seed 2-opt shipped.** ibm04 1.2899 → **1.2797**
(−0.0102, fully recovering the 3-DP regression). `--all` avg
1.4471 → **1.4464**.

The fix is candidate #2 below (run the final 2-opt from each DP basin,
keep the global minimum). Two corrections to the original analysis,
both established empirically this session:

  - **The tags were muddled.** Real ibm04 DP proxies are lo-fix 1.3588,
    hi-mov 1.3210, hi-fix 1.3188. The 2-DP winner was **hi-mov** (1.3210),
    and the 3-DP hijacker was **hi-fix** (1.3188), beating hi-mov by only
    0.0022.
  - **Fix candidate #1 (margin gate) was DISPROVEN.** Adding a 0.005
    acceptance margin so hi-fix can't displace hi-mov as best_pl gave
    1.2913 — *worse* than 3-DP's 1.2899, and nowhere near 1.2797. The
    1.2797 was never a property of the best_pl seed; it was a property
    of the whole 2-DP configuration. Even with hi-mov kept as best_pl,
    the mere presence of the hi-fix candidate perturbs plc state (Phase
    5b uses the last-scored plc state) and adds a Phase 7 chain. So a
    best_pl gate alone (== S7) cannot reproduce the 2-DP trajectory.

**What shipped (candidate #2):** the final 2-opt now runs from `best_pl`
PLUS each DP candidate basin in `dp_placements`, keeping the lowest
result. hi-mov's basin 2-opts to 1.2797 even though it lost the best_pl
race. The win generalizes — ibm09 also improved (1.1035 → 1.1026, via
the dp[hi-fix] basin). Implementation notes:

  - **Selection is by a fresh `_exact_proxy`, never the
    IncrementalScorer's `final_score`.** The incremental WL drifts
    seed-dependently (ibm01 dp[lo-fix]: internal 1.1309 vs true 1.1506).
    A first cut that compared internal scores picked a phantom winner
    and regressed ibm01 1.1317 → 1.1506. Re-scoring each finalist
    exactly fixed it (and incidentally cleaned up the cross-seed plc
    state leakage). The change is strictly additive: the `best` seed
    reproduces the committed single-seed 2-opt, and a seed is kept only
    if its true proxy beats the true-scored incumbent.
  - **Pruning (`DP_SEED_2OPT_WINDOW = 0.02`):** a DP seed whose raw
    proxy is > 0.02 above best_score can't catch up (max observed 2-opt
    gain ~0.04; both wins sit at +0.011 / +0.002), so it's skipped. This
    is provably score-neutral and cut `--all` wall-clock from ~1198s
    (no prune) back to ~722s (committed 3-DP was ~628s). 35 seeds pruned
    across the suite.

**Remaining (not pursued):** candidate #3 (full per-DP plc-state +
best_pl isolation) would let the pipeline reproduce each DP's standalone
trajectory, possibly squeezing a bit more, but it's much more invasive
and the cheap candidate #2 already recovered the regression.

### O3. Soft-macro repositioning (CLOSED 2026-05-26 — confirmed dead lever)

**Status: closed, no headroom.** Soft macros stay at `initial.plc`
throughout the non-DP pipeline; the earlier estimate was ~0.01-0.02 of
recoverable proxy. A measure-first investigation closed it: `initial.plc` soft
positions sit at a robust local proxy optimum, and every repositioning
method tested makes proxy equal-or-worse.

| Method (probe) | targets | result on stale-soft benches |
|---|---|---|
| WL net-centroid blend (a sweep) | wirelength | best ~−0.002 (a≈0.05), often 0 |
| congestion-gradient bulk soft move | congestion | strictly worse |
| density-spread bulk soft move | density | strictly worse |

Why: wirelength is only ~5% of proxy and the entire soft-WL swing is
~0.005; the dominant density+congestion terms are driven by HARD
placement + net routing, not soft positions. Clustering softs (WL min)
spikes density; spreading them spikes WL + congestion; moving them down
the congestion gradient just relocates congestion. The `initial.plc`
spread (from the prior EDA flow) already balances all three.

Seed analysis (`--all` run4): 15/17 win via the `best` seed and 4 large
benches (ibm08/10/12/16) have NO DP candidate → their softs are
definitely `initial.plc` — yet even those showed zero headroom. So this
isn't a "softs happen to be good on DP benches" artifact; it's structural.

**Do not revisit** without a fundamentally different objective (e.g. a
soft model that DREAMPlace's density-aware NLP optimizes jointly with
hard — but that's the DP path we already have, and DP only wins 2/17).

### O4. The pre-flight skip guard occasionally fires on benign WSL2 clock drift

`time.monotonic()` covers the inside-the-placer paths but the
harness's own `time.time()` reporting (in `evaluate.py`) still
occasionally shows wall-clocks of 36000+ seconds. The harness's
3600s cap uses host wall-clock, so a single Windows-host suspend
during a real submission run could blow the cap.

**Mitigation options:**
- Wrap the harness call with a wrapper that uses a Linux clock that
  pauses during suspend (e.g., `CLOCK_MONOTONIC` not just for the
  placer but for the cap timer).
- File a bug / patch against the harness.
- Run inside a container that has reliable wall-clock under suspend.

Not blocking; submission should run on a non-WSL Linux box where this
doesn't manifest.

### O5. IncrementalScorer relies on clean plc state at init (RESOLVED 2026-05-26)

**Status: fixed.** `IncrementalScorer.__init__` now sets
`plc._last_pos_cache = None` before `_fast_set_placement`, forcing a full
re-set of every macro. After the fix every seed's internal `final_score`
equals the true `_exact_proxy` (`incr==true` across the spot set), so the
seed-dependent drift is structurally gone, not just worked around. The
multi-seed path's true-rescore selection is retained as defence-in-depth.
Root cause and original analysis below.

**Surfaced 2026-05-25 during O2 candidate #2.** `IncrementalScorer.__init__`
calls `_fast_set_placement(plc, current_placement_np)`, which is
"idempotent if positions match `last_pos_cache`". When a second scorer is
built right after a prior 2-opt has mutated plc (the multi-seed case),
the idempotency cache can skip setting some positions, so the WL baseline
(`_compute_per_net_hpwl_full`) is computed against a mismatched plc state.
Result: the scorer's `final_score` drifts from the true `_exact_proxy`
(ibm01 dp[lo-fix]: internal 1.1309 vs true 1.1506).

**Currently mitigated, not fixed.** O2's multi-seed path works around it by
calling `_exact_proxy` (a clean full set) between seeds and selecting on
the true proxy, never the internal score. The single-seed path was never
affected (one scorer, built from a clean-enough state — ibm01 matched).

**Robust fix (~5 lines, defensive):** force a full placement set in
`IncrementalScorer.__init__` (bypass / invalidate the idempotency cache)
so the scorer's baseline is always self-consistent regardless of prior plc
state. Removes the implicit "caller must hand me a clean plc" contract and
unblocks any future code that builds multiple scorers.

**Risk:** low. Worst case is one redundant full set (~ms) at init.

---

## Speculative score improvements (not started)

### S1. Basin-hopping 2-opt — cong-grad kick between passes (DISPROVEN 2026-05-26 — kept dormant)

**Result:** enabling sliced basin-hopping (5s passes + cong-grad kick,
`S1_MAX_KICKS=2`) on top of P3 regressed `--all`: 6/7 benchmarks worse, 1 tie,
0 better before the run was stopped (ibm01 1.1269→1.1306, ibm04 1.2686→1.2777,
ibm08 1.4978→1.5023; cumulative +0.025 over 7). **Slicing the 15s into 5s
passes starves the productive deadline-bound 2-opt search**, and the kicks
perturb away from the optimum without recovering. The "more accepts" signal
that looked promising on a single ibm04 run (671→1072) was misleading — the
extra accepts were repairing kick damage, not net-improving; and the one
ibm04=1.2293 run was a lucky noise draw (ibm04 swings ~0.05 run-to-run).
Even ibm01, which converges early (where S1 *should* help), regressed.
**Kept dormant** (`S1_MAX_KICKS=0` = single full-15s pass); code retained for
reference. A gentler non-sliced variant (full-deadline pass, kick only with
leftover budget after early convergence) is low-EV: it fires only on small
benchmarks with ~1-2s to spare and never on the large average-movers.

**Original idea (for the record):**

**Idea:** 2-opt only PERMUTES existing macro slots — it can never reach a
position no macro occupies. After a pass converges to a swap-only local
min, inject a `_routing_congestion_perturb` KICK (continuous move of the
hottest macros against the live congestion field), legalize, and run 2-opt
again to clean up. Accept-on-true-proxy, keeping the running best across
passes. Per seed: up to `S1_MAX_KICKS+1` passes of `S1_PASS_BUDGET`=5s each
within the same 15s/seed envelope.

**Implemented** in the multi-seed 2-opt loop (basin-hop while-loop, RNG-
isolated via a local RandomState). Currently `S1_MAX_KICKS=0` (DORMANT) =
single full-15s pass = byte-identical to the committed single-pass code, so
P3 can be measured without S1 confounding.

**Key finding (2026-05-26):** the original "kick only on early convergence"
trigger never fired — at k=20/iters=6 the 2-opt is deadline-bound (uses the
full 15s without converging) on *every* benchmark, even small ones. So a
full-deadline pass leaves no budget to kick. Two consequences:
  1. Slicing (5s passes) is required to make kicks fire. An early sliced test
     on ibm04 showed the kick genuinely surfaces NEW improving swaps (accepts
     671→1072), but single-benchmark proxy is too noisy to judge (ibm04 swings
     ~0.05 run-to-run because the deadline-bound greedy path is CPU-load
     sensitive — 1.2293 vs 1.2846 on identical-algorithm reruns).
  2. **P3 changes the regime:** with ~25% faster scoring, small/mid benchmarks
     now converge before 15s (ibm04 12.8–14.3s), freeing budget for kicks.
     So S1 should be re-enabled (`S1_MAX_KICKS=2`) and --all-tested *after*
     P3 lands — it's P3 that makes S1 viable.

**Cost:** ~60 lines (shipped). RNG-isolated.
**Expected gain:** unknown until tested on top of P3; the accept-count jump
is suggestive but noise-dominated at the single-benchmark level.

### S2. Wider 2-opt k_neighbors (SHIPPED 2026-05-26 — k=20)

k_neighbors 10 → 15 → 20 in the multi-seed 2-opt-on-winner.
  - k=10 → 15: all 17 improved, avg 1.4464 → 1.4443 (−0.0021).
  - k=15 → 20: avg 1.4443 → **1.4435** (−0.0008), 15/17 improved. The two
    regressions (ibm13 +0.0004, ibm14 +0.0003) are deadline-bound — wider
    k means fewer total passes fit the 15s budget on large benchmarks — but
    noise-level and outweighed by broad small gains (ibm04 −0.0040).
Wall-clock ~826s. k=25+ not pursued: the deadline-bound regime is
expanding, so further widening likely hurts large benchmarks more than it
helps small ones. An adaptive-k (wider on fast benchmarks) is the next
lever if this is revisited. (Update: see S11 — the *R2-cleanup* 2-opt k was
later cut 20→16 to free scoring time; this 2-opt-on-winner pass stays k=20.)

### S11. Scoring-cost reduction — WL-delta prefilters + R2-cleanup k (SHIPPED 2026-06-06, avg 1.1423)

Per-operator profiling on ibm13 showed **hard_2opt eats ~48% of scoring time for
the smallest per-move gains** (median 2.8e-6, ~20–50× below the soft operators);
**soft_relocation (28%) is the score MVP**. Three cheap, accept-gate-safe cuts
(only change which candidates get exact-scored — every accept still validated).
Full `--all`: **1.1423, all 17 VALID, 0 overlaps, 3434s** — new best (prior 1.1500;
1.1500 → 1.1496 2-opt-only → 1.1423 with soft-reloc). Freed budget converts to
deeper refinement on the deadline-bound benchmarks.

0. **soft_relocation WL-delta prefilter = 1e-4** (the biggest win) — skips ~37% of
   `_trial_at_soft` calls (~10% of total scoring time). soft_relocation commits the
   best candidate per group, so skipping non-best improving candidates is free —
   the calibrated per-candidate "loss" (7.1% at 1e-4) massively over-counts. ibm15
   replicates: **1.2136 vs 1.2219 off (−0.008, and faster)**; ibm13 no regression at
   any threshold. New bit-exact `wl_delta_move_soft` (verified
   `_verify_wl_delta_move_soft.py`). Env `SOFT_RELOC_WL_PREFILTER`.

1. **soft_2opt WL-delta prefilter 0.01 → 3e-4.** The 0.01 default rejected
   *nothing* on ibm13 (no soft swap's wl_delta exceeds it). Calibrated via
   `test/diagnostic/_calibrate_wl_prefilter.py`: 3e-4 skips ~23% of
   `score_swap_soft` calls, drops only ~0.2% of improving swaps. Env override
   `SOFT_2OPT_WL_PREFILTER`.
2. **R2-cleanup hard_2opt k_neighbors 20 → 16** (the per-round 2-opt pass). Fewer
   spatial-kNN candidates → less scoring → freed time for the productive soft
   passes (the reallocation thesis; helps the budget-bound large benchmarks where
   S2 noted wider-k hurts by fitting fewer passes). The multi-seed
   *2-opt-on-winner* stays k=20 (S2). Env override `HARD_2OPT_K`.

A WL-delta prefilter was also added for hard_2opt (`wl_delta_swap`, bit-exact —
`test/verification/_verify_wl_delta_swap.py`, Δ≤1e-18, zero side-effects) but
**shipped OFF**: calibration showed hard spatial-kNN swaps have tiny WL deltas
(improving max 1.5e-4), so any safe threshold rejects <0.5% while adding wl_delta
cost to all candidates — net-negative. The method + `HARD_2OPT_WL_PREFILTER` knob
+ a `wl_delta` trace feature remain for experiments.

**Validation (spot-check, no env):** ibm13 −0.008 (1.0341 → 1.0259) and faster
(227 vs 240 s); ibm15 within its noise band (~1.224 ship vs ~1.220 baseline;
ibm15 single-run noise ±0.008). hardk12 was rejected (clear ibm15 regression
+0.03 → **k=16 is the safe value**). The 2-opt-only `--all` was 1.1496; with the
soft-relocation prefilter (item 0 above) the combined headline is **1.1423**.
(A `--all` ibm01 wall-clock of 29,795 s seen during this work was a machine-suspend
artifact — `monotonic()` counted sleep; solo ibm01 re-runs at ~137 s.)

**Invariant (do not let the GNN/propose-all work overshadow this).** These cuts
live in the *sequential* prep→trial path. The Phase-C propose-all / CUDA-batch
relocation path (`V2_RELOC_PROPOSE_ALL`, currently hard-only + default off) replaces
that loop and bypasses the prefilters, so it must stay opt-in until it beats the
prefiltered CPU default (now 1.1403) on the deadline-bound IBM benchmarks — see
constraint 6 in `../ml_nn/04-gnn-routing-fill-surrogate.md`. `_soft_relocation_moves`
has no propose-all branch today, so the soft prefilter (the biggest win) is always
active on the default path; keep it so.

### S15. Spending the numba-freed cap headroom — budget DEAD, width NEUTRAL (2026-06-07)

The S13/S14 speedups cut `--all` to ~35 min (vs the 1 h cap), freeing ~1400 s of
slack. Two attempts to convert it to score, both negative:
- **Raise per-benchmark budget (`V2_TIME_BUDGET`): no effect.** The "budget-bound"
  benchmarks actually **converge** ~200–235 s (the numba speedup already let them
  reach convergence — that's what drove 1.1403→1.1379). ibm13 control is bit-identical
  at budget 150 vs 350; runtimes don't scale with budget; proxy variation is
  restart-RNG noise. *Time is no longer the constraint — the reachable move set is.*
- **Wider exploration (`HARD_2OPT_K=20` + `V2_SOFT_TGT=40`): net wash.** Single-
  benchmark sweeps looked promising (ibm12 tgt40 −0.022) but were RNG-noise: the full
  `--all` is **1.1376 vs 1.1379** (−0.0003, slower), just shuffling per-benchmark
  wins/losses (ibm13 −0.018 but ibm08 +0.016). A single global width can't win
  everywhere (no per-benchmark branching). Not shipped; defaults stay tgt32/k16.

**Conclusion: at the practical floor for this move set on IBM.** Budget and width
are exhausted; further gains need generic LSMC seed/kick/descent improvements or
new move types — bigger bets with diminishing IBM return (we already beat the
leaderboard 1.4076 by 19%). Env knobs `V2_TIME_BUDGET` / `V2_SOFT_TGT` /
`HARD_2OPT_K` kept for future experiments.

### S14. Hand-JIT the post-numba scoring hot paths (2026-06-07, --all 2563s→2117s)

After S13 (numba on), cProfile on ibm13 showed three vectorized-numpy scoring
functions with no JIT path dominating: `_apply_macro_routing` (22.8s tottime, the
per-cell macro routing scatter), `_macro_occ` (14.4s, density footprint), and
`_compute_per_net_hpwl_subset` (12.2s, per-net HPWL). Wrote explicit-loop numba
versions of each (`_apply_macro_routing_scatter_jit`, `_macro_occ_jit`,
`_hpwl_subset_jit`), matching numpy's accumulation order → **bit-exact** (stress
verifier Hcong/Vcong ~1e-15, density Δ=0, swap Δ=0; score_move Δ≤1e-9).

`--all` **2563s → 2117s (~17% faster; ~39% vs the no-numba 3486s), ~35 min** — a
big cap-safety margin under the 1 h limit. Avg unchanged at **1.1379** (bit-exact,
so pure speed). ibm13 trajectory: 200s (no-numba) → 162s (numba) → 130s (+macro
routing JIT) → 119s (+all 3). Remaining profiled chunk: `_resmooth_h_cols/_v_rows`
(~10s, cumsum-based — numba won't beat numpy's C cumsum, so deprioritized) and
`get_ref_node_id` (TILOS plc_client, external/read-only).

### S13. numba JIT was silently disabled — re-enabled (2026-06-07, avg 1.1380, ~26% faster)

cProfile on ibm13 found the routing-apply (`_apply_net_routing_struct`, ~114 s
cumtime — half the run) running the **numpy fallbacks** (`_apply_3pin_routing_vec_numpy`
etc.), not the JIT paths. Root cause: **numba was not installed** (`HAS_NUMBA=False`).
numba is declared in `v2/requirements.txt` (`numba>=0.59`) but **not** in
`pyproject.toml`, so `uv sync` alone never installs it, and `config.py` falls back
to numpy **silently**. So every measurement this session (incl. the 1.1403 headline)
ran ~25 % slower than intended.

Fix: install numba (0.65.1 resolves on py3.14). `--all` then drops **3486 s → 2563 s
(~26 % faster)** and the avg improves **1.1403 → 1.1380** (the freed wall-clock budget
converts to more rounds on the deadline-bound benchmarks). Per-move the routing-apply
JIT is ~3–5× the numpy path (`ARCHITECTURE.md` §5.3).

**Impact + risk.** Without numba the placer still runs (graceful fallback) but ~25 %
slower → `--all` ~58 min, *near the 1 h cap*, and avg 1.1403. So numba is both a
score lever (−0.0023) and a cap-safety margin. `config.py` now emits a warning when
numba is missing. **The eval environment must install `v2/requirements.txt`** (or
numba must reach the `uv sync` path) to realize 1.1380. Other post-JIT hot spots
(cProfile): `np.unique`/`_unique1d` (~10 s, 1.5 M calls in the subset-cumsum
strip-batch) and `get_ref_node_id` (TILOS plc_client, ~8 s) — next CPU candidates.

### S12. Spend the S11 freed budget + adaptive budget control (2026-06-07, avg 1.1403)

S11 freed ~15–20 % of scoring time. Two follow-ups to spend it well:

**SHIPPED — soft_relocation `n_targets` 24 → 32.** Each soft-reloc group is ~37 %
cheaper post-prefilter, so the freed budget buys more per-macro target depth on the
score MVP. `--all` **1.1403** (17/17 VALID). Per-benchmark: ibm13 −0.012, ibm17
−0.0054, ibm15 neutral. Widening `top_hot` too (128→192) **over-widens** — worse on
ibm13 + ibm15 and finishes early (under-uses budget), so only `n_targets` moved.
Env `V2_SOFT_TGT` / `V2_SOFT_HOT` / `V2_SOFT_HOT_BOOSTED`.

**SHELVED (negative) — adaptive per-pass budget control.** Tracked each pass's
cumulative yield (proxy gain / budget-second) and scaled its deadline cap by
`clamp(yield/mean, lo, hi)`. Both full-adaptive (`[0.4, 2.5]`, boost+shrink) and
boost-only (`[1.0, 2.5]`) were consistently **worse** on deadline-bound ibm13/15/17
(+0.002 to +0.008): the shrink path makes `round_improved` flip false sooner →
early termination → worse basin; the boost path saturates without using the extra
time. The static caps + `skip-if-empty` are already a near-optimal allocation —
**the budget allocation isn't the lever; the moves are.** Kept env-gated
(`V2_ADAPTIVE_BUDGET` / `V2_ADAPTIVE_LO` / `V2_ADAPTIVE_HI`, default off, zero
overhead when off — the timing/gain bookkeeping is guarded) for future iteration.

### S3. Phase 8 with extended TOP-K set ({3, 5, 7, 10, 15, 20, 30, 50})

Currently k ∈ {5, 10, 20}. Some benchmarks may benefit from finer
gradations.

**Cost:** ~10 lines (extend the for-loop).
**Risk:** budget displacement.

### S4. 2-opt from multiple seed placements (PARTIALLY SHIPPED — see O2)

The multi-seed 2-opt framework shipped 2026-05-25 (O2 candidate #2):
2-opt now runs from best_pl + each DP candidate basin, with true-proxy
selection and window-0.02 pruning. Remaining cheap extensions, now that
the framework + `twoopt_seeds` list exist (each is ~1-2 lines):
  - **`baseline_pos` as a seed** — catches benchmarks where the refined
    best_pl landed in a worse basin than the raw legalized baseline.
  - **top-K noise restarts as seeds** — requires tracking the best few
    noise placements (more state); defer unless baseline_pos pays off.

**Cost:** baseline_pos ~2 lines; noise restarts moderate.
**Expected gain:** −0.001 to −0.010 (speculative); pruning keeps the
cost near-zero on benchmarks where these can't win.

### S5. Cong-grad with adaptive frac per cell

Currently `frac=0.04` is a single global parameter. Per-cell
adaptive perturbation magnitude based on local congestion ratio
could provide more targeted moves.

**Cost:** modify `_routing_congestion_perturb`.
**Expected gain:** small, depends on whether the simple linear
`move_scale = scale * local_cong` (already present) captures most
of the benefit.

### S6. Phase 7 starting from best_pl alternatives

Currently Phase 7 chains start from each DP placement. Phase 8 chains
start from best_pl. Could try Phase 7 starting from:
  - Each DP candidate (current).
  - Each noise restart in the top-K by score.
  - baseline_pos with cong-grad applied N times beforehand.

**Cost:** moderate.
**Risk:** budget displacement.

### S7. Acceptance criterion for DP candidates (DISPROVEN 2026-05-25 — see O2)

Tested: a 0.005 best_pl acceptance margin on ibm04 gave 1.2913, *worse*
than 3-DP's 1.2899. The path-dependency isn't in the seed choice (plc
state + Phase 7 chain count also shift), so a best_pl gate can't help.
Superseded by O2's candidate #2 (multi-seed 2-opt), which shipped.

### S8. Phase 9 random-order: increase trial count

Currently N=3 random-tiebreak trials. With B3p4 + 2-opt widening,
budget is freed; more trials might find better legalizations on
benchmarks with many same-area macros.

**Cost:** 1 line.
**Expected gain:** small.

### S9. Congestion-aware 2-opt candidate selection (SHIPPED 2026-05-26 — 1.4424 → 1.4422)

Two layered changes inside `_two_opt_proxy_swap`, gated on a `macro_cong`
(per-macro local `max(H,V)` snapshot taken at seed time):
  - **Variant 1 — hot-first outer ordering.** Iterate macros by descending
    local congestion instead of by index. On deadline-bound benchmarks the
    swaps evaluated before the budget expires are then the hotspot ones —
    the dominant proxy term. Pure budget reallocation (can't beat the
    deadline-free convergence point).
  - **Variant 2 — cold-region teleport augmentation.** Spatial kNN can only
    swap nearby macros, so a routing-heavy macro can never relocate across
    the chip (intermediate local swaps all reject). For the `cong_hot_k`=20
    hottest macros, append the `cong_cold_k`=8 coldest as extra candidates —
    a long-range edge that expands the reachable placement set. Size-
    incompatible teleports fail the free conflict check before scoring.

The proxy gate validates every swap, so this only changes WHICH candidates
are tried, never accepts a worse placement. `macro_cong=None` reproduces the
prior index-order / spatial-only behavior exactly.

**Result:** --all 1.4424 → **1.4422** (−0.0002). 12/17 improved, 5 slightly
worse, cumulative −0.0042 (ibm06 −0.0023, ibm14 −0.0011 the standouts; ibm01
+0.0015 the worst). 12/17 same-direction is ≈7% by chance, so likely-real but
**marginal — edge-of-noise.** All 17 VALID / 0 overlaps; teleports confirmed
firing (ibm10 accepts 1168→1327). Theoretically the higher-ceiling of the two
candidate-selection variants (expands reachability vs reorders a fixed set);
kept because it's net-positive, consistent-direction, and correctness-safe.

**Theory note (vs S1):** unlike S1 (which sliced the budget and starved the
search → regressed), S9 keeps the full pass and only changes candidate choice
— every accepted teleport strictly lowers proxy, so no budget-waste damage.

---

### S10. ML candidate ranker — per-operator XGBoost (SHIPPED AS DEFAULT 2026-06-11; equal-budget compare 2026-06-05 was comparable-or-better)

**Equal-budget head-to-head (2026-06-05).** Compared the wired `hard_relocation`
filter against the production interleave at *equal scoring budget*: config A =
production narrow-16 (no ML); config B = filter (`ML_HARD_RELOCATION_N_TARGETS=32`
generates a wide-32 pool, `ML_FILTER_TOP_K=16` exact-scores the model's best 16).
Both score ~16 candidates/group, so this isolates "model's 16-of-32 vs heuristic's
nearest-16." Model used: `ml_data/models/clean-wide32-holdout-ibm13-001`
(hard_relocation only). Fresh single-benchmark runs, interleaved A/B to control
machine drift. Raw logs in `ml_data/compare/`.

10-benchmark Δ (filter − interleave; negative = filter better): ibm01 +0.0010,
ibm09 +0.0029, ibm10 **−0.0221**, ibm11 **−0.0092**, ibm12 +0.0052, ibm13
**−0.0048**, ibm14 −0.0008, ibm15 +0.0197, ibm17 −0.0015, ibm18 +0.0072. Net
**−0.0024 / 10**.

**Variance correction.** ibm10 and ibm15 (the two big movers) were re-run 2×
each. ibm10 is a **robust win** (Δ −0.0221/−0.0084/−0.0216, filter wins every
rep). ibm15's +0.0197 was **mostly timing noise** (re-runs −0.0013 / +0.0085;
interleave swings 1.2175–1.2317, filter stable ~1.2304) — true gap ≈ +0.009, one
rep flipped to a filter win. Corrected net ≈ **−0.008 / 10**.

**Conclusion.** At equal budget the filter is **comparable-or-better** than the
exhaustive interleave. Robust wins (ibm10/ibm11/ibm13) concentrate on the
budget-bound benchmarks, exactly as the design predicted; everything else sits
within the ±0.005–0.01 run-to-run timing noise floor, and **no benchmark robustly
regresses** (the worst-looking, ibm15, collapsed under repetition). This did NOT
require retraining: **`best_recall@16` ≈ 1.0 on every benchmark** (ibm11 0.9987,
ibm12 1.0, ibm13 0.9984 — see `test/diagnostic/_filter_recall_by_benchmark.py`),
so the model almost never drops the true-best move; the per-benchmark swings are a
budget/diversity *trajectory* effect, not a ranking-quality one. The remaining
upside lever is **budget-aware pruning** (prune only under time pressure, score
the full wide-32 pool when the search is converging early with budget to spare) —
NOT a better ranker. The earlier "filter regresses ibm11" reading was a baseline
artifact: it had compared the filter against `wide32_nofilter`, which scores all
32 (more budget than the filter's 16), not against the equal-budget interleave.

**Status (2026-06-11): shipped as production default.** `src/main.py` enables
config B (`ML_HARD_RELOCATION_N_TARGETS=32`, `ML_FILTER_TOP_K=16`, model
`clean-wide32-holdout-ibm13-001`) whenever no `ML_*` env var is set and the
model artifact + `xgboost` are present; otherwise the pure-heuristic narrow-16
path runs unchanged. Any preset `ML_*` var (including `ML_FILTER_OPERATORS=""`
as an explicit opt-out) skips the defaults so trace/shadow/sweep workflows keep
their exact semantics. Wiring verified by
`test/verification/_verify_ml_filter_wiring.py` and an ibm01 end-to-end run
(`R2 hard relocation ML filter on (pool=32, top_k=16)` in the log; proxy
0.9146, VALID, 71s, reproduced twice). First `--all` rep with the default on:
**avg 1.1252, 17/17 VALID, 0 overlaps, 2337s** (per-benchmark table in
PROGRESS.md 2026-06-11). **Acceptance gate met same day via paired multi-seed
`--all`**: 3 seeds × (ON, OFF) run sequentially under same-day machine
conditions — Δ(ON−OFF) = −0.0051 (default seed), −0.0044 (s43), −0.0029
(s44); mean **−0.0041, filter wins 3/3 pairs**, all 6 runs 17/17 VALID /
0 overlaps. ON mean 1.1245 vs OFF mean 1.1286. Logs:
`ml_data/compare/all_20260611_{on,off}_s{def,43,44}.log`. Methodology note:
day-to-day drift (prior-day OFF reference 1.1272 vs same-day OFF 1.1303)
exceeds the effect size, so unpaired cross-day comparisons cannot resolve
filter-sized deltas — use paired same-day runs for all future filter
experiments. NG45 spot-check remains open but expected ~neutral
(hard_relocation is near-idle on NG45, see the 2026-06-05 re-check below).

**Recall-vs-width study (2026-06-05) — GNN routing-fill prefilter feasibility.**
Tested whether a cheap surrogate can triage *wide* candidate pools (the premise of
a GNN that evaluates 1000+ and verifies only the top-K). Data:
`ml_data/recall_study/{ibm10,ibm13}_w{64,128,256}.jsonl.gz` (filter off, all
candidates exact-labelled); analyzer `test/diagnostic/_recall_at_width.py`. Two
decisive findings: (1) the *legal* pool saturates at median **94 / max 168** even
at `N_TARGETS=256`, so 1000-wide is only reachable **cross-macro** (the
evaluate-all-then-commit restructuring). (2) `improving_recall@5` collapses with
width (0.78→0.67→0.36 at pool 18/45/94 on held-out ibm13) and a **fresh wide-pool
-trained** surrogate does **not** recover it (0.36 vs 0.33) — the collapse is
fundamental, not OOD. BUT gains are near-tied so the right metric, gain-regret, is
benign: **top-10 captures ~95 % of achievable gain at width-94** (regret@10 5.3 %).
Verdict: a cheap XGBoost already triages wide pools well enough; ranking quality is
not the bottleneck, so the GNN is **not justified** for IBM/hard-relocation — the
73 % strip-gen cost is better attacked by vectorizing the *exact* kernel
cross-macro. Full roadmap + gates: [`../ml_nn/04-gnn-routing-fill-surrogate.md`](../ml_nn/04-gnn-routing-fill-surrogate.md).

**NG45 re-check (2026-06-05).** Re-ran on the 4 NG45 designs (`ml_data/recall_study_ng45/`):
the verdict holds, for stronger reasons. (1) NG45 **converges with ~40 % budget to
spare** (150 s budget, 90–97 s elapsed) — not deadline-bound, so the filter's
"free budget → more rounds" premise is void (the downside-only regime of
`../ml_nn/02`). (2) hard-relocation is **near-idle** on NG45 (1.9–3.0 % improving
groups vs 20–25 % on IBM); the productive operators are **soft_2opt (34 %)** and
soft_relocation (13 %). Caveat: this tier is coarse-grid (504–1404 cells), *not* the
large-grid/deadline-bound industrial regime where a learned routing-fill surrogate
would pay off — neither contest tier reaches that scale.

---

### S10-orig. ML candidate ranker design notes — per-operator XGBoost (DATA COLLECTED 2026-06-04)

A learned filter to make the R2 local search spend its scoring budget on
candidates likely to improve, while keeping the exact accept-on-true-proxy gate
as the final arbiter (so the search stays **strictly non-regressing** — the
model only reorders/prunes what gets scored, it never accepts a move).

**Why it can help.** The search is deadline-bound, and the worst benchmarks
(ibm12/14/15/17/18) are the slow-to-score large ones that get few R2 rounds.
Pruning losers before scoring → more accepts per second → lower score where the
headroom is. The exact gate means the only failure mode is *under-improvement*
(model drops the true-best candidate), never a regression.

**Where it plugs in (two ranking decisions, per `relocation.py` structure):**
- **Across groups — which hot macros to `prep` and attempt.** `_prepare_move`
  (the routing-apply, ~30%/move) is the per-group cost; gating which macros to
  attempt is the *higher-leverage* lever.
- **Within a group — which target cells to trial.** The "score only top-K"
  decision. Lower leverage because each `_trial_at` is already cheap post-prep,
  but free to add.

**Design.** Separate models for **hard relocation**, **soft relocation**, and
**hard 2-opt**, with the cong/density `field` as a feature (not separate models).
Two heads per operator from the same trace:
- *Target ranker* (within-group): LambdaMART (`rank:pairwise`/`rank:ndcg`),
  groups = `group_id`, labels = `dataset.add_group_relevance`. → top-K targets.
- *Group gater* (across-group): `binary:logistic` on "did this macro's group
  contain any improving move?" (aggregate `improves` per group). → which macros
  to prep.
Features must be **pre-score and cheap** (already what `CandidateTrace` records:
net degree, source/target cong & density, displacement, hot/cold rank, size,
position). `state_score`/`trial_score` are labels, never features. Do **not**
feed `benchmark.name` (leakage + the no-per-benchmark-branching rule).

**Validation plan.** Train per operator with **whole-benchmark holdout** (never
random rows — within-group rows are correlated). Pick K from an offline
**recall@K of the true-best target per group** curve *before* any `--all`. Then
confirm online: `--all` with the model but no tracing, per-benchmark
non-regression, watching that freed budget beats predict overhead on the large
benchmarks. Stratify the holdout across IBM **and** NG45 so the metric reflects
cross-design transfer.

**Known risk — distribution shift.** Models are trained on the states the greedy
loop visits; once they reorder evaluation the loop visits different states and
the model is off-distribution. Budget one **DAgger** cycle (train v0 → collect
traces from the states it induces → retrain on the union).

**Data status (2026-06-04).** `scripts/collect_ml_data.sh` produced
~12.6M candidate rows across seeds 42/43/44 for IBM (`--all`) + NG45 (`--ng45`)
in `ml_data/traces/`. Per-operator counts: hard_2opt 5.88M, soft_relocation
3.97M, soft_2opt 2.38M, **hard_relocation 190k (the lean one — NG45 cross-design
data matters most here)**, hard_soft_swap 71k, hard_soft_soft_cycle 62k. Training
deps (`xgboost`, `scikit-learn`) are offline-only in `requirements.txt`. Next
step is the offline training scaffold (rank + gater heads, recall@K curve) under
`test/diagnostic/`, starting with hard_relocation. See README "ML candidate-ranker
data collection" for the collection workflow, and `../ml_nn/` for the
conceptual design (why it can improve, selection mechanism).

---

## Speculative performance ideas (not started)

### P1. B5 GIL-aware parallel scoring

**Untested.** If `plc.get_*_cost` C++ paths release the GIL,
ThreadPoolExecutor with 2-4 workers could double or quadruple score
throughput. Currently single-threaded.

**Verification first:** spawn 2 threads each calling `_exact_proxy`
on the same plc. If wall-clock ≈ 50% of single-thread, GIL releases.

**Implementation if it works:** 2-opt could try 2-4 swap candidates
in parallel, score each, accept the best improvement. Requires
careful synchronization since accept changes shared state.

**Expected gain:** doubles or quadruples 2-opt accept rate, may
translate to small score improvement on candidate-bound benchmarks.

### P2. B6 batched `_fast_set_placement`

Currently Python loop calls `set_pos` per-macro. With ~1500 macros
on large benchmarks, iteration overhead is ~2ms per call. Multiple
candidates per score → potentially 5-10ms per benchmark per `--all`.

**Status:** plc may not have a batched API. Worst case requires a
binding contribution.
**Expected gain:** small wall-clock save.

### P3. Per-net incremental DENSITY (IMPLEMENTED 2026-05-26 — verifying via --all)

B3 phase 4 made congestion routing incremental. Density was the last
full-recompute in `score_swap` (`plc.get_density_cost` scatters ALL
soft+hard macros into the occupancy grid each call). On a 2-opt swap
only macros i, j move, so the occupancy delta is a handful of cells.

`IncrementalScorer` now maintains `grid_occupied` as state:
`score_swap` subtracts i,j's OLD footprints + adds NEW (via
`_macro_occ`, an exact per-macro replica of the full overlap math),
takes top-10% over the grid, then reverts the touched cells;
`commit_swap` persists the delta. `_compute_density_cost` mirrors
`get_density_cost` (0.5 × mean of top floor(0.1·n_cells) nonzero cells).

**Verified:** `_verify_incremental_scorer.py` — score_swap (incl. density)
matches `_exact_proxy` to ≤4.4e-16 (machine eps) on ibm01/04/10, both
trial swaps and sequential commits (no drift).
**Measured speedup:** score_swap −22% to −29%
(ibm01 1.77→1.36ms, ibm04 1.46→1.14ms, ibm10 2.03→1.44ms, ibm16
1.99→1.47ms). Translates to ~40–56% more 2-opt scores in the 15s
deadline (ibm04 7914→11058; ibm10 4784→7482).

**Why it matters:** the 2-opt is deadline-bound on large benchmarks
(ibm10/12/16 use the full 15s), so more-scores-per-second converts
directly to more accepts → lower proxy on the congestion-heavy
benchmarks that dominate the --all average. On small/mid benchmarks
(ibm01/04) the speedup lets 2-opt *converge* before 15s — which is the
budget S1 needs to fire (see S1).

### P4. Skip `_routing_congestion_perturb` on Phase 9 trials (OBSOLETE)

Phase 9 (random-order legalize) doesn't use congestion gradient —
it just legalizes from init_pos with a shuffled order. The
`_routing_congestion_perturb` calls in Phase 1/2/3/5b/5c/7/8 are
retired cong-grad calls from the old pipeline. Phase 9 trials currently don't
call `_routing_congestion_perturb` (correct), and the active pipeline no longer
has cong-grad phases to prune here.

**Status:** obsolete after cong-grad deletion.

### P5. Interleave speedup — shipped as a different stack (RESOLVED 2026-05-29)

**Final outcome (different from the original proposal):** the original P5 plan
("shared scorer eliminates ~60–75 s/benchmark of redundant inits") was retired
when fixed-overhead profiling showed the per-pass `IncrementalScorer.__init__` is
21–48 ms (not seconds) and `_exact_proxy` is 5–12 ms — total per-pass fixed
overhead ~0.1–0.28 s/round / ~0.7–1.7 s/benchmark, not the projected 60–75 s.
A shared-scorer refactor would have saved <1.7 s/benchmark and put the
verified-bit-exact core at risk. **Not implemented.**

Instead, the actual shipped speedup is a *different, mutually compounding*
stack of bit-exact changes:

- **Incremental congestion cost (RESOLVED 2026-05-29).** `IncrementalScorer`
  caches the smoothed normalized H/V (a 2D `grid_row × grid_col` array) and per
  move re-smooths only the touched-net pin-bbox columns/rows from raw flats.
  The box filter is separable (H per column, V per row, each independent), and
  recomputing from raw — not accumulating deltas — keeps it bit-identical to a
  full re-smooth (no drift). All 6 move paths
  (`score_swap`/`commit_swap`/`score_move`/`commit_move`/`score_move_soft`/
  `commit_move_soft`) wired through `_resmooth_bbox`. Verified: swap Δ≤4.4e-16,
  hard/soft move ≤1.8e-9, no drift over commits. **Isolated `--all`: 1.2799 →
  1.2767** (−0.0032).
- **Idea #1 subset-cumsum strip-batch (RESOLVED 2026-05-29).** `_apply_h/v_strips_batch`
  now `np.unique`s the touched rows/cols and cumsums only those (per-row/col
  independent in the diff-array → bit-identical to full-grid). Both incremental
  and full-build routing paths share it.
- **Idea #2 topology-struct cache (RESOLVED 2026-05-29).** Split
  `_apply_net_routing_subset` into `_build_net_routing_struct` (placement-
  independent: gather indices, lengths, 2/3/≥4-pin classification, ≥4-pin sink
  index layout) and `_apply_net_routing_struct` (position-dependent fill).
  Scorer caches the struct per module — single-macro paths hit on every move and
  every −1/+1 within. Swap builds once per call (pair-keyed, not cached). The
  init path keeps the original `_apply_net_routing_subset` (additive — de-risks
  the full-build path).
- **A: round-3 cong soft-reloc hard cap (RESOLVED 2026-05-29).** Cong soft pass
  saturates by round 3 (ibm09 round 4+ accepts ≤2 moves, ~zero gain). Hard-skip
  on `_r2 >= R3_CONG_MAX_ROUNDS=3`. Frees ~4–5 s/round.
- **C: density `top_hot` boost 128→192 on rounds 4–6 (RESOLVED 2026-05-29).** On
  the rounds where cong is skipped, density gets a wider candidate set so the
  freed time is spent on more density attempts instead of returning early.

**Combined `--all`:** 1.2767 → **1.2755** (−0.0012 on top of incremental cong
cost; total speedup-stack contribution 1.2799 → 1.2755 = **−0.0044**). 12/17
wins vs the cong-only baseline, biggest movers ibm17 −0.034, ibm16 −0.019, ibm07
−0.015. WSL run reported 3860s wall (host-suspend O4 inflated ibm06 to 1509s
wall vs ~125s real); `monotonic` budget held — no benchmark returned baseline.

**Diagnostics retained** (`v2/test/diagnostic/`): `_profile_move.py` (per-move
breakdown — cong=20%, density=0.7%, routing-apply=67%), `_profile_move_internals.py`
(cProfile attribution that pointed at `_apply_net_routing_subset`),
`_profile_move_realistic.py` (same-macro/nearby vs random-k pattern, isolates
#2's cache benefit).

---

## Maintenance items

### M1. Stale failing test in `test/test_varrahan_v2_congestion.py`

The test asserts H+V combined-perturb behavior, but
`_routing_congestion_perturb` uses `max(H, V)` per a documented A/B
test. The actual code is correct; the test was authored when the
code briefly used H+V and never re-synced.

**Constraint:** `test/` is read-only per CLAUDE.md. Requires user to
update the test or grant write permission.

### M2. Harness `time.time()` exposure (see O4)

Cosmetic but confusing — the harness's "Total runtime" output
sometimes shows 36000+ seconds while actual elapsed is <600s. Caused
by WSL2 wall-clock drift in the harness's own timing (which we don't
control).

### G1. Out-of-bounds soft swaps (RESOLVED 2026-06-09 — caught by the synthetic suite)

First catch from the anti-overfitting suite
(`test/benchmarks/`): 9/10 synthetic benchmarks came back INVALID with
0.15–0.52um overhangs, all on SOFT macros. The soft-2opt swap
(`soft_moves.py`) exchanged positions with **no bounds check**, so a
larger soft macro inheriting a smaller one's edge-flush slot overhung
the canvas. IBM never trips it because the hand-tuned `initial.plc`
seeds keep softs off the canvas edge. Fix: clamp swap targets by each
macro's own half-size (softs may overlap, so clamping stays legal),
plus tighten the `EPS=0.05` overhang allowance in the hard bounds
checks (`two_opt.py`, `relocation.py`, `legalize/swap.py`) to strict —
`validate_placement` has zero tolerance, so those were latent INVALID
sources too. The rerun then exposed a second leak with the same shape:
`hard_soft.py` (HXS swap + HS3 3-cycle) bounds-checked only the HARD
macro's destination, never the softs inheriting slots in the exchange
(`syn07_ports` still INVALID, 0.253um). Same fix: clamp each soft's
inherited slot by its own half-size, strict hard bounds. Verified:
all 10 synthetics VALID, proxies within noise or slightly better
(syn03 4.3063 vs 4.3113); ibm01 unaffected before and after
(0.9111 VALID).

### G2. Budget overrun at scale (OPEN — found by syn10_xl)

`syn10_xl` (820 hard / 2000 soft / 50×50 grid) ran **504s against a
90s budget**; `syn09_seedless` ran 173s. Something in the pipeline is
not deadline-gated at this scale (IBM tops out at 786 hard macros and
the floor-reservation allocator was tuned on IBM-sized cases). Worth
profiling before any rule change adds bigger benchmarks — a 17-case
run of syn10-sized designs would blow the 1-hour harness cap.

### G3. Seed dependence (OPEN — quantified by syn09_seedless)

With a scrambled seed (same netlist style as syn08), v2 lands at proxy
3.27 vs ~1.14 with a coherent seed. Overlaps are recovered (240 → 0)
but global structure is not rebuilt — consistent with the "initial.plc
is already a good seed" note, and a real exposure on any future
benchmark without a curated seed. A cheap global-restructure phase
(e.g. cluster-aware seeding when the seed scores terribly) is the
obvious lever.

---

## What's NOT in this list (resolved or rejected — see commits)

The session 2026-05-23 → 2026-05-25 closed the following. They're
documented in the commit messages; no need to track them here.

- B1 — `--all` wall-clock timeout (cumulative-budget guard).
- A1 — 2-opt-on-winner uses displacement, not proxy (proxy 2-opt shipped).
- B3 phases 1-4 — incremental scoring (position cache, per-net HPWL,
  numpy abu, per-net routing).
- B4 — `_vectorized_get_routing` dispatch cache.
- A3 — DREAMPlace diagnostic re-run.
- A6 axis #1 (TOP-K cong-grad / Phase 8) and axis #4 (random-order /
  Phase 9).
- 2-opt widening (k_neighbors=10, max_iters=6, Phase 8 multi-iter chains).
- A2 — DP soft_macros_movable diversification (lo-fix + hi-mov).
- WSL2 clock-drift hardening (`time.monotonic()` throughout).
- NG45 design disambiguation in `_load_plc`.
- A5 — Phase 7 retro-eval (gate + RNG isolation, bit-stable).
- Rejected: A4 (DP gating), A6 axes #2 (drop lo) and #3 (fine-noise
  from best), B7 (score cache), B8 (adaptive max_iters), B9 (smart
  ordering), 2-opt cache memoization, stale soft-resnap helpers (only
  in v1).
