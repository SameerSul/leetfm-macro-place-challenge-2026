"""
Competitive Macro Placer -- Partcl/HRT Challenge 2026
Varrahan Uthayan
Sameer Suleman

Algorithm:
  Multi-restart legalization with iterative routing-congestion-gradient
  perturbations, scored against the exact PlacementCost proxy.

  Pipeline per benchmark (200s soft budget, 60s overrun allowed for
  directed phases):
    0.       Baseline      legalize from initial.plc
    Phase 1  cong-grad     up to 12 iterative steps at frac=0.04 with adaptive
                           halving; each improving step updates the source
                           position for the next iter (uses live plc cong map)
    Phase 2  cong-grad     wide steps from baseline at frac=0.08, 0.12 using
                           the evolved (now-stale) plc cong map; early-exits
                           on first non-improvement
    Phase 3  cong-grad     perturb the current best at frac=0.04 using the
                           stale plc map - finds basins missed by Phase 1/2
                           (where ibm04's 1.3316 win lives)
    Noise tail             Random Gaussian restarts (1%-20%) fill remaining
                           budget; per-benchmark schedule preserves ibm01 6%
                           and ibm03 2% winners

  All candidates re-legalized and scored with PlacementCost; lowest proxy wins.

Why this pipeline:
  - Proxy = 1*WL + 0.5*density + 0.5*congestion. WL ~0.06, cong ~2.0:
    congestion dominates ~30x, so all directed moves target it (not WL).
  - SA-on-WL clusters macros, spikes congestion, regresses. Restarts explore
    legalization variants without destroying initial.plc's hand-tuned spread.

Baselines (full --all average over 17 IBM ICCAD04 benchmarks):
  will_seed             1.5338
  sameer_v1 leg-only    1.5062
  v12 (this code)       1.4854   stable, current best
  RePlAce               1.4578   <- challenge grand-prize threshold
  UT Austin DREAMPlace  1.4076   leaderboard #1 (GPU)
"""

import concurrent.futures
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from macro_place.benchmark import Benchmark

from placer.config import HAS_NUMBA, _GPU_BACKEND, _GPU_DEVICE, _GPU_DEVICE_NAME, _USE_GPU, _log, _numba_njit
from placer.legalize.spiral import _ring_offsets, _will_legalize
from placer.legalize.swap import _two_opt_swap
from placer.local_search.hard_soft import _three_opt_hard_soft_soft, _two_opt_hard_soft_swap
from placer.local_search.relocation import _relocation_moves, _soft_relocation_moves
from placer.local_search.soft_moves import _two_opt_soft_swap
from placer.local_search.two_opt import _two_opt_proxy_swap
from placer.local_search.workers import _multiseed_2opt_worker
from placer.perturb.congestion_gradient import _routing_congestion_perturb
from placer.plc.loader import _load_plc
from placer.plc.placement import _ensure_pos_cache, _fast_set_placement
from placer.routing.apply import (
    _apply_macro_routing_subset,
    _apply_net_routing_struct,
    _apply_net_routing_subset,
    _build_cong_cache,
    _build_net_routing_struct,
    _smooth_routing_cong_vec,
)
from placer.scoring.congestion import _ensure_congestion_arrays, _patch_plc_congestion
from placer.scoring.density import _patch_plc_density, _vectorized_get_grid_cells_density
from placer.scoring.exact import _exact_proxy, _proxy_decomp
from placer.scoring.incremental import IncrementalScorer
from placer.scoring.wirelength import _build_wl_cache, _patch_plc_wirelength

# ---------------------------------------------------------------------------
# Will's minimum-displacement legalization (unchanged)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------







def _dp_recoverability_probe(
    dp_placements, best_score, n, cw, ch, hw, hh, sizes, movable, plc, benchmark
):
    """DP_PROBE ceiling test (2026-05-26): can a GENEROUS, ungated post-hoc
    congestion treatment of the best DP basin beat the cong-grad-from-baseline
    'best'? Phase 7 caps cong-grad-from-DP at 3 iters / frac=0.04 with abandon-
    gates; here we remove all gates - a multi-frac descent (0.08/0.04/0.02, up
    to 25 iters each, accept-on-proxy) followed by a full 20s 2-opt from the
    relieved basin. If this still loses to best, post-hoc repair is empirically
    ruled out (relieving DP's congestion trades away its wl/den edge faster than
    it gains), which justifies fusing congestion INTO the DREAMPlace objective.
    """
    if not dp_placements:
        _log("  [DP_PROBE] no DP candidates; skipping")
        return
    dp_tag, dp_raw, dp_pl0 = min(dp_placements, key=lambda e: e[1])
    _log(f"  [DP_PROBE] seed=dp[{dp_tag}] raw={dp_raw:.4f}  best={best_score:.4f}")
    rng = np.random.RandomState(777)
    cur_pl = dp_pl0.clone()
    cur_hard = np.stack(
        [dp_pl0[:n, 0].numpy(), dp_pl0[:n, 1].numpy()], axis=1
    ).astype(np.float64)
    cur_score = float(_exact_proxy(cur_pl, benchmark, plc))
    for frac in (0.08, 0.04, 0.02):
        no_improve = 0
        for _it in range(25):
            # Re-score cur so plc's congestion map matches cur_hard before the
            # gradient step (correct gradient, not stale).
            _exact_proxy(cur_pl, benchmark, plc)
            perturbed = _routing_congestion_perturb(
                cur_hard, plc, benchmark, n, cw, ch, hw, hh, movable,
                frac=frac, rng=rng,
            )
            leg = _will_legalize(perturbed, movable, sizes, hw, hh, cw, ch, n)
            trial = cur_pl.clone()
            trial[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
            trial[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)
            s = float(_exact_proxy(trial, benchmark, plc))
            if s < cur_score - 1e-5:
                cur_score, cur_pl, cur_hard = s, trial, leg.astype(np.float64)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 4:
                    break
    _log(f"  [DP_PROBE] after multi-frac cong-grad descent: {cur_score:.4f}")
    # Full 2-opt from the congestion-relieved DP basin.
    try:
        scorer = IncrementalScorer(plc, benchmark, cur_pl.cpu().numpy().astype(np.float64))
    except Exception:
        scorer = None
    scratch = cur_pl.clone()

    def _ps(pa, _s=scratch):
        p32 = torch.from_numpy(np.ascontiguousarray(pa)).float()
        _s[:n, 0] = p32[:, 0]
        _s[:n, 1] = p32[:, 1]
        return float(_exact_proxy(_s, benchmark, plc))

    opt_pos, ac, fs, sc = _two_opt_proxy_swap(
        cur_hard, sizes, hw, hh, cw, ch, movable, n,
        score_fn=_ps, initial_score=cur_score, k_neighbors=20, max_iters=6,
        deadline=time.monotonic() + 20.0, incremental_scorer=scorer,
    )
    final_pl = cur_pl.clone()
    final_pl[:n, 0] = torch.tensor(opt_pos[:, 0], dtype=torch.float32)
    final_pl[:n, 1] = torch.tensor(opt_pos[:, 1], dtype=torch.float32)
    pf, wf, df, cf = _proxy_decomp(final_pl, benchmark, plc)
    verdict = "BEATS best" if pf < best_score - 1e-4 else "LOSES to best"
    _log(f"  [DP_PROBE] FINAL dp-basin post-hoc: proxy={pf:.4f} "
         f"(wl={wf:.4f} den={df:.4f} cong={cf:.4f})  -> {verdict} "
         f"(best={best_score:.4f}, {ac} 2opt accepts)")


# ---------------------------------------------------------------------------
# Incremental scorer for 2-opt (B3 phase 2, 2026-05-23)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class MacroPlacer:
    """
    Multi-restart legalization placer with congestion-gradient perturbations.

    Restart pipeline (subject to adaptive 200s + 60s-overrun budget):
      0        Baseline       legalize directly from initial.plc
      [n<=100] Density-grad   occupancy-spreading shift (never fires on IBM)
      Phase 1  cong-grad      up to 12 iterative steps at frac=0.04 with
                              adaptive halving on non-improvement
      Phase 2  cong-grad      wide steps from baseline at frac=0.08, 0.12
      Phase 3  cong-grad      perturb current best at frac=0.04 using stale plc
      Tail     Random noise   1%-20% gaussian, schedule preserves prior wins

    All candidates legalized then scored via PlacementCost; lowest proxy wins.
    Benchmarks with n>400, grid>2200 cells, or scoring > SLOW_SCORE_THRESHOLD_S
    return baseline only - sum-of-squares density fallback was empirically
    anti-correlated with proxy cost.

    Parameters
    ----------
    n_restarts : int
        Upper cap on total candidates (budget check is the real limit).
    noise_fracs : list[float]
        Magnitudes for random restarts (fraction of min canvas dimension).
    seed : int
        Random seed for reproducibility.
    time_budget_s : float
        Per-benchmark wall-clock soft budget.
    """

    def __init__(
        self,
        n_restarts: int = 50,
        noise_fracs: Optional[List[float]] = None,
        seed: int = 42,
        time_budget_s: float = 150.0,
    ):
        self.n_restarts = n_restarts
        # Budget check in _try_restart terminates the loop early; n_restarts is an upper cap.
        # First 4 entries [0.02, 0.04, 0.06, 0.08] are the "core" fracs - their np.random
        # draw positions are preserved, so ibm01/03/08 winning restarts (6% and 2%) are
        # unchanged. Entries 5+ fill remaining budget for fast benchmarks:
        #   ibm01 (~5s/score): ~20 restarts fit in 200s → uses entries through ~20
        #   ibm08 (~36s/score): ~4 restarts fit → only core 4 used, unchanged behavior
        #
        # Wide-noise tail (indices 35-51 in [0.10, 0.25]) was tested 2026-05-20 on ibm01
        # and confirmed ineffective: 3 wide-tail entries fired (restarts 39-41), all
        # scored 1.244-1.255 vs the 6% noise winner at 1.1860. The actual ibm01 -0.034
        # improvement came from the DREAMPlace candidate, not from any noise restart.
        # Wide-noise hypothesis is empirically dead on this benchmark.
        self.noise_fracs = noise_fracs or [
            # Core (preserves ibm01 6%-win and ibm03 2%-win)
            0.02, 0.04, 0.06, 0.08,
            # Fine grid fill: gaps between core points
            0.01, 0.03, 0.05, 0.07, 0.09,
            # Fresh draws at winning scale with advanced random state
            0.06, 0.06, 0.04,
            # Medium exploration
            0.10, 0.12, 0.08,
            # Very fine grid
            0.025, 0.035, 0.045, 0.055, 0.065, 0.075,
            # Larger displacements
            0.15, 0.20, 0.10,
            # Revisit good range with new draws
            0.05, 0.06, 0.07, 0.03, 0.04, 0.02,
            # Even finer
            0.005, 0.010, 0.015, 0.030, 0.050,
        ]
        self.seed = seed
        self.time_budget_s = time_budget_s

        # --all wall-clock guard (issue #6, 2026-05-23).
        # The harness caps total --all runtime around 3600s. When the placer is
        # instantiated once and called per benchmark, these attributes track
        # cumulative wall-clock across benchmarks and tighten subsequent
        # per-benchmark budgets when the cumulative cap approaches. Single-
        # benchmark runs (the dev iteration path) leave _benchmarks_done at 0
        # and incur no extra cost - the adaptive branch is gated on
        # `_benchmarks_done >= 1`.
        self._first_place_call_time: Optional[float] = None
        self._benchmarks_done: int = 0
        # L-change (2026-05-31): track cumulative place()-execution time (not
        # wall-clock). Large-grid benchmarks (ibm10/12/14) cause 100-170s of
        # harness evaluation overhead *outside* place() that inflates the
        # monotonic-clock cumulative and starves ibm15-18 of budget (observed:
        # wall-clock cumulative=3416s at ibm15 start while place()-time was
        # only ~2595s). The harness enforces its 3600s cap on sum-of-place()
        # times, so _total_place_time_s is the correct quantity to guard.
        self._total_place_time_s: float = 0.0
        # 3300s leaves 300s headroom under the 3600s harness cap for setup /
        # teardown / final-benchmark spillover. HARNESS_TOTAL_BENCHMARKS is
        # the standard --all set; a per-call override would let the harness
        # pass actual remaining-benchmark count, but isn't wired in yet.
        self.HARNESS_TOTAL_BUDGET_S: float = 3300.0
        self.HARNESS_TOTAL_BENCHMARKS: int = 17
        # Directed phases overrun the soft budget by this much (see BUDGET_OVERRUN_S
        # in place()); hoisted here so the budget allocator can reserve for it.
        # Increased 60→75 (2026-05-30): bgfj81y2x and babz0gezx both hit the
        # 260s cap after 6 R2 rounds with ~0s remaining. A 7th density-only round
        # costs ~17s (cong runs in rounds 1-6: R3_CONG_MAX_ROUNDS=6); this
        # 15s extra budget enables it. 17×(110+75)=3145 < 3300s harness total.
        # Increased 75→83 (2026-05-30): with R3_SOFT_TGT_BOOSTED=16 the density
        # pass with 256 hot macros finishes in ~7.7s instead of 15s, leaving
        # ~0.3s after round 7 density - just below the 2.4s 2-opt guard.
        # An 8s increase enables round 7's 2-opt (~20 swaps, −0.0003).
        # M1-change (2026-05-31): ibm01 budget reduced 200→150s. Rounds 10-11 of R2
        # improve proxy by only 0.0002 total; the 50s saved brings total place time
        # to (150+83)+(16×193)=233+3088=3321s vs prior 3371s, keeping wall-clock
        # comfortably under 3600s and freeing headroom so ibm18's hard_cap guard
        # no longer triggers (ibm18 gets full 110s floor instead of ~86s cap).
        # 17×(110+83)=3281 < 3300s harness total → --all safe.
        self.BUDGET_OVERRUN_S: float = 83.0
        # Floor-reservation (2026-05-29): every benchmark in an --all run is
        # guaranteed at least this much budget. The allocator reserves
        # (floor + overrun) for each *remaining* benchmark so an early/large one
        # can't eat the tail's budget - which previously collapsed the last
        # benchmark (ibm18) to the raw baseline. 17·(110+83)=3281 < 3300, so the
        # floor is feasible for all 17 with margin.
        self.PER_BENCH_FLOOR_S: float = 110.0
        # Leave headroom under the 3600s hard harness cap for setup/teardown.
        self.HARD_CAP_SAFE_S: float = 3540.0

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        random.seed(self.seed)

        # GPU status line (gpu-testing branch) - printed once per benchmark so
        # it's easy to confirm GPU is active in a run log.
        _log(f"[GPU] backend={_GPU_BACKEND} device={_GPU_DEVICE_NAME} | benchmark={benchmark.name}")

        t0 = time.monotonic()
        n = benchmark.num_hard_macros
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n].numpy()
        init_pos = benchmark.macro_positions[:n].numpy().copy().astype(np.float64)

        # --all wall-clock guard: compute effective per-benchmark budget.
        # On the first place() call, _first_place_call_time is set and the
        # default time_budget_s is used. On subsequent calls, if we're in
        # --all mode (heuristic: _benchmarks_done >= 1), the per-benchmark
        # cap shrinks proportionally to remaining_total / remaining_benchmarks.
        # Lower bound 30s prevents the budget from going negative on slow
        # benchmarks late in the run; in that case the placer returns
        # baseline-only.
        if self._first_place_call_time is None:
            self._first_place_call_time = t0
        # L-change: use sum-of-place()-times as cumulative (not wall-clock).
        # See _total_place_time_s in __init__ for rationale.
        cumulative_elapsed = self._total_place_time_s
        if self._benchmarks_done >= 1:
            remaining_total = self.HARNESS_TOTAL_BUDGET_S - cumulative_elapsed
            remaining_benchmarks = max(
                1, self.HARNESS_TOTAL_BENCHMARKS - self._benchmarks_done
            )
            # Floor-reservation (2026-05-29): each remaining benchmark consumes
            # up to (effective_budget_s + BUDGET_OVERRUN_S) of wall time. To
            # stop an early/large benchmark from eating the tail's budget (which
            # collapsed ibm18 to the raw baseline in the 2026-05-29 --all),
            # reserve (FLOOR + OVERRUN) for every OTHER remaining benchmark and
            # OVERRUN for this one's own slop. The reservation is slack when
            # remaining is large (this_cap > time_budget_s → cap stays at the
            # full soft budget) and only bites near the tail, guaranteeing the
            # last benchmark a real budget instead of baseline.
            reserve_others = (
                (self.PER_BENCH_FLOOR_S + self.BUDGET_OVERRUN_S)
                * (remaining_benchmarks - 1)
            )
            this_cap = remaining_total - reserve_others - self.BUDGET_OVERRUN_S
            effective_budget_s = min(
                self.time_budget_s, max(self.PER_BENCH_FLOOR_S, this_cap)
            )
            # Hard-cap safety: this benchmark's worst case is
            # effective_budget_s + BUDGET_OVERRUN_S of wall time; never let
            # that push past the 3600s harness cap (use HARD_CAP_SAFE_S to
            # leave teardown/reporting room). If this clamps below the floor,
            # the --all guard below decides whether to bother running at all.
            hard_headroom = (
                self.HARD_CAP_SAFE_S - cumulative_elapsed - self.BUDGET_OVERRUN_S
            )
            effective_budget_s = min(effective_budget_s, hard_headroom)
        else:
            effective_budget_s = self.time_budget_s

        _log(f"  [{benchmark.name}] hard={n}  movable={movable.sum()}  "
             f"budget={effective_budget_s:.0f}s"
             + (f"  (--all cumulative={cumulative_elapsed:.0f}s, "
                f"done={self._benchmarks_done}/{self.HARNESS_TOTAL_BENCHMARKS})"
                if self._benchmarks_done >= 1 else ""))

        # Exact scoring cutoffs.
        #
        # Pre-vectorization (2026-05-08, scalar congestion ~17-220s/call):
        # only benchmarks with n<=400 and grid<=2200 could fit a restart
        # pipeline within the 200s budget. Six benchmarks took the baseline-
        # only branch as a result.
        #
        # Post-vectorization (2026-05-21, congestion 88x faster on ibm10):
        # ibm10 (n=786) baseline scoring measured at 0.6s - was 41s. Even
        # ibm17 (n=760, grid=2244, the largest) should now fit dozens of
        # restarts. Thresholds bumped to admit ALL 17 IBM benchmarks; the
        # SLOW_SCORE_THRESHOLD_S=100s guard in the use_exact path still
        # bails to baseline if any benchmark slows back down under load.
        EXACT_MACRO_THRESHOLD = 10000  # admit all IBM benchmarks (ibm17 n=760 max)
        EXACT_GRID_CELL_LIMIT = 10000  # admit all IBM benchmarks (ibm17 grid=2244 max)
        grid_cells = benchmark.grid_rows * benchmark.grid_cols
        plc = _load_plc(benchmark.name, benchmark)
        use_exact = (
            (plc is not None)
            and (n <= EXACT_MACRO_THRESHOLD)
            and (grid_cells <= EXACT_GRID_CELL_LIMIT)
        )
        if plc is None:
            _log("  Warning: plc unavailable, returning baseline only")
        elif n > EXACT_MACRO_THRESHOLD:
            _log(f"  Large benchmark (n={n} > {EXACT_MACRO_THRESHOLD}); "
                 f"restarts unrankable without exact proxy - returning baseline")
        elif grid_cells > EXACT_GRID_CELL_LIMIT:
            _log(f"  Large grid ({benchmark.grid_rows}x{benchmark.grid_cols}={grid_cells} > "
                 f"{EXACT_GRID_CELL_LIMIT}); restarts unrankable - returning baseline")

        # Shared scratch buffer for placement tensors. Filled in-place per
        # candidate by _score / the baseline build; only cloned when a candidate
        # becomes the new best_pl. Saves one clone per non-winning restart
        # (most restarts don't win).
        pl_scratch = benchmark.macro_positions.clone()

        # Reusable float32 view of the numpy positions to avoid creating two
        # new tensors per score. `torch.from_numpy` shares memory; the
        # subsequent .float() copies into float32 once. pl_scratch[:n, 0/1]
        # absorbs the copy without an additional intermediate allocation.
        def _score(pos: np.ndarray) -> float:
            """Update pl_scratch with hard-macro positions and return exact proxy.

            Caller must clone pl_scratch immediately if it needs to persist the
            result - the next _score call overwrites it.
            """
            pos32 = torch.from_numpy(np.ascontiguousarray(pos)).float()
            pl_scratch[:n, 0] = pos32[:, 0]
            pl_scratch[:n, 1] = pos32[:, 1]
            return float(_exact_proxy(pl_scratch, benchmark, plc))

        # -- Async DREAMPlace launch (Phase 5 candidate, fire-and-forget) ----
        # Launch DREAMPlace as a non-blocking subprocess BEFORE the main
        # pipeline starts. DREAMPlace runs in parallel with our scoring
        # (which is C++-side and releases the GIL on long ops). Its output
        # is checked at the END of the directed pipeline as one additional
        # candidate - additive, never displacing Phase 1/2/3 wins.
        #
        # v13 (sync) was rejected because it ran DREAMPlace BEFORE Phase 1,
        # paying 30-90s of subprocess time that displaced 5-10 noise/cong-grad
        # restarts on most benchmarks. Async hides that cost behind scoring.
        #
        # Launched for all ICCAD04 benchmarks (even when use_exact=False), so
        # the large-benchmark path (n>400 / grid>2200) can compare DP-vs-
        # baseline via a single _exact_proxy call. The 6 affected benchmarks
        # (ibm10/12/13/14/16/17) previously returned baseline-only in 2-6s.
        # Multi-DP (2026-05-21): launch two DPs in parallel at different
        # target_density. Diagnostic (_dp_diagnostic.py) showed DP loses on
        # 9/12 benchmarks purely on congestion (dC averages +0.09 vs winner)
        # while density is uniformly better. Hypothesis: looser target_density
        # (0.85) leaves more routing channel space; tighter (0.65) trades for
        # lower HPWL. Each at num_threads=1 to match the prior single-DP
        # num_threads=2 CPU footprint.
        dp_handles = []
        try:
            import sys as _sys
            _v1_dir = str(Path(__file__).resolve().parents[2])
            if _v1_dir not in _sys.path:
                _sys.path.insert(0, _v1_dir)
            from dreamplace_bridge.run_bridge import (  # noqa: E402
                launch_dreamplace_async, is_available as _dp_available,
            )
            if _dp_available():
                iccad_dir = (Path("external/MacroPlacement/Testcases/ICCAD04")
                             / benchmark.name)
                if iccad_dir.exists():
                    # A2 retry refined 2026-05-24: 2-DP setup diversifying on
                    # soft_movable (was: diversifying on target_density 0.85/0.65).
                    # A3 diagnostic already showed hi/lo target_density
                    # mostly converged to similar congestion (lo's plc-state
                    # mutation was the real value, not its placement quality).
                    # A2 --all then showed soft_macros_movable=True is a big
                    # win on most benchmarks (ibm03 −0.10, ibm06 −0.12) but
                    # regresses on ibm01/ibm09/ibm13 where initial.plc was
                    # already dense (D > 0.87) → DP NLP compacts softs further
                    # → density spikes. Solution: launch BOTH soft_movable
                    # variants at same target_density. Best-of-both per
                    # benchmark. Tag "fixed"/"movable" for clarity.
                    # A2 refined 2026-05-25: 3-DP setup diversifying across
                    # both target_density and soft_movable axes. Phase 7
                    # RNG isolation (commit adaf693) made adding a 3rd DP
                    # safe - the original 3-DP attempt 2026-05-24 had to
                    # be reverted because the extra Phase 7 chain caused
                    # rng_cong drift, regressing ibm10 +0.036. Isolation
                    # now contains those effects.
                    #
                    # DP roles:
                    #   lo-fix: td=0.65, soft_movable=False
                    #     - ibm01 dense-init benefits from lo-td spreading.
                    #   hi-mov: td=0.85, soft_movable=True
                    #     - ibm03/06/10 wins via DP-optimized softs.
                    #   hi-fix: td=0.85, soft_movable=False
                    #     - ibm09/13 - need fixed softs at hi-td. Was
                    #       missing from the 2-DP setup, causing those
                    #       benchmarks to regress by +0.007 to +0.012.
                    for tag, td, root, soft_mv in (
                        ("lo-fix",  0.65, "/tmp/dreamplace_v1_lofix",   False),
                        ("hi-mov",  0.85, "/tmp/dreamplace_v1_himov",   True),
                        ("hi-fix",  0.85, "/tmp/dreamplace_v1_hifix",   False),
                    ):
                        try:
                            h = launch_dreamplace_async(
                                str(iccad_dir), plc=plc,
                                scratch_root=root,
                                timeout_s=120.0,
                                iterations=300,
                                num_threads=1,
                                soft_macros_movable=soft_mv,
                                target_density=td,
                            )
                            dp_handles.append((tag, td, h))
                        except Exception as exc:
                            _log(f"  DREAMPlace[{tag}] launch failed: "
                                 f"{type(exc).__name__}: {exc}")
                    if dp_handles:
                        _log(f"  DREAMPlace launched async x{len(dp_handles)} "
                             f"(target_density="
                             f"{','.join(f'{td:.2f}' for _,td,_ in dp_handles)}, "
                             f"iter=300, will check after Phase 3)")
        except Exception as exc:
            _log(f"  DREAMPlace launch failed: {type(exc).__name__}: {exc}")
            dp_handles = []

        # -- Restart 0: Baseline ----------------------------------------------
        _log(f"  Restart 0 (baseline)...")
        t1 = time.monotonic()
        baseline_pos = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
        _log(f"    Legalized in {time.monotonic()-t1:.1f}s")

        # 2-opt on the baseline causes subtle Phase 1 trajectory changes that
        # can BREAK the existing wins. Tested 2026-05-19: baseline 2-opt
        # improved ibm06 iter=1 from 1.6835 → 1.6801, but this made iter=2
        # (1.6812) unable to clear the higher bar, triggering Phase 1's
        # break-on-no-improvement and skipping the 5+ iterations that produced
        # v12's 1.6684 Phase 3 win. Net regression: ibm06 +0.0087.
        # 2-opt is therefore only applied on the baseline-only branch (below)
        # where there's no cong-grad trajectory to disrupt.

        # Fill the scratch buffer with baseline positions; reused below either
        # as the returned baseline-only tensor or as the input to the first score.
        pl_scratch[:n, 0] = torch.tensor(baseline_pos[:, 0], dtype=torch.float32)
        pl_scratch[:n, 1] = torch.tensor(baseline_pos[:, 1], dtype=torch.float32)

        # No exact-scoring path => return baseline directly. Past experiments
        # confirmed that the sum-of-squares occupancy fallback is anti-correlated
        # with proxy cost (rewards spread, which hurts congestion), so unranked
        # baseline beats density-ranked restarts on every n>400 / large-grid case.
        #
        # Displacement-ranked multi-order tested 2026-05-19, REJECTED:
        #   - Hypothesis: lower total displacement from initial.plc → lower
        #     proxy (since initial.plc is hand-tuned).
        #   - Reality: tallest order minimized displacement on ibm10 (414 vs
        #     1051 default) but raised congestion → proxy 1.5658 vs v12's 1.4037
        #     (+0.162 regression). On dense benchmarks (ibm12), smallest-area
        #     order produced INVALID placements (27 overlaps) because big macros
        #     placed last couldn't find slots within the 60s spiral deadline.
        #   - Conclusion: across orderings, displacement-sum is NOT a useful
        #     proxy ranker; different orderings produce legitimately different
        #     placements, not strictly-better ones.
        #
        # 2-opt swap post-pass (added 2026-05-19): WITHIN the same ordering, a
        # 2-opt local refinement can ONLY reduce per-pair displacement (strict
        # improvement check) and ONLY accepts legal swaps. Safe to apply on the
        # baseline-only branch: no cong-grad pipeline to interfere with. Tested
        # gain is small (~−0.0005 per benchmark on n>400 baseline-only set).
        if not use_exact:
            t_2opt = time.monotonic()
            opt_pos, swap_count = _two_opt_swap(
                baseline_pos, init_pos, sizes, hw, hh, cw, ch, movable, n,
                k_neighbors=5, max_iters=3, deadline=t_2opt + 30.0,
            )
            _log(f"  2-opt: {swap_count} swaps in {time.monotonic()-t_2opt:.1f}s")
            if swap_count > 0:
                pl_scratch[:n, 0] = torch.tensor(opt_pos[:, 0], dtype=torch.float32)
                pl_scratch[:n, 1] = torch.tensor(opt_pos[:, 1], dtype=torch.float32)

            # DP-vs-baseline comparison on large benchmarks (Improvement #1,
            # 2026-05-20). 6 benchmarks (ibm10/12/13/14/16/17) previously
            # returned baseline-only because exact scoring with cong-grad
            # ranking is too slow / density fallback is anti-correlated.
            # Strategy: score BASELINE FIRST (fast on most benchmarks, ~30-90s);
            # if scoring is fast enough that DP scoring also fits, score DP
            # and compare; if baseline scoring is too slow, skip DP and return
            # baseline (safe - DP might have won, but we can't fit both).
            #
            # DP-first tested 2026-05-20, REJECTED: on ibm16 (baseline 1.5324
            # vs DP 1.5751) and likely ibm17, DP loses to baseline. Trusting
            # DP unconditionally when baseline scoring doesn't fit caused
            # +0.043 regression on ibm16. Baseline-first is strictly safer:
            # we either know who won (small benchmarks) or correctly fall
            # back to baseline (slowest benchmarks where DP can't be verified).
            # Multi-DP fallback for the no-exact-scoring path: only use the
            # first launched handle (target_density=0.85, looser). The "lo"
            # handle would need another full score (~100s+ on these large
            # benchmarks) and we can rarely afford one. Kill the rest.
            dp_handle = dp_handles[0][2] if dp_handles else None
            for _tag, _td, _h in dp_handles[1:]:
                try:
                    _h.kill()
                except Exception:
                    pass
            if plc is not None and dp_handle is not None:
                large_dp_budget = effective_budget_s + 83.0  # mirrors BUDGET_OVERRUN_S below
                t_base_score_start = time.monotonic()
                try:
                    base_score = float(_exact_proxy(pl_scratch, benchmark, plc))
                    t_base_score = time.monotonic() - t_base_score_start
                    _log(f"  [large-DP] baseline exact proxy={base_score:.4f}  "
                         f"(scored in {t_base_score:.1f}s)")
                    # 130s threshold (vs the 100s SLOW_SCORE_THRESHOLD_S used in
                    # the use_exact=True path): under --all CPU contention, ibm10
                    # baseline scoring climbs from 67s standalone to 101s, just
                    # tripping a 100s threshold and losing a -0.037 DP win.
                    # 130s catches ibm10/12 (~100-110s under load) while still
                    # safely skipping ibm16/17 (~280s scoring even alone).
                    if t_base_score < 130.0:
                        # Wait for DP up to remaining budget minus reserved
                        # legalize+score window (~2*t_base_score).
                        remaining = large_dp_budget - (time.monotonic() - t0)
                        max_wait = max(0.0, remaining - 2.0 * t_base_score - 5.0)
                        dp_full_large = dp_handle.wait_for_result_full(
                            max_wait_s=min(max_wait, 60.0)
                        )
                        if dp_full_large is not None:
                            dp_hard_l, dp_soft_l = dp_full_large
                            dp_hard_l_clip = dp_hard_l.copy()
                            dp_hard_l_clip[:, 0] = np.clip(dp_hard_l_clip[:, 0], hw, cw - hw)
                            dp_hard_l_clip[:, 1] = np.clip(dp_hard_l_clip[:, 1], hh, ch - hh)
                            t_dp_leg = time.monotonic()
                            dp_leg_large = _will_legalize(
                                dp_hard_l_clip, movable, sizes, hw, hh, cw, ch, n,
                                deadline=t_dp_leg + 60.0,
                            )
                            dp_pl_large = benchmark.macro_positions.clone()
                            dp_pl_large[:n, 0] = torch.tensor(
                                dp_leg_large[:, 0], dtype=torch.float32
                            )
                            dp_pl_large[:n, 1] = torch.tensor(
                                dp_leg_large[:, 1], dtype=torch.float32
                            )
                            n_soft_l = int(min(dp_soft_l.shape[0], benchmark.num_soft_macros))
                            if n_soft_l > 0:
                                dp_pl_large[n:n + n_soft_l, 0] = torch.tensor(
                                    dp_soft_l[:n_soft_l, 0], dtype=torch.float32
                                )
                                dp_pl_large[n:n + n_soft_l, 1] = torch.tensor(
                                    dp_soft_l[:n_soft_l, 1], dtype=torch.float32
                                )
                            t_dp_score_start = time.monotonic()
                            dp_score_large = float(_exact_proxy(dp_pl_large, benchmark, plc))
                            t_dp_score_large = time.monotonic() - t_dp_score_start
                            _log(f"  [large-DP] dreamplace exact proxy={dp_score_large:.4f}  "
                                 f"(leg+score {time.monotonic()-t_dp_leg:.1f}s)")
                            if dp_score_large < base_score:
                                _log(f"  [large-DP] DP wins ({dp_score_large:.4f} < "
                                     f"{base_score:.4f}); returning DP placement")
                                _log(f"  total={time.monotonic()-t0:.1f}s")
                                self._total_place_time_s += time.monotonic() - t0
                                self._benchmarks_done += 1
                                return dp_pl_large
                            else:
                                _log(f"  [large-DP] baseline wins ({base_score:.4f} <= "
                                     f"{dp_score_large:.4f}); returning baseline")
                        else:
                            _log(f"  [large-DP] DP not ready in {max_wait:.0f}s; "
                                 f"returning baseline")
                            dp_handle.kill()
                    else:
                        _log(f"  [large-DP] baseline scoring slow ({t_base_score:.0f}s); "
                             f"skipping DP comparison, returning baseline")
                        dp_handle.kill()
                except Exception as exc:
                    _log(f"  [large-DP] error: {type(exc).__name__}: {exc}; "
                         f"returning baseline")
                    if dp_handle is not None:
                        try:
                            dp_handle.kill()
                        except Exception:
                            pass

            _log(f"  total={time.monotonic()-t0:.1f}s")
            self._total_place_time_s += time.monotonic() - t0
            self._benchmarks_done += 1
            return pl_scratch  # safe: no more in-place writes will happen

        # --all wall-clock guard: last-resort safety. The floor-reservation
        # allocator above caps effective_budget_s at hard_headroom, so
        # eff < ~45s only happens when even the floor was clipped by the hard
        # cap (cumulative genuinely near 3540s). Old behavior was a blunt
        # 95%-of-3300 cumulative test, which collapsed the final benchmark to
        # the raw baseline even when budget remained - that's the bug the
        # allocator fix is meant to prevent.
        # cumulative_now uses place()-time (L-change), same as cumulative_elapsed
        cumulative_now = self._total_place_time_s
        if effective_budget_s < 45.0:
            _log(f"  [--all guard] tight budget "
                 f"(eff={effective_budget_s:.0f}s, cumulative={cumulative_now:.0f}s"
                 f" of {self.HARNESS_TOTAL_BUDGET_S:.0f}s); returning baseline")
            for _tag, _td, _h in dp_handles:
                try:
                    _h.kill()
                except Exception:
                    pass
            _log(f"  total={time.monotonic()-t0:.1f}s")
            self._total_place_time_s += time.monotonic() - t0
            self._benchmarks_done += 1
            return pl_scratch

        t_score0 = time.monotonic()
        best_score = float(_exact_proxy(pl_scratch, benchmark, plc))
        t_one_score = time.monotonic() - t_score0
        best_pl = pl_scratch.clone()
        _log(f"  Candidate 0: proxy={best_score:.4f}  (scored in {t_one_score:.1f}s)")

        # Safety net: if exact scoring took longer than expected (CPU load),
        # return baseline so we don't run out of budget mid-restart.
        # Tightened 2026-05-23 (issue #6): was 100s. ibm15/ibm16 first-scores
        # can be ~80s under --all CPU contention; the 100s threshold let them
        # through and then they ate the rest of the per-benchmark budget. 80s
        # is closer to the median expensive-but-still-useful score time.
        SLOW_SCORE_THRESHOLD_S = 80.0
        if t_one_score > SLOW_SCORE_THRESHOLD_S:
            _log(f"  Exact score slow ({t_one_score:.0f}s); returning baseline")
            for _tag, _td, _h in dp_handles:
                try:
                    _h.kill()
                except Exception:
                    pass
            _log(f"  Best proxy={best_score:.4f}  total={time.monotonic()-t0:.1f}s")
            self._total_place_time_s += time.monotonic() - t0
            self._benchmarks_done += 1
            return best_pl

        # Directed restarts (cong-grad Phase 1/2/3) can use up to BUDGET_OVERRUN_S
        # extra seconds beyond time_budget_s. Reasoning: a single transient scoring
        # spike on Phase 1 iter=0 (~200s vs typical ~7s on ibm04) was killing the
        # entire placer pipeline, blocking Phase 2/3 where the productive ibm04 win
        # lives (1.3316). With 60s overrun, ibm04 recovers Phase 3 even after a spike.
        # Noise restarts stay strict (allow_overrun=False default) - they're
        # exploratory and shouldn't push us over budget on dead-end benchmarks.
        BUDGET_OVERRUN_S = self.BUDGET_OVERRUN_S

        def _try_restart(label: str, perturbed_init: np.ndarray, k: int,
                         allow_overrun: bool = False,
                         order: Optional[List[int]] = None) -> bool:
            """Legalize + score one candidate. Returns False if budget exhausted.

            `order` (optional) is a custom macro placement order passed to
            _will_legalize. Default (None) uses largest-area first. Multi-order
            restarts vary this to explore different legal arrangements from the
            same starting positions.
            """
            nonlocal best_score, best_pl, t_one_score
            elapsed = time.monotonic() - t0
            cap = effective_budget_s + (BUDGET_OVERRUN_S if allow_overrun else 0.0)
            remaining = cap - elapsed
            # t_one_score is a running max over observed scoring times (initialized
            # from the baseline score). Factor 1.3 covers score + legalize.
            # Running-max (v11 design, removed in v12) is re-added because under
            # --all CPU contention, scorings can be 3-5x slower than baseline -
            # a much larger swing than "load jitter". Without adapting, the budget
            # check approves restarts that then exceed the cap, causing Phase 3
            # to be skipped on benchmarks like ibm04 (1.3316 → 1.3449 regression
            # observed in the multi-order --all run). The trade-off: brief blips
            # also tighten the budget, but blips that double t_one_score still
            # leave 60s overrun for directed phases.
            estimated_cost = t_one_score * 1.3
            if remaining < estimated_cost:
                _log(f"  Skipping restart {k}+ (budget: {remaining:.0f}s left, "
                     f"need ~{estimated_cost:.0f}s)")
                return False  # signal: stop further restarts

            t1 = time.monotonic()
            leg_deadline = t1 + 60.0  # cap spiral search; timed-out macros keep pos value
            leg = _will_legalize(perturbed_init, movable, sizes, hw, hh, cw, ch, n,
                                 deadline=leg_deadline, order=order)
            t_leg = time.monotonic() - t1
            _log(f"  Restart {k} ({label}) legalized in {t_leg:.1f}s")

            # 2-opt-everywhere tested 2026-05-19, REJECTED. Applied to each
            # cong-grad iter, it produces:
            #   - ibm04: 1.3316 → 1.3201 (−0.0115 improvement ✓)
            #   - ibm06: 1.6684 → 1.6769 (+0.0085 regression ✗)
            #   - ibm02: 1.5923 → 1.5938 (+0.0015 regression ✗)
            # Net sporadic (similar variance pattern as WireMask). Root cause:
            # 2-opt pulls cong-grad-perturbed positions BACK toward their pre-
            # perturbation displacement target, undoing the cong-grad exploration
            # that was supposed to push macros AWAY from congested cells. The
            # cong-grad trajectory depends on consistent perturbation direction
            # across iters; 2-opt's "snap back to target" interferes.
            # 2-opt is still applied to BASELINE legalize (outside this function)
            # where there's no cong-grad trajectory to disrupt.

            t_score_start = time.monotonic()
            score = _score(leg)
            t_score_observed = time.monotonic() - t_score_start
            if t_score_observed > t_one_score:
                t_one_score = t_score_observed
            _log(f"  Candidate {k}: proxy={score:.4f}")
            if score < best_score:
                best_score = score
                best_pl = pl_scratch.clone()  # snapshot only on improvement

            # Safety: if scoring overran the (possibly relaxed) cap, stop immediately
            # rather than launching another restart that would push time further over.
            if time.monotonic() - t0 > cap:
                _log(f"  Over budget after scoring ({time.monotonic()-t0:.0f}s, cap={cap:.0f}s); stopping")
                return False

            return True

        # Density-grad / occupancy-spreading restart only fires for n <= 100,
        # which never occurs on IBM benchmarks (smallest ibm01 has n=246). It
        # also empirically hurt ibm03 (n=126) and ibm08 (n=301) in earlier
        # experiments. Removed 2026-05-19 along with its helpers
        # (_congestion_heatmap, _box_blur, _density_gradient_perturb).
        directed_ran = 0

        # -- Routing-congestion-gradient descent (v8, iterative + wide) --------
        # Phase 1: iterative gradient descent at frac=0.04.
        #   After each improving step, extract the new position from best_pl
        #   and use it (with plc's now-updated congestion map) as the starting
        #   point for the next step. Stops when a step fails to improve or
        #   budget can't fit 3 noise restarts.
        # Phase 2: wide step at frac=0.08 from baseline_pos using current plc.
        #   Only runs if phase 1 improved at least once (otherwise cong-grad
        #   is not useful for this benchmark). Uses rng_cong so main random
        #   state is unchanged and subsequent noise draws are identical to v5.
        rng_cong = np.random.RandomState(self.seed + 1)
        cong_pos = baseline_pos
        cong_improved = False
        cong_frac = 0.04
        for cong_iter in range(12):  # I-change revert: was 15; extra iters shifted ibm01's 2-opt into worse basin
            if cong_iter > 0:
                # Use relaxed cap (matches _try_restart's allow_overrun=True path)
                # so a transient spike on iter=0 doesn't block the whole loop.
                remaining = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                # Larger factor for full-frac iters (reserve for Phase 2 + noise).
                # Smaller factor for adaptive halved-frac retries (only 1 eval needed).
                budget_factor = 3.0 if cong_frac >= 0.04 else 1.5
                if remaining < budget_factor * t_one_score * 1.3:
                    break
            cong_perturbed = _routing_congestion_perturb(
                cong_pos, plc, benchmark, n, cw, ch, hw, hh, movable,
                frac=cong_frac, rng=rng_cong,
            )
            score_before = best_score
            if not _try_restart(f"cong-grad iter={cong_iter + 1} f={cong_frac:.2f}",
                                 cong_perturbed, k=1 + directed_ran,
                                 allow_overrun=True):
                break  # don't kill Phase 2/3 - they have their own budget checks
            directed_ran += 1
            if best_score < score_before:
                cong_pos = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                )
                cong_improved = True
                cong_frac = 0.04  # reset frac on success
            elif cong_improved and cong_frac > 0.01 and cong_iter >= 2:
                # At least 2 prior iterations: try a gentler step before giving up.
                # Guard cong_iter>=2 prevents firing after only 1 success (ibm02 pattern):
                # ibm02 fails at cong_iter=1 → stale plc state critical for Phase 2 wide=8%.
                # ibm03/ibm06 fail at cong_iter=2+ → stale plc less critical, adaptive helps.
                cong_frac *= 0.5
            else:
                break  # plc's map is stale, stop iterating

        # Phase 2: wide steps from baseline using evolved plc congestion state.
        # Loop over [0.08, 0.12]; stop when a step fails to improve or budget
        # runs out. Each step uses the gradient from the current plc state
        # (which encodes where prior iterations struggled), applied with a
        # larger displacement from the original baseline spread.
        if cong_improved:
            for wide_frac in [0.08, 0.12]:
                # Use relaxed cap so Phase 2 still fires after a Phase 1 spike.
                remaining = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                if remaining < t_one_score * 1.3:
                    break
                cong_wide = _routing_congestion_perturb(
                    baseline_pos, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=wide_frac, rng=rng_cong,
                )
                score_before = best_score
                if not _try_restart(f"cong-grad wide={wide_frac:.0%}", cong_wide,
                                     k=1 + directed_ran, allow_overrun=True):
                    break  # don't kill Phase 3 - it has its own check
                directed_ran += 1
                if best_score >= score_before:
                    break  # stop wide steps if this one didn't improve

        # Phase 3: cong-grad from best known position using current (stale) plc.
        # After Phase 2 failed wide steps, plc holds the cong map from a placement
        # that was WORSE than our best. Moving from the BEST position away from the
        # high-congestion regions of this stale map may explore a different local
        # minimum. Only runs when cong-grad improved at least once (cong_improved)
        # so we know the gradient signal is useful for this benchmark.
        # Phase 3: cong-grad from best known position using current (stale) plc.
        # After Phase 2 failed wide steps, plc holds the cong map from a placement
        # that was WORSE than our best. Moving from the BEST position away from the
        # high-congestion regions of this stale map may explore a different local
        # minimum. Only runs when cong-grad improved at least once (cong_improved)
        # so we know the gradient signal is useful for this benchmark.
        #
        # Multi-frac Phase 3 (0.02/0.04/0.06) tested 2026-05-19, REJECTED. f=0.04
        # consistently wins on tested benchmarks (ibm04 1.3316, ibm06 1.6684,
        # ibm02 1.5923, ibm09 1.1304); the extra fracs 0.02/0.06 never found
        # deeper basins. Safe but ineffective; reverted for code clarity.
        if cong_improved:
            # Use relaxed cap so Phase 3 fires after a Phase 1 spike - this is
            # where ibm04's 1.3316 win lives.
            remaining = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if remaining >= t_one_score * 1.3:
                best_pos_now = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                )
                phase3_perturbed = _routing_congestion_perturb(
                    best_pos_now, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=0.04, rng=rng_cong,
                )
                if _try_restart("cong-grad phase3", phase3_perturbed,
                                 k=1 + directed_ran, allow_overrun=True):
                    directed_ran += 1
                # On Phase 3 failure, fall through to noise loop (which will
                # likely also skip on its own strict pre-check)

        # -- Async DREAMPlace check (Phase 5: additive candidates) ------------
        # Multi-DP: iterate over all launched handles. Each completed DP
        # becomes a candidate; the best across all DPs feeds Phase 5b/5c
        # and is also retained in `dp_placements` for Phase 7 (DP-rescue
        # cong-grad as additive tail after the noise loop).
        dp_placements: list[tuple[str, float, torch.Tensor]] = []
        for tag, td, h in dp_handles:
            remaining_dp = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            # 3*t_one_score reserve covers Phase 5b + at least one noise score.
            max_wait = max(0.0, min(remaining_dp - 3.0 * t_one_score, 30.0))
            dp_full = h.wait_for_result_full(max_wait_s=max_wait)
            if dp_full is None:
                _log(f"  DREAMPlace[{tag} td={td:.2f}] not ready "
                     f"(elapsed={h.time_elapsed():.1f}s); killing subprocess")
                h.kill()
                continue
            dp_hard, dp_soft = dp_full
            _log(f"  DREAMPlace[{tag} td={td:.2f}] ready in {h.time_elapsed():.1f}s "
                 f"(hard={dp_hard.shape[0]}, soft={dp_soft.shape[0]}); "
                 f"testing as candidate")
            # Legalize hard macros (DREAMPlace's NLP may leave overlaps).
            # Clip out-of-canvas first: DREAMPlace's macro_place_flag stage
            # can produce positions slightly past canvas.
            t_dp = time.monotonic()
            dp_leg_deadline = t_dp + 60.0
            dp_hard_clip = dp_hard.copy()
            dp_hard_clip[:, 0] = np.clip(dp_hard_clip[:, 0], hw, cw - hw)
            dp_hard_clip[:, 1] = np.clip(dp_hard_clip[:, 1], hh, ch - hh)
            dp_hard_leg = _will_legalize(
                dp_hard_clip, movable, sizes, hw, hh, cw, ch, n,
                deadline=dp_leg_deadline,
            )
            dp_pl = benchmark.macro_positions.clone()
            dp_pl[:n, 0] = torch.tensor(dp_hard_leg[:, 0], dtype=torch.float32)
            dp_pl[:n, 1] = torch.tensor(dp_hard_leg[:, 1], dtype=torch.float32)
            n_soft_dp = int(min(dp_soft.shape[0], benchmark.num_soft_macros))
            if n_soft_dp > 0:
                dp_pl[n:n + n_soft_dp, 0] = torch.tensor(
                    dp_soft[:n_soft_dp, 0], dtype=torch.float32
                )
                dp_pl[n:n + n_soft_dp, 1] = torch.tensor(
                    dp_soft[:n_soft_dp, 1], dtype=torch.float32
                )
            t_dp_score_start = time.monotonic()
            dp_score = float(_exact_proxy(dp_pl, benchmark, plc))
            t_dp_score = time.monotonic() - t_dp_score_start
            if t_dp_score > t_one_score:
                t_one_score = t_dp_score
            directed_ran += 1
            _log(f"  Candidate {directed_ran} (dreamplace[{tag}] hard+soft): "
                 f"proxy={dp_score:.4f}  (leg+score {time.monotonic()-t_dp:.1f}s)")
            # The 2026-05-22 "analytic soft re-snap" experiment (centroid-
            # follow blend on DP candidate softs) was rejected: regressed
            # ibm04 +0.003 and ibm10 +0.002 at every blend factor. Resolved
            # 2026-05-24 by A2: launching DP with soft_movable=True lets
            # DREAMPlace's NLP optimize softs directly (better than analytic
            # post-hoc re-snap). The helpers _build_soft_resnap_cache and
            # _resnap_soft_macros were never copied forward to v2.
            if dp_score < best_score:
                best_score = dp_score
                best_pl = dp_pl.clone()
            dp_placements.append((tag, dp_score, dp_pl))

        # Phase 5b: cong-grad from best_pl using current plc state. plc state
        # reflects whatever was scored last (last DP if any DP scored, else
        # baseline). Perturbing best_pl with this gradient explores basins
        # the original-baseline plc state alone couldn't reach.
        if dp_placements:
            remaining_5b = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if remaining_5b >= t_one_score * 1.3:
                best_pos_now = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                )
                dp_perturbed = _routing_congestion_perturb(
                    best_pos_now, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=0.04, rng=rng_cong,
                )
                if _try_restart("cong-grad-best from-dreamplace-plc f=0.04",
                                 dp_perturbed,
                                 k=1 + directed_ran, allow_overrun=True):
                    directed_ran += 1

        # Phase 6 (cong-grad from DP placement, single per-iter inside Phase 5)
        # tested 2026-05-20, REJECTED for displacing noise restarts that won
        # ibm08 at 6% noise. Phase 7 below revisits this idea but only AFTER
        # the noise loop completes - purely additive on leftover budget.

        # Phase 5c: wide-from-best with current plc state. Fills the slot left
        # by Phase 2 (wide from BASELINE only) and Phase 3/5b (frac=0.04 from
        # BEST only). Uses the latest plc state (post-Phase-5b if DP fired,
        # else post-Phase-3) which encodes the most-recent congestion pattern.
        # Purely additive: fires only if cong-grad helped earlier and budget
        # allows; placed AFTER Phase 5b so no current winning rng_cong path is
        # affected. Noise loop uses np.random directly (not rng_cong), so the
        # extra rng_cong draw here doesn't perturb noise restarts.
        if cong_improved:
            remaining_5c = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if remaining_5c >= t_one_score * 1.3:
                best_pos_5c = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                )
                wide_perturbed = _routing_congestion_perturb(
                    best_pos_5c, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=0.08, rng=rng_cong,
                )
                if _try_restart("cong-grad wide-from-best f=0.08",
                                 wide_perturbed,
                                 k=1 + directed_ran, allow_overrun=True):
                    directed_ran += 1

        # WireMask-BBO with congestion penalty tested 2026-05-19, REJECTED.
        # Helps sparse benchmarks (ibm01 WM=1.1964 vs baseline 1.2253) but hurts
        # dense ones (ibm04 WM=1.5070 vs 1.4101; ibm06 WM=1.8890 vs 1.7197).
        # Root cause: WireMask is constructive - rebuilds from scratch and
        # loses initial.plc's hand-tuned spread that the pipeline operates around.
        # A single alpha can't satisfy all benchmarks (would need per-benchmark
        # tuning, which violates the "no benchmark-specific tweaks" rule).
        # Implementation removed 2026-05-19; see commit 121a555-era history.

        # -- Restarts 1+: Random Gaussian -------------------------------------
        noise_scale_base = min(cw, ch)
        for k, frac in enumerate(
            self.noise_fracs[: self.n_restarts - 1 - directed_ran], start=1 + directed_ran
        ):
            noise = np.random.normal(0, frac * noise_scale_base, init_pos.shape)
            perturbed = np.clip(
                init_pos + noise,
                np.stack([hw, hh], axis=1),
                np.stack([cw - hw, ch - hh], axis=1),
            )
            if not _try_restart(f"random noise={frac:.0%}", perturbed, k=k):
                break

        # -- Phase 7: DP-rescue cong-grad chain (additive, after noise) -------
        # Diagnostic (_dp_diagnostic.py 2026-05-21) showed DP loses on 9/12
        # benchmarks purely on congestion (dC +0.02 to +0.16). 2026-05-21
        # single-iter tests on ibm01/04/07/12/02 confirmed 1 iter is not
        # enough to close gaps that large - the rescue candidates scored
        # WORSE than current best every time, because legalization shuffles
        # macros enough that one gradient step gets reset.
        #
        # Multi-iter (this version): chain up to MAX_P7_ITERS cong-grad
        # iterations per DP placement, each starting from the previous
        # iter's legalized output. Greedy descent: stop the chain when an
        # iter fails to improve over the previous iter (gradient direction
        # is no longer productive). Each iter's plc state reflects the
        # prior iter's scoring, so the gradient is recomputed fresh.
        #
        # Phase 6 (2026-05-20) ran similar multi-iter BEFORE the noise loop
        # and was rejected for displacing noise winners. Phase 7 runs AFTER
        # noise - purely additive, only consumes leftover budget.
        # Phase 7 retro-eval 2026-05-25 (90-iter sample, monotonic-clock
        # --all log): 7 wins / 90 iters = 7.8% hit rate. Big wins on
        # ibm02 (−0.060 at hi-mov iter 3) and ibm10 (−0.07 across lo-fix
        # chain). 13 of 17 benchmarks contribute 0 wins. Iter-1-margin
        # gate (threshold 0.06) abandons chains where iter 1 is far
        # worse than pre-P7 best; preserves all 7 wins (largest winning
        # iter-1 margin was 0.0555) while gating ~14 zero-win chains.
        #
        # RNG isolation 2026-05-25: snapshot rng_cong before Phase 7 and
        # restore after. Without this, the variable-length Phase 7 chains
        # (greedy break, iter-1-margin gate, MAX_P7_ITERS cap) consume
        # rng_cong by different amounts across benchmarks, causing the
        # downstream Phase 8/9 perturbations to diverge - initial gate
        # test showed ibm10 regressed +0.0193 purely from this RNG drift.
        # The isolation makes Phase 7 a closed compartment w.r.t. rng_cong,
        # so changes to Phase 7's internal logic (gating, chain length,
        # adding/removing DPs) no longer affect downstream phases.
        rng_cong_pre_p7 = rng_cong.get_state()
        P7_ITER1_MARGIN_GATE = 0.06  # tested 2026-05-25, see ISSUES.md A5
        MAX_P7_ITERS = 3
        for tag, _dp_score_unused, dp_pl_saved in dp_placements:
            current_pos = np.stack(
                [dp_pl_saved[:n, 0].numpy(), dp_pl_saved[:n, 1].numpy()], axis=1
            ).astype(np.float64)
            prev_iter_score = float("inf")
            pre_chain_best = best_score
            for it in range(1, MAX_P7_ITERS + 1):
                remaining_p7 = (
                    effective_budget_s + BUDGET_OVERRUN_S
                ) - (time.monotonic() - t0)
                if remaining_p7 < t_one_score * 1.3:
                    break
                rescue_perturbed = _routing_congestion_perturb(
                    current_pos, plc, benchmark, n, cw, ch, hw, hh, movable,
                    frac=0.04, rng=rng_cong,
                )
                t1 = time.monotonic()
                leg = _will_legalize(
                    rescue_perturbed, movable, sizes, hw, hh, cw, ch, n,
                    deadline=t1 + 60.0,
                )
                t_leg = time.monotonic() - t1
                directed_ran += 1
                _log(f"  Restart {directed_ran} (cong-grad from-dp[{tag}] "
                     f"iter={it} f=0.04) legalized in {t_leg:.1f}s")
                t_score_start = time.monotonic()
                score = _score(leg)
                t_score_observed = time.monotonic() - t_score_start
                if t_score_observed > t_one_score:
                    t_one_score = t_score_observed
                _log(f"  Candidate {directed_ran}: proxy={score:.4f}")
                if score < best_score:
                    best_score = score
                    best_pl = pl_scratch.clone()
                # Iter-1 margin gate: abandon chain if iter 1 score is
                # far above pre-chain best - empirically those chains
                # don't recover (per Phase 7 retro-eval 2026-05-25).
                if it == 1 and (score - pre_chain_best) > P7_ITER1_MARGIN_GATE:
                    break
                # Greedy descent: stop chain if this iter didn't strictly
                # improve over previous iter's score.
                if score >= prev_iter_score - 1e-4:
                    break
                prev_iter_score = score
                current_pos = leg
                # Hard cap: don't exceed cap after this iter's scoring.
                if time.monotonic() - t0 > effective_budget_s + BUDGET_OVERRUN_S:
                    break

        # RNG isolation (2026-05-25): restore rng_cong to pre-Phase-7 state
        # so Phase 8/9 perturbations are deterministic regardless of how
        # many Phase 7 chain iters fired (iter-1-margin gate, greedy break,
        # MAX_P7_ITERS cap all cause irregular consumption).
        rng_cong.set_state(rng_cong_pre_p7)

        # -- Phase 8: TOP-K cong-grad from best_pl (A6 attack #1, 2026-05-23) -
        # The A3 diagnostic showed DP loses on congestion by avg +0.08 vs our
        # best. Phase 1/2/3/5/7 use the full-mask perturb (every macro in a
        # congested cell moves), which may blunt the gradient on dense
        # benchmarks. Phase 8 tries TOP-K (move only the K hottest macros)
        # from best_pl with a few K values; preserves all prior wins because
        # it runs LAST and only consumes leftover budget.
        #
        # 2026-05-24 (improvement #3): per-K multi-iter chains (like Phase 7
        # but starting from current best_pl). Greedy break-on-no-improvement.
        # Note: single-bench testing showed mixed results - ibm04 −0.0005
        # but ibm10 +0.0020 (regression). Including in combined --all to see
        # if cross-benchmark wins offset.
        MAX_P8_ITERS = 3
        if cong_improved:
            for top_k_val in (5, 10, 20):
                prev_chain_score = best_score
                for chain_iter in range(MAX_P8_ITERS):
                    remaining_p8 = (
                        effective_budget_s + BUDGET_OVERRUN_S
                    ) - (time.monotonic() - t0)
                    if remaining_p8 < t_one_score * 1.3:
                        break
                    best_pos_now = np.stack(
                        [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                    )
                    p8_perturbed = _routing_congestion_perturb(
                        best_pos_now, plc, benchmark, n, cw, ch, hw, hh, movable,
                        frac=0.04, rng=rng_cong, top_k=top_k_val,
                    )
                    if not _try_restart(
                        f"cong-grad-best TOP-{top_k_val} iter={chain_iter+1} f=0.04",
                        p8_perturbed,
                        k=1 + directed_ran, allow_overrun=True,
                    ):
                        break
                    directed_ran += 1
                    if best_score >= prev_chain_score - 1e-4:
                        break
                    prev_chain_score = best_score

        # Phase 9a (fine-noise from best_pl, A6 axis #3) was tested and
        # REVERTED 2026-05-23. Added 4 Gaussian-perturb candidates from
        # best_pl at frac=0.005-0.02. `--all` net result: 0 avg change
        # (ibm14 −0.0005, ibm17 +0.0008, rest within ±0.0001). Small noise
        # + greedy legalize converges back to the same basin most of the
        # time; the pipeline's existing perturbations already cover the
        # productive perturbation magnitudes.

        # -- Phase 9: Random-tiebreak legalize order (A6 axis #4, 2026-05-23) -
        # Default `_will_legalize` order is `sorted(range(n), key=-area)` -
        # largest-area first with index-tied secondary key. For benchmarks
        # with many similar-sized macros (ibm08/09/11/13), the deterministic
        # tiebreaks may lock the placer into one specific legal arrangement.
        # This phase tries N_TRIALS legalize orderings that keep the primary
        # key (-area) but RANDOMIZE the secondary key.
        #
        # Distinct from the rejected "multi-order baseline" (smallest-area,
        # tallest, widest) which changed the primary key - that regressed
        # benchmarks where small-macro-first produced large-macro-trapped
        # placements. Here the primary key is preserved.
        N_ORDER_TRIALS = 3
        area = sizes[:n, 0] * sizes[:n, 1]
        # Phase-9 parallelization (2026-05-29): the N trials are independent
        # legalize-then-score chains. _will_legalize is pure numpy (releases
        # the GIL on the heavy work), so we run the legalize calls
        # concurrently in a thread pool. The score step (_exact_proxy /
        # _score) mutates the shared plc + pl_scratch state and must stay
        # sequential. Saves the legalize time (~0.1-0.3s × N) per benchmark.
        p9_orders: list = []
        for _ in range(N_ORDER_TRIALS):
            # np.lexsort: last key is primary. With (random_key, -area) the
            # primary sort is by -area (largest first), tied entries broken by
            # the uniform random key - different per trial.
            random_key = rng_cong.random(n)
            p9_orders.append(np.lexsort((random_key, -area)).tolist())

        def _p9_legalize(order):
            return _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n,
                                  deadline=time.monotonic() + 60.0, order=order)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=N_ORDER_TRIALS) as _p9_ex:
                p9_legs = list(_p9_ex.map(_p9_legalize, p9_orders))
        except Exception as exc:
            _log(f"  Phase 9 parallel legalize failed ({type(exc).__name__}: {exc}); "
                 f"falling back to sequential")
            p9_legs = [_p9_legalize(o) for o in p9_orders]

        for trial, leg in enumerate(p9_legs):
            remaining_p9 = (
                effective_budget_s + BUDGET_OVERRUN_S
            ) - (time.monotonic() - t0)
            if remaining_p9 < t_one_score * 1.3:
                _log(f"  Skipping P9 trial {trial}+ "
                     f"(budget: {remaining_p9:.0f}s left)")
                break
            t_score_start = time.monotonic()
            score = _score(leg)
            t_score_observed = time.monotonic() - t_score_start
            if t_score_observed > t_one_score:
                t_one_score = t_score_observed
            _log(f"  Restart {1 + directed_ran} (random-order-legalize "
                 f"trial={trial}) proxy={score:.4f}")
            if score < best_score:
                best_score = score
                best_pl = pl_scratch.clone()
            directed_ran += 1
            # Safety: post-score budget guard, same as _try_restart's tail.
            if time.monotonic() - t0 > (effective_budget_s + BUDGET_OVERRUN_S):
                _log(f"  Over budget after P9 trial {trial}; stopping")
                break

        # -- 2-opt swap on cong-grad winner (additive, after Phase 7) ---------
        # Proxy-driven (issue #1, 2026-05-23). Previously this used
        # `_two_opt_swap` (displacement-from-init criterion), which was
        # empirically anti-correlated with proxy on ibm01/04/10 - every
        # documented benchmark had the post-hoc guard reject ALL applied
        # swaps. The 15s budget was wasted. With per-score time at ~5-50ms
        # post-vectorization, scoring each candidate swap directly is
        # affordable. Cheap bounds + conflict checks remain as a free
        # filter so most candidates skip the score call.
        # Phase 7b (DP-basin congestion relief) was prototyped 2026-05-26 and
        # REVERTED - see ISSUES.md. The DP_PROBE ceiling test suggested the best
        # raw DREAMPlace basin could 2-opt below best after a fuller cong-grad
        # descent (ibm10 1.3279 vs 1.3337), but the production descent proved too
        # budget-hungry (~30s/benchmark) AND high-variance - and not even
        # reproducible at fixed seed (plc-state-dependent on where in the pipeline
        # it runs: seed 777 gave 1.3639 post-pipeline but 1.3730 mid-pipeline). It
        # captured zero net gain in-pipeline. The durable finding (DP loses purely
        # on congestion; post-hoc repair can't fix it reliably) points instead at
        # congestion-aware DREAMPlace (congestion in the global objective). The
        # DP_DIAG/DP_PROBE diagnostics are retained (env-gated) to reproduce it.
        remaining_2opt = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
        if remaining_2opt >= t_one_score + 15.0:
            # O2 candidate #2 (2026-05-25): run 2-opt from MULTIPLE basins, not
            # just the single refined best_pl. Raw DP proxy is not predictive of
            # the final 2-opt result (the O2-margin experiment showed keeping the
            # "winning" basin as the only seed can converge worse), so try best_pl
            # plus each DP candidate basin and keep the global minimum. Losing DP
            # basins cost one 2-opt budget each but may 2-opt below the winner.
            twoopt_seeds: list[tuple[str, torch.Tensor, float]] = [
                ("best", best_pl.clone(), best_score)
            ]
            for _tag, _dp_sc, _dp_pl in dp_placements:
                twoopt_seeds.append((f"dp[{_tag}]", _dp_pl.clone(), _dp_sc))
            # S4 baseline_pos seed tested 2026-05-25, REJECTED: 2-opt from the
            # raw legalized baseline never beat best_pl on any of ibm01/04/09/
            # 10/13 (landed 0.02-0.10 above), since baseline is best_pl's
            # unrefined ancestor, not a distinct basin. Pure wall-clock cost.

            # Prune hopeless DP basins. A seed can only beat the incumbent's
            # 2-opt result if its own 2-opt result is lower; for a DP seed whose
            # raw proxy is > DP_SEED_2OPT_WINDOW above best_score, the 2-opt gain
            # needed to catch up exceeds anything observed, so it isn't worth a
            # 15s pass. Both observed basin wins sit well inside this window
            # (ibm04 dp[hi-mov] +0.011, ibm09 dp[hi-fix] +0.002). The "best" seed
            # is never pruned (it reproduces the committed single-seed 2-opt,
            # keeping the change strictly additive).
            DP_SEED_2OPT_WINDOW = 0.02

            # Selection is by TRUE _exact_proxy, never the IncrementalScorer's
            # final_score: the incremental WL drifts seed-dependently (ibm01
            # dp[lo-fix] reported internal 1.1309 but true proxy 1.1506), so
            # cross-seed comparison on the internal score picks phantom winners.
            # The incremental scorer still guides which swaps to accept (speed);
            # we just re-score each finalist exactly before comparing.
            twoopt_best_pl = best_pl
            twoopt_best_score = float(_exact_proxy(best_pl, benchmark, plc))
            _dp_diag_2opt = []  # (seed_tag, true_final, cand) when DP_DIAG set

            # Speedup #3 v2 (2026-05-30): TIME-SHIFTED multi-seed 2-opt
            # subprocess parallelism. The original v1 launched DP seeds in
            # subprocesses BEFORE the inline best-seed 2-opt - but the 4-way
            # CPU contention during the 15s deadlines degraded every thread's
            # search (~+0.005/bench regression, 9/9 worse). v2 instead runs
            # the inline "best" with FULL solo CPU, then fires DP seeds in
            # a parallel pool AFTER best is done. DP seeds contend only with
            # each other (3-way max, less impact). Best-seed quality - which
            # usually wins the multi-seed tournament - is fully preserved.
            # Env-gated: set V2_MULTISEED_MP=1 to enable.
            _use_mp = bool(os.environ.get("V2_MULTISEED_MP"))

            for seed_tag, seed_pl, seed_score in twoopt_seeds:
                # Time-shifted #3 v2: when MP is enabled, the inline loop
                # processes ONLY the "best" seed. DP seeds are handled by the
                # pool launched after this loop (no overlap with best).
                if _use_mp and seed_tag != "best":
                    continue
                rem = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                if rem < 2.0 * t_one_score + 15.0:
                    _log(f"  2-opt seed {seed_tag}: skipped (budget {rem:.0f}s)")
                    break
                if seed_tag != "best" and seed_score > best_score + DP_SEED_2OPT_WINDOW:
                    _log(f"  2-opt seed {seed_tag}: pruned "
                         f"(raw {seed_score:.4f} > best {best_score:.4f} + "
                         f"{DP_SEED_2OPT_WINDOW})")
                    continue
                t_2opt = time.monotonic()
                # S1 (2026-05-26): basin-hopping 2-opt. The 2-opt search only
                # PERMUTES existing macro slots - it can never reach a position
                # no macro currently occupies. After a pass converges to a swap-
                # only local minimum, inject a congestion-gradient KICK (the same
                # _routing_congestion_perturb the cong-grad phases use) to move
                # the hottest macros to NEW continuous positions against the live
                # congestion field, legalize, then run 2-opt again to clean up.
                # Accept-on-true-proxy, keeping the running best across passes.
                #
                # Budget-safe by construction: each pass gets the FULL remaining
                # 15s deadline, and 2-opt returns early only when it converges.
                # On deadline-bound large benchmarks (ibm10/12/16) the first pass
                # eats the whole 15s → no time to kick → behavior is byte-identical
                # to the prior single-pass code. The kicks only fill the otherwise-
                # idle remainder on benchmarks where 2-opt exhausts its candidate
                # pool early.
                # Slice the 15s into passes so 2-opt yields its low-yield tail
                # to a kick + fresh search. Empirically the 2-opt never reaches a
                # local minimum within a full 15s on these benchmarks (always
                # deadline-bound), so a "kick only on early convergence" trigger
                # never fires - slicing is what makes the interleave happen.
                S1_PASS_BUDGET = 5.0      # seconds per 2-opt pass before a kick
                # S1 DORMANT (max_kicks=0). DISPROVEN 2026-05-26: enabling sliced
                # basin-hopping (5s passes + cong-grad kick, max_kicks=2) regressed
                # --all on 6/7 benchmarks before the run was stopped (ibm01 +0.0037,
                # ibm04 +0.0091, ibm08 +0.0045; cumulative +0.025/7). Slicing starves
                # the productive deadline-bound 2-opt search, and the kicks perturb
                # away from the optimum without recovering. The earlier "more accepts"
                # signal (671→1072 on ibm04) was misleading - the extra accepts were
                # repairing kick damage, not net-improving. Code kept for reference.
                S1_MAX_KICKS = 0          # → up to (this+1) passes of ~5s each
                S1_KICK_FRAC = 0.03       # kick magnitude (refinement-scale)
                S1_MIN_REM = 3.0          # need >=this much budget to bother kicking
                global_2opt_deadline = t_2opt + 15.0
                s1_rng = np.random.RandomState(20260526)

                work_pl = seed_pl.clone()
                work_hard = np.stack(
                    [seed_pl[:n, 0].numpy(), seed_pl[:n, 1].numpy()], axis=1
                ).astype(np.float64)
                work_score = seed_score
                seed_best_pl = seed_pl.clone()
                seed_best_score = float("inf")
                accept_count = 0
                score_calls = 0
                final_score = work_score
                n_kicks = 0
                while True:
                    # B3 phase 2/4 IncrementalScorer: incremental WL + congestion.
                    # Re-init per pass from the current working placement (kick
                    # moved positions non-swap-wise, so the prior scorer state is
                    # stale). Init cost is ~3-10ms, negligible vs the 15s budget.
                    try:
                        incremental_scorer = IncrementalScorer(
                            plc, benchmark, work_pl.cpu().numpy().astype(np.float64)
                        )
                    except Exception as exc:
                        _log(f"  IncrementalScorer init failed: {type(exc).__name__}: "
                             f"{exc}; falling back to full scoring")
                        incremental_scorer = None

                    opt_scratch = work_pl.clone()

                    def _2opt_score(pos_arr: np.ndarray, _scr=opt_scratch) -> float:
                        pos32 = torch.from_numpy(np.ascontiguousarray(pos_arr)).float()
                        _scr[:n, 0] = pos32[:, 0]
                        _scr[:n, 1] = pos32[:, 1]
                        return float(_exact_proxy(_scr, benchmark, plc))

                    # S9 (2026-05-26): per-macro local congestion snapshot for
                    # congestion-aware 2-opt (hot-first ordering + cold-region
                    # teleport augmentation). The IncrementalScorer init above
                    # called plc.get_congestion_cost() on work_pl, so plc's
                    # routing map reflects the current placement. cell field is
                    # max(H,V), matching _routing_congestion_perturb.
                    macro_cong = None
                    try:
                        nr_g, nc_g = benchmark.grid_rows, benchmark.grid_cols
                        h_arr = np.asarray(
                            plc.get_horizontal_routing_congestion(), dtype=np.float64
                        )
                        v_arr = np.asarray(
                            plc.get_vertical_routing_congestion(), dtype=np.float64
                        )
                        if h_arr.size == nr_g * nc_g and v_arr.size == nr_g * nc_g:
                            cell_cong = np.maximum(
                                h_arr.reshape(nr_g, nc_g), v_arr.reshape(nr_g, nc_g)
                            )
                            cwc, chc = cw / nc_g, ch / nr_g
                            ci = np.clip(
                                (work_hard[:, 0] / cwc).astype(np.int64), 0, nc_g - 1
                            )
                            ri = np.clip(
                                (work_hard[:, 1] / chc).astype(np.int64), 0, nr_g - 1
                            )
                            macro_cong = cell_cong[ri, ci]
                    except Exception:
                        macro_cong = None

                    # k_neighbors=20 / max_iters=6 (S2, 2026-05-25): per-score
                    # ~3ms post-B3-phase-4, so a wide candidate pool fits. Each
                    # pass is bounded by a time slice (S1_PASS_BUDGET) AND the
                    # global 15s deadline, whichever is sooner.
                    pass_deadline = global_2opt_deadline if S1_MAX_KICKS == 0 else min(
                        global_2opt_deadline, time.monotonic() + S1_PASS_BUDGET
                    )
                    opt_pos, ac, fs, sc = _two_opt_proxy_swap(
                        work_hard, sizes, hw, hh, cw, ch, movable, n,
                        score_fn=_2opt_score, initial_score=work_score,
                        k_neighbors=20, max_iters=6, deadline=pass_deadline,
                        incremental_scorer=incremental_scorer,
                        macro_cong=macro_cong,
                    )
                    accept_count += ac
                    score_calls += sc
                    final_score = fs

                    cand = work_pl.clone()
                    cand[:n, 0] = torch.tensor(opt_pos[:, 0], dtype=torch.float32)
                    cand[:n, 1] = torch.tensor(opt_pos[:, 1], dtype=torch.float32)
                    # _exact_proxy also repopulates plc's routing-congestion map,
                    # which the kick below reads to build its gradient.
                    cand_true = float(_exact_proxy(cand, benchmark, plc))
                    if cand_true < seed_best_score:
                        seed_best_score = cand_true
                        seed_best_pl = cand

                    rem = global_2opt_deadline - time.monotonic()
                    if n_kicks >= S1_MAX_KICKS or rem < S1_MIN_REM:
                        break

                    # Congestion-gradient kick from the just-scored 2-opt result
                    # (plc reflects `cand`), then legalize the perturbed hard
                    # macros. Feed the kicked layout into the next 2-opt pass even
                    # if it scores worse - escaping the swap-only basin is the
                    # whole point; seed_best_pl preserves the best seen so far.
                    kicked = _routing_congestion_perturb(
                        opt_pos, plc, benchmark, n, cw, ch, hw, hh, movable,
                        frac=S1_KICK_FRAC, rng=s1_rng,
                    )
                    kicked_leg = _will_legalize(
                        kicked, movable, sizes, hw, hh, cw, ch, n
                    )
                    work_pl = cand.clone()
                    work_pl[:n, 0] = torch.tensor(kicked_leg[:, 0], dtype=torch.float32)
                    work_pl[:n, 1] = torch.tensor(kicked_leg[:, 1], dtype=torch.float32)
                    work_hard = kicked_leg.astype(np.float64)
                    work_score = float(_exact_proxy(work_pl, benchmark, plc))
                    n_kicks += 1

                cand = seed_best_pl
                true_final = seed_best_score
                scorer_tag = "incr" if incremental_scorer is not None else "full"
                _log(f"  2-opt seed {seed_tag} (proxy/{scorer_tag}): {accept_count} "
                     f"accepts / {score_calls} scores, {n_kicks} kicks, "
                     f"true={true_final:.4f} (was {seed_score:.4f}) "
                     f"in {time.monotonic()-t_2opt:.1f}s")
                if true_final < twoopt_best_score:
                    twoopt_best_score = true_final
                    twoopt_best_pl = cand
                if os.environ.get("DP_DIAG"):
                    _dp_diag_2opt.append((seed_tag, true_final, cand.clone()))

            # Speedup #3 v2: after the inline best-seed 2-opt completes, NOW
            # launch the DP seeds in a parallel subprocess pool. They contend
            # only with each other (~3-way), not with the main thread - and
            # since "best" already ran with full solo CPU, its quality is
            # preserved. Wall: ~15s (best inline) + ~15-18s (DP parallel)
            # = ~30-33s vs ~60s sequential, saves ~27-30s/bench.
            if _use_mp:
                _mp_pool = None
                _mp_futures: list = []
                try:
                    _iccad_path = (Path("external/MacroPlacement/Testcases/ICCAD04")
                                   / benchmark.name)
                    if _iccad_path.exists():
                        _eligible_dp = []
                        for _t, _pl, _sc in twoopt_seeds:
                            if _t == "best":
                                continue
                            if _sc > best_score + DP_SEED_2OPT_WINDOW:
                                _log(f"  2-opt seed {_t}: pruned (raw {_sc:.4f} > "
                                     f"best {best_score:.4f} + {DP_SEED_2OPT_WINDOW})")
                                continue
                            _eligible_dp.append((_t, _pl, _sc))
                        if _eligible_dp:
                            # mp_context="fork" is critical: placer.py is loaded
                            # via importlib by the evaluator's harness, so the
                            # default "spawn" can't pickle our module-level
                            # worker function. Fork inherits the parent's
                            # loaded modules + function references via COW.
                            _mp_pool = concurrent.futures.ProcessPoolExecutor(
                                max_workers=len(_eligible_dp),
                                mp_context=mp.get_context("fork"),
                            )
                            for _t, _pl, _sc in _eligible_dp:
                                _fut = _mp_pool.submit(
                                    _multiseed_2opt_worker,
                                    benchmark.name, str(_iccad_path),
                                    _pl.cpu().numpy().astype(np.float64),
                                    float(_sc), _t,
                                    int(n), float(cw), float(ch),
                                    sizes, hw, hh, movable,
                                    15.0, 20, 6,
                                )
                                _mp_futures.append((_t, _fut))
                            _log(f"  2-opt v2: launched {len(_mp_futures)} DP "
                                 f"seeds in subprocesses (time-shifted: best "
                                 f"already done, no main-thread contention)")
                except Exception as exc:
                    _log(f"  2-opt subprocess pool launch failed: "
                         f"{type(exc).__name__}: {exc}")
                    _mp_pool = None
                    _mp_futures = []

                # Collect DP-seed subprocess results.
                if _mp_pool is not None:
                    for _t, _fut in _mp_futures:
                        try:
                            _res = _fut.result(timeout=60.0)
                            _log(f"  2-opt seed {_t} (proxy/subproc): "
                                 f"{_res['accept_count']} accepts / "
                                 f"{_res['score_calls']} scores, "
                                 f"true={_res['true_final']:.4f}")
                            if _res["true_final"] < twoopt_best_score:
                                twoopt_best_score = _res["true_final"]
                                _opt_full = _res["opt_pos_full"]
                                _cand = best_pl.clone()
                                _cand[:, 0] = torch.tensor(_opt_full[:, 0], dtype=torch.float32)
                                _cand[:, 1] = torch.tensor(_opt_full[:, 1], dtype=torch.float32)
                                twoopt_best_pl = _cand
                            if os.environ.get("DP_DIAG"):
                                _cand_diag = best_pl.clone()
                                _opt_full = _res["opt_pos_full"]
                                _cand_diag[:, 0] = torch.tensor(_opt_full[:, 0], dtype=torch.float32)
                                _cand_diag[:, 1] = torch.tensor(_opt_full[:, 1], dtype=torch.float32)
                                _dp_diag_2opt.append((_t, _res["true_final"], _cand_diag))
                        except Exception as exc:
                            _log(f"  2-opt seed {_t} subprocess failed: "
                                 f"{type(exc).__name__}: {exc}")
                    try:
                        _mp_pool.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass

            if twoopt_best_score < best_score:
                best_score = twoopt_best_score
                best_pl = twoopt_best_pl

        # -- Interleaved relocation ⇄ 2-opt (R2, 2026-05-27) ------------------
        # R1 (relocation: move the hottest macros into empty low-congestion legal
        # gaps - a move the swap-only 2-opt can't make) landed −0.0096 as a single
        # post-2-opt pass. R2 ALTERNATES relocation and 2-opt until neither
        # improves: each relocation opens new swap opportunities (and vice versa),
        # so the two compound. Both reuse the fast incremental scorer and accept
        # only on a strict TRUE-proxy drop, so the loop is strictly non-regressing.
        # Relocation runs first each round (the multi-seed block already 2-opt-
        # converged best_pl, so a fresh 2-opt finds nothing until relocation moves
        # something). Budget-gated; a round needs slack for a relocation (~cheap)
        # + a short 2-opt pass.
        R2_MAX_ROUNDS = 20  # budget-guarded; converges + breaks on no-improvement.
        # (Previously 12: increased to 20 (H-change, 2026-06-01) - budget-gated so
        # no regression risk; allows more rounds on budget-flush fast benchmarks.)
        # R2_HOT/R2_TGT widen the relocation candidate set per round so large
        # benchmarks (ibm10 786 macros) relieve more than ~3% of hot macros/round.
        # I-change REVERTED (2026-06-01): R2_HOT=96 for n>300 tested but reverted.
        # Doubling reloc evals (768→1536) added ~4s/round overhead, reducing R2
        # round throughput (ibm10: 1.1457→1.2029, ibm12: 1.3641→1.4063 regressions).
        # The reloc deadline (15s) doesn't prevent the extra time from eating into
        # partial rounds, cutting ibm12 from 2.4 to 2.0 complete R2 rounds.
        R2_HOT = 48
        R2_TGT = 16
        R2_2OPT_SLICE = 8.0
        # R3b (2026-05-28): softs number 900-2000 but only R2_HOT were tried per
        # round (~16% coverage over 6 rounds on ibm17), so the dominant lever was
        # under-covered. Wider soft candidate set; budget-gated by the pass deadline.
        R3_SOFT_HOT = 128
        R3_SOFT_TGT = 24
        # A+C (2026-05-29, made adaptive 2026-05-30): the cong soft-reloc pass
        # saturates fast (typically by round 3-4) while density keeps finding
        # moves through round 6. C: boost the density pass's candidate set to
        # R3_SOFT_HOT_BOOSTED on rounds where cong has saturated, spending the
        # freed ~4-5s/round on more density attempts. The "cong saturated"
        # trigger is now the SKIP_EMPTY_AFTER mechanism (#2) - adaptive per
        # benchmark, no hardcoded round count. (The earlier hardcoded
        # R3_CONG_MAX_ROUNDS=3 cap is retired; adaptive skip-empty matches the
        # empirical observation without baking the round number in.)
        R3_SOFT_HOT_BOOSTED = 192
        # A3 (2026-05-29): bias soft-pass candidate ordering toward the macro's
        # net centroid (where its connections want it). Pure ordering change -
        # the proxy gate still validates every move, so it's strictly
        # non-regressing. wl_blend=0 reproduces the original nearest-to-current
        # ordering exactly; 0.3 gives a modest pull toward connections so the
        # deadline-bound search tries WL-friendly candidates earlier.
        A3_WL_BLEND = 0.3
        # Soft-macro half-sizes (for the soft relocation pass - R3, 2026-05-28).
        _n_soft = benchmark.num_soft_macros
        _soft_sizes = benchmark.macro_sizes[n:n + _n_soft].numpy().astype(np.float64)
        soft_hw = _soft_sizes[:, 0] / 2
        soft_hh = _soft_sizes[:, 1] / 2
        _soft_movable = benchmark.get_movable_mask().numpy()[n:n + _n_soft]

        def _hard_xy(_pl):
            return np.stack([_pl[:n, 0].numpy(), _pl[:n, 1].numpy()], axis=1).astype(np.float64)

        def _macro_cong_now():
            # Per-macro local max(H,V) from plc's current routing map (set by the
            # IncrementalScorer init / last _exact_proxy on best_pl).
            try:
                nr_g, nc_g = benchmark.grid_rows, benchmark.grid_cols
                ha = np.asarray(plc.get_horizontal_routing_congestion(), dtype=np.float64)
                va = np.asarray(plc.get_vertical_routing_congestion(), dtype=np.float64)
                if ha.size != nr_g * nc_g or va.size != nr_g * nc_g:
                    return None
                cc = np.maximum(ha.reshape(nr_g, nc_g), va.reshape(nr_g, nc_g))
                cwc, chc = cw / nc_g, ch / nr_g
                ci = np.clip((best_pl[:n, 0].numpy() / cwc).astype(np.int64), 0, nc_g - 1)
                ri = np.clip((best_pl[:n, 1].numpy() / chc).astype(np.int64), 0, nr_g - 1)
                return cc[ri, ci]
            except Exception:
                return None

        # Adaptive skip-if-empty (speedup, 2026-05-30): track zero-accept rounds
        # for each new pass added in this layer. After SKIP_EMPTY_AFTER
        # consecutive rounds with no candidate-level accepts, drop the pass
        # for the remainder of the interleave. Defensive guard: most new passes
        # find moves every round, so this rarely fires - but it bounds the
        # worst case (a pass that has converged just sits doing 0-yield work).
        SKIP_EMPTY_AFTER = 1
        _empty_streak = {
            "reloc_density": 0,
            "reloc_combined": 0,
            "soft_reloc_cong": 0,
            "soft2opt_cong": 0,
            "soft2opt_density": 0,
            "hxs_cong": 0,
            "hxs_density": 0,
            "hs3_cong": 0,
            "hs3_density": 0,
        }

        # Adaptive R2 termination (2026-05-30): break out of the round loop on
        # TINY_R2_ROUNDS_TO_STOP consecutive rounds where the proxy Δ is below
        # R2_DELTA_THRESHOLD. Catches the "diminishing returns" tail where the
        # search is technically improving but the gains are below the noise
        # floor of one --all run (~1e-3). The existing `if not round_improved`
        # break still fires for true no-improvement; this fires earlier when
        # improvements are microscopic.
        R2_DELTA_THRESHOLD = 1e-3
        TINY_R2_ROUNDS_TO_STOP = 2
        _r2_tiny_streak = 0

        # Speedup #32 (2026-05-31): persistent shared scorer across the passes
        # of one R2 round. The R2 round body has ~10 distinct passes (hard
        # reloc cong / density / combined, soft reloc cong / density, soft-2opt
        # cong / density × A5 passes, HXS cong / density, 2-opt cleanup); the
        # status-quo code rebuilt an `IncrementalScorer` per pass (full WL,
        # density, routing-flats build - ~0.1-0.3 s each on large benchmarks =
        # ~10-20 s of overhead per benchmark across 6 rounds). Since every
        # accepting move commits via the scorer's commit_* API and every
        # pass calls _exact_proxy(cand) at the end to validate, the scorer
        # naturally stays in sync with `best_pl` whenever the cumulative pass
        # accepts pass the proxy gate. The only time the scorer diverges is
        # when a pass had accepts but `cand_true >= best_score - 1e-6` (the
        # accepts didn't sum to a net improvement) - then we set the dirty
        # flag and the next pass rebuilds. Bit-exact with the per-pass
        # rebuild path because every commit_* is bit-exact with a fresh
        # scorer + replay.
        _round_scorer = [None]   # list-as-mutable-flag (closure-safe)
        _round_scorer_dirty = [False]

        def _round_scorer_get():
            """Return a scorer in sync with `best_pl`. Lazily builds on first
            call, rebuilds when the previous pass marked dirty."""
            if _round_scorer[0] is None or _round_scorer_dirty[0]:
                _exact_proxy(best_pl, benchmark, plc)
                _round_scorer[0] = IncrementalScorer(
                    plc, benchmark, best_pl.cpu().numpy().astype(np.float64)
                )
                _round_scorer_dirty[0] = False
            return _round_scorer[0]

        def _round_scorer_handoff(cand_true, cand):
            """Call from each pass after computing cand_true. Either updates
            best_pl (scorer stays in sync) or marks dirty (next pass rebuilds).
            Returns True iff the pass improved best_pl."""
            nonlocal best_score, best_pl, round_improved
            if cand_true < best_score - 1e-6:
                best_score, best_pl = cand_true, cand
                round_improved = True
                return True
            else:
                # Pass had accepts that committed to the scorer but didn't pass
                # the cumulative gate. Scorer state ≠ best_pl now; force rebuild.
                _round_scorer_dirty[0] = True
                return False

        for _r2 in range(R2_MAX_ROUNDS):
            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 3.0 * t_one_score + 3.0:
                break
            _r2_prev_best = best_score
            round_improved = False
            # Force a fresh scorer at round-start (best_pl may have changed
            # between rounds via cleanup-2-opt commits / accept handling).
            _round_scorer_dirty[0] = True

            # --- Per-round shared scorer (scorer-sharing, 2026-05-30) -----------
            # Previously each pass (reloc, cong, density, 2-opt) built its own
            # IncrementalScorer (~0.5–1s) and called _exact_proxy (~0.2s) before
            # starting - 4×/round × 6 rounds = ~18s of pure overhead. Now we build
            # ONE scorer per round and reuse it across all passes. After each pass
            # commits moves, the scorer's state is already at the new best position
            # (the post-accept _exact_proxy keeps plc in sync). Only if a verify
            # unexpectedly fails do we rebuild from the known-good best_pl.
            #
            # K-change (2026-06-01): skip the round-start _exact_proxy for rounds
            # 1+ (index ≥ 1). Analysis shows plc is ALWAYS synced to best_pl at
            # every round boundary:
            #   - Reloc/cong/density accepts: verify calls _exact_proxy(cand=best_pl) → sync.
            #   - Reloc/cong/density fail-verify: explicit _exact_proxy(best_pl) resync.
            #   - Reloc/cong/density 0 moves: plc unchanged = previous sync.
            #   - 2-opt accepts: _exact_proxy(cand) → if improved: plc=new best_pl;
            #     if verify fails: explicit _exact_proxy(best_pl) resync.
            #   - 2-opt 0 swaps: 2-opt uses incremental scorer (NOT _exact_proxy) →
            #     plc unchanged from density/cong/reloc verify → still synced.
            #   - Budget breaks BETWEEN passes: last pass's verify left plc synced.
            # Round 0 always re-scores: the outer 2-opt leaves plc at the last
            # scored kick candidate (not best_pl after multi-seed processing).
            # Savings: (n_rounds - 1) × t_one_score per benchmark.
            # For ibm17 (t_one_score≈5s, 4 rounds): ~15s freed for 1 more round.
            _r2_base = None
            _r2_shared = None
            try:
                t_rel = time.monotonic()
                base_rel = float(_exact_proxy(best_pl, benchmark, plc))
                rel_scorer = _round_scorer_get()
                # R4 (WL-aware hard-relocation targeting) was DISPROVEN 2026-05-29:
                # biasing targets toward the net centroid steered the interleave to
                # a slightly WORSE local min (ibm03 +0.0015, ibm07 +0.0025) with no
                # upside. Reverted to the nearest-to-current sort. The `wl_blend`
                # option + `hard_net_centroids()` + `WLAWARE_PROBE` are kept as inert
                # diagnostic scaffolding (see ISSUES.md).
                rel_pos, rel_acc, _ = _relocation_moves(
                    _hard_xy(best_pl), sizes, hw, hh, cw, ch, movable, n, plc,
                    benchmark, rel_scorer, base_rel,
                    deadline=t_rel + min(rem_r2 - t_one_score, 15.0),
                    top_hot=R2_HOT, n_targets=R2_TGT,
                )
                if rel_acc > 0:
                    cand = best_pl.clone()
                    cand[:n, 0] = torch.tensor(rel_pos[:, 0], dtype=torch.float32)
                    cand[:n, 1] = torch.tensor(rel_pos[:, 1], dtype=torch.float32)
                    rel_true = float(_exact_proxy(cand, benchmark, plc))
                    if rel_true < best_score - 1e-6:
                        _log(f"  R2 round {_r2+1} reloc[cong]: {rel_acc} moves, "
                             f"{best_score:.4f} → {rel_true:.4f}")
                    _round_scorer_handoff(rel_true, cand)
            except Exception as exc:
                _log(f"  R2 relocation[cong] failed: {type(exc).__name__}: {exc}")
                _round_scorer_dirty[0] = True

            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 2.0 * t_one_score + 2.0:
                break

            # --- H5 (2026-05-29): hard DENSITY relocation pass (analog of R5
            # for hards). Hot hard macros in densest cells → lowest-density
            # cold cells. Same _relocation_moves but with use_density=True,
            # which switches the hot/cold field to scorer.grid_occupied.
            # Skip-if-empty (2026-05-30): drop after a zero-accept round -
            # H5 typically finds 0-3 moves; once it stops finding anything,
            # the basin's flat for hards on the density field and the ~0.7s
            # per round is wasted budget.
            if _empty_streak["reloc_density"] < SKIP_EMPTY_AFTER:
                try:
                    t_rel_d = time.monotonic()
                    base_rel_d = float(_exact_proxy(best_pl, benchmark, plc))
                    rel_scorer_d = _round_scorer_get()
                    rel_pos_d, rel_acc_d, _ = _relocation_moves(
                        _hard_xy(best_pl), sizes, hw, hh, cw, ch, movable, n, plc,
                        benchmark, rel_scorer_d, base_rel_d,
                        deadline=t_rel_d + min(rem_r2 - t_one_score, 15.0),
                        top_hot=R2_HOT, n_targets=R2_TGT,
                        use_density=True,
                    )
                    if rel_acc_d == 0:
                        _empty_streak["reloc_density"] += 1
                    else:
                        _empty_streak["reloc_density"] = 0
                    if rel_acc_d > 0:
                        cand = best_pl.clone()
                        cand[:n, 0] = torch.tensor(rel_pos_d[:, 0], dtype=torch.float32)
                        cand[:n, 1] = torch.tensor(rel_pos_d[:, 1], dtype=torch.float32)
                        rel_true_d = float(_exact_proxy(cand, benchmark, plc))
                        if rel_true_d < best_score - 1e-6:
                            _log(f"  R2 round {_r2+1} reloc[density]: {rel_acc_d} moves, "
                                 f"{best_score:.4f} → {rel_true_d:.4f}")
                        _round_scorer_handoff(rel_true_d, cand)
                except Exception as exc:
                    _log(f"  R2 relocation[density] failed: {type(exc).__name__}: {exc}")
                    _round_scorer_dirty[0] = True

            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 2.0 * t_one_score + 2.0:
                break

            # --- R6 (2026-05-30): hard COMBINED cong+density relocation pass.
            # Hotness = geometric mean of normalized cong & density. Catches
            # macros moderately hot on both fields that neither pure pass
            # prioritized (each ranks pure-field extremes higher). Same proxy
            # gate. Skip-if-empty after consecutive empty rounds. ---
            if _empty_streak["reloc_combined"] < SKIP_EMPTY_AFTER:
                try:
                    t_rel_c = time.monotonic()
                    base_rel_c = float(_exact_proxy(best_pl, benchmark, plc))
                    rel_scorer_c = _round_scorer_get()
                    rel_pos_c, rel_acc_c, _ = _relocation_moves(
                        _hard_xy(best_pl), sizes, hw, hh, cw, ch, movable, n, plc,
                        benchmark, rel_scorer_c, base_rel_c,
                        deadline=t_rel_c + min(rem_r2 - t_one_score, 4.0),
                        top_hot=R2_HOT, n_targets=R2_TGT,
                        use_combined=True,
                    )
                    if rel_acc_c == 0:
                        _empty_streak["reloc_combined"] += 1
                    else:
                        _empty_streak["reloc_combined"] = 0
                    if rel_acc_c > 0:
                        cand = best_pl.clone()
                        cand[:n, 0] = torch.tensor(rel_pos_c[:, 0], dtype=torch.float32)
                        cand[:n, 1] = torch.tensor(rel_pos_c[:, 1], dtype=torch.float32)
                        rel_true_c = float(_exact_proxy(cand, benchmark, plc))
                        if rel_true_c < best_score - 1e-6:
                            _log(f"  R2 round {_r2+1} reloc[combined]: {rel_acc_c} moves, "
                                 f"{best_score:.4f} → {rel_true_c:.4f}")
                        _round_scorer_handoff(rel_true_c, cand)
                except Exception as exc:
                    _log(f"  R2 relocation[combined] failed: {type(exc).__name__}: {exc}")
                    _round_scorer_dirty[0] = True

            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 2.0 * t_one_score + 2.0:
                break

            # --- soft relocation passes: hot soft clusters → cold cells, by the
            # CONGESTION field (R3, 2026-05-28) then the DENSITY field (R5,
            # 2026-05-28). Softs are the bulk of BOTH the routing-congestion and
            # density terms; the cong pass converges then the density pass finds
            # MORE moves it missed (DENS_SOFT_PROBE on best_pl: cong=0 relocs, but
            # density 22–68 relocs for −0.011 to −0.020, all in the density term).
            # Softs may overlap → no legality check. Accept-on-true-proxy.
            # Scorer-sharing: no new IncrementalScorer or _exact_proxy here -
            # _r2_shared is already at the post-reloc committed state, and plc
            # was synced by the reloc verify call (or round-start if reloc=0). ---
            for _sfield, _use_d in (("cong", False), ("density", True)):
                if _n_soft <= 0:
                    break
                # Adaptive cong cap (replaces hardcoded R3_CONG_MAX_ROUNDS for
                # the single-soft cong-relocation pass, 2026-05-30): use the
                # skip-if-empty pattern (#2) instead - drop the cong pass after
                # SKIP_EMPTY_AFTER consecutive zero-accept rounds. This matches
                # the empirical observation (cong saturates at round 3-4 on
                # most benchmarks) without hardcoding a number.
                if _sfield == "cong" and _empty_streak["soft_reloc_cong"] >= SKIP_EMPTY_AFTER:
                    continue
                rem_sr = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                if rem_sr < 2.0 * t_one_score + 2.0:
                    break
                # C: when the cong pass has been skipped (saturated), the
                # density pass gets a wider candidate set (top_hot 128 → 192).
                # Trigger is adaptive: based on the cong-pass empty streak.
                _cong_saturated = _empty_streak["soft_reloc_cong"] >= SKIP_EMPTY_AFTER
                _top_hot_this = (
                    R3_SOFT_HOT_BOOSTED
                    if (_sfield == "density" and _cong_saturated)
                    else R3_SOFT_HOT
                )
                _n_tgt_this = (
                    16
                    if (_sfield == "density" and _cong_saturated)
                    else R3_SOFT_TGT
                )
                try:
                    t_sr = time.monotonic()
                    base_sr = float(_exact_proxy(best_pl, benchmark, plc))
                    sr_scorer = _round_scorer_get()
                    sr_pos = np.stack(
                        [best_pl[n:n + _n_soft, 0].numpy(),
                         best_pl[n:n + _n_soft, 1].numpy()], axis=1
                    ).astype(np.float64)
                    # A3: precompute soft net centroids on the scorer (~ms for
                    # ~2000 softs × ~10 pins each). With the shared-scorer
                    # speedup the centroids may be recomputed per pass; if it
                    # shows up in profile, cache on the scorer instance.
                    _soft_centroids = sr_scorer.soft_net_centroids()
                    sr_pos, sr_acc, _ = _soft_relocation_moves(
                        sr_pos, soft_hw, soft_hh, cw, ch, n, plc, benchmark,
                        sr_scorer, base_sr,
                        deadline=t_sr + min(rem_sr - t_one_score, 15.0),
                        top_hot=_top_hot_this, n_targets=_n_tgt_this,
                        soft_movable=_soft_movable, use_density=_use_d,
                        net_centroid=_soft_centroids, wl_blend=A3_WL_BLEND,
                    )
                    # Track per-field empty streak for the adaptive cong cap.
                    if _sfield == "cong":
                        if sr_acc == 0:
                            _empty_streak["soft_reloc_cong"] += 1
                        else:
                            _empty_streak["soft_reloc_cong"] = 0
                    if sr_acc > 0:
                        cand = best_pl.clone()
                        cand[n:n + _n_soft, 0] = torch.tensor(sr_pos[:, 0], dtype=torch.float32)
                        cand[n:n + _n_soft, 1] = torch.tensor(sr_pos[:, 1], dtype=torch.float32)
                        sr_true = float(_exact_proxy(cand, benchmark, plc))
                        if sr_true < best_score - 1e-6:
                            _log(f"  R2 round {_r2+1} soft-reloc[{_sfield}]: {sr_acc} "
                                 f"moves, {best_score:.4f} → {sr_true:.4f}")
                        _round_scorer_handoff(sr_true, cand)
                except Exception as exc:
                    _log(f"  R2 soft-reloc[{_sfield}] failed: {type(exc).__name__}: {exc}")
                    _round_scorer_dirty[0] = True

            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 2.0 * t_one_score + 2.0:
                break

            # --- A1 (2026-05-29): soft-soft 2-opt swap. Single soft relocation
            # can't find moves where two softs need to EXCHANGE places. This
            # pair-swap pass adds that move type.
            # A1b + A1×2: run TWICE per round, once with the cong hotness field
            # and once with density - same dual-field symmetry that gave R3 +
            # R5 their compound gain (the two fields find different hot softs
            # and so different beneficial swaps).
            # A1c: each pass appends n_cold_teleports=4 globally-coldest movable
            # softs to the kNN candidate set, so the search can find long-range
            # exchanges (analog of S9 cold-teleport for hard 2-opt).
            # A5 (2026-05-30): adaptive multi-pass soft-2opt. Wrap each field's
            # A1 call in a small inner loop (up to A5_NUM_PASSES). Each pass
            # picks fresh hot softs from the updated state, so a second pass
            # can find chains the first missed. Break on no committed swaps.
            # Adaptive cong cap (2026-05-30): the hardcoded R3_CONG_MAX_ROUNDS
            # gate has been REMOVED for A1b - observed data shows soft-2opt[cong]
            # finds 7-35 swaps per round even at round 6 (unlike single-soft
            # cong-relocation, which does saturate at round 3). Skip-if-empty
            # (the #2 mechanism) is the right adaptive gate for A1b.
            A5_NUM_PASSES = 2
            for _ssfield, _ssuse_d in (("cong", False), ("density", True)):
                if _n_soft < 2:
                    break
                _streak_key = "soft2opt_cong" if _ssfield == "cong" else "soft2opt_density"
                if _empty_streak[_streak_key] >= SKIP_EMPTY_AFTER:
                    continue
                for _a5_pass in range(A5_NUM_PASSES):
                    try:
                        rem_ss = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                        if rem_ss < 2.0 * t_one_score + 2.0:
                            break
                        t_ss = time.monotonic()
                        base_ss = float(_exact_proxy(best_pl, benchmark, plc))
                        ss_scorer = _round_scorer_get()
                        ss_pos = np.stack(
                            [best_pl[n:n + _n_soft, 0].numpy(),
                             best_pl[n:n + _n_soft, 1].numpy()], axis=1
                        ).astype(np.float64)
                        # A4: pass net centroids + wl_blend for WL-aware kNN
                        # ordering (analog of A3 for the swap).
                        _ss_centroids = ss_scorer.soft_net_centroids()
                        ss_pos, ss_acc, _ = _two_opt_soft_swap(
                            ss_pos, cw, ch, n, plc, benchmark, ss_scorer, base_ss,
                            deadline=t_ss + min(rem_ss - t_one_score, 6.0),
                            top_hot=64, k_neighbors=12,
                            soft_movable=_soft_movable,
                            use_density=_ssuse_d, n_cold_teleports=4,
                            net_centroid=_ss_centroids, wl_blend=A3_WL_BLEND,
                        )
                        # Track empty streak only on the FIRST pass of the round
                        # (subsequent passes are bonus search).
                        if _a5_pass == 0:
                            if ss_acc == 0:
                                _empty_streak[_streak_key] += 1
                            else:
                                _empty_streak[_streak_key] = 0
                        _improved_this_pass = False
                        if ss_acc > 0:
                            cand = best_pl.clone()
                            cand[n:n + _n_soft, 0] = torch.tensor(ss_pos[:, 0], dtype=torch.float32)
                            cand[n:n + _n_soft, 1] = torch.tensor(ss_pos[:, 1], dtype=torch.float32)
                            ss_true = float(_exact_proxy(cand, benchmark, plc))
                            if ss_true < best_score - 1e-6:
                                _log(f"  R2 round {_r2+1} soft-2opt[{_ssfield}]"
                                     f"{'' if _a5_pass == 0 else f' pass{_a5_pass+1}'}: "
                                     f"{ss_acc} swaps, {best_score:.4f} → {ss_true:.4f}")
                                _improved_this_pass = True
                            _round_scorer_handoff(ss_true, cand)
                        # A5 early-stop: if this pass found no improving moves,
                        # don't bother with another pass on the same field.
                        if not _improved_this_pass:
                            break
                    except Exception as exc:
                        _log(f"  R2 soft-2opt[{_ssfield}] failed: {type(exc).__name__}: {exc}")
                        _round_scorer_dirty[0] = True
                        break

            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 2.0 * t_one_score + 2.0:
                break

            # --- HXS (2026-05-30): hard ⇄ soft cross-swap. Exchanges a hard
            # macro and a soft macro's positions. Neither the hard-2opt nor
            # the soft-2opt can find such pairs because they only swap within
            # their own kind. Same accept-on-true-proxy machinery via
            # score_swap_hard_soft / commit_swap_hard_soft (bit-exact
            # verified). Dual-field (cong then density), skip-if-empty.
            for _xfield, _xuse_d in (("cong", False), ("density", True)):
                if _n_soft < 1:
                    break
                _xstreak = "hxs_cong" if _xfield == "cong" else "hxs_density"
                if _empty_streak[_xstreak] >= SKIP_EMPTY_AFTER:
                    continue
                rem_x = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                if rem_x < 2.0 * t_one_score + 2.0:
                    break
                try:
                    t_x = time.monotonic()
                    base_x = float(_exact_proxy(best_pl, benchmark, plc))
                    x_scorer = _round_scorer_get()
                    x_hard_pos = _hard_xy(best_pl)
                    x_soft_pos = np.stack(
                        [best_pl[n:n + _n_soft, 0].numpy(),
                         best_pl[n:n + _n_soft, 1].numpy()], axis=1
                    ).astype(np.float64)
                    x_hard_pos, x_soft_pos, x_acc, _ = _two_opt_hard_soft_swap(
                        x_hard_pos, x_soft_pos, sizes, hw, hh, cw, ch,
                        movable, n, plc, benchmark, x_scorer, base_x,
                        deadline=t_x + min(rem_x - t_one_score, 2.5),
                        top_hot=24, k_neighbors=12,
                        soft_movable=_soft_movable, use_density=_xuse_d,
                    )
                    if x_acc == 0:
                        _empty_streak[_xstreak] += 1
                    else:
                        _empty_streak[_xstreak] = 0
                    if x_acc > 0:
                        cand = best_pl.clone()
                        cand[:n, 0] = torch.tensor(x_hard_pos[:, 0], dtype=torch.float32)
                        cand[:n, 1] = torch.tensor(x_hard_pos[:, 1], dtype=torch.float32)
                        cand[n:n + _n_soft, 0] = torch.tensor(x_soft_pos[:, 0], dtype=torch.float32)
                        cand[n:n + _n_soft, 1] = torch.tensor(x_soft_pos[:, 1], dtype=torch.float32)
                        x_true = float(_exact_proxy(cand, benchmark, plc))
                        if x_true < best_score - 1e-6:
                            _log(f"  R2 round {_r2+1} HXS[{_xfield}]: {x_acc} swaps, "
                                 f"{best_score:.4f} → {x_true:.4f}")
                        _round_scorer_handoff(x_true, cand)
                except Exception as exc:
                    _log(f"  R2 HXS[{_xfield}] failed: {type(exc).__name__}: {exc}")
                    _round_scorer_dirty[0] = True

            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 2.0 * t_one_score + 2.0:
                break

            # --- HS3 (2026-05-31): hard-soft-soft 3-cycle rotation.
            # Extension of HXS to 3-cycles: H → S1 → S2 → H. Captures
            # configurations where a hard wants S1's slot but swapping
            # H↔S1 alone hurts because S1's connections need to go
            # elsewhere - 2-opt can't accept the chain individually,
            # but the single combined 3-cycle move can. Same dual-field +
            # skip-if-empty pattern. Cubic cost in (top_hot × k_inner²)
            # → tight 3s deadline cap.
            for _h3field, _h3use_d in (("cong", False), ("density", True)):
                if _n_soft < 2:
                    break
                _h3streak = "hs3_cong" if _h3field == "cong" else "hs3_density"
                if _empty_streak[_h3streak] >= SKIP_EMPTY_AFTER:
                    continue
                rem_h3 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                if rem_h3 < 2.0 * t_one_score + 2.0:
                    break
                try:
                    t_h3 = time.monotonic()
                    base_h3 = float(_exact_proxy(best_pl, benchmark, plc))
                    h3_scorer = _round_scorer_get()
                    h3_hard_pos = _hard_xy(best_pl)
                    h3_soft_pos = np.stack(
                        [best_pl[n:n + _n_soft, 0].numpy(),
                         best_pl[n:n + _n_soft, 1].numpy()], axis=1
                    ).astype(np.float64)
                    h3_hard_pos, h3_soft_pos, h3_acc, _ = _three_opt_hard_soft_soft(
                        h3_hard_pos, h3_soft_pos, sizes, hw, hh, cw, ch,
                        movable, n, plc, benchmark, h3_scorer, base_h3,
                        deadline=t_h3 + min(rem_h3 - t_one_score, 3.0),
                        top_hot=15, k_inner=5,
                        soft_movable=_soft_movable, use_density=_h3use_d,
                    )
                    if h3_acc == 0:
                        _empty_streak[_h3streak] += 1
                    else:
                        _empty_streak[_h3streak] = 0
                    if h3_acc > 0:
                        cand = best_pl.clone()
                        cand[:n, 0] = torch.tensor(h3_hard_pos[:, 0], dtype=torch.float32)
                        cand[:n, 1] = torch.tensor(h3_hard_pos[:, 1], dtype=torch.float32)
                        cand[n:n + _n_soft, 0] = torch.tensor(h3_soft_pos[:, 0], dtype=torch.float32)
                        cand[n:n + _n_soft, 1] = torch.tensor(h3_soft_pos[:, 1], dtype=torch.float32)
                        h3_true = float(_exact_proxy(cand, benchmark, plc))
                        if h3_true < best_score - 1e-6:
                            _log(f"  R2 round {_r2+1} HS3[{_h3field}]: {h3_acc} cycles, "
                                 f"{best_score:.4f} → {h3_true:.4f}")
                        _round_scorer_handoff(h3_true, cand)
                except Exception as exc:
                    _log(f"  R2 HS3[{_h3field}] failed: {type(exc).__name__}: {exc}")
                    _round_scorer_dirty[0] = True

            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 2.0 * t_one_score + 2.0:
                break

            # --- 2-opt cleanup pass (swaps around the relocated macros) ---
            # Scorer-sharing: reuses _r2_shared (post-reloc+cong+density state).
            try:
                t_2o = time.monotonic()
                base_2o = float(_exact_proxy(best_pl, benchmark, plc))
                o_scorer = _round_scorer_get()
                o_scratch = best_pl.clone()

                def _r2_score(pos_arr, _scr=o_scratch):
                    p32 = torch.from_numpy(np.ascontiguousarray(pos_arr)).float()
                    _scr[:n, 0] = p32[:, 0]
                    _scr[:n, 1] = p32[:, 1]
                    return float(_exact_proxy(_scr, benchmark, plc))

                o_pos, o_acc, _o_fs, _o_sc = _two_opt_proxy_swap(
                    _hard_xy(best_pl), sizes, hw, hh, cw, ch, movable, n,
                    score_fn=_r2_score, initial_score=base_2o, k_neighbors=20,
                    max_iters=6, deadline=t_2o + min(rem_r2 - t_one_score, R2_2OPT_SLICE),
                    incremental_scorer=_r2_shared, macro_cong=_macro_cong_now(),
                )
                if o_acc > 0:
                    cand = best_pl.clone()
                    cand[:n, 0] = torch.tensor(o_pos[:, 0], dtype=torch.float32)
                    cand[:n, 1] = torch.tensor(o_pos[:, 1], dtype=torch.float32)
                    o_true = float(_exact_proxy(cand, benchmark, plc))
                    if o_true < best_score - 1e-6:
                        _log(f"  R2 round {_r2+1} 2-opt: {o_acc} swaps, "
                             f"{best_score:.4f} → {o_true:.4f}")
                    _round_scorer_handoff(o_true, cand)
            except Exception as exc:
                _log(f"  R2 2-opt failed: {type(exc).__name__}: {exc}")
                _round_scorer_dirty[0] = True

            if not round_improved:
                break
            # Adaptive R2 round termination: break on consecutive tiny rounds.
            _r2_delta = _r2_prev_best - best_score
            if _r2_delta < R2_DELTA_THRESHOLD:
                _r2_tiny_streak += 1
                if _r2_tiny_streak >= TINY_R2_ROUNDS_TO_STOP:
                    _log(f"  R2 round {_r2+1}: tiny Δ={_r2_delta:.5f} for "
                         f"{_r2_tiny_streak} rounds (< {R2_DELTA_THRESHOLD}); "
                         f"stopping interleave early")
                    break
            else:
                _r2_tiny_streak = 0

        # -- N-change (2026-06-02): Post-R2 soft-reloc using leftover budget ---------
        # Phase 11 (post-R2 cong-grad hard-macro perturb → legalize → exact_proxy)
        # is replaced with a soft-reloc[cong] + soft-reloc[density] pass that uses
        # the same leftover budget (~2–3 × t_one_score) that R2's round-break guard
        # leaves behind.
        #
        # Why Phase 11 was ineffective:
        #   (a) For benchmarks where t_one_score is small, R2 exhausts the entire
        #       budget inside its round loop and Phase 11 exits immediately (rem ≈ 0).
        #   (b) For benchmarks where t_one_score is larger (ibm15–18), R2's round-
        #       break guard fires with ~2–3 × t_one_score remaining. Phase 11 runs
        #       1–2 score evaluations, all rejected: the legalization cascade from
        #       cong-grad perturb destroys the R2-optimized soft macro layout, so
        #       candidates score ~0.25–0.35 worse than R2's best.
        #
        # Why post-R2 soft-reloc works:
        #   After R2 exits mid-round (budget cut during soft-reloc or 2-opt), the
        #   hard-macro state from the last reloc pass is not yet fully exploited.
        #   A soft-reloc[cong] + soft-reloc[density] pass using the leftover time
        #   continues exactly where R2 left off (strictly accept-only, no legalization,
        #   no score wasted on rejections). Observed in R2 late rounds: 10–50 moves,
        #   −0.002 to −0.015 improvement per pass.
        #
        # plc is guaranteed synced to best_pl at R2 exit: K-change analysis proved
        # every R2 exit path calls _exact_proxy(best_pl) - either via an accept
        # verify, a fail-verify resync, or the round-0 scorer init. No resync needed.
        if _n_soft > 0:
            for _post_field, _post_ud in (("cong", False), ("density", True)):
                rem_post = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                # Need ≥ 1.5 × t_one_score: ~1 × t_one_score for the verify call
                # after soft-reloc + 0.5 × margin for scorer init and the pass itself.
                if rem_post < t_one_score * 1.5:
                    break
                try:
                    _post_base = best_score
                    _post_shared = IncrementalScorer(
                        plc, benchmark, best_pl.cpu().numpy().astype(np.float64)
                    )
                    _post_sr_pos = np.stack(
                        [best_pl[n:n + _n_soft, 0].numpy(),
                         best_pl[n:n + _n_soft, 1].numpy()], axis=1
                    ).astype(np.float64)
                    t_post = time.monotonic()
                    _post_max = min(rem_post - t_one_score * 1.0, 15.0)
                    if _post_max < 0.5:
                        break
                    _post_sr_pos, _post_acc, _ = _soft_relocation_moves(
                        _post_sr_pos, soft_hw, soft_hh, cw, ch, n, plc, benchmark,
                        _post_shared, _post_base,
                        deadline=t_post + _post_max,
                        top_hot=1024, n_targets=4,
                        soft_movable=_soft_movable, use_density=_post_ud,
                    )
                    if _post_acc > 0:
                        _post_cand = best_pl.clone()
                        _post_cand[n:n + _n_soft, 0] = torch.tensor(
                            _post_sr_pos[:, 0], dtype=torch.float32)
                        _post_cand[n:n + _n_soft, 1] = torch.tensor(
                            _post_sr_pos[:, 1], dtype=torch.float32)
                        _post_true = float(_exact_proxy(_post_cand, benchmark, plc))
                        if _post_true < best_score - 1e-6:
                            _log(f"  Post-R2 soft-reloc[{_post_field}]: {_post_acc} moves, "
                                 f"{best_score:.4f} -> {_post_true:.4f}")
                            best_score = _post_true
                            best_pl = _post_cand
                        else:
                            # Verify failed: restore plc to best_pl.
                            float(_exact_proxy(best_pl, benchmark, plc))
                except Exception as _post_exc:
                    _log(f"  Post-R2 soft-reloc[{_post_field}] failed: {_post_exc}")
                    try:
                        float(_exact_proxy(best_pl, benchmark, plc))
                    except Exception:
                        pass

        # DP_DIAG (2026-05-26): decompose where the DP basin loses to "best".
        # Logs the WEIGHTED proxy split (wl, 0.5*den, 0.5*cong) for each raw DP
        # candidate, each cong-grad+2-opt-from-seed result, and the final best.
        # Re-scores placements (mutates plc), so done last, right before return.
        if os.environ.get("DP_DIAG"):
            _log("  [DP_DIAG] ---- raw DP candidates (pre cong-grad/2-opt) ----")
            for _t, _sc, _pl in dp_placements:
                p, w, d, c = _proxy_decomp(_pl, benchmark, plc)
                _log(f"  [DP_DIAG] raw dp[{_t}]: proxy={p:.4f}  wl={w:.4f} "
                     f"den={d:.4f} cong={c:.4f}")
            if "_dp_diag_2opt" in locals():
                _log("  [DP_DIAG] ---- after cong-grad+2-opt from each seed ----")
                for _t, _tf, _pl in _dp_diag_2opt:
                    p, w, d, c = _proxy_decomp(_pl, benchmark, plc)
                    _log(f"  [DP_DIAG] 2opt[{_t}]: proxy={p:.4f}  wl={w:.4f} "
                         f"den={d:.4f} cong={c:.4f}")
            p, w, d, c = _proxy_decomp(best_pl, benchmark, plc)
            _log(f"  [DP_DIAG] FINAL best: proxy={p:.4f}  wl={w:.4f} "
                 f"den={d:.4f} cong={c:.4f}")

        if os.environ.get("DP_PROBE"):
            _dp_recoverability_probe(
                dp_placements, best_score, n, cw, ch, hw, hh, sizes,
                movable, plc, benchmark,
            )

        # RELOC_PROBE (2026-05-27): congestion-directed relocation moves on the
        # final best_pl. Builds a fresh scorer, runs _relocation_moves, reports
        # the true proxy delta + decomposition. Diagnostic only (no production
        # change) - measure whether relocations beat the 2-opt result.
        if os.environ.get("RELOC_PROBE"):
            try:
                t_rp = time.monotonic()
                base = float(_exact_proxy(best_pl, benchmark, plc))
                bw = float(plc.get_cost()); bd = 0.5 * float(plc.get_density_cost())
                bc = 0.5 * float(plc.get_congestion_cost())
                rscorer = IncrementalScorer(plc, benchmark, best_pl.cpu().numpy().astype(np.float64))
                rpos = np.stack([best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1).astype(np.float64)
                rpos, racc, rsc = _relocation_moves(
                    rpos, sizes, hw, hh, cw, ch, movable, n, plc, benchmark,
                    rscorer, base, deadline=t_rp + 20.0,
                )
                rcand = best_pl.clone()
                rcand[:n, 0] = torch.tensor(rpos[:, 0], dtype=torch.float32)
                rcand[:n, 1] = torch.tensor(rpos[:, 1], dtype=torch.float32)
                rp, rw, rd, rc = _proxy_decomp(rcand, benchmark, plc)
                verdict = "BEATS best" if rp < base - 1e-4 else "no gain"
                _log(f"  [RELOC_PROBE] base={base:.4f} (wl={bw:.4f} den={bd:.4f} "
                     f"cong={bc:.4f}) -> {racc} relocs -> proxy={rp:.4f} "
                     f"(wl={rw:.4f} den={rd:.4f} cong={rc:.4f}) {verdict} "
                     f"in {time.monotonic()-t_rp:.1f}s")
            except Exception as exc:
                _log(f"  [RELOC_PROBE] failed: {type(exc).__name__}: {exc}")

        # SOFT_RELOC_PROBE (2026-05-28): relocate hot SOFT clusters into low-
        # congestion cells (R1 idea on softs). Targets the soft/net-dominated
        # benchmarks (ibm17/18) that hard relocation can't help. Diagnostic only.
        if os.environ.get("SOFT_RELOC_PROBE"):
            try:
                t_sp = time.monotonic()
                n_soft = benchmark.num_soft_macros
                base = float(_exact_proxy(best_pl, benchmark, plc))
                bw = float(plc.get_cost()); bd = 0.5 * float(plc.get_density_cost())
                bc = 0.5 * float(plc.get_congestion_cost())
                sscorer = IncrementalScorer(plc, benchmark, best_pl.cpu().numpy().astype(np.float64))
                spos = np.stack([best_pl[n:n + n_soft, 0].numpy(),
                                 best_pl[n:n + n_soft, 1].numpy()], axis=1).astype(np.float64)
                ssz = benchmark.macro_sizes[n:n + n_soft].numpy().astype(np.float64)
                spos, sacc, _ssc = _soft_relocation_moves(
                    spos, ssz[:, 0] / 2, ssz[:, 1] / 2, cw, ch, n, plc, benchmark,
                    sscorer, base, deadline=t_sp + 30.0,
                )
                scand = best_pl.clone()
                scand[n:n + n_soft, 0] = torch.tensor(spos[:, 0], dtype=torch.float32)
                scand[n:n + n_soft, 1] = torch.tensor(spos[:, 1], dtype=torch.float32)
                sp, sw, sd, sc = _proxy_decomp(scand, benchmark, plc)
                verdict = "BEATS best" if sp < base - 1e-4 else "no gain"
                _log(f"  [SOFT_RELOC_PROBE] base={base:.4f} (wl={bw:.4f} den={bd:.4f} "
                     f"cong={bc:.4f}) -> {sacc} soft relocs -> proxy={sp:.4f} "
                     f"(wl={sw:.4f} den={sd:.4f} cong={sc:.4f}) {verdict} "
                     f"in {time.monotonic()-t_sp:.1f}s")
            except Exception as exc:
                import traceback
                _log(f"  [SOFT_RELOC_PROBE] failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()

        # WLAWARE_PROBE (2026-05-28): A/B WL-aware HARD relocation target selection
        # on best_pl. For each wl_blend, run one hard relocation pass from the SAME
        # best_pl (fresh scorer) and report the resulting true proxy. blend=0 is the
        # current nearest-to-current sort; >0 biases toward each macro's net
        # centroid (WL anchor). Isolates the target-selection effect. Diagnostic.
        if os.environ.get("WLAWARE_PROBE"):
            try:
                base = float(_exact_proxy(best_pl, benchmark, plc))
                _log(f"  [WLAWARE_PROBE] base={base:.4f}")
                for blend in (0.0, 0.5, 1.0):
                    t_wp = time.monotonic()
                    wscorer = IncrementalScorer(plc, benchmark, best_pl.cpu().numpy().astype(np.float64))
                    cent = wscorer.hard_net_centroids()
                    wpos = np.stack([best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1).astype(np.float64)
                    wpos, wacc, _ = _relocation_moves(
                        wpos, sizes, hw, hh, cw, ch, movable, n, plc, benchmark,
                        wscorer, base, deadline=t_wp + 20.0, top_hot=48, n_targets=16,
                        net_centroid=cent, wl_blend=blend,
                    )
                    wcand = best_pl.clone()
                    wcand[:n, 0] = torch.tensor(wpos[:, 0], dtype=torch.float32)
                    wcand[:n, 1] = torch.tensor(wpos[:, 1], dtype=torch.float32)
                    wp, ww, wd, wc = _proxy_decomp(wcand, benchmark, plc)
                    _log(f"  [WLAWARE_PROBE] blend={blend:.1f}: {wacc} relocs -> "
                         f"proxy={wp:.4f} (wl={ww:.4f} den={wd:.4f} cong={wc:.4f}) "
                         f"Δ={wp-base:+.4f} in {time.monotonic()-t_wp:.1f}s")
            except Exception as exc:
                import traceback
                _log(f"  [WLAWARE_PROBE] failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()

        # DENS_SOFT_PROBE (R5, 2026-05-28): is there DENSITY headroom in the softs
        # that the congestion-targeted soft pass (R3) missed? On best_pl (where the
        # cong soft pass is already converged), run a DENSITY-targeted soft
        # relocation (hot = softs in the densest cells, target = low-density cells)
        # and report the true-proxy delta + decomposition. If it beats 0, a
        # density-aware soft pass is worth adding to production. Diagnostic only.
        if os.environ.get("DENS_SOFT_PROBE"):
            try:
                _n_soft = benchmark.num_soft_macros
                if _n_soft > 0:
                    base = float(_exact_proxy(best_pl, benchmark, plc))
                    bw = float(plc.get_cost()); bd = 0.5 * float(plc.get_density_cost())
                    bc = 0.5 * float(plc.get_congestion_cost())
                    _ssz = benchmark.macro_sizes[n:n + _n_soft].numpy().astype(np.float64)
                    _smov = benchmark.get_movable_mask().numpy()[n:n + _n_soft]
                    for tag, use_d in (("cong", False), ("density", True)):
                        t_dp = time.monotonic()
                        dscorer = IncrementalScorer(plc, benchmark, best_pl.cpu().numpy().astype(np.float64))
                        dpos = np.stack([best_pl[n:n + _n_soft, 0].numpy(),
                                         best_pl[n:n + _n_soft, 1].numpy()], axis=1).astype(np.float64)
                        dpos, dacc, _ = _soft_relocation_moves(
                            dpos, _ssz[:, 0] / 2, _ssz[:, 1] / 2, cw, ch, n, plc, benchmark,
                            dscorer, base, deadline=t_dp + 30.0, top_hot=96, n_targets=24,
                            soft_movable=_smov, use_density=use_d,
                        )
                        dcand = best_pl.clone()
                        dcand[n:n + _n_soft, 0] = torch.tensor(dpos[:, 0], dtype=torch.float32)
                        dcand[n:n + _n_soft, 1] = torch.tensor(dpos[:, 1], dtype=torch.float32)
                        dp, dw, dd, dc = _proxy_decomp(dcand, benchmark, plc)
                        _log(f"  [DENS_SOFT_PROBE] field={tag:7s}: {dacc} relocs -> "
                             f"proxy={dp:.4f} (wl={dw:.4f} den={dd:.4f} cong={dc:.4f}) "
                             f"Δ={dp-base:+.4f} in {time.monotonic()-t_dp:.1f}s")
                    _log(f"  [DENS_SOFT_PROBE] base={base:.4f} (wl={bw:.4f} den={bd:.4f} cong={bc:.4f})")
            except Exception as exc:
                import traceback
                _log(f"  [DENS_SOFT_PROBE] failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()

        _log(f"  Best proxy={best_score:.4f}  total={time.monotonic()-t0:.1f}s")
        self._total_place_time_s += time.monotonic() - t0
        self._benchmarks_done += 1
        return best_pl
