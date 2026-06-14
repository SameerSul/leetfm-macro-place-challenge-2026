"""Main macro-placement pipeline."""

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

from placer.config import _GPU_BACKEND, _GPU_DEVICE_NAME, _log
from placer.legalize.spiral import _will_legalize
from placer.legalize.swap import _two_opt_swap
from placer.local_search.fields import _congestion_field
from placer.local_search.hard_soft import _three_opt_hard_soft_soft, _two_opt_hard_soft_swap
from placer.local_search.lsmc_explore import _explore_enabled, _lsmc_explore
from placer.local_search.relocation import _relocation_moves, _soft_relocation_moves
from placer.local_search.soft_moves import _two_opt_soft_swap
from placer.local_search.two_opt import _two_opt_proxy_swap
from placer.local_search.workers import _multiseed_2opt_worker
from placer.ml.data_collection import get_candidate_trace
from placer.ml.shadow import is_filter_enabled
from placer.perturb.congestion_gradient import _routing_congestion_perturb
from placer.plc.loader import _load_plc
from placer.scoring.exact import _exact_proxy
from placer.scoring.incremental import IncrementalScorer

_TRUE_ENV = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_FALSE_ENV = {"", "0", "false", "FALSE", "no", "NO", "off", "OFF"}


def _reloc_propose_all_enabled(raw: str | None, gpu_backend: str) -> bool:
    """Return whether hard relocation should use the propose-all path."""
    value = (raw or "").strip()
    if value in _TRUE_ENV:
        return True
    if value in _FALSE_ENV:
        return False
    if value.lower() in {"auto", "cuda", "gpu"}:
        return gpu_backend == "cuda"
    return False


class MacroPlacer:
    """Budgeted placer with restarts, DREAMPlace seeds, and local search."""

    def __init__(
        self,
        n_restarts: int = 50,
        noise_fracs: Optional[List[float]] = None,
        seed: int = 42,
        time_budget_s: float = 150.0,
    ):
        self.n_restarts = n_restarts
        # Noise sizes for restart placements. Order affects seeded randomness.
        self.noise_fracs = noise_fracs or [
            0.02, 0.04, 0.06, 0.08,
            0.01, 0.03, 0.05, 0.07, 0.09,
            0.06, 0.06, 0.04,
            0.10, 0.12, 0.08,
            0.025, 0.035, 0.045, 0.055, 0.065, 0.075,
            0.15, 0.20, 0.10,
            0.05, 0.06, 0.07, 0.03, 0.04, 0.02,
            0.005, 0.010, 0.015, 0.030, 0.050,
        ]
        self.seed = seed
        self.time_budget_s = time_budget_s
        # Optional override for experiments and slower/faster machines.
        _env_budget = os.environ.get("V2_TIME_BUDGET")
        if _env_budget:
            try:
                self.time_budget_s = float(_env_budget)
            except ValueError:
                pass

        # Track budget across an evaluate --all run.
        self._first_place_call_time: Optional[float] = None
        self._benchmarks_done: int = 0
        # Count only time spent inside place().
        self._total_place_time_s: float = 0.0
        self.HARNESS_TOTAL_BUDGET_S: float = 3300.0
        self.HARNESS_TOTAL_BENCHMARKS: int = 17
        # Directed phases may overrun the soft budget by this much.
        self.BUDGET_OVERRUN_S: float = 83.0
        # Reserve enough time so late benchmarks are not starved.
        self.PER_BENCH_FLOOR_S: float = 110.0
        self.HARD_CAP_SAFE_S: float = 3540.0

    def _effective_budget(self, t0: float) -> "tuple[float, float]":
        """Return this benchmark's soft budget and total used time."""
        if self._first_place_call_time is None:
            self._first_place_call_time = t0
        cumulative_elapsed = self._total_place_time_s
        if self._benchmarks_done >= 1:
            remaining_total = self.HARNESS_TOTAL_BUDGET_S - cumulative_elapsed
            remaining_benchmarks = max(
                1, self.HARNESS_TOTAL_BENCHMARKS - self._benchmarks_done
            )
            reserve_others = (
                (self.PER_BENCH_FLOOR_S + self.BUDGET_OVERRUN_S)
                * (remaining_benchmarks - 1)
            )
            this_cap = remaining_total - reserve_others - self.BUDGET_OVERRUN_S
            effective_budget_s = min(
                self.time_budget_s, max(self.PER_BENCH_FLOOR_S, this_cap)
            )
            hard_headroom = (
                self.HARD_CAP_SAFE_S - cumulative_elapsed - self.BUDGET_OVERRUN_S
            )
            effective_budget_s = min(effective_budget_s, hard_headroom)
        else:
            effective_budget_s = self.time_budget_s
        return effective_budget_s, cumulative_elapsed

    def _launch_dreamplace_seeds(self, benchmark: Benchmark, plc) -> list:
        """Start DREAMPlace seed runs in the background when available."""
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
                    # Try a few settings to get different starting layouts.
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
        return dp_handles

    @staticmethod
    def _clamp_in_bounds(pl: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Keep movable macro centers inside the canvas."""
        sizes = benchmark.macro_sizes
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        hw = sizes[:, 0] / 2.0
        hh = sizes[:, 1] / 2.0
        mov = benchmark.get_movable_mask().to(torch.bool)
        out = pl.clone()
        cx = torch.minimum(torch.maximum(out[:, 0], hw), cw - hw)
        cy = torch.minimum(torch.maximum(out[:, 1], hh), ch - hh)
        out[:, 0] = torch.where(mov, cx, out[:, 0])
        out[:, 1] = torch.where(mov, cy, out[:, 1])
        return out

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Final guard for any soft macro positions copied from input or DP.
        return self._clamp_in_bounds(self._place_impl(benchmark), benchmark)

    def _place_impl(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        random.seed(self.seed)

        _log(f"[GPU] backend={_GPU_BACKEND} device={_GPU_DEVICE_NAME} | benchmark={benchmark.name}")

        t0 = time.monotonic()
        n = benchmark.num_hard_macros
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n].numpy()
        init_pos = benchmark.macro_positions[:n].numpy().copy().astype(np.float64)

        effective_budget_s, cumulative_elapsed = self._effective_budget(t0)

        _log(f"  [{benchmark.name}] hard={n}  movable={movable.sum()}  "
             f"budget={effective_budget_s:.0f}s"
             + (f"  (--all cumulative={cumulative_elapsed:.0f}s, "
                f"done={self._benchmarks_done}/{self.HARNESS_TOTAL_BENCHMARKS})"
                if self._benchmarks_done >= 1 else ""))

        _ml_trace = get_candidate_trace()
        if _ml_trace is not None:
            _ml_trace.start_benchmark(
                benchmark=benchmark,
                seed=self.seed,
                effective_budget_s=effective_budget_s,
                benchmark_index=self._benchmarks_done,
                config={
                    "n_restarts": self.n_restarts,
                    "noise_fracs": self.noise_fracs,
                    "time_budget_s": self.time_budget_s,
                    "budget_overrun_s": self.BUDGET_OVERRUN_S,
                },
            )
            _ml_trace.set_context(phase="pipeline", elapsed_s=0.0)

        def _ml_finish(reason: str, final_score=None) -> None:
            if _ml_trace is not None:
                _ml_trace.set_context(
                    phase="complete",
                    elapsed_s=time.monotonic() - t0,
                    current_best_score=final_score,
                )
                _ml_trace.event("benchmark_end", reason=reason, final_score=final_score)
                _ml_trace.flush()

        # Exact-scoring cutoffs; the slow-score guard still handles bad load.
        EXACT_MACRO_THRESHOLD = 10000
        EXACT_GRID_CELL_LIMIT = 10000
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

        # Shared buffer reused for candidate scores.
        pl_scratch = benchmark.macro_positions.clone()

        def _score(pos: np.ndarray) -> float:
            """Update pl_scratch with hard-macro positions and return exact proxy.

            Caller must clone pl_scratch immediately if it needs to persist the
            result - the next _score call overwrites it.
            """
            pos32 = torch.from_numpy(np.ascontiguousarray(pos)).float()
            pl_scratch[:n, 0] = pos32[:, 0]
            pl_scratch[:n, 1] = pos32[:, 1]
            return float(_exact_proxy(pl_scratch, benchmark, plc))

        # Launch DREAMPlace seeds while the main pipeline runs.
        dp_handles = self._launch_dreamplace_seeds(benchmark, plc)

        # Baseline
        _log("  Restart 0 (baseline)...")
        t1 = time.monotonic()
        baseline_pos = _will_legalize(init_pos, movable, sizes, hw, hh, cw, ch, n)
        _log(f"    Legalized in {time.monotonic()-t1:.1f}s")

        # Fill scratch with baseline positions.
        pl_scratch[:n, 0] = torch.tensor(baseline_pos[:, 0], dtype=torch.float32)
        pl_scratch[:n, 1] = torch.tensor(baseline_pos[:, 1], dtype=torch.float32)

        # Without exact scoring, return the legalized baseline after a legal
        # displacement-reducing 2-opt pass.
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

            # On large cases, compare only the first DREAMPlace seed if time remains.
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
                    # Skip DP comparison when another exact score is unlikely to fit.
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
                            dp_score_large = float(_exact_proxy(dp_pl_large, benchmark, plc))
                            _log(f"  [large-DP] dreamplace exact proxy={dp_score_large:.4f}  "
                                 f"(leg+score {time.monotonic()-t_dp_leg:.1f}s)")
                            if dp_score_large < base_score:
                                _log(f"  [large-DP] DP wins ({dp_score_large:.4f} < "
                                     f"{base_score:.4f}); returning DP placement")
                                _log(f"  total={time.monotonic()-t0:.1f}s")
                                self._total_place_time_s += time.monotonic() - t0
                                _ml_finish("large_dp_early_return", float(dp_score_large))
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
            _ml_finish("insufficient_budget")
            self._benchmarks_done += 1
            return pl_scratch  # safe: no more in-place writes will happen

        # Last-resort guard: do not start a restart that cannot finish.
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
            _ml_finish("exact_scoring_unavailable")
            self._benchmarks_done += 1
            return pl_scratch

        t_score0 = time.monotonic()
        best_score = float(_exact_proxy(pl_scratch, benchmark, plc))
        t_one_score = time.monotonic() - t_score0
        best_pl = pl_scratch.clone()
        _log(f"  Candidate 0: proxy={best_score:.4f}  (scored in {t_one_score:.1f}s)")

        # Safety net: if the first exact score is slow (CPU contention), it would
        # eat the whole per-benchmark budget, so return baseline instead.
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
            _ml_finish("baseline_score_too_slow", best_score)
            self._benchmarks_done += 1
            return best_pl

        # Directed restarts get a small overrun allowance; noise restarts do not.
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
            # t_one_score is a running max over observed scoring times (x1.3 for
            # score + legalize). Running max, not baseline-only: --all CPU
            # contention makes scores 3-5x slower, and a stale estimate would
            # approve restarts that then overrun.
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

            # No 2-opt here: it pulls cong-grad-perturbed positions back toward
            # their target, undoing the exploration away from congested cells.
            # 2-opt runs only on the baseline (no cong-grad trajectory to undo).

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

        directed_ran = 0

        # Routing-congestion restarts use their own RNG so noise draws stay stable.
        rng_cong = np.random.RandomState(self.seed + 1)
        cong_pos = baseline_pos
        cong_improved = False
        cong_frac = 0.04
        for cong_iter in range(12):
            if cong_iter > 0:
                # Match the relaxed cap used by directed restarts.
                remaining = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                # Smaller retry steps need less reserve.
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
                break
            directed_ran += 1
            if best_score < score_before:
                cong_pos = np.stack(
                    [best_pl[:n, 0].numpy(), best_pl[:n, 1].numpy()], axis=1
                )
                cong_improved = True
                cong_frac = 0.04  # reset frac on success
            elif cong_improved and cong_frac > 0.01 and cong_iter >= 2:
                # Try a gentler step before giving up.
                cong_frac *= 0.5
            else:
                break  # plc's map is stale, stop iterating

        # Wide steps from baseline after an improving congestion pass.
        if cong_improved:
            for wide_frac in [0.08, 0.12]:
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
                    break
                directed_ran += 1
                if best_score >= score_before:
                    break  # stop wide steps if this one didn't improve

        # Congestion step from the current best placement.
        if cong_improved:
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

        # Score any DREAMPlace seeds that finished in time.
        dp_placements: list[tuple[str, float, torch.Tensor]] = []
        for tag, td, h in dp_handles:
            remaining_dp = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            # Keep enough reserve for follow-up search.
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
            # Legalize hard macros (DREAMPlace's NLP may leave overlaps); clip
            # out-of-canvas first (macro_place_flag can push slightly past canvas).
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
            if dp_score < best_score:
                best_score = dp_score
                best_pl = dp_pl.clone()
            dp_placements.append((tag, dp_score, dp_pl))

        # Congestion step after DREAMPlace candidates.
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

        # Wide step from current best.
        # Stage 4: "wide-from-best f=0.08" restart PRUNED by default for pipeline
        # simplification. NB the paired gate showed keep is marginally BETTER
        # (seed1 +0.0014, seed2 partial +0.0051) — 5c does real work on
        # ibm09/12/17, so this trades a near-noise score regression for a leaner
        # pipeline. Restore with V2_PRUNE_P5C=0.
        _prune_p5c = os.environ.get("V2_PRUNE_P5C", "1").strip() not in {"0", "false", "off"}
        if cong_improved and not _prune_p5c:
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

        # Random Gaussian restarts.
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

        # Try short congestion chains from DREAMPlace placements.
        rng_cong_pre_p7 = rng_cong.get_state()
        P7_ITER1_MARGIN_GATE = 0.06
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
                # Drop chains that start too far behind the current best.
                if it == 1 and (score - pre_chain_best) > P7_ITER1_MARGIN_GATE:
                    break
                # Greedy descent: stop if this iter didn't improve over the last.
                if score >= prev_iter_score - 1e-4:
                    break
                prev_iter_score = score
                current_pos = leg
                # Hard cap after scoring.
                if time.monotonic() - t0 > effective_budget_s + BUDGET_OVERRUN_S:
                    break

        # Restore RNG so later steps do not depend on DP-chain length.
        rng_cong.set_state(rng_cong_pre_p7)

        # Move only the hottest macros in short greedy chains.
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

        # Retry legalization with random tie-breaks.
        N_ORDER_TRIALS = 3
        area = sizes[:n, 0] * sizes[:n, 1]
        # Legalize trials can run in parallel; scoring remains sequential.
        p9_orders: list = []
        for _ in range(N_ORDER_TRIALS):
            # Largest first, random order within equal-area groups.
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

        # -- Proxy-driven 2-opt swap on the cong-grad winner (additive) ---------
        # Bounds and conflict checks filter candidates before proxy scoring.
        remaining_2opt = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
        # Stage 4: multi-seed 2-opt is PRUNED by default — paired gate (2/2
        # seeds, mean -0.0029, best 1.1169) showed it is net-harmful, steering
        # R2 into worse basins on more benchmarks than it helps. R2 now starts
        # from the cong-grad best_pl. Restore the phase with V2_PRUNE_MULTISEED_2OPT=0.
        _prune_ms2opt = os.environ.get("V2_PRUNE_MULTISEED_2OPT", "1").strip() not in {"0", "false", "off"}
        if _prune_ms2opt:
            _log("  multi-seed 2-opt: PRUNED (default; set V2_PRUNE_MULTISEED_2OPT=0 to restore)")
        if not _prune_ms2opt and remaining_2opt >= t_one_score + 15.0:
            # Run 2-opt from MULTIPLE basins (best_pl + each DP basin), not just
            # best_pl: raw DP proxy doesn't predict the final 2-opt result, so a
            # losing basin can still 2-opt below the winner. Keep the global min.
            twoopt_seeds: list[tuple[str, torch.Tensor, float]] = [
                ("best", best_pl.clone(), best_score)
            ]
            for _tag, _dp_sc, _dp_pl in dp_placements:
                twoopt_seeds.append((f"dp[{_tag}]", _dp_pl.clone(), _dp_sc))

            # Prune hopeless DP basins: a seed whose raw proxy is >DP_SEED_2OPT_WINDOW
            # above best_score can't close the gap. "best" is never pruned.
            DP_SEED_2OPT_WINDOW = 0.02

            # Compare seeds by TRUE _exact_proxy, not the scorer's final_score: the
            # incremental WL drifts per-seed, so cross-seed comparison on it picks
            # phantom winners. The scorer still guides which swaps to accept.
            twoopt_best_pl = best_pl
            twoopt_best_score = float(_exact_proxy(best_pl, benchmark, plc))
            # Optional time-shifted multi-seed parallelism (env-gated): run only
            # "best" inline with solo CPU, then the DP seeds in a pool after.
            # Running all seeds at once degraded every search.
            _use_mp = bool(os.environ.get("V2_MULTISEED_MP"))

            for seed_tag, seed_pl, seed_score in twoopt_seeds:
                # With MP on, DP seeds are handled by the pool after this loop.
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
                global_2opt_deadline = t_2opt + 15.0

                work_pl = seed_pl.clone()
                work_hard = np.stack(
                    [seed_pl[:n, 0].numpy(), seed_pl[:n, 1].numpy()], axis=1
                ).astype(np.float64)
                work_score = seed_score
                seed_best_pl = seed_pl.clone()
                seed_best_score = float("inf")
                accept_count = 0
                score_calls = 0
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

                # S9: per-macro local max(H,V) snapshot for congestion-aware
                # 2-opt ordering and cold-region teleport augmentation.
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

                if _ml_trace is not None:
                    _ml_trace.set_context(
                        phase="multi_seed_2opt",
                        pass_name="hard_2opt",
                        seed_tag=seed_tag,
                        elapsed_s=time.monotonic() - t0,
                        remaining_budget_s=rem,
                        current_best_score=work_score,
                    )
                opt_pos, ac, _fs, sc = _two_opt_proxy_swap(
                    work_hard, sizes, hw, hh, cw, ch, movable, n,
                    score_fn=_2opt_score, initial_score=work_score,
                    k_neighbors=20, max_iters=6, deadline=global_2opt_deadline,
                    incremental_scorer=incremental_scorer,
                    macro_cong=macro_cong,
                )
                accept_count += ac
                score_calls += sc

                cand = work_pl.clone()
                cand[:n, 0] = torch.tensor(opt_pos[:, 0], dtype=torch.float32)
                cand[:n, 1] = torch.tensor(opt_pos[:, 1], dtype=torch.float32)
                cand_true = float(_exact_proxy(cand, benchmark, plc))
                if cand_true < seed_best_score:
                    seed_best_score = cand_true
                    seed_best_pl = cand

                cand = seed_best_pl
                true_final = seed_best_score
                scorer_tag = "incr" if incremental_scorer is not None else "full"
                _log(f"  2-opt seed {seed_tag} (proxy/{scorer_tag}): {accept_count} "
                     f"accepts / {score_calls} scores, true={true_final:.4f} "
                     f"(was {seed_score:.4f}) "
                     f"in {time.monotonic()-t_2opt:.1f}s")
                if true_final < twoopt_best_score:
                    twoopt_best_score = true_final
                    twoopt_best_pl = cand

            # After the inline best-seed 2-opt, run the DP seeds in a subprocess
            # pool (they contend only with each other; "best" already had solo
            # CPU). ~30-33s vs ~60s sequential, saves ~27-30s/bench.
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
                            # fork (not spawn): inherits loaded modules + sys.path,
                            # skipping re-import / numba recompile. The worker calls
                            # _force_worker_cpu() first (CUDA can't survive fork).
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

        # R2: keep trying local moves while they still improve the score.
        R2_MAX_ROUNDS = 20
        R2_HOT = 48
        R2_TGT = 16
        _ml_hard_reloc_targets = os.environ.get("ML_HARD_RELOCATION_N_TARGETS")
        if _ml_hard_reloc_targets:
            try:
                R2_TGT = max(R2_TGT, int(_ml_hard_reloc_targets))
            except ValueError:
                _log(
                    f"  ignoring invalid ML_HARD_RELOCATION_N_TARGETS="
                    f"{_ml_hard_reloc_targets!r}"
                )
        if is_filter_enabled("hard_relocation"):
            _log(
                f"  R2 hard relocation ML filter on (pool={R2_TGT}, "
                f"top_k={os.environ.get('ML_FILTER_TOP_K') or 'all'})"
            )
        _reloc_propose_all_raw = os.environ.get("V2_RELOC_PROPOSE_ALL", "")
        R2_RELOC_PROPOSE_ALL = _reloc_propose_all_enabled(
            _reloc_propose_all_raw,
            _GPU_BACKEND,
        )
        if (
            _reloc_propose_all_raw.strip().lower() in {"auto", "cuda", "gpu"}
            and not R2_RELOC_PROPOSE_ALL
        ):
            _log(
                "  R2 hard relocation propose-all auto disabled "
                f"(backend={_GPU_BACKEND})"
            )
        R2_RELOC_PROPOSE_TOP_M = None
        _reloc_propose_top_m = os.environ.get("V2_RELOC_PROPOSE_TOP_M")
        if _reloc_propose_top_m:
            try:
                R2_RELOC_PROPOSE_TOP_M = max(1, int(_reloc_propose_top_m))
            except ValueError:
                _log(
                    f"  ignoring invalid V2_RELOC_PROPOSE_TOP_M="
                    f"{_reloc_propose_top_m!r}"
                )
        if R2_RELOC_PROPOSE_ALL:
            _log(
                "  R2 hard relocation propose-all enabled "
                f"(top_m={R2_RELOC_PROPOSE_TOP_M or 'all'})"
            )
        R2_2OPT_SLICE = 8.0

        def _env_int(name, default):
            v = os.environ.get(name)
            try:
                return max(1, int(v)) if v not in (None, "") else default
            except ValueError:
                _log(f"  ignoring invalid {name}={v!r}")
                return default
        R3_SOFT_HOT = _env_int("V2_SOFT_HOT", 128)
        R3_SOFT_TGT = _env_int("V2_SOFT_TGT", 32)
        # Density keeps helping after congestion moves dry up.
        R3_SOFT_HOT_BOOSTED = _env_int("V2_SOFT_HOT_BOOSTED", 192)
        # Soft move ordering blends distance-to-current and distance-to-net-center.
        A3_WL_BLEND = 0.3
        _n_soft = benchmark.num_soft_macros
        _soft_sizes = benchmark.macro_sizes[n:n + _n_soft].numpy().astype(np.float64)
        soft_hw = _soft_sizes[:, 0] / 2
        soft_hh = _soft_sizes[:, 1] / 2
        _soft_movable = benchmark.get_movable_mask().numpy()[n:n + _n_soft]

        def _hard_xy(_pl):
            return np.stack([_pl[:n, 0].numpy(), _pl[:n, 1].numpy()], axis=1).astype(np.float64)

        def _macro_cong_now():
            # Read each hard macro's current congestion cell.
            try:
                nr_g, nc_g = benchmark.grid_rows, benchmark.grid_cols
                cc = _congestion_field(plc, nr_g, nc_g)
                if cc is None:
                    return None
                cwc, chc = cw / nc_g, ch / nr_g
                ci = np.clip((best_pl[:n, 0].numpy() / cwc).astype(np.int64), 0, nc_g - 1)
                ri = np.clip((best_pl[:n, 1].numpy() / chc).astype(np.int64), 0, nr_g - 1)
                return cc[ri, ci]
            except Exception:
                return None

        # Stop a pass after repeated rounds with no accepted moves.
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

        # Stop when rounds are no longer making useful progress.
        R2_DELTA_THRESHOLD = 1e-3
        R2_DELTA_HARD_STOP = 3e-4
        TINY_R2_ROUNDS_TO_STOP = 2
        _r2_tiny_streak = 0

        # Share one scorer across a round and rebuild only if it falls out of sync.
        _round_scorer = [None]   # list-as-mutable-flag (closure-safe)
        _round_scorer_dirty = [False]

        # Optional: give more time to passes that have been paying off.
        _ADAPTIVE_BUDGET = os.environ.get("V2_ADAPTIVE_BUDGET", "").strip() in {
            "1", "true", "TRUE", "yes", "YES", "on", "ON"
        }

        def _env_float(name, default):
            v = os.environ.get(name)
            try:
                return float(v) if v not in (None, "") else default
            except ValueError:
                return default
        _ADAPTIVE_LO = _env_float("V2_ADAPTIVE_LO", 0.4)
        _ADAPTIVE_HI = _env_float("V2_ADAPTIVE_HI", 2.5)
        _pass_gain = {}
        _pass_time = {}
        _cur_pass = [None]
        _cur_pass_t0 = [0.0]

        def _pass_close():
            if _cur_pass[0] is not None:
                _pass_time[_cur_pass[0]] = _pass_time.get(_cur_pass[0], 0.0) + (
                    time.monotonic() - _cur_pass_t0[0]
                )
                _cur_pass[0] = None

        def _pass_cap(name, base):
            """Return this pass's time cap."""
            if not _ADAPTIVE_BUDGET:
                return base
            ys = {p: _pass_gain.get(p, 0.0) / t for p, t in _pass_time.items() if t > 1e-6}
            if name not in ys or len(ys) < 2:
                return base
            mean_y = sum(ys.values()) / len(ys)
            if mean_y <= 1e-12:
                return base
            return base * max(_ADAPTIVE_LO, min(_ADAPTIVE_HI, ys[name] / mean_y))

        def _round_scorer_get():
            """Return a scorer that matches `best_pl`."""
            if _round_scorer[0] is None or _round_scorer_dirty[0]:
                _exact_proxy(best_pl, benchmark, plc)
                _round_scorer[0] = IncrementalScorer(
                    plc, benchmark, best_pl.cpu().numpy().astype(np.float64)
                )
                _round_scorer_dirty[0] = False
            return _round_scorer[0]

        def _round_scorer_handoff(cand_true, cand):
            """Accept a pass result or mark the scorer for rebuild."""
            nonlocal best_score, best_pl, round_improved
            if _ADAPTIVE_BUDGET and _cur_pass[0] is not None and cand_true < best_score:
                _pass_gain[_cur_pass[0]] = _pass_gain.get(_cur_pass[0], 0.0) + (
                    best_score - cand_true
                )
            if cand_true < best_score - 1e-6:
                best_score, best_pl = cand_true, cand
                round_improved = True
                return True
            else:
                # The pass changed the scorer but not the incumbent placement.
                _round_scorer_dirty[0] = True
                return False

        def _ml_r2_context(pass_name: str, field: str | None, remaining_s: float, **extra):
            if _ADAPTIVE_BUDGET:
                _pass_close()
                _cur_pass[0] = pass_name
                _cur_pass_t0[0] = time.monotonic()
            if _ml_trace is not None:
                _ml_trace.set_context(
                    phase="r2",
                    r2_round=_r2 + 1,
                    pass_name=pass_name,
                    field=field,
                    elapsed_s=time.monotonic() - t0,
                    remaining_budget_s=remaining_s,
                    current_best_score=best_score,
                    round_start_score=_r2_prev_best,
                    hard_relocation_n_targets=R2_TGT,
                    **extra,
                )

        for _r2 in range(R2_MAX_ROUNDS):
            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 3.0 * t_one_score + 3.0:
                break
            _t_round_start = time.monotonic()
            _r2_prev_best = best_score
            round_improved = False
            # best_pl may have changed in the previous round.
            _round_scorer_dirty[0] = True

            try:
                _ml_r2_context("hard_relocation", "congestion", rem_r2)
                t_rel = time.monotonic()
                base_rel = float(_exact_proxy(best_pl, benchmark, plc))
                rel_scorer = _round_scorer_get()
                rel_pos, rel_acc, _ = _relocation_moves(
                    _hard_xy(best_pl), sizes, hw, hh, cw, ch, movable, n, plc,
                    benchmark, rel_scorer, base_rel,
                    deadline=t_rel + min(rem_r2 - t_one_score, _pass_cap("hard_relocation", 15.0)),
                    top_hot=R2_HOT, n_targets=R2_TGT,
                    propose_all=R2_RELOC_PROPOSE_ALL,
                    propose_top_m=R2_RELOC_PROPOSE_TOP_M,
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

            # Move hard macros out of dense cells.
            if _empty_streak["reloc_density"] < SKIP_EMPTY_AFTER:
                try:
                    _ml_r2_context("hard_relocation", "density", rem_r2)
                    t_rel_d = time.monotonic()
                    base_rel_d = float(_exact_proxy(best_pl, benchmark, plc))
                    rel_scorer_d = _round_scorer_get()
                    rel_pos_d, rel_acc_d, _ = _relocation_moves(
                        _hard_xy(best_pl), sizes, hw, hh, cw, ch, movable, n, plc,
                        benchmark, rel_scorer_d, base_rel_d,
                        deadline=t_rel_d + min(rem_r2 - t_one_score, _pass_cap("hard_relocation", 15.0)),
                        top_hot=R2_HOT, n_targets=R2_TGT,
                        use_density=True,
                        propose_all=R2_RELOC_PROPOSE_ALL,
                        propose_top_m=R2_RELOC_PROPOSE_TOP_M,
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

            # Move hard macros that are moderately bad on both fields.
            if _empty_streak["reloc_combined"] < SKIP_EMPTY_AFTER:
                try:
                    _ml_r2_context("hard_relocation", "combined", rem_r2)
                    t_rel_c = time.monotonic()
                    base_rel_c = float(_exact_proxy(best_pl, benchmark, plc))
                    rel_scorer_c = _round_scorer_get()
                    rel_pos_c, rel_acc_c, _ = _relocation_moves(
                        _hard_xy(best_pl), sizes, hw, hh, cw, ch, movable, n, plc,
                        benchmark, rel_scorer_c, base_rel_c,
                        deadline=t_rel_c + min(rem_r2 - t_one_score, _pass_cap("hard_relocation", 4.0)),
                        top_hot=R2_HOT, n_targets=R2_TGT,
                        use_combined=True,
                        propose_all=R2_RELOC_PROPOSE_ALL,
                        propose_top_m=R2_RELOC_PROPOSE_TOP_M,
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

            # --- Soft relocation passes: hot soft clusters -> cold cells, by the
            # congestion field then the density field. Softs are the bulk of both
            # terms, and the density pass finds moves the cong pass misses. Softs
            # may overlap (no legality check); accept-on-true-proxy. Reuses the
            # shared scorer (already at the post-reloc committed state). ---
            for _sfield, _use_d in (("cong", False), ("density", True)):
                if _n_soft <= 0:
                    break
                # Drop the congestion pass after it stops accepting moves.
                if _sfield == "cong" and _empty_streak["soft_reloc_cong"] >= SKIP_EMPTY_AFTER:
                    continue
                rem_sr = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                if rem_sr < 2.0 * t_one_score + 2.0:
                    break
                # After congestion stalls, let density try a wider set.
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
                    _ml_r2_context(
                        "soft_relocation",
                        _sfield,
                        rem_sr,
                        congestion_saturated=_cong_saturated,
                    )
                    t_sr = time.monotonic()
                    base_sr = float(_exact_proxy(best_pl, benchmark, plc))
                    sr_scorer = _round_scorer_get()
                    sr_pos = np.stack(
                        [best_pl[n:n + _n_soft, 0].numpy(),
                         best_pl[n:n + _n_soft, 1].numpy()], axis=1
                    ).astype(np.float64)
                    # Net centroids guide target ordering.
                    _soft_centroids = sr_scorer.soft_net_centroids()
                    sr_pos, sr_acc, _ = _soft_relocation_moves(
                        sr_pos, soft_hw, soft_hh, cw, ch, n, plc, benchmark,
                        sr_scorer, base_sr,
                        deadline=t_sr + min(rem_sr - t_one_score, _pass_cap("soft_relocation", 15.0)),
                        top_hot=_top_hot_this, n_targets=_n_tgt_this,
                        soft_movable=_soft_movable, use_density=_use_d,
                        net_centroid=_soft_centroids, wl_blend=A3_WL_BLEND,
                    )
                    # Track empty congestion passes.
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

            # Soft-soft swaps catch exchanges relocation misses.
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
                        _ml_r2_context(
                            "soft_2opt",
                            _ssfield,
                            rem_ss,
                            inner_pass=_a5_pass + 1,
                        )
                        t_ss = time.monotonic()
                        base_ss = float(_exact_proxy(best_pl, benchmark, plc))
                        ss_scorer = _round_scorer_get()
                        ss_pos = np.stack(
                            [best_pl[n:n + _n_soft, 0].numpy(),
                             best_pl[n:n + _n_soft, 1].numpy()], axis=1
                        ).astype(np.float64)
                        # Net centroids guide swap ordering.
                        _ss_centroids = ss_scorer.soft_net_centroids()
                        ss_pos, ss_acc, _ = _two_opt_soft_swap(
                            ss_pos, cw, ch, n, plc, benchmark, ss_scorer, base_ss,
                            deadline=t_ss + min(rem_ss - t_one_score, _pass_cap("soft_2opt", 6.0)),
                            top_hot=64, k_neighbors=12,
                            soft_movable=_soft_movable,
                            use_density=_ssuse_d, n_cold_teleports=4,
                            net_centroid=_ss_centroids, wl_blend=A3_WL_BLEND,
                        )
                        # Count only the first pass for skip gating.
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
                        # Stop this field when another pass does not help.
                        if not _improved_this_pass:
                            break
                    except Exception as exc:
                        _log(f"  R2 soft-2opt[{_ssfield}] failed: {type(exc).__name__}: {exc}")
                        _round_scorer_dirty[0] = True
                        break

            rem_r2 = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            if rem_r2 < 2.0 * t_one_score + 2.0:
                break

            # Swap hard and soft macros when same-type swaps miss the pair.
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
                    _ml_r2_context("hard_soft_swap", _xfield, rem_x)
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
                        deadline=t_x + min(rem_x - t_one_score, _pass_cap("hard_soft_swap", 2.5)),
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

            # Rotate one hard and two soft macros as one move.
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
                    _ml_r2_context("hard_soft_soft_cycle", _h3field, rem_h3)
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
                        deadline=t_h3 + min(rem_h3 - t_one_score, _pass_cap("hard_soft_soft_cycle", 3.0)),
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
            try:
                _ml_r2_context("hard_2opt", "congestion", rem_r2)
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
                    score_fn=_r2_score, initial_score=base_2o, k_neighbors=16,
                    max_iters=6, deadline=t_2o + min(rem_r2 - t_one_score, _pass_cap("hard_2opt", R2_2OPT_SLICE)),
                    incremental_scorer=o_scorer, macro_cong=_macro_cong_now(),
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
            # Stop when a round's gain is too small.
            _r2_delta = _r2_prev_best - best_score
            _r2_round_time = time.monotonic() - _t_round_start
            if _r2_delta < R2_DELTA_HARD_STOP:
                _log(f"  R2 round {_r2+1}: negligible Δ={_r2_delta:.5f} "
                     f"(< {R2_DELTA_HARD_STOP}) in {_r2_round_time:.1f}s; "
                     f"stopping interleave early")
                break
            elif _r2_delta < R2_DELTA_THRESHOLD:
                _r2_tiny_streak += 1
                if _r2_tiny_streak >= TINY_R2_ROUNDS_TO_STOP:
                    _log(f"  R2 round {_r2+1}: tiny Δ={_r2_delta:.5f} for "
                         f"{_r2_tiny_streak} rounds (< {R2_DELTA_THRESHOLD}) "
                         f"in {_r2_round_time:.1f}s; stopping interleave early")
                    break
            else:
                _r2_tiny_streak = 0

        # Spend leftover time on one last soft relocation pass.
        if _n_soft > 0:
            for _post_field, _post_ud in (("cong", False), ("density", True)):
                rem_post = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
                # Keep enough time to verify the result.
                if rem_post < t_one_score * 1.5:
                    break
                try:
                    if _ml_trace is not None:
                        _ml_trace.set_context(
                            phase="post_r2",
                            pass_name="soft_relocation",
                            field=_post_field,
                            elapsed_s=time.monotonic() - t0,
                            remaining_budget_s=rem_post,
                            current_best_score=best_score,
                        )
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
                            # Restore plc to the incumbent placement.
                            float(_exact_proxy(best_pl, benchmark, plc))
                except Exception as _post_exc:
                    _log(f"  Post-R2 soft-reloc[{_post_field}] failed: {_post_exc}")
                    try:
                        float(_exact_proxy(best_pl, benchmark, plc))
                    except Exception:
                        pass

        # LSMC exploration (GPU-ops.md Stage 2a) - opt-in via V2_GPU_EXPLORE.
        # Final quality phase: kicks from the fully-refined optimum and accepts
        # on exact post-descent score. Earlier placements (pre-R2, pre-post-soft)
        # were tested and disproven - any refinement AFTER the accept gate makes
        # the gate compare the wrong objective.
        if _explore_enabled(_GPU_BACKEND):
            _exp_time = float(os.environ.get("V2_GPU_EXPLORE_TIME_S", "30.0"))
            _rem_exp = (effective_budget_s + BUDGET_OVERRUN_S) - (time.monotonic() - t0)
            _exp_slice = min(_exp_time, _rem_exp - t_one_score * 1.5)
            if _exp_slice > t_one_score:
                try:
                    best_pl, best_score, _exp_iters, _exp_acc = _lsmc_explore(
                        best_pl, best_score, benchmark, plc, _exact_proxy,
                        sizes, hw, hh, cw, ch, movable, n,
                        time_budget_s=_exp_slice, log=_log,
                        soft_hw=soft_hw, soft_hh=soft_hh,
                        soft_movable=_soft_movable, n_soft=_n_soft,
                    )
                    _log(f"  LSMC explore: {_exp_acc} accepts / {_exp_iters} iters, "
                         f"best={best_score:.4f}")
                except Exception as exc:
                    _log(f"  LSMC explore failed: {type(exc).__name__}: {exc}")

        _log(f"  Best proxy={best_score:.4f}  total={time.monotonic()-t0:.1f}s")
        _ml_finish("completed", best_score)
        self._total_place_time_s += time.monotonic() - t0
        self._benchmarks_done += 1
        return best_pl
