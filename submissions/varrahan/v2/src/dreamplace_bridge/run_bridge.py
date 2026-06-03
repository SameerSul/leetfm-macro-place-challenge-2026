"""End-to-end DREAMPlace bridge: TILOS pb.txt → Bookshelf → DREAMPlace global → back.

Wraps the three steps so the placer can call a single function to obtain
DREAMPlace's analytic placement of the hard macros.

Subprocess strategy: DREAMPlace is launched as a separate Python process
because (a) its `import Params` style pollutes our `sys.modules`, (b) its
torch usage can fight with ours over global state, and (c) subprocesses
take a hard timeout cleanly. Cost: ~5-10s of Python startup per call.

For each benchmark we materialise a scratch dir under `/tmp/dreamplace_v1/`
holding the 5 Bookshelf files + JSON config + DREAMPlace's results dir.
The dir is reused across calls (positions are deterministic given a fixed
random_seed), so a second call is fast — actually no, we still re-run
DREAMPlace every call. Caching to a `.npy` would be a future optimization.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve()
REPO_ROOT = next(
    p for p in HERE.parents
    if (p / "pyproject.toml").exists() and (p / "macro_place").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macro_place._plc import PlacementCost  # noqa: E402


# ---------------------------------------------------------------------------
# Disk cache — DREAMPlace output is deterministic given fingerprint inputs.
# ---------------------------------------------------------------------------
# The (hard_pos, soft_pos) result depends only on: input netlist + initial
# placement + config (iterations, seed, num_threads, soft_macros_movable,
# random_center_init). Threads affect determinism only via NUM_THREADS env
# (deterministic_flag=1 + fixed seed gives bit-identical output on same
# threads). We fingerprint the input files by (size, mtime_ns) so dev edits
# invalidate the cache automatically, and bake config into the key.
CACHE_VERSION = "v1"


def _file_fingerprint(p: Path) -> str:
    try:
        st = p.stat()
    except FileNotFoundError:
        return "missing"
    return f"{st.st_size}:{st.st_mtime_ns}"


def _cache_key(benchmark_dir: Path, iterations: int, random_seed: int,
               num_threads: int, soft_macros_movable: bool,
               random_center_init: bool, routability_opt: bool = False,
               routopt_sig: str = "") -> str:
    netlist_fp = _file_fingerprint(benchmark_dir / "netlist.pb.txt")
    init_fp = _file_fingerprint(benchmark_dir / "initial.plc")
    raw = (
        f"{CACHE_VERSION}|{benchmark_dir.name}|netlist={netlist_fp}|"
        f"init={init_fp}|iter={iterations}|seed={random_seed}|"
        f"threads={num_threads}|soft={int(soft_macros_movable)}|"
        f"rci={int(random_center_init)}"
    )
    # Append only when set, so existing (non-routopt) cache keys are unchanged.
    # routopt_sig encodes the calibration knobs so swept configs don't collide.
    if routability_opt:
        raw += f"|ro=1|{routopt_sig}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_paths(work_dir: Path, key: str) -> "tuple[Path, Path]":
    """Returns (npz_path, meta_path) for a given cache key."""
    return (work_dir / f"cached_{key}.npz", work_dir / f"cached_{key}.json")


def _try_load_cache(work_dir: Path, key: str) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    npz_path, meta_path = _cache_paths(work_dir, key)
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        with np.load(npz_path) as data:
            hard = data["hard_pos"]
            soft = data["soft_pos"]
        return (np.ascontiguousarray(hard), np.ascontiguousarray(soft))
    except Exception:
        return None


def _write_cache(work_dir: Path, key: str, hard: np.ndarray, soft: np.ndarray) -> None:
    npz_path, meta_path = _cache_paths(work_dir, key)
    try:
        np.savez_compressed(npz_path, hard_pos=hard, soft_pos=soft)
        meta_path.write_text(json.dumps({"key": key, "ts": time.time()}))
    except Exception:
        # Cache write is best-effort; never fail the placer because of it.
        pass

# Sibling-module imports — work whether this package is loaded as
# `dreamplace_bridge.run_bridge` (placer-side, after sys.path injection)
# or as `submissions.varrahan.v1.dreamplace_bridge.run_bridge` (CLI from repo root).
try:
    from .pb_to_bookshelf import convert
    from .bookshelf_to_pb import read_dreamplace_positions, read_dreamplace_positions_full
except ImportError:
    from pb_to_bookshelf import convert  # type: ignore
    from bookshelf_to_pb import read_dreamplace_positions, read_dreamplace_positions_full  # type: ignore


# Where DREAMPlace lives (set up by Phase 1 build).
DREAMPLACE_INSTALL = (
    REPO_ROOT / "submissions" / "varrahan" / "dreamplace_build" / "install"
)
DREAMPLACE_PLACER = DREAMPLACE_INSTALL / "dreamplace" / "Placer.py"

# Repo's project venv (uv-managed).
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _default_dreamplace_config(aux_input: str, result_dir: str,
                                random_seed: int = 1000,
                                iterations: int = 200,
                                num_threads: int = 4,
                                random_center_init: bool = False,
                                target_density: float = 0.75,
                                routability_opt: bool = False,
                                route_num_bins: int = 64,
                                route_num_bins_x: "int | None" = None,
                                route_num_bins_y: "int | None" = None,
                                unit_h_cap: "float | None" = None,
                                unit_v_cap: "float | None" = None,
                                max_route_opt_adjust_rate: float = 2.0,
                                max_num_area_adjust: int = 3) -> dict:
    """Single CPU-only global-placement stage. Tuned to be fast: 200 iters,
    64x64 density bins. No legalization or detailed placement (we do those
    in our own pipeline).

    random_center_init=False (default): warm-start from TILOS initial.plc
    positions (passed in via the .pl file). Hard macros only move incrementally,
    keeping soft-macro neighborhoods intact for the proxy evaluator.

    random_center_init=True: cold-start from canvas center; produces a
    fundamentally different placement. Useful for exploring entirely new
    basins, but soft macros end up mismatched (high congestion penalty)."""
    cfg = {
        "aux_input": aux_input,
        "gpu": 0,
        "num_bins_x": 64,
        "num_bins_y": 64,
        "global_place_stages": [{
            "num_bins_x": 64, "num_bins_y": 64,
            "iteration": iterations, "learning_rate": 0.01,
            "wirelength": "weighted_average", "optimizer": "nesterov",
            "Llambda_density_weight_iteration": 1, "Lsub_iteration": 1,
        }],
        "target_density": target_density,
        "density_weight": 5e-3,
        "gamma": 4.0,
        "random_seed": random_seed,
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 0,
        "gp_noise_ratio": 0.025,
        "global_place_flag": 1,
        "legalize_flag": 0,
        "detailed_place_flag": 0,
        "detailed_place_engine": "",
        "detailed_place_command": "",
        "stop_overflow": 0.10,
        "dtype": "float32",
        "plot_flag": 0,
        "random_center_init_flag": 1 if random_center_init else 0,
        "gift_init_flag": 0,
        "sort_nets_by_degree": 0,
        "num_threads": num_threads,
        "deterministic_flag": 1,
        "macro_place_flag": 1,
        "use_bb": 1,
        "result_dir": result_dir,
    }
    if routability_opt:
        # DP1 (2026-05-27): congestion-aware global placement. DREAMPlace
        # computes a RUDY/RISA routing-congestion map mid-placement (once
        # overflow drops below node_area_adjust_overflow) and inflates node
        # areas in congested regions, so the density penalty spreads cells out
        # of routing hotspots. This bakes congestion INTO the global objective —
        # the DP_DIAG diagnostic showed our DP candidates lose to best purely on
        # congestion, and post-hoc repair can't fix it (trades away DP's wl/den
        # edge). route_num_bins is coarser than DREAMPlace's 512 default to
        # roughly match the ICCAD04 routing grids (~35-55 cols). Unit capacities
        # are left at DREAMPlace defaults for the first cut — to be calibrated
        # against the ICCAD04 routes-per-micron if the proxy-cong signal is weak.
        cfg.update({
            "routability_opt_flag": 1,
            "route_num_bins_x": route_num_bins_x or route_num_bins,
            "route_num_bins_y": route_num_bins_y or route_num_bins,
            "adjust_rudy_area_flag": 1,
            "adjust_pin_area_flag": 1,
            "adjust_nctugr_area_flag": 0,
            "node_area_adjust_overflow": 0.15,
            "max_num_area_adjust": max_num_area_adjust,
            "max_route_opt_adjust_rate": max_route_opt_adjust_rate,
            "route_opt_adjust_exponent": 2.0,
        })
        # Per-tech routing capacity (tracks per unit distance, DREAMPlace coords).
        # RUDY utilization = demand / (bin_area * unit_capacity); a larger
        # capacity lowers perceived congestion → gentler area inflation.
        if unit_h_cap is not None:
            cfg["unit_horizontal_capacity"] = unit_h_cap
        if unit_v_cap is not None:
            cfg["unit_vertical_capacity"] = unit_v_cap
    return cfg


def is_available() -> bool:
    """Quick check: is DREAMPlace built and importable?"""
    return DREAMPLACE_PLACER.exists() and VENV_PYTHON.exists()


def run_dreamplace(
    benchmark_dir: str,
    plc: Optional[PlacementCost] = None,
    scratch_root: str = "/tmp/dreamplace_v1",
    timeout_s: float = 120.0,
    iterations: int = 200,
    random_seed: int = 1000,
    num_threads: int = 4,
    soft_macros_movable: bool = False,
    random_center_init: bool = False,
    keep_log: bool = False,
    target_density: float = 0.75,
    routability_opt: bool = False,
    route_num_bins: int = 64,
) -> np.ndarray:
    """Run the full DREAMPlace pipeline on a benchmark.

    Returns hard-macro positions [num_hard_macros, 2] in TILOS microns,
    indexed identically to `plc.hard_macro_indices`.

    Parameters
    ----------
    benchmark_dir : str
        Path to TILOS benchmark dir (containing netlist.pb.txt).
    plc : PlacementCost, optional
        If provided, used for the back-conversion. If None, a fresh one is
        loaded from the benchmark dir.
    scratch_root : str
        Parent dir for per-benchmark scratch space.
    timeout_s : float
        Hard timeout for the DREAMPlace subprocess. If exceeded, a
        TimeoutExpired exception propagates.
    iterations : int
        Global-placement iteration count (DREAMPlace stops earlier if
        overflow drops below stop_overflow=0.10).
    random_seed : int
        DREAMPlace's RNG seed. Same seed → deterministic output.
    num_threads : int
        CPU thread count for DREAMPlace.
    soft_macros_movable : bool
        Forward-conversion option. False → soft macros are terminals (stay
        at initial positions). True → DREAMPlace re-places them too.
    keep_log : bool
        If True, the DREAMPlace stdout/stderr log is kept at
        {scratch_root}/{design}/dreamplace.log; otherwise discarded.
    """
    if not is_available():
        raise RuntimeError(
            f"DREAMPlace not installed at {DREAMPLACE_INSTALL}. "
            f"See v1/dreamplace_bridge/ docs to rebuild."
        )

    benchmark_dir = Path(benchmark_dir).resolve()
    design = benchmark_dir.name
    work_dir = Path(scratch_root).resolve() / design
    work_dir.mkdir(parents=True, exist_ok=True)

    # Disk-cache fast path (same fingerprint scheme as the async launcher).
    cache_key = _cache_key(
        benchmark_dir, iterations, random_seed, num_threads,
        soft_macros_movable, random_center_init, routability_opt,
    )
    cached = _try_load_cache(work_dir, cache_key)
    if cached is not None:
        hard_pos, _soft_pos = cached
        print(f"  [dreamplace] {design}: {hard_pos.shape[0]} hard macros "
              f"(cache hit, skipped subprocess)")
        return hard_pos

    # Phase 2: forward convert (re-use caller's plc when available)
    convert(str(benchmark_dir), str(work_dir), design=design,
            soft_macros_movable=soft_macros_movable, plc=plc)

    # Write JSON config
    result_dir = work_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg = _default_dreamplace_config(
        aux_input=str(work_dir / f"{design}.aux"),
        result_dir=str(result_dir),
        random_seed=random_seed,
        iterations=iterations,
        num_threads=num_threads,
        random_center_init=random_center_init,
        target_density=target_density,
        routability_opt=routability_opt,
        route_num_bins=route_num_bins,
    )
    cfg_path = work_dir / f"{design}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # Run DREAMPlace
    env = os.environ.copy()
    pythonpath = f"{DREAMPLACE_INSTALL}:{DREAMPLACE_INSTALL / 'dreamplace'}"
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + ":" + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    # Cap CPU footprint to match num_threads (see launch_dreamplace_async for
    # rationale).
    nt = str(max(1, int(num_threads)))
    env["OMP_NUM_THREADS"] = nt
    env["MKL_NUM_THREADS"] = nt
    env["OPENBLAS_NUM_THREADS"] = nt
    env["NUMEXPR_NUM_THREADS"] = nt

    log_target = (work_dir / "dreamplace.log").open("w") if keep_log else subprocess.DEVNULL
    t0 = time.time()
    try:
        subprocess.run(
            [str(VENV_PYTHON), str(DREAMPLACE_PLACER), str(cfg_path)],
            cwd=str(DREAMPLACE_INSTALL),
            env=env,
            stdout=log_target,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout_s,
        )
    finally:
        if hasattr(log_target, "close"):
            log_target.close()
    dp_time = time.time() - t0

    # Phase 3a: back-convert
    if plc is None:
        plc = PlacementCost(str(benchmark_dir / "netlist.pb.txt"))
        init_plc = benchmark_dir / "initial.plc"
        if init_plc.exists():
            plc.restore_placement(str(init_plc), ifInital=True, ifReadComment=True)

    # Use the full reader so we can persist both hard + soft to cache. The
    # callers expect ONLY hard back, so we return that and stash soft on disk.
    try:
        hard_pos, soft_pos = read_dreamplace_positions_full(plc, str(work_dir), design)
        _write_cache(work_dir, cache_key, hard_pos, soft_pos)
        pos = hard_pos
    except Exception:
        pos = read_dreamplace_positions(plc, str(work_dir), design)
    print(f"  [dreamplace] {design}: {pos.shape[0]} hard macros placed "
          f"(global-place {dp_time:.1f}s)")
    return pos


# ---------------------------------------------------------------------------
# Async launch: fire DREAMPlace as non-blocking subprocess
# ---------------------------------------------------------------------------

class _CachedDreamplaceHandle:
    """Drop-in stand-in for AsyncDreamplaceHandle when a disk-cache hit
    means the subprocess never has to run. Implements the same surface the
    placer uses: poll/is_done/time_elapsed/wait_for_result*/kill."""

    def __init__(self, result: "tuple[np.ndarray, np.ndarray]", start_time: float):
        self._result = result
        self.start_time = start_time

    def poll(self) -> Optional[int]:
        return 0

    def is_done(self) -> bool:
        return True

    def time_elapsed(self) -> float:
        return time.time() - self.start_time

    def wait_for_result(self, max_wait_s: float = 0.0) -> Optional[np.ndarray]:
        return self._result[0]

    def wait_for_result_full(
        self, max_wait_s: float = 0.0
    ) -> "Optional[tuple[np.ndarray, np.ndarray]]":
        return self._result

    def kill(self) -> None:
        return


class AsyncDreamplaceHandle:
    """Handle to a DREAMPlace subprocess launched via `launch_dreamplace_async`.

    Use `is_done()`/`poll()` to check completion without blocking, then
    `wait_for_result(max_wait_s)` to retrieve positions. `kill()` aborts a
    still-running subprocess (call at place() exit if DREAMPlace hasn't
    finished and we don't want to wait).

    The async pattern saves the v13 failure mode: v13 ran DREAMPlace
    synchronously BEFORE Phase 1, paying 30-90s of subprocess time that
    displaced cong-grad/noise restarts on most benchmarks. Async overlaps
    that subprocess time with our own scoring (which is C++-side and can
    release the GIL), so DREAMPlace becomes "free" budget on benchmarks
    where scoring is also slow (ibm08/11/15/etc).
    """

    def __init__(self, popen: "subprocess.Popen", work_dir: Path,
                 design: str, plc: PlacementCost, start_time: float,
                 timeout_s: float, log_handle,
                 cache_key: Optional[str] = None):
        self.popen = popen
        self.work_dir = work_dir
        self.design = design
        self.plc = plc
        self.start_time = start_time
        self.timeout_s = timeout_s
        self._log_handle = log_handle
        self._cache_key = cache_key
        # Internally stores the (hard_pos, soft_pos) tuple from
        # read_dreamplace_positions_full once the subprocess completes.
        # wait_for_result extracts the hard component for backward-compatible callers.
        self._result: "Optional[tuple[np.ndarray, np.ndarray]]" = None
        self._failed = False
        self._kill_called = False
        self._watchdog_thread = None

    def _start_watchdog(self) -> None:
        """Spawn a daemon thread that enforces timeout_s by killing the DP
        subprocess if it runs past the deadline. Necessary because the placer
        may be blocked in scoring (which doesn't release the GIL fast enough
        for wait_for_result to fire) while DP keeps eating CPU. Without this,
        a hung or slow DP can saturate cores and slow scoring 100x (verified
        2026-05-20 on ibm06 in --all)."""
        import threading

        def _watch():
            try:
                self.popen.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                # Subprocess exceeded its budget; tear it (and its process group)
                # down so it stops competing for CPU.
                self._kill_called = True
                try:
                    import os, signal
                    os.killpg(os.getpgid(self.popen.pid), signal.SIGKILL)
                except Exception:
                    try:
                        self.popen.kill()
                    except Exception:
                        pass
                try:
                    self.popen.wait(timeout=2.0)
                except Exception:
                    pass
            except Exception:
                pass

        t = threading.Thread(target=_watch, daemon=True, name=f"dp-watchdog-{self.design}")
        t.start()
        self._watchdog_thread = t

    def poll(self) -> Optional[int]:
        """Return exit code if process done, None if still running."""
        if self._kill_called:
            return self.popen.returncode
        return self.popen.poll()

    def is_done(self) -> bool:
        return self.poll() is not None

    def time_elapsed(self) -> float:
        return time.time() - self.start_time

    def wait_for_result(self, max_wait_s: float = 0.0) -> Optional[np.ndarray]:
        """Wait up to max_wait_s seconds for completion. Returns hard-macro
        positions [num_hard_macros, 2] on success, None on timeout/failure.

        max_wait_s=0 means non-blocking check (returns None if not yet done).

        Note: returns ONLY hard positions for compatibility. Use
        `wait_for_result_full()` if you also need soft positions (which
        requires the bridge to have been launched with
        `soft_macros_movable=True` for the soft positions to be meaningful).
        """
        full = self.wait_for_result_full(max_wait_s=max_wait_s)
        return None if full is None else full[0]

    def wait_for_result_full(
        self, max_wait_s: float = 0.0
    ) -> "Optional[tuple[np.ndarray, np.ndarray]]":
        """Like `wait_for_result` but returns BOTH hard and soft positions.

        Returns (hard_pos [num_hard, 2], soft_pos [num_soft, 2]) on success,
        None on timeout/failure. The soft positions are meaningful only if
        the bridge was launched with `soft_macros_movable=True` — otherwise
        DREAMPlace treated softs as fixed and the back-converter falls back
        to `node.get_pos()` which equals their initial.plc positions.
        """
        if self._result is not None:
            return self._result
        if self._failed:
            return None
        try:
            self.popen.wait(timeout=max_wait_s if max_wait_s > 0 else 0.001)
        except subprocess.TimeoutExpired:
            return None
        finally:
            if hasattr(self._log_handle, "close"):
                try:
                    self._log_handle.close()
                except Exception:
                    pass

        if self.popen.returncode != 0:
            self._failed = True
            return None

        try:
            self._result = read_dreamplace_positions_full(
                self.plc, str(self.work_dir), self.design
            )
            if self._cache_key is not None and self._result is not None:
                _write_cache(self.work_dir, self._cache_key,
                             self._result[0], self._result[1])
            return self._result
        except Exception:
            self._failed = True
            return None

    def kill(self) -> None:
        """Abort the subprocess if still running. Safe to call multiple times.
        Sends SIGKILL to the entire process group (matched by start_new_session
        at launch) so any child threads / processes DP spawned also die."""
        if self._kill_called:
            return
        self._kill_called = True
        if self.popen.poll() is None:
            try:
                import os, signal
                os.killpg(os.getpgid(self.popen.pid), signal.SIGKILL)
            except Exception:
                try:
                    self.popen.kill()
                except Exception:
                    pass
            try:
                self.popen.wait(timeout=2.0)
            except Exception:
                pass
        if hasattr(self._log_handle, "close"):
            try:
                self._log_handle.close()
            except Exception:
                pass


def launch_dreamplace_async(
    benchmark_dir: str,
    plc: Optional[PlacementCost] = None,
    scratch_root: str = "/tmp/dreamplace_v1",
    timeout_s: float = 120.0,
    iterations: int = 200,
    random_seed: int = 1000,
    num_threads: int = 2,
    soft_macros_movable: bool = False,
    random_center_init: bool = False,
    target_density: float = 0.75,
    routability_opt: bool = False,
    route_num_bins: int = 64,
    route_num_bins_x: "int | None" = None,
    route_num_bins_y: "int | None" = None,
    unit_h_cap: "float | None" = None,
    unit_v_cap: "float | None" = None,
    max_route_opt_adjust_rate: float = 2.0,
    max_num_area_adjust: int = 3,
) -> AsyncDreamplaceHandle:
    """Launch DREAMPlace as a non-blocking subprocess. Returns immediately
    with a handle for polling.

    Setup steps (forward conversion, JSON config) run synchronously here
    (~1s total). The DREAMPlace subprocess itself runs asynchronously.

    Use the returned handle:
        h = launch_dreamplace_async(benchmark_dir, plc=plc)
        # ... do other work ...
        pos = h.wait_for_result(max_wait_s=5.0)
        if pos is None:
            h.kill()  # didn't finish in time
        else:
            # use pos
    """
    if not is_available():
        raise RuntimeError(
            f"DREAMPlace not installed at {DREAMPLACE_INSTALL}. "
            f"See v1/dreamplace_bridge/ docs to rebuild."
        )

    benchmark_dir_p = Path(benchmark_dir).resolve()
    design = benchmark_dir_p.name
    work_dir = Path(scratch_root).resolve() / design
    work_dir.mkdir(parents=True, exist_ok=True)

    # Disk-cache check: DREAMPlace output is deterministic given fingerprint
    # (input files + iterations + seed + threads + softs + rci). If we have
    # a matching .npz, skip the whole subprocess pipeline and return a stub
    # handle that produces the cached arrays immediately.
    routopt_sig = (
        f"rb={route_num_bins_x or route_num_bins}x{route_num_bins_y or route_num_bins}"
        f"|uh={unit_h_cap}|uv={unit_v_cap}|rate={max_route_opt_adjust_rate}"
        f"|adj={max_num_area_adjust}|td={target_density:.3f}"
    ) if routability_opt else ""
    cache_key = _cache_key(
        benchmark_dir_p, iterations, random_seed, num_threads,
        soft_macros_movable, random_center_init, routability_opt, routopt_sig,
    )
    cached = _try_load_cache(work_dir, cache_key)
    if cached is not None:
        return _CachedDreamplaceHandle(cached, start_time=time.time())

    # Forward convert (~1s). Pass the caller's plc through if provided so the
    # converter doesn't re-parse netlist.pb.txt (saves ~0.5-2s).
    convert(str(benchmark_dir_p), str(work_dir), design=design,
            soft_macros_movable=soft_macros_movable, plc=plc)

    result_dir = work_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg = _default_dreamplace_config(
        aux_input=str(work_dir / f"{design}.aux"),
        result_dir=str(result_dir),
        random_seed=random_seed,
        iterations=iterations,
        num_threads=num_threads,
        random_center_init=random_center_init,
        target_density=target_density,
        routability_opt=routability_opt,
        route_num_bins=route_num_bins,
        route_num_bins_x=route_num_bins_x,
        route_num_bins_y=route_num_bins_y,
        unit_h_cap=unit_h_cap,
        unit_v_cap=unit_v_cap,
        max_route_opt_adjust_rate=max_route_opt_adjust_rate,
        max_num_area_adjust=max_num_area_adjust,
    )
    cfg_path = work_dir / f"{design}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    env = os.environ.copy()
    pythonpath = f"{DREAMPLACE_INSTALL}:{DREAMPLACE_INSTALL / 'dreamplace'}"
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + ":" + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    # Cap DP's CPU footprint to match its declared num_threads. Without this,
    # DREAMPlace's internal OMP/MKL pools default to all available cores and
    # oversubscribe alongside the parent's scoring (torch + PlacementCost C++),
    # causing 100x scoring slowdowns on contended runs (ibm06 in --all 2026-05-20
    # saw scoring take 1599s vs the typical 14s, triggering the SLOW_SCORE_THRESHOLD
    # safety bail and regressing -0.051 from the v8 PROGRESS.md result).
    nt = str(max(1, int(num_threads)))
    env["OMP_NUM_THREADS"] = nt
    env["MKL_NUM_THREADS"] = nt
    env["OPENBLAS_NUM_THREADS"] = nt
    env["NUMEXPR_NUM_THREADS"] = nt

    # Lazy-load plc for back-conversion (done before launch so we don't pay
    # this cost on the critical path when the user calls wait_for_result).
    if plc is None:
        plc = PlacementCost(str(benchmark_dir_p / "netlist.pb.txt"))
        init_plc = benchmark_dir_p / "initial.plc"
        if init_plc.exists():
            plc.restore_placement(str(init_plc), ifInital=True, ifReadComment=True)

    log_handle = (work_dir / "dreamplace.log").open("w")
    # start_new_session=True puts DP in its own process group so a kill can
    # tear down any child threads DP might spawn (defensive — DREAMPlace
    # doesn't typically spawn children, but cleanup safety matters when we
    # may kill it from a watchdog thread).
    popen = subprocess.Popen(
        [str(VENV_PYTHON), str(DREAMPLACE_PLACER), str(cfg_path)],
        cwd=str(DREAMPLACE_INSTALL),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    handle = AsyncDreamplaceHandle(
        popen=popen, work_dir=work_dir, design=design, plc=plc,
        start_time=time.time(), timeout_s=timeout_s, log_handle=log_handle,
        cache_key=cache_key,
    )
    handle._start_watchdog()
    return handle


def _main():
    """Quick CLI smoke test."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--scratch", default="/tmp/dreamplace_v1")
    ap.add_argument("--keep-log", action="store_true")
    args = ap.parse_args()

    pos = run_dreamplace(args.benchmark, scratch_root=args.scratch,
                         keep_log=args.keep_log)
    print(f"Got positions: shape={pos.shape}, dtype={pos.dtype}")
    print(f"  x range: [{pos[:, 0].min():.2f}, {pos[:, 0].max():.2f}]")
    print(f"  y range: [{pos[:, 1].min():.2f}, {pos[:, 1].max():.2f}]")


if __name__ == "__main__":
    _main()
