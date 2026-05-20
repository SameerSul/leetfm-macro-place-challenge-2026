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

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macro_place._plc import PlacementCost  # noqa: E402

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
                                random_center_init: bool = False) -> dict:
    """Single CPU-only global-placement stage. Tuned to be fast: 200 iters,
    64x64 density bins. No legalization or detailed placement (we do those
    in our own pipeline).

    random_center_init=False (default): warm-start from TILOS initial.plc
    positions (passed in via the .pl file). Hard macros only move incrementally,
    keeping soft-macro neighborhoods intact for the proxy evaluator.

    random_center_init=True: cold-start from canvas center; produces a
    fundamentally different placement. Useful for exploring entirely new
    basins, but soft macros end up mismatched (high congestion penalty)."""
    return {
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
        "target_density": 0.75,
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
        "macro_place_flag": 0,
        "result_dir": result_dir,
    }


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

    # Phase 2: forward convert
    convert(str(benchmark_dir), str(work_dir), design=design,
            soft_macros_movable=soft_macros_movable)

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
    )
    cfg_path = work_dir / f"{design}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # Run DREAMPlace
    env = os.environ.copy()
    pythonpath = f"{DREAMPLACE_INSTALL}:{DREAMPLACE_INSTALL / 'dreamplace'}"
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + ":" + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

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

    pos = read_dreamplace_positions(plc, str(work_dir), design)
    print(f"  [dreamplace] {design}: {pos.shape[0]} hard macros placed "
          f"(global-place {dp_time:.1f}s)")
    return pos


# ---------------------------------------------------------------------------
# Async launch: fire DREAMPlace as non-blocking subprocess
# ---------------------------------------------------------------------------

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
                 timeout_s: float, log_handle):
        self.popen = popen
        self.work_dir = work_dir
        self.design = design
        self.plc = plc
        self.start_time = start_time
        self.timeout_s = timeout_s
        self._log_handle = log_handle
        # Internally stores the (hard_pos, soft_pos) tuple from
        # read_dreamplace_positions_full once the subprocess completes.
        # wait_for_result extracts the hard component for backward-compatible callers.
        self._result: "Optional[tuple[np.ndarray, np.ndarray]]" = None
        self._failed = False
        self._kill_called = False

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
            return self._result
        except Exception:
            self._failed = True
            return None

    def kill(self) -> None:
        """Abort the subprocess if still running. Safe to call multiple times."""
        if self._kill_called:
            return
        self._kill_called = True
        if self.popen.poll() is None:
            try:
                self.popen.kill()
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

    # Forward convert (~1s)
    convert(str(benchmark_dir_p), str(work_dir), design=design,
            soft_macros_movable=soft_macros_movable)

    result_dir = work_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg = _default_dreamplace_config(
        aux_input=str(work_dir / f"{design}.aux"),
        result_dir=str(result_dir),
        random_seed=random_seed,
        iterations=iterations,
        num_threads=num_threads,
        random_center_init=random_center_init,
    )
    cfg_path = work_dir / f"{design}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    env = os.environ.copy()
    pythonpath = f"{DREAMPLACE_INSTALL}:{DREAMPLACE_INSTALL / 'dreamplace'}"
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + ":" + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    # Lazy-load plc for back-conversion (done before launch so we don't pay
    # this cost on the critical path when the user calls wait_for_result).
    if plc is None:
        plc = PlacementCost(str(benchmark_dir_p / "netlist.pb.txt"))
        init_plc = benchmark_dir_p / "initial.plc"
        if init_plc.exists():
            plc.restore_placement(str(init_plc), ifInital=True, ifReadComment=True)

    log_handle = (work_dir / "dreamplace.log").open("w")
    popen = subprocess.Popen(
        [str(VENV_PYTHON), str(DREAMPLACE_PLACER), str(cfg_path)],
        cwd=str(DREAMPLACE_INSTALL),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    return AsyncDreamplaceHandle(
        popen=popen, work_dir=work_dir, design=design, plc=plc,
        start_time=time.time(), timeout_s=timeout_s, log_handle=log_handle,
    )


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
