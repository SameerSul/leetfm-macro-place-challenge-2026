"""Run DREAMPlace through a TILOS-to-Bookshelf bridge."""

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


# Disk cache keyed by input files and DREAMPlace settings.
CACHE_VERSION = "v1"


def _file_fingerprint(p: Path) -> str:
    try:
        st = p.stat()
    except FileNotFoundError:
        return "missing"
    return f"{st.st_size}:{st.st_mtime_ns}"


def _cache_key(benchmark_dir: Path, iterations: int, random_seed: int,
               num_threads: int, soft_macros_movable: bool,
               random_center_init: bool) -> str:
    netlist_fp = _file_fingerprint(benchmark_dir / "netlist.pb.txt")
    init_fp = _file_fingerprint(benchmark_dir / "initial.plc")
    raw = (
        f"{CACHE_VERSION}|{benchmark_dir.name}|netlist={netlist_fp}|"
        f"init={init_fp}|iter={iterations}|seed={random_seed}|"
        f"threads={num_threads}|soft={int(soft_macros_movable)}|"
        f"rci={int(random_center_init)}"
    )
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

# Support package and script-style imports.
try:
    from .pb_to_bookshelf import convert
    from .bookshelf_to_pb import read_dreamplace_positions, read_dreamplace_positions_full
except ImportError:
    from pb_to_bookshelf import convert  # type: ignore
    from bookshelf_to_pb import read_dreamplace_positions, read_dreamplace_positions_full  # type: ignore


# Where DREAMPlace is installed.
DREAMPLACE_INSTALL = (
    REPO_ROOT / "system" / "dreamplace_build" / "install"
)
DREAMPLACE_PLACER = DREAMPLACE_INSTALL / "dreamplace" / "Placer.py"

# DREAMPlace extensions must run with the Python used to build them.
_DP_BUILD_PYTHON = (
    REPO_ROOT / "system" / "dreamplace_build" / "dpenv" / "bin" / "python"
)
_REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
VENV_PYTHON = _DP_BUILD_PYTHON if _DP_BUILD_PYTHON.exists() else _REPO_VENV_PYTHON


def _default_dreamplace_config(aux_input: str, result_dir: str,
                                random_seed: int = 1000,
                                iterations: int = 200,
                                num_threads: int = 4,
                                random_center_init: bool = False,
                                target_density: float = 0.75) -> dict:
    """Build the DREAMPlace config used for fast global placement."""
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
) -> np.ndarray:
    """Run DREAMPlace and return hard-macro center positions."""
    if not is_available():
        raise RuntimeError(
            f"DREAMPlace not installed at {DREAMPLACE_INSTALL}. "
            f"See system/v2/README.md to rebuild."
        )

    benchmark_dir = Path(benchmark_dir).resolve()
    design = benchmark_dir.name
    work_dir = Path(scratch_root).resolve() / design
    work_dir.mkdir(parents=True, exist_ok=True)

    # Disk-cache fast path.
    cache_key = _cache_key(
        benchmark_dir, iterations, random_seed, num_threads,
        soft_macros_movable, random_center_init,
    )
    cached = _try_load_cache(work_dir, cache_key)
    if cached is not None:
        hard_pos, _soft_pos = cached
        print(f"  [dreamplace] {design}: {hard_pos.shape[0]} hard macros "
              f"(cache hit, skipped subprocess)")
        return hard_pos

    # Convert to Bookshelf.
    convert(str(benchmark_dir), str(work_dir), design=design,
            soft_macros_movable=soft_macros_movable, plc=plc)

    # Write DREAMPlace config.
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
    )
    cfg_path = work_dir / f"{design}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # Run DREAMPlace.
    env = os.environ.copy()
    pythonpath = f"{DREAMPLACE_INSTALL}:{DREAMPLACE_INSTALL / 'dreamplace'}"
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + ":" + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    # Cap DREAMPlace CPU pools to the requested thread count.
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

    # Convert results back to TILOS coordinates.
    if plc is None:
        plc = PlacementCost(str(benchmark_dir / "netlist.pb.txt"))
        init_plc = benchmark_dir / "initial.plc"
        if init_plc.exists():
            plc.restore_placement(str(init_plc), ifInital=True, ifReadComment=True)

    # Cache hard and soft positions; return only hard positions.
    try:
        hard_pos, soft_pos = read_dreamplace_positions_full(plc, str(work_dir), design)
        _write_cache(work_dir, cache_key, hard_pos, soft_pos)
        pos = hard_pos
    except Exception:
        pos = read_dreamplace_positions(plc, str(work_dir), design)
    print(f"  [dreamplace] {design}: {pos.shape[0]} hard macros placed "
          f"(global-place {dp_time:.1f}s)")
    return pos


class _CachedDreamplaceHandle:
    """Async-style handle for a cached DREAMPlace result."""

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
    """Non-blocking DREAMPlace subprocess handle."""

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
        # wait_for_result returns only hard positions for compatibility.
        self._result: "Optional[tuple[np.ndarray, np.ndarray]]" = None
        self._failed = False
        self._kill_called = False
        self._watchdog_thread = None

    def _start_watchdog(self) -> None:
        """Kill the subprocess from a daemon thread if it exceeds timeout_s."""
        import threading

        def _watch():
            try:
                self.popen.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                # Kill the process group so child processes also stop.
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
        """Return hard positions, or None if the process is not ready."""
        full = self.wait_for_result_full(max_wait_s=max_wait_s)
        return None if full is None else full[0]

    def wait_for_result_full(
        self, max_wait_s: float = 0.0
    ) -> "Optional[tuple[np.ndarray, np.ndarray]]":
        """Return hard and soft positions, or None if unavailable."""
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
        """Abort the subprocess if still running."""
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
) -> AsyncDreamplaceHandle:
    """Launch DREAMPlace in the background and return a polling handle."""
    if not is_available():
        raise RuntimeError(
            f"DREAMPlace not installed at {DREAMPLACE_INSTALL}. "
            f"See system/v2/README.md to rebuild."
        )

    benchmark_dir_p = Path(benchmark_dir).resolve()
    design = benchmark_dir_p.name
    work_dir = Path(scratch_root).resolve() / design
    work_dir.mkdir(parents=True, exist_ok=True)

    # Return a cached handle when the inputs match.
    cache_key = _cache_key(
        benchmark_dir_p, iterations, random_seed, num_threads,
        soft_macros_movable, random_center_init,
    )
    cached = _try_load_cache(work_dir, cache_key)
    if cached is not None:
        return _CachedDreamplaceHandle(cached, start_time=time.time())

    # Convert to Bookshelf; reuse the caller's plc when available.
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
    )
    cfg_path = work_dir / f"{design}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    env = os.environ.copy()
    pythonpath = f"{DREAMPLACE_INSTALL}:{DREAMPLACE_INSTALL / 'dreamplace'}"
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + ":" + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    # Keep DREAMPlace from oversubscribing CPU pools.
    nt = str(max(1, int(num_threads)))
    env["OMP_NUM_THREADS"] = nt
    env["MKL_NUM_THREADS"] = nt
    env["OPENBLAS_NUM_THREADS"] = nt
    env["NUMEXPR_NUM_THREADS"] = nt

    # Load PLC before launch so result conversion stays off the wait path.
    if plc is None:
        plc = PlacementCost(str(benchmark_dir_p / "netlist.pb.txt"))
        init_plc = benchmark_dir_p / "initial.plc"
        if init_plc.exists():
            plc.restore_placement(str(init_plc), ifInital=True, ifReadComment=True)

    log_handle = (work_dir / "dreamplace.log").open("w")
    # Use a process group so the watchdog can terminate all descendants.
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
