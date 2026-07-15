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
    p for p in HERE.parents if (p / "pyproject.toml").exists() and (p / "macro_place").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macro_place._plc import PlacementCost  # noqa: E402

# Disk cache keyed by input files and DREAMPlace settings.
CACHE_VERSION = "v3"


def dreamplace_design_name(benchmark_dir: str | Path) -> str:
    """Return a stable, unique Bookshelf design name for a benchmark source dir."""
    benchmark_dir = Path(benchmark_dir).resolve()
    if benchmark_dir.name == "output_CT_Grouping" and benchmark_dir.parent.name == "netlist":
        return f"{benchmark_dir.parent.parent.name}_{benchmark_dir.name}"
    return benchmark_dir.name


def _file_fingerprint(p: Path) -> str:
    try:
        st = p.stat()
    except FileNotFoundError:
        return "missing"
    return f"{st.st_size}:{st.st_mtime_ns}"


def _cache_key(
    benchmark_dir: Path,
    iterations: int,
    random_seed: int,
    num_threads: int,
    soft_macros_movable: bool,
    random_center_init: bool,
    group_sig: str = "",
) -> str:
    netlist_fp = _file_fingerprint(benchmark_dir / "netlist.pb.txt")
    init_fp = _file_fingerprint(benchmark_dir / "initial.plc")
    design = dreamplace_design_name(benchmark_dir)
    raw = (
        f"{CACHE_VERSION}|{design}|path={benchmark_dir.as_posix()}|netlist={netlist_fp}|"
        f"init={init_fp}|iter={iterations}|seed={random_seed}|"
        f"threads={num_threads}|soft={int(soft_macros_movable)}|"
        f"rci={int(random_center_init)}|grp={group_sig}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _group_sig(cluster_groups, group_weight) -> str:
    """Stable fingerprint of grouping config for the cache key."""
    if not cluster_groups or group_weight <= 0:
        return ""
    n_groups = len(cluster_groups)
    n_members = sum(len(g) for g in cluster_groups)
    return f"w{int(group_weight)}-g{n_groups}-m{n_members}"


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
DREAMPLACE_INSTALL = REPO_ROOT / "dreamplace_build" / "install"
DREAMPLACE_PLACER = DREAMPLACE_INSTALL / "dreamplace" / "Placer.py"

# DREAMPlace extensions must run with the Python used to build them.
_DP_BUILD_PYTHON = REPO_ROOT / "dreamplace_build" / "dpenv" / "bin" / "python"
_REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
VENV_PYTHON = _DP_BUILD_PYTHON if _DP_BUILD_PYTHON.exists() else _REPO_VENV_PYTHON
_AVAILABILITY_CACHE: Optional[tuple[bool, str]] = None


def _probe_install(timeout_s: float = 30.0) -> tuple[bool, str]:
    """Import DREAMPlace and representative native ops in its build Python."""
    missing = [str(path) for path in (DREAMPLACE_PLACER, VENV_PYTHON) if not path.exists()]
    if missing:
        return False, "missing required path(s): " + ", ".join(missing)

    env = os.environ.copy()
    paths = [str(DREAMPLACE_INSTALL), str(DREAMPLACE_INSTALL / "dreamplace")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    code = """
import dreamplace.configure as configure
import dreamplace.NonLinearPlace
from dreamplace.ops.density_map import density_map
from dreamplace.ops.hpwl import hpwl
from dreamplace.ops.move_boundary import move_boundary
assert configure.compile_configurations
"""
    try:
        proc = subprocess.run(
            [str(VENV_PYTHON), "-c", code],
            cwd=DREAMPLACE_INSTALL,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"native import probe failed: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        return False, f"native import probe exited {proc.returncode}: {detail}"
    return True, "native imports passed"


def _default_dreamplace_config(
    aux_input: str,
    result_dir: str,
    random_seed: int = 1000,
    iterations: int = 200,
    num_threads: int = 4,
    random_center_init: bool = False,
    target_density: float = 0.75,
) -> dict:
    """Build the DREAMPlace config used for fast global placement."""
    cfg = {
        "aux_input": aux_input,
        "gpu": 0,
        "num_bins_x": 64,
        "num_bins_y": 64,
        "global_place_stages": [
            {
                "num_bins_x": 64,
                "num_bins_y": 64,
                "iteration": iterations,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
                "Llambda_density_weight_iteration": 1,
                "Lsub_iteration": 1,
            }
        ],
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


def is_available(refresh: bool = False) -> bool:
    """Return whether the required DREAMPlace install actually imports."""
    global _AVAILABILITY_CACHE
    if refresh or _AVAILABILITY_CACHE is None:
        _AVAILABILITY_CACHE = _probe_install()
    return _AVAILABILITY_CACHE[0]


def availability_error(refresh: bool = False) -> str:
    """Return actionable detail from the DREAMPlace availability probe."""
    is_available(refresh=refresh)
    assert _AVAILABILITY_CACHE is not None
    if _AVAILABILITY_CACHE[0]:
        return ""
    return (
        f"{_AVAILABILITY_CACHE[1]}. Run scripts/dreamplace/bootstrap.sh all "
        "to build and verify the pinned toolchain."
    )


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
    cluster_groups: "Optional[list]" = None,
    group_weight: int = 0,
    return_full: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run DREAMPlace and return hard-macro center positions."""
    if not is_available():
        raise RuntimeError(f"DREAMPlace unavailable: {availability_error()}")

    benchmark_dir = Path(benchmark_dir).resolve()
    design = dreamplace_design_name(benchmark_dir)
    work_dir = Path(scratch_root).resolve() / design
    work_dir.mkdir(parents=True, exist_ok=True)

    # Disk-cache fast path.
    cache_key = _cache_key(
        benchmark_dir,
        iterations,
        random_seed,
        num_threads,
        soft_macros_movable,
        random_center_init,
        group_sig=_group_sig(cluster_groups, group_weight),
    )
    cached = _try_load_cache(work_dir, cache_key)
    if cached is not None:
        hard_pos, soft_pos = cached
        print(
            f"  [dreamplace] {design}: {hard_pos.shape[0]} hard macros "
            f"(cache hit, skipped subprocess)"
        )
        if return_full:
            return hard_pos, soft_pos
        return hard_pos

    # Convert to Bookshelf.
    convert(
        str(benchmark_dir),
        str(work_dir),
        design=design,
        soft_macros_movable=soft_macros_movable,
        plc=plc,
        cluster_groups=cluster_groups,
        group_weight=group_weight,
    )

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
        if return_full:
            print(
                f"  [dreamplace] {design}: {hard_pos.shape[0]} hard macros placed "
                f"(global-place {dp_time:.1f}s)"
            )
            return hard_pos, soft_pos
        pos = hard_pos
    except Exception:
        pos = read_dreamplace_positions(plc, str(work_dir), design)
    print(
        f"  [dreamplace] {design}: {pos.shape[0]} hard macros placed "
        f"(global-place {dp_time:.1f}s)"
    )
    return pos


def _main():
    """Quick CLI smoke test."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--scratch", default="/tmp/dreamplace_v1")
    ap.add_argument("--keep-log", action="store_true")
    args = ap.parse_args()

    pos = run_dreamplace(args.benchmark, scratch_root=args.scratch, keep_log=args.keep_log)
    print(f"Got positions: shape={pos.shape}, dtype={pos.dtype}")
    print(f"  x range: [{pos[:, 0].min():.2f}, {pos[:, 0].max():.2f}]")
    print(f"  y range: [{pos[:, 1].min():.2f}, {pos[:, 1].max():.2f}]")


if __name__ == "__main__":
    _main()
