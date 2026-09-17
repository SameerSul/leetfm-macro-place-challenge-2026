#!/usr/bin/env python3
"""Verify that the pinned DREAMPlace install and native extensions can load."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_ROOT = ROOT / "dreamplace_build"
RUNTIME_FILES = (
    "NonLinearPlace.py",
    "PlaceObj.py",
    "NesterovAcceleratedGradientOptimizer.py",
    "ops/adjust_node_area/adjust_node_area.py",
    "ops/move_boundary/move_boundary.py",
)


def verify_toolchain(build_root: Path) -> None:
    """Require the locked package archives, including their checksums."""
    lock = ROOT / "scripts/dreamplace/environment-linux-64.lock"
    expected = {line for line in lock.read_text().splitlines() if line.startswith("https://")}
    actual = set()
    for path in (build_root / "mamba/envs/dptool/conda-meta").glob("*.json"):
        package = json.loads(path.read_text())
        actual.add(f"{package.get('url')}#{package.get('md5')}")
    if actual != expected:
        raise RuntimeError(
            "DREAMPlace toolchain differs from environment-linux-64.lock; "
            "use a fresh DREAMPLACE_BUILD_ROOT to reproduce the pinned build"
        )


def verify_runtime(source: Path, build_root: Path) -> None:
    """Catch install-only edits that a rebuild would lose."""
    for rel in RUNTIME_FILES:
        a = source / "dreamplace" / rel
        b = build_root / "install/dreamplace" / rel
        if a.read_bytes() != b.read_bytes():
            raise RuntimeError(f"DREAMPlace source/install mismatch: {rel}")


def probe(build_root: Path, timeout_s: float = 30.0) -> tuple[bool, str]:
    """Run imports in the Python ABI that compiled DREAMPlace."""
    install = build_root / "install"
    python = build_root / "dpenv" / "bin" / "python"
    placer = install / "dreamplace" / "Placer.py"
    missing = [str(path) for path in (python, placer) if not path.exists()]
    if missing:
        return False, "missing required path(s): " + ", ".join(missing)

    env = os.environ.copy()
    paths = [str(install), str(install / "dreamplace")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    code = """
import json
import sys
from importlib.metadata import version
import torch
import dreamplace.configure as configure
import dreamplace.NonLinearPlace
from dreamplace.NesterovAcceleratedGradientOptimizer import NesterovAcceleratedGradientOptimizer
from dreamplace.ops.density_map import density_map
from dreamplace.ops.hpwl import hpwl
from dreamplace.ops.move_boundary import move_boundary
assert callable(getattr(NesterovAcceleratedGradientOptimizer, "step_bb", None))
assert sys.version_info[:3] == (3, 10, 20), sys.version
assert torch.__version__ == "2.4.1+cu121", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert not torch._C._GLIBCXX_USE_CXX11_ABI
assert configure.compile_configurations.get("CUDA_FOUND") == "TRUE"
with open(sys.argv[1]) as stream:
    for line in stream:
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.strip().split("==")
        assert version(name) == expected, (name, version(name), expected)
print(json.dumps({
    "python": sys.version.split()[0],
    "python_torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "dreamplace_cuda_found": configure.compile_configurations.get("CUDA_FOUND"),
    "bb_nesterov": True,
    "native_ops": [density_map.__name__, hpwl.__name__, move_boundary.__name__],
}))
"""
    try:
        proc = subprocess.run(
            [str(python), "-c", code, str(ROOT / "scripts/dreamplace/requirements.txt")],
            cwd=install,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"preflight process failed: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        return False, f"native import probe exited {proc.returncode}: {detail}"
    return True, proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "dreamplace_src")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        verify_toolchain(args.build_root.resolve())
        verify_runtime(args.source_dir.resolve(), args.build_root.resolve())
    except (OSError, RuntimeError) as exc:
        ok, detail = False, str(exc)
    else:
        ok, detail = probe(args.build_root.resolve())
    if args.json:
        print(json.dumps({"ok": ok, "detail": detail}))
    elif ok:
        print(f"DREAMPlace preflight passed: {detail}")
    else:
        print(f"DREAMPlace preflight failed: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
