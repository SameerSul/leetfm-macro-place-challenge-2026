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
import torch
import dreamplace.configure as configure
from dreamplace.ops.density_map import density_map
from dreamplace.ops.hpwl import hpwl
from dreamplace.ops.move_boundary import move_boundary
print(json.dumps({
    "python_torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "dreamplace_cuda_found": configure.compile_configurations.get("CUDA_FOUND"),
    "native_ops": [density_map.__name__, hpwl.__name__, move_boundary.__name__],
}))
"""
    try:
        proc = subprocess.run(
            [str(python), "-c", code],
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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
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
