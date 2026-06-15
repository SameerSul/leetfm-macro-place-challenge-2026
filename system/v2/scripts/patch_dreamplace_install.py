"""Repair a stale import in the (gitignored) DREAMPlace install tree.

The DREAMPlace install under `system/dreamplace_build/install/` is a build
artifact and is gitignored, so fixes there are NOT captured by git and are lost
on any rebuild. One such fix: after the 2026-06-11 repo restructure, a single op
file kept an absolute import with the old `varrahan.` path prefix:

    ops/move_boundary/move_boundary.py:
      import varrahan.dreamplace_build.install.dreamplace.configure as configure

That module path no longer resolves -> `ModuleNotFoundError: No module named
'varrahan'` killed EVERY DREAMPlace subprocess ~4s after launch, masked by the
bridge as a benign "not ready; killing subprocess" log line. Net effect: DP
produced zero seeds (placer ran a basin short, ~+0.011 proxy). See ISSUES.md.

This script idempotently rewrites that import to the convention every other op
uses (`import dreamplace.configure as configure`). Run it once after building /
reinstalling DREAMPlace:

    uv run python system/v2/scripts/patch_dreamplace_install.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL = REPO_ROOT / "system" / "dreamplace_build" / "install" / "dreamplace"

PATCHES = [
    # (relative path under install/dreamplace, bad line, good line)
    (
        "ops/move_boundary/move_boundary.py",
        "import varrahan.dreamplace_build.install.dreamplace.configure as configure",
        "import dreamplace.configure as configure",
    ),
]


def main() -> int:
    if not INSTALL.is_dir():
        print(f"[patch] DREAMPlace install not found at {INSTALL}; nothing to do.")
        return 0
    changed = 0
    for rel, bad, good in PATCHES:
        f = INSTALL / rel
        if not f.is_file():
            print(f"[patch] {rel}: missing, skipped")
            continue
        text = f.read_text()
        if bad in text:
            f.write_text(text.replace(bad, good))
            print(f"[patch] {rel}: fixed stale import")
            changed += 1
        elif good in text:
            print(f"[patch] {rel}: already correct")
        else:
            print(f"[patch] {rel}: neither bad nor good import found (manual check)")
    print(f"[patch] done ({changed} file(s) changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
