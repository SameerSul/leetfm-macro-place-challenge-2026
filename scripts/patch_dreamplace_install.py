"""Repair or verify the portable configure import in a DREAMPlace install."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL = REPO_ROOT / "dreamplace_build" / "install"

PATCHES = [
    # (relative path under install/dreamplace, bad line, good line)
    (
        "ops/move_boundary/move_boundary.py",
        "import varrahan.dreamplace_build.install.dreamplace.configure as configure",
        "import dreamplace.configure as configure",
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, default=INSTALL)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    changed = 0
    for rel, bad, good in PATCHES:
        f = args.install_dir / "dreamplace" / rel
        text = f.read_text()
        if bad in text:
            if args.check:
                raise RuntimeError(f"stale DREAMPlace configure import in {f}")
            f.write_text(text.replace(bad, good))
            print(f"[patch] {rel}: fixed stale import")
            changed += 1
        elif good in text:
            print(f"[patch] {rel}: already correct")
        else:
            raise RuntimeError(f"DREAMPlace configure import not found in {f}")
    print(f"[patch] done ({changed} file(s) changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
