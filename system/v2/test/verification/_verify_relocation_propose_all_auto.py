"""Verify hard-relocation propose-all env parsing, including CUDA auto mode.

Usage:
  PYTHONPATH=system/v2/src \
  uv run python system/v2/test/verification/_verify_relocation_propose_all_auto.py
"""

from __future__ import annotations

from placer.pipeline.macro_placer import _reloc_propose_all_enabled


def main() -> int:
    cases = [
        ("", "cpu", False),
        ("0", "cuda", False),
        ("false", "cuda", False),
        ("1", "cpu", True),
        ("yes", "cpu", True),
        ("auto", "cpu", False),
        ("auto", "cuda", True),
        ("cuda", "cpu", False),
        ("cuda", "cuda", True),
        ("gpu", "cpu", False),
        ("gpu", "cuda", True),
        ("nonsense", "cuda", False),
    ]
    for raw, backend, expected in cases:
        actual = _reloc_propose_all_enabled(raw, backend)
        if actual is not expected:
            raise AssertionError(
                f"V2_RELOC_PROPOSE_ALL={raw!r}, backend={backend!r}: "
                f"expected {expected}, got {actual}"
            )
    print("PASS relocation_propose_all_auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
