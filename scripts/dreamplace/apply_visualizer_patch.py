"""Apply and verify the opt-in VivaPlace progress protocol in DREAMPlace.

The source and install trees are generated artifacts. Bootstrap runs this
tracked patcher before building and after installing so both copies receive the
same protocol even when the accepted local DREAMPlace revision changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    ROOT / "dreamplace_src/dreamplace/NonLinearPlace.py",
    ROOT / "dreamplace_build/install/dreamplace/NonLinearPlace.py",
)

PROTOCOL_VERSION = 1
BLOCK_START = "        # BEGIN VIVAPLACE_PROGRESS_PROTOCOL_V1\n"
BLOCK_END = "        # END VIVAPLACE_PROGRESS_PROTOCOL_V1\n"
LEGACY_START = '        vivaplace_sample_every = int(os.environ.get("VIVAPLACE_PROGRESS_EVERY"'
CALL_MARKER = (
    "                            emit_vivaplace_progress("
    "model.data_collections.pos[0], placedb, iteration)\n"
)
ITERATION_MARKER = "                            iteration += 1\n"


def _protocol_block() -> str:
    return f'''{BLOCK_START}        vivaplace_sample_every = int(os.environ.get("VIVAPLACE_PROGRESS_EVERY", "0") or 0)

        def emit_vivaplace_progress(pos, placedb, current_iteration):
            """Emit compact lower-left movable-node coordinates for diagnostics."""
            if vivaplace_sample_every <= 0 or current_iteration % vivaplace_sample_every:
                return
            try:
                count = int(placedb.num_movable_nodes)
                nodes = int(placedb.num_nodes)
                raw = pos.detach().cpu().numpy()
                xy = np.column_stack((raw[:count], raw[nodes : nodes + count])).astype(
                    np.float32, copy=False
                )
                payload = {{
                    "protocol": {PROTOCOL_VERSION},
                    "iteration": int(current_iteration),
                    "count": count,
                    "dtype": "float32",
                    "coordinates": base64.b64encode(xy.tobytes(order="C")).decode("ascii"),
                }}
                print("VIVAPLACE_PROGRESS " + json.dumps(payload, separators=(",", ":")), flush=True)
            except Exception:
                # Visualization must never alter placement success.
                pass
{BLOCK_END}'''


def _ensure_import(text: str, name: str) -> str:
    marker = f"import {name}\n"
    if marker in text:
        return text
    anchor = "import inspect\n"
    if anchor not in text:
        raise RuntimeError(f"DREAMPlace import marker not found while adding {name}")
    return text.replace(anchor, anchor + marker, 1)


def _replace_protocol_block(text: str, path: Path) -> str:
    block = _protocol_block()
    if BLOCK_START in text:
        start = text.index(BLOCK_START)
        end_marker = text.find(BLOCK_END, start)
        if end_marker < 0:
            raise RuntimeError(f"unterminated VivaPlace protocol block in {path}")
        end = end_marker + len(BLOCK_END)
        return text[:start] + block + text[end:]

    legacy = text.find(LEGACY_START)
    if legacy >= 0:
        end = text.find("        all_metrics = []\n", legacy)
        if end < 0:
            raise RuntimeError(f"legacy VivaPlace protocol boundary not found in {path}")
        return text[:legacy] + block + "\n" + text[end:]

    anchor = "        iteration = 0\n"
    if text.count(anchor) != 1:
        raise RuntimeError(f"DREAMPlace iteration declaration changed in {path}")
    return text.replace(anchor, anchor + block + "\n", 1)


def _normalize(text: str, path: Path) -> str:
    text = _ensure_import(text, "base64")
    text = _ensure_import(text, "json")
    text = _replace_protocol_block(text, path)
    if CALL_MARKER not in text:
        if text.count(ITERATION_MARKER) != 1:
            raise RuntimeError(f"DREAMPlace optimizer update marker changed in {path}")
        text = text.replace(ITERATION_MARKER, ITERATION_MARKER + CALL_MARKER, 1)
    return text


def _validate(text: str, path: Path) -> None:
    required = (
        "import base64\n",
        "import json\n",
        BLOCK_START,
        BLOCK_END,
        f'"protocol": {PROTOCOL_VERSION}',
        CALL_MARKER,
    )
    missing = [marker.strip() for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"incomplete VivaPlace protocol in {path}: {missing}")
    if text.count(BLOCK_START) != 1 or text.count(BLOCK_END) != 1:
        raise RuntimeError(f"duplicate VivaPlace protocol block in {path}")
    if text.count(CALL_MARKER) != 1:
        raise RuntimeError(f"duplicate VivaPlace optimizer progress call in {path}")


def patch(path: Path, *, check: bool = False) -> bool:
    if not path.is_file():
        return False
    original = path.read_text()
    normalized = _normalize(original, path)
    _validate(normalized, path)
    changed = normalized != original
    if check and changed:
        raise RuntimeError(f"VivaPlace DREAMPlace protocol patch is stale in {path}")
    if changed:
        path.write_text(normalized)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without modifying files")
    args = parser.parse_args(argv)
    existing = [path for path in TARGETS if path.is_file()]
    changed = [str(path) for path in existing if patch(path, check=args.check)]
    mode = "verified" if args.check else f"{len(changed)} file(s) changed"
    print(f"[patch] DREAMPlace visualizer protocol v{PROTOCOL_VERSION}: {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
