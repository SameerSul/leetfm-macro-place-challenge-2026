"""Idempotently add the opt-in VivaPlace progress protocol to DREAMPlace.

The DREAMPlace source/install trees are generated artifacts, so bootstrap calls
this tracked patcher before building and after installing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    ROOT / "dreamplace_src/dreamplace/NonLinearPlace.py",
    ROOT / "dreamplace_build/install/dreamplace/NonLinearPlace.py",
)

IMPORT_MARKER = "import base64\nimport json\n"
FUNCTION_MARKER = "        def emit_vivaplace_progress(pos, placedb, current_iteration):\n"
CALL_MARKER = (
    "                            emit_vivaplace_progress("
    "model.data_collections.pos[0], placedb, iteration)\n"
)


def patch(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text()
    original = text
    if IMPORT_MARKER not in text:
        text = text.replace("import inspect\n", "import inspect\n" + IMPORT_MARKER, 1)
    if FUNCTION_MARKER not in text:
        needle = "        iteration = 0\n"
        addition = '''        vivaplace_sample_every = int(os.environ.get("VIVAPLACE_PROGRESS_EVERY", "0") or 0)

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
                payload = {
                    "iteration": int(current_iteration),
                    "count": count,
                    "dtype": "float32",
                    "coordinates": base64.b64encode(xy.tobytes(order="C")).decode("ascii"),
                }
                print("VIVAPLACE_PROGRESS " + json.dumps(payload, separators=(",", ":")), flush=True)
            except Exception:
                # Visualization must never alter placement success.
                pass

'''
        if needle not in text:
            raise RuntimeError(f"DREAMPlace iteration marker not found in {path}")
        text = text.replace(needle, needle + addition, 1)
    if CALL_MARKER not in text:
        needle = "                            iteration += 1\n"
        if needle not in text:
            raise RuntimeError(f"DREAMPlace update marker not found in {path}")
        text = text.replace(needle, needle + CALL_MARKER, 1)
    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> int:
    changed = [str(path) for path in TARGETS if patch(path)]
    print(f"[patch] DREAMPlace visualizer protocol: {len(changed)} file(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
