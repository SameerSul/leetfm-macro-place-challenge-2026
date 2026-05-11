"""Bookshelf → TILOS positions back-converter.

Reads DREAMPlace's output `.gp.pl` (the global-placement result) plus the
`.nodes` and `.scale` files we generated alongside the input, and returns a
[num_hard_macros, 2] numpy array of CENTER coordinates in TILOS microns,
indexed identically to `plc.hard_macro_indices`.

The forward converter (`pb_to_bookshelf.py`) sanitizes TILOS macro names
into Bookshelf-legal tokens and writes a 5-file Bookshelf bundle with all
positions and sizes scaled up by an integer factor (default 1000) so they
fit Bookshelf's integer-only `.nodes`/`.pl`/`.scl` requirement. This module
inverts both transforms:

    bookshelf_x_ll * (size / 2) → micron x_center
    micron_x = bookshelf_x_center / scale

Soft macros and ports are NOT returned — they're either fixed in the
forward conversion (default) or out of scope for the active placer's
hard-macro-only restart slot. The caller can layer soft macros from
`benchmark.macro_positions[n:]` if it wants a complete `[num_macros, 2]`.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macro_place._plc import PlacementCost  # noqa: E402

# Same character set as forward converter's _sanitize() — keep in sync.
_BOOKSHELF_TOKEN_RE = re.compile(r"^[A-Za-z0-9_/.\-]+$")


def _sanitize(name: str) -> str:
    """Mirror of pb_to_bookshelf._sanitize. Replace illegal chars with _."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in "_/-.":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


@dataclass
class _NodeSize:
    width: float   # in scaled units (Bookshelf-side)
    height: float


def _read_node_sizes(nodes_file: Path) -> Dict[str, _NodeSize]:
    """Parse our generated .nodes file: lines look like '\\tname\\tWIDTH\\tHEIGHT[\\tterminal]'."""
    sizes: Dict[str, _NodeSize] = {}
    with nodes_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("UCLA"):
                continue
            if line.startswith("NumNodes") or line.startswith("NumTerminals"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name, w, h = parts[0], parts[1], parts[2]
            try:
                sizes[name] = _NodeSize(width=float(w), height=float(h))
            except ValueError:
                continue
    return sizes


def _read_pl(pl_file: Path) -> Dict[str, tuple]:
    """Parse a Bookshelf .pl file. Returns {name: (x_ll, y_ll, fixed)}."""
    out: Dict[str, tuple] = {}
    with pl_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("UCLA"):
                continue
            # name x y : ORIENT [/FIXED]
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            head = parts[0].split()
            if len(head) < 3:
                continue
            name = head[0]
            try:
                x_ll = float(head[1])
                y_ll = float(head[2])
            except ValueError:
                continue
            fixed = "/FIXED" in parts[1].upper() or "/FIX" in parts[1].upper()
            out[name] = (x_ll, y_ll, fixed)
    return out


def read_dreamplace_positions(
    plc: PlacementCost,
    bookshelf_dir: str,
    design: str,
    output_pl: Optional[str] = None,
) -> np.ndarray:
    """Read DREAMPlace's output .gp.pl and return TILOS hard macro positions.

    Parameters
    ----------
    plc : PlacementCost
        The TILOS-parsed benchmark. Used to map names back to indices.
    bookshelf_dir : str
        Directory that holds the .nodes/.scale files (the FORWARD conversion
        output dir we created earlier).
    design : str
        Design name prefix (e.g. "ibm04"). Files looked up:
          {bookshelf_dir}/{design}.nodes
          {bookshelf_dir}/{design}.scale
    output_pl : Optional[str]
        Path to DREAMPlace's output .gp.pl. If None, defaults to
        {bookshelf_dir}/results/{design}/{design}.gp.pl (DREAMPlace's standard
        output location given result_dir={bookshelf_dir}/results).

    Returns
    -------
    pos : np.ndarray, shape [num_hard_macros, 2], dtype float64
        Center coordinates in TILOS microns, indexed identically to
        `plc.hard_macro_indices`. Macros not found in the .gp.pl file
        (shouldn't happen if the forward conversion was clean) keep their
        current `plc.get_node_location()` value.
    """
    bookshelf_dir = Path(bookshelf_dir).resolve()
    if output_pl is None:
        output_pl = bookshelf_dir / "results" / design / f"{design}.gp.pl"
    output_pl = Path(output_pl)
    if not output_pl.exists():
        raise FileNotFoundError(f"DREAMPlace output not found: {output_pl}")

    nodes_file = bookshelf_dir / f"{design}.nodes"
    scale_file = bookshelf_dir / f"{design}.scale"
    if not nodes_file.exists():
        raise FileNotFoundError(f"missing .nodes file: {nodes_file}")
    if not scale_file.exists():
        raise FileNotFoundError(f"missing .scale file: {scale_file}")

    scale = float(scale_file.read_text().strip())
    sizes = _read_node_sizes(nodes_file)
    pl = _read_pl(output_pl)

    # Build sanitized-name → TILOS hard macro idx map.
    # Forward converter dedupes on collision with `_1`, `_2` suffixes; if any
    # of our hard macro names collide we'd see those, but in practice IBM
    # benchmark names are already Bookshelf-legal and unique.
    name_to_hard_idx: Dict[str, int] = {}
    for tilos_idx in plc.hard_macro_indices:
        node = plc.modules_w_pins[tilos_idx]
        sanitized = _sanitize(node.get_name())
        name_to_hard_idx[sanitized] = tilos_idx

    n_hard = len(plc.hard_macro_indices)
    out = np.zeros((n_hard, 2), dtype=np.float64)

    # Order outputs to match `plc.hard_macro_indices` order (the convention
    # the rest of the placer uses).
    for out_idx, tilos_idx in enumerate(plc.hard_macro_indices):
        node = plc.modules_w_pins[tilos_idx]
        sanitized = _sanitize(node.get_name())
        if sanitized in pl and sanitized in sizes:
            x_ll, y_ll, _fixed = pl[sanitized]
            sz = sizes[sanitized]
            x_center_scaled = x_ll + sz.width / 2.0
            y_center_scaled = y_ll + sz.height / 2.0
            out[out_idx, 0] = x_center_scaled / scale
            out[out_idx, 1] = y_center_scaled / scale
        else:
            # Fallback: keep current TILOS position
            try:
                cur_x, cur_y = node.get_pos()
                out[out_idx, 0] = float(cur_x)
                out[out_idx, 1] = float(cur_y)
            except Exception:
                pass

    return out


def _main():
    """Quick CLI for sanity testing: print first few positions."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="TILOS benchmark dir (containing netlist.pb.txt)")
    ap.add_argument("--bookshelf-dir", required=True,
                    help="Directory with the forward-converted Bookshelf files")
    ap.add_argument("--design", default=None,
                    help="Design name prefix (defaults to benchmark dir name)")
    ap.add_argument("--output-pl", default=None,
                    help="Override path to DREAMPlace .gp.pl output")
    args = ap.parse_args()

    bench_dir = Path(args.benchmark).resolve()
    netlist = bench_dir / "netlist.pb.txt"
    plc_init = bench_dir / "initial.plc"
    plc = PlacementCost(str(netlist))
    if plc_init.exists():
        plc.restore_placement(str(plc_init), ifInital=True, ifReadComment=True)

    design = args.design or bench_dir.name
    pos = read_dreamplace_positions(plc, args.bookshelf_dir, design, args.output_pl)

    print(f"Loaded {pos.shape[0]} hard macro positions from DREAMPlace output.")
    print("Index | TILOS name | Old (x,y) | New from DREAMPlace (x,y)")
    for i, tilos_idx in enumerate(plc.hard_macro_indices[:8]):
        node = plc.modules_w_pins[tilos_idx]
        old = node.get_pos()
        new = pos[i]
        print(f"  {i:3d}  {node.get_name():12s}  ({old[0]:8.2f},{old[1]:8.2f})  →  "
              f"({new[0]:8.2f},{new[1]:8.2f})")


if __name__ == "__main__":
    _main()
