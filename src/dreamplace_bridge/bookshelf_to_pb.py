"""Read DREAMPlace Bookshelf output back into TILOS coordinates."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macro_place._plc import PlacementCost  # noqa: E402

try:  # source of truth lives in the forward converter
    from .pb_to_bookshelf import _sanitize  # noqa: E402
except ImportError:  # standalone (non-package) invocation
    from pb_to_bookshelf import _sanitize  # type: ignore  # noqa: E402


@dataclass
class _NodeSize:
    width: float   # in scaled units (Bookshelf-side)
    height: float


def _read_node_sizes(nodes_file: Path) -> Dict[str, _NodeSize]:
    """Parse node sizes from our generated .nodes file."""
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
    """Read DREAMPlace .gp.pl and return hard-macro center positions."""
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

    # Build sanitized-name to TILOS hard-macro index map.
    name_to_hard_idx: Dict[str, int] = {}
    for tilos_idx in plc.hard_macro_indices:
        node = plc.modules_w_pins[tilos_idx]
        sanitized = _sanitize(node.get_name())
        name_to_hard_idx[sanitized] = tilos_idx

    n_hard = len(plc.hard_macro_indices)
    out = np.zeros((n_hard, 2), dtype=np.float64)

    # Keep the same order as `plc.hard_macro_indices`.
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


def read_dreamplace_positions_full(
    plc: PlacementCost,
    bookshelf_dir: str,
    design: str,
    output_pl: Optional[str] = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Read DREAMPlace's .gp.pl output and return BOTH hard and soft positions.

    This is the soft-macros-movable counterpart to `read_dreamplace_positions`.
    When `soft_macros_movable=True` is passed to the forward converter, DREAMPlace
    places hard AND soft macros; this function reads both sets.

    Returns
    -------
    hard_pos : np.ndarray [num_hard_macros, 2] float64
        Same as `read_dreamplace_positions`.
    soft_pos : np.ndarray [num_soft_macros, 2] float64
        Center coordinates in TILOS microns, indexed identically to
        `plc.soft_macro_indices`. Macros not found in the .gp.pl file keep
        their current `node.get_pos()` value (which is `initial.plc` if the
        plc was just loaded).
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

    def _read_set(tilos_idxs):
        out = np.zeros((len(tilos_idxs), 2), dtype=np.float64)
        for out_idx, tilos_idx in enumerate(tilos_idxs):
            node = plc.modules_w_pins[tilos_idx]
            sanitized = _sanitize(node.get_name())
            if sanitized in pl and sanitized in sizes:
                x_ll, y_ll, _fixed = pl[sanitized]
                sz = sizes[sanitized]
                out[out_idx, 0] = (x_ll + sz.width / 2.0) / scale
                out[out_idx, 1] = (y_ll + sz.height / 2.0) / scale
            else:
                try:
                    cur_x, cur_y = node.get_pos()
                    out[out_idx, 0] = float(cur_x)
                    out[out_idx, 1] = float(cur_y)
                except Exception:
                    pass
        return out

    hard_pos = _read_set(plc.hard_macro_indices)
    soft_pos = _read_set(plc.soft_macro_indices)
    return hard_pos, soft_pos


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
