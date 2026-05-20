"""TILOS pb.txt → Bookshelf converter for DREAMPlace.

Reads a TILOS-format benchmark via PlacementCost (the existing parser),
extracts nodes / nets / pin offsets / canvas, and writes the 5-file
Bookshelf format that DREAMPlace's place_io op consumes:

    <design>.aux    -- index file
    <design>.nodes  -- list of cells/macros with sizes; "terminal" for fixed
    <design>.nets   -- net list with pin offsets
    <design>.pl     -- initial placement (LL corner coords) and /Fixed flag
    <design>.scl    -- site/row spec (we use a single row covering the canvas)

Conventions:
- TILOS positions are macro CENTER coordinates; Bookshelf .pl uses LL corner.
  We convert center → LL on write.
- TILOS pin x_offset/y_offset are relative to macro center; Bookshelf is the same.
  Direct passthrough.
- Hard macros are movable unless their TILOS fix_flag is set.
- Soft macros are marked terminal+/Fixed (stand-ins for stdcell clusters that
  the existing pipeline doesn't move). Can be flipped to movable later.
- I/O ports become tiny (1×1 micron) terminal nodes at their fixed positions.
- Net direction: TILOS driver pin → Bookshelf 'O', TILOS sink pins → 'I'.

Run as a script:
    uv run python submissions/varrahan/v1/dreamplace_bridge/pb_to_bookshelf.py \
        --benchmark external/MacroPlacement/Testcases/ICCAD04/ibm04 \
        --output /tmp/ibm04_bookshelf
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# Allow running as a script: add repo root so `macro_place` imports work.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macro_place._plc import PlacementCost  # noqa: E402


@dataclass
class BookshelfNode:
    name: str
    width: float
    height: float
    is_terminal: bool        # appears with "terminal" in .nodes
    x_ll: float              # lower-left corner X
    y_ll: float              # lower-left corner Y
    fixed: bool              # appears with /Fixed in .pl


@dataclass
class BookshelfNetPin:
    node_name: str
    direction: str           # 'I' | 'O' | 'B'
    x_offset: float          # relative to node center
    y_offset: float


@dataclass
class BookshelfNet:
    name: str
    pins: List[BookshelfNetPin] = field(default_factory=list)


def _safe_pos(node) -> Tuple[float, float]:
    """Read (x, y) from a TILOS node, defaulting to (0, 0)."""
    try:
        return node.get_pos()
    except Exception:
        return (0.0, 0.0)


def _safe_size(node) -> Tuple[float, float]:
    try:
        return (node.get_width(), node.get_height())
    except Exception:
        return (0.0, 0.0)


def _safe_fixed(node) -> bool:
    try:
        return bool(node.get_fix_flag())
    except Exception:
        return False


def _sanitize(name: str) -> str:
    """Bookshelf token characters: letters, digits, underscore, slash, hyphen, dot.
    Replace anything else with underscore."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in "_/-.":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def extract_bookshelf_data(
    plc: PlacementCost,
    soft_macros_movable: bool = False,
    port_size: float = 1.0,
    scale: int = 1000,
) -> Tuple[List[BookshelfNode], List[BookshelfNet], float, float, int]:
    """Extract Bookshelf-shaped data from a parsed TILOS PlacementCost.

    Bookshelf .nodes/.pl/.scl require INTEGER coordinates. TILOS data uses
    sub-micron decimals (e.g. macro 0.96x0.8). We multiply all positions/sizes
    by `scale` (default 1000) and round to integer. Pin offsets in .nets
    accept floats and are also scaled. The caller divides results by `scale`
    to get back to TILOS micron coordinates.

    Returns (nodes, nets, canvas_width_scaled, canvas_height_scaled, scale).
    """
    canvas_w, canvas_h = plc.get_canvas_width_height()

    nodes: List[BookshelfNode] = []
    name_seen = {}  # sanitized_name -> sanitized_name (deduped)

    def add_node(raw_name: str, w: float, h: float, x_center: float, y_center: float,
                 terminal: bool, fixed: bool) -> str:
        name = _sanitize(raw_name)
        # If duplicate sanitized name, append a suffix
        if name in name_seen:
            i = 1
            while f"{name}_{i}" in name_seen:
                i += 1
            name = f"{name}_{i}"
        name_seen[name] = name
        # Scale + round to int. Use max(1, ...) on size so no degenerate 0x0
        # nodes (DREAMPlace asserts on zero-size terminals).
        sw = max(1, round(float(w) * scale))
        sh = max(1, round(float(h) * scale))
        sx_ll = round((float(x_center) - float(w) / 2.0) * scale)
        sy_ll = round((float(y_center) - float(h) / 2.0) * scale)
        nodes.append(BookshelfNode(
            name=name,
            width=float(sw),
            height=float(sh),
            is_terminal=bool(terminal),
            x_ll=float(sx_ll),
            y_ll=float(sy_ll),
            fixed=bool(fixed),
        ))
        return name

    # Map TILOS macro name -> Bookshelf sanitized name (for net pin lookup later)
    tilos_to_bookshelf: dict = {}

    # Hard macros
    for idx in plc.hard_macro_indices:
        m = plc.modules_w_pins[idx]
        raw = m.get_name()
        w, h = _safe_size(m)
        x, y = _safe_pos(m)
        fixed = _safe_fixed(m)
        bs_name = add_node(raw, w, h, x, y, terminal=fixed, fixed=fixed)
        tilos_to_bookshelf[raw] = bs_name

    # Soft macros — by default mark as terminal+fixed
    for idx in plc.soft_macro_indices:
        m = plc.modules_w_pins[idx]
        raw = m.get_name()
        w, h = _safe_size(m)
        x, y = _safe_pos(m)
        is_terminal = (not soft_macros_movable) or _safe_fixed(m)
        is_fixed = is_terminal
        bs_name = add_node(raw, w, h, x, y, terminal=is_terminal, fixed=is_fixed)
        tilos_to_bookshelf[raw] = bs_name

    # I/O ports — tiny terminal nodes
    for idx in plc.port_indices:
        p = plc.modules_w_pins[idx]
        raw = p.get_name()
        x, y = _safe_pos(p)
        bs_name = add_node(raw, port_size, port_size, x, y, terminal=True, fixed=True)
        tilos_to_bookshelf[raw] = bs_name

    # Sort: movable first, terminals last (Bookshelf NumTerminals is trailing count)
    nodes.sort(key=lambda n: (n.is_terminal, n.name))

    # Build net topology from plc.nets: {driver_pin_name: [sink_pin_names]}
    nets: List[BookshelfNet] = []
    net_idx = 0

    def pin_to_node_and_offset(pin_name: str) -> Optional[Tuple[str, float, float]]:
        """Resolve a TILOS pin name to (bookshelf_node_name, x_offset, y_offset).
        Returns None if the pin can't be resolved.
        Ports are pins themselves (no parent); use port name with offset (0, 0)."""
        idx = plc.mod_name_to_indices.get(pin_name)
        if idx is None:
            return None
        pin = plc.modules_w_pins[idx]
        # Hard / soft macro pin: look up parent macro
        parent_raw = getattr(pin, "macro_name", None)
        if parent_raw and parent_raw in tilos_to_bookshelf:
            x_off = float(getattr(pin, "x_offset", 0.0)) * scale
            y_off = float(getattr(pin, "y_offset", 0.0)) * scale
            return (tilos_to_bookshelf[parent_raw], x_off, y_off)
        # Port (no parent macro): the port is its own node, offset is 0
        if pin_name in tilos_to_bookshelf:
            return (tilos_to_bookshelf[pin_name], 0.0, 0.0)
        return None

    for driver_pin_name, sink_pin_names in plc.nets.items():
        net_idx += 1
        net_name = f"n{net_idx}"
        bs_net = BookshelfNet(name=net_name)

        driver = pin_to_node_and_offset(driver_pin_name)
        if driver is not None:
            node_name, x_off, y_off = driver
            bs_net.pins.append(BookshelfNetPin(node_name, "O", x_off, y_off))

        for sink_name in sink_pin_names:
            sink = pin_to_node_and_offset(sink_name)
            if sink is None:
                continue
            node_name, x_off, y_off = sink
            bs_net.pins.append(BookshelfNetPin(node_name, "I", x_off, y_off))

        if len(bs_net.pins) >= 2:
            nets.append(bs_net)

    return nodes, nets, float(canvas_w) * scale, float(canvas_h) * scale, scale


def _write_aux(out_dir: Path, design: str) -> None:
    (out_dir / f"{design}.aux").write_text(
        f"RowBasedPlacement : {design}.nodes {design}.nets {design}.pl {design}.scl\n"
    )


def _write_nodes(out_dir: Path, design: str, nodes: List[BookshelfNode]) -> None:
    num_terminals = sum(1 for n in nodes if n.is_terminal)
    lines = ["UCLA nodes 1.0", "",
             f"NumNodes : {len(nodes)}",
             f"NumTerminals : {num_terminals}",
             ""]
    for n in nodes:
        suffix = "\tterminal" if n.is_terminal else ""
        # widths/heights are pre-scaled to integer values; emit as int
        lines.append(f"\t{n.name}\t{int(n.width)}\t{int(n.height)}{suffix}")
    (out_dir / f"{design}.nodes").write_text("\n".join(lines) + "\n")


def _write_nets(out_dir: Path, design: str, nets: List[BookshelfNet]) -> None:
    num_pins = sum(len(net.pins) for net in nets)
    lines = ["UCLA nets 1.0", "",
             f"NumNets : {len(nets)}",
             f"NumPins : {num_pins}",
             ""]
    for net in nets:
        lines.append(f"NetDegree : {len(net.pins)}\t{net.name}")
        for p in net.pins:
            lines.append(f"\t{p.node_name}\t{p.direction} : {p.x_offset:f}\t{p.y_offset:f}")
    (out_dir / f"{design}.nets").write_text("\n".join(lines) + "\n")


def _write_pl(out_dir: Path, design: str, nodes: List[BookshelfNode]) -> None:
    lines = ["UCLA pl 1.0", ""]
    for n in nodes:
        suffix = " /FIXED" if n.fixed else ""
        # positions are pre-scaled to integer values; emit as int
        lines.append(f"{n.name}\t{int(n.x_ll)}\t{int(n.y_ll)}\t: N{suffix}")
    (out_dir / f"{design}.pl").write_text("\n".join(lines) + "\n")


def _write_scl(out_dir: Path, design: str, canvas_w: float, canvas_h: float) -> None:
    """Single row covering the entire canvas. DREAMPlace's analytic global
    placer doesn't use rows for macro positioning — they exist for stdcell
    legalization, which we don't care about here.

    The .scl parser requires INTEGER values for Height/Coordinate/SubrowOrigin/
    NumSites — even though .pl/.nodes accept floats. We round up to ensure
    the row covers the canvas, then site grid is 1x1 unit."""
    row_height = max(1, int(canvas_h) + 1)  # +1 to ensure full coverage after int()
    sites = max(1, int(canvas_w) + 1)
    lines = [
        "UCLA scl 1.0", "",
        "NumRows : 1", "",
        "CoreRow Horizontal",
        "  Coordinate    :   0",
        f"  Height        :   {row_height}",
        "  Sitewidth     :    1",
        "  Sitespacing   :    1",
        "  Siteorient    :    1",
        "  Sitesymmetry  :    1",
        f"  SubrowOrigin  :   0\tNumSites  :  {sites}",
        "End",
    ]
    (out_dir / f"{design}.scl").write_text("\n".join(lines) + "\n")


def convert(benchmark_dir: str, output_dir: str, design: Optional[str] = None,
            soft_macros_movable: bool = False) -> Path:
    """Convert a TILOS benchmark dir to Bookshelf. Returns the .aux path."""
    benchmark_dir = Path(benchmark_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    netlist_file = benchmark_dir / "netlist.pb.txt"
    plc_file = benchmark_dir / "initial.plc"
    if not netlist_file.exists():
        raise FileNotFoundError(f"netlist not found: {netlist_file}")

    if design is None:
        design = benchmark_dir.name

    plc = PlacementCost(str(netlist_file))
    if plc_file.exists():
        plc.restore_placement(str(plc_file), ifInital=True, ifReadComment=True)

    nodes, nets, cw, ch, scale = extract_bookshelf_data(
        plc, soft_macros_movable=soft_macros_movable
    )

    print(f"  benchmark={benchmark_dir.name}: {len(nodes)} nodes "
          f"({sum(1 for n in nodes if n.is_terminal)} terminals), "
          f"{len(nets)} nets, canvas={cw:.0f}x{ch:.0f} (scale={scale})")

    _write_aux(output_dir, design)
    _write_nodes(output_dir, design, nodes)
    _write_nets(output_dir, design, nets)
    _write_pl(output_dir, design, nodes)
    _write_scl(output_dir, design, cw, ch)
    # Persist the scale factor for the inverse converter (positions out of DREAMPlace
    # are in scaled coords; divide by scale to get TILOS microns).
    (output_dir / f"{design}.scale").write_text(str(scale) + "\n")

    aux_path = output_dir / f"{design}.aux"
    print(f"  wrote {aux_path} (scale factor {scale})")
    return aux_path


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="Path to TILOS benchmark dir (containing netlist.pb.txt)")
    ap.add_argument("--output", required=True,
                    help="Directory to write Bookshelf files into")
    ap.add_argument("--design", default=None,
                    help="Design name prefix (defaults to benchmark dir name)")
    ap.add_argument("--soft-movable", action="store_true",
                    help="Mark soft macros as movable instead of terminal+fixed")
    args = ap.parse_args()

    convert(args.benchmark, args.output, args.design,
            soft_macros_movable=args.soft_movable)


if __name__ == "__main__":
    _main()
