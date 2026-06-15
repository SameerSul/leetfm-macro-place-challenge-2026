"""Convert a TILOS benchmark to Bookshelf files for DREAMPlace."""

from __future__ import annotations

import argparse
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
    """Replace characters Bookshelf cannot use in names."""
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
    cluster_groups: "Optional[List[List[str]]]" = None,
    group_weight: int = 0,
) -> Tuple[List[BookshelfNode], List[BookshelfNet], float, float, int]:
    """Extract scaled Bookshelf data from a parsed TILOS placement.

    When `cluster_groups` (lists of TILOS hard-macro names) and `group_weight>0`
    are given, append `group_weight` synthetic clique nets per cluster. These
    extra nets add weighted bounding-box HPWL over each subsystem, so DREAMPlace
    softly keeps connected macros together. They carry no real pins/offsets and
    are ignored on readback (which maps by macro name only). DREAMPlace's
    `ignore_net_degree` skips clusters larger than that cap automatically.
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
        # Use integer units and avoid zero-size nodes.
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

    # Map TILOS names to Bookshelf names for pin lookup.
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

    # Soft macros - by default mark as terminal+fixed
    for idx in plc.soft_macro_indices:
        m = plc.modules_w_pins[idx]
        raw = m.get_name()
        w, h = _safe_size(m)
        x, y = _safe_pos(m)
        is_terminal = (not soft_macros_movable) or _safe_fixed(m)
        is_fixed = is_terminal
        bs_name = add_node(raw, w, h, x, y, terminal=is_terminal, fixed=is_fixed)
        tilos_to_bookshelf[raw] = bs_name

    # I/O ports become tiny fixed nodes.
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

    # Synthetic cluster-grouping nets: one clique net per cluster, duplicated
    # group_weight times to scale its pull. Members are TILOS names mapped to
    # Bookshelf node names; unresolved names are skipped.
    if cluster_groups and group_weight > 0:
        for gi, member_names in enumerate(cluster_groups):
            mbs = [tilos_to_bookshelf[m] for m in member_names
                   if m in tilos_to_bookshelf]
            if len(mbs) < 2:
                continue
            for k in range(int(group_weight)):
                net_idx += 1
                g_net = BookshelfNet(name=f"grp{gi}_{k}")
                g_net.pins.append(BookshelfNetPin(mbs[0], "O", 0.0, 0.0))
                for m in mbs[1:]:
                    g_net.pins.append(BookshelfNetPin(m, "I", 0.0, 0.0))
                nets.append(g_net)

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


def _write_scl(out_dir: Path, design: str, canvas_w: float, canvas_h: float,
               num_rows_target: int = 8) -> None:
    """Write a simple multi-row site grid for DREAMPlace."""
    canvas_w_i = max(1, int(canvas_w) + 1)
    canvas_h_i = max(1, int(canvas_h) + 1)
    # Use ceil so the rows cover the canvas.
    row_height = max(1, (canvas_h_i + num_rows_target - 1) // num_rows_target)
    num_rows = num_rows_target
    sites = canvas_w_i

    lines = ["UCLA scl 1.0", "",
             f"NumRows : {num_rows}", ""]
    for r in range(num_rows):
        y = r * row_height
        lines += [
            "CoreRow Horizontal",
            f"  Coordinate    :   {y}",
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
            soft_macros_movable: bool = False,
            plc: Optional[PlacementCost] = None,
            cluster_groups: "Optional[List[List[str]]]" = None,
            group_weight: int = 0) -> Path:
    """Convert a TILOS benchmark dir to Bookshelf and return the .aux path."""
    benchmark_dir = Path(benchmark_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    netlist_file = benchmark_dir / "netlist.pb.txt"
    plc_file = benchmark_dir / "initial.plc"
    if not netlist_file.exists():
        raise FileNotFoundError(f"netlist not found: {netlist_file}")

    if design is None:
        design = benchmark_dir.name

    if plc is None:
        plc = PlacementCost(str(netlist_file))
        if plc_file.exists():
            plc.restore_placement(str(plc_file), ifInital=True, ifReadComment=True)

    nodes, nets, cw, ch, scale = extract_bookshelf_data(
        plc, soft_macros_movable=soft_macros_movable,
        cluster_groups=cluster_groups, group_weight=group_weight,
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
