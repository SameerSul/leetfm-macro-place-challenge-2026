"""Design -> Benchmark builder: the bridge from EDA files to the placer.

Strategy: rather than teaching the placer a second native format, convert
the merged Design into the ICCAD04 protobuf + plc pair the whole stack
already understands, then load it through the standard `load_benchmark`.
That gives exact TILOS scoring and identical placer behavior for free.

What the conversion does, for any input combination:
  - hard macros   = components whose LEF CLASS is BLOCK (or, with no BLOCK
                    masters anywhere, components much larger than the median)
  - soft macros   = standard cells clustered into groups (by location when
                    placed, by connectivity when not)
  - blockages     = dummy fixed hard macros covering the keep-out rect
  - die area      = DEF DIEAREA, else a square sized from total area
  - missing seeds = shelf-packed spread (the placer expects a seed)
  - net weights   = SDC/Liberty weights, carried on benchmark.net_weights
"""

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .design import Design

BLOCKAGE_PREFIX = "__blockage_"


@dataclass
class BuildResult:
    benchmark: object  # macro_place.benchmark.Benchmark
    plc: object  # PlacementCost - exact scorer for this design
    design: Design
    hard_names: List[str]  # tensor index -> component name (blockages incl.)
    soft_members: List[List[str]]  # soft index -> std cell names
    soft_seed: List[Tuple[float, float]]  # soft index -> seed center
    port_names: List[str]  # port index -> I/O pin name
    workdir: Path  # where netlist.pb.txt / initial.plc were written
    origin: Tuple[float, float] = (0.0, 0.0)  # die-area lower-left (microns)
    dropped: List[str] = field(default_factory=list)  # comps with no master

    def placements_um(self, placement) -> Dict[str, Tuple[float, float]]:
        """Map a placement tensor back to per-component lower-left microns.

        Hard macros take their placed center; every std cell shifts by its
        cluster's displacement (or sits at the cluster center if it had no
        initial location). Blockage dummies are internal and excluded.
        """
        pos = placement.detach().cpu().numpy().astype(np.float64)
        ox, oy = self.origin
        out = {}
        for i, name in enumerate(self.hard_names):
            if name.startswith(BLOCKAGE_PREFIX):
                continue
            comp = self.design.components[name]
            w, h = self.design.size_of(comp)
            out[name] = (pos[i, 0] - w / 2 + ox, pos[i, 1] - h / 2 + oy)
        nh = len(self.hard_names)
        for m, members in enumerate(self.soft_members):
            cx, cy = pos[nh + m]
            sx, sy = self.soft_seed[m]
            dx, dy = cx - sx, cy - sy  # cluster shift is translation-invariant
            for name in members:
                comp = self.design.components[name]
                w, h = self.design.size_of(comp)
                if comp.pos is not None:
                    out[name] = (comp.pos[0] + dx, comp.pos[1] + dy)
                else:
                    out[name] = (cx - w / 2 + ox, cy - h / 2 + oy)
        return out


def build_benchmark(
    design: Design,
    workdir=None,
    cells_per_cluster: int = 50,
    utilization: float = 0.80,
    grid_base: int = 40,
) -> BuildResult:
    """Convert a Design into a placer-ready Benchmark (+ exact scorer)."""
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="eda_io_"))
    workdir.mkdir(parents=True, exist_ok=True)

    dropped = [n for n, c in design.components.items()
               if c.master not in design.masters
               or design.masters[c.master].width <= 0]
    comps = {n: c for n, c in design.components.items() if n not in dropped}

    hard, cells = _classify(design, comps)
    canvas_w, canvas_h = _canvas(design, comps, utilization)
    # DEF coordinates are absolute; the placer canvas starts at (0, 0)
    ox, oy = design.die_area[:2] if design.die_area else (0.0, 0.0)
    ports, port_pos = _ports(design, canvas_w, canvas_h, (ox, oy))
    clusters, soft_sizes, soft_seed = _cluster_cells(
        design, cells, canvas_w, canvas_h, (ox, oy), cells_per_cluster)

    hard_names = list(hard)
    blockage_sizes, blockage_pos = [], []
    for k, (x0, y0, x1, y1) in enumerate(design.blockages):
        hard_names.append(f"{BLOCKAGE_PREFIX}{k}")
        blockage_sizes.append((x1 - x0, y1 - y0))
        blockage_pos.append(((x0 + x1) / 2 - ox, (y0 + y1) / 2 - oy))

    hard_sizes, hard_pos, fixed = [], [], []
    for name in hard:
        comp = comps[name]
        w, h = design.size_of(comp)
        hard_sizes.append((w, h))
        if comp.pos is not None:
            hard_pos.append((comp.pos[0] + w / 2 - ox, comp.pos[1] + h / 2 - oy))
        else:
            hard_pos.append(None)
        fixed.append(comp.status == "FIXED")
    hard_sizes += blockage_sizes
    hard_pos += blockage_pos
    fixed += [True] * len(blockage_sizes)

    _seed_missing(hard_sizes, hard_pos, fixed, canvas_w, canvas_h)

    cell_cluster = {}
    for m, members in enumerate(clusters):
        for name in members:
            cell_cluster[name] = m
    hard_index = {name: i for i, name in enumerate(hard_names)}
    port_index = {name: p for p, name in enumerate(ports)}
    nets = _map_nets(design, hard_index, cell_cluster, port_index)

    pb_path = workdir / "netlist.pb.txt"
    _write_pb(pb_path, design, canvas_w, canvas_h, hard_names, hard_sizes,
              hard_pos, soft_sizes, soft_seed, ports, port_pos, nets, comps)

    from macro_place._plc import PlacementCost
    plc = PlacementCost(str(pb_path))
    for local_i, idx in enumerate(plc.hard_macro_indices):
        if fixed[local_i]:
            plc.modules_w_pins[idx].set_fix_flag(True)
    grid_cols, grid_rows = _grid(canvas_w, canvas_h, grid_base)
    total_area = sum(w * h for w, h in hard_sizes) + sum(
        w * h for w, h in soft_sizes)
    _write_plc(workdir / "initial.plc", design, canvas_w, canvas_h,
               grid_cols, grid_rows, plc, round(total_area, 4))

    from macro_place.loader import load_benchmark
    benchmark, plc = load_benchmark(str(pb_path), str(workdir / "initial.plc"))
    benchmark.name = design.name
    benchmark._cached_plc = plc  # exact scoring without an ICCAD04 path
    _attach_net_tensors(benchmark, nets, len(hard_names))

    return BuildResult(
        benchmark=benchmark, plc=plc, design=design, hard_names=hard_names,
        soft_members=clusters, soft_seed=soft_seed, port_names=ports,
        workdir=workdir, origin=(ox, oy), dropped=dropped,
    )


# ── Classification / geometry ────────────────────────────────────────────────


def _classify(design, comps):
    """Split components into hard macros and standard cells."""
    have_blocks = any(m.is_block for m in design.masters.values())
    areas = sorted(design.masters[c.master].width * design.masters[c.master].height
                   for c in comps.values())
    median = areas[len(areas) // 2] if areas else 0.0
    hard, cells = [], []
    for name, comp in comps.items():
        master = design.masters[comp.master]
        if master.is_block or (not have_blocks
                               and master.width * master.height > max(10 * median, 4.0)):
            hard.append(name)
        else:
            cells.append(name)
    return hard, cells


def _canvas(design, comps, utilization):
    if design.die_area:
        x0, y0, x1, y1 = design.die_area
        return x1 - x0, y1 - y0
    total = sum(design.masters[c.master].width * design.masters[c.master].height
                for c in comps.values())
    side = math.sqrt(max(total, 1.0) / utilization)
    biggest = max((max(design.size_of(c)) for c in comps.values()), default=1.0)
    side = max(side, biggest * 1.1)
    return side, side


def _grid(canvas_w, canvas_h, base):
    aspect = math.sqrt(canvas_w / canvas_h)
    cols = min(50, max(10, round(base * aspect)))
    rows = min(50, max(10, round(base / aspect)))
    return cols, rows


def _side_of(x, y, w, h):
    d = {"LEFT": x, "RIGHT": w - x, "BOTTOM": y, "TOP": h - y}
    return min(d, key=d.get)


def _ports(design, canvas_w, canvas_h, origin):
    """Order I/O pins; spread any without a location along the perimeter.

    Returns (names, canvas-relative positions).
    """
    ox, oy = origin
    names = list(design.io_pins)
    positions = []
    missing = [n for n in names if design.io_pins[n].pos is None]
    spread = {}
    for k, name in enumerate(missing):
        t = (k + 0.5) / len(missing) * 2 * (canvas_w + canvas_h)
        if t < canvas_w:
            spread[name] = (t, 0.0)
        elif t < canvas_w + canvas_h:
            spread[name] = (canvas_w, t - canvas_w)
        elif t < 2 * canvas_w + canvas_h:
            spread[name] = (2 * canvas_w + canvas_h - t, canvas_h)
        else:
            spread[name] = (0.0, 2 * (canvas_w + canvas_h) - t)
    for name in names:
        pos = design.io_pins[name].pos
        if pos is None:
            positions.append(spread[name])
        else:
            positions.append((min(max(pos[0] - ox, 0.0), canvas_w),
                              min(max(pos[1] - oy, 0.0), canvas_h)))
    return names, positions


def _cluster_cells(design, cells, canvas_w, canvas_h, origin, per_cluster):
    """Group std cells into soft macros: by location if placed, else by nets."""
    if not cells:
        return [], [], []
    ox, oy = origin
    placed = [c for c in cells if design.components[c].pos is not None]
    if len(placed) >= len(cells) // 2:
        groups = _cluster_by_location(design, cells, canvas_w, canvas_h,
                                      origin, per_cluster)
    else:
        groups = _cluster_by_connectivity(design, cells, per_cluster)

    sizes, seeds = [], []
    for members in groups:
        area = wx = wy = wsum = 0.0
        for name in members:
            comp = design.components[name]
            w, h = design.size_of(comp)
            area += w * h
            if comp.pos is not None:
                wx += (comp.pos[0] + w / 2 - ox) * w * h
                wy += (comp.pos[1] + h / 2 - oy) * w * h
                wsum += w * h
        side = math.sqrt(area)
        side = min(side, 0.95 * min(canvas_w, canvas_h))
        sizes.append((side, side))
        if wsum > 0:
            cx = min(max(wx / wsum, side / 2), canvas_w - side / 2)
            cy = min(max(wy / wsum, side / 2), canvas_h - side / 2)
            seeds.append((cx, cy))
        else:
            seeds.append(None)
    _seed_missing(sizes, seeds, [False] * len(seeds), canvas_w, canvas_h)
    return groups, sizes, seeds


def _cluster_by_location(design, cells, canvas_w, canvas_h, origin, per_cluster):
    ox, oy = origin
    g = max(1, min(32, round(math.sqrt(len(cells) / per_cluster))))
    buckets = {}
    for name in cells:
        comp = design.components[name]
        if comp.pos is not None:
            w, h = design.size_of(comp)
            cx, cy = comp.pos[0] + w / 2 - ox, comp.pos[1] + h / 2 - oy
        else:
            cx, cy = canvas_w / 2, canvas_h / 2
        key = (min(g - 1, max(0, int(cx / canvas_w * g))),
               min(g - 1, max(0, int(cy / canvas_h * g))))
        buckets.setdefault(key, []).append(name)
    return [buckets[k] for k in sorted(buckets)]


def _cluster_by_connectivity(design, cells, per_cluster):
    """Greedy net-order agglomeration with a size cap - no placement needed."""
    cell_set = set(cells)
    parent = {c: c for c in cells}
    size = {c: 1 for c in cells}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for net in sorted(design.nets, key=lambda n: len(n.terms)):
        members = [i for i, _ in net.terms if i in cell_set]
        for a, b in zip(members, members[1:]):
            ra, rb = find(a), find(b)
            if ra != rb and size[ra] + size[rb] <= per_cluster:
                parent[rb] = ra
                size[ra] += size[rb]
    groups = {}
    for c in cells:
        groups.setdefault(find(c), []).append(c)
    return [groups[k] for k in sorted(groups)]


def _seed_missing(sizes, pos, fixed, canvas_w, canvas_h):
    """Shelf-pack a seed location for every entry whose position is None."""
    todo = [i for i, p in enumerate(pos) if p is None]
    todo.sort(key=lambda i: -sizes[i][1])
    x = y = row_h = 0.0
    margin = 0.5
    for i in todo:
        w, h = sizes[i]
        if x + w > canvas_w and x > 0:
            x, y = 0.0, y + row_h + margin
            row_h = 0.0
        if y + h > canvas_h:
            y = 0.0  # wrap: overlaps are fine, the placer legalizes
        pos[i] = (min(x + w / 2, canvas_w - w / 2),
                  min(y + h / 2, canvas_h - h / 2))
        x += w + margin
        row_h = max(row_h, h)


# ── Net mapping ──────────────────────────────────────────────────────────────


def _pin_center_offset(design, comp, pin_name):
    """Pin offset from the component's center, honoring orientation."""
    master = design.masters[comp.master]
    pin = master.pins.get(pin_name)
    dx, dy = pin.offset if pin else (master.width / 2, master.height / 2)
    w, h = master.width, master.height
    o = comp.orient
    if o == "S":
        dx, dy = w - dx, h - dy
    elif o == "FN":
        dx = w - dx
    elif o == "FS":
        dy = h - dy
    elif o in ("E", "W", "FE", "FW"):
        rot = {"E": (dy, w - dx), "W": (h - dy, dx),
               "FE": (dy, dx), "FW": (h - dy, w - dx)}[o]
        dx, dy = rot
        w, h = h, w
    return dx - w / 2, dy - h / 2


def _pin_direction(design, comp, pin_name):
    pin = design.masters[comp.master].pins.get(pin_name)
    return pin.direction if pin else "INOUT"


@dataclass
class MappedNet:
    weight: float
    driver: Tuple[str, object]  # ("port", p) | ("hard", (i, pin)) | ("soft", m)
    sinks: List[Tuple[str, object]]


def _map_nets(design, hard_index, cell_cluster, port_index):
    """Resolve each Design net to driver/sink endpoints in placer space."""
    mapped = []
    for net in design.nets:
        ends = []  # (kind, key, is_output)
        seen = set()
        for inst, pin_name in net.terms:
            if inst == "PIN":
                if pin_name in port_index:
                    key = ("port", port_index[pin_name])
                    is_out = design.io_pins[pin_name].direction == "INPUT"
                else:
                    continue
            elif inst in hard_index:
                comp = design.components[inst]
                key = ("hard", (hard_index[inst], pin_name))
                is_out = _pin_direction(design, comp, pin_name) == "OUTPUT"
            elif inst in cell_cluster:
                key = ("soft", cell_cluster[inst])
                comp = design.components[inst]
                is_out = _pin_direction(design, comp, pin_name) == "OUTPUT"
            else:
                continue  # dropped component
            dedup = key if key[0] != "hard" else ("hard", key[1][0])
            if dedup in seen:
                continue
            seen.add(dedup)
            ends.append((key, is_out))
        if len(ends) < 2:
            continue
        driver = next((k for k, out in ends if out), ends[0][0])
        sinks = [k for k, _ in ends if k is not driver]
        mapped.append(MappedNet(net.weight, driver, sinks))
    return mapped


# ── ICCAD04 writers ──────────────────────────────────────────────────────────


def _node(name, inputs, attrs):
    lines = ["node {", f'  name: "{name}"']
    for s in inputs or []:
        lines.append(f'  input: "{s}"')
    for key, kind, val in attrs:
        v = f"{val:.4f}" if kind == "f" else f'"{val}"'
        lines += ["  attr {", f'    key: "{key}"', "    value {",
                  f"      {'f' if kind == 'f' else 'placeholder'}: {v}",
                  "    }", "  }"]
    lines.append("}")
    return "\n".join(lines)


def _sink_name(kind, val):
    if kind == "port":
        return f"p{val}"
    if kind == "soft":
        return f"Grp_{val}/Pinput"
    i, pin_name = val
    return f"a{i}/IP__{pin_name}"


def _write_pb(path, design, canvas_w, canvas_h, hard_names, hard_sizes,
              hard_pos, soft_sizes, soft_seed, ports, port_pos, nets, comps):
    port_inputs = {}
    hard_pin_inputs = {}
    soft_outputs = {}
    hard_pins_used = {}  # hard idx -> {pin_name}
    for net in nets:
        sink_names = [_sink_name(k, v) for k, v in net.sinks]
        for kind, val in [net.driver] + net.sinks:
            if kind == "hard":
                hard_pins_used.setdefault(val[0], set()).add(val[1])
        kind, val = net.driver
        if kind == "port":
            port_inputs.setdefault(val, []).extend(sink_names)
        elif kind == "soft":
            soft_outputs.setdefault(val, []).append(sink_names)
        else:
            hard_pin_inputs.setdefault(val, []).extend(sink_names)

    blocks = [_node("__metadata__", None,
                    [("soft_macro_area_bloating_ratio", "f", 1.0)])]
    for p, pname in enumerate(ports):
        x, y = port_pos[p]
        blocks.append(_node(
            f"p{p}", port_inputs.get(p),
            [("side", "placeholder", _side_of(x, y, canvas_w, canvas_h)),
             ("type", "placeholder", "PORT"), ("x", "f", x), ("y", "f", y)]))
    for i, name in enumerate(hard_names):
        w, h = hard_sizes[i]
        cx, cy = hard_pos[i]
        blocks.append(_node(
            f"a{i}", None,
            [("height", "f", h), ("orientation", "placeholder", "N"),
             ("type", "placeholder", "MACRO"),
             ("width", "f", w), ("x", "f", cx), ("y", "f", cy)]))
        comp = comps.get(name)
        if not hard_pins_used.get(i):
            # the plc parser requires every MACRO to own at least one pin
            blocks.append(_node(
                f"a{i}/IP__dummy", None,
                [("macro_name", "placeholder", f"a{i}"),
                 ("type", "placeholder", "MACRO_PIN"),
                 ("x", "f", cx), ("x_offset", "f", 0.0),
                 ("y", "f", cy), ("y_offset", "f", 0.0)]))
        for pin_name in sorted(hard_pins_used.get(i, ())):
            if comp is not None:
                ox, oy = _pin_center_offset(design, comp, pin_name)
            else:
                ox, oy = 0.0, 0.0  # blockage dummy
            blocks.append(_node(
                f"a{i}/IP__{pin_name}",
                hard_pin_inputs.get((i, pin_name)),
                [("macro_name", "placeholder", f"a{i}"),
                 ("type", "placeholder", "MACRO_PIN"),
                 ("x", "f", cx + ox), ("x_offset", "f", ox),
                 ("y", "f", cy + oy), ("y_offset", "f", oy)]))
    for m, (w, h) in enumerate(soft_sizes):
        cx, cy = soft_seed[m]
        blocks.append(_node(
            f"Grp_{m}", None,
            [("height", "f", h), ("type", "placeholder", "macro"),
             ("width", "f", w), ("x", "f", cx), ("y", "f", cy)]))
        pin_attrs = lambda: [  # noqa: E731 - soft pins sit at the macro center
            ("macro_name", "placeholder", f"Grp_{m}"),
            ("type", "placeholder", "macro_pin"),
            ("x", "f", cx), ("x_offset", "f", 0.0),
            ("y", "f", cy), ("y_offset", "f", 0.0)]
        blocks.append(_node(f"Grp_{m}/Pinput", None, pin_attrs()))
        for j, sink_names in enumerate(soft_outputs.get(m, [])):
            blocks.append(_node(f"Grp_{m}/Poutput_{j}", sink_names, pin_attrs()))
    Path(path).write_text("\n".join(blocks) + "\n")


def _write_plc(path, design, canvas_w, canvas_h, grid_cols, grid_rows, plc,
               total_area):
    lines = [
        "# Placement file for Circuit Training",
        f"# Source : eda_io converter ({design.name})",
        f"# Columns : {grid_cols}  Rows : {grid_rows}",
        f"# Width : {canvas_w:.3f}  Height : {canvas_h:.3f}",
        f"# Area : {total_area}",
        "# Project : eda_io",
        "# Block : unset_block",
        f"# Routes per micron, hor : {design.hroutes_per_micron:.3f}"
        f"  ver : {design.vroutes_per_micron:.3f}",
        "# Routes used by macros, hor : 30.300  ver : 71.300",
        "# Smoothing factor : 2",
        "# Overlap threshold : 0.004",
        "#",
    ]
    for idx in plc.port_indices:
        x, y = plc.modules_w_pins[idx].get_pos()
        lines.append(f"{idx} {x:.4f} {y:.4f} - 0")
    for idx in plc.hard_macro_indices + plc.soft_macro_indices:
        node = plc.modules_w_pins[idx]
        x, y = node.get_pos()
        fix = 1 if node.get_fix_flag() else 0
        lines.append(f"{idx} {x:.4f} {y:.4f} N {fix}")
    Path(path).write_text("\n".join(lines) + "\n")


def _attach_net_tensors(benchmark, nets, num_hard):
    """Expose mapped connectivity + SDC/Liberty weights on the Benchmark."""
    import torch

    net_nodes, weights = [], []
    for net in nets:
        idxs = []
        for kind, val in [net.driver] + net.sinks:
            if kind == "hard":
                idxs.append(val[0])
            elif kind == "soft":
                idxs.append(num_hard + val)
        if len(idxs) >= 2:
            net_nodes.append(torch.tensor(sorted(set(idxs)), dtype=torch.long))
            weights.append(net.weight)
    benchmark.num_nets = len(net_nodes)
    benchmark.net_nodes = net_nodes
    benchmark.net_weights = (torch.tensor(weights, dtype=torch.float32)
                             if weights else torch.zeros(0))
