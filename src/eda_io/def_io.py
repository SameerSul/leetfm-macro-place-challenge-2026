"""DEF reader + writer.

Reader extracts the placement-relevant subset: UNITS, DIEAREA, COMPONENTS
(with status/orientation), PINS (I/O locations), placement BLOCKAGES, and
NETS connectivity. Routing layers, special nets, vias, rows, and tracks are
skipped without error.

Writer produces the updated DEF: same design, with every component's x/y
filled in and flagged PLACED (or FIXED if it was fixed on input). When an
input DEF exists its text is patched line-for-line so unrelated sections
survive byte-identical; otherwise a minimal valid DEF is generated.
"""

import re
from pathlib import Path
from typing import Dict, Optional

from .design import Component, Design, IOPin, Net

_TOKEN = re.compile(r"[()]|[^\s()]+")


def _tokens(text: str):
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        yield from _TOKEN.findall(line)


def parse_def(path, design: Optional[Design] = None) -> Design:
    """Parse a DEF file into (or onto) a Design. Coordinates become microns."""
    design = design or Design()
    toks = list(_tokens(Path(path).read_text()))
    i, n = 0, len(toks)
    dbu = design.dbu_per_micron

    def to_um(v):
        return float(v) / dbu

    while i < n:
        t = toks[i].upper()
        if t == "DESIGN" and i + 2 < n and toks[i + 2] == ";":
            design.name = toks[i + 1]
            i += 3
        elif t == "UNITS":
            dbu = int(toks[i + 3])  # UNITS DISTANCE MICRONS <dbu> ;
            design.dbu_per_micron = dbu
            i += 5
        elif t == "DIEAREA":
            nums = []
            j = i + 1
            while toks[j] != ";":
                if toks[j] not in "()":
                    nums.append(to_um(toks[j]))
                j += 1
            # Two points = the usual rectangle; polygons use their bbox
            xs, ys = nums[0::2], nums[1::2]
            design.die_area = (min(xs), min(ys), max(xs), max(ys))
            i = j + 1
        elif t == "COMPONENTS":
            i = _parse_components(toks, i, design, to_um)
        elif t == "PINS":
            i = _parse_pins(toks, i, design, to_um)
        elif t == "BLOCKAGES":
            i = _parse_blockages(toks, i, design, to_um)
        elif t == "NETS":
            i = _parse_nets(toks, i, design)
        else:
            i += 1
    return design


def _skip_to_end(toks, i, section):
    while not (toks[i].upper() == "END" and toks[i + 1].upper() == section):
        i += 1
    return i + 2


def _parse_components(toks, i, design, to_um):
    i += 2  # COMPONENTS <count> ;
    if toks[i] == ";":
        i += 1
    while toks[i] == "-":
        comp = Component(name=toks[i + 1], master=toks[i + 2])
        i += 3
        while toks[i] != ";":
            t = toks[i].upper()
            if t in ("PLACED", "FIXED", "COVER"):
                comp.status = "FIXED" if t in ("FIXED", "COVER") else "PLACED"
                # + PLACED ( x y ) orient
                comp.pos = (to_um(toks[i + 2]), to_um(toks[i + 3]))
                comp.orient = toks[i + 5]
                i += 6
            elif t == "UNPLACED":
                comp.status = "UNPLACED"
                i += 1
            else:
                i += 1
        i += 1
        design.components[comp.name] = comp
    return _skip_to_end(toks, i, "COMPONENTS")


def _parse_pins(toks, i, design, to_um):
    i += 2
    if toks[i] == ";":
        i += 1
    while toks[i] == "-":
        pin = IOPin(name=toks[i + 1])
        i += 2
        while toks[i] != ";":
            t = toks[i].upper()
            if t == "DIRECTION":
                pin.direction = toks[i + 1].upper()
                i += 2
            elif t in ("PLACED", "FIXED"):
                pin.pos = (to_um(toks[i + 2]), to_um(toks[i + 3]))
                i += 6
            else:
                i += 1
        i += 1
        design.io_pins[pin.name] = pin
    return _skip_to_end(toks, i, "PINS")


def _parse_blockages(toks, i, design, to_um):
    i += 2
    if toks[i] == ";":
        i += 1
    while toks[i] == "-":
        i += 1
        is_placement = toks[i].upper() == "PLACEMENT"
        while toks[i] != ";":
            if toks[i].upper() == "RECT" and is_placement:
                x0, y0 = to_um(toks[i + 2]), to_um(toks[i + 3])
                x1, y1 = to_um(toks[i + 6]), to_um(toks[i + 7])
                design.blockages.append((x0, y0, x1, y1))
                i += 9
            else:
                i += 1
        i += 1
    return _skip_to_end(toks, i, "BLOCKAGES")


def _parse_nets(toks, i, design):
    i += 2
    if toks[i] == ";":
        i += 1
    while toks[i] == "-":
        net = Net(name=toks[i + 1])
        i += 2
        while toks[i] != ";":
            if toks[i] == "(":
                inst, pin = toks[i + 1], toks[i + 2]
                net.terms.append((inst, pin))  # inst == "PIN" marks an I/O
                i += 4
            elif toks[i] == "+" and toks[i + 1].upper() == "WEIGHT":
                net.weight = float(toks[i + 2])
                i += 3
            else:
                i += 1
        i += 1
        design.nets.append(net)
    return _skip_to_end(toks, i, "NETS")


# ── Writer ───────────────────────────────────────────────────────────────────


_COMP_LINE = re.compile(r"^(\s*-\s+(\S+)\s+\S+)(.*)$")


def write_def(
    design: Design,
    placements: Dict[str, tuple],
    out_path,
    template_path=None,
):
    """Write the updated DEF.

    placements maps component name -> (x_ll, y_ll) lower-left microns.
    With template_path the original file is patched (COMPONENTS entries get
    their new location, everything else preserved); without it a minimal
    valid DEF is generated from the Design.
    """
    if template_path:
        text = _patch_def(Path(template_path).read_text(), design, placements)
    else:
        text = _fresh_def(design, placements)
    Path(out_path).write_text(text)


def _placement_clause(design, name, placements):
    comp = design.components[name]
    x, y = placements.get(name, comp.pos or (0.0, 0.0))
    status = "FIXED" if comp.status == "FIXED" else "PLACED"
    dbu = design.dbu_per_micron
    return f" + {status} ( {round(x * dbu)} {round(y * dbu)} ) {comp.orient}"


_STATUS_CLAUSE = re.compile(
    r"\+\s+(?:PLACED|FIXED|UNPLACED|COVER)"
    r"(?:\s+\(\s*-?\d+\s+-?\d+\s*\)\s+\S+)?")


def _patch_def(text, design, placements):
    out, in_components, entry = [], False, []
    for raw in text.splitlines():
        stripped = raw.strip()
        upper = stripped.upper()
        if not in_components and upper.startswith("COMPONENTS"):
            in_components = True
        elif in_components and upper.startswith("END COMPONENTS"):
            in_components = False
        elif in_components and (entry or stripped.startswith("-")):
            # Entries may span lines; join until the terminating ';'
            entry.append(stripped)
            if not stripped.endswith(";"):
                continue
            joined = " ".join(entry)
            entry = []
            m = _COMP_LINE.match(joined)
            if m and m.group(2) in design.components:
                rest = _STATUS_CLAUSE.sub("", m.group(3))
                rest = rest.rstrip().rstrip(";").rstrip()
                joined = (m.group(1) + (" " + rest if rest else "")
                          + _placement_clause(design, m.group(2), placements)
                          + " ;")
            out.append(joined)
            continue
        out.append(raw)
    return "\n".join(out) + "\n"


def _fresh_def(design, placements):
    dbu = design.dbu_per_micron
    x0, y0, x1, y1 = design.die_area or (0, 0, 0, 0)
    lines = [
        "VERSION 5.8 ;",
        'DIVIDERCHAR "/" ;',
        'BUSBITCHARS "[]" ;',
        f"DESIGN {design.name} ;",
        f"UNITS DISTANCE MICRONS {dbu} ;",
        f"DIEAREA ( {round(x0 * dbu)} {round(y0 * dbu)} )"
        f" ( {round(x1 * dbu)} {round(y1 * dbu)} ) ;",
        "",
        f"COMPONENTS {len(design.components)} ;",
    ]
    for name, comp in design.components.items():
        lines.append(f"- {name} {comp.master}"
                     f"{_placement_clause(design, name, placements)} ;")
    lines.append("END COMPONENTS")
    if design.io_pins:
        lines.append("")
        lines.append(f"PINS {len(design.io_pins)} ;")
        for pin in design.io_pins.values():
            entry = f"- {pin.name} + NET {pin.name} + DIRECTION {pin.direction}"
            if pin.pos:
                px, py = round(pin.pos[0] * dbu), round(pin.pos[1] * dbu)
                entry += f" + PLACED ( {px} {py} ) N"
            lines.append(entry + " ;")
        lines.append("END PINS")
    lines += ["", "END DESIGN", ""]
    return "\n".join(lines)
