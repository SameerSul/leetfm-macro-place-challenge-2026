"""Liberty (.lib) reader: cell footprints, pin directions, capacitances.

Liberty complements LEF: it has no geometry but knows electrical facts.
Used to (a) fill pin directions when no LEF pin info exists, (b) provide
cell area as a size fallback when a cell is missing from LEF entirely, and
(c) weight nets by total sink capacitance (heavily loaded nets are pulled
tighter). Timing tables and power groups are skipped.
"""

import re
from pathlib import Path
from typing import Dict

from .design import Design, Master, MasterPin

_CELL = re.compile(r"\bcell\s*\(\s*\"?([\w\[\]\.]+)\"?\s*\)")
_PIN = re.compile(r"\bpin\s*\(\s*\"?([\w\[\]\.]+)\"?\s*\)")
_ATTR = re.compile(r"\b(\w+)\s*:\s*\"?([^;\"]+)\"?\s*;")


def parse_liberty(path, masters: Dict[str, Master] = None) -> Dict[str, Master]:
    """Parse one Liberty file into (or onto) a master dict.

    Brace-depth walk: track the current cell() and pin() scopes and pick up
    `direction`, `capacitance`, and `area` attributes inside them.
    """
    masters = masters if masters is not None else {}
    text = re.sub(r"/\*.*?\*/", " ", Path(path).read_text(), flags=re.S)

    cell = None
    pin = None
    cell_depth = pin_depth = 0
    depth = 0
    for line in text.splitlines():
        line = line.split("//", 1)[0]
        mcell = _CELL.search(line)
        mpin = _PIN.search(line)
        attr_text = line
        if mcell and "{" in line:
            name = mcell.group(1)
            cell = masters.get(name) or Master(name=name)
            masters[name] = cell
            cell_depth = depth
            attr_text = line[line.index("{") + 1:]
        elif mpin and "{" in line and cell is not None:
            name = mpin.group(1)
            pin = cell.pins.get(name) or MasterPin(name=name)
            cell.pins[name] = pin
            pin_depth = depth
            attr_text = line[line.index("{") + 1:]
        for key, val in _ATTR.findall(attr_text):
            val = val.strip()
            if pin is not None:
                if key == "direction":
                    pin.direction = val.upper()
                elif key == "capacitance":
                    pin.capacitance = float(val)
            elif cell is not None and key == "area" and cell.width == 0:
                # No geometry in Liberty: assume a square of that area
                side = float(val) ** 0.5
                cell.width = cell.height = side
        depth += line.count("{") - line.count("}")
        if pin is not None and depth <= pin_depth:
            pin = None
        if cell is not None and depth <= cell_depth:
            cell = None
    return masters


def apply_liberty_weights(design: Design, cap_scale: float = 1.0):
    """Scale each net's weight by its normalized total sink capacitance.

    weight *= 1 + cap_scale * (net_sink_cap / mean_sink_cap - 1), clamped to
    [0.25, 4.0]; nets with no known caps are untouched.
    """
    caps = []
    for net in design.nets:
        total = 0.0
        for inst, pin_name in net.terms:
            if inst == "PIN":
                continue
            comp = design.components.get(inst)
            master = design.masters.get(comp.master) if comp else None
            mp = master.pins.get(pin_name) if master else None
            if mp is not None and mp.direction == "INPUT":
                total += mp.capacitance
        caps.append(total)
    known = [c for c in caps if c > 0]
    if not known:
        return
    mean = sum(known) / len(known)
    for net, cap in zip(design.nets, caps):
        if cap > 0 and net.weight > 0:
            factor = 1.0 + cap_scale * (cap / mean - 1.0)
            net.weight = float(min(4.0, max(0.25, net.weight * factor)))
