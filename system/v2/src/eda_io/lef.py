"""LEF reader: extracts cell dimensions, class, and pin offsets.

Only the placement-relevant subset is parsed (MACRO blocks: CLASS, SIZE,
PIN direction + a representative port location). Layers, obstructions, vias,
and routing rules are skipped without error, so any vendor LEF loads.
"""

import re
from pathlib import Path
from typing import Dict

from .design import Master, MasterPin

_TOKEN = re.compile(r"\S+")


def _tokens(text: str):
    """Statement tokenizer: LEF statements end with ';', comments start '#'."""
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        yield from _TOKEN.findall(line)


def parse_lef(path, masters: Dict[str, Master] = None) -> Dict[str, Master]:
    """Parse one LEF file, merging into an existing master dict if given."""
    masters = masters if masters is not None else {}
    toks = list(_tokens(Path(path).read_text()))
    i = 0
    n = len(toks)
    while i < n:
        if toks[i].upper() != "MACRO":
            i += 1
            continue
        name = toks[i + 1]
        master = masters.get(name) or Master(name=name)
        masters[name] = master
        i += 2
        # Walk the MACRO block until its matching "END <name>"
        while i < n:
            t = toks[i].upper()
            if t == "END" and i + 1 < n and toks[i + 1] == name:
                i += 2
                break
            if t == "CLASS":
                master.cell_class = toks[i + 1].upper()
                i += 2
            elif t == "SIZE":
                master.width = float(toks[i + 1])
                master.height = float(toks[i + 3])  # SIZE w BY h ;
                i += 4
            elif t == "PIN":
                pin_name = toks[i + 1]
                i += 2
                pin = master.pins.get(pin_name) or MasterPin(name=pin_name)
                master.pins[pin_name] = pin
                rect_seen = False
                while i < n:
                    pt = toks[i].upper()
                    if pt == "END" and i + 1 < n and toks[i + 1] == pin_name:
                        i += 2
                        break
                    if pt == "DIRECTION":
                        pin.direction = toks[i + 1].upper()
                        i += 2
                    elif pt == "RECT" and not rect_seen:
                        # First port RECT center = the pin's offset
                        x0, y0, x1, y1 = (float(v) for v in toks[i + 1:i + 5])
                        pin.offset = ((x0 + x1) / 2, (y0 + y1) / 2)
                        rect_seen = True
                        i += 5
                    else:
                        i += 1
            else:
                i += 1
    return masters


def parse_lef_files(paths) -> Dict[str, Master]:
    """Parse multiple LEF files (tech LEF first is fine; non-MACRO content is skipped)."""
    masters: Dict[str, Master] = {}
    for p in paths:
        parse_lef(p, masters)
    return masters
