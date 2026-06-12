"""Structural (gate-level) Verilog netlist reader.

Extracts module ports, instances, and net connectivity from a flat
structural netlist - the form synthesis tools emit. Behavioral constructs
(always blocks, assigns with expressions) are out of scope; plain
`assign a = b;` aliases are honored. Named (.pin(net)) and positional
port connections are both supported; positional connections fall back to
synthetic pin names since pin identity does not affect placement.
"""

import re
from pathlib import Path

from .design import Component, Design, IOPin, Net

_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_MODULE = re.compile(r"\bmodule\s+(\S+?)\s*(?:\(.*?\))?\s*;(.*?)\bendmodule",
                     re.S)
_PORT_DECL = re.compile(r"\b(input|output|inout)\b([^;]*);")
_STATEMENT = re.compile(r"([^;]+);")
_NAMED_CONN = re.compile(r"\.\s*([A-Za-z_\\][\w\[\]\\.]*)\s*\(\s*([^()]*?)\s*\)")
_IDENT = r"[A-Za-z_\\][\w$\[\]\\.]*"
_INSTANCE = re.compile(
    rf"^\s*({_IDENT})\s+({_IDENT})\s*\((.*)\)\s*$", re.S)

_KEYWORDS = {"input", "output", "inout", "wire", "tri", "reg", "supply0",
             "supply1", "assign", "parameter", "localparam", "specify",
             "module", "endmodule", "timescale"}


def _expand_names(decl: str):
    """Names from a port/wire declaration, ignoring ranges like [7:0]."""
    decl = re.sub(r"\[[^\]]*\]", "", decl)
    return [t.strip() for t in decl.split(",") if t.strip()]


def parse_verilog(path, design: Design = None, top: str = None) -> Design:
    """Parse a structural netlist into (or onto) a Design.

    Instances become Components (master = module/cell name); module ports
    become IOPins; each distinct net becomes a Net joining the pins it
    touches. When the file holds several modules, `top` picks one (default:
    a module not instantiated by any other - the usual top detection).
    """
    design = design or Design()
    text = _COMMENTS.sub(" ", Path(path).read_text())
    modules = {m.group(1): m.group(2) for m in _MODULE.finditer(text)}
    if not modules:
        raise ValueError(f"no modules found in {path}")

    if top is None:
        instantiated = set()
        for body in modules.values():
            for st in _STATEMENT.finditer(body):
                m = _INSTANCE.match(st.group(1).strip())
                if m:
                    instantiated.add(m.group(1))
        candidates = [name for name in modules if name not in instantiated]
        top = candidates[0] if candidates else next(iter(modules))
    body = modules[top]
    if design.name == "design":
        design.name = top

    nets = {}  # net name -> list of (component|"PIN", pin name)

    def touch(net_name, inst, pin):
        net_name = net_name.strip()
        if not net_name or net_name.startswith(("1'b", "1'h")):
            return  # unconnected or tied-off
        nets.setdefault(net_name, []).append((inst, pin))

    for direction, decl in _PORT_DECL.findall(body):
        for port in _expand_names(decl):
            design.io_pins.setdefault(
                port, IOPin(name=port, direction=direction.upper()))
            touch(port, "PIN", port)

    for st in _STATEMENT.finditer(body):
        stmt = st.group(1).strip()
        first = stmt.split(None, 1)[0] if stmt else ""
        if not stmt or first in _KEYWORDS or first.startswith(("input", "output")):
            if first == "assign":  # alias: merge both sides into one net
                m = re.match(r"assign\s+(\S+)\s*=\s*(\S+)", stmt)
                if m and m.group(2) in nets:
                    nets.setdefault(m.group(1), []).extend(nets[m.group(2)])
            continue
        m = _INSTANCE.match(stmt)
        if not m:
            continue
        master, inst, conns = m.group(1), m.group(2), m.group(3)
        design.components.setdefault(inst, Component(name=inst, master=master))
        named = _NAMED_CONN.findall(conns)
        if named:
            for pin, net_name in named:
                touch(net_name, inst, pin)
        else:
            for k, net_name in enumerate(_expand_names(conns)):
                touch(net_name, inst, f"_p{k}")

    existing = {n.name for n in design.nets}
    for net_name, terms in nets.items():
        if len(terms) >= 2 and net_name not in existing:
            design.nets.append(Net(name=net_name, terms=terms))
    return design
