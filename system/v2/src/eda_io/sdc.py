"""SDC reader: turns timing constraints into net weights.

The placer optimizes weighted wirelength, so SDC constraints map naturally:
nets on constrained paths get heavier weights (pulled tighter), clock nets
get zero weight (they are routed as trees by CTS, not placement-critical),
and false paths get their weight reduced. This is intentionally coarse -
placement-stage timing only needs to know WHICH nets matter, not exact slack.

Supported commands (others are ignored without error):
    create_clock -period P [get_ports clk] / -name ...
    set_false_path -from ... -to ...
    set_max_delay D -from ... -to ...
    set_multicycle_path N -from ... -to ...
"""

import re
import shlex
from pathlib import Path

from .design import Design

CRITICAL_WEIGHT = 2.0
FALSE_PATH_WEIGHT = 0.25
CLOCK_NET_WEIGHT = 0.0


def _object_names(arg: str):
    """Names inside [get_ports {a b}] / [get_pins x] / bare names."""
    inner = re.findall(r"\[\s*get_\w+\s+(?:\{([^}]*)\}|(\S+?))\s*\]", arg)
    if inner:
        names = []
        for braced, single in inner:
            names += (braced or single).split()
        return names
    return arg.replace("{", " ").replace("}", " ").split()


def parse_sdc(path, design: Design) -> Design:
    """Apply one SDC file's constraints to the design's net weights."""
    critical_endpoints = set()
    false_endpoints = set()

    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            toks = shlex.split(line.replace("[", " [").replace("]", "] "))
        except ValueError:
            continue
        if not toks:
            continue
        cmd = toks[0]
        joined = " ".join(toks[1:])

        if cmd == "create_clock":
            for name in _object_names(joined):
                if name not in ("-period", "-name", "-waveform") \
                        and not _is_number(name) and not name.startswith("-"):
                    design.clock_sources.append(name)
        elif cmd in ("set_max_delay", "set_min_delay", "set_multicycle_path"):
            critical_endpoints.update(_from_to_names(joined))
        elif cmd == "set_false_path":
            false_endpoints.update(_from_to_names(joined))

    _weight_nets(design, critical_endpoints, false_endpoints)
    return design


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _from_to_names(arg: str):
    names = set()
    for m in re.finditer(r"-(?:from|to|through)\s+((?:\[[^\]]*\]|\S)+)", arg):
        names.update(_object_names(m.group(1)))
    return names


def _weight_nets(design, critical, false_paths):
    """A net's weight follows the strongest constraint touching its terms.

    Term matching is by component name, I/O pin name, or "inst/pin" path.
    """
    clock = set(design.clock_sources)
    for net in design.nets:
        keys = {net.name}
        for inst, pin in net.terms:
            keys.add(pin if inst == "PIN" else inst)
            keys.add(f"{inst}/{pin}")
        if keys & clock:
            net.weight = CLOCK_NET_WEIGHT
        elif keys & critical:
            net.weight = max(net.weight, CRITICAL_WEIGHT)
        elif keys & false_paths:
            net.weight = min(net.weight, FALSE_PATH_WEIGHT)
