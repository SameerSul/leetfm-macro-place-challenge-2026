"""Neutral in-memory design representation shared by all EDA readers/writers.

Every parser (LEF, DEF, Verilog, SDC, Liberty) fills part of a Design; the
builder turns a Design into the ICCAD04 files the placer stack understands.
All coordinates here are in microns, lower-left corner convention for
component placements (DEF style); the builder converts to centers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MasterPin:
    """A pin on a library cell: offset from the cell's lower-left corner."""

    name: str
    direction: str = "INOUT"  # INPUT / OUTPUT / INOUT
    offset: Tuple[float, float] = (0.0, 0.0)
    capacitance: float = 0.0  # from Liberty, 0 when unknown


@dataclass
class Master:
    """A library cell (from LEF MACRO or Liberty cell)."""

    name: str
    width: float = 0.0
    height: float = 0.0
    cell_class: str = "CORE"  # LEF CLASS: BLOCK = hard macro, CORE = std cell
    pins: Dict[str, MasterPin] = field(default_factory=dict)

    @property
    def is_block(self) -> bool:
        return self.cell_class.upper().startswith("BLOCK") or \
            self.cell_class.upper() == "RING"


@dataclass
class Component:
    """A placed instance (DEF COMPONENTS entry or Verilog instance)."""

    name: str
    master: str
    pos: Optional[Tuple[float, float]] = None  # lower-left corner, microns
    orient: str = "N"
    status: str = "UNPLACED"  # UNPLACED / PLACED / FIXED


@dataclass
class IOPin:
    """A top-level I/O pin (DEF PINS entry or Verilog module port)."""

    name: str
    direction: str = "INOUT"
    pos: Optional[Tuple[float, float]] = None  # microns


@dataclass
class Net:
    """A net: list of (component_name, pin_name) terms; "PIN" marks I/O pins."""

    name: str
    terms: List[Tuple[str, str]] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class Design:
    """Everything the builder needs, merged from whichever inputs were given."""

    name: str = "design"
    die_area: Optional[Tuple[float, float, float, float]] = None  # x0 y0 x1 y1
    dbu_per_micron: int = 1000
    masters: Dict[str, Master] = field(default_factory=dict)
    components: Dict[str, Component] = field(default_factory=dict)
    io_pins: Dict[str, IOPin] = field(default_factory=dict)
    nets: List[Net] = field(default_factory=list)
    blockages: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # Routing supply (tracks per micron); defaults match the ICCAD04 testcases
    hroutes_per_micron: float = 66.0
    vroutes_per_micron: float = 107.0
    # Timing context from SDC (port/pin names of clock sources, critical terms)
    clock_sources: List[str] = field(default_factory=list)

    def size_of(self, comp: Component) -> Tuple[float, float]:
        """Footprint of a component in microns, accounting for orientation."""
        m = self.masters[comp.master]
        if comp.orient in ("E", "W", "FE", "FW"):
            return m.height, m.width
        return m.width, m.height
