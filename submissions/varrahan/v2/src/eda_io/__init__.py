"""Plug-and-play EDA file I/O for the v2 macro placer.

Inputs (any combination - see read_design):
    LEF      cell geometry, class, pin offsets
    DEF      die area, components, I/O pins, blockages, nets
    Verilog  structural netlist (alternative / supplement to DEF NETS)
    SDC      timing constraints -> net weights
    Liberty  pin directions, capacitance -> weights + size fallback

Outputs:
    updated DEF, Tcl placement script (ICC2 / Innovus), QoR report.

Typical use:
    from eda_io import read_design, build_benchmark
    design = read_design(lef=["tech.lef", "macros.lef"], def_file="floor.def",
                         verilog="top.v", sdc="top.sdc", liberty=["cells.lib"])
    result = build_benchmark(design)
    placement = MacroPlacer().place(result.benchmark)
"""

from .build import BuildResult, build_benchmark
from .def_io import parse_def, write_def
from .design import Component, Design, IOPin, Master, MasterPin, Net
from .lef import parse_lef, parse_lef_files
from .liberty import apply_liberty_weights, parse_liberty
from .report import write_report
from .sdc import parse_sdc
from .tcl_out import write_tcl
from .verilog import parse_verilog


def read_design(
    lef=None,
    def_file=None,
    verilog=None,
    sdc=None,
    liberty=None,
    top=None,
    name=None,
) -> Design:
    """Merge any combination of EDA inputs into one Design.

    Minimum useful input: (LEF or Liberty) for geometry plus (DEF or Verilog)
    for instances. Each extra file refines the picture: DEF adds locations
    and die area, Verilog adds connectivity, SDC and Liberty add net weights.
    """
    if not (lef or liberty):
        raise ValueError("need at least one LEF or Liberty file for cell geometry")
    if not (def_file or verilog):
        raise ValueError("need a DEF or a Verilog netlist for instances")

    design = Design()
    if lef:
        design.masters = parse_lef_files(
            [lef] if isinstance(lef, (str, bytes)) else lef)
    for lib in ([liberty] if isinstance(liberty, (str, bytes)) else liberty or []):
        parse_liberty(lib, design.masters)
    if def_file:
        parse_def(def_file, design)
    if verilog:
        parse_verilog(verilog, design, top=top)
    if sdc:
        for f in [sdc] if isinstance(sdc, (str, bytes)) else sdc:
            parse_sdc(f, design)
    if liberty:
        apply_liberty_weights(design)
    if name:
        design.name = name
    return design
