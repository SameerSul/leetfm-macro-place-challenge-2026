# eda_io - plug-and-play EDA file I/O for the v2 placer

Makes the v2 macro placer usable in any physical-design flow: it accepts the
standard EDA input formats, runs the unchanged placer, and emits the standard
output formats the rest of a toolchain expects. No challenge-specific files
are needed.

## How it works

```
LEF ─┐
DEF ─┤                       ┌─> updated DEF       (def_io.write_def)
 .v ─┼─> Design ─> ICCAD04 ──┼─> Tcl script        (tcl_out.write_tcl)
SDC ─┤   (merge)   pb + plc  ├─> QoR report .rpt   (report.write_report)
.lib ┘                │      └─> visualization .png
                      v
            load_benchmark -> MacroPlacer.place()
```

Rather than teaching the placer a second native format, every input
combination is merged into one neutral `Design` (design.py) and converted to
the ICCAD04 `netlist.pb.txt` + `initial.plc` pair the whole stack already
understands (build.py). The standard loader and the exact TILOS scorer then
work unchanged - external designs are scored and placed exactly like the
challenge benchmarks.

## Supported inputs (mix freely)

| format | what is used | required? |
|---|---|---|
| LEF (repeatable) | cell SIZE, CLASS (BLOCK = hard macro), pin offsets + directions | geometry source (this or Liberty) |
| DEF | DIEAREA, UNITS, COMPONENTS (location/orientation/FIXED), PINS, placement BLOCKAGES, NETS | instance source (this or Verilog) |
| Verilog (.v, structural) | instances, module ports, net connectivity | instance source (this or DEF) |
| SDC | create_clock, set_max_delay/min_delay/multicycle, set_false_path -> net weights | optional |
| Liberty (.lib, repeatable) | pin directions + capacitance (net weighting), cell area (geometry fallback) | optional |
| netlist.pb.txt (+ .plc) | challenge-format passthrough | alternative to all of the above |

Minimum viable combos:
- **LEF + DEF** - the canonical floorplanning handoff (nets from DEF NETS)
- **LEF + Verilog** - no floorplan yet: die area is derived from total cell
  area at 80% utilization, seeds are shelf-packed, I/O pins are spread along
  the perimeter
- **Liberty + Verilog** - no physical library at all: cell geometry falls
  back to squares of the Liberty `area` attribute

Everything else refines the picture: DEF adds locations + die area, Verilog
adds/overrides connectivity, SDC raises weights on timing-critical nets
(clock nets drop to 0, false paths to 0.25, constrained paths to 2.0),
Liberty scales weights by sink capacitance.

## What the conversion does

- **Hard macros**: components whose LEF CLASS is BLOCK. If no BLOCK masters
  exist anywhere, components >10x the median area are treated as macros.
- **Soft macros**: standard cells are clustered (~50 cells each) - by
  location when the DEF has them placed, by net connectivity when not. The
  cluster becomes a square soft macro of the summed cell area.
- **Fixed stays fixed**: DEF `FIXED` components keep their location; the
  placer never moves them.
- **Blockages**: DEF placement blockages become fixed dummy macros, so the
  placer treats keep-out zones as occupied.
- **Die origin**: DEF coordinates are absolute; everything is shifted so the
  placer canvas starts at (0,0) and shifted back on output.
- **Missing placements**: unplaced macros/clusters get a shelf-packed seed
  (the placer expects a seed; overlaps are legalized).
- **Components without geometry** (master in no LEF/Liberty) are dropped
  with a warning and keep their input location in the output DEF.

## Outputs (pick any)

- **Updated DEF** - the input DEF patched in place: every component gets its
  ` + PLACED ( x y ) orient` (or `FIXED` if it was fixed), all other sections
  byte-identical. Without an input DEF a minimal valid DEF is generated.
  Std cells translate along with their cluster.
- **Tcl script** - `set_cell_location` (ICC2, default) or `placeInstance`
  (Innovus, `--tcl-dialect innovus`), plus fixing the placed macros.
- **QoR report (.rpt)** - design stats, HPWL before/after, hard overlaps
  resolved, evaluator validity, exact proxy-cost breakdown
  (wirelength/density/congestion), runtime.
- **Visualization (.png)** - the standard 3-panel placement/density/
  congestion figure.

## Usage

```bash
# canonical: LEF + DEF in, everything out
uv run python submissions/varrahan/v2/src/place_design.py \
    --lef tech.lef --lef macros.lef --def floorplan.def \
    --out-def placed.def --out-tcl place_macros.tcl --report qor.rpt

# Verilog netlist + timing weights, Innovus script out
uv run python .../place_design.py --lef cells.lef --verilog top.v \
    --sdc top.sdc --lib cells.lib --out-tcl place.tcl --tcl-dialect innovus

# no DEF at all (die area derived), report + picture
uv run python .../place_design.py --lef cells.lef --verilog top.v \
    --report qor.rpt --vis placed.png

# challenge-format passthrough
uv run python .../place_design.py --netlist-pb dir/netlist.pb.txt \
    --plc dir/initial.plc --report qor.rpt
```

Useful flags: `--budget <s>` placer time budget (default 150),
`--workdir <dir>` keeps the converted ICCAD04 files for inspection,
`--top <module>` picks the top module in a multi-module Verilog file,
`--design-name` overrides the design name.

Python API:

```python
from eda_io import read_design, build_benchmark, write_def, write_tcl
design = read_design(lef=["cells.lef"], def_file="floor.def", sdc=["t.sdc"])
result = build_benchmark(design)
placement = MacroPlacer().place(result.benchmark)   # exact scoring works:
                                                    # result.plc is attached
write_def(design, result.placements_um(placement), "out.def",
          template_path="floor.def")
```

## Parsing scope (deliberate)

The parsers extract the placement-relevant subset and skip everything else
without error (LEF layers/vias/obstructions, DEF special nets/rows/tracks,
Liberty timing tables, behavioral Verilog). `assign a = b;` aliases are
honored; expressions are not. This keeps the layer dependency-free and
vendor-agnostic - any tool's LEF/DEF loads.

## Files

- `design.py` - neutral `Design` dataclasses all parsers fill
- `lef.py`, `def_io.py`, `verilog.py`, `sdc.py`, `liberty.py` - readers
  (def_io.py also has the DEF writer)
- `build.py` - Design -> ICCAD04 conversion + Benchmark assembly
- `tcl_out.py`, `report.py` - Tcl + QoR writers
- `../place_design.py` - the CLI tying it together

Tests + fixture design (LEF/DEF/Verilog/SDC/Liberty for a 4-macro,
13-cell `chiptop`): `submissions/varrahan/v2/test/eda_io/`.
