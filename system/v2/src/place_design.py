"""Plug-and-play macro placement from standard EDA files.

Reads any workable combination of LEF / DEF / Verilog / SDC / Liberty (or an
ICCAD04 netlist.pb.txt directly), runs the v2 macro placer, and writes any
combination of updated DEF, Tcl placement script, and QoR report.

Usage:
    # canonical flow: LEF + DEF in, everything out
    uv run python system/v2/src/place_design.py \
        --lef tech.lef --lef macros.lef --def floorplan.def \
        --out-def placed.def --out-tcl place_macros.tcl --report qor.rpt

    # netlist from Verilog, timing weights from SDC + Liberty
    ... --lef cells.lef --verilog top.v --sdc top.sdc --lib cells.lib \
        --out-def placed.def

    # no DEF at all (die area derived from utilization), Innovus script out
    ... --lef cells.lef --verilog top.v --out-tcl place.tcl --tcl-dialect innovus

    # challenge-format passthrough
    ... --netlist-pb path/netlist.pb.txt --plc path/initial.plc --report qor.rpt
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group("inputs (mix freely)")
    src.add_argument("--lef", action="append", default=[],
                     help="LEF file (repeatable; tech + cell LEFs)")
    src.add_argument("--def", dest="def_file", help="initial DEF floorplan")
    src.add_argument("--verilog", help="structural Verilog netlist")
    src.add_argument("--sdc", action="append", default=[],
                     help="SDC constraints (repeatable)")
    src.add_argument("--lib", action="append", default=[],
                     help="Liberty file (repeatable)")
    src.add_argument("--netlist-pb", help="ICCAD04 netlist.pb.txt (challenge format)")
    src.add_argument("--plc", help="initial.plc to pair with --netlist-pb")
    src.add_argument("--top", help="top module name when the Verilog has several")
    src.add_argument("--design-name", help="override the design name")

    out = parser.add_argument_group("outputs (pick any)")
    out.add_argument("--out-def", help="write the updated DEF here")
    out.add_argument("--out-tcl", help="write a placement Tcl script here")
    out.add_argument("--tcl-dialect", choices=["icc2", "innovus"], default="icc2")
    out.add_argument("--report", help="write the QoR .rpt here")
    out.add_argument("--vis", help="write a placement visualization .png here")

    parser.add_argument("--budget", type=float, default=150.0,
                        help="placer time budget in seconds (default 150)")
    parser.add_argument("--workdir",
                        help="keep converted ICCAD04 files here (default: temp)")
    args = parser.parse_args()

    if not any([args.out_def, args.out_tcl, args.report, args.vis]):
        parser.error("pick at least one output: --out-def / --out-tcl / "
                     "--report / --vis")
    if args.netlist_pb and (args.lef or args.def_file or args.verilog):
        parser.error("--netlist-pb is exclusive with LEF/DEF/Verilog inputs")
    if args.out_def and args.netlist_pb:
        parser.error("--out-def needs LEF/DEF/Verilog inputs (no component "
                     "names exist in the challenge format)")

    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    t0 = time.time()
    result = None
    if args.netlist_pb:
        from macro_place.loader import load_benchmark
        benchmark, plc = load_benchmark(args.netlist_pb, args.plc)
        benchmark._cached_plc = plc
        print(f"loaded {benchmark}")
    else:
        import eda_io
        design = eda_io.read_design(
            lef=args.lef or None, def_file=args.def_file, verilog=args.verilog,
            sdc=args.sdc or None, liberty=args.lib or None, top=args.top,
            name=args.design_name)
        print(f"design '{design.name}': {len(design.components)} components, "
              f"{len(design.io_pins)} I/O pins, {len(design.nets)} nets")
        result = eda_io.build_benchmark(design, workdir=args.workdir)
        benchmark, plc = result.benchmark, result.plc
        if result.dropped:
            print(f"warning: dropped {len(result.dropped)} components with no "
                  f"master geometry (first: {result.dropped[:3]})")
        print(f"built {benchmark}")
        print(f"converted ICCAD04 files in {result.workdir}")

    from main import MacroPlacer
    placer = MacroPlacer()
    if hasattr(placer, "time_budget_s"):
        placer.time_budget_s = args.budget
    t = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - t

    valid, violations = validate_placement(placement, benchmark)
    initial_costs = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
    final_costs = compute_proxy_cost(placement, benchmark, plc)
    print(f"placed in {runtime:.0f}s: proxy "
          f"{float(initial_costs['proxy_cost']):.4f} -> "
          f"{float(final_costs['proxy_cost']):.4f}, "
          f"{'VALID' if valid else f'INVALID {violations[:2]}'}")

    outputs = {}
    if result is not None:
        placements = result.placements_um(placement)
        if args.out_def:
            from eda_io import write_def
            write_def(result.design, placements, args.out_def,
                      template_path=args.def_file)
            outputs["def"] = args.out_def
        if args.out_tcl:
            from eda_io import write_tcl
            macro_only = {n: placements[n] for n in result.hard_names
                          if n in placements}
            write_tcl(macro_only, result.design, args.out_tcl,
                      dialect=args.tcl_dialect)
            outputs["tcl"] = args.out_tcl
    elif args.out_tcl:
        parser.error("--out-tcl needs LEF/DEF/Verilog inputs")

    if args.vis:
        from macro_place.utils import visualize_placement
        visualize_placement(placement, benchmark, save_path=args.vis, plc=plc)
        outputs["vis"] = args.vis

    if args.report:
        if result is not None:
            from eda_io.report import write_report
            inputs = {k: v for k, v in [
                ("lef", ", ".join(args.lef)), ("def", args.def_file),
                ("verilog", args.verilog), ("sdc", ", ".join(args.sdc)),
                ("lib", ", ".join(args.lib))] if v}
            write_report(args.report, result, placement, runtime,
                         initial_costs=initial_costs, final_costs=final_costs,
                         valid=valid, violations=violations,
                         inputs=inputs, outputs=outputs)
        else:
            _challenge_report(args.report, benchmark, placement, runtime,
                              initial_costs, final_costs, valid, violations)
        outputs["report"] = args.report

    for kind, path in outputs.items():
        print(f"  {kind:>6} -> {path}")
    print(f"total {time.time() - t0:.0f}s")


def _challenge_report(path, benchmark, placement, runtime, initial_costs,
                      final_costs, valid, violations):
    """Minimal QoR report for the --netlist-pb passthrough path."""
    lines = [
        f"design: {benchmark.name}",
        f"hard macros: {benchmark.num_hard_macros}  "
        f"soft macros: {benchmark.num_soft_macros}",
        f"proxy initial: {float(initial_costs['proxy_cost']):.4f}",
        f"proxy placed : {float(final_costs['proxy_cost']):.4f}",
        f"valid: {valid}" + ("" if valid else f"  violations: {violations[:5]}"),
        f"runtime: {runtime:.1f}s",
        "",
    ]
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
