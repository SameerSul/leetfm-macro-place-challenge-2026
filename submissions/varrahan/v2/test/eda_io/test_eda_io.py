"""Tests for the eda_io plug-and-play layer.

Covers every parser, the builder under different input combinations, all
three output writers (round-tripping the DEF through the parser again), and
one end-to-end place_design.py run.

Run:
    uv run pytest submissions/varrahan/v2/test/eda_io/ -v
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SRC = HERE.parents[1] / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eda_io import (  # noqa: E402
    build_benchmark, parse_def, parse_lef, parse_liberty,
    parse_verilog, read_design, write_def, write_tcl,
)

FIX = HERE / "fixtures"
LEF = FIX / "cells.lef"
DEF = FIX / "floorplan.def"
VLG = FIX / "chiptop.v"
SDC = FIX / "chiptop.sdc"
LIB = FIX / "cells.lib"


# ── Parsers ──────────────────────────────────────────────────────────────────


def test_lef_parser():
    masters = parse_lef(LEF)
    assert set(masters) == {"RAM16", "ROM32", "PLL", "INVX1", "NAND2"}
    ram = masters["RAM16"]
    assert ram.is_block and (ram.width, ram.height) == (8.0, 6.0)
    assert not masters["INVX1"].is_block
    assert ram.pins["Q"].direction == "OUTPUT"
    assert ram.pins["Q"].offset == pytest.approx((7.8, 3.0))  # RECT center


def test_def_parser():
    d = parse_def(DEF)
    assert d.name == "chiptop"
    assert d.dbu_per_micron == 2000
    assert d.die_area == (5.0, 5.0, 65.0, 65.0)
    assert len(d.components) == 17
    assert d.components["u_rom"].status == "FIXED"
    assert d.components["u_ram0"].pos == (15.0, 15.0)
    assert d.components["u_ram1"].orient == "FS"
    assert d.components["u_pll"].status == "UNPLACED"
    assert d.components["u11"].pos == (30.0, 30.0)  # multi-line entry
    assert len(d.io_pins) == 4
    assert d.io_pins["clk"].pos == (5.0, 35.0)
    assert d.blockages == [(55.0, 6.0, 63.0, 12.0)]
    assert len(d.nets) == 12
    n3 = next(n for n in d.nets if n.name == "n3")
    assert n3.weight == 3.0
    clk = next(n for n in d.nets if n.name == "clk")
    assert ("PIN", "clk") in clk.terms and ("u_pll", "REF") in clk.terms


def test_verilog_parser():
    d = parse_verilog(VLG)
    assert d.name == "chiptop"
    assert len(d.components) == 17
    assert d.components["u_ram0"].master == "RAM16"
    assert d.components["u_ram0"].status == "UNPLACED"
    assert set(d.io_pins) == {"clk", "rst", "in1", "out1"}
    names = {n.name for n in d.nets}
    assert {"pllout", "n0", "ramq0", "out1"} <= names
    pllout = next(n for n in d.nets if n.name == "pllout")
    assert ("u_pll", "OUT") in pllout.terms and ("u_ram0", "CK") in pllout.terms


def test_liberty_parser():
    masters = parse_liberty(LIB)
    assert masters["INVX1"].pins["A"].capacitance == 0.002
    assert masters["INVX1"].pins["Y"].direction == "OUTPUT"
    # geometry fallback: square from area
    buf = masters["BUFX2"]
    assert buf.width == buf.height == pytest.approx(1.92 ** 0.5)


def test_sdc_weights():
    d = read_design(lef=[LEF], def_file=DEF, sdc=[SDC])
    by_name = {n.name: n for n in d.nets}
    assert by_name["clk"].weight == 0.0  # clock net
    assert by_name["in1"].weight == 2.0  # set_max_delay -from in1
    assert by_name["n0"].weight == 2.0  # -to u_ram0/D
    assert by_name["rst"].weight == 0.25  # false path
    assert by_name["pllout"].weight == 1.0  # untouched


def test_liberty_weights():
    d = read_design(lef=[LEF], def_file=DEF, liberty=[LIB])
    by_name = {n.name: n for n in d.nets}
    # n0 drives RAM D (cap .01) - well above mean -> weight scaled up
    assert by_name["n0"].weight > by_name["out1"].weight


# ── Builder combos ───────────────────────────────────────────────────────────


def _build(tmp_path, **kw):
    design = read_design(**kw)
    return build_benchmark(design, workdir=tmp_path / "work")


def test_build_lef_def(tmp_path):
    r = _build(tmp_path, lef=[LEF], def_file=DEF)
    b = r.benchmark
    assert b.canvas_width == 60.0 and b.canvas_height == 60.0
    # 4 LEF BLOCK instances + 1 blockage dummy
    assert b.num_hard_macros == 5
    assert r.hard_names[:4] == ["u_rom", "u_ram0", "u_ram1", "u_pll"]
    # fixed: u_rom + the blockage
    assert int(b.macro_fixed.sum()) == 2
    # origin shift: u_ram0 ll (15,15) abs -> center (19,18) -> canvas (14,13)
    i = r.hard_names.index("u_ram0")
    assert b.macro_positions[i].tolist() == [14.0, 13.0]
    # u_buf had no LEF/Liberty geometry -> dropped
    assert r.dropped == ["u_buf"]
    assert len(r.port_names) == 4
    assert b.num_soft_macros == len(r.soft_members) >= 1
    assert b.num_nets > 0 and len(b.net_nodes) == b.num_nets
    # exact scorer attached and functional
    from macro_place.objective import compute_proxy_cost
    costs = compute_proxy_cost(b.macro_positions, b, r.plc)
    assert float(costs["proxy_cost"]) > 0


def test_build_lef_verilog_no_def(tmp_path):
    r = _build(tmp_path, lef=[LEF], verilog=VLG, liberty=[LIB])
    b = r.benchmark
    # no die area: canvas derived from area/utilization, square
    assert b.canvas_width == b.canvas_height > 10
    assert b.num_hard_macros == 4  # no blockages without DEF
    # everything unplaced -> all seeds in bounds
    half = b.macro_sizes / 2
    assert torch.all(b.macro_positions - half >= -1e-6)
    assert torch.all(b.macro_positions[:, 0] + half[:, 0] <= b.canvas_width + 1e-6)
    assert torch.all(b.macro_positions[:, 1] + half[:, 1] <= b.canvas_height + 1e-6)
    # connectivity clustering used (nothing placed)
    assert b.num_soft_macros >= 1


def test_build_all_inputs(tmp_path):
    r = _build(tmp_path, lef=[LEF], def_file=DEF, verilog=VLG, sdc=[SDC],
               liberty=[LIB])
    b = r.benchmark
    assert b.num_hard_macros == 5
    assert r.dropped == []  # BUFX2 geometry comes from Liberty
    # SDC criticality survives into the benchmark tensors (nets that touch
    # fewer than two macros/clusters - like the clock net - are not mapped)
    assert float(b.net_weights.max()) >= 2.0
    by_name = {n.name: n for n in r.design.nets}
    assert by_name["clk"].weight == 0.0


# ── Writers ──────────────────────────────────────────────────────────────────


def test_def_writer_patch_roundtrip(tmp_path):
    r = _build(tmp_path, lef=[LEF], def_file=DEF)
    placement = r.benchmark.macro_positions.clone()
    i = r.hard_names.index("u_ram1")
    placement[i] = torch.tensor([20.0, 20.0])  # move one macro
    placements = r.placements_um(placement)
    out = tmp_path / "placed.def"
    write_def(r.design, placements, out, template_path=DEF)

    d2 = parse_def(out)
    assert len(d2.components) == 17
    # moved macro: center (20,20) canvas -> abs ll = 20-4+5, 20-3+5
    assert d2.components["u_ram1"].pos == (21.0, 22.0)
    assert d2.components["u_ram1"].status == "PLACED"
    assert d2.components["u_ram1"].orient == "FS"  # orientation preserved
    assert d2.components["u_rom"].status == "FIXED"
    assert d2.components["u_rom"].pos == (6.0, 6.0)  # fixed never moves
    # std cells follow their cluster, stay in the file
    assert d2.components["u0"].status == "PLACED"
    # untouched sections survive
    assert d2.die_area == (5.0, 5.0, 65.0, 65.0)
    assert len(d2.nets) == 12


def test_def_writer_fresh(tmp_path):
    r = _build(tmp_path, lef=[LEF], verilog=VLG, liberty=[LIB])
    placements = r.placements_um(r.benchmark.macro_positions)
    out = tmp_path / "fresh.def"
    write_def(r.design, placements, out)  # no template
    d2 = parse_def(out)
    assert len(d2.components) == 17
    assert d2.components["u_ram0"].status == "PLACED"
    assert set(d2.io_pins) == {"clk", "rst", "in1", "out1"}


def test_tcl_writer(tmp_path):
    r = _build(tmp_path, lef=[LEF], def_file=DEF)
    placements = r.placements_um(r.benchmark.macro_positions)
    macros = {n: placements[n] for n in r.hard_names if n in placements}

    icc2 = tmp_path / "place.tcl"
    write_tcl(macros, r.design, icc2, dialect="icc2")
    text = icc2.read_text()
    assert "set_cell_location -coordinates {15.0000 15.0000}" in text
    assert "[get_cells {u_ram0}]" in text
    assert "set_fixed_objects" in text

    inv = tmp_path / "place_innovus.tcl"
    write_tcl(macros, r.design, inv, dialect="innovus")
    text = inv.read_text()
    assert "placeInstance u_ram0 15.0000 15.0000 N -fixed" in text

    with pytest.raises(ValueError):
        write_tcl(macros, r.design, tmp_path / "x.tcl", dialect="vivado")


def test_report_writer(tmp_path):
    from eda_io.report import write_report
    r = _build(tmp_path, lef=[LEF], def_file=DEF, sdc=[SDC])
    out = tmp_path / "qor.rpt"
    write_report(out, r, r.benchmark.macro_positions, runtime_s=1.5,
                 valid=True, inputs={"lef": str(LEF)})
    text = out.read_text()
    assert "chiptop" in text
    assert "HPWL initial" in text and "HPWL placed" in text
    assert "hard overlaps placed" in text
    assert "PASS" in text


# ── End to end ───────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_place_design_cli(tmp_path):
    out_def = tmp_path / "placed.def"
    out_tcl = tmp_path / "place.tcl"
    out_rpt = tmp_path / "qor.rpt"
    cmd = [
        sys.executable, str(SRC / "place_design.py"),
        "--lef", str(LEF), "--def", str(DEF), "--sdc", str(SDC),
        "--lib", str(LIB), "--budget", "20",
        "--out-def", str(out_def), "--out-tcl", str(out_tcl),
        "--report", str(out_rpt),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out_def.exists() and out_tcl.exists() and out_rpt.exists()
    d2 = parse_def(out_def)
    # every non-dropped component ends up placed or fixed
    for name, comp in d2.components.items():
        if name == "u_buf":
            continue
        assert comp.status in ("PLACED", "FIXED"), name
        assert comp.pos is not None, name
    assert "proxy" in out_rpt.read_text()


def test_read_design_validation():
    with pytest.raises(ValueError):
        read_design(def_file=DEF)  # no geometry source
    with pytest.raises(ValueError):
        read_design(lef=[LEF])  # no instance source
