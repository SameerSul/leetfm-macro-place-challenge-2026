import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from placer.plc.loader import _trace_benchmark_name


def test_ng45_trace_name_uses_design_directory():
    source = Path("external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping")

    assert _trace_benchmark_name("output_CT_Grouping", source) == "nvdla"


def test_ordinary_trace_name_is_unchanged():
    assert _trace_benchmark_name("ibm10", Path("benchmarks/ibm10")) == "ibm10"
