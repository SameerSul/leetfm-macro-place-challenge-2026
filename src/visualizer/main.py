"""Launch a live VivaPlace run or replay a schema-v1 trace."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from visualizer.events import EventEmitter, SCHEMA_VERSION


def _benchmark_dir(name: str) -> Path:
    from macro_place.evaluate import NG45_BENCHMARKS

    if name in NG45_BENCHMARKS:
        return ROOT / NG45_BENCHMARKS[name]
    path = ROOT / "external/MacroPlacement/Testcases/ICCAD04" / name
    if not (path / "netlist.pb.txt").exists():
        raise ValueError(f"unknown IBM/NG45 benchmark: {name}")
    return path


def _worker(queue, benchmark_name, benchmark_dir, sample_every):
    placer_started = False
    try:
        from macro_place.loader import load_benchmark_from_dir
        from placer.pipeline.macro_placer import MacroPlacer

        # Deterministic score quotas remain authoritative while diagnostic event
        # serialization is excluded from wall-clock safety guards.
        os.environ["HIER_DIAGNOSTIC_NO_DEADLINES"] = "1"
        source = Path(benchmark_dir) if benchmark_dir else _benchmark_dir(benchmark_name)
        benchmark, _plc = load_benchmark_from_dir(source.as_posix())
        benchmark.name = benchmark_name or source.name
        setattr(benchmark, "_source_dir", source)
        sink = EventEmitter(queue.put)
        placer_started = True
        MacroPlacer(event_sink=sink, dreamplace_sample_every=sample_every).place(benchmark)
    except BaseException as exc:
        if not placer_started:
            queue.put(
                {
                    "schema": SCHEMA_VERSION,
                    "type": "error",
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )


def _default_trace(benchmark: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return ROOT / "ml_data/visualizer" / benchmark / f"{run_id}.jsonl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--benchmark", help="IBM or NG45 benchmark name")
    source.add_argument("--benchmark-dir", type=Path, help="directory containing netlist.pb.txt")
    source.add_argument("--replay", type=Path, help="completed or partial JSONL trace")
    parser.add_argument("--trace", type=Path, help="output JSONL path")
    parser.add_argument("--dreamplace-sample-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    from pyqtgraph.Qt import QtWidgets
    from visualizer.qt_dashboard import Dashboard

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    process = None
    event_queue = None
    if args.replay:
        window = Dashboard(replay=args.replay)
    else:
        label = args.benchmark or args.benchmark_dir.name
        trace = args.trace or _default_trace(label)
        context = mp.get_context("spawn")
        event_queue = context.Queue(maxsize=4096)
        process = context.Process(
            target=_worker,
            args=(event_queue, args.benchmark, args.benchmark_dir, args.dreamplace_sample_every),
            daemon=True,
        )
        process.start()
        window = Dashboard(event_queue=event_queue, trace_path=trace)
    window.show()
    code = app.exec()
    if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=2)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
