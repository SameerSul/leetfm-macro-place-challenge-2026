import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtWidgets

from visualizer.events import SCHEMA_VERSION
from visualizer.main import parse_args
from visualizer.qt_dashboard import Dashboard


def test_offscreen_layers_sidebar_and_replay_controls(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    trace = tmp_path / "trace.jsonl"
    rows = [
        {
            "schema": SCHEMA_VERSION,
            "type": "run_metadata",
            "canvas": [100, 80],
            "macro_names": ["hard/a", "soft/a"],
            "macro_sizes": [[10, 8], [6, 6]],
            "macro_fixed": [True, False],
            "num_hard_macros": 1,
            "net_nodes": [[0, 1]],
            "net_weights": [1],
            "port_positions": [],
            "macro_pin_offsets": [[], []],
        },
        {
            "schema": SCHEMA_VERSION,
            "type": "checkpoint",
            "reason": "hierarchy_construction",
            "positions": [[20, 20], [35, 20]],
            "hierarchy": {
                "leaf_labels": [0],
                "leaf_clusters": {"0": [0]},
                "parent_clusters": {"0": [0]},
                "cluster_softs": {"0": [0]},
                "bridge_softs": [],
                "edges": [],
            },
            "metrics": {
                "wirelength": 1,
                "density": 2,
                "congestion": 3,
                "hierarchy": 4,
                "proxy": 3.5,
            },
            "metrics_stale": False,
        },
        {
            "schema": SCHEMA_VERSION,
            "type": "accepted_move",
            "indices": [1],
            "old_positions": [[35, 20]],
            "new_positions": [[40, 22]],
            "metrics": {
                "wirelength": 0.9,
                "density": 2,
                "congestion": 3,
                "hierarchy": 3.9,
                "hierarchy_hard_containment": 0.90,
                "hierarchy_hard_limit": 1.00,
                "hierarchy_hard_headroom": 0.10,
                "hierarchy_cluster_compactness": 0.20,
                "hierarchy_cluster_compactness_limit": 0.25,
                "hierarchy_cluster_compactness_headroom": 0.05,
                "proxy": 3.4,
            },
            "metrics_stale": False,
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = parse_args(
        [
            "--replay",
            str(trace),
            "--export-mp4",
            str(tmp_path / "cli-demo.mp4"),
            "--export-speed",
            "0.02",
        ]
    )
    assert args.export_speed == 0.02
    newest = parse_args(["--replay", str(tmp_path)])
    assert newest.replay == trace
    cached = parse_args(["--benchmark", "ibm10", "--use-dreamplace-cache"])
    assert cached.use_dreamplace_cache is True
    window = Dashboard(replay=trace)
    window.timeline.setValue(2)
    window._render_current()
    assert window.badge.text() == "EXACT"
    assert "proxy: 3.400000" in window.metric_labels["proxy"].text()
    assert "hard: 0.90000 / 1.00000" in window.contract_metrics.text()
    assert "compact: 0.20000 / 0.25000" in window.contract_metrics.text()
    window.real_toggle.setChecked(True)
    window.hierarchy_toggle.setChecked(True)
    window._previous()
    assert window.timeline.value() == 1
    window._next()
    assert window.timeline.value() == 2
    slow_10x = window.speed.findData(0.1)
    slow_50x = window.speed.findData(0.02)
    assert slow_10x >= 0
    assert slow_50x >= 0
    window.speed.setCurrentIndex(slow_50x)
    assert window.replay_speed == 0.02
    window._toggle_pause()
    assert not window.paused
    assert window.timeline.value() == 0
    for _ in range(49):
        window._tick()
    assert window.timeline.value() == 0
    window._tick()
    assert window.timeline.value() == 1
    window._toggle_pause()
    assert window.paused
    assert window.export_button.isEnabled()

    encoded = []

    class FakeWriter:
        def __init__(self, path, fps):
            assert fps == 3
            path.touch()

        def append_data(self, frame):
            encoded.append(frame.copy())

        def close(self):
            pass

    window._capture_video_frame = lambda: __import__("numpy").zeros((12, 14, 3), dtype="uint8")
    output, written = window.export_video(tmp_path / "demo", speed=0.1, writer_factory=FakeWriter)
    assert output == tmp_path / "demo.mp4"
    assert output.is_file()
    assert written == 2
    assert len(encoded) == 2
    window._select(1)
    assert window.selected == 1
    window.close()
    app.processEvents()


def test_replay_placeholder_has_actionable_error(tmp_path, capsys):
    with pytest.raises(SystemExit, match="2"):
        parse_args(["--replay", str(tmp_path / "TRACE.jsonl")])
    error = capsys.readouterr().err
    assert "example placeholder" in error
    assert "--replay ml_data/visualizer/ibm10" in error


def test_seed_name_remains_visible_during_dreamplace_progress(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    trace = tmp_path / "seed-trace.jsonl"
    rows = [
        {
            "schema": SCHEMA_VERSION,
            "type": "run_metadata",
            "canvas": [100, 80],
            "macro_names": ["hard/a"],
            "macro_sizes": [[10, 8]],
            "macro_fixed": [False],
            "num_hard_macros": 1,
        },
        {
            "schema": SCHEMA_VERSION,
            "type": "checkpoint",
            "reason": "initial_placement",
            "positions": [[20, 20]],
            "metrics_stale": True,
        },
        {
            "schema": SCHEMA_VERSION,
            "type": "seed_status",
            "seed_name": "re2map_recursive_1",
            "status": "building",
            "metrics_stale": True,
        },
        {
            "schema": SCHEMA_VERSION,
            "type": "dreamplace_progress",
            "seed_name": "re2map_recursive_1",
            "iteration": 30,
            "indices": [0],
            "new_positions": [[25, 24]],
            "metrics_stale": True,
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    window = Dashboard(replay=trace)
    window.timeline.setValue(2)
    window._render_current()
    assert window.seed.text() == "Seed: re2map_recursive_1 · building"
    window.timeline.setValue(3)
    window._render_current()
    assert window.seed.text() == "Seed: re2map_recursive_1 · DREAMPlace iteration 30"
    window.close()
    app.processEvents()
