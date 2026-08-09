import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtWidgets

from visualizer.events import SCHEMA_VERSION
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
                "proxy": 3.4,
            },
            "metrics_stale": False,
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    window = Dashboard(replay=trace)
    window.timeline.setValue(2)
    window._render_current()
    assert window.badge.text() == "EXACT"
    assert "proxy: 3.400000" in window.metric_labels["proxy"].text()
    window.real_toggle.setChecked(True)
    window.hierarchy_toggle.setChecked(True)
    window._previous()
    assert window.timeline.value() == 1
    window._next()
    assert window.timeline.value() == 2
    window._select(1)
    assert window.selected == 1
    window.close()
    app.processEvents()
