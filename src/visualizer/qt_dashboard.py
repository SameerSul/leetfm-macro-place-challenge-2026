"""PyQtGraph desktop dashboard. Imported only by the visualizer launcher."""

from __future__ import annotations

import queue as queue_module
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .model import (
    collapse_synthetic_groups,
    extract_real_nets,
    filter_real_nets,
    hierarchy_centroid_edges,
    hierarchy_color,
)
from .trace import TraceReader, TraceWriter


class MacroLayer(pg.GraphicsObject):
    """Paint every macro and hierarchy outline in two batched pictures."""

    def __init__(self, on_select):
        super().__init__()
        self.on_select = on_select
        self.metadata: dict[str, Any] = {}
        self.hierarchy: dict[str, Any] = {}
        self.positions = np.empty((0, 2))
        self.selected: int | None = None
        self.moved: set[int] = set()
        self.vectors: list[tuple[np.ndarray, np.ndarray]] = []
        self._picture = QtGui.QPicture()
        self.setAcceptHoverEvents(True)

    def boundingRect(self):
        canvas = self.metadata.get("canvas", (1.0, 1.0))
        return QtCore.QRectF(0, 0, float(canvas[0]), float(canvas[1]))

    def paint(self, painter, *_args):
        painter.drawPicture(0, 0, self._picture)

    def set_state(self, positions, metadata, hierarchy, moved=(), vectors=(), selected=None):
        self.prepareGeometryChange()
        self.positions = np.asarray(positions, dtype=float)
        self.metadata = metadata or {}
        self.hierarchy = hierarchy or {}
        self.moved = set(int(x) for x in moved)
        self.vectors = list(vectors)
        self.selected = selected
        self._rebuild()
        self.update()

    def _labels(self):
        count = len(self.positions)
        hard = int(self.metadata.get("num_hard_macros", count))
        labels = np.full(count, -1, dtype=int)
        leaf = self.hierarchy.get("leaf_labels", ())
        labels[: min(hard, len(leaf))] = leaf[:hard]
        for cluster, softs in self.hierarchy.get("cluster_softs", {}).items():
            for soft in softs:
                index = hard + int(soft)
                if index < count:
                    labels[index] = int(cluster)
        return labels

    def _rebuild(self):
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        sizes = np.asarray(self.metadata.get("macro_sizes", ()), dtype=float)
        fixed = self.metadata.get("macro_fixed", ())
        hard = int(self.metadata.get("num_hard_macros", len(self.positions)))
        labels = self._labels()

        for parent, members in self.hierarchy.get("parent_clusters", {}).items():
            valid = np.asarray([int(i) for i in members if int(i) < len(self.positions)], dtype=int)
            if not valid.size or sizes.shape[0] < len(self.positions):
                continue
            lo = np.min(self.positions[valid] - sizes[valid] / 2.0, axis=0)
            hi = np.max(self.positions[valid] + sizes[valid] / 2.0, axis=0)
            color = hierarchy_color(int(parent), parent=True)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(pg.mkPen(color, width=2))
            painter.drawRect(QtCore.QRectF(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]))

        for index, (center, size) in enumerate(zip(self.positions, sizes)):
            x, y = center - size / 2.0
            rect = QtCore.QRectF(float(x), float(y), float(size[0]), float(size[1]))
            color = hierarchy_color(int(labels[index]))
            if index >= hard:
                color = (*color[:3], 90)
                pen = pg.mkPen(color[:3], width=1, style=QtCore.Qt.PenStyle.DashLine)
            else:
                pen = pg.mkPen((25, 30, 38), width=1.2)
            painter.setBrush(pg.mkBrush(color))
            painter.setPen(pen)
            painter.drawRect(rect)
            if index < len(fixed) and fixed[index]:
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.setPen(pg.mkPen((230, 55, 55), width=2.5))
                painter.drawRect(rect.adjusted(-1.5, -1.5, 1.5, 1.5))
                painter.drawRect(rect.adjusted(1.5, 1.5, -1.5, -1.5))
            if index in self.moved or index == self.selected:
                glow = (255, 220, 70) if index in self.moved else (255, 255, 255)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.setPen(pg.mkPen(glow, width=3))
                painter.drawRect(rect.adjusted(-2, -2, 2, 2))
        painter.setPen(pg.mkPen((255, 220, 70, 190), width=2))
        for old, new in self.vectors:
            painter.drawLine(QtCore.QPointF(*old), QtCore.QPointF(*new))
        painter.end()
        self._picture = picture

    def _macro_at(self, point) -> int | None:
        if not len(self.positions):
            return None
        sizes = np.asarray(self.metadata.get("macro_sizes", ()), dtype=float)
        if sizes.shape != self.positions.shape:
            return None
        delta = np.abs(self.positions - np.array([point.x(), point.y()]))
        hits = np.flatnonzero(np.all(delta <= sizes / 2.0, axis=1))
        return int(hits[-1]) if hits.size else None

    def mouseClickEvent(self, event):
        index = self._macro_at(event.pos())
        if index is not None:
            self.on_select(index)
            event.accept()

    def hoverMoveEvent(self, event):
        index = self._macro_at(event.pos())
        if index is None:
            self.setToolTip("")
            return
        names = self.metadata.get("macro_names", ())
        sizes = self.metadata.get("macro_sizes", ())
        name = names[index] if index < len(names) else f"macro {index}"
        size = sizes[index] if index < len(sizes) else (0, 0)
        pos = self.positions[index]
        self.setToolTip(
            f"#{index} {name}\ncenter=({pos[0]:.3f}, {pos[1]:.3f})\nsize=({size[0]:.3f}, {size[1]:.3f})"
        )


class WireLayer(pg.GraphicsObject):
    """Paint real, synthetic, and hierarchy wiring in batched form."""

    def __init__(self):
        super().__init__()
        self._picture = QtGui.QPicture()
        self.canvas = (1.0, 1.0)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, float(self.canvas[0]), float(self.canvas[1]))

    def paint(self, painter, *_args):
        painter.drawPicture(0, 0, self._picture)

    def set_state(
        self,
        positions,
        metadata,
        hierarchy,
        *,
        selected=None,
        real_limit=250,
        show_real=False,
        show_synthetic=True,
        show_hierarchy=False,
    ):
        positions = np.asarray(positions, dtype=float)
        self.canvas = metadata.get("canvas", (1.0, 1.0))
        ports = np.asarray(metadata.get("port_positions", ()), dtype=float).reshape((-1, 2))
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        if show_real:
            painter.setPen(pg.mkPen((100, 180, 235, 80), width=1))
            for net in filter_real_nets(extract_real_nets(metadata), real_limit, selected):
                points = []
                for endpoint in net.endpoints:
                    if endpoint.macro is not None and endpoint.macro < len(positions):
                        points.append(positions[endpoint.macro] + np.asarray(endpoint.offset))
                    elif endpoint.port is not None and endpoint.port < len(ports):
                        points.append(ports[endpoint.port])
                self._draw_hyperedge(painter, points)

        if show_synthetic:
            weight = int(metadata.get("group_weight", 1))
            width = 1.0 + min(4.0, np.log2(max(1, weight)) / 3.0)
            painter.setPen(
                pg.mkPen((255, 184, 77, 150), width=width, style=QtCore.Qt.PenStyle.DashLine)
            )
            for group in collapse_synthetic_groups(hierarchy, weight):
                points = [positions[i] for i in group["members"] if i < len(positions)]
                self._draw_hyperedge(painter, points)

        if show_hierarchy:
            clusters = hierarchy.get("leaf_clusters", {})
            centroids = {
                int(cluster): np.mean(positions[np.asarray(members, dtype=int)], axis=0)
                for cluster, members in clusters.items()
                if members
            }
            painter.setPen(pg.mkPen((190, 130, 245, 150), width=2))
            for left, right, _weight in hierarchy_centroid_edges(hierarchy):
                if left in centroids and right in centroids:
                    painter.drawLine(
                        QtCore.QPointF(*centroids[left]), QtCore.QPointF(*centroids[right])
                    )
        painter.end()
        self.prepareGeometryChange()
        self._picture = picture
        self.update()

    @staticmethod
    def _draw_hyperedge(painter, points):
        if len(points) < 2:
            return
        center = np.mean(points, axis=0)
        for point in points:
            painter.drawLine(QtCore.QPointF(*center), QtCore.QPointF(*point))


class Dashboard(QtWidgets.QMainWindow):
    """Live queue consumer and trace replay window."""

    def __init__(self, *, event_queue=None, trace_path: Path | None = None, replay=None):
        super().__init__()
        self.setWindowTitle("VivaPlace Live Visualizer")
        self.resize(1500, 920)
        self.event_queue = event_queue
        self.writer = TraceWriter(trace_path) if trace_path else None
        self.events: list[dict[str, Any]] = []
        self.frames: list[np.ndarray | None] = []
        self.metadata: dict[str, Any] = {}
        self.hierarchy: dict[str, Any] = {}
        self.current_positions: np.ndarray | None = None
        self.selected: int | None = None
        self.is_replay = replay is not None
        self.paused = replay is not None
        self.replay_speed = 1.0
        self._replay_accum = 0.0
        self._build_ui()
        if replay is not None:
            for event, positions in TraceReader(replay).frames():
                self._append_event(event, positions=positions, write=False)
            if self.events:
                self.timeline.setValue(0)
                self._render_index(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _build_ui(self):
        root = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(root)
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.wires = WireLayer()
        self.macros = MacroLayer(self._select)
        self.plot.addItem(self.wires)
        self.plot.addItem(self.macros)
        layout.addWidget(self.plot, 1)

        side = QtWidgets.QWidget()
        panel = QtWidgets.QVBoxLayout(side)
        self.algorithm = QtWidgets.QLabel("Waiting for placement…")
        self.algorithm.setWordWrap(True)
        self.badge = QtWidgets.QLabel("STALE")
        self.badge.setStyleSheet("color:#ffb84d;font-weight:bold")
        panel.addWidget(self.algorithm)
        panel.addWidget(self.badge)
        self.metric_labels = {}
        for key in ("wirelength", "density", "congestion", "hierarchy", "proxy"):
            label = QtWidgets.QLabel(f"{key}: —")
            self.metric_labels[key] = label
            panel.addWidget(label)
        self.trend = pg.PlotWidget()
        self.trend.setMaximumHeight(180)
        self.trend.addLegend()
        panel.addWidget(self.trend)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search macro name")
        self.search.returnPressed.connect(self._search)
        panel.addWidget(self.search)
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(self._reset_view)
        panel.addWidget(reset)

        self.real_toggle = QtWidgets.QCheckBox("Real nets")
        self.synthetic_toggle = QtWidgets.QCheckBox("Synthetic grouping nets")
        self.synthetic_toggle.setChecked(True)
        self.hierarchy_toggle = QtWidgets.QCheckBox("Hierarchy graph")
        for toggle in (self.real_toggle, self.synthetic_toggle, self.hierarchy_toggle):
            toggle.toggled.connect(self._render_current)
            panel.addWidget(toggle)
        self.net_limit = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.net_limit.setRange(0, 1000)
        self.net_limit.setValue(250)
        self.net_limit.valueChanged.connect(self._render_current)
        panel.addWidget(QtWidgets.QLabel("Real-net limit"))
        panel.addWidget(self.net_limit)

        controls = QtWidgets.QHBoxLayout()
        for text, callback in (
            ("◀", self._previous),
            ("Pause", self._pause),
            ("Live", self._live),
            ("▶", self._next),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        panel.addLayout(controls)
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.sliderMoved.connect(self._scrub)
        panel.addWidget(self.timeline)
        self.speed = QtWidgets.QComboBox()
        for value in (0.25, 0.5, 1, 2, 4, 8):
            self.speed.addItem(f"{value:g}×", value)
        self.speed.setCurrentText("1×")
        self.speed.currentIndexChanged.connect(
            lambda: setattr(self, "replay_speed", float(self.speed.currentData()))
        )
        panel.addWidget(self.speed)
        panel.addStretch(1)
        side.setMaximumWidth(360)
        layout.addWidget(side)
        self.setCentralWidget(root)

    def _tick(self):
        newest = None
        if self.event_queue is not None:
            while True:
                try:
                    event = self.event_queue.get_nowait()
                except queue_module.Empty:
                    break
                self._append_event(event)
                newest = len(self.events) - 1
        if newest is not None and not self.paused:
            self.timeline.setValue(newest)
            self._render_index(newest)
        elif self.is_replay and not self.paused and self.events:
            self._replay_accum += self.replay_speed
            while self._replay_accum >= 1.0:
                self._replay_accum -= 1.0
                if self.timeline.value() < len(self.events) - 1:
                    self.timeline.setValue(self.timeline.value() + 1)
                    self._render_index(self.timeline.value())

    def _append_event(self, event, *, positions=None, write=True):
        event = dict(event)
        if write and self.writer:
            self.writer.emit(event)
        if event.get("type") == "run_metadata":
            self.metadata = event
        if event.get("metadata_update"):
            self.metadata.update(event["metadata_update"])
        if event.get("hierarchy"):
            self.hierarchy = event["hierarchy"]
        if positions is None:
            if "positions" in event:
                self.current_positions = np.asarray(event["positions"], dtype=float)
            elif event.get("type") in {"accepted_move", "dreamplace_progress"}:
                if self.current_positions is not None:
                    self.current_positions = self.current_positions.copy()
                    self.current_positions[np.asarray(event.get("indices", ()), dtype=int)] = (
                        np.asarray(event.get("new_positions", ()), dtype=float).reshape((-1, 2))
                    )
            positions = self.current_positions
        self.events.append(event)
        self.frames.append(None if positions is None else np.asarray(positions).copy())
        self.timeline.setMaximum(len(self.events) - 1)

    def _render_index(self, index):
        if not (0 <= index < len(self.events)):
            return
        event = self.events[index]
        positions = self.frames[index]
        if positions is None:
            return
        # Reconstruct metadata/hierarchy at the selected point for replay.
        for row in self.events[: index + 1]:
            if row.get("type") == "run_metadata":
                self.metadata = dict(row)
            if row.get("metadata_update"):
                self.metadata.update(row["metadata_update"])
            if row.get("hierarchy"):
                self.hierarchy = row["hierarchy"]
        moved = event.get("indices", ())
        vectors = list(zip(event.get("old_positions", ()), event.get("new_positions", ())))
        self.macros.set_state(
            positions,
            self.metadata,
            self.hierarchy,
            moved=moved,
            vectors=vectors,
            selected=self.selected,
        )
        self._set_wires(positions)
        label = event.get("algorithm") or event.get("label") or event.get("reason") or event["type"]
        suffix = []
        if event.get("round") is not None:
            suffix.append(f"round {event['round']}")
        if event.get("lane"):
            suffix.append(str(event["lane"]))
        self.algorithm.setText(str(label) + (" · " + " · ".join(suffix) if suffix else ""))
        stale = bool(event.get("metrics_stale", event.get("metrics") is None))
        self.badge.setText("STALE" if stale else "EXACT")
        self.badge.setStyleSheet(
            "color:#ffb84d;font-weight:bold" if stale else "color:#62d98b;font-weight:bold"
        )
        metrics = event.get("metrics") or next(
            (
                previous["metrics"]
                for previous in reversed(self.events[:index])
                if previous.get("metrics")
            ),
            {},
        )
        self._update_metrics(index, metrics)

    def _update_metrics(self, index, metrics):
        initial = next((e.get("metrics") for e in self.events if e.get("metrics")), None) or {}
        for key, label in self.metric_labels.items():
            if key not in metrics:
                label.setText(f"{key}: —")
                continue
            delta = float(metrics[key]) - float(initial.get(key, metrics[key]))
            label.setText(f"{key}: {float(metrics[key]):.6f}  ({delta:+.6f})")
        self.trend.clear()
        colors = {
            "wirelength": "c",
            "density": "y",
            "congestion": "m",
            "hierarchy": "g",
            "proxy": "w",
        }
        rows = [
            (i, e.get("metrics"))
            for i, e in enumerate(self.events[: index + 1])
            if e.get("metrics")
        ]
        for key, color in colors.items():
            values = [(i, float(m[key])) for i, m in rows if key in m]
            if values:
                self.trend.plot([v[0] for v in values], [v[1] for v in values], pen=color, name=key)

    def _set_wires(self, positions):
        self.wires.set_state(
            positions,
            self.metadata,
            self.hierarchy,
            selected=self.selected,
            real_limit=self.net_limit.value(),
            show_real=self.real_toggle.isChecked(),
            show_synthetic=self.synthetic_toggle.isChecked(),
            show_hierarchy=self.hierarchy_toggle.isChecked(),
        )

    def _render_current(self, *_args):
        self._render_index(self.timeline.value())

    def _select(self, index):
        self.selected = index
        self._render_current()

    def _search(self):
        query = self.search.text().casefold()
        for index, name in enumerate(self.metadata.get("macro_names", ())):
            if query and query in str(name).casefold():
                self._select(index)
                pos = self.frames[self.timeline.value()][index]
                self.plot.setRange(
                    xRange=(pos[0] - 20, pos[0] + 20), yRange=(pos[1] - 20, pos[1] + 20)
                )
                break

    def _reset_view(self):
        canvas = self.metadata.get("canvas", (1, 1))
        self.plot.setRange(xRange=(0, canvas[0]), yRange=(0, canvas[1]), padding=0.03)

    def _pause(self):
        self.paused = True

    def _live(self):
        self.paused = False
        if self.events:
            self.timeline.setValue(len(self.events) - 1)
            self._render_current()

    def _previous(self):
        self.paused = True
        self.timeline.setValue(max(0, self.timeline.value() - 1))
        self._render_current()

    def _next(self):
        self.paused = True
        self.timeline.setValue(min(len(self.events) - 1, self.timeline.value() + 1))
        self._render_current()

    def _scrub(self, value):
        self.paused = True
        self._render_index(value)

    def closeEvent(self, event):
        if self.writer:
            self.writer.close()
        super().closeEvent(event)
