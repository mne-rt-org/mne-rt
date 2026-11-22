"""Live view of the connections a connectivity modality computes.

Classes
-------
ConnectivityPlot
    One tab per connectivity modality: its pairs drawn as edges on the head
    (sensor space) or between ROIs on a circle (source space), and each pair's
    value over time beside them.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from mne_rt._logging import logger, set_log_level
from mne_rt.viz._layout import fit_to_screen
from mne_rt.viz._sensor_selector import draw_head, head_xy, sensor_positions

_BG = "#0d0d1a"
_DIM = "#3a3a5c"
_TEXT = "#c0c0d8"
# One colour per pair in the trend plot, distinct on the dark background.
_PAIR_COLORS = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#fff176", "#4db6ac"]
# Metrics bounded in [0, 1]; the signed ones span [-1, 1]; others scale to the data.
_UNIT_METRICS = {"coh", "plv", "ciplv", "pli", "wpli", "mic", "graph"}  # graph: normalised
_SIGNED_METRICS = {"imcoh", "corr", "cohy"}


def value_range(metric: str, signed: bool) -> Optional[tuple[float, float]]:
    """The fixed colour range of a metric, or ``None`` to scale to the data."""
    if metric in _SIGNED_METRICS:
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    if metric in _UNIT_METRICS:
        return (0.0, 1.0)
    return None


def circle_positions(nodes: list[str]) -> dict[str, tuple[float, float]]:
    """ROIs on a circle in the unit square, left hemisphere on the left.

    As in MNE-Connectivity's circle plot: ``-lh``/``Left-`` ROIs run down the
    left half, ``-rh``/``Right-`` ones down the right, any other along the
    bottom.
    """

    def side(name: str) -> int:
        low = name.lower()
        if low.endswith("-lh") or low.startswith(("left-", "ctx-lh-", "lh.")):
            return -1
        if low.endswith("-rh") or low.startswith(("right-", "ctx-rh-", "rh.")):
            return 1
        return 0

    sides = [side(n) for n in nodes]
    if not any(sides):  # no hemisphere to go by (e.g. Broca, Wernicke): all around
        angles = np.pi / 2 - 2 * np.pi * np.arange(len(nodes)) / max(len(nodes), 1)
        return {n: (0.5 + 0.42 * np.cos(a), 0.5 + 0.42 * np.sin(a)) for n, a in zip(nodes, angles)}
    groups = {s: [n for n, ns in zip(nodes, sides) if ns == s] for s in (-1, 1, 0)}
    n_mid = len(groups[0])
    # The bottom arc widens with the ROIs on it, and the sides stop above it.
    half = min(0.32, 0.08 * (n_mid + 1)) * np.pi if n_mid else 0.0
    low = -np.pi / 2 + half + 0.06 * np.pi if n_mid else -0.42 * np.pi
    pos: dict[str, tuple[float, float]] = {}
    for s, names in groups.items():
        if not names:
            continue
        if s == 0:
            angles = np.linspace(-np.pi / 2 - half, -np.pi / 2 + half, n_mid + 2)[1:-1][::-1]
        else:  # top to bottom on one side, symmetric across the midline
            span = np.linspace(0.42 * np.pi, low, len(names) + 2)[1:-1]
            angles = span if s == 1 else np.pi - span
        for name, a in zip(names, angles):
            pos[name] = (0.5 + 0.42 * np.cos(a), 0.5 + 0.42 * np.sin(a))
    return pos


def _chord(a: tuple[float, float], b: tuple[float, float], bend: float) -> np.ndarray:
    """A quadratic curve from ``a`` to ``b``, pulled ``bend`` of the way to the centre."""
    ctrl = np.array([0.5, 0.5]) * bend + (np.asarray(a) + np.asarray(b)) / 2 * (1 - bend)
    t = np.linspace(0, 1, 24)[:, None]
    return (1 - t) ** 2 * np.asarray(a) + 2 * (1 - t) * t * ctrl + t**2 * np.asarray(b)


class _Panel(QWidget):
    """One modality: its edges on the left, its pairs' values over time on the right."""

    def __init__(self, name, first, ch_names, info, history, rate) -> None:
        super().__init__()
        self._labels = list(first["labels"])
        self._metric, self._signed = first["metric"], first["signed"]
        self._range = value_range(self._metric, self._signed)
        self._cmaps = {False: pg.colormap.get("viridis"), True: pg.colormap.get("CET-D1")}
        self._history = float(history)
        # Generous: windows come about `rate` a second, and each keeps its time.
        n = max(2, int(round(history * rate * 2)))
        self._times = deque(maxlen=n)
        self._hist = [deque(maxlen=n) for _ in self._labels]
        self._value_hist = deque(maxlen=n)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Edges
        self._space = pg.PlotWidget(background=_BG)
        self._space.hideAxis("bottom")
        self._space.hideAxis("left")
        vb = self._space.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setMenuEnabled(False)
        vb.setAspectLocked(True)
        vb.setRange(xRange=(-0.08, 1.08), yRange=(-0.08, 1.14), padding=0)
        self._nodes = self._node_positions(first, ch_names, info)
        if first["space"] == "sensor":
            draw_head(self._space, pg.mkPen(_DIM, width=1.5))
        else:  # the ring the ROIs sit on, as in MNE-Connectivity's circle plot
            theta = np.linspace(0, 2 * np.pi, 160)
            self._space.plot(
                0.5 + 0.42 * np.cos(theta), 0.5 + 0.42 * np.sin(theta), pen=pg.mkPen(_DIM)
            )
        dots = np.array(list(self._nodes.values())) if self._nodes else np.empty((0, 2))
        self._space.addItem(
            pg.ScatterPlotItem(
                pos=dots, size=6, brush=pg.mkBrush(_DIM), pen=pg.mkPen(None), hoverable=False
            )
        )
        involved = {n for pair in self._labels for n in pair}
        for node, (x, y) in self._nodes.items():
            if node in involved or first["space"] == "source":
                text = pg.TextItem(node, color=_TEXT, anchor=(0.5, -0.3))
                text.setPos(x, y)
                self._space.addItem(text)
        self._edges = []
        bend = 0.0 if first["space"] == "sensor" else 0.55
        for a, b in self._labels:
            path = (
                _chord(self._nodes[a], self._nodes[b], bend)
                if a in self._nodes and b in self._nodes
                else np.empty((0, 2))
            )
            item = pg.PlotCurveItem(path[:, 0], path[:, 1], pen=pg.mkPen(_DIM, width=2))
            self._space.addItem(item)
            self._edges.append(item)
        self._scale = QLabel()
        self._scale.setStyleSheet(f"color:{_TEXT};")
        left = QVBoxLayout()
        left.addWidget(self._space, stretch=1)
        left.addWidget(self._scale)
        layout.addLayout(left, stretch=1)

        # Values over time
        self._trend = pg.PlotWidget(background=_BG)
        self._trend.setLabel("bottom", "Time", units="s")
        self._trend.setLabel("left", self._metric)
        self._trend.getAxis("left").enableAutoSIPrefix(False)  # 0.7, not 700 x 0.001
        self._trend.setMouseEnabled(x=False, y=False)
        self._trend.setXRange(-self._history, 0, padding=0.02)
        self._legend = self._trend.addLegend(offset=(-10, 10), labelTextColor=_TEXT)
        self._curves = []
        for k, (a, b) in enumerate(self._labels):
            pen = pg.mkPen(_PAIR_COLORS[k % len(_PAIR_COLORS)], width=1)
            self._curves.append(self._trend.plot([], [], pen=pen, name=f"{a}–{b}"))
        self._value_curve = self._trend.plot(
            [],
            [],
            pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine),
            name=f"{name} (value)",
        )
        layout.addWidget(self._trend, stretch=1)

    def _node_positions(self, first, ch_names, info) -> dict[str, tuple[float, float]]:
        if first["space"] == "source":
            nodes = first["nodes"] or sorted({n for pair in self._labels for n in pair})
            return circle_positions(nodes)
        names = list(ch_names) if ch_names else sorted({n for p in self._labels for n in p})
        return {
            ch: head_xy(*xy) for ch, xy in zip(names, sensor_positions(names, info, montage=None))
        }

    def show_values(self, snap: dict) -> None:
        """Draw one window's values (not ``update``: that is Qt's repaint request)."""
        edges = np.asarray(snap["edges"], dtype=float)
        # Each window at its own time, so a dropped one leaves a gap in the time
        # axis rather than shifting everything after it.
        self._times.append(float(snap.get("time", len(self._times))))
        for k, hist in enumerate(self._hist):
            hist.append(edges[k] if k < edges.size else np.nan)
        self._value_hist.append(float(snap["value"]))
        t = np.asarray(self._times) - self._times[-1]
        lo, hi = self._color_range(edges)
        cmap = self._cmaps[lo < 0]  # diverging around 0 for signed values
        for item, v in zip(self._edges, edges):
            if not np.isfinite(v):
                item.setPen(pg.mkPen(_DIM, width=1))
                continue
            frac = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
            strength = abs(frac - 0.5) * 2 if lo < 0 else frac
            item.setPen(pg.mkPen(cmap.map(frac, mode="qcolor"), width=1.5 + 4.5 * strength))
        for k, curve in enumerate(self._curves):
            curve.setData(t, np.asarray(self._hist[k]))
            a, b = self._labels[k]
            now = edges[k] if k < edges.size else np.nan
            self._legend.items[k][1].setText(f"{a}–{b}  {now:.3g}")
        self._value_curve.setData(t, np.asarray(self._value_hist))
        self._scale.setText(f"colour: {lo:.3g} → {hi:.3g}  ({self._metric})")

    def _color_range(self, edges: np.ndarray) -> tuple[float, float]:
        if self._range is not None:
            return self._range
        # The values on show, not every one seen: one outlier window no longer
        # dims every edge for the rest of the session.
        shown = np.concatenate([np.asarray(h, dtype=float) for h in self._hist] + [edges])
        shown = shown[np.isfinite(shown)]
        top = float(np.abs(shown).max()) if shown.size else 0.0
        top = top or 1.0
        # A signed metric gone negative is centred on 0.
        return (-top, top) if self._signed and shown.size and shown.min() < 0 else (0.0, top)


class ConnectivityPlot(QMainWindow):
    """Live connectivity: the pairs each connectivity modality computes.

    One tab per modality. Sensor-space pairs are drawn as edges between their
    sensors on the head, source-space ones between their ROIs on a circle, left
    hemisphere on the left; colour and width show each pair's current value.
    Beside them, each pair's value over the last ``history`` seconds, and the
    modality's own value, which the feedback is computed from.

    Parameters
    ----------
    ch_names : list of str | None
        The channels the session acquires, drawn as sensors on the head.
    info : mne.Info | None
        Their measurement info, for the sensor positions.
    history : float, default 30.0
        Seconds of each pair's values to show.
    rate : float, default 2.0
        Updates per second, for the trend's time axis: one per analysis hop.
    verbose : bool | str | None, default None
        Logging level.

    Notes
    -----
    :meth:`~mne_rt.RTStream.record_main` opens it with
    ``show_connectivity=True`` and feeds it every window. It shows the
    configured pairs only: computing every pair would cost the real-time loop
    more than the feedback does.

    .. versionadded:: 1.3.0
    """

    closed = Signal()

    def __init__(
        self,
        ch_names: Optional[list[str]] = None,
        info=None,
        history: float = 30.0,
        rate: float = 2.0,
        verbose=None,
    ) -> None:
        set_log_level(verbose)
        super().__init__()
        self.setWindowTitle("MNE-RT — Connectivity")
        self.setStyleSheet(f"QMainWindow, QWidget {{ background:{_BG}; color:{_TEXT}; }}")
        self._ch_names = list(ch_names) if ch_names is not None else None
        self._info = info
        self._history = float(history)
        self._rate = float(rate)
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)
        self._panels = {}  # modality name -> its _Panel (no annotation: docs mock QWidget)
        fit_to_screen(self, 1100, 520)

    def push(self, values: dict) -> None:
        """Show one window's per-pair values.

        Parameters
        ----------
        values : dict
            ``{modality name: snapshot}``, each snapshot a dict with ``value``
            (the modality's value), ``edges`` (one value per pair), ``labels``
            (the pairs, as names), ``metric``, ``signed``, ``space``
            (``"sensor"`` or ``"source"``) and ``nodes`` (a source modality's
            ROIs).
        """
        for name, snap in values.items():
            panel = self._panels.get(name)
            if panel is None:
                try:
                    panel = _Panel(
                        name, snap, self._ch_names, self._info, self._history, self._rate
                    )
                except Exception:
                    logger.warning("ConnectivityPlot: cannot draw %s.", name, exc_info=True)
                    continue
                self._panels[name] = panel
                self._tabs.addTab(panel, name)
            panel.show_values(snap)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
