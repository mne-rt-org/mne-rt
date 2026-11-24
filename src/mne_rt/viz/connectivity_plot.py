"""Live view of the connections a connectivity modality computes.

Classes
-------
ConnectivityPlot
    The pairs of each connectivity modality: on MNE's head (sensor space) or
    MNE-Connectivity's circle (source space), and each pair's value over time.
"""

from __future__ import annotations

import datetime
from collections import deque
from typing import Optional

import numpy as np
import pyqtgraph as pg
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mne_rt._logging import logger, set_log_level
from mne_rt.viz._layout import fit_to_screen
from mne_rt.viz._sensor_selector import sensor_positions
from mne_rt.viz.raw_plot import _QSS

_BG = "#0d0d1a"
_FG = "#b0b0c8"
_DIM = "#3a3a5c"
_AXIS = "#303050"
_TICKS = "#9090aa"
_BAD = "#ef4444"
# The trend plot: brighter than the other viewers' defaults, so the ticks, the
# frame and the legend read clearly against the dark background.
_SPINE = "#55557a"
_TICK_TEXT = "#c8c8e0"
_LABEL_TEXT = "#d8d8ec"
# One colour per pair, distinct on the dark background.
_PAIR_COLORS = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#fff176", "#4db6ac"]
_HEMI_COLORS = {-1: "#4fc3f7", 1: "#ffb74d", 0: "#8a8aa8"}  # left, right, neither
# Metrics bounded in [0, 1]; the signed ones span [-1, 1]; others scale to the data.
_UNIT_METRICS = {"coh", "plv", "ciplv", "pli", "wpli", "mic", "graph"}  # graph: normalised
_SIGNED_METRICS = {"imcoh", "corr", "cohy"}
_TIME_WINDOWS = [10, 30, 60, 120]


def value_range(metric: str, signed: bool) -> Optional[tuple[float, float]]:
    """The fixed colour range of a metric, or ``None`` to scale to the data."""
    if metric in _SIGNED_METRICS:
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    if metric in _UNIT_METRICS:
        return (0.0, 1.0)
    return None


def hemisphere(name: str, sides: Optional[dict] = None) -> int:
    """-1 for a left-hemisphere ROI, 1 for a right one, 0 for neither.

    From ``sides`` (the modality's, from each ROI's atlas labels) when it has
    the ROI; else read from the name, or for an alias such as ``"Broca"`` from
    its atlas labels, when they all lie in one hemisphere.
    """
    from mne_rt.source import ROI_ALIASES, _name_hemisphere

    if sides and name in sides:
        return int(sides[name])
    side = _name_hemisphere(name)
    if side:
        return side

    for aliases in ROI_ALIASES.values():
        if name in aliases:
            sides = {_name_hemisphere(member) for member in aliases[name]}
            if len(sides) == 1:
                return sides.pop()
    return 0


def circle_order(nodes: list[str], sides: Optional[dict] = None) -> list[str]:
    """ROIs in circle order: left hemisphere, then neither, then the right one
    mirrored, so homologous ROIs face each other across the circle."""
    side = {n: hemisphere(n, sides) for n in nodes}
    left = [n for n in nodes if side[n] < 0]
    mid = [n for n in nodes if side[n] == 0]
    right = [n for n in nodes if side[n] > 0]
    return left + mid + right[::-1]


def _short(name: str, n: int = 12) -> str:
    """A name cut to ``n`` characters for the side panel; the full one is its tooltip."""
    return name if len(name) <= n else name[: n - 1] + "…"


class _Series:
    """One modality's pairs and their values over time."""

    def __init__(self, first: dict, maxlen: int) -> None:
        self.labels = [tuple(p) for p in first["labels"]]
        self.metric = str(first["metric"])
        self.signed = bool(first["signed"])
        self.space = first["space"]
        # Unique, in order: circular_layout refuses a repeated node.
        self.nodes = list(dict.fromkeys(first["nodes"] or []))
        self.sides = dict(first.get("sides") or {})
        self.times = deque(maxlen=maxlen)
        self.hist = [deque(maxlen=maxlen) for _ in self.labels]
        self.value = deque(maxlen=maxlen)
        self.visible = [True] * len(self.labels)

    def add(self, snap: dict) -> None:
        # Each window at its own time, so a dropped one leaves a gap rather than
        # shifting everything after it.
        edges = np.asarray(snap["edges"], dtype=float).ravel()
        values = [edges[k] if k < edges.size else np.nan for k in range(len(self.labels))]
        value = float(snap["value"])
        self.times.append(float(snap.get("time", len(self.times))))
        for hist, v in zip(self.hist, values):
            hist.append(v)
        self.value.append(value)

    def view(self, until: Optional[float] = None):
        """Times, each pair's values and the modality's, up to ``until``, and the
        pairs' values at the last of those times."""
        t = np.asarray(self.times, dtype=float)
        hist = [np.asarray(h, dtype=float) for h in self.hist]
        value = np.asarray(self.value, dtype=float)
        if until is not None and t.size:
            keep = t <= until
            t, hist, value = t[keep], [h[keep] for h in hist], value[keep]
        edges = np.array([h[-1] if h.size else np.nan for h in hist])
        return t, hist, value, edges

    def color_range(self, window: float, until: Optional[float] = None) -> tuple[float, float]:
        """The colour range: the values on show, within the metric's own bounds.

        A fixed [-1, 1] made imcoh around ±0.2 one pale colour, and edges hardly
        changed; scaled to the values on show (not every value seen, so one
        outlier window does not dim the rest of the session), they do.
        """
        fixed = value_range(self.metric, self.signed)
        t, hist, _, edges = self.view(until)
        recent = t >= (t[-1] - window) if t.size else np.zeros(0, bool)
        # The pairs on show only: hiding a strong pair rescales the others.
        shown = [k for k, on in enumerate(self.visible) if on]
        shown = np.concatenate([hist[k][recent] for k in shown] + [edges[shown]])
        shown = shown[np.isfinite(shown)]
        top = float(np.abs(shown).max()) if shown.size else 0.0
        if fixed is not None:
            # Within the metric's bounds, and not below a tenth of them, so a
            # metric near 0 does not show its noise at full contrast.
            bound = max(abs(fixed[0]), abs(fixed[1]))
            top = float(np.clip(top, 0.1 * bound, bound))
            if fixed[0] < 0:  # a signed metric: always centred on 0
                return (-top, top)
        top = top or 1.0
        # A signed metric gone negative is centred on 0.
        return (-top, top) if self.signed and shown.size and shown.min() < 0 else (0.0, top)


class ConnectivityPlot(QMainWindow):
    """Live connectivity: the pairs each connectivity modality computes.

    A sensor-space modality's pairs are drawn as edges on MNE's head outline,
    as in :func:`mne.viz.plot_sensors`; a source-space one's on
    MNE-Connectivity's circle (:func:`mne_connectivity.viz.plot_connectivity_circle`),
    left hemisphere on the left. Colour and width show each pair's current
    value, read on the colour bar. Beside them, each pair's value over time,
    and the modality's own value, which the feedback is computed from. The
    side panel picks the modality, the time window and the pairs shown, and
    pauses the display (the data keeps coming in).

    Parameters
    ----------
    ch_names : list of str | None
        The channels the session acquires: those with a position in ``info``
        are drawn on the head, bad ones outlined in red. ``None`` draws every
        channel of ``info``.
    info : mne.Info | None
        Their measurement info, for the sensor positions. Without positions,
        the sensors are placed by a standard layout or on a circle.
    history : float, default 60.0
        Seconds of each pair's values shown at first; the side panel changes it.
    rate : float, default 2.0
        Updates per second, one per analysis hop: sizes the kept history.
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
        history: float = 60.0,
        rate: float = 2.0,
        verbose=None,
    ) -> None:
        set_log_level(verbose)
        super().__init__()
        self.setWindowTitle("MNE-RT — Connectivity")
        self.setStyleSheet(_QSS)
        self._ch_names = list(ch_names) if ch_names is not None else None
        self._info = info
        self._window = float(min(_TIME_WINDOWS, key=lambda w: abs(w - history)))
        self._maxlen = max(4, int(round(max(_TIME_WINDOWS) * rate * 2)))
        self._series: dict[str, _Series] = {}
        self._current: Optional[str] = None
        self._paused = False
        self._paused_at: dict[str, float] = {}
        self._show_value = True
        self._show_names = True
        self._head_for: Optional[tuple] = None  # what the head was drawn for
        self._circle_for: Optional[tuple] = None
        self._cbar_for: Optional[tuple] = None
        self._circle_edges: list = []
        self._edge_lines: list = []
        self._split_sizes: dict[str, list[int]] = {}
        self._warned: set[tuple[str, str]] = set()
        self._drawn_info = self._drawable_info()

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_panel(), stretch=0)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._build_space())
        self._splitter.addWidget(self._build_trend())
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        layout.addWidget(self._splitter, stretch=1)
        self.setCentralWidget(root)
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        # Several windows can arrive in one tick: one redraw for all of them.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(0)
        self._redraw_timer.timeout.connect(self._redraw)
        fit_to_screen(self, 1500, 640)

    def _drawable_info(self):
        """The channels ``plot_sensors`` draws: one channel type, with positions.

        It draws one type only, and channels without a position would sit at
        the centre; given just those, its dots follow this info's ``ch_names``.
        """
        if self._info is None:
            return None
        import mne

        wanted = set(self._ch_names) if self._ch_names is not None else None
        placed = [
            i
            for i, ch in enumerate(self._info["chs"])
            if (wanted is None or ch["ch_name"] in wanted)
            and np.all(np.isfinite(ch["loc"][:3]))
            and np.any(ch["loc"][:3])
        ]
        if not placed:
            return None
        types = self._info.get_channel_types(picks=placed)
        main = max(set(types), key=types.count)
        return mne.pick_info(self._info, [i for i, t in zip(placed, types) if t == main])

    # UI

    def _build_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(230)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setSpacing(8)
        lay.setContentsMargins(6, 6, 6, 6)

        grp = QGroupBox("Playback")
        g = QVBoxLayout(grp)
        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_pause.setCheckable(True)
        self._btn_pause.toggled.connect(self._set_paused)
        g.addWidget(self._btn_pause)
        lay.addWidget(grp)

        grp = QGroupBox("Modality")
        g = QVBoxLayout(grp)
        self._cmb_mod = QComboBox()
        self._cmb_mod.currentTextChanged.connect(self._select)
        g.addWidget(self._cmb_mod)
        self._lbl_metric = QLabel("")
        self._lbl_metric.setWordWrap(True)
        g.addWidget(self._lbl_metric)
        lay.addWidget(grp)

        grp = QGroupBox("Display")
        g = QVBoxLayout(grp)
        row = QHBoxLayout()
        row.addWidget(QLabel("Time window:"))
        self._cmb_window = QComboBox()
        for w in _TIME_WINDOWS:
            self._cmb_window.addItem(f"{w} s", w)
        self._cmb_window.setCurrentIndex(_TIME_WINDOWS.index(int(self._window)))
        self._cmb_window.currentIndexChanged.connect(self._set_window)
        row.addWidget(self._cmb_window)
        g.addLayout(row)
        self._chk_value = QCheckBox("Show modality value")
        self._chk_value.setChecked(True)
        self._chk_value.toggled.connect(self._set_show_value)
        g.addWidget(self._chk_value)
        self._chk_names = QCheckBox("Show channel names")
        self._chk_names.setChecked(True)
        self._chk_names.toggled.connect(self._set_show_names)
        g.addWidget(self._chk_names)
        lay.addWidget(grp)

        self._grp_pairs = QGroupBox("Pairs")
        self._pairs_lay = QVBoxLayout(self._grp_pairs)
        self._pair_checks: list[QCheckBox] = []
        lay.addWidget(self._grp_pairs)

        lay.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _build_space(self) -> QWidget:
        self._fig = Figure(figsize=(4.2, 4.6), facecolor=_BG)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumWidth(280)
        return self._canvas

    def _build_trend(self) -> QWidget:
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground(_BG)
        self._trend = glw.addPlot(row=0, col=0)
        self._trend.showGrid(x=True, y=True, alpha=0.12)
        self._trend.setMouseEnabled(x=False, y=False)
        self._trend.setMenuEnabled(False)
        self._trend.hideButtons()
        font = self._ui_font(10.5)
        # Left and bottom spines only; their width is set with the curves', for
        # the screen's pixel density.
        for name in ("left", "bottom"):
            ax = self._trend.getAxis(name)
            ax.setTextPen(pg.mkPen(_TICK_TEXT))
            ax.enableAutoSIPrefix(False)  # 0.7, not 700 x 0.001
            ax.setStyle(tickFont=font, tickLength=6, tickTextOffset=8)
        self._trend.getAxis("left").setWidth(70)
        self._trend.getAxis("bottom").setHeight(52)
        self._trend.setLabel(
            "bottom", "Time", units="s", color=_LABEL_TEXT, **{"font-size": "11pt"}
        )
        # Top left: the newest values are drawn at the right edge.
        self._legend = self._trend.addLegend(
            offset=(14, 14),
            labelTextSize="10.5pt",
            labelTextColor=_LABEL_TEXT,
            brush=pg.mkBrush(13, 13, 26, 200),
            verSpacing=-2,
            horSpacing=14,
        )
        self._curves: list = []
        self._value_curve = None
        self._legend_rows: dict = {}  # pair index, or "value" -> its legend label
        return glw

    @staticmethod
    def _ui_font(size: float) -> QFont:
        """The system's UI font at ``size`` points, antialiased and hinted."""
        from qtpy.QtWidgets import QApplication

        font = QFont(QApplication.font())
        font.setPointSizeF(size)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
        return font

    # Modality selection

    def _select(self, name: str) -> None:
        if not name or name not in self._series:
            return
        previous = self._series.get(self._current)
        if previous is not None:  # keep the panel widths set for that view
            self._split_sizes[previous.space] = self._splitter.sizes()
        self._current = name
        series = self._series[name]
        self._lbl_metric.setText(
            f"{series.metric} · {series.space} space · {len(series.labels)} pairs"
        )
        self._rebuild_pairs(series)
        self._rebuild_trend(series)
        self._head_for = self._circle_for = None
        sizes = self._split_sizes.get(series.space)
        if sizes is None:
            # A small head beside wide traces; the circle, with its ROI names, wider.
            total = max(sum(self._splitter.sizes()), 1)
            share = 0.4 if series.space == "source" else 0.26
            sizes = [int(total * share), int(total * (1 - share))]
        self._splitter.setSizes(sizes)
        self._redraw()

    def _rebuild_pairs(self, series: _Series) -> None:
        for chk in self._pair_checks:
            self._pairs_lay.removeWidget(chk)
            chk.deleteLater()
        self._pair_checks = []
        for k, (a, b) in enumerate(series.labels):
            # Short, so long ROI names do not widen the panel past its width.
            chk = QCheckBox(f"{_short(a)} – {_short(b)}")
            chk.setToolTip(f"{a} – {b}")
            chk.setChecked(series.visible[k])
            chk.setStyleSheet(f"color:{_PAIR_COLORS[k % len(_PAIR_COLORS)]};")
            chk.toggled.connect(lambda on, k=k: self._set_pair_visible(k, on))
            self._pairs_lay.addWidget(chk)
            self._pair_checks.append(chk)

    def _rebuild_trend(self, series: _Series) -> None:
        self._trend.clear()
        self._legend.clear()
        self._trend.setLabel("left", series.metric, color=_LABEL_TEXT, **{"font-size": "11pt"})
        self._curves = []
        # pyqtgraph's pens are sized in device pixels: scaled, they keep their
        # weight on a HiDPI screen rather than drawing at half of it.
        px = max(1.0, self.devicePixelRatioF())
        for name in ("left", "bottom"):
            self._trend.getAxis(name).setPen(pg.mkPen(_SPINE, width=2.0 * px))
        # Antialiased explicitly: a handful of curves updated twice a second, and
        # the global default is whatever the last viewer opened set.
        for k, (a, b) in enumerate(series.labels):
            pen = pg.mkPen(_PAIR_COLORS[k % len(_PAIR_COLORS)], width=2.2 * px)
            curve = self._trend.plot([], [], pen=pen, antialias=True)
            curve.setVisible(series.visible[k])
            self._curves.append(curve)
        self._value_curve = self._trend.plot(
            [],
            [],
            pen=pg.mkPen("#f0f0ff", width=1.8 * px, style=Qt.PenStyle.DashLine),
            antialias=True,
        )
        self._value_curve.setVisible(self._show_value)
        self._refresh_legend(series)
        zero = pg.mkPen(_SPINE, width=px, style=Qt.PenStyle.DotLine)
        self._trend.addItem(pg.InfiniteLine(pos=0, angle=0, pen=zero))

    def _refresh_legend(self, series: _Series) -> None:
        """The legend lists the curves on show: a hidden one leaves it."""
        self._legend.clear()
        self._legend_rows = {}
        for k, (a, b) in enumerate(series.labels):
            if series.visible[k]:
                self._legend.addItem(self._curves[k], f"{a} – {b}")
                self._legend_rows[k] = self._legend.items[-1][1]
                self._on_sample_click(self._pair_checks[k])
        if self._show_value:
            self._legend.addItem(self._value_curve, "modality value")
            self._legend_rows["value"] = self._legend.items[-1][1]
            self._on_sample_click(self._chk_value)

    def _on_sample_click(self, box: QCheckBox) -> None:
        """A click on a legend line hides it through its checkbox, so the side
        panel, the legend and the head or circle agree (pyqtgraph's own click
        hid the curve alone)."""
        sample = self._legend.items[-1][0]

        def click(ev, box=box):
            ev.accept()
            # After the event: unticking rebuilds the legend this sample is in.
            QTimer.singleShot(0, lambda: box.setChecked(False))

        sample.mouseClickEvent = click

    # Controls

    def _set_paused(self, on: bool) -> None:
        self._paused = on
        self._btn_pause.setText("▶  Resume" if on else "⏸  Pause")
        # Frozen at the newest window of each modality; the data keeps coming in.
        self._paused_at = {n: s.times[-1] for n, s in self._series.items() if s.times} if on else {}
        self._redraw()

    def _set_window(self, idx: int) -> None:
        self._window = float(self._cmb_window.itemData(idx))
        self._redraw()

    def _set_show_value(self, on: bool) -> None:
        self._show_value = on
        if self._value_curve is not None:
            self._value_curve.setVisible(on)
            series = self._series.get(self._current)
            if series is not None:
                self._refresh_legend(series)
                self._redraw()

    def _set_show_names(self, on: bool) -> None:
        self._show_names = on
        self._head_for = self._circle_for = None
        self._redraw()

    def _set_pair_visible(self, k: int, on: bool) -> None:
        series = self._series.get(self._current)
        if series is None:
            return
        series.visible[k] = on
        self._curves[k].setVisible(on)
        self._refresh_legend(series)
        self._head_for = self._circle_for = None
        self._redraw()

    # Data

    def push(self, values: dict) -> None:
        """Add one window's per-pair values; they are shown at the next event loop pass.

        Parameters
        ----------
        values : dict
            ``{modality name: snapshot}``, each snapshot a dict with ``value``
            (the modality's value), ``edges`` (one value per pair), ``labels``
            (the pairs, as names), ``metric``, ``signed``, ``space``
            (``"sensor"`` or ``"source"``), ``nodes`` (a source modality's
            ROIs), ``sides`` (their hemispheres, where the names do not say)
            and ``time`` (seconds, for the time axis).
        """
        for name, snap in values.items():
            # One modality's bad snapshot must not stop the others being shown.
            try:
                series = self._series.get(name)
                if series is None:
                    series = _Series(snap, self._maxlen)
                    self._series[name] = series
                    self._cmb_mod.addItem(name)  # the first one is selected, and drawn
                series.add(snap)
                if self._paused and name not in self._paused_at:
                    self._paused_at[name] = series.times[-1]  # paused before it came
            except Exception:
                self._warn_once(name, "cannot add its values")
        if not self._paused and not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def _warn_once(self, name: str, what: str) -> None:
        if (name, what) not in self._warned:
            self._warned.add((name, what))
            logger.warning("ConnectivityPlot: %s: %s.", name, what, exc_info=True)

    # Drawing

    def _redraw(self) -> None:
        series = self._series.get(self._current)
        if series is None or not series.times:
            return
        until = self._paused_at.get(self._current) if self._paused else None
        t, hist, value, edges = series.view(until)
        if not t.size:
            return
        try:
            self._draw_trend(series, t, hist, value, edges)
        except Exception:
            self._warn_once(self._current, "cannot draw its values over time")
        try:
            self._draw_space(series, edges, until)
        except Exception:
            self._warn_once(self._current, "cannot draw its pairs")
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        values = "   ".join(
            f"{a}–{b} {v:.3g}" for (a, b), v in zip(series.labels, edges) if np.isfinite(v)
        )
        state = "Paused" if self._paused else "Updated"
        self._status.showMessage(f"{state} {stamp}   {values}   value {value[-1]:.3g}")

    def _draw_trend(self, series, t, hist, value, edges) -> None:
        x = t - t[-1]
        for k, curve in enumerate(self._curves):
            curve.setData(x, hist[k])
            if k in self._legend_rows:
                a, b = series.labels[k]
                now = edges[k]
                text = f"{a} – {b}   {now:.3g}" if np.isfinite(now) else f"{a} – {b}"
                self._legend_rows[k].setText(text)
        self._value_curve.setData(x, value)
        if "value" in self._legend_rows:
            self._legend_rows["value"].setText(f"modality value   {value[-1]:.3g}")
        self._trend.setXRange(-self._window, 0, padding=0.01)

    def _draw_space(self, series: _Series, edges: np.ndarray, until) -> None:
        lo, hi = series.color_range(self._window, until)
        signed = lo < 0
        cmap = colormaps["RdBu_r" if signed else "viridis"]
        norm = Normalize(lo, hi)
        # The head or circle is drawn when what it shows changes; every window
        # only recolours and resizes the edges.
        # Each key is set once its drawing succeeded: one that failed is tried
        # again at the next window.
        key = (self._current, tuple(series.visible), self._show_names)
        if series.space == "source":
            if key != self._circle_for:
                self._cbar_for = None
                self._draw_circle(series)
                self._circle_for = key
            artists = self._circle_edges
        else:
            if key != self._head_for:
                self._cbar_for = None
                self._draw_head(series)
                self._head_for = key
            artists = self._edge_lines
        self._update_edges(artists, edges, cmap, norm, signed)
        if (lo, hi, series.metric) != self._cbar_for:  # its ticks rebuilt on a change only
            self._colorbar.update_normal(ScalarMappable(norm=norm, cmap=cmap))
            self._colorbar.set_label(series.metric, color=_FG)
            self._cbar_for = (lo, hi, series.metric)
        self._canvas.draw_idle()

    def _new_axes(self, **kwargs):
        self._fig.clear()
        ax = self._fig.add_axes((0.04, 0.16, 0.92, 0.82), facecolor=_BG, **kwargs)
        cax = self._fig.add_axes((0.2, 0.08, 0.6, 0.03))
        self._colorbar = self._fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap=colormaps["viridis"]),
            cax=cax,
            orientation="horizontal",
        )
        self._colorbar.outline.set_edgecolor(_AXIS)
        cax.tick_params(colors=_TICKS, labelsize=8)
        return ax

    def _sensor_layout(self, ax) -> dict:
        """Draw the head and sensors, and return each drawn channel's position."""
        info = self._drawn_info
        if info is not None:
            import mne

            mne.viz.plot_sensors(
                info, kind="topomap", axes=ax, show=False, pointsize=14, linewidth=0
            )
            for line in ax.lines:  # the head outline, ears and nose
                line.set_color(_TICKS)
                line.set_linewidth(1.2)
            dots = ax.collections[0]
            dots.set_picker(None)  # its click handler removes ax.texts[0]: our names
            names = list(info["ch_names"])
            pos = dots.get_offsets()
            if len(pos) == len(names):
                bads = set(info["bads"])
                dots.set_facecolor(_DIM)
                dots.set_edgecolor([_BAD if n in bads else "none" for n in names])
                dots.set_linewidth(1.5)
                return {n: pos[i] for i, n in enumerate(names)}
            ax.clear()
            ax.set_facecolor(_BG)
        # No positions to go by: a standard layout, or a circle, inside a plain head.
        names = list(self._ch_names or (self._info["ch_names"] if self._info is not None else []))
        if not names:
            return {}
        xy = np.array(sensor_positions(names, self._info, montage=None), dtype=float)
        xy = np.c_[xy[:, 0] - 0.5, 0.5 - xy[:, 1]]  # y up, centred
        ax.add_patch(Circle((0, 0), 0.5, fill=False, edgecolor=_TICKS, linewidth=1.2))
        ax.scatter(xy[:, 0], xy[:, 1], s=14, c=_DIM, linewidths=0)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.65)
        ax.set_aspect("equal")
        return {n: xy[i] for i, n in enumerate(names)}

    def _draw_head(self, series: _Series) -> None:
        """MNE's head and sensors, as :func:`mne.viz.plot_sensors` draws them."""
        ax = self._new_axes()
        where = self._sensor_layout(ax)
        self._edge_lines = []
        for k, (a, b) in enumerate(series.labels):
            if a not in where or b not in where:
                self._edge_lines.append(None)
                continue
            (line,) = ax.plot(
                [where[a][0], where[b][0]],
                [where[a][1], where[b][1]],
                solid_capstyle="round",
                zorder=3,
                visible=series.visible[k],
            )
            self._edge_lines.append(line)
        if self._show_names:
            involved = {
                n for k, pair in enumerate(series.labels) if series.visible[k] for n in pair
            }
            for n in sorted(involved & set(where)):
                ax.annotate(
                    n,
                    where[n],
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    color=_FG,
                    fontsize=8,
                    zorder=4,
                )
        ax.set_axis_off()

    @staticmethod
    def _update_edges(artists: list, edges: np.ndarray, cmap, norm, signed: bool) -> None:
        """Each edge's colour and width from its value: wide and saturated when strong."""
        for artist, v in zip(artists, edges):
            if artist is None:
                continue
            if not np.isfinite(v):
                artist.set_color(_DIM)
                artist.set_linewidth(1.0)
                continue
            frac = float(np.clip(norm(v), 0.0, 1.0))
            strength = abs(frac - 0.5) * 2 if signed else frac
            artist.set_color(cmap(frac))
            artist.set_linewidth(1.5 + 6.0 * strength)

    def _draw_circle(self, series: _Series) -> None:
        """MNE-Connectivity's circle of the atlas's ROIs, homologous ones facing
        each other, and the modality's pairs as edges across it."""
        from matplotlib.colors import to_rgba
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        nodes = series.nodes or list(dict.fromkeys(n for pair in series.labels for n in pair))
        ax = self._new_axes(projection="polar")
        self._circle_edges = []
        if len(nodes) < 2:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no ROIs to draw", color=_TICKS, ha="center", transform=ax.transAxes)
            return
        import mne
        from mne_connectivity.viz import plot_connectivity_circle

        sides = series.sides
        order = circle_order(nodes, sides)
        n_left = sum(hemisphere(n, sides) < 0 for n in order)
        n_mid = sum(hemisphere(n, sides) == 0 for n in order)
        bounds = sorted({0} | {b for b in (n_left, n_left + n_mid) if 0 < b < len(order)})
        angles = mne.viz.circular_layout(nodes, order, start_pos=90, group_boundaries=bounds)
        # Every ROI of the atlas; those in a shown pair stand out.
        involved = {n for k, pair in enumerate(series.labels) if series.visible[k] for n in pair}
        colors = [
            to_rgba(_HEMI_COLORS[hemisphere(n, sides)], 1.0 if n in involved else 0.35)
            for n in nodes
        ]
        n = len(nodes)
        plot_connectivity_circle(
            np.zeros(0),  # the edges are drawn below, to restyle them every window
            nodes,
            indices=(np.zeros(0, int), np.zeros(0, int)),
            node_angles=angles,
            node_colors=colors,
            facecolor=_BG,
            textcolor=_FG,
            node_edgecolor=_BG,
            vmin=0.0,  # without edges, MNE would take them from none
            vmax=1.0,
            colorbar=False,
            fontsize_names=8 if n <= 30 else 7 if n <= 70 else 6,
            padding=12.0 if n <= 70 else 16.0,
            ax=ax,
            interactive=False,
            show=False,
        )
        for text, name in zip(ax.texts, nodes):
            text.set_visible(self._show_names)
            if name in involved:
                text.set_color("#f0f0ff")
                text.set_fontweight("bold")
            else:
                text.set_color("#6a6a88")
        # The circle plot's own edge shape: a Bezier from node to node through
        # half the radius.
        theta = np.deg2rad(angles)
        where = {name: theta[i] for i, name in enumerate(nodes)}
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        for k, (a, b) in enumerate(series.labels):
            if not series.visible[k] or a not in where or b not in where:
                self._circle_edges.append(None)
                continue
            t0, t1 = where[a], where[b]
            path = Path([(t0, 9.0), (t0, 5.0), (t1, 5.0), (t1, 9.0)], codes)
            patch = PathPatch(path, fill=False, capstyle="round", zorder=3)
            ax.add_patch(patch)
            self._circle_edges.append(patch)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
