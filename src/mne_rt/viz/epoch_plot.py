"""Real-time continuous M/EEG display with epoch/trigger markers.

Shows a scrolling multi-channel raw signal overlaid with event markers:

* Solid vertical line at each trigger (t = 0 of the epoch)
* Semi-transparent shaded region spanning [tmin, tmax] around each trigger
* Dashed boundary lines at the epoch edges (tmin, tmax)

tmin / tmax are adjustable live in the sidebar.  Different event codes
are assigned distinct colours via the ``event_id`` mapping.

Classes
-------
EpochPlot
    Scrolling raw viewer with epoch / trigger overlays.
"""
from __future__ import annotations

import datetime
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

_TRACE_COLORS = [
    "#4fc3f7", "#ef9a9a", "#a5d6a7", "#fff176", "#ffab91",
    "#ce93d8", "#80cbc4", "#ffcc80", "#80deea", "#b39ddb",
    "#f48fb1", "#c5e1a5", "#ffd54f", "#81d4fa", "#dce775",
    "#ff8a65", "#90caf9", "#e6ee9c", "#bcaaa4", "#ffe082",
]

# Trigger colours: red, green, cyan, yellow, magenta, orange
_EVENT_COLORS = [
    "#69f0ae", "#40c4ff", "#ffff00", "#ff9e80", "#ea80fc", "#80cbc4",
]

_TIME_WINDOW_OPTIONS = [2, 5, 10, 20]

_QSS = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", sans-serif;
}
QPushButton {
    background-color: #16213e;
    color: #d0d0e8;
    border: 1px solid #0f3460;
    border-radius: 5px;
    padding: 5px 10px;
    font-size: 12px;
}
QPushButton:hover  { background-color: #0f3460; }
QPushButton:pressed { background-color: #533483; }
QPushButton:checked {
    background-color: #533483;
    border-color: #a882dd;
    color: #ffffff;
}
QComboBox {
    background-color: #16213e;
    color: #d0d0e8;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 3px 6px;
}
QDoubleSpinBox, QSpinBox {
    background-color: #16213e;
    color: #d0d0e8;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 2px 4px;
}
QGroupBox {
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: bold;
    font-size: 11px;
    color: #8888aa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QLabel  { color: #b0b0c8; font-size: 11px; }
QCheckBox { color: #b0b0c8; font-size: 11px; }
QScrollArea { border: none; }
QStatusBar { background-color: #0d0d1a; color: #606080; font-size: 10px; }
QScrollBar:vertical {
    background-color: #0d0d1a;
    width: 14px;
    border: none;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background-color: #2a2a4a;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background-color: #404060; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background-color: #0d0d1a;
}
"""


# ---------------------------------------------------------------------------
# Wheel-event filter (same as RawPlot)
# ---------------------------------------------------------------------------

class _WheelFilter(QObject):
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._cb = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            self._cb(1 if event.angleDelta().y() > 0 else -1)
            return True
        return False


# ---------------------------------------------------------------------------
# EpochPlot
# ---------------------------------------------------------------------------

class EpochPlot(QMainWindow):
    """Real-time scrolling M/EEG viewer with epoch / trigger overlays.

    Displays all channels stacked vertically (identical layout to
    :class:`~mne_rt.viz.RawPlot`) with coloured event markers overlaid on
    the signal.  For each trigger event a solid vertical line marks t = 0,
    a semi-transparent shaded band spans the epoch window [tmin, tmax], and
    dashed boundary lines sit at the epoch edges.

    Parameters
    ----------
    ch_names : list of str
        Channel names.  One row per channel.
    sfreq : float
        Sampling frequency in Hz.
    tmin : float, default -0.1
        Epoch start in seconds relative to each trigger.
    tmax : float, default 0.5
        Epoch end in seconds relative to each trigger.
    n_shown : int, default 20
        Number of channels visible simultaneously.
    time_window : float, default 10.0
        Visible time range in seconds at startup.
    scale_uv : float, default 100.0
        Amplitude scale in µV (half of per-channel row height).
    event_id : dict[str, int] | None, default None
        Maps condition names to integer trigger codes.  Each code gets a
        distinct colour; unmapped codes use the first colour.
    info : mne.Info | None, default None
        If provided, used for channel-type detection and right-click sensor
        position.
    verbose : bool | str | None, default None

    See Also
    --------
    mne_rt.viz.RawPlot : Continuous raw viewer without epoch overlays.
    mne_rt.RTEpochs : Event-triggered epoch accumulator.

    Notes
    -----
    Feed data with :meth:`push` (shape ``(n_ch, n_times)``) and trigger
    events with :meth:`push_trigger`.  Both calls are safe to make from
    an acquisition thread.

    .. versionadded:: 1.1.0
    """

    def __init__(
        self,
        ch_names: list[str],
        sfreq: float,
        tmin: float = -0.1,
        tmax: float = 0.5,
        n_shown: int = 20,
        time_window: float = 10.0,
        scale_uv: float = 100.0,
        event_id: dict[str, int] | None = None,
        info=None,
        verbose=None,
    ) -> None:
        from mne_rt._logging import set_log_level
        set_log_level(verbose)
        super().__init__()

        self._ch_names   = list(ch_names)
        self._n_ch       = len(ch_names)
        self._sfreq      = float(sfreq)
        self._tmin       = float(tmin)
        self._tmax       = float(tmax)
        self._n_shown    = min(int(n_shown), self._n_ch)
        self._time_window = float(time_window)
        self._scale      = float(scale_uv) * 1e-6
        self._event_id   = dict(event_id) if event_id else {}
        self._info       = info
        self._page_start = 0
        self._paused     = False

        # Stream state
        self._total_pushed: int = 0
        n_pts = max(int(sfreq * time_window), 30)
        self._time_axis = np.linspace(0.0, time_window, n_pts)
        self._buf = np.zeros((self._n_ch, n_pts))

        # Trigger history: (abs_sample_idx, event_code)
        self._triggers: deque[tuple[int, int]] = deque(maxlen=500)
        # Overlay items currently on the plot (cleared each redraw)
        self._epoch_overlay_items: list = []

        # Thread-safe pending queue: ('data', ndarray) | ('trigger', int)
        # push()/push_trigger() enqueue here (any thread); _process_pending()
        # drains it in the main Qt thread at 30 Hz via a QTimer.
        self._pending: deque = deque()

        # Build colour map: event_code → colour string
        self._code_colors: dict[int, str] = {}
        for i, (name, code) in enumerate(self._event_id.items()):
            self._code_colors[code] = _EVENT_COLORS[i % len(_EVENT_COLORS)]
        self._default_color = _EVENT_COLORS[0]

        # Per-channel colours
        self._ch_colors = [
            _TRACE_COLORS[i % len(_TRACE_COLORS)] for i in range(self._n_ch)
        ]

        pg.setConfigOptions(antialias=True, foreground="#c0c0d8", background="#0d0d1a")
        self._build_ui()
        self.setWindowTitle("MNE-RT — Epoch Viewer")
        self.resize(1500, 720)

        # 30 Hz render timer — processes queued data in the main Qt thread.
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(33)
        self._render_timer.timeout.connect(self._process_pending)
        self._render_timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setStyleSheet(_QSS)
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 4, 8)
        root.setSpacing(0)

        root.addWidget(self._build_plot_widget(), stretch=5)

        self._ch_scroll = QScrollBar(Qt.Orientation.Vertical)
        self._ch_scroll.setRange(0, max(0, self._n_ch - self._n_shown))
        self._ch_scroll.setPageStep(self._n_shown)
        self._ch_scroll.setSingleStep(1)
        self._ch_scroll.setFixedWidth(14)
        self._ch_scroll.valueChanged.connect(
            lambda v: self._set_page_start(v, source="scrollbar")
        )
        root.addWidget(self._ch_scroll)

        root.addSpacing(4)
        root.addWidget(self._build_control_panel(), stretch=0)

        self._status = self.statusBar()
        self._status.showMessage("Waiting for data …")

    def _build_plot_widget(self) -> pg.GraphicsLayoutWidget:
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground("#0d0d1a")

        self._pi = glw.addPlot(row=0, col=0)
        self._pi.setMouseEnabled(x=False, y=False)
        vb = self._pi.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setMenuEnabled(False)
        self._pi.showGrid(x=True, y=False, alpha=0.25)

        for ax_name in ("left", "bottom"):
            ax = self._pi.getAxis(ax_name)
            ax.setPen(pg.mkPen("#303050"))
            ax.setTextPen(pg.mkPen("#9090aa"))
        self._pi.getAxis("left").setWidth(80)
        self._pi.setLabel("bottom", "Time", units="s", color="#9090aa")
        self._pi.setXRange(0.0, self._time_window, padding=0.01)
        self._pi.setYRange(-0.5, self._n_shown - 0.5, padding=0)

        self._curves: list[pg.PlotCurveItem] = [
            pg.PlotCurveItem(pen=pg.mkPen(color=_TRACE_COLORS[0], width=1))
            for _ in range(self._n_shown)
        ]
        for c in self._curves:
            self._pi.addItem(c)

        sep_pen = pg.mkPen(color=(70, 70, 110, 55), width=1, style=Qt.PenStyle.DotLine)
        for i in range(self._n_shown):
            self._pi.addItem(pg.InfiniteLine(pos=i, angle=0, pen=sep_pen))

        self._update_tick_labels()

        self._wheel_filter = _WheelFilter(self._on_plot_wheel, self)
        glw.viewport().installEventFilter(self._wheel_filter)

        glw.scene().sigMouseClicked.connect(self._on_scene_clicked)

        self._glw = glw
        return glw

    def _build_control_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(230)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(self._grp_playback())
        layout.addWidget(self._grp_amplitude())
        layout.addWidget(self._grp_display())
        layout.addWidget(self._grp_epoch())
        layout.addWidget(self._grp_events())
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    # ── sidebar groups ─────────────────────────────────────────────────

    def _grp_playback(self) -> QGroupBox:
        grp = QGroupBox("Playback")
        lay = QVBoxLayout(grp)
        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_pause.setCheckable(True)
        self._btn_pause.clicked.connect(self._toggle_pause)
        btn_clear = QPushButton("⟳  Clear")
        btn_clear.clicked.connect(self._clear)
        btn_shot = QPushButton("📷  Screenshot")
        btn_shot.clicked.connect(self._screenshot)
        for w in (self._btn_pause, btn_clear, btn_shot):
            lay.addWidget(w)
        return grp

    def _grp_amplitude(self) -> QGroupBox:
        grp = QGroupBox("Amplitude")
        lay = QVBoxLayout(grp)
        self._scale_lbl = QLabel(self._fmt_scale())
        self._scale_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scale_lbl.setStyleSheet("color:#7ec8e3; font-size:12px; font-weight:bold;")
        row = QHBoxLayout()
        btn_dn = QPushButton("÷2"); btn_up = QPushButton("×2")
        for b in (btn_dn, btn_up):
            b.setFixedSize(42, 26)
        btn_dn.clicked.connect(self._scale_down)
        btn_up.clicked.connect(self._scale_up)
        row.addStretch(); row.addWidget(btn_dn); row.addWidget(btn_up); row.addStretch()
        lay.addWidget(self._scale_lbl); lay.addLayout(row)
        return grp

    def _grp_display(self) -> QGroupBox:
        grp = QGroupBox("Display")
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        row.addWidget(QLabel("Time window:"))
        from qtpy.QtWidgets import QComboBox
        self._cmb_tw = QComboBox()
        for secs in _TIME_WINDOW_OPTIONS:
            self._cmb_tw.addItem(f"{secs} s", secs)
        best = min(_TIME_WINDOW_OPTIONS, key=lambda s: abs(s - self._time_window))
        self._cmb_tw.setCurrentIndex(_TIME_WINDOW_OPTIONS.index(best))
        self._cmb_tw.currentIndexChanged.connect(self._change_time_window)
        row.addWidget(self._cmb_tw); lay.addLayout(row)
        chk_grid = QCheckBox("Show grid")
        chk_grid.setChecked(True)
        chk_grid.toggled.connect(
            lambda on: self._pi.showGrid(x=on, y=False, alpha=0.25 if on else 0.0)
        )
        lay.addWidget(chk_grid)
        return grp

    def _grp_epoch(self) -> QGroupBox:
        grp = QGroupBox("Epoch Window")
        lay = QVBoxLayout(grp); lay.setSpacing(5)

        tmin_row = QHBoxLayout()
        tmin_row.addWidget(QLabel("tmin:"))
        self._tmin_spin = QDoubleSpinBox()
        self._tmin_spin.setRange(-5.0, 0.0)
        self._tmin_spin.setValue(self._tmin)
        self._tmin_spin.setSuffix(" s")
        self._tmin_spin.setDecimals(2)
        self._tmin_spin.setSingleStep(0.05)
        tmin_row.addWidget(self._tmin_spin)
        lay.addLayout(tmin_row)

        tmax_row = QHBoxLayout()
        tmax_row.addWidget(QLabel("tmax:"))
        self._tmax_spin = QDoubleSpinBox()
        self._tmax_spin.setRange(0.0, 5.0)
        self._tmax_spin.setValue(self._tmax)
        self._tmax_spin.setSuffix(" s")
        self._tmax_spin.setDecimals(2)
        self._tmax_spin.setSingleStep(0.05)
        tmax_row.addWidget(self._tmax_spin)
        lay.addLayout(tmax_row)

        btn_apply = QPushButton("Apply")
        btn_apply.setStyleSheet(
            "background:#132744; color:#80d8ff; border:1px solid #2a6090;"
            "border-radius:4px; padding:4px; font-size:11px;"
        )
        btn_apply.clicked.connect(self._apply_epoch_window)
        lay.addWidget(btn_apply)

        self._epoch_lbl = QLabel(f"Window: {self._tmin:.2f} → {self._tmax:.2f} s")
        self._epoch_lbl.setStyleSheet("color:#80d8ff; font-size:10px;")
        self._epoch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._epoch_lbl)

        chk_region = QCheckBox("Show epoch region")
        chk_region.setChecked(True)
        chk_region.toggled.connect(self._set_show_epoch_region)
        lay.addWidget(chk_region)
        self._show_epoch_region: bool = True

        chk_bounds = QCheckBox("Show tmin/tmax lines")
        chk_bounds.setChecked(True)
        chk_bounds.toggled.connect(self._set_show_epoch_bounds)
        lay.addWidget(chk_bounds)
        self._show_epoch_bounds: bool = True

        return grp

    def _grp_events(self) -> QGroupBox:
        grp = QGroupBox("Events")
        lay = QVBoxLayout(grp); lay.setSpacing(4)

        self._event_count_lbl = QLabel("No triggers received")
        self._event_count_lbl.setStyleSheet("color:#505070; font-size:10px;")
        self._event_count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._event_count_lbl)

        btn_clear_trigs = QPushButton("Clear triggers")
        btn_clear_trigs.clicked.connect(self._clear_triggers)
        lay.addWidget(btn_clear_trigs)

        # Legend: one colour swatch per event code
        if self._event_id:
            lay.addWidget(QLabel("─── Legend ───"))
            for name, code in self._event_id.items():
                color = self._code_colors.get(code, self._default_color)
                legend_row = QHBoxLayout()
                swatch = QLabel("■")
                swatch.setStyleSheet(f"color:{color}; font-size:14px;")
                legend_row.addWidget(swatch)
                legend_row.addWidget(QLabel(f"{name} (code {code})"))
                legend_row.addStretch()
                lay.addLayout(legend_row)

        return grp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fmt_scale(self) -> str:
        uv = self._scale * 1e6
        if uv >= 1000: return f"{uv / 1000:.4g} mV / row"
        if uv >= 1:    return f"{uv:.4g} µV / row"
        return f"{uv * 1000:.4g} nV / row"

    def _update_tick_labels(self) -> None:
        end = min(self._page_start + self._n_shown, self._n_ch)
        n_actual = end - self._page_start
        ticks = [
            (n_actual - 1 - i, self._ch_names[self._page_start + i])
            for i in range(n_actual)
        ]
        self._pi.getAxis("left").setTicks([ticks, []])

    def _set_page_start(self, new_start: int, source: str = "other") -> None:
        new_start = max(0, min(new_start, max(0, self._n_ch - self._n_shown)))
        if new_start == self._page_start:
            return
        self._page_start = new_start
        self._update_tick_labels()
        if source != "scrollbar":
            self._ch_scroll.blockSignals(True)
            self._ch_scroll.setValue(new_start)
            self._ch_scroll.blockSignals(False)
        self._redraw()

    def _trigger_color(self, code: int) -> str:
        return self._code_colors.get(code, self._default_color)

    # ------------------------------------------------------------------
    # Callbacks — scroll / wheel / click
    # ------------------------------------------------------------------

    def _on_plot_wheel(self, direction: int) -> None:
        step = max(1, self._n_shown // 4)
        self._set_page_start(self._page_start - direction * step, source="wheel")

    def _on_scene_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.RightButton:
            return
        pos = event.scenePos()
        axis = self._pi.getAxis("left")
        if not axis.sceneBoundingRect().contains(pos):
            return
        y_val = self._pi.getViewBox().mapSceneToView(pos).y()
        end = min(self._page_start + self._n_shown, self._n_ch)
        n_actual = end - self._page_start
        vis_idx = int(round(n_actual - 1 - y_val))
        if 0 <= vis_idx < n_actual:
            ch_idx = self._page_start + vis_idx
            self._show_channel_location(self._ch_names[ch_idx])

    def _show_channel_location(self, ch_name: str) -> None:
        if self._info is None:
            self._status.showMessage(f"No Info — cannot show position for {ch_name}")
            return
        try:
            import mne, matplotlib.pyplot as plt
            fig = mne.viz.plot_sensors(
                self._info, show_names=True,
                title=f"Sensor position — {ch_name}", show=False,
            )
            for ax in fig.axes:
                for txt in ax.texts:
                    if txt.get_text() == ch_name:
                        txt.set_color("#cc0000"); txt.set_fontsize(10)
                        txt.set_fontweight("bold")
            plt.show(block=False)
        except Exception as exc:
            self._status.showMessage(f"Could not show sensor position: {exc}")

    # ------------------------------------------------------------------
    # Callbacks — playback / display
    # ------------------------------------------------------------------

    def _toggle_pause(self, checked: bool) -> None:
        self._paused = checked
        self._btn_pause.setText("▶  Resume" if checked else "⏸  Pause")

    def _clear(self) -> None:
        self._pending.clear()
        self._buf[:] = 0.0
        self._triggers.clear()
        self._epoch_overlay_items.clear()
        self._total_pushed = 0
        self._redraw()

    def _screenshot(self) -> None:
        from qtpy.QtWidgets import QFileDialog
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(Path.home() / f"epoch_plot_{ts}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", default, "PNG Image (*.png)"
        )
        if not path: return
        exp = pg.exporters.ImageExporter(self._glw.scene())
        exp.parameters()["width"] = 1920
        exp.export(path)

    def _scale_up(self) -> None:
        self._scale *= 2.0
        self._scale_lbl.setText(self._fmt_scale())
        self._redraw()

    def _scale_down(self) -> None:
        self._scale /= 2.0
        self._scale_lbl.setText(self._fmt_scale())
        self._redraw()

    def _change_time_window(self, idx: int) -> None:
        secs = float(self._cmb_tw.itemData(idx))
        self._time_window = secs
        n_pts = max(int(self._sfreq * secs), 30)
        self._time_axis = np.linspace(0.0, secs, n_pts)
        self._buf = np.zeros((self._n_ch, n_pts))
        # _total_pushed intentionally kept — triggers remain valid relative to stream
        self._pi.setXRange(0.0, secs, padding=0.01)

    def _apply_epoch_window(self) -> None:
        self._tmin = self._tmin_spin.value()
        self._tmax = self._tmax_spin.value()
        self._epoch_lbl.setText(f"Window: {self._tmin:.2f} → {self._tmax:.2f} s")
        self._redraw()

    def _set_show_epoch_region(self, on: bool) -> None:
        self._show_epoch_region = on
        self._redraw()

    def _set_show_epoch_bounds(self, on: bool) -> None:
        self._show_epoch_bounds = on
        self._redraw()

    def _clear_triggers(self) -> None:
        self._triggers.clear()
        self._event_count_lbl.setText("No triggers received")
        self._event_count_lbl.setStyleSheet("color:#505070; font-size:10px;")
        self._redraw()

    # ------------------------------------------------------------------
    # Epoch overlay rendering
    # ------------------------------------------------------------------

    def _redraw_epoch_overlays(self) -> None:
        for item in self._epoch_overlay_items:
            self._pi.removeItem(item)
        self._epoch_overlay_items.clear()

        if not self._triggers:
            return

        buf_size   = self._buf.shape[1]
        buf_start  = self._total_pushed - buf_size   # absolute sample of leftmost buf column
        dash_style = Qt.PenStyle.DashLine

        for (trig_abs, code) in self._triggers:
            # x coordinate of the trigger (t=0) in the current view
            x0 = (trig_abs - buf_start) / self._sfreq
            # Accept if the epoch window overlaps the visible range
            x_lo = x0 + self._tmin
            x_hi = x0 + self._tmax
            if x_hi < 0 or x_lo > self._time_window:
                continue

            color = self._trigger_color(code)

            # ── solid trigger line at t=0 ──────────────────────────────
            trig_line = pg.InfiniteLine(
                pos=x0, angle=90,
                pen=pg.mkPen(color=color, width=2),
            )
            self._pi.addItem(trig_line)
            self._epoch_overlay_items.append(trig_line)

            # ── shaded epoch region ────────────────────────────────────
            if self._show_epoch_region:
                r, g, b = self._hex_to_rgb(color)
                region = pg.LinearRegionItem(
                    values=(x_lo, x_hi),
                    brush=pg.mkBrush(r, g, b, 30),
                    movable=False,
                    pen=pg.mkPen(None),      # no border line on the region itself
                )
                self._pi.addItem(region)
                self._epoch_overlay_items.append(region)

            # ── dashed epoch boundary lines ────────────────────────────
            if self._show_epoch_bounds:
                for x_bnd in (x_lo, x_hi):
                    if 0 <= x_bnd <= self._time_window:
                        bnd_line = pg.InfiniteLine(
                            pos=x_bnd, angle=90,
                            pen=pg.mkPen(color=color, width=1, style=dash_style),
                        )
                        self._pi.addItem(bnd_line)
                        self._epoch_overlay_items.append(bnd_line)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    # ------------------------------------------------------------------
    # Redraw
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        end      = min(self._page_start + self._n_shown, self._n_ch)
        visible  = list(range(self._page_start, end))
        n_actual = len(visible)
        gain     = 1.0 / (self._scale + 1e-300)

        for vis_idx, ch_idx in enumerate(visible):
            raw    = self._buf[ch_idx].copy()
            color  = self._ch_colors[ch_idx]
            self._curves[vis_idx].setPen(pg.mkPen(color=color, width=1))
            offset = float(n_actual - 1 - vis_idx)
            self._curves[vis_idx].setData(self._time_axis, offset + raw * gain)

        for vis_idx in range(n_actual, self._n_shown):
            self._curves[vis_idx].setData([], [])

        self._redraw_epoch_overlays()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(self, data: np.ndarray) -> None:
        """Enqueue a raw data chunk for display.

        Thread-safe: may be called from any thread.  The data is written into
        the circular buffer and rendered by the main-thread timer at ~30 Hz.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            New raw data chunk.  No-op when paused.
        """
        if self._paused:
            return
        self._pending.append(('data', data.copy()))

    def push_trigger(self, code: int = 1) -> None:
        """Enqueue a trigger event at the current stream position.

        Thread-safe: may be called from any thread.  The trigger is placed
        after all data chunks already in the queue, so the sample index is
        computed correctly in the main thread.

        Parameters
        ----------
        code : int, default 1
            Integer event code matched against :attr:`event_id`.
        """
        self._pending.append(('trigger', code))

    def _process_pending(self) -> None:
        """Drain the pending queue and redraw — called in the main thread at 30 Hz."""
        if not self._pending:
            return
        changed = False
        while self._pending:
            kind, payload = self._pending.popleft()
            if kind == 'data':
                n = payload.shape[1]
                self._buf = np.roll(self._buf, -n, axis=1)
                self._buf[:, -n:] = payload
                self._total_pushed += n
                changed = True
            else:  # trigger
                self._triggers.append((self._total_pushed, payload))
                n_t = len(self._triggers)
                self._event_count_lbl.setText(
                    f"{n_t} trigger{'s' if n_t != 1 else ''} received"
                )
                self._event_count_lbl.setStyleSheet("color:#80d8ff; font-size:10px;")
                changed = True
        if changed:
            end = min(self._page_start + self._n_shown, self._n_ch)
            self._status.showMessage(
                f"Streaming  —  ch {self._page_start + 1}–{end} of {self._n_ch}"
                f"  |  triggers: {len(self._triggers)}"
            )
            self._redraw()

    def closeEvent(self, event) -> None:
        self._render_timer.stop()
        super().closeEvent(event)
