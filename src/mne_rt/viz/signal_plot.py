"""Real-time signal monitor.

Dark-themed scrolling window built on PyQt6 + pyqtgraph.

Classes
-------
SignalPlot
    Scrolling multi-channel real-time NF signal monitor.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

_COLORS = [
    "#5DA5A4",  # teal
    "#FF6B6B",  # coral
    "#FFD93D",  # yellow
    "#6BCB77",  # green
    "#4D96FF",  # blue
    "#FF922B",  # orange
    "#CC5DE8",  # purple
    "#F8BBD9",  # pink
]

_LABELS = {
    "sensor_power": "Sensor Power",
    "band_ratio": "Band Ratio",
    "source_power": "Source Power",
    "sensor_connectivity": "Sensor Connectivity",
    "source_connectivity": "Source Connectivity",
    "sensor_graph": "Sensor Graph",
    "source_graph": "Source Graph",
    "entropy": "Entropy",
    "argmax_freq": "Peak Frequency",
    "individual_peak_power": "Peak Power",
    "cfc_sensor": "Sensor CFC",
    "erd_ers": "ERD/ERS",
    "laterality": "Laterality",
    "hjorth": "Hjorth",
    "spectral_centroid": "Spectral Centroid",
}

_UNITS = {
    "sensor_power": "V²/Hz",
    "band_ratio": "",
    "source_power": "a.u.",
    "sensor_connectivity": "",
    "source_connectivity": "",
    "sensor_graph": "",
    "source_graph": "",
    "entropy": "",
    "argmax_freq": "Hz",
    "individual_peak_power": "V²/Hz",
    "cfc_sensor": "",
    "erd_ers": "%",
    "laterality": "",
    "hjorth": "",
    "spectral_centroid": "Hz",
}

_TIME_WINDOW_OPTIONS = [5, 10, 20, 30, 60]

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
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #d0d0e8;
    selection-background-color: #0f3460;
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
"""


class SignalPlot(QMainWindow):
    """Scrolling real-time neurofeedback signal monitor.

    Displays one colour-coded trace per active NF modality in a dark-themed
    :class:`pyqtgraph.PlotWidget`.  Designed to be driven by
    :meth:`~mne_rt.RTStream.record_main` via :meth:`push`.

    Parameters
    ----------
    modalities : list of str
        Names of the active NF modalities.  One scrolling trace is shown
        per entry.  Names must match entries in :data:`_LABELS` for a
        human-readable legend, or the raw string is used as a fallback.
    scales_dict : dict[str, float]
        Mapping from modality name to its physical display scale.  The raw
        NF value is divided by this scale before plotting so that all
        traces occupy a similar vertical range.
    sfreq : float
        Nominal update rate in Hz.  Used to size the ring buffer.
    time_window : float, default 10.0
        Visible time range in seconds at startup (can be changed at runtime
        from the control panel).
    verbose : bool | str | None, default None
        Verbosity level.  See :func:`~ant._logging.set_log_level`.

    See Also
    --------
    mne_rt.viz.BrainPlot : 3D brain activation display.
    mne_rt.RTStream.record_main : Drives both plots from the NF loop.

    Notes
    -----
    The control panel (right sidebar) provides:

    * **Playback** — pause/resume, clear buffer, screenshot.
    * **Display** — time-window selector, grid toggle, auto-range.
    * **Channel Scales** — per-modality amplitude scaling with ``+`` / ``−``
      buttons and a live ``×N`` readout.

    Status bar shows the latest value for every active modality.

    Examples
    --------
    Minimal offline usage:

    >>> app = QApplication([])
    >>> plot = SignalPlot(["sensor_power"], {"sensor_power": 1e-12}, sfreq=100)
    >>> plot.show()
    >>> plot.push([3.2e-13])
    >>> app.exec()

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        modalities: list[str],
        scales_dict: dict[str, float],
        sfreq: float,
        time_window: float = 10.0,
        display_smoothing: float = 0.3,
        verbose=None,
    ) -> None:
        from mne_rt._logging import set_log_level
        set_log_level(verbose)
        super().__init__()
        self._mods = modalities
        self._scales = scales_dict
        self._sfreq = sfreq
        self._time_window = time_window
        self._n = len(modalities)
        self._channel_scales = [1.0] * self._n
        self._paused = False
        self._display_alpha = float(np.clip(display_smoothing, 0.0, 1.0))
        self._ema = np.zeros(self._n)

        # Data buffers — 30 fps × time_window gives real-time resolution
        n_pts = max(int(sfreq * time_window), 30)
        self._time_axis = np.linspace(0.0, time_window, n_pts)
        self._buf = np.zeros((self._n, n_pts))

        pg.setConfigOptions(antialias=True, foreground="#c0c0d8", background="#0d0d1a")
        self._build_ui()
        self.setWindowTitle("ANT — MNE-RT")
        self.resize(1440, max(500, 200 + self._n * 150))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setStyleSheet(_QSS)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 4, 8)
        root.setSpacing(6)

        root.addWidget(self._build_plot_widget(), stretch=5)
        root.addWidget(self._build_control_panel(), stretch=0)

        self._status = self.statusBar()
        self._status.showMessage("Waiting for data …")

    def _build_plot_widget(self) -> pg.GraphicsLayoutWidget:
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground("#0d0d1a")

        self._plots: list[pg.PlotItem] = []
        self._curves: list[pg.PlotDataItem] = []

        for i, mod in enumerate(self._mods):
            color = _COLORS[i % len(_COLORS)]
            is_bottom = (i == self._n - 1)

            pi = glw.addPlot(row=i, col=0)
            self._apply_grid_style(pi, visible=True)
            pi.setMouseEnabled(x=False, y=False)

            # Label the left axis with the modality name in its colour
            pi.setLabel("left", _LABELS.get(mod, mod), color=color, size="10pt")
            pi.getAxis("left").setWidth(110)

            for ax_name in ("left", "bottom"):
                ax = pi.getAxis(ax_name)
                ax.setPen(pg.mkPen("#303050"))
                ax.setTextPen(pg.mkPen("#9090aa"))

            # Only the bottom plot shows the time axis label and tick values
            if is_bottom:
                pi.setLabel("bottom", "Time", units="s", color="#9090aa")
            else:
                pi.getAxis("bottom").setStyle(showValues=False)
                pi.getAxis("bottom").setHeight(0)

            # Fine time-axis ticks so minor grid lines appear
            self._apply_x_tick_spacing(pi, self._time_window)

            # Zero reference line
            pi.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#252545", width=1)))

            # Signal curve
            curve = pi.plot(
                self._time_axis,
                self._buf[i],
                pen=pg.mkPen(color=color, width=2),
            )
            self._curves.append(curve)

            pi.setXRange(0.0, self._time_window, padding=0.01)
            pi.enableAutoRange(axis='y')

            # Link all X axes to the first plot
            if i > 0:
                pi.setXLink(self._plots[0])

            self._plots.append(pi)

        self._glw = glw
        return glw

    def _build_control_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(210)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(self._grp_playback())
        layout.addWidget(self._grp_display())
        layout.addWidget(self._grp_channels())
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

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

    def _grp_display(self) -> QGroupBox:
        grp = QGroupBox("Display")
        lay = QVBoxLayout(grp)

        # Time-window selector
        row = QHBoxLayout()
        row.addWidget(QLabel("Time window:"))
        self._cmb_tw = QComboBox()
        for secs in _TIME_WINDOW_OPTIONS:
            self._cmb_tw.addItem(f"{secs} s", secs)
        # Select closest match
        best = min(_TIME_WINDOW_OPTIONS, key=lambda s: abs(s - self._time_window))
        self._cmb_tw.setCurrentIndex(_TIME_WINDOW_OPTIONS.index(best))
        self._cmb_tw.currentIndexChanged.connect(self._change_time_window)
        row.addWidget(self._cmb_tw)
        lay.addLayout(row)

        # Grid toggle
        chk = QCheckBox("Show grid")
        chk.setChecked(True)
        chk.toggled.connect(self._set_grid)
        lay.addWidget(chk)

        # Auto-range button
        btn_ar = QPushButton("⤢  Auto-range")
        btn_ar.clicked.connect(self._auto_range)
        lay.addWidget(btn_ar)

        return grp

    def _grp_channels(self) -> QGroupBox:
        grp = QGroupBox("Channel Scales")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)
        self._scale_labels: list[QLabel] = []

        for i, mod in enumerate(self._mods):
            color = _COLORS[i % len(_COLORS)]
            row = QHBoxLayout()
            row.setSpacing(3)

            lbl = QLabel(_LABELS.get(mod, mod))
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            scale_lbl = QLabel("×1.0")
            scale_lbl.setFixedWidth(38)
            scale_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            scale_lbl.setStyleSheet("color: #707090; font-size: 10px;")
            self._scale_labels.append(scale_lbl)

            btn_up = QPushButton("+")
            btn_dn = QPushButton("−")
            for btn in (btn_up, btn_dn):
                btn.setFixedSize(22, 22)

            btn_up.clicked.connect(lambda _, idx=i: self._scale_up(idx))
            btn_dn.clicked.connect(lambda _, idx=i: self._scale_down(idx))

            row.addWidget(lbl)
            row.addWidget(scale_lbl)
            row.addWidget(btn_up)
            row.addWidget(btn_dn)
            lay.addLayout(row)

        return grp

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_x_tick_spacing(pi: pg.PlotItem, secs: float) -> None:
        """Set major/minor X tick spacing so the grid is dense but readable."""
        if secs <= 5:
            major, minor = 1.0, 0.25
        elif secs <= 10:
            major, minor = 2.0, 0.5
        elif secs <= 20:
            major, minor = 5.0, 1.0
        elif secs <= 30:
            major, minor = 5.0, 1.0
        else:
            major, minor = 10.0, 2.0
        pi.getAxis("bottom").setTickSpacing(major=major, minor=minor)

    @staticmethod
    def _apply_grid_style(pi: pg.PlotItem, visible: bool = True) -> None:
        """Show grid with white lines and set pen directly on the GridItem."""
        pi.showGrid(x=visible, y=visible, alpha=0.45 if visible else 0.0)
        if visible:
            grid_pen = pg.mkPen(color=(220, 220, 255, 110), width=1, style=Qt.PenStyle.SolidLine)
            for item in pi.items:
                if isinstance(item, pg.GridItem):
                    item.setPen(grid_pen)

    def _set_grid(self, checked: bool) -> None:
        for pi in self._plots:
            self._apply_grid_style(pi, visible=checked)

    def _toggle_pause(self, checked: bool) -> None:
        self._paused = checked
        self._btn_pause.setText("▶  Resume" if checked else "⏸  Pause")

    def _clear(self) -> None:
        self._buf[:] = 0.0
        for i, curve in enumerate(self._curves):
            curve.setData(self._time_axis, self._buf[i])

    def _screenshot(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(Path.home() / f"signal_plot_{ts}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", default, "PNG Image (*.png)"
        )
        if not path:
            return
        exp = pg.exporters.ImageExporter(self._glw.scene())
        exp.parameters()["width"] = 1920
        exp.export(path)

    def _change_time_window(self, idx: int) -> None:
        secs = float(self._cmb_tw.itemData(idx))
        self._time_window = secs
        n_pts = max(int(self._sfreq * secs), 30)
        self._time_axis = np.linspace(0.0, secs, n_pts)
        self._buf = np.zeros((self._n, n_pts))
        for pi in self._plots:
            pi.setXRange(0.0, secs, padding=0.01)
            self._apply_x_tick_spacing(pi, secs)

    def _auto_range(self) -> None:
        for i, pi in enumerate(self._plots):
            row = self._buf[i][self._buf[i] != 0]
            if row.size > 0:
                margin = (row.max() - row.min()) * 0.1 or 1.0
                pi.setYRange(row.min() - margin, row.max() + margin, padding=0)

    def _scale_up(self, idx: int) -> None:
        self._channel_scales[idx] *= 2.0
        self._scale_labels[idx].setText(f"×{self._channel_scales[idx]:.3g}")

    def _scale_down(self, idx: int) -> None:
        self._channel_scales[idx] /= 2.0
        self._scale_labels[idx].setText(f"×{self._channel_scales[idx]:.3g}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(self, new_vals: list[float]) -> None:
        """Append one new sample per modality and refresh all traces.

        This is the main update entry point, called at ~30 fps by the
        Qt pump timer inside :meth:`~mne_rt.RTStream.record_main`.

        Parameters
        ----------
        new_vals : list of float
            Latest NF value for each active modality, in the same order as
            the ``modalities`` list passed to :meth:`__init__`.

        Notes
        -----
        The call is a no-op when the plot is paused (⏸ button pressed).
        Each value is normalised by its entry in ``scales_dict`` before
        being written into the ring buffer, so all traces share a common
        vertical scale.
        """
        if self._paused:
            return

        arr = np.asarray(new_vals, dtype=float)
        norm = np.array([
            (arr[i] / (self._scales[self._mods[i]] + 1e-300)) * self._channel_scales[i]
            for i in range(self._n)
        ])

        if self._display_alpha < 1.0:
            self._ema = self._display_alpha * norm + (1.0 - self._display_alpha) * self._ema
            norm = self._ema

        self._buf = np.roll(self._buf, -1, axis=1)
        self._buf[:, -1] = norm

        status_parts: list[str] = []
        for i, curve in enumerate(self._curves):
            curve.setData(self._time_axis, self._buf[i])
            val = arr[i]
            unit = _UNITS.get(self._mods[i], "")
            status_parts.append(
                f"{_LABELS.get(self._mods[i], self._mods[i])}: {val:.4g}"
                + (f" {unit}" if unit else "")
            )

        self._status.showMessage("  |  ".join(status_parts))

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
