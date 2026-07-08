"""Real-time scalp topomap and band-power display.

Dark-themed window built on Qt (via qtpy) + matplotlib (embedded via
:class:`~matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`).  Displays one
topomap per selected frequency band, updated in real-time from raw EEG/MEG
windows.

Classes
-------
TopomapPlot
    Real-time scalp topomap showing per-band power distribution.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import mne
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "δ  1–4 Hz": (1.0, 4.0),
    "θ  4–8 Hz": (4.0, 8.0),
    "α  8–13 Hz": (8.0, 13.0),
    "β  13–30 Hz": (13.0, 30.0),
    "γ  30–45 Hz": (30.0, 45.0),
}

_CMAPS = ["RdBu_r", "hot", "plasma", "viridis", "Reds", "RdYlBu_r"]

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
QSpinBox {
    background-color: #16213e;
    color: #d0d0e8;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 2px 4px;
}
"""


class TopomapPlot(QMainWindow):
    """Real-time scalp topomap showing per-band power distribution.

    Displays one colour-mapped topomap per selected frequency band,
    updated in real-time by :meth:`push`.  Built on Qt (via qtpy) with a
    matplotlib figure embedded via
    :class:`~matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`.

    Parameters
    ----------
    info : mne.Info
        Channel info containing electrode positions (montage must be set).
    sfreq : float
        Sampling frequency of the incoming data in Hz.
    bands : dict[str, tuple[float, float]] | None, default None
        Frequency bands to display as ``{label: (f_low, f_high)}``.
        Defaults to δ/θ/α/β/γ standard bands.
    cmap : str, default "RdBu_r"
        Initial colourmap.  Must be one of
        ``["RdBu_r", "hot", "plasma", "viridis", "Reds", "RdYlBu_r"]``.
    sensors : bool, default True
        Whether to draw sensor markers on the topomap.
    contours : int, default 6
        Number of contour lines.  Set to ``0`` to disable.
    display_smoothing : float, default 1.0
        EMA factor applied to the per-channel band-power maps before
        rendering.  ``1.0`` disables smoothing (raw per-window estimate
        shown directly — good for artifact monitoring); lower values
        progressively smooth the spatial maps across consecutive windows.
    verbose : bool | str | None, default None
        Verbosity level.  See :func:`~mne_rt._logging.set_log_level`.

    See Also
    --------
    mne_rt.viz.NFPlot : Scrolling NF signal display.
    mne_rt.viz.BrainPlot : 3D brain activation display.
    mne_rt.RTStream.record_main : Main NF loop that drives all plots.

    Notes
    -----
    The control panel (right sidebar) provides:

    * **Playback** — pause/resume, screenshot.
    * **Display** — colormap selector, contours, sensor toggle, colorbar mode.
    * **Bands** — per-band visibility checkboxes.  Toggling rebuilds the
      figure layout automatically.

    Band power is estimated via FFT on each incoming window (Hanning window
    applied to reduce spectral leakage).

    Examples
    --------
    Minimal offline usage:

    >>> from mne import create_info
    >>> import numpy as np
    >>> app = QApplication([])
    >>> info = create_info(["Fp1", "Fz", "Cz", "Oz"], sfreq=256, ch_types="eeg")
    >>> plot = TopomapPlot(info, sfreq=256)
    >>> plot.show()
    >>> data = np.random.randn(4, 256)   # 1 s window
    >>> plot.push(data)
    >>> app.exec()

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        info: mne.Info,
        sfreq: float,
        bands: Optional[dict[str, tuple[float, float]]] = None,
        cmap: str = "RdBu_r",
        sensors: bool = True,
        contours: int = 6,
        display_smoothing: float = 1.0,
        verbose=None,
    ) -> None:
        from mne_rt._logging import set_log_level

        set_log_level(verbose)
        super().__init__()

        self._info = info.copy()
        self._sfreq = float(sfreq)
        self._bands: dict[str, tuple[float, float]] = (
            dict(bands) if bands is not None else dict(_DEFAULT_BANDS)
        )
        self._cmap = cmap if cmap in _CMAPS else "RdBu_r"
        self._sensors = sensors
        self._contours = contours
        self._paused = False
        self._last_data: Optional[np.ndarray] = None
        self._visible_bands: list[str] = list(self._bands.keys())
        self._display_alpha = float(np.clip(display_smoothing, 0.0, 1.0))
        self._bp_ema: dict[str, np.ndarray] = {}

        # Detect channel type for plot_topomap
        self._ch_type = self._detect_ch_type()
        self._topo_picks = self._get_topo_picks()

        self._build_ui()
        self.setWindowTitle("MNE-RT — Topomap")
        self.resize(1200, 540)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _detect_ch_type(self) -> str:
        n_eeg = len(mne.pick_types(self._info, eeg=True, meg=False))
        n_mag = len(mne.pick_types(self._info, meg="mag", eeg=False))
        n_grad = len(mne.pick_types(self._info, meg="grad", eeg=False))
        if n_eeg > 0:
            return "eeg"
        if n_mag > 0:
            return "mag"
        if n_grad > 0:
            return "grad"
        return "eeg"

    def _get_topo_picks(self) -> np.ndarray:
        """Indices of channels to use for the topomap."""
        if self._ch_type == "eeg":
            picks = mne.pick_types(self._info, eeg=True, meg=False, exclude="bads")
        elif self._ch_type == "mag":
            picks = mne.pick_types(self._info, meg="mag", eeg=False, exclude="bads")
        else:
            picks = mne.pick_types(self._info, meg="grad", eeg=False, exclude="bads")
        if len(picks) == 0:
            picks = np.arange(len(self._info["ch_names"]))
        # Drop channels whose 3D position is NaN (e.g. TP9/TP10 not in biosemi64 montage)
        valid = np.array([not np.any(np.isnan(self._info["chs"][p]["loc"][:3])) for p in picks])
        picks = picks[valid]
        return picks

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

        self._canvas_holder = QWidget()
        self._canvas_layout = QVBoxLayout(self._canvas_holder)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._build_canvas()

        root.addWidget(self._canvas_holder, stretch=5)
        root.addWidget(self._build_control_panel(), stretch=0)

        self._status = self.statusBar()
        self._status.showMessage("Waiting for data …")

    def _build_canvas(self) -> None:
        n_vis = max(len(self._visible_bands), 1)
        n_cols = min(n_vis, 5)
        n_rows = (n_vis + n_cols - 1) // n_cols

        fig_w = max(3.2 * n_cols, 6.0)
        fig_h = max(3.2 * n_rows, 3.2)

        self._fig = Figure(figsize=(fig_w, fig_h), facecolor="#0d0d1a")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas_layout.addWidget(self._canvas)

        self._axes: dict[str, object] = {}
        for i, band_name in enumerate(self._visible_bands):
            ax = self._fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_facecolor("#0d0d1a")
            ax.set_title(band_name, color="#b0b0c8", fontsize=10, pad=4)
            self._axes[band_name] = ax

        self._fig.tight_layout(pad=1.2)

    def _rebuild_canvas(self) -> None:
        old = self._canvas_layout.itemAt(0)
        if old is not None:
            w = old.widget()
            self._canvas_layout.removeWidget(w)
            w.deleteLater()

        self._fig.clear()
        self._axes.clear()
        self._build_canvas()

        if self._last_data is not None:
            self._update_topomaps(self._last_data)
        else:
            self._canvas.draw()

    def _build_control_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(215)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(self._grp_playback())
        layout.addWidget(self._grp_display())
        layout.addWidget(self._grp_bands())
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _grp_playback(self) -> QGroupBox:
        grp = QGroupBox("Playback")
        lay = QVBoxLayout(grp)

        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_pause.setCheckable(True)
        self._btn_pause.clicked.connect(self._toggle_pause)

        btn_shot = QPushButton("📷  Screenshot")
        btn_shot.clicked.connect(self._screenshot)

        btn_freeze = QPushButton("🔒  Freeze clim")
        btn_freeze.setCheckable(True)
        btn_freeze.clicked.connect(self._toggle_freeze_clim)
        self._btn_freeze = btn_freeze

        for w in (self._btn_pause, btn_shot, btn_freeze):
            lay.addWidget(w)
        return grp

    def _grp_display(self) -> QGroupBox:
        grp = QGroupBox("Display")
        lay = QVBoxLayout(grp)

        row_cmap = QHBoxLayout()
        row_cmap.addWidget(QLabel("Colormap:"))
        self._cmb_cmap = QComboBox()
        for c in _CMAPS:
            self._cmb_cmap.addItem(c)
        self._cmb_cmap.setCurrentText(self._cmap)
        self._cmb_cmap.currentTextChanged.connect(self._change_cmap)
        row_cmap.addWidget(self._cmb_cmap)
        lay.addLayout(row_cmap)

        row_cont = QHBoxLayout()
        row_cont.addWidget(QLabel("Contours:"))
        self._spn_contours = QSpinBox()
        self._spn_contours.setRange(0, 12)
        self._spn_contours.setValue(self._contours)
        self._spn_contours.valueChanged.connect(self._change_contours)
        row_cont.addWidget(self._spn_contours)
        lay.addLayout(row_cont)

        chk_sensors = QCheckBox("Show sensors")
        chk_sensors.setChecked(self._sensors)
        chk_sensors.toggled.connect(self._toggle_sensors)
        lay.addWidget(chk_sensors)

        return grp

    def _grp_bands(self) -> QGroupBox:
        grp = QGroupBox("Bands")
        self._bands_layout = QVBoxLayout(grp)
        self._bands_layout.setSpacing(3)

        self._band_checks: dict[str, QCheckBox] = {}
        for band_name in self._bands:
            self._add_band_checkbox(band_name)

        # ── Custom frequency range ────────────────────────────────────────
        self._bands_layout.addSpacing(6)
        lbl = QLabel("Custom band (Hz):")
        lbl.setStyleSheet("color:#8888aa; font-size:10px; font-weight:600;")
        self._bands_layout.addWidget(lbl)

        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(4)

        self._custom_flo = QDoubleSpinBox()
        self._custom_flo.setRange(0.1, 200.0)
        self._custom_flo.setSingleStep(0.5)
        self._custom_flo.setValue(1.0)
        self._custom_flo.setDecimals(1)
        self._custom_flo.setSuffix(" Hz")
        self._custom_flo.setFixedWidth(72)
        self._custom_flo.setStyleSheet(
            "background:#16213e; color:#d0d0e8; border:1px solid #0f3460;"
            "border-radius:4px; padding:2px 4px;"
        )

        dash = QLabel("–")
        dash.setStyleSheet("color:#8888aa; font-size:11px;")

        self._custom_fhi = QDoubleSpinBox()
        self._custom_fhi.setRange(0.2, 200.0)
        self._custom_fhi.setSingleStep(0.5)
        self._custom_fhi.setValue(4.0)
        self._custom_fhi.setDecimals(1)
        self._custom_fhi.setSuffix(" Hz")
        self._custom_fhi.setFixedWidth(72)
        self._custom_fhi.setStyleSheet(self._custom_flo.styleSheet())

        btn_add = QPushButton("Add")
        btn_add.setFixedHeight(24)
        btn_add.setStyleSheet(
            "background:#0f3460; color:#d0d0e8; border:1px solid #533483;"
            "border-radius:4px; font-size:11px; padding:0 6px;"
        )
        btn_add.clicked.connect(self._add_custom_band)

        row_lay.addWidget(self._custom_flo)
        row_lay.addWidget(dash)
        row_lay.addWidget(self._custom_fhi)
        row_lay.addWidget(btn_add)
        row_lay.addStretch()
        self._bands_layout.addWidget(row)

        return grp

    def _add_band_checkbox(self, band_name: str) -> None:
        chk = QCheckBox(band_name)
        chk.setChecked(band_name in self._visible_bands)
        chk.toggled.connect(lambda checked, b=band_name: self._toggle_band(b, checked))
        self._bands_layout.addWidget(chk)
        self._band_checks[band_name] = chk

    def _add_custom_band(self) -> None:
        flo = round(self._custom_flo.value(), 1)
        fhi = round(self._custom_fhi.value(), 1)
        if flo >= fhi:
            return
        band_name = f"Custom {flo}–{fhi} Hz"
        if band_name in self._bands:
            return
        self._bands[band_name] = (flo, fhi)
        self._visible_bands.append(band_name)
        self._add_band_checkbox(band_name)
        self._rebuild_canvas()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _toggle_pause(self, checked: bool) -> None:
        self._paused = checked
        self._btn_pause.setText("▶  Resume" if checked else "⏸  Pause")

    def _toggle_freeze_clim(self, checked: bool) -> None:
        self._freeze_clim = checked
        self._btn_freeze.setText("🔓  Unfreeze clim" if checked else "🔒  Freeze clim")

    def _screenshot(self) -> None:
        from qtpy.QtWidgets import QFileDialog

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(Path.home() / f"ant_topo_{ts}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", default, "PNG Image (*.png)")
        if not path:
            return
        self._fig.savefig(path, dpi=150, facecolor=self._fig.get_facecolor())
        self._status.showMessage(f"Screenshot saved: {Path(path).name}")

    def _change_cmap(self, cmap: str) -> None:
        self._cmap = cmap
        if self._last_data is not None and not self._paused:
            self._update_topomaps(self._last_data)

    def _change_contours(self, val: int) -> None:
        self._contours = val
        if self._last_data is not None and not self._paused:
            self._update_topomaps(self._last_data)

    def _toggle_sensors(self, checked: bool) -> None:
        self._sensors = checked
        if self._last_data is not None and not self._paused:
            self._update_topomaps(self._last_data)

    def _toggle_band(self, band_name: str, checked: bool) -> None:
        orig_order = list(self._bands.keys())
        if checked and band_name not in self._visible_bands:
            self._visible_bands = [
                b for b in orig_order if b in self._visible_bands or b == band_name
            ]
        elif not checked and band_name in self._visible_bands:
            self._visible_bands.remove(band_name)
        self._rebuild_canvas()

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def _compute_band_powers(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """Compute per-channel FFT mean power for each frequency band.

        Parameters
        ----------
        data : ndarray of shape (n_channels, n_times)

        Returns
        -------
        dict mapping band label → ndarray of shape (n_topo_picks,)
        """
        topo_data = data[self._topo_picks]
        n_times = topo_data.shape[1]
        win = np.hanning(n_times)
        fft_out = np.fft.rfft(topo_data * win, axis=1)
        freqs = np.fft.rfftfreq(n_times, d=1.0 / self._sfreq)
        psd = np.abs(fft_out) ** 2

        result: dict[str, np.ndarray] = {}
        for band_name, (flo, fhi) in self._bands.items():
            mask = (freqs >= flo) & (freqs <= fhi)
            if not mask.any():
                result[band_name] = np.zeros(len(self._topo_picks))
            else:
                result[band_name] = psd[:, mask].mean(axis=1)
        return result

    def _update_topomaps(self, data: np.ndarray) -> None:
        """Redraw all visible topomaps from a raw data window."""
        band_powers = self._compute_band_powers(data)
        if self._display_alpha < 1.0:
            for band, pw in band_powers.items():
                if band not in self._bp_ema:
                    self._bp_ema[band] = pw.copy()
                else:
                    self._bp_ema[band] = (
                        self._display_alpha * pw + (1.0 - self._display_alpha) * self._bp_ema[band]
                    )
                band_powers[band] = self._bp_ema[band]
        topo_info = mne.pick_info(self._info, self._topo_picks)

        freeze = getattr(self, "_freeze_clim", False)

        for band_name, ax in self._axes.items():
            powers = band_powers.get(band_name)
            if powers is None:
                continue

            if not freeze:
                vmax = float(np.percentile(np.abs(powers), 97)) or 1e-20
                if self._cmap in ("RdBu_r", "RdYlBu_r", "bwr"):
                    vmin, vmax_use = -vmax, vmax
                else:
                    vmin, vmax_use = 0.0, vmax
                self._clim_cache = getattr(self, "_clim_cache", {})
                self._clim_cache[band_name] = (vmin, vmax_use)
            else:
                cache = getattr(self, "_clim_cache", {})
                vmin, vmax_use = cache.get(band_name, (-1e-20, 1e-20))

            ax.clear()
            ax.set_facecolor("#0d0d1a")
            ax.set_title(band_name, color="#b0b0c8", fontsize=10, pad=4)

            try:
                mne.viz.plot_topomap(
                    powers,
                    topo_info,
                    axes=ax,
                    show=False,
                    cmap=self._cmap,
                    vlim=(vmin, vmax_use),
                    sensors=self._sensors,
                    contours=self._contours,
                    res=32,
                    extrapolate="auto",
                )
            except Exception as _e:
                ax.text(
                    0.5,
                    0.5,
                    f"Topo error:\n{_e}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#606080",
                    fontsize=7,
                    wrap=True,
                )

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(self, data: np.ndarray) -> None:
        """Update all visible topomaps from a new raw data window.

        Called at ~5 fps by the Qt pump timer inside
        :meth:`~mne_rt.RTStream.record_main`.

        Parameters
        ----------
        data : ndarray of shape (n_channels, n_times)
            Raw EEG/MEG data for the current analysis window.

        Notes
        -----
        The call is a no-op when paused (⏸ button pressed).  Band power
        is estimated via FFT and the topomaps are redrawn via
        :func:`mne.viz.plot_topomap`.
        """
        if self._paused:
            return
        self._last_data = data
        self._update_topomaps(data)
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._status.showMessage(f"Updated {now}")
