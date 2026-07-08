"""Real-time butterfly plot: all channels overlaid per condition.

Channels are coloured by scalp region (frontal → occipital gradient).
Redraws after every :meth:`update` call as new epochs arrive from
:class:`~mne_rt.RTEpochs`.

Classes
-------
ButterflyPlot
    Real-time butterfly overlay with region-colour coding and interactive
    sidebar.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

try:
    from qtpy.QtCore import Qt, Signal
    from qtpy.QtGui import QColor, QFont
    from qtpy.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    _qt_available = True
except ImportError:
    _qt_available = False

try:
    import pyqtgraph as pg

    _pg_available = True
except ImportError:
    _pg_available = False

try:
    import mne

    _mne_available = True
except ImportError:
    _mne_available = False

from mne_rt._logging import logger, set_log_level

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
_BG = "#0d1117"
_SURFACE = "#161b22"
_BORDER = "#30363d"
_TEXT = "#e6edf3"
_DIM = "#8b949e"
_ACCENT = "#3b82f6"

_COND_COLORS = [
    "#3b82f6",  # blue
    "#ec4899",  # pink
    "#10b981",  # green
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
]

# Region colour gradient stops: (normalised_y, R, G, B)
# y=0 is frontal (top of scalp in standard orientation → small y),
# y=1 is occipital (bottom).
_REGION_STOPS = [
    (0.0, 0x3B, 0x82, 0xF6),  # #3b82f6 blue       – frontal
    (0.3, 0x06, 0xB6, 0xD4),  # #06b6d4 cyan       – fronto-central
    (0.5, 0x10, 0xB9, 0x81),  # #10b981 green      – central
    (0.7, 0xF5, 0x9E, 0x0B),  # #f59e0b amber      – parieto-occipital
    (1.0, 0xEF, 0x44, 0x44),  # #ef4444 red        – occipital
]

_SIDEBAR_W = 210  # px


# ---------------------------------------------------------------------------
# Sidebar helpers (mirrors erp_plot.py)
# ---------------------------------------------------------------------------


def _sep(parent: QWidget) -> QFrame:
    f = QFrame(parent)
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{_BORDER};")
    return f


def _section(text: str, parent: QWidget) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setStyleSheet(
        f"color:{_DIM}; font-size:10px; font-weight:700; letter-spacing:1px; padding-top:6px;"
    )
    return lbl


def _row(parent: QWidget, spacing: int = 5) -> tuple[QWidget, QHBoxLayout]:
    w = QWidget(parent)
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 1, 0, 1)
    lay.setSpacing(spacing)
    return w, lay


def _val_lbl(text: str, parent: QWidget, color: str = _ACCENT) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setStyleSheet(f"color:{color}; font-size:11px; font-weight:600;")
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return lbl


def _key_lbl(text: str, parent: QWidget) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setStyleSheet(f"color:{_TEXT}; font-size:11px;")
    return lbl


def _slider(parent: QWidget, lo: int, hi: int, val: int) -> QSlider:
    sl = QSlider(Qt.Orientation.Horizontal, parent)
    sl.setRange(lo, hi)
    sl.setValue(val)
    return sl


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _region_color(xn: float, yn: float) -> tuple[int, int, int]:
    """Return an RGB tuple for a channel at normalised scalp position.

    Parameters
    ----------
    xn : float
        Normalised x position (0 = left, 1 = right) — unused but kept for
        future lateral gradient extension.
    yn : float
        Normalised y position (0 = frontal, 1 = occipital).

    Returns
    -------
    tuple of int
        ``(R, G, B)`` each in 0–255.
    """
    t = float(np.clip(yn, 0.0, 1.0))
    for i in range(len(_REGION_STOPS) - 1):
        t0, r0, g0, b0 = _REGION_STOPS[i]
        t1, r1, g1, b1 = _REGION_STOPS[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
            return (
                int(r0 + alpha * (r1 - r0)),
                int(g0 + alpha * (g1 - g0)),
                int(b0 + alpha * (b1 - b0)),
            )
    # Clamp to last stop
    return (_REGION_STOPS[-1][1], _REGION_STOPS[-1][2], _REGION_STOPS[-1][3])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ButterflyPlot(QMainWindow):
    """Real-time butterfly plot: all EEG/MEG channels overlaid per condition.

    Each condition gets its own :class:`pyqtgraph.PlotItem` stacked
    vertically in a :class:`pyqtgraph.GraphicsLayoutWidget`.  Channels
    are drawn as thin lines coloured by scalp region (blue → cyan → green
    → amber → red from frontal to occipital).

    Parameters
    ----------
    ch_names : list of str
        Electrode names in data order.
    sfreq : float
        Sampling frequency in Hz.
    tmin : float
        Epoch start (s).
    tmax : float
        Epoch end (s).
    event_id : dict[str, int]
        Condition label → marker integer.
    info : mne.Info or None
        When provided, used for exact scalp positioning via
        :func:`mne.channels.find_layout`.
    montage : str, default ``"standard_1020"``
        Fallback montage when *info* is absent or has no dig points.
    baseline : tuple or None, default ``(None, 0)``
        Baseline correction interval.
    window_size : tuple of int, default ``(1440, 900)``
        Initial window size in pixels.
    verbose : bool, str, or None

    .. versionadded:: 1.0.0

    See Also
    --------
    mne_rt.RTEpochs : Drives this plot via :meth:`update`.
    """

    # Emitted from any thread; Qt delivers it to _redraw on the main thread.
    _redraw_sig = Signal(int)

    def __init__(
        self,
        ch_names: list[str],
        sfreq: float,
        tmin: float,
        tmax: float,
        event_id: dict[str, int],
        info=None,
        montage: str = "standard_1020",
        baseline: Optional[tuple] = (None, 0),
        window_size: tuple[int, int] = (1440, 900),
        verbose: Union[bool, str, None] = None,
    ) -> None:
        if not _qt_available or not _pg_available:
            raise ImportError(
                "A Qt binding (PyQt6 or PySide6) and pyqtgraph are required for ButterflyPlot.\n"
                "Install with: pip install 'mne-rt[full]'"
            )
        _app = QApplication.instance() or QApplication([])  # noqa: F841

        super().__init__()
        self._redraw_sig.connect(self._redraw)
        set_log_level(verbose)

        self.ch_names = list(ch_names)
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        self.event_id = event_id
        self.montage = montage
        self.baseline = baseline
        self._info = info

        self._conditions = list(event_id.keys())
        self._cmap = {
            c: _COND_COLORS[i % len(_COND_COLORS)] for i, c in enumerate(self._conditions)
        }
        self._n_ch = len(ch_names)
        self._n_t = int(round((tmax - tmin) * sfreq)) + 1
        self._times = np.linspace(tmin, tmax, self._n_t)

        self._epoch_buf: dict[str, list[np.ndarray]] = {c: [] for c in self._conditions}
        self._n_per: dict[str, int] = {c: 0 for c in self._conditions}

        # Display state
        self._yscale = 1.0
        self._linewidth = 0.8
        self._show_bl = True
        self._show_grid = False
        self._x_start = tmin
        self._x_end = tmax

        # Scalp positions for region colouring
        self._norm_pos = self._compute_positions(info, montage)

        self.setWindowTitle("MNE-RT — Butterfly Plot")
        self.resize(*window_size)
        self._apply_styles()
        self._build_ui()

        logger.info(
            "ButterflyPlot: %d channels, %.0f–%.0f ms, %d conditions",
            self._n_ch,
            tmin * 1000,
            tmax * 1000,
            len(self._conditions),
        )

    # -----------------------------------------------------------------------
    # Scalp layout (for region colouring only)
    # -----------------------------------------------------------------------

    def _compute_positions(self, info, montage_name: str) -> list[tuple[float, float]]:
        """Return normalised (x, y) positions for each channel.

        y=0 is frontal, y=1 is occipital (standard EEG orientation).
        """
        if _mne_available:
            if info is not None:
                pos = self._from_layout(mne.channels.find_layout(info))
                if pos is not None:
                    return pos
            try:
                tmp = mne.create_info(self.ch_names, sfreq=1.0, ch_types="eeg", verbose=False)
                mont = mne.channels.make_standard_montage(montage_name)
                tmp.set_montage(mont, on_missing="ignore", verbose=False)
                pos = self._from_layout(mne.channels.find_layout(tmp))
                if pos is not None:
                    return pos
            except Exception as exc:
                logger.debug("ButterflyPlot: montage layout failed: %s", exc)

        logger.warning("ButterflyPlot: falling back to circular channel arrangement.")
        return self._circular_fallback()

    def _from_layout(self, layout) -> Optional[list[tuple[float, float]]]:
        if layout is None:
            return None
        name_xy: dict[str, tuple[float, float]] = {}
        for name, pos in zip(layout.names, layout.pos):
            xc = float(pos[0] + pos[2] / 2.0)
            yc = float(pos[1] + pos[3] / 2.0)
            # y is already 0=top (frontal) after standard layout convention;
            # keep as-is so y≈0 → frontal, y≈1 → occipital.
            name_xy[name] = (xc, float(yc))

        n_matched = sum(1 for c in self.ch_names if c in name_xy)
        if n_matched < self._n_ch // 2:
            logger.debug(
                "ButterflyPlot: only %d/%d channels matched layout — skipping.",
                n_matched,
                self._n_ch,
            )
            return None

        positions: list[tuple[float, float]] = []
        fb = 0
        for ch in self.ch_names:
            if ch in name_xy:
                positions.append(name_xy[ch])
            else:
                positions.append((0.5, fb * 0.05))
                fb += 1
        return positions

    def _circular_fallback(self) -> list[tuple[float, float]]:
        n = self._n_ch
        return [
            (
                0.5 + 0.42 * math.cos(2 * math.pi * i / n - math.pi / 2),
                0.5 + 0.42 * math.sin(2 * math.pi * i / n - math.pi / 2),
            )
            for i in range(n)
        ]

    # -----------------------------------------------------------------------
    # Styles
    # -----------------------------------------------------------------------

    def _apply_styles(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget  {{ background:{_BG}; color:{_TEXT}; }}
            QLabel                {{ color:{_TEXT}; font-size:12px; }}
            QCheckBox             {{ color:{_TEXT}; font-size:12px; spacing:6px; }}
            QCheckBox::indicator  {{ width:14px; height:14px; border-radius:3px;
                                     border:1px solid {_BORDER};
                                     background:{_SURFACE}; }}
            QCheckBox::indicator:checked {{ background:{_ACCENT};
                                           border-color:{_ACCENT}; }}
            QPushButton           {{ background:{_SURFACE}; color:{_TEXT};
                                     border:1px solid {_BORDER};
                                     border-radius:5px; padding:4px 10px;
                                     font-size:11px; }}
            QPushButton:hover     {{ background:{_BORDER}; }}
            QSlider::groove:horizontal {{
                height:4px; background:{_BORDER}; border-radius:2px; }}
            QSlider::handle:horizontal {{
                width:14px; height:14px; margin:-5px 0;
                border-radius:7px; background:{_ACCENT}; }}
            QScrollArea           {{ border: none; }}
            QScrollBar:vertical   {{ background:{_BG}; width:6px; }}
            QScrollBar::handle:vertical {{ background:{_BORDER}; border-radius:3px; }}
        """)

    # -----------------------------------------------------------------------
    # UI build
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Canvas ──────────────────────────────────────────────────────
        pg.setConfigOptions(antialias=True, background=_BG, foreground=_DIM)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(_BG)
        self._glw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._glw, stretch=1)

        # ── Sidebar ──────────────────────────────────────────────────────
        root.addWidget(self._build_sidebar())

        # ── Condition plots ──────────────────────────────────────────────
        # curves[cond][ch_idx] → PlotCurveItem
        self._cond_plots: dict[str, pg.PlotItem] = {}
        self._curves: dict[str, list[pg.PlotCurveItem]] = {}
        self._cond_titles: dict[str, pg.TextItem] = {}
        self._grid_items: list[pg.GridItem] = []
        self._t0_lines: list[pg.InfiniteLine] = []

        n_conds = len(self._conditions)
        for row_idx, cond in enumerate(self._conditions):
            plot = self._glw.addPlot(row=row_idx, col=0)
            self._style_plot(plot, cond)
            self._cond_plots[cond] = plot

            # t = 0 dashed line
            t0_line = pg.InfiniteLine(
                pos=self.tmin + (self.tmax - self.tmin) * 0,  # actual pos=0s
                angle=90,
                pen=pg.mkPen(_BORDER, width=1, style=Qt.PenStyle.DashLine),
            )
            t0_line.setValue(0.0)
            plot.addItem(t0_line)
            self._t0_lines.append(t0_line)

            # GridItem (works with hidden axes unlike showGrid)
            grid = pg.GridItem(
                pen=pg.mkPen(_BORDER, width=0.5),
                textPen=None,
            )
            grid.setVisible(False)
            plot.getViewBox().addItem(grid)
            self._grid_items.append(grid)

            # One curve per channel
            curves: list[pg.PlotCurveItem] = []
            for ch_i in range(self._n_ch):
                xn, yn = self._norm_pos[ch_i]
                r, g, b = _region_color(xn, yn)
                curve = plot.plot(
                    self._times,
                    np.zeros(self._n_t),
                    pen=pg.mkPen((r, g, b, 180), width=self._linewidth),
                )
                curves.append(curve)
            self._curves[cond] = curves

            # Condition title (top-left TextItem)
            title = pg.TextItem(
                f"{cond}  (n = 0)",
                color=_TEXT,
                anchor=(0, 0),
            )
            title.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
            plot.addItem(title)
            title.setPos(self.tmin, 0)
            self._cond_titles[cond] = title

        # Ensure equal row stretches
        for row_idx in range(n_conds):
            self._glw.ci.layout.setRowStretchFactor(row_idx, 1)

    def _style_plot(self, plot: pg.PlotItem, cond_title: str) -> None:
        """Configure a condition PlotItem: hide axes, disable mouse."""
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setMouseEnabled(x=False, y=False)
        for ax in ("bottom", "left", "top", "right"):
            plot.showAxis(ax, False)
        plot.setContentsMargins(0, 0, 0, 0)
        plot.getViewBox().setBackgroundColor(_BG)
        plot.setXRange(self.tmin, self.tmax, padding=0)

    # -----------------------------------------------------------------------
    # Sidebar build
    # -----------------------------------------------------------------------

    def _build_sidebar(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setFixedWidth(_SIDEBAR_W + 14)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background:{_SURFACE}; border-left:1px solid {_BORDER}; }}"
        )

        sb = QWidget()
        sb.setStyleSheet(f"background:{_SURFACE};")
        ly = QVBoxLayout(sb)
        ly.setSpacing(4)
        ly.setContentsMargins(10, 12, 10, 12)

        # ── Header ───────────────────────────────────────────────────────
        hdr = QLabel("BUTTERFLY CONTROLS")
        hdr.setStyleSheet(f"color:{_TEXT}; font-size:11px; font-weight:700; letter-spacing:1.5px;")
        ly.addWidget(hdr)
        ly.addWidget(_sep(sb))

        # ── CONDITIONS ───────────────────────────────────────────────────
        ly.addWidget(_section("CONDITIONS", sb))
        self._cond_checks: dict[str, QCheckBox] = {}
        self._cond_n_lbl: dict[str, QLabel] = {}

        for cond in self._conditions:
            col = self._cmap[cond]
            row_w, row_l = _row(sb)
            cb = QCheckBox()
            cb.setChecked(True)
            cb.setStyleSheet(
                f"QCheckBox::indicator:checked{{background:{col};border-color:{col};}}"
            )
            cb.toggled.connect(lambda chk, c=cond: self._toggle_cond(c, chk))
            self._cond_checks[cond] = cb

            dot = QLabel(f"● {cond}")
            dot.setStyleSheet(f"color:{col};font-size:11px;font-weight:600;")
            dot.setWordWrap(True)

            n_lbl = _val_lbl("n = 0", sb, color=_DIM)
            self._cond_n_lbl[cond] = n_lbl

            row_l.addWidget(cb)
            row_l.addWidget(dot, stretch=1)
            row_l.addWidget(n_lbl)
            ly.addWidget(row_w)

        ly.addWidget(_sep(sb))

        # ── AMPLITUDE ────────────────────────────────────────────────────
        ly.addWidget(_section("AMPLITUDE", sb))

        r1, l1 = _row(sb)
        l1.addWidget(_key_lbl("Y scale", sb), stretch=1)
        self._sv_lbl = _val_lbl("×1.0", sb)
        l1.addWidget(self._sv_lbl)
        ly.addWidget(r1)

        self._scale_sl = _slider(sb, 1, 50, 5)
        self._scale_sl.valueChanged.connect(self._on_scale)
        ly.addWidget(self._scale_sl)

        ra, la = _row(sb, spacing=6)
        auto_btn = QPushButton("Auto scale")
        auto_btn.clicked.connect(self._auto_scale)
        la.addWidget(auto_btn)
        ly.addWidget(ra)

        ly.addWidget(_sep(sb))

        # ── TIME WINDOW ──────────────────────────────────────────────────
        ly.addWidget(_section("TIME WINDOW", sb))

        tmin_ms = int(self.tmin * 1000)
        tmax_ms = int(self.tmax * 1000)

        r4, l4 = _row(sb)
        l4.addWidget(_key_lbl("Start", sb), stretch=1)
        self._xstart_lbl = _val_lbl(f"{tmin_ms} ms", sb)
        l4.addWidget(self._xstart_lbl)
        ly.addWidget(r4)

        self._xstart_sl = _slider(sb, tmin_ms, 0, tmin_ms)
        self._xstart_sl.valueChanged.connect(self._on_xrange)
        ly.addWidget(self._xstart_sl)

        r5, l5 = _row(sb)
        l5.addWidget(_key_lbl("End", sb), stretch=1)
        self._xend_lbl = _val_lbl(f"{tmax_ms} ms", sb)
        l5.addWidget(self._xend_lbl)
        ly.addWidget(r5)

        self._xend_sl = _slider(sb, 0, tmax_ms, tmax_ms)
        self._xend_sl.valueChanged.connect(self._on_xrange)
        ly.addWidget(self._xend_sl)

        ly.addWidget(_sep(sb))

        # ── APPEARANCE ───────────────────────────────────────────────────
        ly.addWidget(_section("APPEARANCE", sb))

        r2, l2 = _row(sb)
        l2.addWidget(_key_lbl("Line width", sb), stretch=1)
        self._lw_lbl = _val_lbl("0.8", sb)
        l2.addWidget(self._lw_lbl)
        ly.addWidget(r2)

        # 1–8 → 0.5–4.0 px  (step 0.5)
        self._lw_sl = _slider(sb, 1, 8, 2)
        self._lw_sl.valueChanged.connect(self._on_linewidth)
        ly.addWidget(self._lw_sl)

        self._bl_chk = QCheckBox("Baseline line")
        self._bl_chk.setChecked(True)
        self._bl_chk.toggled.connect(self._toggle_bl)
        ly.addWidget(self._bl_chk)

        self._grid_chk = QCheckBox("Grid lines")
        self._grid_chk.setChecked(False)
        self._grid_chk.toggled.connect(self._toggle_grid)
        ly.addWidget(self._grid_chk)

        ly.addWidget(_sep(sb))

        # ── DATA ─────────────────────────────────────────────────────────
        ly.addWidget(_section("DATA", sb))

        self._total_lbl = QLabel("Total: 0 trials")
        self._total_lbl.setStyleSheet(f"color:{_TEXT};font-size:11px;")
        ly.addWidget(self._total_lbl)

        ly.addSpacing(4)

        export_btn = QPushButton("Export PNG …")
        export_btn.setToolTip("Save the current butterfly plot as a PNG image")
        export_btn.clicked.connect(self._export_png)
        ly.addWidget(export_btn)

        ly.addStretch()
        scroll.setWidget(sb)
        return scroll

    # -----------------------------------------------------------------------
    # Sidebar callbacks
    # -----------------------------------------------------------------------

    def _toggle_cond(self, cond: str, visible: bool) -> None:
        """Show/hide all channel curves for a condition."""
        for curve in self._curves[cond]:
            curve.setVisible(visible)

    def _on_scale(self, value: int) -> None:
        # slider 1–50; centre value 5 → ×1.0; step 0.2
        self._yscale = value * 0.2
        self._sv_lbl.setText(f"×{self._yscale:.1f}")
        self._apply_y_range()

    def _apply_y_range(self) -> None:
        avgs = []
        for cond in self._conditions:
            if self._cond_checks.get(cond, QCheckBox()).isChecked():
                buf = self._epoch_buf[cond]
                if buf:
                    avgs.append(np.mean(np.stack(buf, 0), 0))
        if not avgs:
            return
        amp = float(np.percentile(np.abs(np.stack(avgs, 0)), 99)) or 1e-12
        half = amp * self._yscale
        for plot in self._cond_plots.values():
            plot.setYRange(-half, half, padding=0.05)

    def _auto_scale(self) -> None:
        self._scale_sl.setValue(5)
        for plot in self._cond_plots.values():
            plot.enableAutoRange(axis="y")

    def _on_linewidth(self, value: int) -> None:
        self._linewidth = value * 0.5
        self._lw_lbl.setText(f"{self._linewidth:.1f}")
        for cond in self._conditions:
            for ch_i, curve in enumerate(self._curves[cond]):
                xn, yn = self._norm_pos[ch_i]
                r, g, b = _region_color(xn, yn)
                curve.setPen(pg.mkPen((r, g, b, 180), width=self._linewidth))

    def _on_xrange(self) -> None:
        x1 = self._xstart_sl.value() / 1000.0
        x2 = self._xend_sl.value() / 1000.0
        if x1 >= x2:
            return
        self._x_start = x1
        self._x_end = x2
        self._xstart_lbl.setText(f"{int(x1 * 1000)} ms")
        self._xend_lbl.setText(f"{int(x2 * 1000)} ms")
        for plot in self._cond_plots.values():
            plot.setXRange(x1, x2, padding=0)

    def _toggle_bl(self, visible: bool) -> None:
        for line in self._t0_lines:
            line.setVisible(visible)

    def _toggle_grid(self, visible: bool) -> None:
        for grid in self._grid_items:
            grid.setVisible(visible)

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Butterfly Plot",
            "butterfly_plot.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)",
        )
        if path:
            self.grab().save(path)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(
        self,
        data: np.ndarray,
        conditions: list[str],
    ) -> None:
        """Redraw all butterfly plots with updated condition averages.

        Thread-safe — may be called from the acquisition thread.

        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_channels, n_times)
            All accepted epochs so far.
        conditions : list of str
            Condition label for each epoch; length == ``data.shape[0]``.
        """
        n_total = len(conditions)
        for cond in self._conditions:
            mask = np.array([c == cond for c in conditions])
            self._epoch_buf[cond] = list(data[mask]) if mask.any() else []
            self._n_per[cond] = int(mask.sum())

        self._redraw_sig.emit(n_total)

    def _redraw(self, n_total: int) -> None:
        """Slot — always runs on the main/GUI thread."""
        for cond in self._conditions:
            buf = self._epoch_buf[cond]
            avg = np.mean(np.stack(buf, 0), 0) if buf else np.zeros((self._n_ch, self._n_t))
            for ch_i, curve in enumerate(self._curves[cond]):
                curve.setData(self._times, avg[ch_i])

            # Update per-condition title with trial count
            self._cond_titles[cond].setText(f"{cond}  (n = {self._n_per[cond]})")
            self._cond_n_lbl[cond].setText(f"n = {self._n_per[cond]}")

        self._total_lbl.setText(f"Total: {n_total} trials")

        if self._scale_sl.value() != 5:
            self._apply_y_range()
