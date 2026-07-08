"""Real-time scalp-layout ERP / evoked-potential display.

Live-updating equivalent of :func:`mne.viz.plot_evoked_topo`: channels
are placed at their exact 2-D scalp positions (from
:func:`mne.channels.find_layout`), using PyQtGraph's scene for absolute
positioning rather than a collapsible grid.  Redraws after every
:meth:`update` call as new epochs arrive from :class:`~mne_rt.RTEpochs`.

Classes
-------
TopoPlot
    Real-time scalp-layout ERP display with interactive sidebar.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

try:
    from qtpy.QtCore import QRectF, Qt, Signal
    from qtpy.QtGui import QColor, QFont
    from qtpy.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
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

_BG_PRESETS = [
    ("Dark", "#0d1117", _TEXT),
    ("Navy", "#050d1a", _TEXT),
    ("Slate", "#1e2030", _TEXT),
    ("Dim", "#2d333b", _TEXT),
    ("Light", "#f1f5f9", "#111827"),
]

_SIDEBAR_W = 230

_SW, _SH = 1000, 920
_PW, _PH = 76, 64

_MASTOID_NAMES = frozenset(["M1", "M2", "TP9", "TP10", "A1", "A2", "Mastoid", "mastoid"])


# ---------------------------------------------------------------------------
# Sidebar helpers
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


def _slider(parent, lo: int, hi: int, val: int) -> QSlider:
    sl = QSlider(Qt.Orientation.Horizontal, parent)
    sl.setRange(lo, hi)
    sl.setValue(val)
    return sl


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TopoPlot(QMainWindow):
    """Real-time scalp-layout ERP display.

    One mini :class:`pyqtgraph.PlotItem` per electrode, placed at the
    channel's true 2-D scalp position from
    :func:`mne.channels.find_layout`.  Condition averages (with optional
    ±1 SEM shading) are redrawn after every :meth:`update` call.

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
        When provided, :func:`mne.channels.find_layout` is called on
        this object for exact scalp positioning and channel-type detection
        (EEG → µV, MEG mag → fT, MEG grad → fT/cm).
        Pass ``epochs_stream.info`` from :class:`~mne_rt.RTEpochs`.
    montage : str, default "standard_1020"
        Fallback montage when ``info`` is not given or has no dig points.
    baseline : tuple or None, default (None, 0)
        Baseline interval — drawn as a shaded region.
    window_size : tuple of int, default (1440, 900)
        Initial window size in pixels.
    verbose : bool or str or None

    .. versionadded:: 1.0.0

    See Also
    --------
    mne_rt.RTEpochs : Drives this plot via :meth:`update`.
    mne_rt.viz.ButterflyPlot : All-channel overlay alternative.
    mne_rt.viz.CompareEvoked : Large per-channel view with SEM ribbons.
    """

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
                "A Qt binding (PyQt6 or PySide6) and pyqtgraph are required for TopoPlot.\n"
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
        self._linewidth = 1.6
        self._smooth_ms = 0.0
        self._show_sem = False
        self._plot_bg = _BG
        self._x_start = tmin
        self._x_end = tmax

        # Unit / re-reference
        self._unit, self._unit_scale = self._detect_unit(info)
        self._reref_mode = "none"
        self._mastoid_idx = self._find_mastoids()

        self._norm_pos = self._compute_positions(info, montage)

        self.setWindowTitle("MNE-RT — Topo ERP")
        self.resize(*window_size)
        self._apply_styles()
        self._build_ui()

        logger.info(
            "TopoPlot(ERP): %d ch, %.0f–%.0f ms, unit=%s, layout=%s",
            self._n_ch,
            tmin * 1000,
            tmax * 1000,
            self._unit,
            "from info" if info is not None else "montage/fallback",
        )

    # -----------------------------------------------------------------------
    # Unit / mastoid helpers
    # -----------------------------------------------------------------------

    def _detect_unit(self, info) -> tuple[str, float]:
        if info is None or not _mne_available:
            return "µV", 1e6
        try:
            ct = mne.channel_type(info, 0)
            if ct == "mag":
                return "fT", 1e15
            elif ct == "grad":
                return "fT/cm", 1e13
        except Exception:
            pass
        return "µV", 1e6

    def _find_mastoids(self) -> list[int]:
        return [i for i, ch in enumerate(self.ch_names) if ch in _MASTOID_NAMES]

    # -----------------------------------------------------------------------
    # Scalp layout
    # -----------------------------------------------------------------------

    def _compute_positions(self, info, montage_name: str) -> list[tuple[float, float]]:
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
                logger.debug("TopoPlot: montage layout failed: %s", exc)
        logger.warning("TopoPlot: falling back to circular layout.")
        return self._circular_fallback()

    def _from_layout(self, layout) -> Optional[list[tuple[float, float]]]:
        if layout is None:
            return None
        name_xy: dict[str, tuple[float, float]] = {}
        for name, pos in zip(layout.names, layout.pos):
            xc = float(pos[0] + pos[2] / 2.0)
            yc = float(pos[1] + pos[3] / 2.0)
            name_xy[name] = (xc, 1.0 - yc)
        n_matched = sum(1 for c in self.ch_names if c in name_xy)
        if n_matched < self._n_ch // 2:
            return None
        positions: list[tuple[float, float]] = []
        fb = 0
        for ch in self.ch_names:
            if ch in name_xy:
                positions.append(name_xy[ch])
            else:
                positions.append((0.02, fb * 0.05))
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
    # UI build
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
            QComboBox             {{ background:{_SURFACE}; color:{_TEXT};
                                     border:1px solid {_BORDER};
                                     border-radius:4px; padding:2px 6px;
                                     font-size:11px; }}
            QComboBox::drop-down  {{ border:none; width:20px; }}
            QScrollArea           {{ border: none; }}
            QScrollBar:vertical   {{ background:{_BG}; width:6px; }}
            QScrollBar::handle:vertical {{ background:{_BORDER}; border-radius:3px; }}
        """)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOptions(antialias=True, background=_BG, foreground=_DIM)

        self._gview = pg.GraphicsView()
        self._gview.setBackground(_BG)
        self._gview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._gview, stretch=1)
        self._scene = self._gview.sceneObj

        root.addWidget(self._build_sidebar())

        self._plots: list[pg.PlotItem] = []
        self._ch_labels: list[pg.TextItem] = []
        self._curves: dict[str, list[pg.PlotCurveItem]] = {c: [] for c in self._conditions}
        self._sem_upper: dict[str, list[pg.PlotCurveItem]] = {c: [] for c in self._conditions}
        self._sem_lower: dict[str, list[pg.PlotCurveItem]] = {c: [] for c in self._conditions}
        self._sem_fills: dict[str, list] = {c: [] for c in self._conditions}
        self._zl_items: list[pg.InfiniteLine] = []

        for ch_idx, ch in enumerate(self.ch_names):
            xn, yn = self._norm_pos[ch_idx]
            margin = 0.05
            cx = (margin + xn * (1.0 - 2 * margin)) * _SW
            cy = (margin + yn * (1.0 - 2 * margin)) * _SH

            plot = pg.PlotItem()
            plot.setGeometry(QRectF(cx - _PW / 2, cy - _PH / 2, _PW, _PH))
            self._scene.addItem(plot)
            lbl = self._style_plot(plot, ch)
            self._ch_labels.append(lbl)

            zl = pg.InfiniteLine(
                pos=0,
                angle=90,
                pen=pg.mkPen(_BORDER, width=1, style=Qt.PenStyle.DashLine),
            )
            plot.addItem(zl)
            self._zl_items.append(zl)

            for cond in self._conditions:
                col = self._cmap[cond]
                curve = plot.plot(
                    self._times,
                    np.zeros(self._n_t),
                    pen=pg.mkPen(col, width=self._linewidth),
                )
                self._curves[cond].append(curve)

                upper = plot.plot(self._times, np.zeros(self._n_t), pen=None)
                lower = plot.plot(self._times, np.zeros(self._n_t), pen=None)
                qcol = QColor(col)
                qcol.setAlpha(55)
                fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(qcol))
                fill.setVisible(False)
                plot.addItem(fill)
                self._sem_upper[cond].append(upper)
                self._sem_lower[cond].append(lower)
                self._sem_fills[cond].append(fill)

            self._plots.append(plot)

    def _style_plot(self, plot: pg.PlotItem, ch: str) -> pg.TextItem:
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setMouseEnabled(x=False, y=False)
        for ax in ("bottom", "left", "top", "right"):
            plot.showAxis(ax, False)
        plot.setContentsMargins(0, 0, 0, 0)
        lbl = pg.TextItem(ch, color=_DIM, anchor=(0, 1))
        lbl.setFont(QFont("Helvetica", 6))
        plot.addItem(lbl)
        lbl.setPos(self.tmin, 0)
        return lbl

    def _fit_view(self) -> None:
        if not self._plots:
            return
        rects = [p.geometry() for p in self._plots]
        x0 = min(r.x() for r in rects)
        y0 = min(r.y() for r in rects)
        x1 = max(r.x() + r.width() for r in rects)
        y1 = max(r.y() + r.height() for r in rects)
        pad_x = (x1 - x0) * 0.08
        pad_y = (y1 - y0) * 0.10
        self._gview.fitInView(
            QRectF(x0 - pad_x, y0 - pad_y, x1 - x0 + 2 * pad_x, y1 - y0 + 2 * pad_y),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._fit_view()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_view()

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
        hdr = QLabel("ERP CONTROLS")
        hdr.setStyleSheet(f"color:{_TEXT}; font-size:11px; font-weight:700; letter-spacing:1.5px;")
        ly.addWidget(hdr)

        # Unit badge next to header
        self._unit_lbl = QLabel(f"[{self._unit}]")
        self._unit_lbl.setStyleSheet(
            f"color:{_ACCENT}; font-size:10px; font-weight:600; "
            f"background:{_SURFACE}; border:1px solid {_BORDER}; "
            "border-radius:3px; padding:1px 5px;"
        )
        row_hdr, lhdr = _row(sb)
        lhdr.addWidget(hdr, stretch=1)
        lhdr.addWidget(self._unit_lbl)
        ly.addWidget(row_hdr)
        ly.addWidget(_sep(sb))

        # ── CONDITIONS ───────────────────────────────────────────────────
        ly.addWidget(_section("CONDITIONS", sb))
        self._cond_checks: dict[str, QCheckBox] = {}
        self._cond_n_lbl: dict[str, QLabel] = {}

        for cond in self._conditions:
            col = self._cmap[cond]
            rw, rl = _row(sb)
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
            rl.addWidget(cb)
            rl.addWidget(dot, stretch=1)
            rl.addWidget(n_lbl)
            ly.addWidget(rw)

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
        self._yscale = 1.0

        r2, l2 = _row(sb)
        l2.addWidget(_key_lbl("Line width", sb), stretch=1)
        self._lw_lbl = _val_lbl("1.5", sb)
        l2.addWidget(self._lw_lbl)
        ly.addWidget(r2)
        self._lw_sl = _slider(sb, 1, 8, 3)
        self._lw_sl.valueChanged.connect(self._on_linewidth)
        ly.addWidget(self._lw_sl)

        rbt, lbt = _row(sb, 6)
        auto_btn = QPushButton("Auto scale")
        auto_btn.clicked.connect(self._auto_scale)
        lbt.addWidget(auto_btn)
        ly.addWidget(rbt)

        ly.addWidget(_sep(sb))

        # ── SMOOTHING ────────────────────────────────────────────────────
        ly.addWidget(_section("SMOOTHING", sb))
        r3, l3 = _row(sb)
        l3.addWidget(_key_lbl("Window", sb), stretch=1)
        self._sm_lbl = _val_lbl("Off", sb)
        l3.addWidget(self._sm_lbl)
        ly.addWidget(r3)
        self._smooth_sl = _slider(sb, 0, 50, 0)
        self._smooth_sl.valueChanged.connect(self._on_smooth)
        ly.addWidget(self._smooth_sl)

        ly.addWidget(_sep(sb))

        # ── RE-REFERENCE ─────────────────────────────────────────────────
        ly.addWidget(_section("RE-REFERENCE", sb))

        self._reref_cb = QComboBox(sb)
        self._reref_cb.addItem("None (raw)")
        self._reref_cb.addItem("Average reference")
        mastoid_names = [self.ch_names[i] for i in self._mastoid_idx]
        if mastoid_names:
            self._reref_cb.addItem(f"Mastoids ({', '.join(mastoid_names)})")
        else:
            self._reref_cb.addItem("Mastoids (not found)")
            self._reref_cb.model().item(2).setEnabled(False)
        self._reref_cb.currentIndexChanged.connect(self._on_reref)
        ly.addWidget(self._reref_cb)

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

        ly.addWidget(_key_lbl("Background", sb))
        sw_row, sw_l = _row(sb, 5)
        self._bg_swatches: list[QPushButton] = []
        for label, hex_col, _ in _BG_PRESETS:
            btn = QPushButton()
            btn.setFixedSize(28, 22)
            btn.setToolTip(label)
            active = hex_col == _BG
            border = f"2px solid {_ACCENT}" if active else f"1px solid {_BORDER}"
            btn.setStyleSheet(f"background:{hex_col}; border:{border}; border-radius:4px;")
            btn.clicked.connect(lambda _, c=hex_col: self._set_bg(c))
            sw_l.addWidget(btn)
            self._bg_swatches.append(btn)
        sw_l.addStretch()
        ly.addWidget(sw_row)
        ly.addSpacing(4)

        self._sem_chk = QCheckBox("±1 SEM shading")
        self._sem_chk.setChecked(False)
        self._sem_chk.toggled.connect(self._toggle_sem)
        ly.addWidget(self._sem_chk)

        self._labels_chk = QCheckBox("Channel labels")
        self._labels_chk.setChecked(True)
        self._labels_chk.toggled.connect(self._toggle_labels)
        ly.addWidget(self._labels_chk)

        self._zl_chk = QCheckBox("Stimulus line")
        self._zl_chk.setChecked(True)
        self._zl_chk.toggled.connect(self._toggle_zl)
        ly.addWidget(self._zl_chk)

        ly.addWidget(_sep(sb))

        # ── DATA ─────────────────────────────────────────────────────────
        ly.addWidget(_section("DATA", sb))
        self._total_lbl = QLabel("Total: 0 trials")
        self._total_lbl.setStyleSheet(f"color:{_TEXT};font-size:11px;")
        ly.addWidget(self._total_lbl)
        ly.addSpacing(4)

        reset_btn = QPushButton("Reset epochs")
        reset_btn.setToolTip("Clear all accumulated epochs")
        reset_btn.clicked.connect(self._reset_epochs)
        ly.addWidget(reset_btn)

        export_btn = QPushButton("Export PNG …")
        export_btn.setToolTip("Save current plot as PNG")
        export_btn.clicked.connect(self._export_png)
        ly.addWidget(export_btn)

        ly.addStretch()
        scroll.setWidget(sb)
        return scroll

    # -----------------------------------------------------------------------
    # Sidebar callbacks
    # -----------------------------------------------------------------------

    def _toggle_cond(self, cond: str, visible: bool) -> None:
        for c in self._curves[cond]:
            c.setVisible(visible)
        for f in self._sem_fills[cond]:
            f.setVisible(visible and self._show_sem)

    def _on_scale(self, value: int) -> None:
        self._yscale = value / 5.0
        self._sv_lbl.setText(f"×{self._yscale:.1f}")
        self._apply_y_range()

    def _apply_y_range(self) -> None:
        avgs = []
        for cond in self._conditions:
            if self._cond_checks.get(cond, QCheckBox()).isChecked():
                buf = self._epoch_buf[cond]
                if buf:
                    avg = np.mean(np.stack(buf, 0), 0) * self._unit_scale
                    avgs.append(self._apply_reref(avg))
        if not avgs:
            return
        amp = float(np.percentile(np.abs(np.stack(avgs, 0)), 99)) or 1e-12
        half = amp * self._yscale
        for plot in self._plots:
            plot.setYRange(-half, half, padding=0.05)

    def _auto_scale(self) -> None:
        self._scale_sl.setValue(5)  # triggers _on_scale → _apply_y_range

    def _on_linewidth(self, value: int) -> None:
        self._linewidth = value * 0.5
        self._lw_lbl.setText(f"{self._linewidth:.1f}")
        for cond in self._conditions:
            col = self._cmap[cond]
            for curve in self._curves[cond]:
                curve.setPen(pg.mkPen(col, width=self._linewidth))

    def _on_smooth(self, value: int) -> None:
        self._smooth_ms = float(value)
        self._sm_lbl.setText("Off" if value == 0 else f"{value} ms")
        total = sum(self._n_per.values())
        if total > 0:
            self._redraw(total)

    def _smooth(self, y: np.ndarray) -> np.ndarray:
        if self._smooth_ms <= 0:
            return y
        n = max(1, int(self._smooth_ms * 1e-3 * self.sfreq))
        if n < 2:
            return y
        return np.convolve(y, np.ones(n) / n, mode="same")

    def _on_reref(self, index: int) -> None:
        modes = ["none", "average", "mastoids"]
        self._reref_mode = modes[index] if index < len(modes) else "none"
        total = sum(self._n_per.values())
        if total > 0:
            self._redraw(total)

    def _apply_reref(self, avg: np.ndarray) -> np.ndarray:
        """Apply re-referencing to avg (n_ch, n_times). Returns copy."""
        if self._reref_mode == "average":
            return avg - avg.mean(0, keepdims=True)
        if self._reref_mode == "mastoids" and self._mastoid_idx:
            ref = avg[self._mastoid_idx].mean(0)
            return avg - ref
        return avg

    def _on_xrange(self) -> None:
        x1 = self._xstart_sl.value() / 1000.0
        x2 = self._xend_sl.value() / 1000.0
        if x1 >= x2:
            return
        self._x_start = x1
        self._x_end = x2
        self._xstart_lbl.setText(f"{int(x1 * 1000)} ms")
        self._xend_lbl.setText(f"{int(x2 * 1000)} ms")
        for plot in self._plots:
            plot.setXRange(x1, x2, padding=0)

    def _set_bg(self, color: str) -> None:
        self._plot_bg = color
        self._gview.setBackground(color)
        for plot in self._plots:
            plot.getViewBox().setBackgroundColor(color)
        for btn, (_, hex_col, _) in zip(self._bg_swatches, _BG_PRESETS):
            active = hex_col == color
            border = f"2px solid {_ACCENT}" if active else f"1px solid {_BORDER}"
            btn.setStyleSheet(f"background:{hex_col}; border:{border}; border-radius:4px;")

    def _toggle_sem(self, visible: bool) -> None:
        self._show_sem = visible
        for cond in self._conditions:
            checked = self._cond_checks.get(cond, QCheckBox()).isChecked()
            for f in self._sem_fills[cond]:
                f.setVisible(visible and checked)
        if visible:
            total = sum(self._n_per.values())
            if total > 0:
                self._redraw(total)

    def _toggle_labels(self, visible: bool) -> None:
        for lbl in self._ch_labels:
            lbl.setVisible(visible)

    def _toggle_zl(self, v: bool) -> None:
        for item in self._zl_items:
            item.setVisible(v)

    def _reset_epochs(self) -> None:
        for cond in self._conditions:
            self._epoch_buf[cond] = []
            self._n_per[cond] = 0
            self._cond_n_lbl[cond].setText("n = 0")
            for curve in self._curves[cond]:
                curve.setData(self._times, np.zeros(self._n_t))
            for upper, lower, fill in zip(
                self._sem_upper[cond], self._sem_lower[cond], self._sem_fills[cond]
            ):
                upper.setData(self._times, np.zeros(self._n_t))
                lower.setData(self._times, np.zeros(self._n_t))
                fill.setVisible(False)
        self._total_lbl.setText("Total: 0 trials")

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Topo ERP Plot",
            "topo_plot.png",
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
        """Redraw all channel plots with updated condition averages.

        Thread-safe — may be called from the acquisition thread.

        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_channels, n_times)
            All accepted epochs so far.
        conditions : list of str
            Condition label for each epoch; ``len(conditions) == data.shape[0]``.
        """
        n_total = len(conditions)
        for cond in self._conditions:
            mask = np.array([c == cond for c in conditions])
            self._epoch_buf[cond] = list(data[mask]) if mask.any() else []
            self._n_per[cond] = int(mask.sum())
        self._redraw_sig.emit(n_total)

    def _redraw(self, n_total: int) -> None:
        self._total_lbl.setText(f"Total: {n_total} trials")

        for cond in self._conditions:
            self._cond_n_lbl[cond].setText(f"n = {self._n_per[cond]}")
            buf = self._epoch_buf[cond]
            n = len(buf)

            if buf:
                stack = np.stack(buf, 0)
                avg = np.mean(stack, 0)  # (n_ch, n_t)
                sem = np.std(stack, 0, ddof=1) / np.sqrt(n) if n >= 2 else np.zeros_like(avg)
            else:
                avg = np.zeros((self._n_ch, self._n_t))
                sem = np.zeros_like(avg)

            # Apply re-reference then unit scaling
            avg = self._apply_reref(avg) * self._unit_scale
            sem = sem * self._unit_scale

            checked = self._cond_checks.get(cond, QCheckBox()).isChecked()

            for ch_i, curve in enumerate(self._curves[cond]):
                y = avg[ch_i]
                if len(y) != self._n_t:
                    y = np.interp(self._times, np.linspace(self.tmin, self.tmax, len(y)), y)
                curve.setData(self._times, self._smooth(y))

                if self._show_sem and n >= 2:
                    s = sem[ch_i]
                    self._sem_upper[cond][ch_i].setData(self._times, y + s)
                    self._sem_lower[cond][ch_i].setData(self._times, y - s)
                    self._sem_fills[cond][ch_i].setVisible(checked)

        self._apply_y_range()
