"""Real-time multi-condition evoked comparison for selected channels.

Shows N user-selected channels as large individual plots with all
conditions overlaid, ±1 SEM shading, visible time/amplitude axes, and
auto-detected peak markers.  Channels are chosen by clicking on a
mini scalp-topomap in the sidebar.  Redraws after every :meth:`update`
call as new epochs arrive from :class:`~mne_rt.RTEpochs`.

Classes
-------
CompareEvoked
    Real-time per-channel condition-overlay display with SEM shading,
    peak detection, and interactive topomap channel selector.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

try:
    from qtpy.QtCore import Qt, Signal
    from qtpy.QtGui import QFont, QColor
    from qtpy.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QVBoxLayout, QHBoxLayout, QLabel,
        QCheckBox, QSlider, QPushButton,
        QFrame, QSizePolicy, QScrollArea,
        QFileDialog,
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
_BG      = "#0d1117"
_SURFACE = "#161b22"
_BORDER  = "#30363d"
_TEXT    = "#e6edf3"
_DIM     = "#8b949e"
_ACCENT  = "#3b82f6"

_COND_COLORS = [
    "#3b82f6",   # blue
    "#ec4899",   # pink
    "#10b981",   # green
    "#f59e0b",   # amber
    "#8b5cf6",   # violet
    "#06b6d4",   # cyan
]

# Auto-channel preference list (case-insensitive matching)
_PREF_CHANNELS = ["cz", "pz", "oz", "fz", "fcz", "cpz"]

_SIDEBAR_W = 210   # px


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
        f"color:{_DIM}; font-size:10px; font-weight:700; "
        "letter-spacing:1px; padding-top:6px;"
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
    lbl.setStyleSheet(
        f"color:{color}; font-size:11px; font-weight:600;"
    )
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
# Unit/scale helpers
# ---------------------------------------------------------------------------

def _detect_unit(info, ch_names: list[str]) -> tuple[str, float]:
    if info is None or not _mne_available:
        return ("µV", 1e6)
    try:
        ch_type = mne.channel_type(info, 0)
        if ch_type == "eeg":
            return ("µV", 1e6)
        elif ch_type == "mag":
            return ("fT", 1e15)
        elif ch_type == "grad":
            return ("fT/cm", 1e13)
        else:
            return ("µV", 1e6)
    except Exception:
        return ("µV", 1e6)


def _auto_channels(ch_names: list[str]) -> list[str]:
    lower_to_orig = {ch.lower(): ch for ch in ch_names}
    selected: list[str] = []
    for pref in _PREF_CHANNELS:
        if pref in lower_to_orig:
            selected.append(lower_to_orig[pref])
        if len(selected) == 3:
            break
    if not selected:
        selected = ch_names[:3]
    return selected


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CompareEvoked(QMainWindow):
    """Real-time per-channel condition overlay with SEM shading and peak markers.

    Shows N user-selected channels as large individual :class:`pyqtgraph.PlotItem`
    rows.  Each plot overlays all conditions with solid curves, ±1 SEM shading,
    and a scatter point marking the peak latency in the post-stimulus window.

    Channels are chosen interactively via a clickable mini scalp-topomap
    in the sidebar: click any electrode dot to add or remove it from the
    display (up to :data:`_MAX_DISP_CH` channels simultaneously).

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
    channels : list of str or None
        Channels shown on startup.  If ``None``, auto-selects from
        ``['Cz','Pz','Oz','Fz','FCz','CPz']`` or the first 3 channels.
    info : mne.Info or None
        Used for unit/scale detection and scalp layout.
    montage : str, default ``"standard_1020"``
        Fallback montage for electrode positions when *info* has no dig.
    baseline : tuple or None, default ``(None, 0)``
        Baseline correction interval (informational only).
    window_size : tuple of int, default ``(1200, 800)``
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
        ch_names:    list[str],
        sfreq:       float,
        tmin:        float,
        tmax:        float,
        event_id:    dict[str, int],
        channels:    Optional[list[str]] = None,
        info=None,
        montage:     str = "standard_1020",
        baseline:    Optional[tuple] = (None, 0),
        window_size: tuple[int, int] = (1200, 800),
        verbose:     Union[bool, str, None] = None,
    ) -> None:
        if not _qt_available or not _pg_available:
            raise ImportError(
                "A Qt binding (PyQt6 or PySide6) and pyqtgraph are required for CompareEvoked.\n"
                "Install with: pip install 'mne-rt[full]'"
            )
        _app = QApplication.instance() or QApplication([])  # noqa: F841

        super().__init__()
        self._redraw_sig.connect(self._redraw)
        set_log_level(verbose)

        self.ch_names   = list(ch_names)
        self.sfreq      = sfreq
        self.tmin       = tmin
        self.tmax       = tmax
        self.event_id   = event_id
        self.montage    = montage
        self.baseline   = baseline
        self._info      = info

        self._conditions = list(event_id.keys())
        self._cmap       = {
            c: _COND_COLORS[i % len(_COND_COLORS)]
            for i, c in enumerate(self._conditions)
        }
        self._n_ch   = len(ch_names)
        self._n_t    = int(round((tmax - tmin) * sfreq)) + 1
        self._times  = np.linspace(tmin, tmax, self._n_t)

        # Display channels (mutable — changed by topomap clicks)
        if channels is None:
            self._disp_channels: list[str] = _auto_channels(self.ch_names)
        else:
            self._disp_channels = [ch for ch in channels if ch in self.ch_names]
            if not self._disp_channels:
                logger.warning(
                    "CompareEvoked: none of the requested channels found; "
                    "falling back to auto-selection."
                )
                self._disp_channels = _auto_channels(self.ch_names)
        # (no hard cap — let the user decide via the topomap)

        # Index in ch_names for each displayed channel
        self._disp_idx: list[int] = [
            self.ch_names.index(ch) for ch in self._disp_channels
        ]

        # Unit / scale
        self._unit_label, self._unit_scale = _detect_unit(info, self.ch_names)

        # Epoch buffer (always over all conditions)
        self._epoch_buf: dict[str, list[np.ndarray]] = {
            c: [] for c in self._conditions
        }
        self._n_per: dict[str, int] = {c: 0 for c in self._conditions}

        # Display state
        self._yscale    = 1.0
        self._show_sem  = True
        self._show_peak = True

        # Scalp positions for all channels (normalised, yn=0=frontal)
        self._norm_pos = self._compute_positions(info, montage)

        # Topomap scatter item (assigned in _build_topo_widget)
        self._topo_scatter: Optional[pg.ScatterPlotItem] = None

        # Canvas data structures (rebuilt by _rebuild_canvas)
        self._ch_plots:   dict[str, pg.PlotItem]               = {}
        self._curves:     dict[str, dict[str, pg.PlotCurveItem]] = {}
        self._sem_upper:  dict[str, dict[str, pg.PlotCurveItem]] = {}
        self._sem_lower:  dict[str, dict[str, pg.PlotCurveItem]] = {}
        self._sem_fills:  dict[str, dict[str, pg.FillBetweenItem]] = {}
        self._peaks:      dict[str, dict[str, pg.ScatterPlotItem]] = {}
        self._t0_lines:   list[pg.InfiniteLine]                = []

        self.setWindowTitle("MNE-RT — Compare Evoked")
        self.resize(*window_size)
        self._apply_styles()
        self._build_ui()

        logger.info(
            "CompareEvoked: %d channels displayed (%s), %d conditions, "
            "unit=%s",
            len(self._disp_channels),
            ", ".join(self._disp_channels),
            len(self._conditions),
            self._unit_label,
        )

    # -----------------------------------------------------------------------
    # Scalp layout
    # -----------------------------------------------------------------------

    def _compute_positions(
        self, info, montage_name: str
    ) -> list[tuple[float, float]]:
        """Return normalised (xn, yn) for each channel, yn=0=frontal."""
        if _mne_available:
            if info is not None:
                pos = self._from_layout(mne.channels.find_layout(info))
                if pos is not None:
                    return pos
            try:
                tmp = mne.create_info(
                    self.ch_names, sfreq=1.0, ch_types="eeg", verbose=False
                )
                mont = mne.channels.make_standard_montage(montage_name)
                tmp.set_montage(mont, on_missing="ignore", verbose=False)
                pos = self._from_layout(mne.channels.find_layout(tmp))
                if pos is not None:
                    return pos
            except Exception as exc:
                logger.debug("CompareEvoked: montage layout failed: %s", exc)
        logger.warning("CompareEvoked: falling back to circular layout.")
        return self._circular_fallback()

    def _from_layout(self, layout) -> Optional[list[tuple[float, float]]]:
        if layout is None:
            return None
        name_xy: dict[str, tuple[float, float]] = {}
        for name, pos in zip(layout.names, layout.pos):
            xc = float(pos[0] + pos[2] / 2.0)
            yc = float(pos[1] + pos[3] / 2.0)
            name_xy[name] = (xc, 1.0 - yc)   # yn=0 = frontal
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
            (0.5 + 0.42 * math.cos(2 * math.pi * i / n - math.pi / 2),
             0.5 + 0.42 * math.sin(2 * math.pi * i / n - math.pi / 2))
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

        pg.setConfigOptions(antialias=True, background=_BG, foreground=_DIM)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(_BG)
        self._glw.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._glw, stretch=1)
        root.addWidget(self._build_sidebar())

        self._rebuild_canvas()

    def _rebuild_canvas(self) -> None:
        """Create or recreate PlotItems for the current _disp_channels."""
        self._glw.clear()

        self._ch_plots  = {}
        self._curves    = {}
        self._sem_upper = {}
        self._sem_lower = {}
        self._sem_fills = {}
        self._peaks     = {}
        self._t0_lines  = []
        self._disp_idx  = [self.ch_names.index(ch) for ch in self._disp_channels]

        n_rows   = len(self._disp_channels)
        times_ms = self._times * 1000.0

        for row_idx, ch in enumerate(self._disp_channels):
            is_last = row_idx == n_rows - 1

            plot = self._glw.addPlot(row=row_idx, col=0)
            self._style_plot(plot, ch, is_last)
            self._ch_plots[ch] = plot

            t0_line = pg.InfiniteLine(
                pos=0.0, angle=90,
                pen=pg.mkPen(_BORDER, width=1, style=Qt.PenStyle.DashLine),
            )
            plot.addItem(t0_line)
            self._t0_lines.append(t0_line)

            self._curves[ch]    = {}
            self._sem_upper[ch] = {}
            self._sem_lower[ch] = {}
            self._sem_fills[ch] = {}
            self._peaks[ch]     = {}

            for cond in self._conditions:
                col  = self._cmap[cond]
                qcol = QColor(col)
                r, g, b = qcol.red(), qcol.green(), qcol.blue()

                curve = pg.PlotCurveItem(
                    times_ms, np.zeros(self._n_t),
                    pen=pg.mkPen(col, width=1.8), antialias=True,
                )
                plot.addItem(curve)
                self._curves[ch][cond] = curve

                sem_up = pg.PlotCurveItem(
                    times_ms, np.zeros(self._n_t),
                    pen=pg.mkPen((r, g, b, 0), width=0.5),
                )
                sem_lo = pg.PlotCurveItem(
                    times_ms, np.zeros(self._n_t),
                    pen=pg.mkPen((r, g, b, 0), width=0.5),
                )
                plot.addItem(sem_up)
                plot.addItem(sem_lo)
                self._sem_upper[ch][cond] = sem_up
                self._sem_lower[ch][cond] = sem_lo

                fill = pg.FillBetweenItem(
                    sem_up, sem_lo, brush=pg.mkBrush(r, g, b, 40),
                )
                fill.setVisible(False)
                plot.addItem(fill)
                self._sem_fills[ch][cond] = fill

                scatter = pg.ScatterPlotItem(
                    size=8, pen=pg.mkPen(col, width=1.5),
                    brush=pg.mkBrush(r, g, b, 200), symbol="o",
                )
                scatter.setVisible(False)
                plot.addItem(scatter)
                self._peaks[ch][cond] = scatter

        for row_idx in range(n_rows):
            self._glw.ci.layout.setRowStretchFactor(row_idx, 1)

        self._update_x_ticks()

        # If data already accumulated, refresh immediately
        total = sum(self._n_per.values())
        if total > 0:
            self._redraw(total)

    def _style_plot(
        self, plot: pg.PlotItem, ch_name: str, is_last: bool
    ) -> None:
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setMouseEnabled(x=False, y=False)
        plot.getViewBox().setBackgroundColor(_BG)

        plot.showAxis("left", True)
        left_ax = plot.getAxis("left")
        left_ax.setStyle(tickLength=4, showValues=False)
        left_ax.setPen(pg.mkPen(_BORDER, width=1))
        left_ax.setTextPen(pg.mkPen(_DIM))

        plot.showAxis("top", False)
        plot.showAxis("right", False)
        plot.showAxis("bottom", is_last)
        if is_last:
            bottom_ax = plot.getAxis("bottom")
            bottom_ax.setPen(pg.mkPen(_BORDER, width=1))
            bottom_ax.setTextPen(pg.mkPen(_DIM))
            bottom_ax.setLabel("Time (ms)", color=_DIM)

        plot.setContentsMargins(0, 0, 0, 0)
        plot.setXRange(self.tmin * 1000.0, self.tmax * 1000.0, padding=0)

        title = pg.TextItem(ch_name, color=_TEXT, anchor=(0, 0))
        title.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
        plot.addItem(title)
        title.setPos(self.tmin * 1000.0, 0)

    # -----------------------------------------------------------------------
    # Custom time axis ticks
    # -----------------------------------------------------------------------

    def _update_x_ticks(self) -> None:
        if not self._disp_channels:
            return
        last_ch    = self._disp_channels[-1]
        plot       = self._ch_plots[last_ch]
        bottom_ax  = plot.getAxis("bottom")
        tmin_ms    = self.tmin * 1000.0
        tmax_ms    = self.tmax * 1000.0
        span_ms    = tmax_ms - tmin_ms

        for interval in [25, 50, 100, 200, 250, 500]:
            if 4 <= span_ms / interval <= 10:
                break
        else:
            interval = 100

        start = math.ceil(tmin_ms / interval) * interval
        ticks: list[tuple[float, str]] = []
        t = start
        while t <= tmax_ms + 1e-6:
            ticks.append((t, str(int(t))))
            t += interval
        bottom_ax.setTicks([ticks])

    # -----------------------------------------------------------------------
    # Topomap channel selector
    # -----------------------------------------------------------------------

    def _build_topo_widget(self, parent: QWidget) -> pg.PlotWidget:
        """Return a 184×184 pyqtgraph PlotWidget with clickable electrode dots."""
        pw = pg.PlotWidget(parent=parent)
        pw.setFixedSize(184, 184)
        pw.setBackground(_SURFACE)
        pw.hideAxis("bottom")
        pw.hideAxis("left")
        pw.getViewBox().setMouseEnabled(x=False, y=False)
        pw.getViewBox().setAspectLocked(True)
        # View range with a small margin so the nose isn't clipped
        pw.getViewBox().setRange(
            xRange=(-0.06, 1.06), yRange=(-0.06, 1.14), padding=0,
        )

        # ── Head circle ──────────────────────────────────────────────────
        theta = np.linspace(0, 2 * np.pi, 160)
        cx = 0.5 + 0.48 * np.cos(theta)
        cy = 0.5 + 0.48 * np.sin(theta)
        pw.plot(cx, cy, pen=pg.mkPen(_BORDER, width=1.5))

        # ── Nose (small triangle at top, y > 0.98) ───────────────────────
        nose_x = [0.47, 0.5, 0.53, 0.47]
        nose_y = [0.97, 1.06, 0.97, 0.97]
        pw.plot(nose_x, nose_y, pen=pg.mkPen(_BORDER, width=1.2))

        # ── Left/right ear bumps ─────────────────────────────────────────
        for side in (-1, 1):
            ear_x_vals = np.linspace(0.48 * side, 0.56 * side, 8)
            ear_y_vals = 0.5 + 0.06 * np.sin(np.linspace(0, np.pi, 8))
            pw.plot(
                0.5 + ear_x_vals, ear_y_vals,
                pen=pg.mkPen(_BORDER, width=1.2),
            )

        # ── Channel dots ─────────────────────────────────────────────────
        # yn=0 = frontal = top of display (y large in pg's y-up coords)
        spots = []
        for i, ch in enumerate(self.ch_names):
            xn, yn = self._norm_pos[i]
            # Map into the circle: keep within 0.5±0.45
            tx = 0.5 + (xn - 0.5) * 0.9
            ty = 0.5 + (yn - 0.5) * 0.9   # yn=0→top, yn=1→bottom; y-axis: 0=bottom
            # pg default: y increases upward, so we flip yn
            ty = 1.0 - ty  # now ty=1 → frontal top, ty=0 → occipital bottom
            selected = ch in self._disp_channels
            spots.append({
                "pos": (tx, ty),
                "data": ch,
                "brush": pg.mkBrush(_ACCENT if selected else _BORDER),
                "pen": pg.mkPen(None),
                "size": 10 if selected else 6,
            })

        self._topo_scatter = pg.ScatterPlotItem(
            spots=spots, hoverable=True,
            tip=lambda x, y, data: str(data) if data else "",
        )
        self._topo_scatter.sigClicked.connect(self._on_topo_click)
        pw.addItem(self._topo_scatter)

        # ── Hint text ────────────────────────────────────────────────────
        hint = pg.TextItem("click to select", color=_DIM, anchor=(0.5, 1))
        hint.setFont(QFont("Helvetica", 7))
        hint.setPos(0.5, -0.02)
        pw.addItem(hint)

        return pw

    def _update_topo_colors(self) -> None:
        """Refresh dot appearance after selection changes."""
        if self._topo_scatter is None:
            return
        spots = []
        for i, ch in enumerate(self.ch_names):
            xn, yn = self._norm_pos[i]
            tx = 0.5 + (xn - 0.5) * 0.9
            ty = 1.0 - (0.5 + (yn - 0.5) * 0.9)
            selected = ch in self._disp_channels
            spots.append({
                "pos": (tx, ty),
                "data": ch,
                "brush": pg.mkBrush(_ACCENT if selected else _BORDER),
                "pen": pg.mkPen(None),
                "size": 10 if selected else 6,
            })
        self._topo_scatter.setData(spots=spots)

    def _on_topo_click(self, *args) -> None:
        """Handle a click on the topomap scatter plot.

        PyQtGraph passes ``(scatter, points, event)``; we accept ``*args``
        for version compatibility.
        """
        # points is the second-to-last argument across pg versions
        points = args[-2] if len(args) >= 2 else args[0]
        for pt in points:
            ch = pt.data()
            if ch is None or ch not in self.ch_names:
                continue
            if ch in self._disp_channels:
                if len(self._disp_channels) > 1:   # always keep at least 1
                    self._disp_channels.remove(ch)
            else:
                self._disp_channels.append(ch)
        self._update_topo_colors()
        # Update title
        self._sel_lbl.setText(
            f"{', '.join(self._disp_channels)}"
        )
        self._rebuild_canvas()

    # -----------------------------------------------------------------------
    # Sidebar build
    # -----------------------------------------------------------------------

    def _build_sidebar(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setFixedWidth(_SIDEBAR_W + 14)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background:{_SURFACE}; "
            f"border-left:1px solid {_BORDER}; }}"
        )

        sb = QWidget()
        sb.setStyleSheet(f"background:{_SURFACE};")
        ly = QVBoxLayout(sb)
        ly.setSpacing(4)
        ly.setContentsMargins(10, 12, 10, 12)

        # ── Header ───────────────────────────────────────────────────────
        hdr = QLabel("COMPARE EVOKED")
        hdr.setStyleSheet(
            f"color:{_TEXT}; font-size:11px; font-weight:700; letter-spacing:1.5px;"
        )
        ly.addWidget(hdr)
        ly.addWidget(_sep(sb))

        # ── CHANNEL SELECTOR (topomap) ────────────────────────────────────
        ly.addWidget(_section("CHANNELS", sb))

        # Hint: how many selected / max
        self._sel_lbl = QLabel(", ".join(self._disp_channels))
        self._sel_lbl.setStyleSheet(
            f"color:{_ACCENT}; font-size:10px; font-weight:600;"
        )
        self._sel_lbl.setWordWrap(True)
        ly.addWidget(self._sel_lbl)

        cap_lbl = QLabel("click to toggle")
        cap_lbl.setStyleSheet(f"color:{_DIM}; font-size:9px;")
        ly.addWidget(cap_lbl)

        ly.addSpacing(4)
        topo_w = self._build_topo_widget(sb)
        # Centre the topomap in the sidebar
        topo_row = QWidget(sb)
        topo_lay = QHBoxLayout(topo_row)
        topo_lay.setContentsMargins(0, 0, 0, 0)
        topo_lay.addStretch()
        topo_lay.addWidget(topo_w)
        topo_lay.addStretch()
        ly.addWidget(topo_row)

        ly.addWidget(_sep(sb))

        # ── CONDITIONS ───────────────────────────────────────────────────
        ly.addWidget(_section("CONDITIONS", sb))
        self._cond_checks: dict[str, QCheckBox] = {}
        self._cond_n_lbl:  dict[str, QLabel]    = {}

        for cond in self._conditions:
            col = self._cmap[cond]
            row_w, row_l = _row(sb)
            cb = QCheckBox()
            cb.setChecked(True)
            cb.setStyleSheet(
                f"QCheckBox::indicator:checked{{"
                f"background:{col};border-color:{col};}}"
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

        # ── DISPLAY ──────────────────────────────────────────────────────
        ly.addWidget(_section("DISPLAY", sb))

        self._sem_chk = QCheckBox("SEM shading")
        self._sem_chk.setChecked(True)
        self._sem_chk.toggled.connect(self._toggle_sem)
        ly.addWidget(self._sem_chk)

        self._peak_chk = QCheckBox("Peak markers")
        self._peak_chk.setChecked(True)
        self._peak_chk.toggled.connect(self._toggle_peaks)
        ly.addWidget(self._peak_chk)

        ly.addSpacing(4)

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

        # ── DATA ─────────────────────────────────────────────────────────
        ly.addWidget(_section("DATA", sb))

        self._total_lbl = QLabel("Total: 0 trials")
        self._total_lbl.setStyleSheet(f"color:{_TEXT};font-size:11px;")
        ly.addWidget(self._total_lbl)

        ly.addSpacing(4)

        export_btn = QPushButton("Export PNG …")
        export_btn.setToolTip("Save the current compare-evoked plot as a PNG image")
        export_btn.clicked.connect(self._export_png)
        ly.addWidget(export_btn)

        ly.addStretch()
        scroll.setWidget(sb)
        return scroll

    # -----------------------------------------------------------------------
    # Sidebar callbacks
    # -----------------------------------------------------------------------

    def _toggle_cond(self, cond: str, visible: bool) -> None:
        for ch in self._disp_channels:
            self._curves[ch][cond].setVisible(visible)
            if visible and self._show_sem:
                self._sem_fills[ch][cond].setVisible(True)
            else:
                self._sem_fills[ch][cond].setVisible(False)
            if not visible:
                self._peaks[ch][cond].setVisible(False)

    def _toggle_sem(self, visible: bool) -> None:
        self._show_sem = visible
        for ch in self._disp_channels:
            for cond in self._conditions:
                cond_vis = self._cond_checks.get(cond, QCheckBox()).isChecked()
                self._sem_fills[ch][cond].setVisible(visible and cond_vis)

    def _toggle_peaks(self, visible: bool) -> None:
        self._show_peak = visible
        for ch in self._disp_channels:
            for cond in self._conditions:
                cond_vis = self._cond_checks.get(cond, QCheckBox()).isChecked()
                self._peaks[ch][cond].setVisible(
                    visible and cond_vis and bool(self._epoch_buf[cond])
                )

    def _on_scale(self, value: int) -> None:
        self._yscale = value * 0.2
        self._sv_lbl.setText(f"×{self._yscale:.1f}")
        self._apply_y_range()

    def _apply_y_range(self) -> None:
        avgs_scaled: list[np.ndarray] = []
        for cond in self._conditions:
            if self._cond_checks.get(cond, QCheckBox()).isChecked():
                buf = self._epoch_buf[cond]
                if buf:
                    avg = np.mean(np.stack(buf, 0), 0) * self._unit_scale
                    avgs_scaled.append(avg)
        if not avgs_scaled:
            return
        amp = float(np.percentile(np.abs(np.stack(avgs_scaled, 0)), 99)) or 1e-12
        half = amp * self._yscale
        for plot in self._ch_plots.values():
            if plot.isVisible():
                plot.setYRange(-half, half, padding=0.05)

    def _auto_scale(self) -> None:
        self._scale_sl.setValue(5)
        for plot in self._ch_plots.values():
            if plot.isVisible():
                plot.enableAutoRange(axis="y")

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Compare Evoked", "compare_evoked.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)",
        )
        if path:
            self.grab().save(path)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(
        self,
        data:       np.ndarray,
        conditions: list[str],
    ) -> None:
        """Redraw all channel plots with updated condition averages.

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
            self._n_per[cond]     = int(mask.sum())

        self._redraw_sig.emit(n_total)

    def _redraw(self, n_total: int) -> None:
        """Slot — always runs on the main/GUI thread."""
        times_ms = self._times * 1000.0
        t0_idx   = int(np.searchsorted(times_ms, 0.0))

        for cond in self._conditions:
            buf      = self._epoch_buf[cond]
            n        = len(buf)
            cond_vis = self._cond_checks.get(cond, QCheckBox()).isChecked()

            if buf:
                stack = np.stack(buf, 0)
                avg   = np.mean(stack, 0)
                sem   = (np.std(stack, axis=0, ddof=1) / math.sqrt(n)
                         if n >= 2 else np.zeros_like(avg))
            else:
                avg = np.zeros((self._n_ch, self._n_t))
                sem = np.zeros((self._n_ch, self._n_t))

            for disp_pos, ch in enumerate(self._disp_channels):
                ch_i       = self._disp_idx[disp_pos]
                avg_scaled = avg[ch_i] * self._unit_scale
                sem_scaled = sem[ch_i] * self._unit_scale

                self._curves[ch][cond].setData(times_ms, avg_scaled)
                self._curves[ch][cond].setVisible(cond_vis)

                if n >= 2 and self._show_sem and cond_vis:
                    self._sem_upper[ch][cond].setData(times_ms, avg_scaled + sem_scaled)
                    self._sem_lower[ch][cond].setData(times_ms, avg_scaled - sem_scaled)
                    self._sem_fills[ch][cond].setVisible(True)
                else:
                    self._sem_fills[ch][cond].setVisible(False)

                if buf and self._show_peak and cond_vis:
                    post = avg_scaled[t0_idx:]
                    if post.size > 0:
                        pk_local = int(np.argmax(np.abs(post)))
                        self._peaks[ch][cond].setData(
                            [times_ms[t0_idx + pk_local]],
                            [avg_scaled[t0_idx + pk_local]],
                        )
                        self._peaks[ch][cond].setVisible(True)
                else:
                    self._peaks[ch][cond].setVisible(False)

            self._cond_n_lbl[cond].setText(f"n = {self._n_per[cond]}")

        self._total_lbl.setText(f"Total: {n_total} trials")

        if self._scale_sl.value() != 5:
            self._apply_y_range()
