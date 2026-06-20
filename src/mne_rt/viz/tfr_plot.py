"""Real-time time-frequency representation (TFR) display.

Computes Morlet wavelet power for selected channels and displays
as colour-coded heatmaps (time × frequency) after each new batch
of epochs arrives via :meth:`TFRPlot.update`.

Classes
-------
TFRPlot
    Real-time TFR display with interactive sidebar.
"""
from __future__ import annotations

import math
import threading
from typing import Optional, Union

import numpy as np

try:
    from qtpy.QtCore import Qt, QRectF, Signal
    from qtpy.QtGui import QFont, QColor, QTransform
    from qtpy.QtWidgets import (QApplication, QMainWindow, QWidget,
        QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QSlider, QPushButton,
        QFrame, QSizePolicy, QScrollArea, QFileDialog, QComboBox,
        QDoubleSpinBox)
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
    import mne.time_frequency
    _mne_available = True
except ImportError:
    _mne_available = False

from mne_rt._logging import logger, set_log_level


# ---------------------------------------------------------------------------
# Palette  (shared with ERPPlot)
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

_SIDEBAR_W = 210   # px

# Fallback thermal colormap stops when matplotlib is unavailable
_THERMAL_POS   = [0.0, 0.25, 0.5, 0.75, 1.0]
_THERMAL_COLOR = ["#000000", "#1a237e", "#e53935", "#ffeb3b", "#ffffff"]


# ---------------------------------------------------------------------------
# Sidebar helpers  (same API as erp_plot.py)
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
# Main class
# ---------------------------------------------------------------------------

class TFRPlot(QMainWindow):
    """Real-time time-frequency representation (TFR).

    Computes Morlet wavelet power for selected channels and displays
    as colour-coded heatmaps (time × frequency) after each new batch
    of epochs arrives via :meth:`update`.

    Two modes:

    - **induced**: average of per-epoch TFR → total power (including
      non-phase-locked oscillations).
    - **evoked**: TFR of the trial average → only phase-locked power.

    Baseline correction uses dB change: ``10 * log10(power / baseline_mean)``.

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
    freqs : ndarray or None
        Frequencies of interest (Hz).  Default ``np.arange(4, 50, 2)``.
    n_cycles : ndarray, float, or None
        Number of cycles per frequency for Morlet wavelets.  Default
        ``freqs / 2`` (half-cycle per frequency).
    channels : list of str or None
        Channels to display.  When ``None``, up to 4 channels are
        auto-selected (see :meth:`_auto_channels`).
    mode : {'induced', 'evoked'}
        Computation mode.  ``'induced'`` averages per-epoch TFRs;
        ``'evoked'`` computes the TFR of the trial average.
    baseline : tuple or None, default (None, 0)
        Baseline interval ``(tmin, tmax)`` in seconds used for dB
        normalisation.  ``None`` on either side means epoch edge.
    decim : int, default 4
        Decimation factor applied along the time axis before display.
    info : mne.Info or None
        Unused at present; reserved for future layout features.
    window_size : tuple of int, default (1440, 900)
        Initial window size in pixels.
    verbose : bool, str, or None

    .. versionadded:: 1.0.0

    See Also
    --------
    mne_rt.RTEpochs : Drives this plot via :meth:`update`.
    """

    # Emitted from the worker thread; Qt delivers it to _redraw on the main
    # thread, keeping all widget mutations on the GUI thread.
    _redraw_sig = Signal(int)

    def __init__(
        self,
        ch_names:    list[str],
        sfreq:       float,
        tmin:        float,
        tmax:        float,
        event_id:    dict[str, int],
        freqs:       Optional[np.ndarray] = None,
        n_cycles:    Union[np.ndarray, float, None] = None,
        channels:    Optional[list[str]] = None,
        mode:        str = "induced",
        baseline:    Optional[tuple] = (None, 0),
        decim:       int = 4,
        info=None,
        montage:     str = "standard_1020",
        window_size: tuple[int, int] = (1440, 900),
        verbose:     Union[bool, str, None] = None,
    ) -> None:
        if not _qt_available or not _pg_available:
            raise ImportError(
                "A Qt binding (PyQt6 or PySide6) and pyqtgraph are required for TFRPlot.\n"
                "Install with: pip install 'mne-rt[full]'"
            )
        _app = QApplication.instance() or QApplication([])  # noqa: F841

        super().__init__()
        self._redraw_sig.connect(self._redraw)
        set_log_level(verbose)

        # ── Public attributes ────────────────────────────────────────────
        self.ch_names   = list(ch_names)
        self.sfreq      = float(sfreq)
        self.tmin       = float(tmin)
        self.tmax       = float(tmax)
        self.event_id   = event_id
        self.mode       = mode.lower()
        self.baseline   = baseline
        self.decim      = max(1, int(decim))

        # ── Conditions ───────────────────────────────────────────────────
        self._conditions = list(event_id.keys())
        self._cmap = {
            c: _COND_COLORS[i % len(_COND_COLORS)]
            for i, c in enumerate(self._conditions)
        }

        # ── Time axis ────────────────────────────────────────────────────
        self._n_t   = int(round((tmax - tmin) * sfreq)) + 1
        self._times = np.linspace(tmin, tmax, self._n_t)
        # Decimated time axis — mirrors what tfr_array_morlet returns with
        # decim applied along the last axis.
        self._times_dec = self._times[::self.decim]
        self._n_t_dec   = len(self._times_dec)

        # ── Frequencies ──────────────────────────────────────────────────
        if freqs is None:
            freqs = np.arange(4, 50, 2, dtype=float)
        self._freqs = np.asarray(freqs, dtype=float)

        if n_cycles is None:
            self._n_cycles = self._freqs / 2.0
        else:
            self._n_cycles = n_cycles

        self._clip_freqs()

        self.montage = montage

        # ── Display channels ─────────────────────────────────────────────
        self._display_chs = self._auto_channels(channels)
        self._display_idx = [self.ch_names.index(c) for c in self._display_chs]

        # Scalp positions for topomap channel selector (normalised, yn=0=frontal)
        self._norm_pos    = self._compute_positions(info, montage)
        self._topo_scatter: Optional[pg.ScatterPlotItem] = None

        # ── Normalisation mode ───────────────────────────────────────────
        # 'db'  — dB change from baseline  (default)
        # 'raw' — raw power in µV²/Hz (no normalisation)
        self._norm_mode = "db"

        # ── Thread-safety state ──────────────────────────────────────────
        self._computing     = False
        self._latest_data:  Optional[np.ndarray] = None
        self._latest_conds: list[str]            = []
        self._tfr_result:   dict[str, np.ndarray] = {}

        # ── Colormap & color limits ──────────────────────────────────────
        self._cmap_name = "hot"
        self._cmap_lut  = self._build_colormap()
        self._vmin: Optional[float] = None   # None = auto
        self._vmax: Optional[float] = None

        # ── Widget dicts (populated in _build_ui) ────────────────────────
        self._image_items:  list[list[pg.ImageItem]] = []   # [cond_i][ch_i]
        self._plot_items:   list[list[pg.PlotItem]]  = []   # [cond_i][ch_i]
        self._ch_row_checks: dict[str, QCheckBox]    = {}
        self._cond_n_lbl:   dict[str, QLabel]        = {}

        # ── Build window ─────────────────────────────────────────────────
        self.setWindowTitle("MNE-RT — TFR Plot")
        self.resize(*window_size)
        self._apply_styles()
        self._build_ui()

        logger.info(
            "TFRPlot: %d display channels, %d conditions, "
            "freqs %.0f–%.0f Hz, mode=%s",
            len(self._display_chs), len(self._conditions),
            self._freqs[0], self._freqs[-1], self.mode,
        )

    # -----------------------------------------------------------------------
    # Frequency clipping
    # -----------------------------------------------------------------------

    def _clip_freqs(self) -> None:
        """Ensure no Morlet wavelet exceeds the epoch length.

        MNE uses ``n_sigmas = 5`` in :func:`mne.time_frequency.morlet`, so the
        wavelet fits when ``n_cycles < (n_t - 1) * π * f / (5 * sfreq)``.

        Strategy: first remove frequencies where even 2 cycles would not fit
        (too few for a meaningful TFR); then clip ``n_cycles`` on the remaining
        frequencies so the wavelet stays within the epoch.
        """
        nc = (np.asarray(self._n_cycles, float)
              if not np.isscalar(self._n_cycles)
              else np.full(len(self._freqs), float(self._n_cycles)))

        # Maximum n_cycles that fits: nc < (n_t-1)*pi*f / (5*sfreq)
        nc_max = (self._n_t - 1) * np.pi * self._freqs / (5.0 * self.sfreq)

        # Drop frequencies where even 2 cycles cannot fit
        mask = nc_max >= 2.0
        if not mask.all():
            logger.info(
                "TFRPlot: removed %d freq(s) below %.1f Hz "
                "(epoch too short for ≥2 cycles).",
                int((~mask).sum()),
                float(self._freqs[mask][0]) if mask.any() else 0.0,
            )
            self._freqs = self._freqs[mask]
            nc          = nc[mask]
            nc_max      = nc_max[mask]

        if len(self._freqs) == 0:
            # Epoch is very short — use the highest plausible frequency range
            f_min = max(10.0 * self.sfreq / ((self._n_t - 1) * np.pi), 8.0)
            self._freqs = np.arange(f_min, min(f_min + 30.0, 80.0), 2.0)
            nc_max      = (self._n_t - 1) * np.pi * self._freqs / (5.0 * self.sfreq)
            self._n_cycles = np.maximum(nc_max * 0.90, 1.0)
            logger.warning(
                "TFRPlot: epoch too short for standard TFR — "
                "using %.0f–%.0f Hz with reduced n_cycles.",
                float(self._freqs[0]), float(self._freqs[-1]),
            )
            return

        # Clip n_cycles to fit within epoch (10 % safety margin)
        self._n_cycles = np.minimum(nc, np.maximum(nc_max * 0.90, 1.0))

    # -----------------------------------------------------------------------
    # Scalp layout (for topomap channel selector)
    # -----------------------------------------------------------------------

    def _compute_positions(
        self, info, montage_name: str
    ) -> list[tuple[float, float]]:
        import math as _math
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
                logger.debug("TFRPlot: montage layout failed: %s", exc)
        n = len(self.ch_names)
        return [
            (0.5 + 0.42 * _math.cos(2 * _math.pi * i / n - _math.pi / 2),
             0.5 + 0.42 * _math.sin(2 * _math.pi * i / n - _math.pi / 2))
            for i in range(n)
        ]

    def _from_layout(self, layout) -> Optional[list[tuple[float, float]]]:
        if layout is None:
            return None
        name_xy: dict[str, tuple[float, float]] = {}
        for name, pos in zip(layout.names, layout.pos):
            xc = float(pos[0] + pos[2] / 2.0)
            yc = float(pos[1] + pos[3] / 2.0)
            name_xy[name] = (xc, 1.0 - yc)
        n_matched = sum(1 for c in self.ch_names if c in name_xy)
        if n_matched < len(self.ch_names) // 2:
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

    # -----------------------------------------------------------------------
    # Grid rebuild (called when channel selection changes)
    # -----------------------------------------------------------------------

    def _rebuild_grid(self) -> None:
        """Clear and recreate TFR grid for the current _display_chs."""
        self._gl_widget.clear()
        self._image_items = [[] for _ in range(len(self._conditions))]
        self._plot_items  = [[] for _ in range(len(self._conditions))]
        self._display_idx = [self.ch_names.index(c) for c in self._display_chs]
        self._build_tfr_grid()
        if self._latest_data is not None and not self._computing:
            threading.Thread(target=self._compute_and_emit, daemon=True).start()

    # -----------------------------------------------------------------------
    # Channel auto-selection
    # -----------------------------------------------------------------------

    def _auto_channels(self, channels: Optional[list[str]]) -> list[str]:
        """Return up to 4 channels to display.

        If *channels* is given, return those that exist in ``ch_names``
        (warn about missing ones).  Otherwise prefer the canonical set
        ``['Cz','Pz','Oz','Fz','C3','C4']``, padding with the first
        available channels when fewer than 4 are found.
        """
        if channels is not None:
            valid = [c for c in channels if c in self.ch_names]
            missing = [c for c in channels if c not in self.ch_names]
            if missing:
                logger.warning(
                    "TFRPlot: channels not found in ch_names and will be "
                    "ignored: %s", missing
                )
            return valid or self.ch_names[:4]

        preferred = ["Cz", "Pz", "Oz", "Fz", "C3", "C4"]
        found: list[str] = []
        # Case-sensitive first pass
        for c in preferred:
            if c in self.ch_names and c not in found:
                found.append(c)
        # Case-insensitive second pass
        ch_upper = [c.upper() for c in self.ch_names]
        for c in preferred:
            if c not in found:
                try:
                    idx = ch_upper.index(c.upper())
                    found.append(self.ch_names[idx])
                except ValueError:
                    pass

        # Pad with the first N channels if fewer than 4 found
        for c in self.ch_names:
            if len(found) >= 4:
                break
            if c not in found:
                found.append(c)

        return found[:4]

    # -----------------------------------------------------------------------
    # Colormap
    # -----------------------------------------------------------------------

    def _build_colormap(self) -> np.ndarray:
        """Return a 256×3 uint8 LUT for the current ``_cmap_name``."""
        name = getattr(self, "_cmap_name", "hot")
        try:
            cmap = pg.colormap.get(name, source="matplotlib")
            return cmap.getLookupTable(0.0, 1.0, 256)
        except Exception:
            pass
        # Inline fallbacks for the most important maps
        if name == "RdBu_r":
            stops  = [0.0, 0.25, 0.5, 0.75, 1.0]
            colors = ["#053061", "#4393c3", "#f7f7f7", "#d6604d", "#67001f"]
        elif name == "viridis":
            stops  = [0.0, 0.33, 0.67, 1.0]
            colors = ["#440154", "#31688e", "#35b779", "#fde725"]
        elif name == "plasma":
            stops  = [0.0, 0.33, 0.67, 1.0]
            colors = ["#0d0887", "#cc4778", "#f89441", "#f0f921"]
        else:
            stops  = _THERMAL_POS
            colors = _THERMAL_COLOR
        cmap = pg.ColorMap(stops, [QColor(c).getRgb()[:3] for c in colors])
        return cmap.getLookupTable(0.0, 1.0, 256)

    # -----------------------------------------------------------------------
    # Style sheet
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
            QComboBox             {{ background:{_SURFACE}; color:{_TEXT};
                                     border:1px solid {_BORDER};
                                     border-radius:4px; padding:2px 6px;
                                     font-size:11px; }}
            QComboBox::drop-down  {{ border:none; }}
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

        pg.setConfigOptions(antialias=False, background=_BG, foreground=_DIM)

        # ── Canvas ──────────────────────────────────────────────────────
        self._gl_widget = pg.GraphicsLayoutWidget()
        self._gl_widget.setBackground(_BG)
        self._gl_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._gl_widget, stretch=1)

        # ── Sidebar ─────────────────────────────────────────────────────
        root.addWidget(self._build_sidebar())

        # ── TFR grid ─────────────────────────────────────────────────────
        self._build_tfr_grid()

    def _build_tfr_grid(self) -> None:
        """Populate the GraphicsLayoutWidget with PlotItem / ImageItem cells."""
        layout = self._gl_widget.ci   # the central GraphicsLayout

        n_conds = len(self._conditions)
        n_chs   = len(self._display_chs)

        # Row 0: condition title labels
        for cond_i, cond in enumerate(self._conditions):
            col = _COND_COLORS[cond_i % len(_COND_COLORS)]
            title_lbl = layout.addLabel(
                cond, row=0, col=cond_i, color=col
            )
            title_lbl.setText(
                f'<span style="color:{col};font-size:10pt;font-weight:700;">'
                f'{cond}</span>'
            )

        # Rows 1…n_chs: one row per display channel, one column per condition
        self._image_items = [[] for _ in range(n_conds)]
        self._plot_items  = [[] for _ in range(n_conds)]

        is_last_row  = lambda ch_i: ch_i == n_chs - 1  # noqa: E731
        is_first_col = lambda cond_i: cond_i == 0       # noqa: E731

        for ch_i, ch_name in enumerate(self._display_chs):
            for cond_i in range(n_conds):
                show_x = is_last_row(ch_i)
                show_y = is_first_col(cond_i)

                ch_title = ch_name   # show channel label in every column

                plot = layout.addPlot(row=ch_i + 1, col=cond_i)
                self._style_plot(plot, show_x=show_x, show_y=show_y,
                                 title=ch_title)

                img = pg.ImageItem()
                img.setLookupTable(self._cmap_lut)
                plot.addItem(img)

                # Set axis transform: x = time in ms, y = frequency in Hz
                self._set_image_transform(img)

                self._image_items[cond_i].append(img)
                self._plot_items[cond_i].append(plot)

        # ── Align all axes explicitly ─────────────────────────────────────
        f_min = float(self._freqs[0])  if len(self._freqs) else 0.0
        f_max = float(self._freqs[-1]) if len(self._freqs) else 100.0
        t0_ms = float(self._times_dec[0]  * 1000.0)
        t1_ms = float(self._times_dec[-1] * 1000.0)
        for ch_i in range(n_chs):
            for cond_i in range(n_conds):
                self._plot_items[cond_i][ch_i].setXRange(t0_ms, t1_ms, padding=0)
                self._plot_items[cond_i][ch_i].setYRange(f_min, f_max, padding=0)

    def _set_image_transform(self, img: "pg.ImageItem") -> None:
        """Apply QTransform so image axes show ms / Hz values."""
        t0_ms  = self._times_dec[0]  * 1000.0
        t1_ms  = self._times_dec[-1] * 1000.0
        f0_hz  = float(self._freqs[0])
        f1_hz  = float(self._freqs[-1])
        n_t    = max(1, len(self._times_dec))
        n_f    = max(1, len(self._freqs))

        tr = QTransform()
        tr.translate(t0_ms, f0_hz)
        tr.scale((t1_ms - t0_ms) / n_t, (f1_hz - f0_hz) / n_f)
        img.setTransform(tr)

    def _style_plot(
        self,
        plot: "pg.PlotItem",
        show_x: bool = False,
        show_y: bool = False,
        title: str = "",
    ) -> None:
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setMouseEnabled(x=False, y=False)
        plot.getViewBox().disableAutoRange()   # prevent auto-range from breaking alignment
        plot.showAxis("top",    False)
        plot.showAxis("right",  False)
        plot.showAxis("bottom", show_x)
        plot.showAxis("left",   show_y)
        if show_x:
            plot.getAxis("bottom").setLabel("Time", units="ms")
            plot.getAxis("bottom").setStyle(tickFont=QFont("Helvetica", 7))
        if show_y:
            plot.getAxis("left").setLabel("Freq", units="Hz")
            plot.getAxis("left").setStyle(tickFont=QFont("Helvetica", 7))
        if title:
            plot.setTitle(title, color=_DIM, size="9pt")
        plot.setContentsMargins(0, 0, 0, 0)

    # -----------------------------------------------------------------------
    # Sidebar
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
        hdr = QLabel("TFR CONTROLS")
        hdr.setStyleSheet(
            f"color:{_TEXT}; font-size:11px; font-weight:700; letter-spacing:1.5px;"
        )
        ly.addWidget(hdr)
        ly.addWidget(_sep(sb))

        # ── CHANNELS (topomap selector) ───────────────────────────────────
        ly.addWidget(_section("CHANNELS", sb))

        self._sel_lbl = QLabel(", ".join(self._display_chs))
        self._sel_lbl.setStyleSheet(
            f"color:{_ACCENT}; font-size:10px; font-weight:600;"
        )
        self._sel_lbl.setWordWrap(True)
        ly.addWidget(self._sel_lbl)

        cap_lbl = QLabel("click to toggle  (≤8 for performance)")
        cap_lbl.setStyleSheet(f"color:{_DIM}; font-size:9px;")
        ly.addWidget(cap_lbl)

        ly.addSpacing(4)
        topo_w = self._build_topo_widget(sb)
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
        self._cond_n_lbl = {}
        for cond in self._conditions:
            col = self._cmap[cond]
            row_w, row_l = _row(sb)
            dot = QLabel(f"● {cond}")
            dot.setStyleSheet(f"color:{col}; font-size:11px; font-weight:600;")
            dot.setWordWrap(True)
            n_lbl = _val_lbl("n = 0", sb, color=_DIM)
            self._cond_n_lbl[cond] = n_lbl
            row_l.addWidget(dot, stretch=1)
            row_l.addWidget(n_lbl)
            ly.addWidget(row_w)

        ly.addWidget(_sep(sb))

        # ── MODE ─────────────────────────────────────────────────────────
        ly.addWidget(_section("MODE", sb))
        self._mode_combo = QComboBox(sb)
        self._mode_combo.addItem("Induced (total power)")
        self._mode_combo.addItem("Evoked (phase-locked)")
        self._mode_combo.setCurrentIndex(0 if self.mode == "induced" else 1)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_change)
        ly.addWidget(self._mode_combo)

        ly.addWidget(_sep(sb))

        # ── FREQ RANGE ───────────────────────────────────────────────────
        ly.addWidget(_section("FREQ RANGE", sb))

        r_fstart, l_fstart = _row(sb)
        l_fstart.addWidget(_key_lbl("Start", sb), stretch=1)
        self._fstart_lbl = _val_lbl(f"{int(self._freqs[0])} Hz", sb)
        l_fstart.addWidget(self._fstart_lbl)
        ly.addWidget(r_fstart)

        self._fstart_sl = _slider(sb, 2, 50, int(self._freqs[0]))
        self._fstart_sl.valueChanged.connect(self._on_freq_range)
        ly.addWidget(self._fstart_sl)

        r_fend, l_fend = _row(sb)
        l_fend.addWidget(_key_lbl("End", sb), stretch=1)
        self._fend_lbl = _val_lbl(f"{int(self._freqs[-1])} Hz", sb)
        l_fend.addWidget(self._fend_lbl)
        ly.addWidget(r_fend)

        self._fend_sl = _slider(sb, 4, 100, int(self._freqs[-1]))
        self._fend_sl.valueChanged.connect(self._on_freq_range)
        ly.addWidget(self._fend_sl)

        ly.addWidget(_sep(sb))

        # ── NORMALISATION ────────────────────────────────────────────────
        ly.addWidget(_section("NORMALIZATION", sb))
        self._norm_combo = QComboBox(sb)
        self._norm_combo.addItem("dB change (baseline)")
        self._norm_combo.addItem("Raw power (µV²/Hz)")
        self._norm_combo.setCurrentIndex(0)
        self._norm_combo.currentIndexChanged.connect(self._on_norm_change)
        ly.addWidget(self._norm_combo)

        ly.addWidget(_sep(sb))

        # ── COLORMAP ─────────────────────────────────────────────────────
        ly.addWidget(_section("COLORMAP", sb))
        self._cmap_combo = QComboBox(sb)
        for label in ["Hot", "RdBu (div.)", "Viridis", "Plasma", "Turbo", "Greys"]:
            self._cmap_combo.addItem(label)
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_change)
        ly.addWidget(self._cmap_combo)

        ly.addWidget(_sep(sb))

        # ── COLOR LIMITS ─────────────────────────────────────────────────
        ly.addWidget(_section("COLOR LIMITS", sb))

        self._auto_lvl_chk = QCheckBox("Auto")
        self._auto_lvl_chk.setChecked(True)
        self._auto_lvl_chk.toggled.connect(self._on_auto_levels)
        ly.addWidget(self._auto_lvl_chk)

        r_vmin, l_vmin = _row(sb)
        l_vmin.addWidget(_key_lbl("vmin", sb), stretch=1)
        self._vmin_spin = QDoubleSpinBox(sb)
        self._vmin_spin.setRange(-200.0, 0.0)
        self._vmin_spin.setValue(-3.0)
        self._vmin_spin.setSingleStep(0.5)
        self._vmin_spin.setDecimals(1)
        self._vmin_spin.setEnabled(False)
        self._vmin_spin.setStyleSheet(
            f"QDoubleSpinBox{{background:{_SURFACE};color:{_TEXT};"
            f"border:1px solid {_BORDER};border-radius:3px;padding:1px 4px;}}"
        )
        self._vmin_spin.valueChanged.connect(self._on_vmin_change)
        l_vmin.addWidget(self._vmin_spin)
        ly.addWidget(r_vmin)

        r_vmax, l_vmax = _row(sb)
        l_vmax.addWidget(_key_lbl("vmax", sb), stretch=1)
        self._vmax_spin = QDoubleSpinBox(sb)
        self._vmax_spin.setRange(0.0, 200.0)
        self._vmax_spin.setValue(3.0)
        self._vmax_spin.setSingleStep(0.5)
        self._vmax_spin.setDecimals(1)
        self._vmax_spin.setEnabled(False)
        self._vmax_spin.setStyleSheet(
            f"QDoubleSpinBox{{background:{_SURFACE};color:{_TEXT};"
            f"border:1px solid {_BORDER};border-radius:3px;padding:1px 4px;}}"
        )
        self._vmax_spin.valueChanged.connect(self._on_vmax_change)
        l_vmax.addWidget(self._vmax_spin)
        ly.addWidget(r_vmax)

        ly.addWidget(_sep(sb))

        # ── DATA ─────────────────────────────────────────────────────────
        ly.addWidget(_section("DATA", sb))

        self._total_lbl = QLabel("Total: 0 trials")
        self._total_lbl.setStyleSheet(f"color:{_TEXT}; font-size:11px;")
        ly.addWidget(self._total_lbl)

        ly.addSpacing(4)

        export_btn = QPushButton("Export PNG …")
        export_btn.setToolTip("Save the current TFR plot as a PNG image")
        export_btn.clicked.connect(self._export_png)
        ly.addWidget(export_btn)

        ly.addStretch()
        scroll.setWidget(sb)
        return scroll

    # -----------------------------------------------------------------------
    # Sidebar callbacks
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Topomap channel selector
    # -----------------------------------------------------------------------

    def _build_topo_widget(self, parent: QWidget) -> pg.PlotWidget:
        pw = pg.PlotWidget(parent=parent)
        pw.setFixedSize(184, 184)
        pw.setBackground(_SURFACE)
        pw.hideAxis("bottom")
        pw.hideAxis("left")
        pw.getViewBox().setMouseEnabled(x=False, y=False)
        pw.getViewBox().setAspectLocked(True)
        pw.getViewBox().setRange(
            xRange=(-0.06, 1.06), yRange=(-0.06, 1.14), padding=0,
        )

        theta = np.linspace(0, 2 * np.pi, 160)
        pw.plot(0.5 + 0.48 * np.cos(theta), 0.5 + 0.48 * np.sin(theta),
                pen=pg.mkPen(_BORDER, width=1.5))
        pw.plot([0.47, 0.5, 0.53, 0.47], [0.97, 1.06, 0.97, 0.97],
                pen=pg.mkPen(_BORDER, width=1.2))

        spots = []
        for i, ch in enumerate(self.ch_names):
            xn, yn = self._norm_pos[i]
            tx = 0.5 + (xn - 0.5) * 0.9
            ty = 1.0 - (0.5 + (yn - 0.5) * 0.9)
            selected = ch in self._display_chs
            spots.append({
                "pos": (tx, ty), "data": ch,
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

        hint = pg.TextItem("click to select", color=_DIM, anchor=(0.5, 1))
        hint.setFont(QFont("Helvetica", 7))
        hint.setPos(0.5, -0.02)
        pw.addItem(hint)
        return pw

    def _update_topo_colors(self) -> None:
        if self._topo_scatter is None:
            return
        spots = []
        for i, ch in enumerate(self.ch_names):
            xn, yn = self._norm_pos[i]
            tx = 0.5 + (xn - 0.5) * 0.9
            ty = 1.0 - (0.5 + (yn - 0.5) * 0.9)
            selected = ch in self._display_chs
            spots.append({
                "pos": (tx, ty), "data": ch,
                "brush": pg.mkBrush(_ACCENT if selected else _BORDER),
                "pen": pg.mkPen(None),
                "size": 10 if selected else 6,
            })
        self._topo_scatter.setData(spots=spots)

    def _on_topo_click(self, *args) -> None:
        points = args[-2] if len(args) >= 2 else args[0]
        for pt in points:
            ch = pt.data()
            if ch is None or ch not in self.ch_names:
                continue
            if ch in self._display_chs:
                if len(self._display_chs) > 1:
                    self._display_chs.remove(ch)
            else:
                if len(self._display_chs) < 8:
                    self._display_chs.append(ch)
        self._update_topo_colors()
        self._sel_lbl.setText(", ".join(self._display_chs))
        self._rebuild_grid()

    def _on_mode_change(self, idx: int) -> None:
        self.mode = "induced" if idx == 0 else "evoked"
        self._maybe_recompute()

    def _on_freq_range(self) -> None:
        fmin = self._fstart_sl.value()
        fmax = self._fend_sl.value()
        if fmin >= fmax:
            return
        self._fstart_lbl.setText(f"{fmin} Hz")
        self._fend_lbl.setText(f"{fmax} Hz")
        self._freqs    = np.arange(fmin, fmax, 2, dtype=float)
        if len(self._freqs) == 0:
            self._freqs = np.array([float(fmin)])
        self._n_cycles = self._freqs / 2.0
        self._clip_freqs()
        # Refresh transforms and realign all axes
        f_min = float(self._freqs[0])  if len(self._freqs) else 0.0
        f_max = float(self._freqs[-1]) if len(self._freqs) else 100.0
        for cond_i in range(len(self._conditions)):
            for ch_i in range(len(self._display_chs)):
                self._set_image_transform(self._image_items[cond_i][ch_i])
                self._plot_items[cond_i][ch_i].setXRange(
                    float(self._times_dec[0]*1000), float(self._times_dec[-1]*1000), padding=0
                )
                self._plot_items[cond_i][ch_i].setYRange(f_min, f_max, padding=0)
        self._maybe_recompute()

    def _on_norm_change(self, idx: int) -> None:
        self._norm_mode = "db" if idx == 0 else "raw"
        self._maybe_recompute()

    def _on_cmap_change(self, idx: int) -> None:
        _map = ["hot", "RdBu_r", "viridis", "plasma", "turbo", "greys"]
        self._cmap_name = _map[idx % len(_map)]
        self._cmap_lut  = self._build_colormap()
        for ci in range(len(self._conditions)):
            for chi in range(len(self._display_chs)):
                self._image_items[ci][chi].setLookupTable(self._cmap_lut)

    def _on_auto_levels(self, checked: bool) -> None:
        self._vmin_spin.setEnabled(not checked)
        self._vmax_spin.setEnabled(not checked)
        if checked:
            self._vmin = None
            self._vmax = None
        else:
            self._vmin = self._vmin_spin.value()
            self._vmax = self._vmax_spin.value()
        self._maybe_recompute()

    def _on_vmin_change(self, val: float) -> None:
        if not self._auto_lvl_chk.isChecked():
            self._vmin = val
            self._maybe_recompute()

    def _on_vmax_change(self, val: float) -> None:
        if not self._auto_lvl_chk.isChecked():
            self._vmax = val
            self._maybe_recompute()

    def _maybe_recompute(self) -> None:
        """Trigger recompute if we already have data."""
        if self._latest_data is not None and not self._computing:
            threading.Thread(
                target=self._compute_and_emit, daemon=True
            ).start()

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export TFR Plot", "tfr_plot.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)",
        )
        if path:
            self.grab().save(path)

    # -----------------------------------------------------------------------
    # TFR computation
    # -----------------------------------------------------------------------

    def _compute_tfr(
        self, data: np.ndarray, conditions: list[str]
    ) -> "dict[str, np.ndarray]":
        """Compute Morlet TFR for each condition.

        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_channels, n_times)
            All accumulated epochs.
        conditions : list of str
            Condition label for each epoch.

        Returns
        -------
        dict
            Mapping condition → power array of shape
            ``(n_display_ch, n_freqs, n_times_dec)``.
        """
        result: dict[str, np.ndarray] = {}

        for cond in self._conditions:
            mask = np.array([c == cond for c in conditions])
            if not mask.any():
                result[cond] = np.zeros(
                    (len(self._display_idx), len(self._freqs), self._n_t_dec)
                )
                continue

            ep = data[mask]            # (n_ep, n_ch, n_t)

            if self.mode == "evoked":
                ep = ep.mean(0, keepdims=True)  # TFR of trial average

            power = mne.time_frequency.tfr_array_morlet(
                ep.astype(np.float64),
                sfreq=self.sfreq,
                freqs=self._freqs,
                n_cycles=self._n_cycles,
                output="power",
                decim=self.decim,
                zero_mean=True,
            )  # (n_ep, n_ch, n_freqs, n_times_dec)

            avg_power = power.mean(0)   # (n_ch, n_freqs, n_times_dec)

            if self._norm_mode == "db":
                # Baseline correction: dB change from mean baseline power
                bl_mask = self._times_dec <= 0
                if bl_mask.any():
                    bl_power = (
                        avg_power[:, :, bl_mask].mean(-1, keepdims=True) + 1e-30
                    )
                    avg_power = 10.0 * np.log10(avg_power / bl_power)
                # If no pre-stimulus baseline, still convert to log scale
                else:
                    avg_power = 10.0 * np.log10(avg_power + 1e-30)

            result[cond] = avg_power[self._display_idx]  # select display chs

        return result

    # -----------------------------------------------------------------------
    # Threading
    # -----------------------------------------------------------------------

    def update(self, data: np.ndarray, conditions: list[str]) -> None:
        """Receive new epoch data and schedule a TFR recompute.

        Thread-safe.  If a computation is already running the new data is
        stored and will be used at the next opportunity; the current run is
        not interrupted.

        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_channels, n_times)
            All accepted epochs accumulated so far.
        conditions : list of str
            Condition label for each epoch; ``len(conditions) == data.shape[0]``.
        """
        self._latest_data  = data.copy()
        self._latest_conds = list(conditions)
        if not self._computing:
            threading.Thread(
                target=self._compute_and_emit, daemon=True
            ).start()

    def _compute_and_emit(self) -> None:
        self._computing = True
        try:
            tfr = self._compute_tfr(self._latest_data, self._latest_conds)
            self._tfr_result = tfr
            self._redraw_sig.emit(len(self._latest_conds))
        except Exception as exc:
            logger.warning("TFRPlot compute error: %s", exc)
        finally:
            self._computing = False

    # -----------------------------------------------------------------------
    # Redraw  (main thread)
    # -----------------------------------------------------------------------

    def _redraw(self, n_total: int) -> None:
        """Update all image items from the latest TFR result.

        Always called on the main (GUI) thread via the Qt signal mechanism.
        Uses a **shared** colour scale across all conditions so the two
        heatmaps are directly comparable.
        """
        self._total_lbl.setText(f"Total: {n_total} trials")

        for cond_i, cond in enumerate(self._conditions):
            n = sum(1 for c in self._latest_conds if c == cond)
            self._cond_n_lbl[cond].setText(f"n = {n}")

        # ── Shared colour limits across ALL conditions and channels ───────
        if self._vmin is not None and self._vmax is not None:
            g_vmin, g_vmax = float(self._vmin), float(self._vmax)
        else:
            samples = []
            for cond in self._conditions:
                pwr = self._tfr_result.get(cond)
                if pwr is not None:
                    for ch_i in range(len(self._display_chs)):
                        flat = pwr[ch_i].ravel()
                        if flat.any():
                            samples.append(flat)
            if samples:
                all_vals = np.concatenate(samples)
                g_vmin = float(np.percentile(all_vals, 5))
                g_vmax = float(np.percentile(all_vals, 95))
                if g_vmin == g_vmax:
                    g_vmin, g_vmax = g_vmin - 1.0, g_vmax + 1.0
            else:
                g_vmin, g_vmax = -3.0, 3.0

        # ── Update images ─────────────────────────────────────────────────
        for cond_i, cond in enumerate(self._conditions):
            pwr = self._tfr_result.get(cond)
            if pwr is None:
                continue
            for ch_i in range(len(self._display_chs)):
                # ImageItem expects (n_times_dec, n_freqs)
                self._image_items[cond_i][ch_i].setImage(
                    pwr[ch_i].T, levels=(g_vmin, g_vmax)
                )

        # ── Re-lock axis ranges on every cell ─────────────────────────────
        if self._plot_items and len(self._freqs):
            f_min = float(self._freqs[0])
            f_max = float(self._freqs[-1])
            t0_ms = float(self._times_dec[0]  * 1000.0)
            t1_ms = float(self._times_dec[-1] * 1000.0)
            for ci in range(len(self._conditions)):
                for chi in range(len(self._display_chs)):
                    if ci < len(self._plot_items) and chi < len(self._plot_items[ci]):
                        self._plot_items[ci][chi].setXRange(t0_ms, t1_ms, padding=0)
                        self._plot_items[ci][chi].setYRange(f_min, f_max, padding=0)
