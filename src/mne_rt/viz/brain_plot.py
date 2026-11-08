"""Real-time 3D brain activation display.

Built on PyVista / pyvistaqt with a Qt control panel, hemisphere toggles,
surface switching, view presets, and screenshot support.

Classes
-------
BrainPlot
    Interactive real-time 3D brain surface with NF activity overlay.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import pyvista as pv

    _pyvista_available = True
except ImportError:
    pv = None  # type: ignore[assignment]
    _pyvista_available = False

try:
    from pyvistaqt import BackgroundPlotter as _BackgroundPlotter

    _pyvistaqt_available = True
except ImportError:
    _BackgroundPlotter = None  # type: ignore[assignment]
    _pyvistaqt_available = False

from mne_rt._logging import logger
from mne_rt.tools import setup_surface
from mne_rt.viz._smoothing import check_smoothing, ema_step

# Constants

_CMAPS = ["hot", "plasma", "viridis", "Reds", "YlOrRd", "RdBu_r"]

_SURFACES = ["inflated", "pial", "white", "sphere"]

# Display modes: label shown in the scalar bar title
_DISPLAY_MODES = [
    "Source Activation",
    "Alpha Power  (8–13 Hz)",
    "Beta Power   (13–30 Hz)",
    "Theta Power  (4–8 Hz)",
    "Gamma Power  (30–80 Hz)",
    "SMR Power    (12–15 Hz)",
]

# The band each display mode shows; None is the broadband activation.
# :attr:`BrainPlot.band` hands it to the computation in record_main.
_MODE_BANDS: dict[str, Optional[tuple[float, float]]] = {
    "Source Activation": None,
    "Alpha Power  (8–13 Hz)": (8.0, 13.0),
    "Beta Power   (13–30 Hz)": (13.0, 30.0),
    "Theta Power  (4–8 Hz)": (4.0, 8.0),
    "Gamma Power  (30–80 Hz)": (30.0, 80.0),
    "SMR Power    (12–15 Hz)": (12.0, 15.0),
}

# Sulcal depth is drawn in grey under the activity, light on gyri.
_SULC_CLIM = (-1.0, 1.5)
_SULC_GREYS = (0.30, 0.82)  # grey levels of sulci and gyri

_VIEW_PRESETS: dict[str, dict] = {
    "lateral_lh": {"position": (-450, 0, 0), "focal": (0, 0, 0), "up": (0, 0, 1)},
    "lateral_rh": {"position": (450, 0, 0), "focal": (0, 0, 0), "up": (0, 0, 1)},
    "dorsal": {"position": (0, 0, 450), "focal": (0, 0, 0), "up": (0, 1, 0)},
    "frontal": {"position": (0, -450, 0), "focal": (0, 0, 0), "up": (0, 0, 1)},
    "ventral": {"position": (0, 0, -450), "focal": (0, 0, 0), "up": (0, 1, 0)},
}

# Dark stylesheet for the Qt control panel — matches the dark UI of other plots
_PANEL_DARK_STYLE = """
    QWidget {
        background: #161b22;
        color: #e6edf3;
        font-family: -apple-system, 'Segoe UI', sans-serif;
        font-size: 11px;
    }
    QGroupBox {
        border: 1px solid #30363d;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 8px;
        color: #8b949e;
        font-weight: 600;
        font-size: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: #8b949e;
    }
    QLabel { color: #c9d1d9; background: transparent; }
    QComboBox {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 5px;
        padding: 4px 8px;
        color: #e6edf3;
    }
    QComboBox::drop-down { border: none; width: 18px; }
    QComboBox QAbstractItemView {
        background: #161b22;
        border: 1px solid #30363d;
        color: #e6edf3;
        selection-background-color: #1f6feb;
    }
    QSlider::groove:horizontal {
        height: 3px;
        background: #30363d;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #58a6ff;
        width: 12px;
        height: 12px;
        margin: -5px 0;
        border-radius: 6px;
    }
    QSlider::sub-page:horizontal {
        background: #388bfd;
        border-radius: 2px;
    }
    QPushButton {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 5px;
        padding: 5px 8px;
        color: #c9d1d9;
        font-weight: 500;
    }
    QPushButton:hover { background: #30363d; border-color: #8b949e; color: #e6edf3; }
    QPushButton:pressed { background: #1f6feb; border-color: #388bfd; color: #ffffff; }
    QCheckBox { spacing: 6px; color: #c9d1d9; }
    QCheckBox::indicator {
        width: 13px; height: 13px;
        border-radius: 3px;
        border: 1px solid #30363d;
        background: #21262d;
    }
    QCheckBox::indicator:checked { background: #238636; border-color: #2ea043; }
    QDoubleSpinBox {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 5px;
        padding: 3px 6px;
        color: #e6edf3;
    }
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        background: #30363d; border: none; width: 14px;
    }
    QDockWidget { color: #e6edf3; background: #0d1117; }
    QDockWidget::title {
        background: #161b22;
        border-bottom: 1px solid #30363d;
        padding: 6px 8px;
        color: #e6edf3;
        font-weight: 700;
    }
    QScrollBar:vertical { background: #0d1117; width: 8px; border-radius: 4px; }
    QScrollBar::handle:vertical {
        background: #30363d; border-radius: 4px; min-height: 20px;
    }
"""

# Background colour presets  (bottom_hex, top_hex)
_BACKGROUNDS: dict[str, tuple[str, str]] = {
    "Deep space": ("#040810", "#0b1628"),
    "Midnight": ("#050d1a", "#0d1b35"),
    "Slate": ("#0d1117", "#1c2333"),
    "Charcoal": ("#111111", "#1e1e1e"),
    "Black": ("#000000", "#050505"),
    "Light": ("#e8eaf0", "#ffffff"),  # publication / presentations
}

_DEFAULT_BG = "Midnight"


#: Vertices per hemisphere of the ico-5 cortex the display draws.
N_ICO5 = 10242


def to_display(values: np.ndarray, vertices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """One value per source, as ``(lh, rh)`` arrays over the display's vertices.

    ``vertices`` are the source estimate's vertex numbers per hemisphere; a
    display vertex without a source is NaN, drawn without activity.
    """
    n_lh = len(vertices[0])
    out = []
    for vertno, part in zip(vertices, (values[:n_lh], values[n_lh:])):
        full = np.full(N_ICO5, np.nan)
        full[vertno] = part
        out.append(full)
    return out[0], out[1]


def _lut(cmap: str) -> np.ndarray:
    """256 RGB rows of a matplotlib colormap, in [0, 1]."""
    import matplotlib

    return matplotlib.colormaps[cmap](np.linspace(0.0, 1.0, 256))[:, :3]


class BrainPlot:
    """Interactive real-time 3D brain activation display.

    Renders bilateral ``fsaverage`` cortical surfaces with a colour-mapped
    activity overlay and a Qt control panel docked on the right.  Designed
    to run alongside :class:`~mne_rt.viz.NFPlot` inside a shared Qt
    event loop — call :meth:`update_from_arrays` or :meth:`update` from the
    acquisition thread's pump timer.

    Parameters
    ----------
    subjects_fs_dir : str | Path
        FreeSurfer subjects directory.  The MNE-bundled ``fsaverage``
        template is used automatically (auto-downloaded on first use);
        this path is accepted for API compatibility.
    clim : tuple of float, default (0.0, 0.6)
        Initial ``(min, max)`` colour-map range for the activity overlay.
    hemi_distance : float, default 20.0
        Gap in mm between the medial walls of the two hemispheres.
    surf : {"inflated", "pial", "white", "sphere"}, default "inflated"
        Initial cortical surface geometry to display.
    cmap : str, default "hot"
        Initial colour map name.  Must be one of
        ``["hot", "plasma", "viridis", "Reds", "YlOrRd", "RdBu_r"]``.
    opacity : float, default 0.6
        Initial opacity of the activity overlay (0 = transparent, 1 = opaque).
    window_size : tuple of int, default (1600, 1000)
        Width × height of the render window in pixels.
    display_smoothing : float, default 0.3
        EMA factor applied to the per-vertex activation arrays before each
        render, in ``(0, 1]``.  ``1.0`` disables smoothing; lower values blend
        consecutive frames so the cortical map transitions smoothly instead of
        jumping between analysis windows.
    verbose : bool | str | None, default None
        Verbosity level.

    See Also
    --------
    mne_rt.viz.NFPlot : Scrolling real-time NF signal plot.
    mne_rt.RTStream.record_main : Main NF loop that drives both plots.

    Notes
    -----
    The cortex is drawn at the resolution of an ico-5 source space: the 10 242
    vertices per hemisphere the activity is estimated on, which are the first
    10 242 of fsaverage's. Values are interpolated across each triangle. The
    full 163 842-vertex surface made every update upload sixteen times more
    data to the GPU for no extra information, about 0.2 s per frame.

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        subjects_fs_dir: Union[str, Path],
        clim: tuple[float, float] = (0.0, 0.6),
        hemi_distance: float = 20.0,
        surf: str = "inflated",
        cmap: str = "hot",
        opacity: float = 0.6,
        window_size: tuple[int, int] = (1600, 1000),
        display_smoothing: float = 0.3,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        if not _pyvista_available or not _pyvistaqt_available:
            raise ImportError(
                "pyvista and pyvistaqt are required for BrainPlot. "
                "Install them with:  pip install 'mne-rt[viz]'"
            )
        from mne_rt._logging import set_log_level

        set_log_level(verbose)

        if cmap not in _CMAPS:
            raise ValueError(f"cmap {cmap!r} not recognised.  Choose from {_CMAPS}.")
        if surf not in _SURFACES:
            raise ValueError(f"surf {surf!r} not recognised.  Choose from {_SURFACES}.")
        display_smoothing = check_smoothing(display_smoothing)

        self._subjects_fs_dir = Path(subjects_fs_dir)
        self._clim = list(clim)
        self._hemi_distance = hemi_distance
        self._surf = surf
        self._opacity = opacity
        self._threshold = 0.0
        self._cmap_idx = _CMAPS.index(cmap)
        self._hemi_visible = {"lh": True, "rh": True}
        self._bg_name = _DEFAULT_BG
        self._display_mode = _DISPLAY_MODES[0]
        self._recording = False
        self._video_writer = None
        self._video_path: Path | None = None
        self._display_alpha = float(display_smoothing)
        self._lh_ema: np.ndarray | None = None
        self._rh_ema: np.ndarray | None = None

        # Custom band overlay state
        self._custom_band_hz: tuple[float, float] | None = None

        # Parcellation border overlay state
        self._parc_name: str | None = None
        self._parc_actor = None
        self._parc_edges: np.ndarray | None = None  # (n_edges, 2) vertex index pairs

        logger.info("Loading %s surface …", surf)
        self._load_surface(surf)

        self._plotter = self._build_plotter(window_size)
        self._add_key_bindings()
        self._add_overlays()
        logger.info("BrainPlot ready.")

    # Surface management

    def _load_surface(self, surf: str) -> None:
        (
            self._hemi_offsets,
            self._scalars_full,
            self._mesh,
            self._verts_stc,
            self._nn_map,
        ) = setup_surface(
            str(self._subjects_fs_dir),
            hemi_distance=self._hemi_distance,
            surf=surf,
            grade=5,
        )
        self._n_lh = int(self._hemi_offsets["rh"])
        t = np.clip((self._mesh["base"] - _SULC_CLIM[0]) / np.ptp(_SULC_CLIM), 0.0, 1.0)
        grey = _SULC_GREYS[0] + (_SULC_GREYS[1] - _SULC_GREYS[0]) * (1.0 - t)
        self._base_rgb = np.repeat(grey[:, np.newaxis], 3, axis=1)
        self._mesh["rgb"] = (self._base_rgb * 255).astype(np.uint8)

    # Plotter construction

    def _build_plotter(self, window_size: tuple[int, int]) -> "_BackgroundPlotter":
        p = _BackgroundPlotter(
            window_size=tuple(window_size),
            lighting="three lights",
            title="MNE-RT — Brain Activation",
            toolbar=False,
            menu_bar=False,
            editor=False,
            # Rendered when there is something new (record_main's brain timer),
            # not also on pyvistaqt's own timer, five times a second regardless.
            auto_update=False,
        )

        bg_bot, bg_top = _BACKGROUNDS[_DEFAULT_BG]
        p.set_background(bg_bot, top=bg_top)

        self._add_cortex_actor(p)
        self._add_scalar_bar(p)

        # FXAA only: SSAO (a 256-sample kernel), SSAA (the frame rendered several
        # times over) and eye-dome lighting (made for point clouds) cost a weak
        # GPU far more than they add, and made offscreen VTK report errors.
        p.enable_anti_aliasing("fxaa")
        p.add_camera_orientation_widget()
        p.camera_position = "yz"
        p.camera.azimuth = 45

        self._add_control_panel(p)
        return p

    def _add_cortex_actor(self, p: "_BackgroundPlotter") -> None:
        """One actor, coloured on the CPU: the sulcal grey blended with activity.

        Two overlapping actors, one translucent, doubled the geometry, needed
        transparency sorting and flickered where the two coincided.
        """
        self._actor = p.add_mesh(
            self._mesh, scalars="rgb", rgb=True, smooth_shading=True, show_scalar_bar=False
        )
        # The array the actor draws from, written in place on each update.
        self._rgb = self._actor.mapper.dataset.point_data["rgb"]
        self._refresh_scalars(deferred=True)

    def _add_scalar_bar(self, p: "_BackgroundPlotter") -> None:
        from vtkmodules.vtkCommonCore import vtkLookupTable
        from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor

        self._bar_lut = vtkLookupTable()
        self._bar = vtkScalarBarActor()
        self._bar.SetLookupTable(self._bar_lut)
        self._bar.SetPosition(0.02, 0.20)
        self._bar.SetWidth(0.04)
        self._bar.SetHeight(0.38)
        self._bar.SetNumberOfLabels(0)
        p.add_actor(self._bar)
        self._update_scalar_bar()

    def _update_scalar_bar(self) -> None:
        lut = _lut(_CMAPS[self._cmap_idx])
        self._bar_lut.SetNumberOfTableValues(len(lut))
        for i, (r, g, b) in enumerate(lut):
            self._bar_lut.SetTableValue(i, r, g, b, 1.0)
        self._bar_lut.SetRange(*self._clim)
        self._bar_lut.Build()

    # Qt control panel

    def _add_control_panel(self, p: "_BackgroundPlotter") -> None:
        """Dock a Qt control panel on the right side of the brain window."""
        from qtpy.QtCore import Qt as Qt_
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDockWidget,
            QDoubleSpinBox,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )

        # helpers
        def _hslider(lo: int = 0, hi: int = 100, val: int = 0) -> QSlider:
            sl = QSlider(Qt_.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(val)
            return sl

        def _group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
            grp = QGroupBox(title)
            ly = QVBoxLayout(grp)
            ly.setSpacing(4)
            ly.setContentsMargins(6, 12, 6, 6)
            return grp, ly

        # root panel
        panel = QWidget()
        panel.setMinimumWidth(210)
        panel.setMaximumWidth(260)
        panel.setStyleSheet(_PANEL_DARK_STYLE)
        root = QVBoxLayout(panel)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)

        # Surface
        grp, ly = _group("Surface")
        surf_combo = QComboBox()
        surf_combo.addItems(_SURFACES)
        surf_combo.setCurrentText(self._surf)
        surf_combo.currentTextChanged.connect(
            lambda s: self.set_surface(s) if s != self._surf else None
        )
        ly.addWidget(surf_combo)
        root.addWidget(grp)

        # Parcellation
        grp, ly = _group("Parcellation")
        _PARC_ITEMS = ["None", "aparc  (Desikan)", "aparc.a2009s  (Destrieux)"]
        self._parc_combo = QComboBox()
        self._parc_combo.addItems(_PARC_ITEMS)
        self._parc_combo.currentTextChanged.connect(self._set_parcellation)
        ly.addWidget(self._parc_combo)
        self._parc_status_lbl = QLabel("")
        self._parc_status_lbl.setWordWrap(True)
        self._parc_status_lbl.setStyleSheet("color:#8b949e; font-size:9px;")
        ly.addWidget(self._parc_status_lbl)
        root.addWidget(grp)

        # Hemispheres
        grp, ly = _group("Hemispheres")
        hl = QHBoxLayout()
        lh_cb = QCheckBox("Left")
        lh_cb.setChecked(True)
        rh_cb = QCheckBox("Right")
        rh_cb.setChecked(True)

        def _on_lh(state: int) -> None:
            self._hemi_visible["lh"] = bool(state)
            self._refresh_scalars()

        def _on_rh(state: int) -> None:
            self._hemi_visible["rh"] = bool(state)
            self._refresh_scalars()

        lh_cb.stateChanged.connect(_on_lh)
        rh_cb.stateChanged.connect(_on_rh)
        hl.addWidget(lh_cb)
        hl.addWidget(rh_cb)
        ly.addLayout(hl)
        root.addWidget(grp)

        # Colormap
        grp, ly = _group("Colormap")
        cmap_combo = QComboBox()
        cmap_combo.addItems(_CMAPS)
        cmap_combo.setCurrentText(_CMAPS[self._cmap_idx])

        def _on_cmap(name: str) -> None:
            idx = _CMAPS.index(name)
            if idx == self._cmap_idx:
                return
            self._cmap_idx = idx
            self._update_scalar_bar()
            self._refresh_scalars()

        cmap_combo.currentTextChanged.connect(_on_cmap)
        ly.addWidget(cmap_combo)
        root.addWidget(grp)

        # Color range (clim)
        grp, ly = _group("Color Range")
        clim_max_lbl = QLabel(f"max  {self._clim[1]:.2f}")
        clim_max_sl = _hslider(0, 100, int(self._clim[1] * 100))
        clim_min_lbl = QLabel(f"min  {self._clim[0]:.2f}")
        clim_min_sl = _hslider(0, 100, int(self._clim[0] * 100))

        def _on_clim_max(v: int) -> None:
            self._clim[1] = v / 100.0
            clim_max_lbl.setText(f"max  {self._clim[1]:.2f}")
            self._update_scalar_bar()
            self._refresh_scalars()

        def _on_clim_min(v: int) -> None:
            self._clim[0] = v / 100.0
            clim_min_lbl.setText(f"min  {self._clim[0]:.2f}")
            self._update_scalar_bar()
            self._refresh_scalars()

        clim_max_sl.valueChanged.connect(_on_clim_max)
        clim_min_sl.valueChanged.connect(_on_clim_min)
        ly.addWidget(clim_max_lbl)
        ly.addWidget(clim_max_sl)
        ly.addWidget(clim_min_lbl)
        ly.addWidget(clim_min_sl)
        root.addWidget(grp)

        # Opacity
        grp, ly = _group("Opacity")
        op_lbl = QLabel(f"{self._opacity:.2f}")
        op_sl = _hslider(0, 100, int(self._opacity * 100))

        def _on_opacity(v: int) -> None:
            self._opacity = v / 100.0
            op_lbl.setText(f"{self._opacity:.2f}")
            self._refresh_scalars()

        op_sl.valueChanged.connect(_on_opacity)
        ly.addWidget(op_lbl)
        ly.addWidget(op_sl)
        root.addWidget(grp)

        # Threshold
        grp, ly = _group("Threshold  (0 → 1)")
        thr_lbl = QLabel(f"{self._threshold:.3f}")
        thr_sl = _hslider(0, 100, 0)

        def _on_threshold(v: int) -> None:
            self._threshold = v / 100.0
            thr_lbl.setText(f"{self._threshold:.3f}")
            self._refresh_scalars()

        thr_sl.valueChanged.connect(_on_threshold)
        ly.addWidget(thr_lbl)
        ly.addWidget(thr_sl)
        root.addWidget(grp)

        # Background
        grp, ly = _group("Background")
        bg_combo = QComboBox()
        bg_combo.addItems(list(_BACKGROUNDS.keys()))
        bg_combo.setCurrentText(_DEFAULT_BG)

        def _on_bg(name: str) -> None:
            self._bg_name = name
            bot, top = _BACKGROUNDS[name]
            p.set_background(bot, top=top)
            p.render()

        bg_combo.currentTextChanged.connect(_on_bg)
        ly.addWidget(bg_combo)
        root.addWidget(grp)

        # View presets
        grp, ly = _group("View Presets")
        btn_row = QHBoxLayout()
        for label, key, tip in [
            ("L", "lateral_lh", "Left lateral"),
            ("R", "lateral_rh", "Right lateral"),
            ("D", "dorsal", "Dorsal (top)"),
            ("F", "frontal", "Frontal"),
            ("V", "ventral", "Ventral (bottom)"),
        ]:
            btn = QPushButton(label)
            btn.setFixedSize(32, 28)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, k=key: self._set_view(k))
            btn_row.addWidget(btn)
        ly.addLayout(btn_row)
        root.addWidget(grp)

        # Display Mode
        grp, ly = _group("Display Mode")
        mode_combo = self._mode_combo = QComboBox()
        mode_combo.addItems(_DISPLAY_MODES)
        mode_combo.setCurrentText(self._display_mode)
        mode_lbl = QLabel(
            "<span style='color:#8b949e;font-size:9px;'>"
            "Activation, or source power in the chosen band; "
            "applies from the next window"
            "</span>"
        )
        mode_lbl.setWordWrap(True)

        def _on_display_mode(name: str) -> None:
            self.set_display_mode(name)

        mode_combo.currentTextChanged.connect(_on_display_mode)
        ly.addWidget(mode_combo)
        ly.addWidget(mode_lbl)
        root.addWidget(grp)

        # Custom frequency band
        grp, ly = _group("Custom Band  (Hz)")
        band_row = QHBoxLayout()
        self._custom_lo_spin = QDoubleSpinBox()
        self._custom_lo_spin.setRange(0.5, 200.0)
        self._custom_lo_spin.setValue(13.0)
        self._custom_lo_spin.setSuffix(" Hz")
        self._custom_lo_spin.setDecimals(1)
        dash_lbl2 = QLabel("–")
        dash_lbl2.setStyleSheet("color:#8b949e;")
        self._custom_hi_spin = QDoubleSpinBox()
        self._custom_hi_spin.setRange(1.0, 500.0)
        self._custom_hi_spin.setValue(30.0)
        self._custom_hi_spin.setSuffix(" Hz")
        self._custom_hi_spin.setDecimals(1)
        band_row.addWidget(self._custom_lo_spin)
        band_row.addWidget(dash_lbl2)
        band_row.addWidget(self._custom_hi_spin)
        ly.addLayout(band_row)

        btn_band_row = QHBoxLayout()
        btn_set_band = QPushButton("Set")

        def _on_set_band() -> None:
            lo, hi = self._custom_lo_spin.value(), self._custom_hi_spin.value()
            if lo < hi:  # an empty band is ignored, as before
                self.set_custom_band(lo, hi)

        btn_set_band.clicked.connect(_on_set_band)
        btn_clr_band = QPushButton("Clear")
        btn_clr_band.clicked.connect(self.clear_custom_band)
        btn_band_row.addWidget(btn_set_band)
        btn_band_row.addWidget(btn_clr_band)
        ly.addLayout(btn_band_row)

        self._custom_band_lbl = QLabel("No custom band set")
        self._custom_band_lbl.setStyleSheet("color:#8b949e; font-size:10px;")
        self._custom_band_lbl.setWordWrap(True)
        ly.addWidget(self._custom_band_lbl)
        root.addWidget(grp)

        # Actions
        grp, ly = _group("Actions")
        reset_btn = QPushButton("Reset activity  (r)")
        reset_btn.clicked.connect(self.reset_activity)
        shot_btn = QPushButton("Screenshot  (s)")
        shot_btn.clicked.connect(lambda: self.screenshot())
        ly.addWidget(reset_btn)
        ly.addWidget(shot_btn)
        root.addWidget(grp)

        root.addStretch()

        # Keyboard hint
        hint = QLabel(
            "<span style='color:#8b949e; font-size:10px;'>"
            "Keys: 1-5 views · i/p/w/h surface · s screenshot · r reset"
            "</span>"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Dock widget
        dock = QDockWidget("Brain Controls")
        dock.setWidget(panel)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable,
        )
        p.app_window.addDockWidget(Qt_.DockWidgetArea.RightDockWidgetArea, dock)

    def _add_key_bindings(self) -> None:
        p = self._plotter
        p.add_key_event("1", lambda: self._set_view("lateral_lh"))
        p.add_key_event("2", lambda: self._set_view("lateral_rh"))
        p.add_key_event("3", lambda: self._set_view("dorsal"))
        p.add_key_event("4", lambda: self._set_view("frontal"))
        p.add_key_event("5", lambda: self._set_view("ventral"))
        p.add_key_event("s", self.screenshot)
        p.add_key_event("r", self.reset_activity)
        p.add_key_event("i", lambda: self.set_surface("inflated"))
        p.add_key_event("p", lambda: self.set_surface("pial"))
        p.add_key_event("w", lambda: self.set_surface("white"))
        p.add_key_event("h", lambda: self.set_surface("sphere"))

    def _add_overlays(self) -> None:
        self._rec_label = self._plotter.add_text(
            "",
            position="lower_right",
            font_size=12,
            color="#FF4444",
        )

    # Internal helpers

    def _refresh_scalars(self, deferred: bool = False) -> None:
        """Colour the cortex: activity over the sulcal grey, where it is shown.

        Activity is shown where it is finite, at least the threshold, and in a
        visible hemisphere; the opacity sets how much of the grey it covers.
        """
        v = self._scalars_full
        lo, hi = self._clim
        with np.errstate(invalid="ignore"):
            shown = np.isfinite(v) & (v >= self._threshold)
            t = np.clip((v - lo) / ((hi - lo) or 1.0), 0.0, 1.0)
        if not self._hemi_visible["lh"]:
            shown[: self._n_lh] = False
        if not self._hemi_visible["rh"]:
            shown[self._n_lh :] = False
        idx = np.nan_to_num(t * 255.0).astype(np.intp)
        w = (self._opacity * shown)[:, np.newaxis]
        rgb = self._base_rgb * (1.0 - w) + _lut(_CMAPS[self._cmap_idx])[idx] * w
        self._rgb[:] = (rgb * 255.0).astype(np.uint8)
        if not deferred:
            self._plotter.render()

    def _set_view(self, preset: str) -> None:
        if preset not in _VIEW_PRESETS:
            return
        v = _VIEW_PRESETS[preset]
        self._plotter.camera_position = [v["position"], v["focal"], v["up"]]
        self._plotter.render()

    def _rebuild_actors(self) -> None:
        """Remove and re-add the cortex actor after a surface swap."""
        p = self._plotter
        p.remove_actor(self._actor)
        self._remove_parc_actor()
        self._add_cortex_actor(p)

        # Re-overlay parcellation borders on the new surface geometry
        if self._parc_edges is not None:
            self._build_parc_actor()

        p.render()

    # Parcellation border overlay

    def _set_parcellation(self, parc_label: str) -> None:
        """Load annotation and draw parcel boundary edges on the surface."""
        self._remove_parc_actor()
        self._parc_edges = None
        self._parc_name = None

        if parc_label == "None":
            if hasattr(self, "_parc_status_lbl"):
                self._parc_status_lbl.setText("")
            return

        parc_id = parc_label.split()[0]  # "aparc  (Desikan)" → "aparc"
        logger.info("Loading parcellation %r …", parc_id)
        if hasattr(self, "_parc_status_lbl"):
            self._parc_status_lbl.setText("Loading …")

        try:
            from pathlib import Path as _Path

            import mne

            fs_dir = _Path(mne.datasets.fetch_fsaverage(verbose=False))
            subjects_dir = str(fs_dir.parent)

            labels_lh = mne.read_labels_from_annot(
                "fsaverage",
                parc=parc_id,
                hemi="lh",
                subjects_dir=subjects_dir,
                verbose=False,
            )
            labels_rh = mne.read_labels_from_annot(
                "fsaverage",
                parc=parc_id,
                hemi="rh",
                subjects_dir=subjects_dir,
                verbose=False,
            )

            n_lh = self._n_lh
            n_rh = self._mesh.n_points - n_lh
            parcel_lh = np.full(n_lh, -1, dtype=np.int32)
            parcel_rh = np.full(n_rh, -1, dtype=np.int32)

            for i, lbl in enumerate(labels_lh):
                verts = lbl.vertices[lbl.vertices < n_lh]
                parcel_lh[verts] = i
            for i, lbl in enumerate(labels_rh):
                verts = lbl.vertices[lbl.vertices < n_rh]
                parcel_rh[verts] = i

            parcel_all = np.concatenate([parcel_lh, parcel_rh])

            # Extract triangle edges and keep those crossing parcel boundaries
            faces = self._mesh.faces.reshape(-1, 4)[:, 1:]  # (n_faces, 3)
            e01 = np.sort(faces[:, :2], axis=1)
            e12 = np.sort(faces[:, 1:], axis=1)
            e02 = np.sort(faces[:, [0, 2]], axis=1)
            all_edges = np.unique(np.vstack([e01, e12, e02]), axis=0)

            mask = parcel_all[all_edges[:, 0]] != parcel_all[all_edges[:, 1]]
            self._parc_edges = all_edges[mask]
            self._parc_name = parc_id

            self._build_parc_actor()
            n_bnd = len(self._parc_edges)
            logger.info("Parcellation %r: %d boundary edges", parc_id, n_bnd)
            if hasattr(self, "_parc_status_lbl"):
                self._parc_status_lbl.setText(f"{n_bnd:,} boundary edges")

        except Exception as exc:
            self._parc_edges = None
            self._parc_name = None
            logger.warning("Parcellation load failed: %s", exc)
            if hasattr(self, "_parc_status_lbl"):
                self._parc_status_lbl.setText(f"Error: {exc}")

    def _build_parc_actor(self) -> None:
        """Create a pyvista line mesh from stored boundary edges and add it."""
        if self._parc_edges is None or len(self._parc_edges) == 0:
            return
        pts = self._mesh.points
        n_segs = len(self._parc_edges)

        seg_pts = np.empty((n_segs * 2, 3), dtype=np.float32)
        seg_pts[0::2] = pts[self._parc_edges[:, 0]]
        seg_pts[1::2] = pts[self._parc_edges[:, 1]]

        lines_conn = np.empty(n_segs * 3, dtype=np.int_)
        lines_conn[0::3] = 2
        lines_conn[1::3] = np.arange(0, n_segs * 2, 2)
        lines_conn[2::3] = np.arange(1, n_segs * 2, 2)

        bnd = pv.PolyData(seg_pts, lines=lines_conn)
        self._parc_actor = self._plotter.add_mesh(
            bnd,
            color="#ffffff",
            line_width=2.0,
            render_lines_as_tubes=False,
            show_scalar_bar=False,
        )

    def _remove_parc_actor(self) -> None:
        if self._parc_actor is not None:
            self._plotter.remove_actor(self._parc_actor)
            self._parc_actor = None

    # Custom frequency band

    def set_custom_band(self, lo: float, hi: float) -> None:
        """Show source power in ``lo``–``hi`` Hz, from the next window on.

        Parameters
        ----------
        lo, hi : float
            Band edges in Hz, ``lo < hi``.
        """
        if not 0 < lo < hi:
            raise ValueError(f"Need 0 < lo < hi, got ({lo}, {hi}).")
        # On top of the display mode, which stays as it was: Clear goes back to it.
        self._custom_band_hz = (float(lo), float(hi))
        if hasattr(self, "_custom_band_lbl"):
            self._custom_band_lbl.setText(f"Active: {lo:.1f}–{hi:.1f} Hz")
            self._custom_band_lbl.setStyleSheet("color:#80d8ff; font-size:10px;")
        self._reset_smoothing()
        logger.info("BrainPlot custom band → %.1f–%.1f Hz", lo, hi)

    def set_band(self, band: tuple[float, float] | None) -> None:
        """Show activation (``None``) or source power in ``band``, in Hz.

        A band that one of the display modes covers selects that mode; any
        other becomes a custom band.
        """
        if band is None:
            self.set_display_mode("Source Activation")
            return
        band = (float(band[0]), float(band[1]))
        for mode, mode_band in _MODE_BANDS.items():
            if mode_band == band:
                self.set_display_mode(mode)
                return
        self.set_custom_band(*band)

    def clear_custom_band(self) -> None:
        """Go back to the display mode's own band."""
        self._custom_band_hz = None
        if hasattr(self, "_custom_band_lbl"):
            self._custom_band_lbl.setText("No custom band set")
            self._custom_band_lbl.setStyleSheet("color:#8b949e; font-size:10px;")
        self._reset_smoothing()
        logger.info("BrainPlot custom band cleared")

    @property
    def custom_band(self) -> tuple[float, float] | None:
        """Active custom frequency band ``(lo, hi)`` in Hz, or ``None``."""
        return self._custom_band_hz

    @property
    def band(self) -> tuple[float, float] | None:
        """What to compute for the display: a band in Hz, or ``None`` for activation.

        The custom band when one is set, else the display mode's.
        :meth:`~mne_rt.RTStream.record_main` reads it for every window, so the
        controls change what is computed, not just the label.
        """
        if self._custom_band_hz is not None:
            return self._custom_band_hz
        return _MODE_BANDS.get(self._display_mode)

    def _reset_smoothing(self) -> None:
        # What is shown changes meaning: blending in the old values would put
        # the previous band's map under the new label for the next frames.
        self._lh_ema = None
        self._rh_ema = None

    # Public interface

    def set_surface(self, surf: str) -> None:
        """Switch the cortical surface geometry.

        Parameters
        ----------
        surf : {"inflated", "pial", "white", "sphere"}
            Target surface geometry.
        """
        if surf not in _SURFACES:
            raise ValueError(f"surf {surf!r} not recognised.  Choose from {_SURFACES}.")
        if surf == self._surf:
            return

        logger.info("Switching surface: %s → %s", self._surf, surf)
        saved_scalars = self._scalars_full.copy()
        self._surf = surf
        self._load_surface(surf)

        if len(saved_scalars) == len(self._scalars_full):
            self._scalars_full[:] = saved_scalars

        self._rebuild_actors()
        logger.info("Surface changed to %r.", surf)

    def update_from_arrays(
        self,
        lh_scalars: np.ndarray,
        rh_scalars: np.ndarray,
        mode: str = "power",
        deferred: bool = False,
    ) -> None:
        """Update the brain display from per-source-vertex values.

        Parameters
        ----------
        lh_scalars, rh_scalars : ndarray, shape (10242,)
            One value per ico-5 vertex of each hemisphere (NaN where there is
            none, which is drawn without activity).
        mode : str, default "power"
            Kept for API symmetry; unused internally.
        deferred : bool, default False
            If ``True``, skip the immediate ``render()`` call.
        """
        lh_scalars = np.asarray(lh_scalars, dtype=float)
        rh_scalars = np.asarray(rh_scalars, dtype=float)
        if self._display_alpha < 1.0:
            if self._lh_ema is None:
                self._lh_ema, self._rh_ema = lh_scalars.copy(), rh_scalars.copy()
            else:
                a = self._display_alpha
                self._lh_ema = ema_step(self._lh_ema, lh_scalars, a)
                self._rh_ema = ema_step(self._rh_ema, rh_scalars, a)
            # A vertex with no value this frame is drawn blank, not with the
            # average it had: what is shown is never older than the data.
            lh_scalars = np.where(np.isfinite(lh_scalars), self._lh_ema, np.nan)
            rh_scalars = np.where(np.isfinite(rh_scalars), self._rh_ema, np.nan)

        n_lh = self._n_lh
        self._scalars_full[:n_lh] = lh_scalars[self._nn_map["lh"]]
        self._scalars_full[n_lh:] = rh_scalars[self._nn_map["rh"]]
        self._refresh_scalars(deferred=deferred)

    def update(self, stc, mode: str = "power") -> None:
        """Show one frame of a surface source estimate on fsaverage.

        Parameters
        ----------
        stc : mne.SourceEstimate
            On an fsaverage ico-5 (or coarser) surface source space.
        mode : {"power", "activation"}, default "power"
            ``"power"`` displays mean squared amplitude;
            ``"activation"`` displays the time-averaged amplitude.
        """
        vals = np.mean(stc.data**2, axis=1) if mode == "power" else stc.data.mean(axis=1)
        self.update_from_arrays(*to_display(vals, stc.vertices))

    def set_display_mode(self, mode: str) -> None:
        """Choose what the display shows, from the next window on.

        Parameters
        ----------
        mode : str
            One of :data:`_DISPLAY_MODES`: ``"Source Activation"``, or source
            power in one of the named bands (see :attr:`band`). Clears any
            custom band.
        """
        if mode not in _DISPLAY_MODES:
            raise ValueError(f"mode {mode!r} not recognised.  Choose from {_DISPLAY_MODES}.")
        # The mode first: `band` reads the custom band before the mode, so a
        # window submitted in between still gets one of the two, not neither.
        self._display_mode = mode
        self._custom_band_hz = None
        if hasattr(self, "_custom_band_lbl"):
            self._custom_band_lbl.setText("No custom band set")
            self._custom_band_lbl.setStyleSheet("color:#8b949e; font-size:10px;")
        combo = getattr(self, "_mode_combo", None)
        if combo is not None and combo.currentText() != mode:
            # Set from code (record_main): the menu has to say so, or picking
            # what it shows would change nothing.
            combo.blockSignals(True)
            combo.setCurrentText(mode)
            combo.blockSignals(False)
        self._reset_smoothing()
        logger.info("BrainPlot display mode → %r (band %s)", mode, self.band)

    @property
    def display_mode(self) -> str:
        """Current display mode string (e.g. ``"Alpha Power  (8–13 Hz)"``).

        ``"Custom  lo–hi Hz"`` while a custom band is set. What is computed
        is :attr:`band`.
        """
        if self._custom_band_hz is not None:
            lo, hi = self._custom_band_hz
            return f"Custom  {lo:.1f}–{hi:.1f} Hz"
        return self._display_mode

    def reset_activity(self) -> None:
        """Zero out all activity scalars and refresh the display."""
        self._scalars_full[:] = 0.0
        self._reset_smoothing()  # or the next update blends into what was cleared
        self._refresh_scalars()

    # Video recording

    def record_video(self, path: Union[str, Path, None] = None) -> Path:
        """Start recording the brain display to an MP4 video file.

        Parameters
        ----------
        path : str | Path | None, default None
            Destination file.  Defaults to ``~/ant_brain_<timestamp>.mp4``.

        Returns
        -------
        path : Path
        """
        import imageio

        if self._recording:
            raise RuntimeError("Already recording. Call stop_recording() first.")
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path.home() / f"ant_brain_{ts}.mp4"
        path = Path(path)
        self._video_writer = imageio.get_writer(
            str(path), fps=24, macro_block_size=None, plugin="ffmpeg"
        )
        self._video_path = path
        self._recording = True
        self._rec_label.SetInput("● REC")
        self._plotter.render()
        logger.info("Brain video recording started: %s", path)
        return path

    def stop_recording(self) -> Union[Path, None]:
        """Stop recording and finalise the video file.

        Returns
        -------
        path : Path | None
        """
        if not self._recording:
            return None
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        self._recording = False
        self._rec_label.SetInput("")
        self._plotter.render()
        path = self._video_path
        self._video_path = None
        logger.info("Brain video saved: %s", path)
        return path

    def write_frame_if_recording(self) -> None:
        """Capture a video frame if recording is active (no-op otherwise)."""
        if not self._recording or self._video_writer is None:
            return
        try:
            frame = self._plotter.screenshot(return_img=True)
            self._video_writer.append_data(frame)
        except Exception:
            if not getattr(self, "_frame_warned", False):
                self._frame_warned = True
                logger.warning("Writing a video frame failed.", exc_info=True)

    def screenshot(self, path: Union[str, Path, None] = None) -> Path:
        """Save a PNG screenshot of the current brain view.

        Parameters
        ----------
        path : str | Path | None, default None
            Destination file.  Defaults to ``~/ant_brain_<timestamp>.png``.

        Returns
        -------
        path : Path
        """
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path.home() / f"ant_brain_{ts}.png"
        self._plotter.screenshot(str(path))
        logger.info("Screenshot saved: %s", path)
        return Path(path)

    @property
    def plotter(self) -> "_BackgroundPlotter":
        """The underlying :class:`pyvistaqt.BackgroundPlotter` instance."""
        return self._plotter

    @property
    def surf(self) -> str:
        """Currently displayed surface geometry name."""
        return self._surf
