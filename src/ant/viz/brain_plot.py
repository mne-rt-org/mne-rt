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
import time
from pathlib import Path
from typing import Union

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

from ant._logging import logger
from ant.tools import setup_surface


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# Suggested clim ranges for each display mode
_DISPLAY_CLIM_HINTS: dict[str, tuple[float, float]] = {
    "Source Activation":   (0.0, 0.6),
    "Alpha Power  (8–13 Hz)":  (0.0, 0.5),
    "Beta Power   (13–30 Hz)": (0.0, 0.3),
    "Theta Power  (4–8 Hz)":   (0.0, 0.5),
    "Gamma Power  (30–80 Hz)": (0.0, 0.2),
    "SMR Power    (12–15 Hz)": (0.0, 0.4),
}

_VIEW_PRESETS: dict[str, dict] = {
    "lateral_lh": {"position": (-450, 0, 0),  "focal": (0, 0, 0), "up": (0, 0, 1)},
    "lateral_rh": {"position": (450, 0, 0),   "focal": (0, 0, 0), "up": (0, 0, 1)},
    "dorsal":     {"position": (0, 0, 450),   "focal": (0, 0, 0), "up": (0, 1, 0)},
    "frontal":    {"position": (0, -450, 0),  "focal": (0, 0, 0), "up": (0, 0, 1)},
    "ventral":    {"position": (0, 0, -450),  "focal": (0, 0, 0), "up": (0, 1, 0)},
}

# Background colour presets  (bottom_hex, top_hex)
_BACKGROUNDS: dict[str, tuple[str, str]] = {
    "Deep space":   ("#040810", "#0b1628"),
    "Midnight":     ("#050d1a", "#0d1b35"),
    "Slate":        ("#0d1117", "#1c2333"),
    "Charcoal":     ("#111111", "#1e1e1e"),
    "Black":        ("#000000", "#050505"),
    "Light":        ("#e8eaf0", "#ffffff"),   # publication / presentations
}

_DEFAULT_BG = "Midnight"


class BrainPlot:
    """Interactive real-time 3D brain activation display.

    Renders bilateral ``fsaverage`` cortical surfaces with a colour-mapped
    activity overlay and a Qt control panel docked on the right.  Designed
    to run alongside :class:`~ant.viz.NFSignalPlot` inside a shared Qt
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
    verbose : bool | str | None, default None
        Verbosity level.

    See Also
    --------
    ant.viz.NFSignalPlot : Scrolling real-time NF signal plot.
    ant.NFRealtime.record_main : Main NF loop that drives both plots.

    Notes
    -----
    Activity values are spread from the 10 242 ico-5 source vertices to all
    163 842 fsaverage surface vertices via nearest-neighbour interpolation,
    giving a smooth spatial appearance.

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
        verbose: Union[bool, str, None] = None,
    ) -> None:
        if not _pyvista_available or not _pyvistaqt_available:
            raise ImportError(
                "pyvista and pyvistaqt are required for BrainPlot. "
                "Install them with:  pip install 'ANT[viz]'"
            )
        from ant._logging import set_log_level
        set_log_level(verbose)

        if cmap not in _CMAPS:
            raise ValueError(f"cmap {cmap!r} not recognised.  Choose from {_CMAPS}.")
        if surf not in _SURFACES:
            raise ValueError(f"surf {surf!r} not recognised.  Choose from {_SURFACES}.")

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

        logger.info("Loading %s surface …", surf)
        self._load_surface(surf)

        self._plotter = self._build_plotter(window_size)
        self._add_key_bindings()
        self._add_overlays()
        logger.info("BrainPlot ready.")

    # ------------------------------------------------------------------
    # Surface management
    # ------------------------------------------------------------------

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
        )
        self._n_lh = int(self._hemi_offsets["rh"])

    # ------------------------------------------------------------------
    # Plotter construction
    # ------------------------------------------------------------------

    def _build_plotter(self, window_size: tuple[int, int]) -> "_BackgroundPlotter":
        p = _BackgroundPlotter(
            window_size=tuple(window_size),
            lighting="three lights",
            title="Advanced Neurofeedback Toolbox — Brain Activation",
            toolbar=False,
            menu_bar=False,
            editor=False,
        )

        # Background gradient
        bg_bot, bg_top = _BACKGROUNDS[_DEFAULT_BG]
        p.set_background(bg_bot, top=bg_top)

        # ── Base sulcal-depth mesh ────────────────────────────────────────
        # sulc values: negative = gyri (light), positive = sulci (dark)
        # "Greys_r" maps low→white, high→black → correct neuroimaging convention
        self._base_actor = p.add_mesh(
            self._mesh,
            scalars="base",
            cmap="Greys_r",
            clim=[-1.0, 1.5],
            smooth_shading=True,
            show_scalar_bar=False,
        )

        # ── Activity overlay ──────────────────────────────────────────────
        self._act_actor = p.add_mesh(
            self._mesh,
            scalars="activity",
            cmap=_CMAPS[self._cmap_idx],
            opacity=self._opacity,
            clim=self._clim,
            smooth_shading=True,
            show_scalar_bar=False,
            interpolate_before_map=True,
            nan_opacity=0.0,
        )

        # ── Scalar bar (no tick numbers) ──────────────────────────────────
        p.add_scalar_bar(
            title="Activity",
            italic=True,
            vertical=True,
            position_x=0.02,
            position_y=0.20,
            height=0.38,
            width=0.04,
            color="white",
            title_font_size=12,
            label_font_size=10,
            n_labels=0,
        )

        # ── Lighting & post-processing ────────────────────────────────────
        p.enable_eye_dome_lighting()
        p.enable_ssao(radius=0.5, bias=0.005, kernel_size=256)
        p.enable_anti_aliasing("ssaa")
        p.add_camera_orientation_widget()
        p.camera_position = "yz"
        p.camera.azimuth = 45

        self._add_control_panel(p)
        return p

    # ------------------------------------------------------------------
    # Qt control panel
    # ------------------------------------------------------------------

    def _add_control_panel(self, p: "_BackgroundPlotter") -> None:
        """Dock a Qt control panel on the right side of the brain window."""
        from PyQt6.QtWidgets import (
            QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QSlider, QComboBox, QCheckBox, QPushButton, QGroupBox,
        )
        from PyQt6.QtCore import Qt as Qt_

        # ── helpers ──────────────────────────────────────────────────────
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

        # ── root panel ────────────────────────────────────────────────────
        panel = QWidget()
        panel.setMinimumWidth(210)
        panel.setMaximumWidth(260)
        root = QVBoxLayout(panel)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Surface ───────────────────────────────────────────────────────
        grp, ly = _group("Surface")
        surf_combo = QComboBox()
        surf_combo.addItems(_SURFACES)
        surf_combo.setCurrentText(self._surf)
        surf_combo.currentTextChanged.connect(
            lambda s: self.set_surface(s) if s != self._surf else None
        )
        ly.addWidget(surf_combo)
        root.addWidget(grp)

        # ── Hemispheres ───────────────────────────────────────────────────
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

        # ── Colormap ──────────────────────────────────────────────────────
        grp, ly = _group("Colormap")
        cmap_combo = QComboBox()
        cmap_combo.addItems(_CMAPS)
        cmap_combo.setCurrentText(_CMAPS[self._cmap_idx])

        def _on_cmap(name: str) -> None:
            idx = _CMAPS.index(name)
            if idx == self._cmap_idx:
                return
            self._cmap_idx = idx
            self._sync_scalar_bar_lut()
            p.render()

        cmap_combo.currentTextChanged.connect(_on_cmap)
        ly.addWidget(cmap_combo)
        root.addWidget(grp)

        # ── Color range (clim) ────────────────────────────────────────────
        grp, ly = _group("Color Range")
        clim_max_lbl = QLabel(f"max  {self._clim[1]:.2f}")
        clim_max_sl = _hslider(0, 100, int(self._clim[1] * 100))
        clim_min_lbl = QLabel(f"min  {self._clim[0]:.2f}")
        clim_min_sl = _hslider(0, 100, int(self._clim[0] * 100))

        def _on_clim_max(v: int) -> None:
            self._clim[1] = v / 100.0
            clim_max_lbl.setText(f"max  {self._clim[1]:.2f}")
            self._act_actor.GetMapper().SetScalarRange(*self._clim)
            p.render()

        def _on_clim_min(v: int) -> None:
            self._clim[0] = v / 100.0
            clim_min_lbl.setText(f"min  {self._clim[0]:.2f}")
            self._act_actor.GetMapper().SetScalarRange(*self._clim)
            p.render()

        clim_max_sl.valueChanged.connect(_on_clim_max)
        clim_min_sl.valueChanged.connect(_on_clim_min)
        ly.addWidget(clim_max_lbl)
        ly.addWidget(clim_max_sl)
        ly.addWidget(clim_min_lbl)
        ly.addWidget(clim_min_sl)
        root.addWidget(grp)

        # ── Opacity ───────────────────────────────────────────────────────
        grp, ly = _group("Opacity")
        op_lbl = QLabel(f"{self._opacity:.2f}")
        op_sl = _hslider(0, 100, int(self._opacity * 100))

        def _on_opacity(v: int) -> None:
            self._opacity = v / 100.0
            op_lbl.setText(f"{self._opacity:.2f}")
            self._act_actor.GetProperty().SetOpacity(self._opacity)
            p.render()

        op_sl.valueChanged.connect(_on_opacity)
        ly.addWidget(op_lbl)
        ly.addWidget(op_sl)
        root.addWidget(grp)

        # ── Threshold ─────────────────────────────────────────────────────
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

        # ── Background ────────────────────────────────────────────────────
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

        # ── View presets ──────────────────────────────────────────────────
        grp, ly = _group("View Presets")
        btn_row = QHBoxLayout()
        for label, key, tip in [
            ("L", "lateral_lh", "Left lateral"),
            ("R", "lateral_rh", "Right lateral"),
            ("D", "dorsal",     "Dorsal (top)"),
            ("F", "frontal",    "Frontal"),
            ("V", "ventral",    "Ventral (bottom)"),
        ]:
            btn = QPushButton(label)
            btn.setFixedSize(32, 28)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, k=key: self._set_view(k))
            btn_row.addWidget(btn)
        ly.addLayout(btn_row)
        root.addWidget(grp)

        # ── Display Mode ──────────────────────────────────────────────────
        grp, ly = _group("Display Mode")
        mode_combo = QComboBox()
        mode_combo.addItems(_DISPLAY_MODES)
        mode_combo.setCurrentText(self._display_mode)
        mode_lbl = QLabel(
            "<span style='color:#666;font-size:9px;'>"
            "Configure NFRealtime modality to match selected mode"
            "</span>"
        )
        mode_lbl.setWordWrap(True)

        def _on_display_mode(name: str) -> None:
            self.set_display_mode(name)

        mode_combo.currentTextChanged.connect(_on_display_mode)
        ly.addWidget(mode_combo)
        ly.addWidget(mode_lbl)
        root.addWidget(grp)

        # ── Actions ───────────────────────────────────────────────────────
        grp, ly = _group("Actions")
        reset_btn = QPushButton("Reset activity  (r)")
        reset_btn.clicked.connect(self.reset_activity)
        shot_btn = QPushButton("Screenshot  (s)")
        shot_btn.clicked.connect(lambda: self.screenshot())
        ly.addWidget(reset_btn)
        ly.addWidget(shot_btn)
        root.addWidget(grp)

        root.addStretch()

        # ── Keyboard hint ─────────────────────────────────────────────────
        hint = QLabel(
            "<span style='color:#555; font-size:10px;'>"
            "Keys: 1-5 views · i/p/w/h surface · s screenshot · r reset"
            "</span>"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        # ── Dock widget ───────────────────────────────────────────────────
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_scalars(self, deferred: bool = False) -> None:
        """Recompute display scalars respecting threshold and hemisphere masks."""
        display = self._scalars_full.copy()
        display[display < self._threshold] = np.nan
        if not self._hemi_visible["lh"]:
            display[: self._n_lh] = np.nan
        if not self._hemi_visible["rh"]:
            display[self._n_lh :] = np.nan
        self._mesh["activity"] = display
        if not deferred:
            self._plotter.render()

    def _set_view(self, preset: str) -> None:
        if preset not in _VIEW_PRESETS:
            return
        v = _VIEW_PRESETS[preset]
        self._plotter.camera_position = [v["position"], v["focal"], v["up"]]
        self._plotter.render()

    def _rebuild_actors(self) -> None:
        """Remove and re-add mesh actors after a surface swap."""
        p = self._plotter
        p.remove_actor(self._base_actor)
        p.remove_actor(self._act_actor)

        self._base_actor = p.add_mesh(
            self._mesh,
            scalars="base",
            cmap="Greys_r",
            clim=[-1.0, 1.5],
            smooth_shading=True,
            show_scalar_bar=False,
        )
        self._act_actor = p.add_mesh(
            self._mesh,
            scalars="activity",
            cmap=_CMAPS[self._cmap_idx],
            opacity=self._opacity,
            clim=self._clim,
            smooth_shading=True,
            show_scalar_bar=False,
            interpolate_before_map=True,
            nan_opacity=0.0,
        )
        self._sync_scalar_bar_lut()
        p.render()

    def _sync_scalar_bar_lut(self) -> None:
        """Point the scalar bar LUT at the current activity actor's mapper."""
        import matplotlib
        cmap_obj = matplotlib.colormaps[_CMAPS[self._cmap_idx]]
        lut = self._act_actor.GetMapper().GetLookupTable()
        lut.SetNumberOfColors(256)
        for i in range(256):
            r, g, b, a = cmap_obj(i / 255.0)
            lut.SetTableValue(i, r, g, b, a)
        lut.Build()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
            self._mesh["activity"] = saved_scalars

        self._rebuild_actors()
        logger.info("Surface changed to %r.", surf)

    def update_from_arrays(
        self,
        lh_scalars: np.ndarray,
        rh_scalars: np.ndarray,
        mode: str = "power",
        deferred: bool = False,
    ) -> None:
        """Update the brain display from pre-computed per-source-vertex scalars.

        Source values (10 242 per hemisphere) are spread to all 163 842
        surface vertices via nearest-neighbour interpolation.

        Parameters
        ----------
        lh_scalars : ndarray, shape (10242,)
            Activity values for left-hemisphere source vertices.
        rh_scalars : ndarray, shape (10242,)
            Activity values for right-hemisphere source vertices.
        mode : str, default "power"
            Kept for API symmetry; unused internally.
        deferred : bool, default False
            If ``True``, skip the immediate ``render()`` call.
        """
        n_lh = self._n_lh
        self._scalars_full[:n_lh] = lh_scalars[self._nn_map["lh"]]
        self._scalars_full[n_lh:] = rh_scalars[self._nn_map["rh"]]
        self._refresh_scalars(deferred=deferred)

    def update(
        self,
        stc,
        mode: str = "power",
        interval: float = 0.05,
    ) -> None:
        """Animate the brain from a source time-course estimate.

        Parameters
        ----------
        stc : mne.SourceEstimate
            Source estimate returned by ``apply_inverse_raw`` or similar.
        mode : {"power", "activation"}, default "power"
            ``"power"`` displays mean squared amplitude;
            ``"activation"`` displays the time-averaged amplitude.
        interval : float, default 0.05
            Seconds to pause between frame updates.
        """
        n_times = stc.lh_data.shape[1]
        block = max(n_times // 2, 1)
        n_lh = self._n_lh

        for b_start in range(0, n_times, block):
            b_end = min(b_start + block, n_times)
            for hemi in ("lh", "rh"):
                raw_d = stc.rh_data if hemi == "rh" else stc.lh_data
                chunk = raw_d[:, b_start:b_end]
                src_vals = (
                    np.mean(chunk ** 2, axis=1)
                    if mode == "power"
                    else chunk.mean(axis=1)
                )
                nn = self._nn_map[hemi]
                if hemi == "lh":
                    self._scalars_full[:n_lh] = src_vals[nn]
                else:
                    self._scalars_full[n_lh:] = src_vals[nn]
            self._refresh_scalars()
            time.sleep(interval)

    def set_display_mode(self, mode: str) -> None:
        """Switch the display mode label and reset clim to sensible defaults.

        The display mode is a **label only** — it tells both the operator and
        the visualisation what kind of values are being streamed in.  Configure
        :meth:`~ant.NFRealtime.record_main` with the matching ``modality`` to
        pass the correct data.

        Parameters
        ----------
        mode : str
            One of :data:`_DISPLAY_MODES`.  Choosing a band-power mode will
            also set the colour-range (clim) to a typical power scale;
            choosing "Source Activation" restores the activation scale.

        Notes
        -----
        For **band power** modes pass per-source-vertex spectral power (e.g.,
        from ``modality="source_power"``) to :meth:`update_from_arrays`.
        For **activation** mode pass eLORETA / dSPM amplitude values from
        :meth:`update` (``stc`` from MNE inverse operator).
        """
        if mode not in _DISPLAY_MODES:
            raise ValueError(f"mode {mode!r} not recognised.  Choose from {_DISPLAY_MODES}.")
        self._display_mode = mode
        clim_hint = _DISPLAY_CLIM_HINTS.get(mode, (0.0, 0.6))
        self._clim[0] = clim_hint[0]
        self._clim[1] = clim_hint[1]
        self._act_actor.GetMapper().SetScalarRange(*self._clim)
        label = mode.split("(")[0].strip()
        self._plotter.scalar_bar.SetTitle(label)
        self._plotter.render()
        logger.info("BrainPlot display mode → %r  (clim %s)", mode, clim_hint)

    @property
    def display_mode(self) -> str:
        """Current display mode string (e.g. ``"Alpha Power  (8–13 Hz)"``).

        Read this in the acquisition loop to decide which modality to compute
        and pass to :meth:`update_from_arrays`.
        """
        return self._display_mode

    def reset_activity(self) -> None:
        """Zero out all activity scalars and refresh the display."""
        self._scalars_full[:] = 0.0
        self._refresh_scalars()

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def _toggle_recording(self) -> None:
        if self._recording:
            self.stop_recording()
        else:
            self.record_video()

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
            pass

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
