"""Real-time 3D brain activation display.

Built on PyVista with interactive sliders, hemisphere toggles, surface
switching, view presets, and screenshot support.

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

from ant._logging import logger
from ant.tools import setup_surface


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CMAPS = ["hot", "plasma", "viridis", "Reds", "YlOrRd", "RdBu_r"]

_SURFACES = ["inflated", "pial", "white", "sphere"]

_VIEW_PRESETS: dict[str, dict] = {
    "lateral_lh": {"position": (-450, 0, 0),  "focal": (0, 0, 0), "up": (0, 0, 1)},
    "lateral_rh": {"position": (450, 0, 0),   "focal": (0, 0, 0), "up": (0, 0, 1)},
    "dorsal":     {"position": (0, 0, 450),   "focal": (0, 0, 0), "up": (0, 1, 0)},
    "frontal":    {"position": (0, -450, 0),  "focal": (0, 0, 0), "up": (0, 0, 1)},
    "ventral":    {"position": (0, 0, -450),  "focal": (0, 0, 0), "up": (0, 1, 0)},
}

_KEY_HELP = (
    "View keys\n"
    "  1 → Left lateral\n"
    "  2 → Right lateral\n"
    "  3 → Dorsal\n"
    "  4 → Frontal\n"
    "  5 → Ventral\n"
    "  s → Screenshot\n"
    "  v → Toggle video rec\n"
    "  r → Reset activity\n"
    "  i/p/w/h → Surface\n"
    "     (inflated/pial/white/sphere)"
)


class BrainPlot:
    """Interactive real-time 3D brain activation display.

    Renders bilateral ``fsaverage5`` cortical surfaces with a colour-mapped
    activity overlay.  Designed to run alongside
    :class:`~ant.viz.NFSignalPlot` inside a shared Qt event loop — call
    :meth:`update_from_arrays` or :meth:`update` from the acquisition
    thread's pump timer.

    Parameters
    ----------
    subjects_fs_dir : str | Path
        FreeSurfer subjects directory (must contain the ``fsaverage5``
        subject folder).
    clim : tuple of float, default (0.0, 0.6)
        Initial ``(min, max)`` colour-map range for the activity overlay.
    hemi_distance : float, default 100.0
        Lateral separation in mm between left and right hemispheres.
    surf : {"inflated", "pial", "white", "sphere"}, default "inflated"
        Initial cortical surface geometry to display.
    cmap : str, default "hot"
        Initial colour map name.  Must be one of
        ``["hot", "plasma", "viridis", "Reds", "YlOrRd", "RdBu_r"]``.
    opacity : float, default 0.6
        Initial opacity of the activity overlay (0 = transparent, 1 = opaque).
    window_size : tuple of int, default (1600, 1000)
        Width × height of the PyVista render window in pixels.
    verbose : bool | str | None, default None
        Verbosity level.  ``None`` uses the current ANT log level.
        See :func:`~ant._logging.set_log_level` for accepted values.

    Attributes
    ----------
    plotter : pyvista.Plotter
        The underlying PyVista plotter instance.

    Raises
    ------
    ValueError
        If ``cmap`` or ``surf`` is not a recognised value.
    ValueError
        If ``subjects_fs_dir`` does not exist or does not contain
        ``fsaverage5``.

    See Also
    --------
    ant.viz.NFSignalPlot : Scrolling real-time NF signal plot.
    ant.NFRealtime.record_main : Main NF loop that drives both plots.

    Notes
    -----
    The window is opened immediately in :meth:`__init__` via
    ``plotter.show(interactive_update=True, auto_close=False)`` so that it
    participates in the Qt event loop from the start.

    Examples
    --------
    Minimal offline usage (no acquisition stream):

    >>> bp = BrainPlot("/path/to/freesurfer/subjects")
    >>> bp.update_from_arrays(lh_scalars, rh_scalars)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        subjects_fs_dir: Union[str, Path],
        clim: tuple[float, float] = (0.0, 0.6),
        hemi_distance: float = 100.0,
        surf: str = "inflated",
        cmap: str = "hot",
        opacity: float = 0.6,
        window_size: tuple[int, int] = (1600, 1000),
        verbose: Union[bool, str, None] = None,
    ) -> None:
        if not _pyvista_available:
            raise ImportError(
                "pyvista is required for BrainPlot. "
                "Install it with:  pip install 'ANT[viz]'"
            )
        from ant._logging import set_log_level
        set_log_level(verbose)

        if cmap not in _CMAPS:
            raise ValueError(
                f"cmap {cmap!r} not recognised.  Choose from {_CMAPS}."
            )
        if surf not in _SURFACES:
            raise ValueError(
                f"surf {surf!r} not recognised.  Choose from {_SURFACES}."
            )

        self._subjects_fs_dir = Path(subjects_fs_dir)
        self._clim = list(clim)
        self._hemi_distance = hemi_distance
        self._surf = surf
        self._opacity = opacity
        self._threshold = 0.0
        self._cmap_idx = _CMAPS.index(cmap)
        self._hemi_visible = {"lh": True, "rh": True}
        self._recording = False
        self._video_writer = None
        self._video_path: Path | None = None

        logger.info("Loading %s surface from %s …", surf, subjects_fs_dir)
        self._load_surface(surf)

        self._plotter = self._build_plotter(window_size)
        self._add_sliders()
        self._add_hemisphere_toggles()
        self._add_key_bindings()
        self._add_overlays()
        logger.info("BrainPlot ready.")

    # ------------------------------------------------------------------
    # Surface management
    # ------------------------------------------------------------------

    def _load_surface(self, surf: str) -> None:
        """Load mesh for *surf* and initialise scalar arrays."""
        (
            self._hemi_offsets,
            self._scalars_full,
            self._mesh,
            self._verts_stc,
        ) = setup_surface(
            str(self._subjects_fs_dir),
            hemi_distance=self._hemi_distance,
            surf=surf,
        )
        self._n_lh = int(self._hemi_offsets["rh"])

    # ------------------------------------------------------------------
    # Plotter construction
    # ------------------------------------------------------------------

    def _build_plotter(self, window_size: tuple[int, int]) -> pv.Plotter:
        p = pv.Plotter(
            window_size=list(window_size),
            lighting="three lights",
            title="Advanced Neurofeedback Toolbox — Brain Activation",
        )
        p.set_background("#060c18")

        self._base_actor = p.add_mesh(
            self._mesh,
            scalars="base",
            cmap="Greys",
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

        p.add_scalar_bar(
            title="Activity",
            italic=True,
            vertical=True,
            position_x=0.88,
            position_y=0.12,
            height=0.55,
            color="white",
            title_font_size=13,
            label_font_size=11,
        )
        p.enable_eye_dome_lighting()
        p.add_camera_orientation_widget()
        p.camera_position = "yz"
        p.camera.azimuth = 45
        p.show(interactive_update=True, auto_close=False)
        return p

    def _add_sliders(self) -> None:
        p = self._plotter

        def _on_threshold(val: float) -> None:
            self._threshold = val
            self._refresh_scalars()

        p.add_slider_widget(
            callback=_on_threshold,
            rng=[0.0, float(self._clim[1])],
            value=0.0,
            title="Threshold",
            pointa=(0.04, 0.90), pointb=(0.28, 0.90),
            style="modern", color="#4ECDC4", title_opacity=0.9,
            tube_width=0.003, slider_width=0.012,
        )

        def _on_opacity(val: float) -> None:
            self._opacity = val
            self._act_actor.GetProperty().SetOpacity(val)
            p.render()

        p.add_slider_widget(
            callback=_on_opacity,
            rng=[0.0, 1.0],
            value=self._opacity,
            title="Opacity",
            pointa=(0.32, 0.90), pointb=(0.56, 0.90),
            style="modern", color="#FFD93D", title_opacity=0.9,
            tube_width=0.003, slider_width=0.012,
        )

        def _on_cmap(val: float) -> None:
            idx = max(0, min(int(round(val)), len(_CMAPS) - 1))
            if idx == self._cmap_idx:
                return
            self._cmap_idx = idx
            self._sync_scalar_bar_lut()
            self._cmap_label.SetInput(f"Colormap: {_CMAPS[idx]}")
            p.render()

        p.add_slider_widget(
            callback=_on_cmap,
            rng=[0, len(_CMAPS) - 1],
            value=float(self._cmap_idx),
            title="Colormap  (0–5)",
            pointa=(0.60, 0.90), pointb=(0.84, 0.90),
            style="modern", color="#CC5DE8", title_opacity=0.9,
            fmt="%.0f",
            tube_width=0.003, slider_width=0.012,
        )
        self._cmap_label = p.add_text(
            f"Colormap: {_CMAPS[self._cmap_idx]}",
            position=(10, 48),
            font_size=9,
            color="#CC5DE8",
        )

        # Surface label (updated by set_surface)
        self._surf_label = p.add_text(
            f"Surface: {self._surf}",
            position=(10, 28),
            font_size=9,
            color="#4ECDC4",
        )

    def _add_hemisphere_toggles(self) -> None:
        p = self._plotter

        def _toggle_lh(state: bool) -> None:
            self._hemi_visible["lh"] = state
            self._refresh_scalars()

        def _toggle_rh(state: bool) -> None:
            self._hemi_visible["rh"] = state
            self._refresh_scalars()

        p.add_checkbox_button_widget(
            callback=_toggle_lh, value=True, position=(12, 12),
            size=24, border_size=3, color_on="#4ECDC4", color_off="#1a2744",
        )
        p.add_text("LH", position=(42, 16), font_size=9, color="#4ECDC4")

        p.add_checkbox_button_widget(
            callback=_toggle_rh, value=True, position=(82, 12),
            size=24, border_size=3, color_on="#FF6B6B", color_off="#1a2744",
        )
        p.add_text("RH", position=(112, 16), font_size=9, color="#FF6B6B")

    def _add_key_bindings(self) -> None:
        p = self._plotter
        p.add_key_event("1", lambda: self._set_view("lateral_lh"))
        p.add_key_event("2", lambda: self._set_view("lateral_rh"))
        p.add_key_event("3", lambda: self._set_view("dorsal"))
        p.add_key_event("4", lambda: self._set_view("frontal"))
        p.add_key_event("5", lambda: self._set_view("ventral"))
        p.add_key_event("s", self.screenshot)
        p.add_key_event("v", self._toggle_recording)
        p.add_key_event("r", self.reset_activity)
        p.add_key_event("i", lambda: self.set_surface("inflated"))
        p.add_key_event("p", lambda: self.set_surface("pial"))
        p.add_key_event("w", lambda: self.set_surface("white"))
        p.add_key_event("h", lambda: self.set_surface("sphere"))

    def _add_overlays(self) -> None:
        self._plotter.add_text(
            _KEY_HELP,
            position="upper_right",
            font_size=8,
            color="#5a6a8a",
        )
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
            self._mesh, scalars="base", cmap="Greys",
            smooth_shading=True, show_scalar_bar=False,
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
        # Re-sync scalar bar LUT to the new actor's mapper
        self._sync_scalar_bar_lut()
        p.render()

    def _sync_scalar_bar_lut(self) -> None:
        """Point the scalar bar's LUT at the current activity actor's mapper."""
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

        Reloads the FreeSurfer surface file and rebuilds the PyVista actors
        in-place.  Current activity scalars are preserved.

        Parameters
        ----------
        surf : {"inflated", "pial", "white", "sphere"}
            Target surface geometry.

        Raises
        ------
        ValueError
            If *surf* is not one of the recognised surface names.

        Examples
        --------
        >>> bp = BrainPlot("/path/to/subjects")
        >>> bp.set_surface("pial")
        """
        if surf not in _SURFACES:
            raise ValueError(
                f"surf {surf!r} not recognised.  Choose from {_SURFACES}."
            )
        if surf == self._surf:
            return

        logger.info("Switching surface: %s → %s", self._surf, surf)
        saved_scalars = self._scalars_full.copy()
        self._surf = surf
        self._load_surface(surf)

        # Preserve activity values if vertex count matches
        if len(saved_scalars) == len(self._scalars_full):
            self._scalars_full[:] = saved_scalars
            self._mesh["activity"] = saved_scalars

        self._rebuild_actors()
        self._surf_label.SetInput(f"Surface: {self._surf}")
        logger.info("Surface changed to %r.", surf)

    def update_from_arrays(
        self,
        lh_scalars: np.ndarray,
        rh_scalars: np.ndarray,
        mode: str = "power",
        deferred: bool = False,
    ) -> None:
        """Update the brain display from pre-computed per-vertex scalars.

        Intended to be called from the Qt main thread by the acquisition
        pump timer (see :meth:`~ant.NFRealtime.record_main`).

        Parameters
        ----------
        lh_scalars : ndarray, shape (n_lh_verts,)
            Activity values for left-hemisphere source vertices.
        rh_scalars : ndarray, shape (n_rh_verts,)
            Activity values for right-hemisphere source vertices.
        mode : str, default "power"
            Kept for API symmetry; the calling code passes the correct
            values so this argument is not used internally.
        """
        self._scalars_full[
            self._verts_stc["lh"] + self._hemi_offsets["lh"]
        ] = lh_scalars
        self._scalars_full[
            self._verts_stc["rh"] + self._hemi_offsets["rh"]
        ] = rh_scalars
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

        for b_start in range(0, n_times, block):
            b_end = min(b_start + block, n_times)
            for hemi in ("lh", "rh"):
                raw_d = stc.rh_data if hemi == "rh" else stc.lh_data
                chunk = raw_d[:, b_start:b_end]
                scalars = (
                    np.mean(chunk ** 2, axis=1)
                    if mode == "power"
                    else chunk.mean(axis=1)
                )
                verts = self._verts_stc[hemi] + self._hemi_offsets[hemi]
                self._scalars_full[verts] = scalars
            self._refresh_scalars()
            time.sleep(interval)

    def reset_activity(self) -> None:
        """Zero out all activity scalars and refresh the display.

        Examples
        --------
        >>> bp.reset_activity()
        """
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

        Each subsequent call to :meth:`write_frame_if_recording` (issued
        automatically by :meth:`~ant.NFRealtime.record_main`'s brain pump
        timer) appends one frame.  Press **v** in the window or call
        :meth:`stop_recording` to finish.

        Parameters
        ----------
        path : str | Path | None, default None
            Destination file path.  Defaults to
            ``~/ant_brain_<timestamp>.mp4``.

        Returns
        -------
        path : Path
            Path of the video file being written.

        Raises
        ------
        RuntimeError
            If recording is already in progress.

        Examples
        --------
        >>> bp.record_video("session.mp4")
        >>> # ... run NF session ...
        >>> bp.stop_recording()
        """
        import imageio
        if self._recording:
            raise RuntimeError(
                "Already recording. Call stop_recording() first."
            )
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path.home() / f"ant_brain_{ts}.mp4"
        path = Path(path)
        self._video_writer = imageio.get_writer(
            str(path), fps=24, macro_block_size=None
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
            Path of the saved video file, or ``None`` if no recording
            was active.

        Examples
        --------
        >>> saved = bp.stop_recording()
        >>> print("Video saved to:", saved)
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
        """Capture a video frame if recording is active.

        Called automatically from
        :meth:`~ant.NFRealtime.record_main`'s brain pump timer after
        each :meth:`~pyvista.Plotter.render` call.  Safe to call even
        when not recording (no-op).
        """
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
            Path to the saved file.

        Examples
        --------
        >>> saved = bp.screenshot("~/my_view.png")
        """
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path.home() / f"ant_brain_{ts}.png"
        self._plotter.screenshot(str(path))
        logger.info("Screenshot saved: %s", path)
        return Path(path)

    @property
    def plotter(self) -> pv.Plotter:
        """The underlying :class:`pyvista.Plotter` instance."""
        return self._plotter

    @property
    def surf(self) -> str:
        """Currently displayed surface geometry name."""
        return self._surf
