"""
NF signal monitor & brain activation display
=============================================

ANT provides two live visualisation windows during a closed-loop session:

* :class:`~ant.viz.NFSignalPlot` — scrolling multi-modality NF time-series
* :class:`~ant.viz.BrainPlot` — interactive 3D cortical activation surface

Because both windows are Qt-based they cannot be captured by Sphinx-Gallery
automatically.  Instead this example generates synthetic NF data, re-renders
each window as an animated GIF using matplotlib (NF signal) and PyVista
off-screen rendering (brain), saves them to ``docs/source/_static/``, and
embeds them below.
"""

# sphinx_gallery_thumbnail_path = '_static/nf_signal_thumb.png'

# %%
# Synthetic NF data
# ------------------
# We synthesise 30 s of realistic NF feature trajectories for four modalities.
# Each signal is a slow oscillation with additive noise — representative of
# what the live :class:`~ant.NFRealtime` loop produces at 1 sample/s.

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from pathlib import Path

rng = np.random.default_rng(42)

MODS   = ["sensor_power", "band_ratio", "entropy", "hjorth"]
COLORS = ["#5DA5A4", "#FF6B6B", "#FFD93D", "#6BCB77"]
LABELS = {
    "sensor_power": "Sensor Power",
    "band_ratio":   "Band Ratio",
    "entropy":      "Entropy",
    "hjorth":       "Hjorth",
}

N_SEC = 30   # seconds of NF data at 1 sample/s

t_nf = np.arange(N_SEC, dtype=float)

# Each modality: slow drift + alpha oscillation + noise
raw_signals: dict[str, np.ndarray] = {
    "sensor_power": (
        0.5 * np.sin(2 * np.pi * t_nf / 12)
        + 0.2 * np.sin(2 * np.pi * t_nf / 4)
        + 0.15 * rng.standard_normal(N_SEC)
    ),
    "band_ratio": (
        0.4 * np.cos(2 * np.pi * t_nf / 10)
        + 0.15 * rng.standard_normal(N_SEC)
    ),
    "entropy": (
        0.3 * np.sin(2 * np.pi * t_nf / 8 + 1)
        + 0.2 * rng.standard_normal(N_SEC)
    ),
    "hjorth": (
        0.5 * np.cos(2 * np.pi * t_nf / 15 + 0.5)
        + 0.25 * rng.standard_normal(N_SEC)
    ),
}

# Normalise to a common visual range
norm_signals: dict[str, np.ndarray] = {}
for m, v in raw_signals.items():
    span = v.max() - v.min()
    norm_signals[m] = (v - v.mean()) / (span + 1e-300)

print("Synthetic NF signals:")
for m, v in norm_signals.items():
    print(f"  {m:20s}  mean={v.mean():.3f}  std={v.std():.3f}")

# %%
# NF signal monitor (animated GIF)
# ----------------------------------
# We render a scrolling animation that matches the look of the live
# :class:`~ant.viz.NFSignalPlot` window, including the dark theme and the
# fine graph-paper grid.  The 30-fps interpolation between 1-s window
# estimates is also reproduced.

N_MODS      = len(MODS)
TIME_WINDOW = 10       # seconds visible in the scrolling window
FPS         = 30       # animation frame rate
n_pts       = TIME_WINDOW * FPS
t_axis      = np.linspace(0.0, TIME_WINDOW, n_pts)
buf         = np.zeros((N_MODS, n_pts))

# ── build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    N_MODS, 1, figsize=(10, N_MODS * 1.9),
    facecolor="#0d0d1a", sharex=True,
)
fig.subplots_adjust(hspace=0.04, left=0.13, right=0.97, top=0.95, bottom=0.06)
fig.text(0.5, 0.98, "ANT — NF Signal Monitor",
         ha="center", va="top", color="#c0c0d8", fontsize=11, fontweight="bold")

lines = []
for i, (ax, mod, color) in enumerate(zip(axes, MODS, COLORS)):
    ax.set_facecolor("#0d0d1a")
    ax.set_xlim(0, TIME_WINDOW)
    ax.set_ylim(-2.5, 2.5)
    ax.set_ylabel(LABELS[mod], fontsize=9, color=color, labelpad=6)
    ax.tick_params(colors="#9090aa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#303050")
    # Fine graph-paper grid: major every 2 s, minor every 0.5 s
    ax.set_xticks(np.arange(0, TIME_WINDOW + 0.01, 2))
    ax.set_xticks(np.arange(0, TIME_WINDOW + 0.01, 0.5), minor=True)
    ax.grid(True, which="both", color="#303050", linewidth=0.5, alpha=0.6)
    ax.axhline(0, color="#252545", lw=0.8, zorder=1)
    if i < N_MODS - 1:
        ax.tick_params(labelbottom=False)
    line, = ax.plot(t_axis, buf[i], color=color, lw=1.8, zorder=2)
    lines.append(line)

axes[-1].set_xlabel("Time (s)", color="#9090aa", fontsize=9)

# ── animation: 30-fps linear interpolation between 1-s NF estimates ───────────
n_nf     = N_SEC
n_frames = min(n_nf * FPS, 25 * FPS)   # cap at 25 s of animation


def _update(frame: int):
    nf_idx  = frame // FPS
    step    = frame %  FPS
    nf_prev = max(nf_idx - 1, 0)
    alpha   = step / FPS
    for i, mod in enumerate(MODS):
        v      = norm_signals[mod]
        interp = v[nf_prev] * (1 - alpha) + v[min(nf_idx, len(v) - 1)] * alpha
        buf[i] = np.roll(buf[i], -1)
        buf[i, -1] = interp
        lines[i].set_ydata(buf[i])
    return lines


anim = mpl_animation.FuncAnimation(
    fig, _update, frames=n_frames, interval=1000 / FPS, blit=True
)

# Resolve docs/_static relative to this examples/ directory.
# sphinx-gallery doesn't set __file__, so fall back to cwd (which
# sphinx-gallery sets to the examples/ directory during execution).
try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path.cwd()
_static_dir = _here.parent / "docs" / "source" / "_static"
_static_dir.mkdir(parents=True, exist_ok=True)

gif_path = _static_dir / "nf_signal_demo.gif"
anim.save(str(gif_path), writer=mpl_animation.PillowWriter(fps=FPS), dpi=90)
plt.close(fig)
print(f"NF signal GIF saved → {gif_path}")

# Save a static thumbnail (5 s into the animation) for the gallery card
buf[:] = 0.0
for f in range(FPS * 5):
    _update(f)

fig_thumb, axs_t = plt.subplots(
    N_MODS, 1, figsize=(8, N_MODS * 1.5), facecolor="#0d0d1a", sharex=True
)
fig_thumb.subplots_adjust(hspace=0.04, left=0.14, right=0.97, top=0.94, bottom=0.07)
for i, (ax, mod, color) in enumerate(zip(axs_t, MODS, COLORS)):
    ax.set_facecolor("#0d0d1a")
    ax.plot(t_axis, buf[i], color=color, lw=1.8)
    ax.set_xlim(0, TIME_WINDOW)
    ax.set_ylabel(LABELS[mod], fontsize=8, color=color)
    ax.tick_params(colors="#9090aa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#303050")
    ax.grid(True, which="both", color="#303050", lw=0.5, alpha=0.6)
    ax.axhline(0, color="#252545", lw=0.8)
    if i < N_MODS - 1:
        ax.tick_params(labelbottom=False)
axs_t[-1].set_xlabel("Time (s)", color="#9090aa", fontsize=9)
fig_thumb.savefig(
    str(_static_dir / "nf_signal_thumb.png"), dpi=120, facecolor="#0d0d1a"
)
plt.close(fig_thumb)

# %%
# The animation below shows the NF Signal Monitor during a live session.
# Each row corresponds to one NF modality, scrolling left at real-time speed.
# The fine grid updates automatically when the time window is changed via
# the control panel.
#
# .. raw:: html
#
#    <p style="text-align:center">
#      <img src="../_static/nf_signal_demo.gif"
#           alt="ANT NF Signal Monitor"
#           style="max-width:100%;border-radius:6px;background:#0d0d1a">
#    </p>

# %%
# Brain activation display (animated GIF)
# ----------------------------------------
# The :class:`~ant.viz.BrainPlot` renders the ``fsaverage5`` cortical surface
# with a source-power activity overlay, updated every ~200 ms during a live
# session.  The animation below is generated offline using PyVista's
# off-screen backend.  If PyVista or FreeSurfer is not found this section
# falls back to a sensor-space topomap.

import os


def _find_fs_dir() -> "Path | None":
    candidates = []
    for var in ("FREESURFER_HOME", "SUBJECTS_DIR"):
        val = os.environ.get(var)
        if val:
            p = Path(val) / "subjects" if var == "FREESURFER_HOME" else Path(val)
            candidates.append(p)
    candidates += [
        Path("/Applications/freesurfer/dev/subjects"),
        Path("/usr/local/freesurfer/subjects"),
    ]
    return next(
        (d for d in candidates if d.is_dir() and (d / "fsaverage5").is_dir()), None
    )


_fs_dir = _find_fs_dir()
_brain_ok = False

try:
    if _fs_dir is None:
        raise RuntimeError("FreeSurfer subjects dir not found.")

    import pyvista as pv
    from ant.tools import setup_surface

    hemi_offsets, scalars_full, mesh, verts_stc = setup_surface(
        str(_fs_dir), hemi_distance=100, surf="inflated"
    )
    n_verts = len(scalars_full)

    plotter = pv.Plotter(off_screen=True, window_size=[900, 560])
    plotter.set_background("#060c18")
    plotter.add_mesh(
        mesh, scalars="base", cmap="Greys",
        smooth_shading=True, show_scalar_bar=False,
    )
    plotter.add_mesh(
        mesh, scalars="activity", cmap="hot",
        clim=[0.0, 0.55], smooth_shading=True,
        show_scalar_bar=True, interpolate_before_map=True, nan_opacity=0.0,
    )
    plotter.camera_position = [(-450, -80, 180), (0, 0, 0), (0, 0, 1)]

    N_BRAIN_FRAMES = 60
    frames_brain = []
    for f in range(N_BRAIN_FRAMES):
        phase = f / N_BRAIN_FRAMES * 2 * np.pi
        act = np.clip(
            0.28 + 0.28 * np.sin(np.linspace(0, 3 * np.pi, n_verts) + phase), 0, 1
        )
        mesh["activity"] = act
        frames_brain.append(plotter.screenshot(return_img=True))
    plotter.close()

    from PIL import Image as _PILImage

    pil_frames = [_PILImage.fromarray(fr) for fr in frames_brain]
    brain_gif  = _static_dir / "brain_demo.gif"
    pil_frames[0].save(
        str(brain_gif), save_all=True, append_images=pil_frames[1:],
        loop=0, duration=int(1000 / 15), optimize=False,
    )
    print(f"Brain GIF saved → {brain_gif}")
    _brain_ok = True

except Exception as exc:
    print(f"Brain animation skipped: {exc}")

# %%
# .. raw:: html
#
#    <p id="ant-brain-gif" style="text-align:center">
#      <img src="../_static/brain_demo.gif"
#           alt="ANT Brain Activation Display"
#           style="max-width:100%;border-radius:6px;background:#060c18"
#           onerror="this.parentElement.style.display='none'">
#    </p>

if not _brain_ok:
    # Sensor-space topomap as a fallback illustration
    import mne
    import ant as _ant_pkg
    from scipy.signal import welch as _welch

    _data_dir = Path(_ant_pkg.__file__).parent.parent.parent / "data" / "sample"
    _raw = mne.io.read_raw_brainvision(
        str(_data_dir / "sample_data.vhdr"), preload=True, verbose=False
    )
    _raw.drop_channels(
        [c for c in _raw.ch_names if c in ("HRli", "HRre")], on_missing="ignore"
    )
    _raw.rename_channels({"FPz": "Fpz"}, verbose=False)
    _raw.set_montage("easycap-M1", on_missing="warn", verbose=False)
    _raw_eeg = _raw.copy().pick("eeg").filter(8, 12, verbose=False)
    _ch_data = _raw_eeg.get_data(tmin=5, tmax=25)
    _freqs, _psd = _welch(_ch_data, fs=_raw_eeg.info["sfreq"], nperseg=256)
    _alpha_pwr = _psd[:, (_freqs >= 8) & (_freqs <= 12)].mean(axis=-1)

    fig_topo, ax_topo = plt.subplots(figsize=(5, 5))
    mne.viz.plot_topomap(
        _alpha_pwr, _raw_eeg.info, axes=ax_topo, show=False, cmap="hot",
        vlim=(
            np.percentile(_alpha_pwr, 5),
            np.percentile(_alpha_pwr, 95),
        ),
    )
    ax_topo.set_title(
        "Alpha power per electrode\n(sensor-space approximation of BrainPlot)",
        fontsize=10,
    )
    plt.show()

# %%
# Summary
# --------
# During a live ANT session both windows run in parallel, driven by a shared
# Qt event loop at ~30 fps, without blocking EEG/MEG acquisition:
#
# * **NF signal monitor** — 30-fps scrolling display with fine graph-paper
#   grid; per-modality auto-range; linear interpolation between window
#   estimates for a smooth trace.
# * **Brain activation display** — PyVista 3D cortical surface updated at
#   ~5 fps; interactive threshold, opacity, and colour-map sliders; hemisphere
#   toggles; keyboard shortcuts for view presets.
