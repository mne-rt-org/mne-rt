"""
Three-window real-time display
================================

ANT provides three simultaneous real-time visualisation windows during a
neurofeedback session:

1. **Raw signal** — live scrolling M/EEG channel traces (mne-lsl StreamViewer).
2. **NF signal** — scrolling plot of the extracted neurofeedback feature
   (:class:`~ant.viz.NFSignalPlot`).
3. **Brain activation** — interactive 3D cortical surface coloured by
   estimated source power (:class:`~ant.viz.BrainPlot`).

This example runs a short mock session headlessly and reproduces all three
displays as static matplotlib figures, illustrating what each window looks
like during a live session.
"""

# %%
# Run a headless mock session
# ----------------------------
# We record 20 s of simulated EEG and extract three sensor-space modalities.
# All GUI windows are disabled via ``show_*=False`` so the session runs
# non-interactively.

import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ant import NFRealtime

subjects_dir = tempfile.mkdtemp(prefix="ant_3win_")

nf = NFRealtime(
    subject_id="demo",
    visit=1,
    session="main",
    subjects_dir=subjects_dir,
    montage="easycap-M1",
    data_type="eeg",
    verbose=False,
)

nf.connect_to_lsl(mock_lsl=True, timeout=30.0, verbose=False)

nf.record_main(
    duration=20,
    modality=["sensor_power", "band_ratio", "entropy"],
    winsize=1.0,
    show_raw_signal=False,
    show_nf_signal=False,
    show_brain_activation=False,
    verbose=False,
)

print("Session complete.")
print("Modalities:", list(nf.nf_data.keys()))
print("Samples per modality:", {k: len(v) for k, v in nf.nf_data.items()})

# %%
# Window 1 — Raw EEG signal
# --------------------------
# During a live session the StreamViewer scrolls through all EEG channels in
# real time.  We reconstruct the same view by loading the source file directly.

import mne
import ant as _ant_pkg

_data_dir = Path(_ant_pkg.__file__).parent.parent.parent / "data" / "sample"
raw = mne.io.read_raw_brainvision(
    str(_data_dir / "sample_data.vhdr"), preload=True, verbose=False
)

# Drop non-EEG channels and set montage
raw.drop_channels([c for c in raw.ch_names if c in ("HRli", "HRre")], on_missing="ignore")
raw.rename_channels({"FPz": "Fpz"}, verbose=False)
raw.set_montage("easycap-M1", on_missing="warn", verbose=False)

# Grab a 5-second snippet and 8 channels for display
tmin, tmax = 5.0, 10.0
ch_picks = raw.ch_names[:8]
data, times = raw.get_data(picks=ch_picks, tmin=tmin, tmax=tmax, return_times=True)

fig1, axes = plt.subplots(len(ch_picks), 1, figsize=(10, 7), sharex=True)
fig1.suptitle("Window 1 — Raw EEG signal (StreamViewer)", fontweight="bold", fontsize=12)

for i, (ax, ch, row) in enumerate(zip(axes, ch_picks, data)):
    ax.plot(times, row * 1e6, lw=0.6, color="#2E86AB")
    ax.set_yticks([])
    ax.set_ylabel(ch, fontsize=7, rotation=0, labelpad=35, va="center")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    if i < len(ch_picks) - 1:
        ax.axhline(0, lw=0.3, color="lightgrey")

axes[-1].set_xlabel("Time (s)")
fig1.text(0.005, 0.5, "EEG channels", va="center", rotation="vertical", fontsize=9)
fig1.tight_layout()
plt.show()

# %%
# Window 2 — NF signal monitor
# -----------------------------
# The :class:`~ant.viz.NFSignalPlot` shows the extracted feature value(s)
# as a scrolling time-series.  We sub-sample the stored values to one per
# second for a clean representation.

PALETTE = ["#2E86AB", "#E84855", "#4CAF50"]
LABELS = {
    "sensor_power": "Alpha power",
    "band_ratio":   "θ/β ratio",
    "entropy":      "Spectral entropy",
}

mods = list(nf.nf_data.keys())

fig2, axes2 = plt.subplots(len(mods), 1, figsize=(10, 5), sharex=True)
fig2.suptitle("Window 2 — NF signal monitor (NFSignalPlot)", fontweight="bold", fontsize=12)

for ax, mod, color in zip(axes2, mods, PALETTE):
    raw_vals = np.asarray(nf.nf_data[mod], dtype=float)
    # Down-sample to ~1 sample/s for a representative view
    step = max(1, len(raw_vals) // 20)
    vals = raw_vals[::step]
    t = np.arange(len(vals))
    ax.plot(t, vals, lw=1.5, color=color, label=mod)
    ax.fill_between(t, vals, alpha=0.12, color=color)
    ax.axhline(np.mean(vals), lw=0.9, ls="--", color="grey", alpha=0.7, label="mean")
    ax.set_ylabel(LABELS.get(mod, mod), fontsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

axes2[-1].set_xlabel("Time (s)")
fig2.tight_layout()
plt.show()

# %%
# Window 3 — Brain activation (2-D topomap)
# ------------------------------------------
# In a live session the :class:`~ant.viz.BrainPlot` displays a 3D cortical
# surface coloured by source power.  We reproduce the spatial intuition with
# an EEG sensor-space topomap, computing alpha-band power per channel from
# the sample recording.

from scipy.signal import welch as _welch

raw_eeg = raw.copy().pick("eeg")
raw_alpha = raw_eeg.copy().filter(l_freq=8.0, h_freq=12.0, verbose=False)
ch_data = raw_alpha.get_data(tmin=5.0, tmax=25.0)

freqs, psd = _welch(ch_data, fs=raw_eeg.info["sfreq"], nperseg=256)
alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
alpha_power = psd[:, alpha_mask].mean(axis=-1)

fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))
fig3.suptitle("Window 3 — Brain activation (BrainPlot)", fontweight="bold", fontsize=12)

mne.viz.plot_topomap(
    alpha_power,
    raw_eeg.info,
    axes=ax3,
    show=False,
    cmap="RdBu_r",
    vlim=(np.percentile(alpha_power, 5), np.percentile(alpha_power, 95)),
)
ax3.set_title("Alpha power (8–12 Hz) per electrode", fontsize=10)
fig3.tight_layout()
plt.show()

# %%
# Summary
# --------
# All three windows provide complementary views of the same real-time signal:
#
# * **Window 1** lets the experimenter monitor raw signal quality and spot
#   artifacts while the session is running.
# * **Window 2** shows the closed-loop feedback signal the participant
#   responds to, scrolling in real time.
# * **Window 3** provides spatial context — which brain regions are driving
#   the feedback signal — updated every second from the 3D surface model.
#
# During a live session all three windows run in parallel, driven by a shared
# Qt event loop at ~30 fps, without blocking M/EEG acquisition.
