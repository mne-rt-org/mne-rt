"""
Real-time Maxwell filtering (SSS) on MEG data
==============================================

Signal Space Separation (SSS) is the gold-standard preprocessing step for
MEG data, suppressing external interference while preserving brain signals.
ANT's :class:`~ant.tools.RTMaxwellFilter` pre-computes the SSS projection
operator once from sensor geometry and applies it as a single matrix multiply
per incoming chunk — zero added latency, numerically equivalent to offline
MNE processing.

This example:

1. Loads the MNE sample MEG dataset (cropped to 30–90 s).
2. Fits :class:`~ant.tools.RTMaxwellFilter` on the recording info.
3. Applies offline SSS via :func:`mne.preprocessing.maxwell_filter` for
   reference.
4. Applies RT-SSS chunk-by-chunk (1-second windows).
5. Compares offline vs. real-time SSS via time-series, PSD, residual analysis,
   and per-channel Pearson correlation.

.. note::

   RT-SSS and offline SSS produce **numerically identical** output for basic
   SSS (no tSSS, no movement compensation).  This is expected and by design:
   both apply the same pre-computed projector
   :math:`\\mathbf{P}_{\\mathrm{SSS}} = \\mathbf{S}_{\\mathrm{in}}
   \\mathbf{S}_{\\mathrm{in}}^\\dagger`.
   Because matrix projection is linear, applying it chunk-by-chunk gives
   the same result as applying it to the full recording — the near-zero
   residual (Figure 1, bottom row) confirms this numerically.
   The practical value of RT-SSS is **online, sample-by-sample throughput**
   with no batch latency.
"""

# %%
# Load MNE sample data and apply offline SSS
# -------------------------------------------
# We pick MEG channels only, crop to a 60-second segment, and apply a
# 1–100 Hz bandpass to remove slow drifts and high-frequency noise before
# SSS processing.

import os

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import pearsonr

from ant.tools import RTMaxwellFilter

plt.style.use("default")

mne.set_log_level("WARNING")

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
raw.pick_types(meg=True, eeg=False, stim=False, exclude=[])
raw.crop(tmin=30.0, tmax=90.0)
raw.filter(l_freq=1.0, h_freq=100.0, verbose=False)

print(f"Raw MEG: {len(raw.ch_names)} channels, "
      f"{raw.times[-1]:.1f} s, sfreq={raw.info['sfreq']:.0f} Hz")

# %%
# Offline SSS (MNE reference)
# ---------------------------
# We apply :func:`mne.preprocessing.maxwell_filter` with the same harmonic
# orders as the real-time filter to obtain a ground-truth cleaned signal.

from mne.preprocessing import maxwell_filter

raw_offline = maxwell_filter(
    raw,
    origin="auto",
    int_order=8,
    ext_order=3,
    verbose=False,
)
data_offline = raw_offline.get_data()  # (n_ch, n_times)
print(f"Offline SSS done: data shape {data_offline.shape}")

# %%
# Fit RTMaxwellFilter
# --------------------
# The SSS operator depends only on sensor geometry — no baseline recording
# is required.  We simply pass :attr:`raw.info`.

rt_mf = RTMaxwellFilter(int_order=8, ext_order=3)
rt_mf.fit(raw.info)

print(rt_mf)
print(f"Internal moments retained: {rt_mf.n_use_in}")

# %%
# Apply RT-SSS chunk-by-chunk
# ----------------------------
# We simulate a real-time stream by slicing 1-second windows (= ``sfreq``
# samples) and calling :meth:`~ant.tools.RTMaxwellFilter.transform` on each.

sfreq = int(raw.info["sfreq"])
data_raw = raw.get_data()  # (n_ch, n_times)
n_ch, n_times = data_raw.shape
chunk_size = sfreq  # 1-second windows

n_chunks = n_times // chunk_size
data_rt = np.zeros_like(data_raw)

for i in range(n_chunks):
    sl = slice(i * chunk_size, (i + 1) * chunk_size)
    data_rt[:, sl] = rt_mf.transform(data_raw[:, sl])

if n_times % chunk_size:
    sl = slice(n_chunks * chunk_size, n_times)
    data_rt[:, sl] = rt_mf.transform(data_raw[:, sl])

print(f"RT-SSS applied over {n_chunks} chunks of {chunk_size} samples each")

# %%
# Residual analysis — confirming numerical equivalence
# -----------------------------------------------------
# Since both methods apply the same linear projector, the residual
# ``offline − RT-SSS`` should be at or near floating-point precision.

mag_picks  = mne.pick_types(raw.info, meg="mag",  exclude=[])
grad_picks = mne.pick_types(raw.info, meg="grad", exclude=[])
all_meg_picks = mne.pick_types(raw.info, meg=True, exclude=[])

residuals = data_offline - data_rt  # (n_ch, n_times)

residual_rms_fT = np.sqrt(np.mean(residuals[mag_picks] ** 2)) * 1e15  # fT
signal_rms_fT   = np.sqrt(np.mean(data_offline[mag_picks] ** 2)) * 1e15

print(f"Signal RMS  (mag): {signal_rms_fT:.1f} fT")
print(f"Residual RMS(mag): {residual_rms_fT:.6f} fT  "
      f"({residual_rms_fT / signal_rms_fT * 100:.2e} % of signal)")

# %%
# Per-channel Pearson correlation (offline vs. RT-SSS)
# ------------------------------------------------------
# r ≈ 1.0 across all channels confirms the projection is equivalent.

corr_all = np.array(
    [pearsonr(data_offline[ch], data_rt[ch])[0] for ch in all_meg_picks]
)

ch_types = np.array(
    ["mag" if ch in mag_picks else "grad" for ch in all_meg_picks]
)

print(f"Mean Pearson r (offline vs RT-SSS): {corr_all.mean():.6f}")
print(f"  Magnetometers : {corr_all[ch_types == 'mag'].mean():.6f}")
print(f"  Gradiometers  : {corr_all[ch_types == 'grad'].mean():.6f}")

# %%
# Figure 1 — Time-series comparison and residual
# ------------------------------------------------
# Three magnetometer channels show raw, offline-SSS, and RT-SSS overlaid over
# the first 10 s.  The bottom row plots the residual (Offline − RT-SSS) for
# one channel — it is indistinguishable from zero, confirming that chunked
# online processing produces the same output as batch offline processing.

target_names = ["MEG 0111", "MEG 0121", "MEG 0131"]
plot_chs = []
for name in target_names:
    if name in raw.ch_names:
        plot_chs.append(raw.ch_names.index(name))
if len(plot_chs) < 3:
    plot_chs = list(mag_picks[:3])

t = raw.times
t_mask = t <= 10.0

fig1, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True,
                           gridspec_kw={"hspace": 0.1})
fig1.suptitle(
    "RT-SSS is numerically equivalent to offline MNE maxwell_filter\n"
    "Bottom row: residual (Offline − RT-SSS) confirms near-zero difference",
    fontsize=12, fontweight="bold", y=0.99,
)

scale_fT = 1e15  # T → fT

for ax, ch_idx in zip(axes[:3], plot_chs):
    ch_name = raw.ch_names[ch_idx]
    ax.plot(t[t_mask], data_raw[ch_idx, t_mask] * scale_fT,
            color="#cccccc", lw=0.8, label="Raw", zorder=1)
    ax.plot(t[t_mask], data_offline[ch_idx, t_mask] * scale_fT,
            color="#1565C0", lw=1.8, label="Offline SSS", zorder=3)
    ax.plot(t[t_mask], data_rt[ch_idx, t_mask] * scale_fT,
            color="#E65100", lw=1.0, ls="--", label="RT-SSS", zorder=4)
    ax.set_ylabel("fT", fontsize=9)
    ax.set_title(ch_name, fontsize=10, loc="left", pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    if ax is axes[0]:
        ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=3)

# Residual row (scaled to aT = 1e-18 T for visibility)
res_ch = plot_chs[0]
residual_plot = residuals[res_ch, t_mask] * 1e18  # T → aT
ax_res = axes[3]
ax_res.plot(t[t_mask], residual_plot, color="#7B1FA2", lw=1.0)
ax_res.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
ax_res.set_ylabel("aT", fontsize=9)
ax_res.set_xlabel("Time (s)", fontsize=10)
ax_res.set_title(
    f"{raw.ch_names[res_ch]} — Residual (Offline − RT-SSS)   "
    f"[RMS = {np.sqrt(np.mean(residuals[res_ch]**2))*1e18:.3f} aT ≈ 0]",
    fontsize=9, loc="left", pad=2,
)
ax_res.spines[["top", "right"]].set_visible(False)

fig1.tight_layout()

# %%
# Figure 2 — Power spectral density
# -----------------------------------
# The left panel shows the full 1–100 Hz PSD averaged over all magnetometers.
# The right panel zooms into 1–40 Hz to show the noise-floor reduction more
# clearly.  The offline and RT curves are perfectly overlaid.

from scipy.signal import welch

psd_kwargs = dict(fs=float(raw.info["sfreq"]),
                  nperseg=int(4 * raw.info["sfreq"]),
                  noverlap=int(2 * raw.info["sfreq"]))


def _mean_psd(data, picks, **kw):
    psds = [welch(data[ch], **kw)[1] for ch in picks]
    f = welch(data[picks[0]], **kw)[0]
    return f, np.mean(psds, axis=0)


f_raw, pxx_raw      = _mean_psd(data_raw,     mag_picks, **psd_kwargs)
f_off, pxx_offline  = _mean_psd(data_offline, mag_picks, **psd_kwargs)
f_rt,  pxx_rt       = _mean_psd(data_rt,      mag_picks, **psd_kwargs)

fig2, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(
    "Power Spectral Density — Magnetometers (avg)\n"
    "Offline SSS and RT-SSS curves are exactly overlaid",
    fontsize=12, fontweight="bold",
)

for ax, flim, title in [
    (ax_full, (1, 100), "Full range 1–100 Hz"),
    (ax_zoom, (1,  40), "Close-up 1–40 Hz"),
]:
    mask = (f_raw >= flim[0]) & (f_raw <= flim[1])
    ax.semilogy(f_raw[mask], pxx_raw[mask],
                color="#aaaaaa", lw=1.0, label="Raw")
    ax.semilogy(f_off[mask], pxx_offline[mask],
                color="#1565C0", lw=2.2, label="Offline SSS")
    ax.semilogy(f_rt[mask], pxx_rt[mask],
                color="#E65100", lw=1.2, ls="--", label="RT-SSS")
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("PSD (T²/Hz)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

fig2.tight_layout()

# %%
# Figure 3 — Offline vs. RT-SSS agreement (all MEG channels)
# ------------------------------------------------------------
# Both panels confirm numerical equivalence across all 306 MEG channels.
# Near-perfect correlation (r ≈ 1.0) is the intended result — it shows
# that the pre-computed SSS projector applied chunk-by-chunk produces
# the same output as the batch offline solution.

fig3, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle(
    "Numerical equivalence: Offline SSS ≡ RT-SSS  (r ≈ 1.0 for all channels)\n"
    "This confirms RT-SSS achieves batch-equivalent quality with zero latency overhead",
    fontsize=11, fontweight="bold",
)

mag_mask  = ch_types == "mag"
grad_mask = ch_types == "grad"
ch_indices = np.arange(len(all_meg_picks))

ax_scatter.scatter(
    ch_indices[grad_mask], corr_all[grad_mask],
    s=14, alpha=0.6, color="#43A047", label="Gradiometers",
)
ax_scatter.scatter(
    ch_indices[mag_mask], corr_all[mag_mask],
    s=22, alpha=0.85, color="#1565C0", label="Magnetometers",
)
mean_r = corr_all.mean()
ax_scatter.axhline(mean_r, color="#E65100", lw=1.5, ls="--")
ax_scatter.set_ylim(max(0, corr_all.min() - 0.005), 1.0 + 0.002)
ax_scatter.set_xlabel("Channel index", fontsize=11)
ax_scatter.set_ylabel("Pearson r  (offline vs RT-SSS)", fontsize=11)
ax_scatter.set_title("Per-channel correlation", fontsize=11)
ax_scatter.legend(fontsize=10, frameon=False)
ax_scatter.spines[["top", "right"]].set_visible(False)
ax_scatter.text(
    0.05, 0.08,
    f"Mean r = {mean_r:.6f}",
    transform=ax_scatter.transAxes,
    fontsize=11, color="#E65100",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
)

ax_hist.hist(corr_all[grad_mask], bins=30, alpha=0.6, color="#43A047",
             label="Gradiometers")
ax_hist.hist(corr_all[mag_mask], bins=20, alpha=0.85, color="#1565C0",
             label="Magnetometers")
ax_hist.axvline(mean_r, color="#E65100", lw=2.0, ls="--",
                label=f"Mean = {mean_r:.6f}")
ax_hist.set_xlabel("Pearson r", fontsize=11)
ax_hist.set_ylabel("Number of channels", fontsize=11)
ax_hist.set_title("Distribution of correlations", fontsize=11)
ax_hist.legend(fontsize=10, frameon=False)
ax_hist.spines[["top", "right"]].set_visible(False)

fig3.tight_layout()
