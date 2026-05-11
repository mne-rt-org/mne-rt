"""
Multi-session NF trend analysis from saved JSON files
======================================================

Neurofeedback training is almost always delivered across multiple sessions
spread over days or weeks.  ANT saves each session's feature time-series to a
``*_nf.json`` file inside ``<subjects_dir>/<subject_id>/neurofeedback/``.

:meth:`~ant.NFRealtime.load_nf_data` reads one of these files and returns a
plain Python dict with two keys:

* ``"meta"`` — subject ID, visit number, session type, modalities, sampling
  frequency, window size, duration, artifact correction, and ISO-8601
  start/end timestamps.
* ``"data"`` — ``{modality: [values, …]}`` mapping each modality to its
  per-window scalar time-series.

This example:

1. Generates synthetic NF files for six sessions that simulate a gradual
   increase in alpha power (a classic neurofeedback learning trajectory).
2. Loads all files with ``NFRealtime.load_nf_data()`` and extracts per-session
   statistics.
3. Plots a session-by-session mean ± std bar chart with a linear trend line.
4. Plots within-session NF traces for all sessions overlaid on one axes,
   with a colour progression (light → dark) to show learning order.
5. Prints a formatted per-session summary table.
"""

# %%
# Generate synthetic NF data files
# ----------------------------------
# Six JSON files are written to a temporary directory; the payload format
# matches exactly what ``NFRealtime.save()`` produces.

import json
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

from ant import NFRealtime

RNG = np.random.default_rng(1)
N_SESSIONS = 6
N_WINDOWS = 120           # 120 one-second windows = 2-minute session

tmp_dir = Path(tempfile.mkdtemp(prefix="ant_multises_"))

session_files = []
for ses_idx in range(N_SESSIONS):
    # Simulate a gradual increase in alpha power across sessions (NF learning).
    trend = 0.05 * ses_idx
    alpha_vals = (RNG.standard_normal(N_WINDOWS) * 0.3 + trend).tolist()

    payload = {
        "meta": {
            "subject_id": "sub01",
            "visit": ses_idx + 1,
            "session": "main",
            "data_type": "eeg",
            "modalities": ["sensor_power"],
            "sfreq_hz": 256.0,
            "winsize_s": 1.0,
            "duration_s": float(N_WINDOWS),
            "n_windows": {"sensor_power": N_WINDOWS},
            "artifact_correction": "False",
            "start_time": f"2024-03-{ses_idx + 1:02d}T10:00:00+00:00",
            "end_time": (
                f"2024-03-{ses_idx + 1:02d}"
                f"T10:{N_WINDOWS // 60:02d}:00+00:00"
            ),
        },
        "data": {"sensor_power": alpha_vals},
    }

    fpath = tmp_dir / f"sub-sub01_vis-{ses_idx + 1:02d}_ses-main_nf.json"
    with open(fpath, "w") as f:
        json.dump(payload, f)
    session_files.append(fpath)

print(f"Wrote {len(session_files)} synthetic session files to {tmp_dir}")

# To load real session files instead, replace the two lines below with:
# session_files = sorted(
#     Path("subjects/sub01/neurofeedback").glob("*_nf.json")
# )

# %%
# Load sessions and extract per-session statistics
# --------------------------------------------------
# We use ``NFRealtime.load_nf_data()`` to read each file.  Metadata and
# feature values are extracted in the same loop so they stay aligned.

session_stats = []  # list of dicts, one per session

for fpath in session_files:
    payload = NFRealtime.load_nf_data(fpath)

    meta = payload["meta"]
    vals = np.asarray(payload["data"]["sensor_power"], dtype=float)

    session_stats.append(
        {
            "visit": meta["visit"],
            "start_time": meta["start_time"],
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "n_windows": int(meta["n_windows"]["sensor_power"]),
            "trace": vals,
        }
    )

visits = np.array([s["visit"] for s in session_stats])
means = np.array([s["mean"] for s in session_stats])
stds = np.array([s["std"] for s in session_stats])

print(f"Loaded {len(session_stats)} sessions  |  visits: {visits.tolist()}")

# %%
# Plot 1 — Session-by-session mean ± std with linear trend
# ---------------------------------------------------------
# Each bar shows the mean NF value for one session; error bars span ±1 SD.
# A linear regression line (scipy.stats.linregress) quantifies the learning
# trend, and its slope and p-value are annotated on the axes.

slope, intercept, r_value, p_value, _ = stats.linregress(visits, means)
trend_y = slope * visits + intercept

fig1, ax1 = plt.subplots(figsize=(7, 4))

ax1.bar(
    visits,
    means,
    yerr=stds,
    color="#2976AE",
    edgecolor="white",
    linewidth=0.8,
    error_kw=dict(ecolor="#555555", capsize=5, linewidth=1.4),
    width=0.6,
    label="Mean ± SD",
    zorder=3,
)

ax1.plot(
    visits,
    trend_y,
    color="#E07B54",
    linewidth=2.0,
    linestyle="--",
    zorder=4,
    label=f"Trend  (slope={slope:+.3f}, p={p_value:.3f})",
)

# Significance marker on the trend line annotation
sig_str = "n.s."
if p_value < 0.001:
    sig_str = "***"
elif p_value < 0.01:
    sig_str = "**"
elif p_value < 0.05:
    sig_str = "*"

ax1.text(
    visits[-1],
    trend_y[-1] + stds[-1] * 0.3,
    sig_str,
    fontsize=12,
    color="#E07B54",
    ha="center",
)

ax1.set_xticks(visits)
ax1.set_xticklabels([f"Ses {v}" for v in visits], fontsize=9)
ax1.set_xlabel("Session (visit number)", fontsize=10)
ax1.set_ylabel("Alpha power (a.u.)", fontsize=10)
ax1.set_title(
    "Session-by-session NF trend — sub01",
    fontsize=11,
    fontweight="bold",
)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(fontsize=9, framealpha=0.7)
ax1.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")

fig1.tight_layout()
plt.show()

# %%
# Plot 2 — Within-session NF traces (all sessions overlaid)
# ----------------------------------------------------------
# Each session's time-series is drawn as a thin, semi-transparent line.
# A colour progression from light to dark encodes session order so that
# later (more experienced) sessions stand out visually.
# The thick black line is the mean trace averaged across all sessions.

# Build a colour array: one colour per session, ranging from light to dark
cmap = plt.get_cmap("Blues")
# Use the range 0.35 – 0.95 to keep colours readable on a white background
session_colors = [cmap(0.35 + 0.6 * i / (N_SESSIONS - 1)) for i in range(N_SESSIONS)]

fig2, ax2 = plt.subplots(figsize=(9, 4))

trace_matrix = np.vstack([s["trace"] for s in session_stats])  # (N_SESSIONS, N_WINDOWS)
t = np.arange(N_WINDOWS)  # time axis in windows (= seconds here)

for i, (s, color) in enumerate(zip(session_stats, session_colors)):
    ax2.plot(
        t,
        s["trace"],
        color=color,
        linewidth=1.0,
        alpha=0.45,
        label=f"Ses {s['visit']}" if i == 0 or i == N_SESSIONS - 1 else "_nolegend_",
    )

# Grand mean across sessions
grand_mean = trace_matrix.mean(axis=0)
ax2.plot(
    t,
    grand_mean,
    color="#1a1a2e",
    linewidth=2.2,
    label="Grand mean",
    zorder=5,
)

# Add a colour-bar-style legend patch showing session order
sm = plt.cm.ScalarMappable(
    cmap=plt.get_cmap("Blues"),
    norm=mcolors.Normalize(vmin=1, vmax=N_SESSIONS),
)
sm.set_array([])
cbar = fig2.colorbar(sm, ax=ax2, orientation="vertical", pad=0.02, aspect=30)
cbar.set_label("Session number", fontsize=9)
cbar.set_ticks([1, N_SESSIONS])
cbar.set_ticklabels(["1 (early)", f"{N_SESSIONS} (late)"])

ax2.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")
ax2.set_xlabel("Analysis window (1 s each)", fontsize=10)
ax2.set_ylabel("Alpha power (a.u.)", fontsize=10)
ax2.set_title(
    "Within-session NF traces — all sessions (sub01)",
    fontsize=11,
    fontweight="bold",
)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(color="#dddddd", linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(
    handles=[
        plt.Line2D([0], [0], color=session_colors[0], lw=1.5, label="Session 1"),
        plt.Line2D(
            [0], [0], color=session_colors[-1], lw=1.5,
            label=f"Session {N_SESSIONS}",
        ),
        plt.Line2D([0], [0], color="#1a1a2e", lw=2.2, label="Grand mean"),
    ],
    fontsize=9,
    framealpha=0.7,
)

fig2.tight_layout()
plt.show()

# %%
# Summary table
# -------------
# Per-session statistics together with the metadata fields most useful for
# quick quality checks (start time, number of windows recorded).

COL_W = [8, 26, 10, 10, 10]
HEADER = ["Visit", "Start time", "Mean", "Std", "N windows"]
SEP = "  ".join("-" * w for w in COL_W)

print(SEP)
print("  ".join(h.ljust(w) for h, w in zip(HEADER, COL_W)))
print(SEP)

for s in session_stats:
    row = [
        str(s["visit"]),
        s["start_time"],
        f"{s['mean']:+.4f}",
        f"{s['std']:.4f}",
        str(s["n_windows"]),
    ]
    print("  ".join(v.ljust(w) for v, w in zip(row, COL_W)))

print(SEP)
overall_mean = np.mean([s["mean"] for s in session_stats])
print(
    f"  {'Grand mean'.ljust(COL_W[0])}  "
    f"{'(across sessions)'.ljust(COL_W[1])}  "
    f"{overall_mean:+.4f}"
)
print(SEP)
print(f"\nLinear trend: slope = {slope:+.4f} per session,  p = {p_value:.4f}")
