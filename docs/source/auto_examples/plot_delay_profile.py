"""
Pipeline delay profile from a saved session
============================================

After a real-time neurofeedback session that was run with
``record_main(estimate_delays=True)``, ANT writes a
``*_delays.json`` file to ``<subjects_dir>/<subject_id>/delays/``.

The JSON has three top-level keys:

* ``"acquisition"`` — wall-clock time for each ``stream.get_data()`` pull.
* ``"artifact_correction"`` — time spent inside the chosen artifact method
  (absent when ``artifact_correction=False``).
* ``"methods"`` — a sub-dict with one entry per modality, each holding the
  time the feature extractor took for one window.

Every entry is a summary dict with keys
``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``, ``p95_ms``, and ``n``.

This example:

1. Creates a synthetic delays dict that mirrors the real JSON structure.
2. Parses it into a flat list of records for easy plotting.
3. Renders a two-panel bar chart (infrastructure vs. feature extraction).
4. Prints a formatted summary table.
"""

# %%
# Generate synthetic delay data
# ------------------------------
# We build a dict that looks exactly like the ``*_delays.json`` produced by
# ``record_main(estimate_delays=True)``.  Swap this block for a
# ``json.load()`` call to analyse a real session.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RNG = np.random.default_rng(0)

synthetic_delays = {
    "acquisition": {
        "mean_ms": 8.2,
        "std_ms": 1.4,
        "min_ms": 5.1,
        "max_ms": 14.3,
        "p95_ms": 11.2,
        "n": 300,
    },
    "artifact_correction": {
        "mean_ms": 2.1,
        "std_ms": 0.3,
        "min_ms": 1.5,
        "max_ms": 3.8,
        "p95_ms": 2.9,
        "n": 300,
    },
    "methods": {
        "sensor_power": {
            "mean_ms": 0.12,
            "std_ms": 0.03,
            "min_ms": 0.08,
            "max_ms": 0.31,
            "p95_ms": 0.18,
            "n": 300,
        },
        "erd_ers": {
            "mean_ms": 0.18,
            "std_ms": 0.04,
            "min_ms": 0.11,
            "max_ms": 0.42,
            "p95_ms": 0.26,
            "n": 300,
        },
        "laterality": {
            "mean_ms": 0.09,
            "std_ms": 0.02,
            "min_ms": 0.06,
            "max_ms": 0.21,
            "p95_ms": 0.14,
            "n": 300,
        },
    },
}

# To load a real session file instead, uncomment:
# import json
# with open(
#     "subjects/sub01/delays/"
#     "sub-sub01_vis-01_ses-main_20240115T103045_delays.json"
# ) as f:
#     synthetic_delays = json.load(f)

delays = synthetic_delays

# %%
# Parse into a flat list of records
# -----------------------------------
# We flatten the nested dict so that each pipeline stage maps to a single row
# with ``stage``, ``mean_ms``, ``std_ms``, and ``p95_ms``.

records = []

if "acquisition" in delays:
    d = delays["acquisition"]
    records.append(
        {
            "stage": "acquisition",
            "group": "infrastructure",
            "mean_ms": d["mean_ms"],
            "std_ms": d["std_ms"],
            "p95_ms": d["p95_ms"],
        }
    )

if "artifact_correction" in delays:
    d = delays["artifact_correction"]
    records.append(
        {
            "stage": "artifact\ncorrection",
            "group": "infrastructure",
            "mean_ms": d["mean_ms"],
            "std_ms": d["std_ms"],
            "p95_ms": d["p95_ms"],
        }
    )

for meth_name, d in delays.get("methods", {}).items():
    records.append(
        {
            "stage": meth_name,
            "group": "features",
            "mean_ms": d["mean_ms"],
            "std_ms": d["std_ms"],
            "p95_ms": d["p95_ms"],
        }
    )

infra_recs = [r for r in records if r["group"] == "infrastructure"]
feat_recs = [r for r in records if r["group"] == "features"]

print(f"Infrastructure stages : {len(infra_recs)}")
print(f"Feature-extraction stages : {len(feat_recs)}")

# %%
# Plot: two-panel latency breakdown
# ----------------------------------
# The left panel shows the infrastructure steps (acquisition and artifact
# correction), which dominate the overall pipeline latency.  The right panel
# zooms in on the feature-extraction methods, whose per-window cost is
# typically an order of magnitude smaller.
#
# Diamonds mark the 95th-percentile latency on each bar.

INFRA_COLORS = ["#2976AE", "#E07B54"]   # steelblue family for infrastructure
FEAT_COLOR = "#3A9E5F"                   # seagreen family for feature methods
P95_MARKER_COLOR = "#1a1a2e"

fig, (ax_left, ax_right) = plt.subplots(
    1, 2, figsize=(11, 4.5), constrained_layout=True
)
fig.suptitle(
    "Real-time pipeline latency breakdown",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

# ---- helper to draw a horizontal bar panel ----

def _draw_panel(ax, recs, colors, title, xlabel="Latency (ms)"):
    labels = [r["stage"] for r in recs]
    means = [r["mean_ms"] for r in recs]
    stds = [r["std_ms"] for r in recs]
    p95s = [r["p95_ms"] for r in recs]
    y_pos = range(len(recs))

    if isinstance(colors, list):
        bar_colors = colors[: len(recs)]
    else:
        bar_colors = [colors] * len(recs)

    bars = ax.barh(
        y_pos,
        means,
        xerr=stds,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw=dict(ecolor="#555555", capsize=4, linewidth=1.2),
        height=0.55,
    )

    # P95 diamond markers
    for yp, p95 in zip(y_pos, p95s):
        ax.plot(
            p95,
            yp,
            marker="D",
            color=P95_MARKER_COLOR,
            markersize=6,
            zorder=5,
            label="P95" if yp == 0 else "_nolegend_",
        )

    # Text labels inside / beside bars
    x_max = max(means) + max(stds)
    for bar, r in zip(bars, recs):
        w = bar.get_width()
        label_x = w + max(stds) * 0.15
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2.0,
            f"{r['mean_ms']:.2f} ± {r['std_ms']:.2f} ms",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333",
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=8)
    ax.set_xlim(0, x_max * 1.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(axis="x", which="major", color="#dddddd", linewidth=0.7, zorder=0)
    ax.grid(axis="x", which="minor", color="#eeeeee", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.6)


_draw_panel(ax_left, infra_recs, INFRA_COLORS, "Infrastructure")
_draw_panel(ax_right, feat_recs, FEAT_COLOR, "Feature extraction")

plt.show()

# %%
# Summary table
# -------------
# A formatted text table makes it easy to report latencies in a paper or
# notebook.  The last row shows the total pipeline mean latency.

COL_W = [22, 10, 10, 10]
HEADER = ["Stage", "Mean (ms)", "Std (ms)", "P95 (ms)"]
SEP = "  ".join("-" * w for w in COL_W)

print(SEP)
print("  ".join(h.ljust(w) for h, w in zip(HEADER, COL_W)))
print(SEP)

total_mean = 0.0
for r in records:
    row = [
        r["stage"].replace("\n", " "),
        f"{r['mean_ms']:.3f}",
        f"{r['std_ms']:.3f}",
        f"{r['p95_ms']:.3f}",
    ]
    print("  ".join(v.ljust(w) for v, w in zip(row, COL_W)))
    total_mean += r["mean_ms"]

print(SEP)
print(f"  {'TOTAL pipeline mean'.ljust(COL_W[0])}  {total_mean:.3f}")
print(SEP)
