"""Generate paper/figure1_architecture.png — the MNE-RT closed loop.

Run with ``python paper/make_figure.py`` to regenerate the figure used in
``paper/paper.md``.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

INK, EDGE = "#1a1a1a", "#4a4a4a"
FILL, ACC, CORE, HL = "#f4f5f7", "#dce8f3", "#eaeff5", "#c9dcee"
LOOP = "#2f5d8a"

W_FIG, H_FIG = 14.0, 5.2
fig, ax = plt.subplots(figsize=(W_FIG, H_FIG))
ax.set_xlim(0, W_FIG)
ax.set_ylim(0, H_FIG)
ax.axis("off")


def box(x, y, w, h, title, sub=None, fc=FILL, ts=11.5, ss=8.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.08",
            linewidth=1.1,
            edgecolor=EDGE,
            facecolor=fc,
        )
    )
    if sub:
        ax.text(
            x + w / 2,
            y + h * 0.72,
            title,
            ha="center",
            va="center",
            fontsize=ts,
            color=INK,
            weight="bold",
        )
        ax.text(
            x + w / 2,
            y + h * 0.32,
            sub,
            ha="center",
            va="center",
            fontsize=ss,
            color="#333333",
            linespacing=1.45,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=ts,
            color=INK,
            weight="bold",
        )


def arrow(p1, p2, rad=0.0, color=EDGE, lw=1.35):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


Y, H = 3.05, 1.32  # main row
row = [  # (x, w, title, subtitle, facecolor)
    (0.25, 1.85, "Acquisition", "LSL inlet\namplifier · file replay\nin-memory array", ACC),
    (2.32, 2.15, "Artifact correction", "LMS · ORICA · GEDAI\nASR · Maxwell/SSS", CORE),
    (
        4.69,
        2.50,
        "Feature extraction",
        "thread pool, one modality per\nthread — 20 modalities",
        CORE,
    ),
    (7.41, 1.95, "Standardise", "online z-score · EMA\nfeature combiners", CORE),
    (9.58, 2.25, "Protocol", "z-score · staircase · RL\noperant · sham · transfer", CORE),
    (12.05, 1.75, "Session record", "BIDS-compatible\nfeatures · rewards\nlatency · SNR", FILL),
]
for x, w, t, s, fc in row:
    box(x, Y, w, H, t, s, fc=fc)
for (x1, w1, *_), (x2, *_) in zip(row, row[1:]):
    arrow((x1 + w1, Y + H / 2), (x2, Y + H / 2))

# --- source-space branch, hanging under Feature extraction ----------------
fx, fw = row[2][0], row[2][1]
sx, sw, sy, sh = fx - 0.42, fw + 0.84, 1.48, 1.28
box(
    sx,
    sy,
    sw,
    sh,
    "Source space",
    "ROI kernel   R = A · W · whitener\nLCMV / MNE — cortical + subcortical ROIs\n"
    "650–800 ms  →  ~1 ms per window",
    fc=HL,
    ts=11.0,
    ss=8.2,
)
arrow((fx + fw / 2 - 0.30, Y), (fx + fw / 2 - 0.30, sy + sh), lw=1.15)
arrow((fx + fw / 2 + 0.30, sy + sh), (fx + fw / 2 + 0.30, Y), lw=1.15)

# --- feedback bar along the bottom ----------------------------------------
bx, bw, by, bh = 2.32, 7.04, 0.38, 0.64
box(bx, by, bw, bh, "Feedback out      Qt displays   ·   LSL outlet   ·   OSC", fc=ACC, ts=11.0)
arrow((row[4][0] + 0.55, Y), (bx + bw, by + bh / 2), rad=-0.22)

# closed loop: feedback back to the participant and the amplifier
ax.add_patch(
    FancyArrowPatch(
        (bx, by + bh / 2),
        (row[0][0] + row[0][1] / 2, Y),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.6,
        color=LOOP,
        connectionstyle="arc3,rad=0.26",
        shrinkA=2,
        shrinkB=0,
    )
)
ax.text(
    0.92, 1.72, "closed loop", fontsize=9.6, color=LOOP, style="italic", ha="center", rotation=62
)

ax.text(0.25, 4.80, "MNE-RT", fontsize=14.5, weight="bold", color=INK)
ax.text(
    1.62, 4.82, "— one analysis window (e.g. 1 s, 50 % overlap)", fontsize=10.6, color="#333333"
)

fig.savefig(
    str(Path(__file__).with_name("figure1_architecture.png")),
    dpi=220,
    bbox_inches="tight",
    facecolor="white",
)
print("ok")
