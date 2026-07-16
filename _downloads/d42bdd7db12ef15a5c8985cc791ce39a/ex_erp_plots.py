"""
Real-time ERP accumulation — RTEpochs + epoch visualisation
=============================================================

Demonstrates the full epoch-based real-time pipeline:

1. Stream a segment of the MNE sample dataset through a mock LSL player.
2. Collect event-triggered epochs trial-by-trial with
   :class:`~mne_rt.RTEpochs`.  A per-trial ``on_trial`` callback receives
   the accumulating buffer after every accepted epoch.
3. Inspect the accumulated evoked responses offline with matplotlib.

The four epoch visualisation windows —
:class:`~mne_rt.viz.TopoPlot`,
:class:`~mne_rt.viz.ButterflyPlot`,
:class:`~mne_rt.viz.CompareEvoked`, and
:class:`~mne_rt.viz.TFRPlot` —
open automatically when running interactively.  To launch them::

    mne-rt demo-erp

or drive them from your own script by passing each widget as a callback
target inside ``on_trial``, exactly as shown in the :ref:`cli` page.
"""

# %%
# Prepare demo data
# -----------------
# We use the MNE auditory sample dataset (downloaded once, ~1.5 GB).
# The recording contains clear N100 / P200 components on a clean stimulus
# channel — ideal for accumulating ERP responses in real time.

import os
import sys
import time
import threading

os.environ.setdefault("MPLBACKEND", "Agg")

import mne
import numpy as np
import matplotlib.pyplot as plt

data_path = mne.datasets.sample.data_path()
raw_full  = mne.io.read_raw_fif(
    str(data_path) + "/MEG/sample/sample_audvis_raw.fif",
    preload=True, verbose=False,
)
raw_full.filter(1.0, 40.0, verbose=False)
raw_demo  = raw_full.copy().crop(tmax=270.0)

mock_path = "/tmp/mne_rt_ex_erp_raw.fif"
raw_demo.save(mock_path, overwrite=True, verbose=False)
print(f"Demo file: {raw_demo.times[-1]:.0f} s, {len(raw_demo.ch_names)} channels")

# %%
# Connect RTEpochs to the mock stream
# -------------------------------------
# :class:`~mne_rt.RTEpochs` wraps :class:`~mne_lsl.stream.EpochsStream` and
# adds a per-trial ``on_trial(n_accepted, data, event_code, condition)``
# callback that fires after every accepted epoch.  This is the hook that the
# Qt plot windows use to update in real time.

from mne_rt import RTEpochs

STIM_CH  = "STI 014"
EVENT_ID = {"auditory/left": 1, "auditory/right": 2}
TMIN, TMAX = -0.1, 0.4
N_TRIALS   = 40

rt = RTEpochs(
    event_id       = EVENT_ID,
    event_channels = STIM_CH,
    tmin           = TMIN,
    tmax           = TMAX,
    baseline       = (None, 0),
    picks          = "eeg",
    reject         = {"eeg": 150e-6},
)
rt.connect_to_lsl(mock_lsl=True, fname=mock_path, timeout=15.0)

epochs_info = rt.epochs_stream_.info.copy()   # save before disconnect
ch_names    = list(epochs_info["ch_names"])
sfreq       = epochs_info["sfreq"]
n_times     = int(round((TMAX - TMIN) * sfreq)) + 1
times       = np.linspace(TMIN, TMAX, n_times)
print(f"RTEpochs connected — {len(ch_names)} EEG channels, {sfreq:.0f} Hz")

# %%
# Register an ``on_trial`` callback and collect epochs
# -----------------------------------------------------
# ``rt._buf_[:n_accepted]`` holds the full accumulated epoch array
# ``(n_accepted, n_channels, n_times)`` at each call.
# ``rt._cond_list_`` mirrors the condition label for every accepted epoch.

epoch_buf: dict[str, list[np.ndarray]] = {c: [] for c in EVENT_ID}
update_times: list[float] = []

def on_trial(n_accepted: int, data: np.ndarray,
             event_code: int, condition: str) -> None:
    t0 = time.perf_counter()
    # Latest epoch is the last row of the buffer
    latest = rt._buf_[n_accepted - 1]          # (n_ch, n_times)
    epoch_buf[condition].append(latest.copy())
    update_times.append((time.perf_counter() - t0) * 1000)

rt.on_trial = on_trial

done = threading.Event()

def _collect():
    rt.run(n_trials=N_TRIALS, show_erp=False)
    done.set()

print(f"Collecting {N_TRIALS} trials …")
threading.Thread(target=_collect, daemon=True).start()
done.wait(timeout=120)
rt.disconnect()

n_left  = len(epoch_buf["auditory/left"])
n_right = len(epoch_buf["auditory/right"])
print(f"Accepted: {n_left} left  +  {n_right} right  =  {n_left + n_right} total")
if update_times:
    print(f"Callback latency: mean={np.mean(update_times):.2f} ms  "
          f"max={np.max(update_times):.2f} ms")

# %%
# Compare evoked responses with MNE
# -----------------------------------
# Wrap each accepted epoch as an :class:`~mne.EvokedArray` and pass a list
# per condition to :func:`mne.viz.plot_compare_evokeds`.  This gives
# automatically computed 95 % confidence intervals via bootstrapping and
# MNE's publication-ready styling.

evokeds: dict[str, list[mne.EvokedArray]] = {}
for cond, buf in epoch_buf.items():
    if buf:
        evokeds[cond] = [
            mne.EvokedArray(trial, info=epochs_info, tmin=TMIN,
                            nave=1, comment=cond)
            for trial in buf
        ]

fig = mne.viz.plot_compare_evokeds(
    evokeds,
    picks     = ["EEG 021"],
    ci        = 0.95,
    show_sensors = "upper right",
    truncate_yaxis = False,
    show      = False,
    title     = "Real-time ERP accumulation — EEG 021",
)
# plot_compare_evokeds returns a list of figures
figs = fig if isinstance(fig, list) else [fig]
for f in figs:
    f.set_size_inches(8, 4)
    f.tight_layout()
plt.show()

# %%
# Single-trial epoch image
# -------------------------
# Assemble all accepted epochs into an :class:`~mne.Epochs` object and use
# :func:`mne.viz.plot_epochs_image` to show the per-trial amplitude at
# EEG 021 together with the running average and an ERP inset.

all_data = np.concatenate(
    [np.stack(epoch_buf[c]) for c in EVENT_ID if epoch_buf[c]], axis=0
)
all_events = np.column_stack([
    np.arange(len(all_data)),
    np.zeros(len(all_data), int),
    np.ones(len(all_data), int),
])
epochs_obj = mne.EpochsArray(
    all_data, info=epochs_info, events=all_events,
    tmin=TMIN, baseline=(None, 0),
)

fig_img = mne.viz.plot_epochs_image(
    epochs_obj,
    picks   = ["EEG 021"],
    show    = False,
    title   = "Single-trial epochs — EEG 021",
)[0]
fig_img.set_size_inches(8, 6)
plt.show()
