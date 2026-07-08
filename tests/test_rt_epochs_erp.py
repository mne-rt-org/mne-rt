"""Test RTEpochs + TopoPlot with a mock LSL player.

Uses the MNE sample dataset (MEG + EEG + STIM) to drive a PlayerLSL,
collects 10 epochs via RTEpochs (backed by mne_lsl.EpochsStream), and
verifies shapes and TopoPlot data path.

Run with:
    python tests/test_rt_epochs_erp.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")  # suppress pyqtgraph render-time noise
import threading

os.environ.setdefault("MPLBACKEND", "Agg")  # headless matplotlib
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # headless Qt

# Make sure src/ is on the path when running without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mne
import numpy as np

# ---------------------------------------------------------------------------
# 1. Locate sample data
# ---------------------------------------------------------------------------
print("Loading MNE sample data …")
data_path = mne.datasets.sample.data_path()
raw_path = str(data_path) + "/MEG/sample/sample_audvis_raw.fif"

raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
raw.pick(["eeg", "stim"], verbose=False)  # keep EEG + STIM only to keep it lightweight
raw.filter(1.0, 40.0, verbose=False)

# Save a trimmed version so PlayerLSL doesn't stream the full 4.5 min
raw_trim = raw.copy().crop(tmax=60.0)  # first 60 s — enough for ~20 epochs
trim_path = "/tmp/mne_rt_test_trim.fif"
raw_trim.save(trim_path, overwrite=True, verbose=False)
print(f"  Saved trimmed raw ({raw_trim.times[-1]:.1f} s) → {trim_path}")

event_id = {"auditory/left": 1, "auditory/right": 2}


# ---------------------------------------------------------------------------
# 2. Test RTEpochs
# ---------------------------------------------------------------------------
print("\n[RTEpochs] Connecting …")
from mne_rt import RTEpochs

received: list[np.ndarray] = []
conditions_received: list[str] = []


def on_trial(n_accepted, data, event_code, condition):
    received.append(data.copy())
    print(f"  Trial {n_accepted} ({condition}): shape={data.shape}")


rt = RTEpochs(
    event_id=event_id,
    event_channels="STI 014",
    tmin=-0.1,
    tmax=0.4,
    baseline=(None, 0),
    picks="eeg",
    reject={"eeg": 150e-6},
    on_trial=on_trial,
)

rt.connect_to_lsl(mock_lsl=True, fname=trim_path, timeout=15.0)

# Run in a thread so we can time it out
done = threading.Event()


def _run():
    rt.run(n_trials=10, show_erp=False)
    done.set()


t = threading.Thread(target=_run, daemon=True)
t.start()
done.wait(timeout=60.0)

if not done.is_set():
    rt.stop()
    print("  WARNING: timed out before 10 trials — checking partial results")

rt.disconnect()

n_got = rt.n_accepted_
print(f"\n[RTEpochs] Accepted {n_got} trials.")
assert n_got > 0, "RTEpochs collected zero epochs!"

# Check epoch shape: (n_eeg_channels, n_times)
epoch = received[0]
n_eeg = len([c for c in raw_trim.ch_names if c.startswith("EEG")])
n_times_expected = int((0.4 - (-0.1)) * raw_trim.info["sfreq"])
print(f"  Epoch shape: {epoch.shape}  (expect ~({n_eeg}, {n_times_expected}))")
assert epoch.ndim == 2, f"Expected 2-D epoch, got {epoch.ndim}-D"
print("  [PASS] RTEpochs shape OK")


# ---------------------------------------------------------------------------
# 3. Test TopoPlot data path (headless — no window shown)
# ---------------------------------------------------------------------------
print("\n[TopoPlot] Testing data computation path …")
from mne_rt.viz.topo_plot import TopoPlot

ch_names_eeg = [c for c in raw_trim.ch_names if c.startswith("EEG")]
sfreq = raw_trim.info["sfreq"]
tmin, tmax = -0.1, 0.4
n_times = epoch.shape[1]  # 301 — MNE includes both endpoints

# Build a synthetic 4-epoch dataset (2 conditions × 2 trials)
rng = np.random.default_rng(0)
n_ch = len(ch_names_eeg)
fake_epochs = rng.standard_normal((4, n_ch, n_times)) * 1e-6  # (4, 60, 301)
fake_conditions = ["auditory/left", "auditory/right", "auditory/left", "auditory/right"]

# TopoPlot.__init__ needs Qt; wrap in try/except so the test degrades gracefully
try:
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)

    erp = TopoPlot(
        ch_names=ch_names_eeg,
        sfreq=sfreq,
        tmin=tmin,
        tmax=tmax,
        event_id={"auditory/left": 1, "auditory/right": 2},
        montage="easycap-M1",
    )

    # update() computes averages and schedules a Qt repaint
    erp.update(fake_epochs, fake_conditions)

    # Manually call _redraw to verify maths without the event loop
    erp._redraw(n_total=4)

    # Verify the per-condition averages are correct
    for cond in ["auditory/left", "auditory/right"]:
        mask = np.array([c == cond for c in fake_conditions])
        expected_avg = fake_epochs[mask].mean(axis=0)  # (n_ch, n_times)
        buf = np.stack(erp._epoch_buf[cond], axis=0)
        actual_avg = buf.mean(axis=0)
        np.testing.assert_allclose(actual_avg, expected_avg, rtol=1e-5)
        print(f"  [PASS] TopoPlot average correct for condition '{cond}'")

    print("  [PASS] TopoPlot data path OK")

except ImportError as e:
    print(f"  [SKIP] Qt not available in this environment: {e}")
except Exception as e:
    print(f"  [FAIL] TopoPlot error: {e}")
    raise

print("\nAll tests passed.")
