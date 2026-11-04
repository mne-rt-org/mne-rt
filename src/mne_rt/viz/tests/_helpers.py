"""Constants and helpers shared by the viewer tests."""

from __future__ import annotations

import numpy as np

from mne_rt._montage import make_standard_montage

SFREQ = 256.0


TMIN = -0.1


TMAX = 0.4


N_TIMES = int((TMAX - TMIN) * SFREQ) + 1  # 129


CH_NAMES = [f"EEG{i:03d}" for i in range(1, 9)]  # 8 synthetic channels


N_CH = len(CH_NAMES)


EVENT_ID = {"left": 1, "right": 2}


# 10-20 channel names for tests that need a real montage (e.g. TopomapPlot)
TOPO_CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]


N_TOPO = len(TOPO_CH_NAMES)


def _make_epochs(n: int, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """Return (data, conditions) with ``n`` epochs split evenly.

    The same ``seed`` gives the same data; pass another for different data.
    """
    # Its own generator: data that depends on which tests ran before is how a
    # failure reproduces in the full suite but not alone.
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, N_CH, N_TIMES)).astype(np.float32) * 1e-6
    half = n // 2
    conds = ["left"] * half + ["right"] * (n - half)
    return data, conds


NF_MODALITIES = ["sensor_power", "erd_ers"]


NF_SCALES = {"sensor_power": 1e-12, "erd_ers": 1.0}


def _reward_visible(w, i):
    """Whether any pooled reward-span region is currently shown for modality ``i``."""
    return any(r.isVisible() for r in w._reward_regions[i])


def _selector(**kw):
    from mne_rt.viz._sensor_selector import SensorSelector, sensor_positions

    return SensorSelector(TOPO_CH_NAMES, sensor_positions(TOPO_CH_NAMES), **kw)


def _bad_p3_info():
    import mne

    names = "P3 C3 F3 Fz F4 C4 P4 Cz Pz Fp1 Fp2 T3 T5 O1 O2 F7 F8 T6 T4".split()
    info = mne.create_info(names, SFREQ, "eeg")
    info.set_montage(make_standard_montage("colin27_1020"))
    info["bads"] = ["P3"]
    return info


class _Click:
    """Just enough of a pyqtgraph click event for mouseClickEvent."""

    def __init__(self, button, pos):
        self._button, self._pos, self.accepted = button, pos, False

    def button(self):
        return self._button

    def pos(self):
        return self._pos

    def accept(self):
        self.accepted = True

    def ignore(self):
        pass
