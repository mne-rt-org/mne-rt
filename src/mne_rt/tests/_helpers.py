"""Helpers shared by the RTStream tests."""

from __future__ import annotations

import numpy as np


def _make_rt_stream(tmp_path, **kwargs):
    from mne_rt import RTStream

    defaults = dict(
        subject_id="sub01",
        session="01",
        subjects_dir=str(tmp_path),
        montage="easycap-M1",
    )
    defaults.update(kwargs)
    return RTStream(**defaults)


def _make_baseline_raw(n_channels=8, sfreq=256.0, duration=2.0):
    import mne

    rng = np.random.default_rng(0)
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = rng.standard_normal((n_channels, int(sfreq * duration))) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


def _make_array_data(info, duration=8.0):
    rng = np.random.default_rng(0)
    n_samples = int(duration * info["sfreq"])
    return rng.standard_normal((len(info["ch_names"]), n_samples)) * 1e-6


def _failing_after(n_ok, exc):
    """A modality stand-in: ``n_ok`` good windows, then ``exc``."""
    calls = {"n": 0}

    def modality(data, **_):
        calls["n"] += 1
        if calls["n"] > n_ok:
            raise exc
        return 1.0, 0.0

    return modality


_HEADLESS = {"show_raw_signal": False, "show_nf_signal": False}


class _CountingProtocol:
    """Records how many times it was evaluated, and with what."""

    def __init__(self, threshold=0.0):
        self.current_threshold = threshold
        self.values = []

    def evaluate(self, value):
        self.values.append(float(value))
        return (float(value) > self.current_threshold, abs(float(value)))
