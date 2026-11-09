"""Fixtures shared by the RTStream tests."""

from __future__ import annotations

import os

import numpy as np
import pytest

# How many times faster than real time `fast_clock` runs a session; lower it
# with MNE_RT_TEST_SPEED on a machine too slow to keep up.
FAST = float(os.environ.get("MNE_RT_TEST_SPEED", "10"))


@pytest.fixture()
def fast_clock(request):
    """Run the in-memory streams, and sessions on them, FAST times faster.

    A session on a real LSL stream keeps the real clock regardless. Compute time
    and thread jitter grow FAST times in session time too, so a test that checks
    timing precisely is marked ``real_time``. Its own MonkeyPatch, so a test's
    ``monkeypatch.undo()`` cannot drop the clock mid-session.
    """
    if request.node.get_closest_marker("real_time"):
        yield
        return
    from mne_lsl.lsl import local_clock

    from mne_rt import rt_stream

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rt_stream, "_CLOCK_ORIGIN", local_clock())
        mp.setattr(rt_stream, "_SPEED", FAST)
        yield


@pytest.fixture()
def subjects_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def short_mock_fif(tmp_path):
    import mne

    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:8]
    sfreq = 256.0
    duration = 8.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)

    rng = np.random.default_rng(0)
    n_samples = int(duration * sfreq)
    data = rng.standard_normal((len(ch_names), n_samples)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)

    fname = tmp_path / "mock-raw.fif"
    raw.save(fname, verbose=False)
    return fname


@pytest.fixture()
def array_info():
    import mne

    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:8]
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    info.set_montage(montage)
    return info
