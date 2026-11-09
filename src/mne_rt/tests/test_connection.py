"""Connecting to data: connect_to_lsl, connect_to_array, ArrayStream and the
mock LSL player.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mne_rt.tests._helpers import _make_array_data, _make_rt_stream

pytestmark = pytest.mark.usefixtures("fast_clock")


def test_connect_to_lsl_and_record_baseline(tmp_path, short_mock_fif):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_lsl(mock_lsl=True, fname=str(short_mock_fif), timeout=15.0)
        assert nf.sfreq == pytest.approx(256.0)
        assert len(nf.rec_info["ch_names"]) == 8

        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        assert nf.raw_baseline is not None
        assert nf.raw_baseline.info["sfreq"] == pytest.approx(256.0)
        assert nf.raw_baseline.n_times >= int(1.0 * 256.0)
    finally:
        nf.save()  # disconnects the stream / mock player


def test_connect_to_array_invalid_shape(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(ValueError, match="2D"):
        nf.connect_to_array(np.zeros((8, 10, 2)), array_info)


def test_connect_to_array_channel_mismatch(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(ValueError, match="channels"):
        nf.connect_to_array(np.zeros((4, 100)), array_info)


def test_connect_to_array_and_record_baseline(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    try:
        nf.connect_to_array(data, array_info)
        assert nf.sfreq == pytest.approx(256.0)
        assert len(nf.rec_info["ch_names"]) == 8
        assert nf.stream.connected

        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        assert nf.raw_baseline is not None
        assert nf.raw_baseline.info["sfreq"] == pytest.approx(256.0)
        assert nf.raw_baseline.n_times >= int(1.0 * 256.0)
    finally:
        nf.save()  # disconnects the stream


def test_record_baseline_rejects_a_winsize_larger_than_the_buffer(tmp_path):
    """The stream returns at most its buffer, so the rest of each chunk was
    silently missing from the baseline."""
    nf = _make_rt_stream(tmp_path)
    nf.bufsize = 3
    with pytest.raises(ValueError, match="buffer"):
        nf.record_baseline(baseline_duration=10, winsize=4)


class _Stop(Exception):
    pass


def test_connect_to_lsl_bufsize_baseline_is_deprecated(tmp_path, monkeypatch):
    """It was documented but never used. The connection itself is cut short:
    the warning is raised before anything is connected."""
    import warnings

    nf = _make_rt_stream(tmp_path)

    def stop():
        raise _Stop

    monkeypatch.setattr(nf, "_teardown_existing_stream", stop)
    with pytest.warns(FutureWarning, match="bufsize_baseline") as record, pytest.raises(_Stop):
        nf.connect_to_lsl(bufsize_baseline=4)
    assert record[0].filename == __file__  # points at the caller, not at mne-rt
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with pytest.raises(_Stop):
            nf.connect_to_lsl()


def test_array_stream_has_unique_sample_clock_timestamps(array_info):
    from mne_rt.rt_stream import ArrayStream

    stream = ArrayStream(_make_array_data(array_info), array_info, n_repeat=np.inf)
    stream.connect()
    try:
        time.sleep(0.15)
        _, timestamps = stream.get_data()
        valid = timestamps[timestamps > 0]
        assert valid.size > 10
        np.testing.assert_allclose(np.diff(valid), 1 / array_info["sfreq"], atol=1e-9)
    finally:
        stream.disconnect()


def test_connect_to_array_pick_types(tmp_path):
    import mne

    ch_names = [f"EEG{i:03d}" for i in range(4)] + ["EOG001"]
    ch_types = ["eeg"] * 4 + ["eog"]
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types=ch_types)
    data = np.random.default_rng(0).standard_normal((5, 256)) * 1e-6

    nf = _make_rt_stream(tmp_path, montage=None)
    try:
        nf.connect_to_array(data, info, pick_types="eeg")
        assert nf.rec_info["ch_names"] == ch_names[:4]
    finally:
        nf.save()


def test_connect_to_array_open_stream_viewer_raises(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    try:
        nf.connect_to_array(data, array_info)
        with pytest.raises(RuntimeError, match="connect_to_array"):
            nf.open_stream_viewer()
    finally:
        nf.save()


def _make_array_stream(n_channels=4, sfreq=100.0, n_samples=500, **kwargs):
    import mne

    from mne_rt.rt_stream import ArrayStream

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = np.random.default_rng(0).standard_normal((n_channels, n_samples)) * 1e-6
    return ArrayStream(data, info, **kwargs)


class TestArrayStream:
    def test_not_connected_raises(self):
        stream = _make_array_stream()
        assert not stream.connected
        with pytest.raises(RuntimeError, match="not connected"):
            stream.get_data()
        with pytest.raises(RuntimeError, match="connect"):
            _ = stream.info

    def test_connect_get_data_resets_n_new_samples(self):
        import time

        stream = _make_array_stream(sfreq=200.0, bufsize=1.0, chunk_size=10, n_repeat=np.inf)
        stream.connect()
        try:
            time.sleep(0.3)
            data, ts = stream.get_data(0.2)
            assert data.shape == (4, 40)
            assert ts.shape == (40,)
            assert stream.n_new_samples == 0
            time.sleep(0.2)
            assert stream.n_new_samples > 0
        finally:
            stream.disconnect()
        assert not stream.connected

    def test_pick_reduces_channels_before_connect(self):
        stream = _make_array_stream(n_channels=4)
        stream.pick(["EEG000", "EEG002"])
        assert stream._info["ch_names"] == ["EEG000", "EEG002"]
        stream.connect()
        try:
            data, _ = stream.get_data(0.1)
            assert data.shape[0] == 2
        finally:
            stream.disconnect()

    def test_pick_preserves_requested_order(self):
        stream = _make_array_stream(n_channels=4)
        stream.pick(["EEG002", "EEG000"])
        assert stream._info["ch_names"] == ["EEG002", "EEG000"]

    def test_pick_after_connect_raises(self):
        stream = _make_array_stream()
        stream.connect()
        try:
            with pytest.raises(RuntimeError, match="before connect"):
                stream.pick("eeg")
        finally:
            stream.disconnect()

    def test_get_data_excludes_bads(self):
        stream = _make_array_stream(n_channels=4)
        stream._info["bads"] = ["EEG001"]
        stream.connect()
        try:
            data, _ = stream.get_data(0.1)
            assert data.shape[0] == 3
        finally:
            stream.disconnect()

    def test_connect_twice_does_not_leak_thread(self):
        stream = _make_array_stream(sfreq=200.0, n_repeat=np.inf)
        stream.connect()
        first_thread = stream._thread
        stream.connect()
        try:
            assert not first_thread.is_alive()
            assert stream.connected
        finally:
            stream.disconnect()

    def test_filter_and_notch_filter_return_self(self):
        stream = _make_array_stream(sfreq=256.0, n_samples=2560)
        assert stream.filter(1.0, 40.0) is stream
        assert stream.notch_filter(50.0) is stream

    def test_n_repeat_stops_advancing(self):
        import time

        stream = _make_array_stream(
            sfreq=100.0, n_samples=20, chunk_size=5, n_repeat=1, bufsize=0.5
        )
        stream.connect()
        try:
            time.sleep(0.5)  # long enough to exhaust the 20-sample array once
            assert stream.n_new_samples == 20
        finally:
            stream.disconnect()


@pytest.mark.real_time
def test_array_stream_timestamps_use_the_lsl_clock(array_info):
    """The test double must share a time base with a real LSL stream.

    `time.time()` and `local_clock()` are ~1.8e9 s apart, so an onset derived
    under this double would otherwise land on a different clock entirely.
    """
    from mne_lsl.lsl import local_clock

    from mne_rt.rt_stream import ArrayStream

    stream = ArrayStream(_make_array_data(array_info, duration=2.0), array_info, n_repeat=np.inf)
    stream.connect()
    try:
        _wait_for = time.time() + 2.0
        while stream.n_new_samples < 10 and time.time() < _wait_for:
            time.sleep(0.01)
        ts = stream.get_data()[1]
        assert ts.size
        assert abs(float(ts[-1]) - local_clock()) < 1.0
    finally:
        stream.disconnect()


def test_mock_connects_to_the_data_not_the_annotations(tmp_path):
    """The mock player publishes a recording's annotations as a second stream
    with the same source_id; matching the source_id alone connected to either
    (the bundled sample's 17 marker channels, more often than not)."""
    from mne_rt import RTStream
    from mne_rt.rt_stream import _REPO_ROOT

    sample = _REPO_ROOT / "data" / "sample" / "sample_data.vhdr"
    if not sample.is_file():
        pytest.skip("needs the bundled sample recording (editable install)")
    nf = RTStream("sub01", "01", str(tmp_path), montage=None)
    try:
        for _ in range(2):
            nf.connect_to_lsl(mock_lsl=True, fname=str(sample), timeout=15.0)
            assert nf.rec_info["nchan"] == 64
            assert set(nf.stream.get_channel_types()) == {"eeg"}
    finally:
        nf.save()
