"""Marker streams, condition latching and feedback gating."""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pytest

from mne_rt.tests._helpers import _CountingProtocol, _make_array_data, _make_rt_stream


class _RecordingSender:
    """Stands in for an OSC or LSL sender, counting what actually goes out."""

    def __init__(self):
        self.sends = []

    def send_all(self, names, values):
        self.sends.append((list(names), list(values)))

    def push(self, names, values):
        self.sends.append((list(names), list(values)))


def _gated_session(tmp_path, array_info, *, codes, onsets, duration=12.0, **kwargs):
    """Run a short gated session against an array stream and a marker schedule."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=40.0), array_info, n_repeat=np.inf)
    nf.connect_marker_array(codes=codes, onsets=onsets, marker_id={"speech": 1, "rest": 2})
    nf.record_main(
        duration=duration,
        winsize=1.0,
        modality="sensor_power",
        show_raw_signal=False,
        show_nf_signal=False,
        **kwargs,
    )
    return nf


def test_marker_array_stream_delivers_each_marker_once():
    """The high-water mark is what stops a marker being consumed twice.

    ``get_data`` returns the *entire* buffer on every call, so a poll loop
    without the mark would re-deliver everything it had already seen.
    """
    from mne_rt import MarkerArrayStream

    stream = MarkerArrayStream([1, 2, 3], [0.05, 0.15, 0.25], bufsize=16)
    stream.connect()
    try:
        seen, hwm = [], 0.0
        deadline = time.time() + 3.0
        while time.time() < deadline and len(seen) < 3:
            data, ts = stream.get_data()
            for i in range(ts.size):
                if ts[i] > hwm:
                    seen.append(int(data[0, i]))
            if ts.size:
                hwm = max(hwm, float(ts.max()))
            time.sleep(0.01)
    finally:
        stream.disconnect()

    assert seen == [1, 2, 3]


def test_marker_array_stream_uses_the_lsl_clock():
    """Markers and window onsets must be comparable without conversion."""
    from mne_lsl.lsl import local_clock

    from mne_rt import MarkerArrayStream

    stream = MarkerArrayStream([7], [0.0], bufsize=4)
    stream.connect()
    try:
        deadline = time.time() + 2.0
        ts = np.zeros(1)
        while time.time() < deadline and not ts.max():
            ts = stream.get_data()[1]
            time.sleep(0.01)
        assert ts.max() > 0
        assert abs(float(ts.max()) - local_clock()) < 1.0
    finally:
        stream.disconnect()


def test_marker_array_stream_zero_winsize_returns_nothing():
    """`ArrayStream` gets this wrong: a `[-0:]` slice returns the whole buffer."""
    from mne_rt import MarkerArrayStream

    stream = MarkerArrayStream([1], [0.0], bufsize=8).connect()
    try:
        data, ts = stream.get_data(0)
        assert data.shape == (1, 0) and ts.size == 0
    finally:
        stream.disconnect()


def test_marker_array_stream_rejects_mismatched_schedule():
    from mne_rt import MarkerArrayStream

    with pytest.raises(ValueError, match="same length"):
        MarkerArrayStream([1, 2], [0.0])
    with pytest.raises(ValueError, match="non-decreasing"):
        MarkerArrayStream([1, 2], [1.0, 0.0])


def test_marker_stream_is_not_spliced_onto_self(tmp_path, array_info):
    """`_finalize_stream` copies every non-dunder callable onto the RTStream.

    Running the marker stream through it would rebind `self.get_data`,
    `self.disconnect` and `self._push` to the marker stream, quietly cutting
    the signal path.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
    signal_get_data = nf.get_data
    try:
        nf.connect_marker_array(codes=[1], onsets=[0.0], marker_id={"speech": 1})
        assert nf.get_data is signal_get_data
        assert nf.get_data.__self__ is nf.stream
    finally:
        nf.stream.disconnect()
        nf.marker_stream.disconnect()


def test_unknown_marker_channel_names_what_is_available(tmp_path):
    """mne-lsl's own message does not list the names the stream does publish."""

    class _Stub:
        info = {"nchan": 1, "sfreq": 0.0, "ch_names": ["0"]}
        disconnected = False

        def disconnect(self):
            self.disconnected = True

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    stub = _Stub()
    with pytest.raises(ValueError, match=r"\['0'\]") as excinfo:
        nf._attach_marker_stream(stub, marker_id={"go": 1}, ch_name="markers")

    # An outlet that does not label its channel surfaces as "0".
    assert "'0', '1'" in str(excinfo.value)
    assert stub.disconnected


def test_gate_conditions_without_a_marker_stream_is_refused(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
    try:
        with pytest.raises(RuntimeError, match="needs a marker stream"):
            nf.record_main(
                duration=1.0,
                modality="sensor_power",
                gate_conditions=["speech"],
                show_raw_signal=False,
                show_nf_signal=False,
            )
    finally:
        nf.stream.disconnect()


def test_gate_condition_that_can_never_match_is_refused(tmp_path, array_info):
    """A typo here would otherwise mute the whole session in silence."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
    nf.connect_marker_array(codes=[1], onsets=[0.0], marker_id={"speech": 1})
    try:
        with pytest.raises(ValueError, match="not in the marker stream"):
            nf.record_main(
                duration=1.0,
                modality="sensor_power",
                gate_conditions=["speach"],
                show_raw_signal=False,
                show_nf_signal=False,
            )
    finally:
        nf.stream.disconnect()
        nf.marker_stream.disconnect()


def test_condition_latches_and_switches_with_the_markers(tmp_path, array_info):
    """Latch, not containment: every window after the first marker is labelled.

    With 50 % overlap most windows contain no marker at all, so a containment
    rule would leave them unlabelled.
    """
    nf = _gated_session(tmp_path, array_info, codes=[1, 2], onsets=[1.0, 5.0], duration=10.0)

    conditions = nf.window_conditions
    assert set(conditions) <= {None, "speech", "rest"}
    # No condition until the first marker: "nothing has been announced yet" is
    # not the same as a condition, and a gated session must not treat it as one.
    assert conditions[0] is None
    assert conditions[-1] == "rest"
    # Each state entered once, in order -- the latch never flaps back.
    switches = [(a, b) for a, b in zip(conditions, conditions[1:]) if a != b]
    assert switches == [(None, "speech"), ("speech", "rest")]


def test_every_marker_is_counted_exactly_once(tmp_path, array_info):
    """Overlapping windows must not double-count a contained marker."""
    nf = _gated_session(
        tmp_path, array_info, codes=[1, 2, 1], onsets=[2.0, 5.0, 8.0], duration=11.0
    )

    # The invariant, not a count that depends on how long setup took: every
    # marker the run saw is attributed to exactly one window.
    assert len(nf.markers) >= 2
    assert max(nf.window_marker_counts) <= 1
    assert sum(nf.window_marker_counts) == len(nf.markers)


def test_gating_mutes_the_protocol_and_the_senders(tmp_path, array_info):
    """A muted window still records a value, but drives nothing outward."""
    proto = _CountingProtocol(threshold=-1e30)
    osc, lsl = _RecordingSender(), _RecordingSender()
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=10.0,
        gate_conditions=["speech"],
        protocol={"sensor_power": proto},
        osc_sender=osc,
        lsl_sender=lsl,
    )

    n_windows = len(nf.nf_data["sensor_power"])
    n_live = n_windows - nf.n_gated_windows
    assert 0 < n_live < n_windows  # the gate really did open and close
    assert len(proto.values) == n_live
    assert len(osc.sends) == n_live
    assert len(lsl.sends) == n_live
    # The trace itself stays dense -- a muted window is recorded, not skipped.
    assert len(nf.window_gated) == n_windows
    assert sum(nf.window_gated) == nf.n_gated_windows


def test_muted_windows_keep_reward_data_aligned(tmp_path, array_info):
    """A missing placeholder would shift every later reward row silently."""
    proto = _CountingProtocol(threshold=-1e30)
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=10.0,
        gate_conditions=["speech"],
        protocol={"sensor_power": proto},
    )

    n_windows = len(nf.nf_data["sensor_power"])
    assert nf.n_gated_windows > 0
    assert len(nf.reward_data["sensor_power"]) == n_windows
    assert len(nf.window_onsets) == n_windows
    assert len(nf.window_conditions) == n_windows
    assert len(nf.window_marker_counts) == n_windows


def test_ungated_session_is_unchanged(tmp_path, array_info):
    """`gate_conditions=None` must behave exactly as it did before."""
    proto = _CountingProtocol(threshold=-1e30)
    osc = _RecordingSender()
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=8.0,
        protocol={"sensor_power": proto},
        osc_sender=osc,
    )

    n_windows = len(nf.nf_data["sensor_power"])
    assert nf.n_gated_windows == 0
    assert len(proto.values) == n_windows
    assert len(osc.sends) == n_windows
    # Windows are still labelled -- annotation happens whether or not it gates.
    assert nf.window_conditions[0] == "speech"


def test_warmup_no_longer_rewards_native_units(tmp_path, array_info):
    """`_apply_zscore` passes the raw value through until warmup completes.

    Evaluating those windows rewarded native-unit power (~1e-11) against a
    z-score-calibrated threshold -- random reinforcement at the start of every
    run.
    """
    proto = _CountingProtocol(threshold=-1e30)
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=40.0), array_info, n_repeat=np.inf)
    nf.record_main(
        duration=10.0,
        winsize=1.0,
        modality="sensor_power",
        protocol={"sensor_power": proto},
        zscore_normalize=True,
        zscore_warmup=6,
        show_raw_signal=False,
        show_nf_signal=False,
    )

    n_windows = len(nf.nf_data["sensor_power"])
    assert n_windows > 6
    assert len(proto.values) == n_windows - 5  # windows 1..5 are pass-through
    # Nothing the protocol saw is still in native units.
    assert max(abs(v) for v in proto.values) > 1e-6


def test_markers_and_gate_round_trip_through_save(tmp_path, array_info):
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=10.0,
        gate_conditions=["speech"],
    )

    path = next(pathlib.Path(nf.subject_dir).rglob("*_beh.json"))
    payload = json.loads(path.read_text())

    # The data block stays modality-only: meta["modalities"], the TSV columns
    # and TransferProtocol all derive from its keys.
    assert set(payload["data"]) == {"sensor_power"}
    assert payload["windows"]["condition"][0] == "speech"
    assert set(payload["windows"]["gated"]) == {0, 1}
    assert payload["meta"]["gate_conditions"] == ["speech"]
    assert payload["meta"]["n_gated_windows"] > 0
    assert payload["meta"]["marker_id"] == {"speech": 1, "rest": 2}
    # The raw markers, as ground truth for anything the columns cannot express.
    assert payload["markers"]["code"] == [1, 2]


def test_beh_tsv_carries_the_marker_columns(tmp_path, array_info):
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=10.0,
        gate_conditions=["speech"],
    )

    tsv = next(pathlib.Path(nf.subject_dir).rglob("*_beh.tsv"))
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    assert header[:5] == ["onset", "duration", "condition", "n_markers", "gated"]
    assert lines[1].split("\t")[2] == "speech"


def test_reconnecting_the_signal_stream_drops_the_marker_stream_loudly(
    tmp_path, array_info, monkeypatch
):
    """Marker timestamps only mean anything against the recording they came with.

    Dropping one silently would leave a session that looks gated but never gates.
    `caplog` cannot see this: the package logger sets ``propagate = False``.
    """
    from mne_rt import rt_stream as rt_stream_mod

    warnings_seen = []
    monkeypatch.setattr(
        rt_stream_mod.logger,
        "warning",
        lambda msg, *a, **kw: warnings_seen.append(str(msg)),
    )

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
    nf.connect_marker_array(codes=[1], onsets=[0.0], marker_id={"speech": 1})
    try:
        nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
        assert nf.marker_stream is None
        assert any("drops the marker stream" in w for w in warnings_seen)
    finally:
        nf.stream.disconnect()


def test_no_marker_stream_leaves_the_table_unchanged(tmp_path, array_info):
    """The marker columns must not appear for a session that had no markers.

    They are appended per window, so placeholders would give every existing
    user's table three new all-empty columns.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf)
    nf.record_main(
        duration=4.0,
        winsize=1.0,
        modality="sensor_power",
        show_raw_signal=False,
        show_nf_signal=False,
    )

    assert nf.window_conditions == []
    payload = json.loads(next(pathlib.Path(nf.subject_dir).rglob("*_beh.json")).read_text())
    assert set(payload["windows"]) == {"onset", "duration", "onset_lsl"}
    assert "markers" not in payload

    tsv = next(pathlib.Path(nf.subject_dir).rglob("*_beh.tsv"))
    header = tsv.read_text().splitlines()[0].split("\t")
    assert header == ["onset", "duration", "sensor_power"]


def test_marker_overflow_is_detectable_before_the_read():
    """`n_new_samples` is reset by `get_data`, so it has to be read first.

    Read afterwards it is always 0, and the one diagnostic that would tell a
    user their markers were overwritten never fires.
    """
    from mne_rt import MarkerArrayStream

    stream = MarkerArrayStream([1, 2, 3, 4], [0.0, 0.01, 0.02, 0.03], bufsize=2)
    stream.connect()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and stream.n_new_samples < 2:
            time.sleep(0.01)
        assert stream.n_new_samples >= 2
        stream.get_data()
        assert stream.n_new_samples == 0
    finally:
        stream.disconnect()


def test_condition_ignores_markers_arriving_after_the_window_ends(tmp_path, array_info):
    """`condition` and `n_markers` must agree about which window a marker is in.

    The post-fetch poll also consumes markers newer than the window's last
    sample; letting one of those label the window would open the gate early.
    """
    nf = _gated_session(tmp_path, array_info, codes=[1, 2], onsets=[1.0, 5.0], duration=10.0)

    counts = nf.window_marker_counts
    conditions = nf.window_conditions
    first_rest = next(i for i, c in enumerate(conditions) if c == "rest")
    marker_windows = [i for i, n in enumerate(counts) if n]
    assert marker_windows
    # The switch cannot precede the window the marker was attributed to.
    assert first_rest >= marker_windows[-1]


def test_sidecar_omits_empty_levels():
    """`"Levels": null` would break a reader doing `Levels.items()`."""
    from mne_rt.tools.bids_io import describe_nf_columns

    cols = describe_nf_columns(
        nf_data={"sensor_power": [1.0]},
        windows={"condition": ["speech"], "n_markers": [1], "gated": [0]},
    )
    assert "Levels" not in cols["condition"]

    cols = describe_nf_columns(
        nf_data={"sensor_power": [1.0]},
        windows={"condition": ["speech"], "n_markers": [1], "gated": [0]},
        meta={"marker_id": {"speech": 1, "rest": 2}},
    )
    assert cols["condition"]["Levels"] == {"rest": "rest", "speech": "speech"}
