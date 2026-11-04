"""Tests for :class:`mne_rt.RTEpochs`, in particular the event-stream path.

``EpochsStream`` type-checks its inputs against ``mne_lsl``'s ``BaseStream``,
so ``ArrayStream`` cannot stand in the way it does for ``RTStream``, and the
only alternative is a live LSL replay.  These tests therefore substitute the
two mne-lsl classes with doubles and assert the *wiring* — which stream is
built, with what, connected in which order, and torn down.
"""

import math

import numpy as np
import pytest

from mne_rt import rt_epochs
from mne_rt.rt_epochs import RTEpochs

pytest.importorskip("mne_lsl")


class _Recorder:
    """Shared call log, so ordering across objects can be asserted."""

    def __init__(self):
        self.events: list[str] = []
        self.streams: list["_FakeStream"] = []
        self.epochs: list["_FakeEpochsStream"] = []


class _FakeStream:
    def __init__(self, rec, bufsize, *, name=None, stype=None, source_id=None):
        self._rec = rec
        self.bufsize = bufsize
        self.name = name
        self.source_id = source_id
        self.connected = False
        self.disconnected = False
        self.connect_kwargs = None
        self.info = {"nchan": 1, "sfreq": 0.0, "ch_names": ["markers"]}
        rec.streams.append(self)

    def connect(self, acquisition_delay=0.001, *, processing_flags=None, timeout=2):
        self.connect_kwargs = dict(
            acquisition_delay=acquisition_delay,
            processing_flags=processing_flags,
            timeout=timeout,
        )
        self.connected = True
        self._rec.events.append(f"connect:{self.name}")
        return self

    def disconnect(self):
        self.connected = False
        self.disconnected = True
        self._rec.events.append(f"disconnect:{self.name}")


class _FakeEpochsStream:
    #: mne-lsl builds ``ceil((tmax - tmin) * sfreq)`` samples, ``endpoint=False``.
    SFREQ = 600.0

    def __init__(self, rec, **kwargs):
        self._rec = rec
        self.kwargs = kwargs
        self.disconnected = False
        self.info = {"nchan": 2, "sfreq": self.SFREQ, "ch_names": ["eeg1", "eeg2"]}
        tmin, tmax = kwargs.get("tmin", -0.2), kwargs.get("tmax", 0.8)
        self.times = np.linspace(tmin, tmax, math.ceil((tmax - tmin) * self.SFREQ), endpoint=False)
        self._served = False
        rec.epochs.append(self)
        rec.events.append("build:epochs")

    @property
    def n_new_epochs(self):
        return 0 if self._served else 1

    @property
    def events(self):
        return np.array([1])

    def get_data(self, n_epochs=None):
        self._served = True
        # All-ones, so any sample the buffer failed to fill shows up as a zero.
        return np.ones((1, self.info["nchan"], self.times.size), dtype=np.float32)

    def connect(self, acquisition_delay=0.001):
        self._rec.events.append("connect:epochs")
        return self

    def disconnect(self):
        self.disconnected = True
        self._rec.events.append("disconnect:epochs")


class _FakePlayer:
    def __init__(self, *args, **kwargs):
        self.stopped = False

    def start(self):
        return self

    def stop(self):
        self.stopped = True


@pytest.fixture
def rec(monkeypatch):
    """Patch mne-lsl out of ``rt_epochs`` and return the shared call log."""
    recorder = _Recorder()

    def make_stream(bufsize, **kwargs):
        stream = _FakeStream(recorder, bufsize, **kwargs)
        # The M/EEG stream is the first one built; give it a plausible info.
        if len(recorder.streams) == 1:
            stream.info = {"nchan": 64, "sfreq": 1000.0, "ch_names": ["eeg1"]}
        return stream

    monkeypatch.setattr(rt_epochs, "StreamLSL", make_stream)
    monkeypatch.setattr(
        rt_epochs, "EpochsStream", lambda **kwargs: _FakeEpochsStream(recorder, **kwargs)
    )
    monkeypatch.setattr(rt_epochs, "PlayerLSL", _FakePlayer)
    return recorder


def _rt(**kwargs):
    kwargs.setdefault("event_id", {"go": 1, "stop": 2})
    kwargs.setdefault("event_channels", "markers")
    return RTEpochs(**kwargs)


# ------------------------------------------------------------------
# No event stream: the pre-existing path must be untouched
# ------------------------------------------------------------------


def test_without_event_stream_only_one_stream_is_opened(rec):
    rt = _rt(event_channels="STI 014").connect_to_lsl(stream_name="eeg")

    assert len(rec.streams) == 1
    assert rt.event_stream_ is None
    assert rec.epochs[0].kwargs["event_stream"] is None


def test_without_event_stream_no_processing_flags_are_set(rec):
    """The single-stream path must behave exactly as it did before."""
    _rt(event_channels="STI 014").connect_to_lsl(stream_name="eeg")

    assert rec.streams[0].connect_kwargs["processing_flags"] is None


# ------------------------------------------------------------------
# With an event stream
# ------------------------------------------------------------------


def test_event_stream_is_built_from_name_and_source_id(rec):
    rt = _rt().connect_to_lsl(
        stream_name="eeg",
        event_stream_name="psychopy_markers",
        event_source_id="psychopy_uid",
    )

    assert len(rec.streams) == 2
    event_stream = rec.streams[1]
    assert event_stream.name == "psychopy_markers"
    assert event_stream.source_id == "psychopy_uid"
    assert rt.event_stream_ is event_stream
    assert rec.epochs[0].kwargs["event_stream"] is event_stream


def test_event_stream_bufsize_is_an_integer_count_of_samples(rec):
    """A marker outlet is irregular, so mne-lsl rejects a float bufsize."""
    _rt().connect_to_lsl(stream_name="eeg", event_stream_name="markers", event_stream_bufsize=64)

    bufsize = rec.streams[1].bufsize
    assert isinstance(bufsize, int) and bufsize == 64


@pytest.mark.parametrize("bad", [0, -1, 0.5, 199.9])
def test_unusable_event_bufsize_is_rejected(rec, bad):
    """Caught before anything is opened -- it can only be a coding error.

    A non-integral value must be refused rather than truncated: mne-lsl needs an
    ``int`` here, and silently rounding 199.9 down would also make the error
    message name a number the caller never passed.
    """
    with pytest.raises(ValueError, match="positive integer") as excinfo:
        _rt().connect_to_lsl(
            stream_name="eeg", event_stream_name="markers", event_stream_bufsize=bad
        )

    assert repr(bad) in str(excinfo.value)
    assert rec.streams == []


def test_event_stream_is_connected_before_the_epochs_stream(rec):
    """mne-lsl raises if the event stream is not already connected."""
    _rt().connect_to_lsl(stream_name="eeg", event_stream_name="markers")

    assert rec.events.index("connect:markers") < rec.events.index("build:epochs")


# ------------------------------------------------------------------
# Clock synchronisation
# ------------------------------------------------------------------


def test_clocksync_is_enabled_on_both_streams(rec):
    """Without it, a marker from a second host lands in the wrong epoch."""
    _rt().connect_to_lsl(stream_name="eeg", event_stream_name="markers")

    assert [s.connect_kwargs["processing_flags"] for s in rec.streams] == ["all", "all"]


def test_explicit_processing_flags_win(rec):
    _rt().connect_to_lsl(
        stream_name="eeg", event_stream_name="markers", processing_flags=("clocksync",)
    )

    assert [s.connect_kwargs["processing_flags"] for s in rec.streams] == [
        ("clocksync",),
        ("clocksync",),
    ]


# ------------------------------------------------------------------
# Event channel validation
# ------------------------------------------------------------------


def test_unknown_event_channel_names_what_is_available(rec):
    with pytest.raises(ValueError, match=r"\['triggers'\].*\['markers'\]"):
        _rt(event_channels="triggers").connect_to_lsl(
            stream_name="eeg", event_stream_name="markers"
        )


def test_unknown_event_channel_explains_the_unlabelled_outlet(rec):
    """An outlet that does not label its channel is exposed as "0"."""
    with pytest.raises(ValueError, match="'0', '1'"):
        _rt(event_channels="triggers").connect_to_lsl(
            stream_name="eeg", event_stream_name="markers"
        )


def test_event_channels_may_be_a_list(rec):
    rt = _rt(event_channels=["markers"]).connect_to_lsl(
        stream_name="eeg", event_stream_name="markers"
    )

    assert rt.event_stream_ is rec.streams[1]


# ------------------------------------------------------------------
# Teardown
# ------------------------------------------------------------------


def test_disconnect_tears_down_the_event_stream_after_the_epochs_stream(rec):
    rt = _rt().connect_to_lsl(stream_name="eeg", event_stream_name="markers")
    rt.disconnect()

    assert all(s.disconnected for s in rec.streams)
    assert rec.epochs[0].disconnected
    assert rec.events.index("disconnect:epochs") < rec.events.index("disconnect:markers")


def test_reconnecting_without_an_event_stream_drops_the_old_one(rec):
    """The stale stream is disconnected, and EpochsStream refuses one of those."""
    rt = _rt().connect_to_lsl(stream_name="eeg", event_stream_name="markers")
    rt.disconnect()

    rt.event_channels = "STI 014"
    rt.connect_to_lsl(stream_name="eeg")

    assert rt.event_stream_ is None
    assert rec.epochs[-1].kwargs["event_stream"] is None


def test_failed_setup_leaves_nothing_connected(rec, monkeypatch):
    """A raise mid-setup must not strand a player and two open streams."""

    def boom(**kwargs):
        raise RuntimeError("no epochs for you")

    monkeypatch.setattr(rt_epochs, "EpochsStream", boom)

    rt = _rt()
    with pytest.raises(RuntimeError, match="no epochs for you"):
        rt.connect_to_lsl(mock_lsl=True, fname="x.fif", event_stream_name="markers")

    assert not any(s.connected for s in rec.streams)
    assert rt._player.stopped
    assert rt._connected is False


def test_string_event_stream_is_reported_against_the_event_stream(rec, monkeypatch):
    """mne-lsl's own message does not say which of the two streams failed."""
    real_connect = _FakeStream.connect

    def connect(self, *args, **kwargs):
        if self.name == "markers":
            raise RuntimeError(
                "The Stream class is designed for numerical types. It does not "
                "support string LSL streams."
            )
        return real_connect(self, *args, **kwargs)

    monkeypatch.setattr(_FakeStream, "connect", connect)

    with pytest.raises(RuntimeError, match="event stream publishes strings"):
        _rt().connect_to_lsl(stream_name="eeg", event_stream_name="markers")


# ------------------------------------------------------------------
# The doubles above are only worth anything if they match the real API
# ------------------------------------------------------------------


def test_the_calls_this_module_makes_bind_against_real_mne_lsl():
    """Pins the doubles to mne-lsl, so an upstream signature change is caught."""
    import inspect

    from mne_lsl.stream import EpochsStream, StreamLSL

    inspect.signature(StreamLSL.__init__).bind(None, bufsize=200, name="markers", source_id="uid")
    inspect.signature(StreamLSL.connect).bind(
        None, acquisition_delay=0.005, processing_flags="all", timeout=10.0
    )
    inspect.signature(EpochsStream.__init__).bind(
        None,
        stream=None,
        bufsize=200,
        event_id={"go": 1},
        event_channels="markers",
        event_stream=None,
        tmin=-0.2,
        tmax=0.8,
        baseline=(None, 0),
        picks=None,
        reject=None,
    )
    inspect.signature(EpochsStream.connect).bind(None, acquisition_delay=0.005)


# ------------------------------------------------------------------
# Epoch buffer geometry
# ------------------------------------------------------------------


def test_epoch_buffer_is_as_wide_as_the_epochs_stream(rec):
    """Sized from ``es.times``, not re-derived.

    ``round((tmax - tmin) * sfreq) + 1`` is one sample wider than the
    ``ceil(...)``-with-``endpoint=False`` array mne-lsl actually produces, which
    left a trailing all-zero sample on every epoch and pushed the reported
    ``tmax`` a sample past the truth.
    """
    rt = _rt(event_channels="STI 014", tmin=-0.1, tmax=0.4)
    rt.connect_to_lsl(stream_name="eeg")
    rt.run(n_trials=1, poll_interval=0.0)

    es = rec.epochs[0]
    assert es.times.size == 300  # ceil(0.5 * 600), not 301
    assert rt._buf_.shape == (1, 2, es.times.size)
    assert np.all(rt._buf_[0] == 1.0)  # nothing left unfilled
