"""Tests for LSLSender — LSL output layer.

Most tests mock the LSL backend so no real LSL runtime is required.  That
convenience is also how a real bug shipped: ``MagicMock`` accepts any argument,
so nothing noticed that ``push`` was handing mne-lsl a Python list, which it
rejects for numeric streams.  Two tests therefore attack the mock — one asserts
the *type* of the pushed sample, and one drops the mock entirely and pushes
through a real outlet, as ``test_osc.py`` does for OSC.
"""

import importlib
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

mne_lsl_available = importlib.util.find_spec("mne_lsl") is not None
requires_mne_lsl = pytest.mark.skipif(
    not mne_lsl_available,
    reason="mne-lsl not installed",
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_mock_lsl():
    """Return (MockStreamInfo, MockStreamOutlet) pair."""
    MockInfo = MagicMock()
    MockOutlet = MagicMock()
    return MockInfo, MockOutlet


def _make_sender(**kwargs):
    """Instantiate LSLSender with a mocked LSL backend."""
    from mne_rt.lsl_output import LSLSender

    MockInfo, MockOutlet = _make_mock_lsl()
    with patch.object(LSLSender, "_import_lsl", staticmethod(lambda: (MockInfo, MockOutlet))):
        sender = LSLSender(**kwargs)
    return sender, MockInfo, MockOutlet


# ------------------------------------------------------------------
# Instantiation
# ------------------------------------------------------------------


def test_default_construction():
    sender, MockInfo, MockOutlet = _make_sender()
    assert sender.n_channels == 8
    assert sender.channel_labels == []
    assert sender.stream_name == "ANT_NF"


def test_custom_params():
    sender, _, _ = _make_sender(stream_name="MyStream", n_channels=4)
    assert sender.stream_name == "MyStream"
    assert sender.n_channels == 4


# ------------------------------------------------------------------
# push
# ------------------------------------------------------------------


def test_push_single_channel():
    from mne_rt.lsl_output import LSLSender

    MockInfo, MockOutlet = _make_mock_lsl()
    outlet_instance = MagicMock()
    MockOutlet.return_value = outlet_instance

    with patch.object(LSLSender, "_import_lsl", staticmethod(lambda: (MockInfo, MockOutlet))):
        sender = LSLSender(n_channels=1)

    sender.push(["alpha_power"], [0.5])
    outlet_instance.push_sample.assert_called_once()
    args = outlet_instance.push_sample.call_args[0][0]
    assert args[0] == pytest.approx(0.5)


def test_push_sets_channel_labels():
    sender, _, _ = _make_sender(n_channels=3)
    sender.push(["alpha", "beta", "gamma"], [1.0, 2.0, 3.0])
    assert sender.channel_labels == ["alpha", "beta", "gamma"]


def test_push_pads_with_zeros():
    from mne_rt.lsl_output import LSLSender

    MockInfo, MockOutlet = _make_mock_lsl()
    outlet_instance = MagicMock()
    MockOutlet.return_value = outlet_instance

    with patch.object(LSLSender, "_import_lsl", staticmethod(lambda: (MockInfo, MockOutlet))):
        sender = LSLSender(n_channels=4)

    sender.push(["alpha"], [0.42])
    args = outlet_instance.push_sample.call_args[0][0]
    assert len(args) == 4
    assert args[0] == pytest.approx(0.42)
    assert args[1] == pytest.approx(0.0)


def test_push_length_mismatch_raises():
    sender, _, _ = _make_sender(n_channels=3)
    with pytest.raises(ValueError):
        sender.push(["alpha", "beta"], [1.0, 2.0, 3.0])


# ------------------------------------------------------------------
# push_value (single-channel convenience)
# ------------------------------------------------------------------


def test_push_value():
    from mne_rt.lsl_output import LSLSender

    MockInfo, MockOutlet = _make_mock_lsl()
    outlet_instance = MagicMock()
    MockOutlet.return_value = outlet_instance

    with patch.object(LSLSender, "_import_lsl", staticmethod(lambda: (MockInfo, MockOutlet))):
        sender = LSLSender(n_channels=1)

    sender.push_value("sensor_power", 3.14)
    outlet_instance.push_sample.assert_called_once()
    args = outlet_instance.push_sample.call_args[0][0]
    assert args[0] == pytest.approx(3.14)


# ------------------------------------------------------------------
# n_channels property
# ------------------------------------------------------------------


def test_n_channels_property():
    sender, _, _ = _make_sender(n_channels=6)
    assert sender.n_channels == 6


# ------------------------------------------------------------------
# close
# ------------------------------------------------------------------


def test_close_sets_outlet_none():
    from mne_rt.lsl_output import LSLSender

    MockInfo, MockOutlet = _make_mock_lsl()
    outlet_instance = MagicMock()
    outlet_instance.close = MagicMock()
    MockOutlet.return_value = outlet_instance

    with patch.object(LSLSender, "_import_lsl", staticmethod(lambda: (MockInfo, MockOutlet))):
        sender = LSLSender()

    sender.close()
    assert sender._outlet is None


def test_close_is_idempotent():
    sender, _, _ = _make_sender()
    sender.close()
    sender.close()  # should not raise


# ------------------------------------------------------------------
# Context manager
# ------------------------------------------------------------------


def test_context_manager():
    from mne_rt.lsl_output import LSLSender

    MockInfo, MockOutlet = _make_mock_lsl()
    outlet_instance = MagicMock()
    outlet_instance.close = MagicMock()
    MockOutlet.return_value = outlet_instance

    with patch.object(LSLSender, "_import_lsl", staticmethod(lambda: (MockInfo, MockOutlet))):
        with LSLSender() as sender:
            sender.push(["alpha"], [0.1])

    assert sender._outlet is None  # closed after context exit


# ------------------------------------------------------------------
# Thread safety — concurrent pushes
# ------------------------------------------------------------------


def test_concurrent_push_does_not_crash():
    sender, _, _ = _make_sender(n_channels=2)
    errors = []

    def push_loop():
        try:
            for _ in range(50):
                sender.push(["a", "b"], [1.0, 2.0])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=push_loop) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------


def test_repr():
    sender, _, _ = _make_sender(stream_name="TestStream")
    r = repr(sender)
    assert "LSLSender" in r
    assert "TestStream" in r


# ------------------------------------------------------------------
# Channel labels reach the StreamInfo
# ------------------------------------------------------------------


def test_push_publishes_channel_names():
    """Labels must land in the StreamInfo, not just be stored on the sender.

    Position-based subscription is ambiguous once two channels share a base
    modality and differ only by instance label.
    """
    sender, MockInfo, _ = _make_sender(n_channels=2)
    sender.push(["source_connectivity@theta", "source_connectivity@alpha"], [0.1, 0.2])

    assert sender.channel_labels == [
        "source_connectivity@theta",
        "source_connectivity@alpha",
    ]
    info = MockInfo.return_value
    info.set_channel_names.assert_called_with(
        ["source_connectivity@theta", "source_connectivity@alpha"]
    )


def test_channel_names_are_padded_to_the_outlet_width():
    sender, MockInfo, _ = _make_sender(n_channels=4)
    sender.push(["sensor_power"], [0.1])
    names = MockInfo.return_value.set_channel_names.call_args[0][0]
    assert names == ["sensor_power", "ch1", "ch2", "ch3"]


def test_push_survives_a_backend_without_set_channel_names():
    """The pylsl fallback has no such setter; labels are never worth a crash."""
    sender, MockInfo, MockOutlet = _make_sender(n_channels=2)
    MockInfo.return_value.set_channel_names.side_effect = AttributeError("no such method")
    sender.push(["sensor_power", "hjorth"], [0.1, 0.2])
    MockOutlet.return_value.push_sample.assert_called()


def test_outlet_is_not_rebuilt_when_labels_are_unchanged():
    sender, MockInfo, _ = _make_sender(n_channels=2)
    sender.push(["sensor_power", "hjorth"], [0.1, 0.2])
    n_after_first = MockInfo.call_count
    sender.push(["sensor_power", "hjorth"], [0.3, 0.4])
    assert MockInfo.call_count == n_after_first


def test_changing_labels_does_not_rebuild_the_outlet():
    """Rebuilding drops subscribers, so it must not happen per sample.

    ``push_value`` passes a different single name each call, which would
    otherwise tear the outlet down on every sample.
    """
    sender, MockInfo, _ = _make_sender(n_channels=2)
    sender.push_value("sensor_power", 0.1)
    n_after_first = MockInfo.call_count
    for _ in range(5):
        sender.push_value("sensor_power", 0.2)
        sender.push_value("hjorth", 0.3)
    assert MockInfo.call_count == n_after_first
    assert sender.channel_labels == ["hjorth"]  # still tracked


def test_channel_names_given_up_front_avoid_any_rebuild():
    sender, MockInfo, _ = _make_sender(
        n_channels=2, channel_names=["sensor_power@alpha", "sensor_power@theta"]
    )
    MockInfo.return_value.set_channel_names.assert_called_with(
        ["sensor_power@alpha", "sensor_power@theta"]
    )
    n_after_init = MockInfo.call_count
    sender.push(["sensor_power@alpha", "sensor_power@theta"], [0.1, 0.2])
    assert MockInfo.call_count == n_after_init  # outlet never rebuilt


# ------------------------------------------------------------------
# The backend's actual contract
# ------------------------------------------------------------------


def test_push_sends_an_ndarray_not_a_list():
    """mne-lsl asserts a NumPy array for numeric streams; a list raises.

    Every test here mocks the outlet, and ``MagicMock`` takes a list happily —
    so pushing a list passed the suite while failing against every real backend,
    and no session ever delivered LSL feedback.  Nothing asserted the type.
    """
    sender, _, MockOutlet = _make_sender(n_channels=2, channel_names=["a", "b"])
    sender.push(["a", "b"], [1.0, 2.0])

    arg = MockOutlet.return_value.push_sample.call_args[0][0]
    assert isinstance(arg, np.ndarray)
    # float32 is the outlet's own dtype, so mne-lsl does not have to copy.
    assert arg.dtype == np.float32
    assert arg.shape == (2,)


def test_push_pads_the_array_without_disturbing_the_values():
    """The zero padding survives the move from list to array."""
    sender, _, MockOutlet = _make_sender(n_channels=4)
    sender.push(["only"], [0.42])

    arg = MockOutlet.return_value.push_sample.call_args[0][0]
    assert isinstance(arg, np.ndarray)
    assert arg.shape == (4,)
    assert arg[0] == pytest.approx(0.42)
    assert np.all(arg[1:] == 0.0)


@requires_mne_lsl
def test_push_reaches_a_real_lsl_outlet():
    """The decisive one: every mocked test passed while this raised.

    Creating an outlet and pushing needs no subscriber and no network peer, the
    same reason ``test_osc.py`` can send UDP with nothing listening.
    """
    from mne_rt.lsl_output import LSLSender

    sender = LSLSender(
        stream_name="mne_rt_test_push",
        n_channels=2,
        channel_names=["alpha", "beta"],
    )
    try:
        sender.push(["alpha", "beta"], [0.25, -0.5])
        # The padded path: one value into a two-channel outlet.
        sender.push_value("alpha", 1.5)
    finally:
        sender.close()
