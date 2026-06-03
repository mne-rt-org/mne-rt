"""Tests for LSLSender — LSL output layer.

All tests mock the LSL backend so no real LSL runtime is required.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest


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
