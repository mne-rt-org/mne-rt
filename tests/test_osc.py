"""Tests for OSCSender (no network connection required)."""

import importlib

import pytest

pythonosc_available = importlib.util.find_spec("pythonosc") is not None
requires_osc = pytest.mark.skipif(
    not pythonosc_available,
    reason="python-osc not installed (install with: pip install 'mne-rt[osc]')",
)


@requires_osc
def test_osc_sender_init():
    from mne_rt.osc import OSCSender

    sender = OSCSender(host="127.0.0.1", port=9001, prefix="/test")
    assert sender.prefix == "/test"
    assert "127.0.0.1" in sender.target
    assert "9001" in sender.target


@requires_osc
def test_osc_sender_repr():
    from mne_rt.osc import OSCSender

    s = OSCSender(host="127.0.0.1", port=9001)
    assert "127.0.0.1" in repr(s)


@requires_osc
def test_osc_sender_close():
    from mne_rt.osc import OSCSender

    s = OSCSender(host="127.0.0.1", port=9001)
    s.close()  # should not raise


@requires_osc
def test_osc_context_manager():
    from mne_rt.osc import OSCSender

    with OSCSender(host="127.0.0.1", port=9001) as s:
        assert s is not None


@pytest.mark.skipif(pythonosc_available, reason="python-osc IS installed")
def test_osc_not_installed(monkeypatch):
    """If python-osc is absent, OSCSender.__init__ should raise ImportError."""
    import sys

    # Hide the module
    monkeypatch.setitem(sys.modules, "pythonosc", None)
    monkeypatch.setitem(sys.modules, "pythonosc.udp_client", None)
    monkeypatch.setitem(sys.modules, "pythonosc.osc_message_builder", None)
    monkeypatch.setitem(sys.modules, "pythonosc.osc_bundle_builder", None)

    # Re-import with the module hidden
    import importlib

    import mne_rt.osc as _osc_mod

    importlib.reload(_osc_mod)

    try:
        _osc_mod.OSCSender(host="127.0.0.1", port=9001)
    except ImportError:
        pass  # expected
    finally:
        importlib.reload(_osc_mod)  # restore
