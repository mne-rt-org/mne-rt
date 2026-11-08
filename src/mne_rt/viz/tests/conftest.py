"""Qt fixtures for the viewer tests: one QApplication, and a deterministic
teardown of every test's widgets (see `_destroy_widgets`).
"""

from __future__ import annotations

import gc

import pytest

from mne_rt._montage import make_standard_montage
from mne_rt.viz.tests._helpers import CH_NAMES, EVENT_ID, SFREQ, TMAX, TMIN, TOPO_CH_NAMES


@pytest.fixture(scope="module")
def qt_app():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


# C++ pointers of every top-level widget already closed and handed to Qt for
# deletion. Module-level because the widgets outlive the tests that made them.
_TORN_DOWN: set[int] = set()


def _qt_handle(widget):
    """The C++ pointer, a stable identity for a Qt object (PyQt6, PySide6).

    ``id()`` only as a last resort: a widget whose wrapper was collected gets a
    new one from ``topLevelWidgets()``, and CPython reuses freed addresses.
    """
    try:  # PyQt6
        from PyQt6 import sip

        return sip.unwrapinstance(widget)
    except Exception:
        try:  # PySide6
            import shiboken6

            return shiboken6.getCppPointer(widget)[0]
        except Exception:  # unknown binding — degrade to per-wrapper identity
            return id(widget)


@pytest.fixture()
def _destroy_widgets(qt_app):
    """Tear each test's Qt widgets down deterministically, once each.

    Opt-in per module (``pytestmark``), not autouse: BrainPlot's tests share
    one window, which must not close after the first.

    ``close()`` only hides a widget; leaving its deletion to race between the
    garbage collector and the event loop aborted CI runs (exit 134). Handing
    the widgets to Qt and collecting *before* draining prevents that: with a
    fixture that only calls ``processEvents()``, these tests segfault.

    Each widget is handled once, by C++ pointer: pyqtgraph's context menus are
    never freed (forcing ``DeferredDelete`` segfaults), and re-sweeping them
    every test made teardown quadratic, timing CI out under Xvfb.
    """
    yield
    from qtpy.QtWidgets import QApplication

    fresh = []
    for widget in QApplication.topLevelWidgets():
        handle = _qt_handle(widget)
        if handle not in _TORN_DOWN:
            _TORN_DOWN.add(handle)
            fresh.append(widget)
    for widget in fresh:
        widget.close()
        widget.deleteLater()
    qt_app.processEvents()
    gc.collect()
    qt_app.processEvents()


@pytest.fixture(scope="module")
def topo_info():
    """mne.Info with standard 10-20 montage for TopomapPlot tests."""
    import mne

    info = mne.create_info(TOPO_CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    montage = make_standard_montage("colin27_1020")  # standard_1020 before MNE 1.13
    info.set_montage(montage, on_missing="ignore")
    return info


@pytest.fixture(scope="module")
def common_kw():
    return dict(
        ch_names=CH_NAMES,
        sfreq=SFREQ,
        tmin=TMIN,
        tmax=TMAX,
        event_id=EVENT_ID,
        baseline=(None, 0),
    )
