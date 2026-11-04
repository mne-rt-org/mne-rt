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
    """A stable identity for a Qt object, surviving Python wrapper churn.

    ``id()`` is not usable here: a widget whose C++ object outlives its Python
    wrapper gets a fresh wrapper the next time ``topLevelWidgets()`` returns it
    (which is why the same window shows up as ``RawPlot`` in one sweep and
    ``QMainWindow`` in the next), and CPython recycles the address of the
    collected wrapper. The underlying C++ pointer does not move.
    """
    try:  # PyQt5 / PyQt6
        from PyQt6 import sip

        return sip.unwrapinstance(widget)
    except Exception:
        try:  # PySide2 / PySide6
            import shiboken6

            return shiboken6.getCppPointer(widget)[0]
        except Exception:  # unknown binding — degrade to per-wrapper identity
            return id(widget)


@pytest.fixture()
def _destroy_widgets(qt_app):
    """Tear each test's Qt widgets down deterministically, exactly once each.

    Used by every viewer test module that builds widgets, through
    ``pytestmark = pytest.mark.usefixtures("_destroy_widgets")``; not autouse,
    because a module that shares one window between its tests (BrainPlot's)
    must not have it closed after the first.

    ``widget.close()`` only hides a widget; the C++ object is destroyed later —
    either when Python garbage-collects the wrapper or when the event loop runs
    a deferred deletion. Leaving those two to race is what made the viewer
    tests (then one file, test_viz_plots.py) abort
    intermittently in CI (``Fatal Python error``, exit 134) on every Python
    version: the event loop would process an event belonging to an object whose
    wrapper had just been collected.

    Verified against a deliberate reproduction (an autouse fixture calling only
    ``processEvents()`` crashes 3/3 runs under
    ``QT_QPA_PLATFORM=offscreen``): handing the widgets to Qt for deletion and
    collecting before draining brings that to 0/3.

    Each widget is handled **once**, tracked by C++ pointer. This matters for
    wall-clock, not correctness: every ``RawPlot`` leaves 16 parentless
    top-level widgets behind (its window plus pyqtgraph's ``ViewBox``/
    ``PlotItem`` context menus), and they are never freed — ``processEvents()``
    does not dispatch ``DeferredDelete``, and forcing it to
    (``sendPostedEvents(None, QEvent.Type.DeferredDelete)``) segfaults, because
    pyqtgraph still references those menus. The leak is load-bearing.

    Re-sweeping the survivors on every test made teardown O(widgets alive), so
    the tests cost O(n^2) overall: 38,912 close/deleteLater pairs across what
    were then 54 tests, against 1,313 live widgets. Offscreen raster hid that locally at
    ~0.06 s per teardown, but under CI's Xvfb/xcb each close is real X11
    traffic, and `TestRawPlot` grew from ~2 s for the whole class to
    52 s -> 105 s -> 212 s -> 423 s per test until the job hit its 30-minute
    timeout. Handling each widget once restores 1,313 calls, one per widget.
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


# Reproducing the abort `_destroy_widgets` prevents: replace it, in the viewer
# test modules, with a fixture that only drains the event queue,
#
#     @pytest.fixture(autouse=True)
#     def _drain(qt_app):
#         yield
#         qt_app.processEvents()
#
# and `pytest src/mne_rt/viz/tests` segfaults on every run (measured when they
# were one file, test_viz_plots.py). Collecting garbage *before* draining, rather than after, is what
# makes the difference — which is how the GC/event-loop race was identified.
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
