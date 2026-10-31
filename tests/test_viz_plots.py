"""Headless tests for the real-time visualisation classes.

Runs in offscreen (no display) mode.  Each test:
  - constructs the widget with synthetic EEG-like data
  - calls the public API (push / update) with small batches of fake data
  - verifies internal state (buffers, condition counts, shapes, flags)
  - does NOT test pixel output — only data paths

Environment setup (offscreen Qt) is handled in conftest.py or via the
QT_QPA_PLATFORM env var set at module level below.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gc

import numpy as np
import pyqtgraph as pg
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SFREQ = 256.0
TMIN = -0.1
TMAX = 0.4
N_TIMES = int((TMAX - TMIN) * SFREQ) + 1  # 129
CH_NAMES = [f"EEG{i:03d}" for i in range(1, 9)]  # 8 synthetic channels
N_CH = len(CH_NAMES)
EVENT_ID = {"left": 1, "right": 2}

# 10-20 channel names for tests that need a real montage (e.g. TopomapPlot)
TOPO_CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
N_TOPO = len(TOPO_CH_NAMES)

RNG = np.random.default_rng(42)


def _make_epochs(n: int) -> tuple[np.ndarray, list[str]]:
    """Return (data, conditions) with ``n`` epochs split evenly."""
    data = RNG.standard_normal((n, N_CH, N_TIMES)).astype(np.float32) * 1e-6
    half = n // 2
    conds = ["left"] * half + ["right"] * (n - half)
    return data, conds


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


@pytest.fixture(autouse=True)
def _destroy_widgets(qt_app):
    """Tear each test's Qt widgets down deterministically, exactly once each.

    ``widget.close()`` only hides a widget; the C++ object is destroyed later —
    either when Python garbage-collects the wrapper or when the event loop runs
    a deferred deletion. Leaving those two to race is what made this file abort
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
    the file cost O(n^2) overall: 38,912 close/deleteLater pairs across 54
    tests, against 1,313 live widgets. Offscreen raster hid that locally at
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


# Reproducing the abort this file's teardown fixture prevents: replace
# `_destroy_widgets` with a fixture that only drains the event queue,
#
#     @pytest.fixture(autouse=True)
#     def _drain(qt_app):
#         yield
#         qt_app.processEvents()
#
# and `QT_QPA_PLATFORM=offscreen pytest tests/test_viz_plots.py` segfaults on
# every run. Collecting garbage *before* draining, rather than after, is what
# makes the difference — which is how the GC/event-loop race was identified.


@pytest.fixture(scope="module")
def topo_info():
    """mne.Info with standard 10-20 montage for TopomapPlot tests."""
    import mne

    info = mne.create_info(TOPO_CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
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


# ---------------------------------------------------------------------------
# TopoPlot  (scalp-layout ERP display)
# ---------------------------------------------------------------------------


class TestTopoPlot:
    def test_construct(self, qt_app, common_kw):
        from mne_rt.viz.topo_plot import TopoPlot

        w = TopoPlot(**common_kw)
        assert w is not None
        assert w.ch_names == CH_NAMES
        assert w.sfreq == SFREQ
        w.close()

    def test_update_populates_buffer(self, qt_app, common_kw):
        from mne_rt.viz.topo_plot import TopoPlot

        w = TopoPlot(**common_kw)
        data, conds = _make_epochs(10)
        w.update(data, conds)
        qt_app.processEvents()
        assert w._n_per["left"] == 5
        assert w._n_per["right"] == 5
        assert len(w._epoch_buf["left"]) == 5
        assert len(w._epoch_buf["right"]) == 5
        w.close()

    def test_update_single_condition(self, qt_app, common_kw):
        from mne_rt.viz.topo_plot import TopoPlot

        w = TopoPlot(**common_kw)
        data, _ = _make_epochs(6)
        conds = ["left"] * 6
        w.update(data, conds)
        qt_app.processEvents()
        assert w._n_per["left"] == 6
        assert w._n_per["right"] == 0
        w.close()

    def test_times_shape(self, qt_app, common_kw):
        from mne_rt.viz.topo_plot import TopoPlot

        w = TopoPlot(**common_kw)
        assert w._times.shape == (N_TIMES,)
        w.close()


# ---------------------------------------------------------------------------
# ButterflyPlot
# ---------------------------------------------------------------------------


class TestButterflyPlot:
    def test_construct(self, qt_app, common_kw):
        from mne_rt.viz.butterfly_plot import ButterflyPlot

        w = ButterflyPlot(**common_kw)
        assert w is not None
        w.close()

    def test_update_buffer_counts(self, qt_app, common_kw):
        from mne_rt.viz.butterfly_plot import ButterflyPlot

        w = ButterflyPlot(**common_kw)
        data, conds = _make_epochs(8)
        w.update(data, conds)
        qt_app.processEvents()
        assert w._n_per["left"] == 4
        assert w._n_per["right"] == 4
        w.close()

    def test_update_incremental(self, qt_app, common_kw):
        from mne_rt.viz.butterfly_plot import ButterflyPlot

        w = ButterflyPlot(**common_kw)
        data1, c1 = _make_epochs(4)
        w.update(data1, c1)
        data2, c2 = _make_epochs(8)
        w.update(data2, c2)
        qt_app.processEvents()
        assert w._n_per["left"] + w._n_per["right"] == 8
        w.close()


# ---------------------------------------------------------------------------
# CompareEvoked
# ---------------------------------------------------------------------------


class TestCompareEvoked:
    def test_construct(self, qt_app, common_kw):
        from mne_rt.viz.compare_evoked import CompareEvoked

        w = CompareEvoked(**common_kw)
        assert w is not None
        w.close()

    def test_construct_with_channels(self, qt_app, common_kw):
        from mne_rt.viz.compare_evoked import CompareEvoked

        kw = {**common_kw, "channels": CH_NAMES[:3]}
        w = CompareEvoked(**kw)
        assert w._disp_channels == CH_NAMES[:3]
        w.close()

    def test_update_buffer(self, qt_app, common_kw):
        from mne_rt.viz.compare_evoked import CompareEvoked

        w = CompareEvoked(**common_kw)
        data, conds = _make_epochs(10)
        w.update(data, conds)
        qt_app.processEvents()
        assert w._n_per["left"] == 5
        assert w._n_per["right"] == 5
        w.close()

    def test_no_channel_cap(self, qt_app, common_kw):
        """All 8 channels selectable — no artificial cap."""
        from mne_rt.viz.compare_evoked import CompareEvoked

        kw = {**common_kw, "channels": CH_NAMES}  # all 8
        w = CompareEvoked(**kw)
        assert len(w._disp_channels) == N_CH
        w.close()


# ---------------------------------------------------------------------------
# TFRPlot
# ---------------------------------------------------------------------------


class TestTFRPlot:
    def test_construct_defaults(self, qt_app, common_kw):
        from mne_rt.viz.tfr_plot import TFRPlot

        w = TFRPlot(**common_kw)
        assert w is not None
        w.close()

    def test_construct_custom_freqs(self, qt_app, common_kw):
        from mne_rt.viz.tfr_plot import TFRPlot

        freqs = np.arange(8.0, 30.0, 2.0)
        w = TFRPlot(**common_kw, freqs=freqs, channels=CH_NAMES[:2])
        assert w._freqs is not None
        w.close()

    def test_clip_freqs_removes_short_epoch_artifacts(self, qt_app, common_kw):
        """_clip_freqs must not error on 0.5-second epochs."""
        from mne_rt.viz.tfr_plot import TFRPlot

        freqs = np.arange(4.0, 80.0, 2.0)
        w = TFRPlot(**common_kw, freqs=freqs, channels=CH_NAMES[:1])
        # After clipping, all remaining freqs must be < Nyquist and
        # n_cycles must be strictly positive
        assert w._n_cycles is None or np.all(np.asarray(w._n_cycles) > 0)
        w.close()

    def test_update_stores_data(self, qt_app, common_kw):
        from mne_rt.viz.tfr_plot import TFRPlot

        w = TFRPlot(**common_kw, channels=CH_NAMES[:2])
        data, conds = _make_epochs(6)
        w.update(data, conds)
        qt_app.processEvents()
        assert w._latest_data is not None
        assert w._latest_data.shape == data.shape
        assert len(w._latest_conds) == 6
        assert w._latest_conds.count("left") == 3
        assert w._latest_conds.count("right") == 3
        w.close()


# ---------------------------------------------------------------------------
# RawPlot
# ---------------------------------------------------------------------------


class TestRawPlot:
    def test_construct(self, qt_app):
        from mne_rt.viz.raw_plot import RawPlot

        w = RawPlot(CH_NAMES, sfreq=SFREQ)
        assert w._n_ch == N_CH
        assert w._sfreq == SFREQ
        assert w._buf.shape == (N_CH, int(SFREQ * 10.0))
        w.close()

    def test_push_fills_buffer(self, qt_app):
        from mne_rt.viz.raw_plot import RawPlot

        w = RawPlot(CH_NAMES, sfreq=SFREQ, time_window=5.0)
        data = RNG.standard_normal((N_CH, 100)).astype(np.float32) * 1e-6
        w.push(data)
        w._flush_data_queue()  # drain queue synchronously (timer-based in production)
        assert np.any(w._buf != 0.0)
        w.close()

    def test_push_respects_pause(self, qt_app):
        from mne_rt.viz.raw_plot import RawPlot

        w = RawPlot(CH_NAMES, sfreq=SFREQ)
        w._paused = True
        data = RNG.standard_normal((N_CH, 50)).astype(np.float32) * 1e-6
        w.push(data)
        qt_app.processEvents()
        assert not np.any(w._buf != 0.0)
        w.close()

    def test_filter_apply_sets_sos_and_zi(self, qt_app):
        from mne_rt.viz.raw_plot import RawPlot

        w = RawPlot(CH_NAMES, sfreq=SFREQ)
        w._cmb_filter.setCurrentIndex(3)  # Band-pass
        w._flo_spin.setValue(1.0)
        w._fhi_spin.setValue(40.0)
        w._apply_filter_settings()
        assert w._filter_sos is not None
        assert w._filter_zi is not None
        assert w._filter_zi.shape == (N_CH, w._filter_sos.shape[0], 2)
        w.close()

    def test_reref_average_applied(self, qt_app):
        from mne_rt.viz.raw_plot import RawPlot

        w = RawPlot(CH_NAMES, sfreq=SFREQ)
        w._cmb_reref.setCurrentIndex(1)  # Average
        w._apply_reref_settings()
        assert w._reref_type == "average"
        # All-ones input after average ref → every channel should be ~0
        data = np.ones((N_CH, 64), dtype=np.float64) * 1e-6
        w.push(data)
        w._flush_data_queue()
        stored = w._buf[:, -64:]
        assert np.allclose(stored, 0.0, atol=1e-12)
        w.close()

    def test_reref_channel(self, qt_app):
        from mne_rt.viz.raw_plot import RawPlot

        w = RawPlot(CH_NAMES, sfreq=SFREQ)
        w._cmb_reref.setCurrentIndex(4)  # Channel
        w._reref_ch_cmb.setCurrentIndex(0)  # first channel as reference
        w._apply_reref_settings()
        assert w._reref_type == "channel"
        assert w._reref_idx == 0
        w.close()


# ---------------------------------------------------------------------------
# TopomapPlot
# ---------------------------------------------------------------------------


class TestTopomapPlot:
    def test_construct(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        assert w._sfreq == SFREQ
        assert len(w._bands) == len(w._visible_bands)
        w.close()

    def test_push_updates_last_data(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        data = RNG.standard_normal((N_TOPO, int(SFREQ))).astype(np.float32) * 1e-6
        w.push(data)
        qt_app.processEvents()
        assert w._last_data is not None
        assert w._last_data.shape == data.shape
        w.close()

    def test_pause_inhibits_update(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        w._paused = True
        data = RNG.standard_normal((N_TOPO, int(SFREQ))).astype(np.float32) * 1e-6
        w.push(data)
        qt_app.processEvents()
        assert w._last_data is None
        w.close()

    def test_custom_band_added(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        initial = len(w._bands)
        w._custom_flo.setValue(15.0)
        w._custom_fhi.setValue(25.0)
        w._add_custom_band()
        assert len(w._bands) == initial + 1
        assert "Custom 15.0–25.0 Hz" in w._bands
        assert "Custom 15.0–25.0 Hz" in w._visible_bands
        w.close()

    def test_toggle_band_visibility(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        first_band = list(w._bands.keys())[0]
        # Disable first band
        w._band_checks[first_band].setChecked(False)
        assert first_band not in w._visible_bands
        # Re-enable
        w._band_checks[first_band].setChecked(True)
        assert first_band in w._visible_bands
        w.close()


# ---------------------------------------------------------------------------
# EpochPlot
# ---------------------------------------------------------------------------


class TestEpochPlot:
    def test_construct(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ, tmin=-0.1, tmax=0.5)
        assert w._n_ch == N_CH
        assert w._sfreq == SFREQ
        assert w._tmin == -0.1
        assert w._tmax == 0.5
        assert w._buf.shape == (N_CH, int(SFREQ * 10.0))
        w.close()

    def test_push_fills_buffer(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ, time_window=5.0)
        data = RNG.standard_normal((N_CH, 100)).astype(np.float32) * 1e-6
        w.push(data)
        w._process_pending()  # drain queue synchronously (timer-based in production)
        assert np.any(w._buf != 0.0)
        assert w._total_pushed == 100
        w.close()

    def test_push_respects_pause(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ)
        w._paused = True
        data = RNG.standard_normal((N_CH, 50)).astype(np.float32) * 1e-6
        w.push(data)
        assert not np.any(w._buf != 0.0)
        assert w._total_pushed == 0
        w.close()

    def test_push_trigger_adds_entry(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ, event_id={"stim": 1})
        data = RNG.standard_normal((N_CH, 64)).astype(np.float32) * 1e-6
        w.push(data)
        w.push_trigger(code=1)
        w._process_pending()  # drain queue synchronously
        assert len(w._triggers) == 1
        trig_abs, trig_code, epoch_id = w._triggers[0]
        assert trig_abs == 64
        assert trig_code == 1
        assert epoch_id == 0
        w.close()

    def test_apply_epoch_window(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ)
        w._tmin_spin.setValue(-0.2)
        w._tmax_spin.setValue(0.8)
        w._apply_epoch_window()
        assert w._tmin == -0.2
        assert w._tmax == 0.8
        w.close()

    def test_clear_triggers(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ)
        data = RNG.standard_normal((N_CH, 32)).astype(np.float32) * 1e-6
        w.push(data)
        w.push_trigger(1)
        w.push_trigger(2)
        w._process_pending()  # drain queue so triggers land in _triggers
        assert len(w._triggers) == 2
        w._clear_triggers()
        assert len(w._triggers) == 0
        w.close()

    def test_epoch_ids_increment_and_survive_clear_triggers(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ)
        w.push_trigger(1)
        w.push_trigger(1)
        w._process_pending()
        ids = [t[2] for t in w._triggers]
        assert ids == [0, 1]
        # next_epoch_id keeps counting monotonically even after clearing
        # the (now orphaned) trigger records.
        w._clear_triggers()
        w.push_trigger(1)
        w._process_pending()
        assert w._triggers[0][2] == 2
        w.close()

    def test_toggle_bad_epoch_marks_and_unmarks(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ)
        w.push_trigger(1)
        w._process_pending()
        assert w.bad_epoch_ids == []
        assert not w.is_epoch_bad(0)

        w._toggle_bad_epoch(0)
        assert w.bad_epoch_ids == [0]
        assert w.is_epoch_bad(0)

        w._toggle_bad_epoch(0)
        assert w.bad_epoch_ids == []
        assert not w.is_epoch_bad(0)
        w.close()

    def test_bad_epoch_renders_in_red(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ, time_window=5.0)
        w.push_trigger(1)
        w._process_pending()
        assert len(w._visible_epoch_spans) == 1
        assert w._visible_epoch_spans[0][0] == 0

        w._toggle_bad_epoch(0)
        # After marking bad, the trigger line for epoch 0 is drawn in the
        # dedicated bad-epoch red, not the default event-code colour.
        trig_lines = [
            item
            for item in w._epoch_overlay_items
            if isinstance(item, pg.InfiniteLine) and item.angle == 90
        ]
        assert any(line.pen.color().name() == w._BAD_EPOCH_COLOR for line in trig_lines)
        w.close()

    def test_epoch_id_at_hit_tests_visible_spans(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ, tmin=-0.1, tmax=0.5, time_window=5.0)
        # Trigger fired "now" (at the right edge); pushing data afterwards
        # scrolls it left to the middle of the visible window, at x0 = 2.5 s.
        w.push_trigger(1)
        n_shift = int(2.5 * SFREQ)
        data = RNG.standard_normal((N_CH, n_shift)).astype(np.float32) * 1e-6
        w.push(data)
        w._process_pending()
        assert w._visible_epoch_spans[0] == (0, pytest.approx(2.4), pytest.approx(3.0))
        assert w._epoch_id_at(2.7) == 0
        assert w._epoch_id_at(4.9) is None
        w.close()

    def test_clear_bad_epochs_resets_marks_but_keeps_triggers(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ)
        w.push_trigger(1)
        w.push_trigger(1)
        w._process_pending()
        w._toggle_bad_epoch(0)
        w._toggle_bad_epoch(1)
        assert w.bad_epoch_ids == [0, 1]

        w._clear_bad_epochs()
        assert w.bad_epoch_ids == []
        assert len(w._triggers) == 2
        w.close()


# ---------------------------------------------------------------------------
# NFPlot  (scrolling real-time neurofeedback signal monitor)
# ---------------------------------------------------------------------------

NF_MODALITIES = ["sensor_power", "erd_ers"]
NF_SCALES = {"sensor_power": 1e-12, "erd_ers": 1.0}


def _reward_visible(w, i):
    """Whether any pooled reward-span region is currently shown for modality ``i``."""
    return any(r.isVisible() for r in w._reward_regions[i])


class TestNFPlot:
    def test_instanced_modalities_render(self, qt_app):
        """An instance inherits its base's scale, label and unit."""
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(
            ["sensor_power@alpha", "sensor_power@theta"],
            {"sensor_power": 1e-12},  # only the base has an entry
            sfreq=30,
        )
        w.push([3.2e-13, 1.1e-13])  # must not KeyError on the missing keys
        assert w._n == 2
        assert w._buf[0, -1] != w._buf[1, -1]
        labels = [pi.getAxis("left").labelText for pi in w._plots]
        assert labels == ["Sensor Power (alpha)", "Sensor Power (theta)"]
        w.close()

    def test_modality_without_a_scale_entry_renders(self, qt_app):
        """instantaneous_phase is a real modality that had no scales_dict entry."""
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(["instantaneous_phase"], {}, sfreq=30)
        w.push([1.5])
        w.close()

    def test_construct(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        assert w._n == 2
        assert w._buf.shape == (2, max(int(30 * 30.0), 30))
        assert w._reward_buf.shape == w._buf.shape
        assert len(w._threshold_lines) == 2
        assert all(not line.isVisible() for line in w._threshold_lines)
        assert len(w._reward_regions) == 2
        assert all(pool == [] for pool in w._reward_regions)
        w.close()

    def test_push_without_thresholds_updates_traces_only(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30, display_smoothing=1.0)
        w.push([3.2e-13, 0.5])
        assert w._buf[0, -1] != 0.0
        assert all(not line.isVisible() for line in w._threshold_lines)
        w.close()

    def test_push_with_threshold_shows_line_at_correct_position(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w.push([3.2e-13, 0.5], thresholds=[2.0e-13, None])
        assert w._threshold_lines[0].isVisible()
        assert w._threshold_lines[0].value() == pytest.approx(0.2)  # 2e-13 / 1e-12
        assert not w._threshold_lines[1].isVisible()
        assert w._threshold_active == [True, False]
        w.close()

    def test_threshold_label_shows_physical_value_not_normalised_position(self, qt_app):
        """InfiniteLine's ``{value}`` placeholder would auto-fill from the
        line's *position* (the normalised/rescaled plot coordinate), not
        the physical threshold -- push() must override it with the real
        value, e.g. ``2e-13``, not the normalised ``0.2``.
        """
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w.push([3.2e-13, 0.5], thresholds=[2.0e-13, None])
        assert w._threshold_lines[0].label.toPlainText() == "thr 2e-13 V²/Hz"
        w.close()

    def test_push_none_threshold_hides_previously_shown_line(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w.push([3.2e-13, 0.5], thresholds=[2.0e-13, None])
        assert w._threshold_lines[0].isVisible()
        w.push([3.2e-13, 0.5], thresholds=[None, None])
        assert not w._threshold_lines[0].isVisible()
        assert w._threshold_active == [False, False]
        w.close()

    def test_push_nan_threshold_hides_line(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w.push([3.2e-13, 0.5], thresholds=[float("nan"), None])
        assert not w._threshold_lines[0].isVisible()
        assert w._threshold_active == [False, False]
        w.close()

    def test_threshold_visibility_toggle(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w.push([3.2e-13, 0.5], thresholds=[2.0e-13, None])
        assert w._threshold_lines[0].isVisible()

        w._set_threshold_visible(False)
        assert not w._threshold_lines[0].isVisible()

        w._set_threshold_visible(True)
        assert w._threshold_lines[0].isVisible()
        # Modality with no active threshold must stay hidden even when re-enabled.
        assert not w._threshold_lines[1].isVisible()
        w.close()

    def test_push_respects_pause(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w._paused = True
        w.push([3.2e-13, 0.5], thresholds=[2.0e-13, None])
        assert not w._threshold_lines[0].isVisible()
        assert np.all(w._buf == 0.0)
        w.close()

    def test_push_without_rewards_leaves_spans_untouched(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w.push([3.2e-13, 0.5])
        assert w._reward_active == [False, False]
        assert all(pool == [] for pool in w._reward_regions)
        w.close()

    def test_push_with_reward_shows_span(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w.push([3.2e-13, 0.5], rewards=[True, None])
        assert _reward_visible(w, 0)
        assert w._reward_buf[0, -1] == 1
        assert not _reward_visible(w, 1)
        assert w._reward_active == [True, False]
        w.close()

    def test_push_reward_false_keeps_span_active_but_off(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w.push([3.2e-13, 0.5], rewards=[False, None])
        # A modality with a driving protocol stays "active" (the span
        # exists) even on windows where the reward condition isn't met --
        # only the most recent sample in the ring buffer is unset.
        assert w._reward_active == [True, False]
        assert w._reward_buf[0, -1] == 0
        w.close()

    def test_reward_visibility_toggle(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w.push([3.2e-13, 0.5], rewards=[True, None])
        assert _reward_visible(w, 0)

        w._set_reward_visible(False)
        assert not _reward_visible(w, 0)

        w._set_reward_visible(True)
        assert _reward_visible(w, 0)
        # Modality with no active reward info must stay hidden even when re-enabled.
        assert not _reward_visible(w, 1)
        w.close()

    def test_push_respects_pause_with_rewards(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        w._paused = True
        w.push([3.2e-13, 0.5], rewards=[True, None])
        assert w._reward_active == [False, False]
        assert np.all(w._reward_buf == 0)
        w.close()

    def test_auto_range_includes_threshold_line(self, qt_app):
        """A threshold far outside the trace's span must not be auto-ranged
        out of view (regression: _auto_range used to only look at _buf)."""
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30, display_smoothing=1.0)
        # Trace sits near 0.2-0.3; threshold is far above that span.
        for _ in range(5):
            w.push([2.5e-13, 0.5], thresholds=[9.0e-13, None])
        w._auto_range()
        lo, hi = w._plots[0].viewRange()[1]
        thr_val = w._threshold_lines[0].value()
        assert lo <= thr_val <= hi
        w.close()

    def test_closed_signal_emitted_on_close(self, qt_app):
        from mne_rt.viz.nf_plot import NFPlot

        w = NFPlot(NF_MODALITIES, NF_SCALES, sfreq=30)
        received = []
        w.closed.connect(lambda: received.append(True))
        w.close()
        assert received == [True]


# ---------------------------------------------------------------------------
# Some regressions
# ---------------------------------------------------------------------------


def test_butterfly_colours_frontal_channels_as_frontal(qt_app, common_kw):
    """MNE layouts grow y towards the nose; taken as is, Fz got the occipital
    colour and Oz the frontal one."""
    from mne_rt.viz.butterfly_plot import ButterflyPlot

    names = ["Fz", "Cz", "Pz", "Oz"]
    w = ButterflyPlot(**{**common_kw, "ch_names": names})
    y = dict(zip(names, (pos[1] for pos in w._norm_pos)))
    assert y["Fz"] < y["Cz"] < y["Pz"] < y["Oz"]  # 0 = frontal, 1 = occipital
    w.close()


@pytest.mark.parametrize("cls_name", ["ButterflyPlot", "TopoPlot"])
def test_y_scale_is_a_gain(qt_app, common_kw, cls_name):
    """×2 must show the traces twice as tall, i.e. halve the y range."""
    import importlib

    module = {"ButterflyPlot": "butterfly_plot", "TopoPlot": "topo_plot"}[cls_name]
    cls = getattr(importlib.import_module(f"mne_rt.viz.{module}"), cls_name)
    w = cls(**common_kw)
    w.update(*_make_epochs(6))
    plot = next(iter(w._cond_plots.values())) if cls_name == "ButterflyPlot" else w._plots[0]
    w._scale_sl.setValue(5)  # ×1
    w._apply_y_range()
    span_1 = np.diff(plot.viewRange()[1])[0]
    w._scale_sl.setValue(10)  # ×2
    w._apply_y_range()
    span_2 = np.diff(plot.viewRange()[1])[0]
    assert span_2 == pytest.approx(span_1 / 2, rel=0.05)
    w.close()


def test_topo_sem_band_surrounds_the_drawn_curve(qt_app, common_kw):
    """The band was drawn around the unsmoothed mean, and raised a shape error
    for epochs whose length is interpolated to the display grid."""
    from mne_rt.viz.topo_plot import TopoPlot

    w = TopoPlot(**common_kw)
    w._show_sem = True
    w._smooth_ms = 20.0
    data, conds = _make_epochs(6)
    w.update(data[:, :, :100], conds)  # shorter than the display grid
    cond = conds[0]
    _, y = w._curves[cond][0].getData()
    _, upper = w._sem_upper[cond][0].getData()
    _, lower = w._sem_lower[cond][0].getData()
    np.testing.assert_allclose((upper + lower) / 2, y, rtol=1e-6, atol=1e-30)
    w.close()


def test_topo_reset_survives_the_next_update(qt_app, common_kw):
    """update() is given every epoch so far, so Reset was undone by the next
    trial."""
    from mne_rt.viz.topo_plot import TopoPlot

    w = TopoPlot(**common_kw)
    data, conds = _make_epochs(8)
    w.update(data[:6], conds[:6])
    w._reset_epochs()
    w.update(data, conds)  # the same six, plus two new ones
    assert sum(w._n_per.values()) == 2
    w.close()


def test_raw_filter_stays_on_after_a_reset(qt_app):
    """Clearing the buffer set the filter state to None, which push() reads as
    "no filter", while the status label still showed it as on. The state is
    kept, not zeroed: the stream goes on, and a zeroed state rings on DC."""
    from mne_rt.viz.raw_plot import RawPlot

    w = RawPlot(CH_NAMES, sfreq=SFREQ)
    w._cmb_filter.setCurrentIndex(3)  # band-pass
    w._apply_filter_settings()
    w.push(RNG.standard_normal((N_CH, 64)) * 1e-6)
    w._flush_data_queue()
    zi = w._filter_zi.copy()
    w._clear()
    assert w._filter_sos is not None
    np.testing.assert_array_equal(w._filter_zi, zi)
    w.close()


@pytest.mark.parametrize("cls_name", ["RawPlot", "EpochPlot"])
def test_ring_buffer_takes_empty_and_oversized_chunks(qt_app, cls_name):
    """`buf[:, -0:]` is the whole buffer, so an empty chunk raised, and so did
    one longer than the buffer."""
    if cls_name == "RawPlot":
        from mne_rt.viz.raw_plot import RawPlot as cls

        drain = "_flush_data_queue"
    else:
        from mne_rt.viz.epoch_plot import EpochPlot as cls

        drain = "_process_pending"
    w = cls(CH_NAMES, sfreq=SFREQ, time_window=1.0)
    n_buf = w._buf.shape[1]
    big = RNG.standard_normal((N_CH, n_buf + 10))
    w.push(big[:, :0])
    w.push(big)
    getattr(w, drain)()
    np.testing.assert_allclose(w._buf, big[:, -n_buf:], rtol=1e-6)
    assert w._total_pushed == n_buf + 10
    w.close()


def test_raw_bad_segment_times_while_the_buffer_fills(qt_app):
    """With 3 s of data in a 10 s window the data sits at x = 7-10 s; a click at
    x = 8 s is 1 s into the recording, not 8 s."""
    from mne_rt.viz.raw_plot import RawPlot

    w = RawPlot(CH_NAMES, sfreq=SFREQ, time_window=10.0)
    w.push(np.zeros((N_CH, int(3 * SFREQ))))
    w._flush_data_queue()
    w._on_bad_seg_click(8.0)
    assert w._bad_seg_click1 == pytest.approx(1.0)
    w.close()


def test_epoch_clear_removes_the_trigger_overlays(qt_app):
    from mne_rt.viz.epoch_plot import EpochPlot

    w = EpochPlot(CH_NAMES, sfreq=SFREQ)
    w.push(np.zeros((N_CH, int(SFREQ))))
    w.push_trigger(1)
    w.push(np.zeros((N_CH, int(SFREQ))))
    w._process_pending()
    drawn = list(w._epoch_overlay_items)
    assert drawn
    w._clear()
    assert not any(item in w._pi.items for item in drawn)
    w.close()


def test_epoch_triggers_are_dropped_while_paused(qt_app):
    """Their data is dropped while paused, so they piled up at one stale index."""
    from mne_rt.viz.epoch_plot import EpochPlot

    w = EpochPlot(CH_NAMES, sfreq=SFREQ)
    w._toggle_pause(True)
    w.push_trigger(1)
    assert not w._pending
    w.close()


def test_nf_trace_recovers_after_a_nan(qt_app):
    """One NaN made the smoothed value NaN for good."""
    from mne_rt.viz.nf_plot import NFPlot

    w = NFPlot(["erd_ers"], {"erd_ers": 1.0}, sfreq=30, display_smoothing=0.3)
    w.push([1.0])
    assert w._buf[0, -1] == pytest.approx(1.0)  # starts at the value, not at 0
    w.push([np.nan])
    assert np.isnan(w._buf[0, -1])  # shown as a gap
    w.push([2.0])
    assert w._buf[0, -1] == pytest.approx(0.3 * 2.0 + 0.7 * 1.0)
    w._auto_range()
    assert np.all(np.isfinite(w._plots[0].viewRange()[1]))
    w.close()


def test_nf_channel_scale_rescales_the_history(qt_app):
    """+/- used to apply only to values arriving after the click."""
    from mne_rt.viz.nf_plot import NFPlot

    w = NFPlot(["erd_ers"], {"erd_ers": 1.0}, sfreq=30, display_smoothing=1.0)
    for v in (1.0, 2.0, 3.0):
        w.push([v])
    w._scale_up(0)
    np.testing.assert_allclose(w._buf[0, -3:], [2.0, 4.0, 6.0])
    w.close()


def test_tfr_baseline_is_the_one_asked_for(qt_app, common_kw):
    from mne_rt.viz.tfr_plot import TFRPlot

    w = TFRPlot(**{**common_kw, "baseline": (-0.05, 0.0)}, channels=CH_NAMES[:1])
    t = w._times_dec[w._baseline_mask()]
    assert t.size and t.min() >= -0.05 and t.max() <= 0.0
    w.close()


def test_tfr_reruns_for_epochs_that_arrive_mid_computation(qt_app, common_kw):
    """Epochs arriving while a TFR was computed were never used, and two close
    updates could start two workers."""
    import threading
    import time

    from mne_rt.viz.tfr_plot import TFRPlot

    w = TFRPlot(**common_kw, channels=CH_NAMES[:1])
    started, release, calls, running = threading.Event(), threading.Event(), [], []

    def slow_tfr(data, conds, params=None):
        running.append(1)
        assert len(running) == 1  # one worker at a time
        calls.append(len(conds))
        started.set()
        release.wait(5)
        running.pop()
        return {}

    w._compute_tfr = slow_tfr
    data, conds = _make_epochs(6)
    w.update(data[:4], conds[:4])
    assert started.wait(5)
    w.update(data, conds)  # arrives while the first pass is running
    release.set()
    deadline = time.monotonic() + 5
    while w._computing and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == [4, 6]
    w.close()


def test_tfr_redraw_ignores_a_result_for_other_channels(qt_app, common_kw):
    """A result computed before the channel selection changed was indexed
    with the new channel count, raising IndexError in the Qt slot."""
    from mne_rt.viz.tfr_plot import TFRPlot

    w = TFRPlot(**common_kw, channels=CH_NAMES[:1])
    shape = (1, len(w._freqs), w._n_t_dec)
    w._tfr_pass = ({c: np.ones(shape) for c in w._conditions}, w._params(), [])
    w._display_chs = CH_NAMES[:2]
    w._rebuild_grid()
    w._redraw(0)  # must not raise
    w.close()


def test_tfr_colour_levels_redraw_without_recomputing(qt_app, common_kw):
    from mne_rt.viz.tfr_plot import TFRPlot

    w = TFRPlot(**common_kw, channels=CH_NAMES[:1])
    scheduled = []
    w._schedule = lambda: scheduled.append(True)
    w._auto_lvl_chk.setChecked(False)
    w._on_vmin_change(-1.0)
    w._on_vmax_change(1.0)
    assert scheduled == []
    w.close()


def test_tfr_computes_only_the_displayed_channels(qt_app, common_kw):
    """The other channels were transformed and thrown away; NaN in one of them
    shows they are no longer touched."""
    from mne_rt.viz.tfr_plot import TFRPlot

    w = TFRPlot(**common_kw, channels=CH_NAMES[:1])
    data, conds = _make_epochs(4)
    data[:, 1:] = np.nan
    result = w._compute_tfr(data, conds)
    assert all(np.all(np.isfinite(p)) for p in result.values())
    w.close()


def test_topo_reset_does_not_hide_a_new_run(qt_app, common_kw):
    """A new run whose first update is longer than the reset point lost its
    first epochs to the old reset."""
    from mne_rt.viz.topo_plot import TopoPlot

    w = TopoPlot(**common_kw)
    data, conds = _make_epochs(4)
    w.update(data, conds)
    w._reset_epochs()  # hides 4 epochs of this run
    new, new_conds = _make_epochs(5)  # a new run, first update has 5
    w.update(new, new_conds)
    assert sum(w._n_per.values()) == 5
    w.close()


def test_nf_rescale_keeps_the_threshold_label(qt_app):
    """setPos() resets an InfiniteLine label to its format string ("thr")."""
    from mne_rt.viz.nf_plot import NFPlot

    w = NFPlot(["erd_ers"], {"erd_ers": 1.0}, sfreq=30, display_smoothing=1.0)
    w.push([1.0], thresholds=[3.2])
    label = w._threshold_lines[0].label.textItem.toPlainText()
    assert label.startswith("thr 3.2")
    w._scale_up(0)
    assert w._threshold_lines[0].label.textItem.toPlainText() == label
    w.close()


def test_brain_display_smoothing_survives_a_nan():
    from mne_rt.viz.brain_plot import _ema_step

    ema = np.array([1.0, np.nan, 1.0])
    out = _ema_step(ema, np.array([np.nan, 2.0, 2.0]), 0.5)
    np.testing.assert_allclose(out, [1.0, 2.0, 1.5])
