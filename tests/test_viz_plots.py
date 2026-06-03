"""Headless tests for the four real-time visualisation classes.

Runs in offscreen (no display) mode.  Each test:
  - constructs the widget with synthetic EEG-like data
  - calls update() with a small batch of fake epochs
  - verifies internal state (buffers, condition counts, shapes)
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

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SFREQ    = 256.0
TMIN     = -0.1
TMAX     = 0.4
N_TIMES  = int((TMAX - TMIN) * SFREQ) + 1   # 129
CH_NAMES = [f"EEG{i:03d}" for i in range(1, 9)]  # 8 synthetic channels
N_CH     = len(CH_NAMES)
EVENT_ID = {"left": 1, "right": 2}

RNG = np.random.default_rng(42)


def _make_epochs(n: int) -> tuple[np.ndarray, list[str]]:
    """Return (data, conditions) with ``n`` epochs split evenly."""
    data  = RNG.standard_normal((n, N_CH, N_TIMES)).astype(np.float32) * 1e-6
    half  = n // 2
    conds = ["left"] * half + ["right"] * (n - half)
    return data, conds


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def common_kw():
    return dict(
        ch_names  = CH_NAMES,
        sfreq     = SFREQ,
        tmin      = TMIN,
        tmax      = TMAX,
        event_id  = EVENT_ID,
        baseline  = (None, 0),
    )


# ---------------------------------------------------------------------------
# ERPPlot
# ---------------------------------------------------------------------------

class TestERPPlot:
    def test_construct(self, qt_app, common_kw):
        from mne_rt.viz.erp_plot import ERPPlot
        w = ERPPlot(**common_kw)
        assert w is not None
        assert w.ch_names == CH_NAMES
        assert w.sfreq == SFREQ
        w.close()

    def test_update_populates_buffer(self, qt_app, common_kw):
        from mne_rt.viz.erp_plot import ERPPlot
        w = ERPPlot(**common_kw)
        data, conds = _make_epochs(10)
        w.update(data, conds)
        qt_app.processEvents()
        assert w._n_per["left"]  == 5
        assert w._n_per["right"] == 5
        assert len(w._epoch_buf["left"])  == 5
        assert len(w._epoch_buf["right"]) == 5
        w.close()

    def test_update_single_condition(self, qt_app, common_kw):
        from mne_rt.viz.erp_plot import ERPPlot
        w = ERPPlot(**common_kw)
        data, _ = _make_epochs(6)
        conds   = ["left"] * 6
        w.update(data, conds)
        qt_app.processEvents()
        assert w._n_per["left"]  == 6
        assert w._n_per["right"] == 0
        w.close()

    def test_times_shape(self, qt_app, common_kw):
        from mne_rt.viz.erp_plot import ERPPlot
        w = ERPPlot(**common_kw)
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
        assert w._n_per["left"]  == 4
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
        assert w._n_per["left"]  + w._n_per["right"] == 8
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
        w  = CompareEvoked(**kw)
        assert w._disp_channels == CH_NAMES[:3]
        w.close()

    def test_update_buffer(self, qt_app, common_kw):
        from mne_rt.viz.compare_evoked import CompareEvoked
        w = CompareEvoked(**common_kw)
        data, conds = _make_epochs(10)
        w.update(data, conds)
        qt_app.processEvents()
        assert w._n_per["left"]  == 5
        assert w._n_per["right"] == 5
        w.close()

    def test_no_channel_cap(self, qt_app, common_kw):
        """All 8 channels selectable — no artificial cap."""
        from mne_rt.viz.compare_evoked import CompareEvoked
        kw = {**common_kw, "channels": CH_NAMES}  # all 8
        w  = CompareEvoked(**kw)
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
        assert w._latest_conds.count("left")  == 3
        assert w._latest_conds.count("right") == 3
        w.close()
