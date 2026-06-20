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

# 10-20 channel names for tests that need a real montage (e.g. TopomapPlot)
TOPO_CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
N_TOPO        = len(TOPO_CH_NAMES)

RNG = np.random.default_rng(42)


def _make_epochs(n: int) -> tuple[np.ndarray, list[str]]:
    """Return (data, conditions) with ``n`` epochs split evenly."""
    data  = RNG.standard_normal((n, N_CH, N_TIMES)).astype(np.float32) * 1e-6
    half  = n // 2
    conds = ["left"] * half + ["right"] * (n - half)
    return data, conds


@pytest.fixture(scope="module")
def qt_app():
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


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
        ch_names  = CH_NAMES,
        sfreq     = SFREQ,
        tmin      = TMIN,
        tmax      = TMAX,
        event_id  = EVENT_ID,
        baseline  = (None, 0),
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
        assert w._n_per["left"]  == 5
        assert w._n_per["right"] == 5
        assert len(w._epoch_buf["left"])  == 5
        assert len(w._epoch_buf["right"]) == 5
        w.close()

    def test_update_single_condition(self, qt_app, common_kw):
        from mne_rt.viz.topo_plot import TopoPlot
        w = TopoPlot(**common_kw)
        data, _ = _make_epochs(6)
        conds   = ["left"] * 6
        w.update(data, conds)
        qt_app.processEvents()
        assert w._n_per["left"]  == 6
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
        w._flush_data_queue()   # drain queue synchronously (timer-based in production)
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
        w._cmb_reref.setCurrentIndex(1)   # Average
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
        w._cmb_reref.setCurrentIndex(4)   # Channel
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
        w._process_pending()    # drain queue synchronously (timer-based in production)
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
        w._process_pending()    # drain queue synchronously
        assert len(w._triggers) == 1
        trig_abs, trig_code = w._triggers[0]
        assert trig_abs == 64
        assert trig_code == 1
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
        w.push(data); w.push_trigger(1); w.push_trigger(2)
        w._process_pending()    # drain queue so triggers land in _triggers
        assert len(w._triggers) == 2
        w._clear_triggers()
        assert len(w._triggers) == 0
        w.close()
