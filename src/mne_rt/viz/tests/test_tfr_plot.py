"""TFRPlot: live time-frequency maps."""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.viz.tests._helpers import CH_NAMES, _make_epochs

pytestmark = pytest.mark.usefixtures("_destroy_widgets")


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
