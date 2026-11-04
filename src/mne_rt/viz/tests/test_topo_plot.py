"""TopoPlot: ERPs laid out on the scalp."""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.viz.tests._helpers import CH_NAMES, N_TIMES, SFREQ, _make_epochs

pytestmark = pytest.mark.usefixtures("_destroy_widgets")


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


def test_topo_reset_does_not_hide_a_new_run(qt_app, common_kw):
    """A new run whose first update is longer than the reset point lost its
    first epochs to the old reset."""
    from mne_rt.viz.topo_plot import TopoPlot

    w = TopoPlot(**common_kw)
    data, conds = _make_epochs(4)
    w.update(data, conds)
    w._reset_epochs()  # hides 4 epochs of this run
    new, new_conds = _make_epochs(5, seed=1)  # a new run, first update has 5
    w.update(new, new_conds)
    assert sum(w._n_per.values()) == 5
    w.close()
