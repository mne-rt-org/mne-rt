"""ButterflyPlot: butterfly ERP display coloured by scalp region."""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.viz.tests._helpers import _make_epochs

pytestmark = pytest.mark.usefixtures("_destroy_widgets")


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
