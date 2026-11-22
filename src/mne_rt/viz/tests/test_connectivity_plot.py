"""ConnectivityPlot: the pairs a connectivity modality computes, live."""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.viz.connectivity_plot import circle_positions, value_range

pytestmark = pytest.mark.usefixtures("_destroy_widgets")

PAIRS = [("C3", "C4"), ("F3", "F4")]


def _snap(edges, labels=PAIRS, metric="coh", space="sensor", nodes=None, signed=True):
    edges = np.asarray(edges, dtype=float)
    return dict(
        value=float(edges.mean()),
        edges=edges,
        labels=labels,
        metric=metric,
        signed=signed,
        space=space,
        nodes=nodes,
    )


def _info():
    import mne

    from mne_rt._montage import make_standard_montage

    names = ["F3", "F4", "C3", "C4", "O1", "O2"]
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage(make_standard_montage("colin27_1020"))
    return names, info


def test_one_tab_per_connectivity_modality(qt_app):
    from mne_rt.viz import ConnectivityPlot

    names, info = _info()
    w = ConnectivityPlot(ch_names=names, info=info)
    w.push({"sensor_connectivity": _snap([0.9, 0.2])})
    w.push({"sensor_connectivity": _snap([0.8, 0.3]), "sensor_graph@beta": _snap([0.1, 0.2])})
    assert [w._tabs.tabText(i) for i in range(w._tabs.count())] == [
        "sensor_connectivity",
        "sensor_graph@beta",
    ]


def test_a_stronger_pair_is_drawn_thicker(qt_app):
    from mne_rt.viz import ConnectivityPlot

    names, info = _info()
    w = ConnectivityPlot(ch_names=names, info=info)
    w.push({"sensor_connectivity": _snap([0.9, 0.1])})
    strong, weak = w._panels["sensor_connectivity"]._edges
    assert strong.opts["pen"].widthF() > weak.opts["pen"].widthF()


def test_each_pair_keeps_its_history_and_shows_its_value(qt_app):
    from mne_rt.viz import ConnectivityPlot

    names, info = _info()
    w = ConnectivityPlot(ch_names=names, info=info, history=5.0, rate=2.0)
    for v in (0.1, 0.2, 0.3):
        w.push({"sensor_connectivity": _snap([v, 0.5])})
    panel = w._panels["sensor_connectivity"]
    _, y = panel._curves[0].getData()
    np.testing.assert_allclose(y[-3:], [0.1, 0.2, 0.3])
    assert "0.3" in panel._legend.items[0][1].text


def test_source_rois_sit_on_a_circle_by_hemisphere(qt_app):
    pos = circle_positions(["insula-lh", "insula-rh", "Left-Hippocampus", "Right-Amygdala"])
    assert pos["insula-lh"][0] < 0.5 < pos["insula-rh"][0]
    assert pos["Left-Hippocampus"][0] < 0.5 < pos["Right-Amygdala"][0]

    from mne_rt.viz import ConnectivityPlot

    w = ConnectivityPlot()
    nodes = ["insula-lh", "insula-rh", "Left-Hippocampus"]
    w.push(
        {"source_connectivity": _snap([0.4], [("insula-lh", "insula-rh")], "plv", "source", nodes)}
    )
    assert set(w._panels["source_connectivity"]._nodes) == set(nodes)


@pytest.mark.parametrize(
    "metric, signed, expected",
    [("coh", True, (0.0, 1.0)), ("imcoh", True, (-1.0, 1.0)), ("imcoh", False, (0.0, 1.0))],
)
def test_the_colour_scale_is_fixed_by_the_metric(metric, signed, expected):
    assert value_range(metric, signed) == expected


def test_an_open_ended_metric_scales_to_the_values_on_show(qt_app):
    """Not to every value seen: one outlier would dim the rest of the session."""
    from mne_rt.viz import ConnectivityPlot

    names, info = _info()
    w = ConnectivityPlot(ch_names=names, info=info, history=1.0, rate=2.0)
    assert value_range("gc", True) is None
    w.push({"x": _snap([5.0, 1.0], metric="gc")})
    panel = w._panels["x"]
    assert panel._color_range(np.array([1.0, 1.0]))[1] == 5.0
    for _ in range(10):  # the outlier scrolls out of the history
        w.push({"x": _snap([1.0, 0.5], metric="gc")})
    assert panel._color_range(np.array([1.0, 0.5])) == (0.0, 1.0)


def test_a_signed_metric_gone_negative_is_centred_on_zero(qt_app):
    from mne_rt.viz import ConnectivityPlot

    names, info = _info()
    w = ConnectivityPlot(ch_names=names, info=info)
    w.push({"x": _snap([-0.6, 0.2], metric="pli2_unbiased", signed=True)})
    assert w._panels["x"]._color_range(np.array([-0.6, 0.2])) == (-0.6, 0.6)


def test_rois_without_a_hemisphere_go_all_around(qt_app):
    """Broca and Wernicke were squeezed onto a short arc at the bottom."""
    pos = circle_positions(["Broca", "Wernicke", "Hippocampus"])
    xs = [x for x, _ in pos.values()]
    assert max(xs) - min(xs) > 0.5
    assert circle_positions(["ctx-lh-insula", "ctx-rh-insula"])["ctx-lh-insula"][0] < 0.5
    mixed = circle_positions(["Broca", "Wernicke", "insula-lh", "insula-rh"])
    assert abs(mixed["Broca"][0] - mixed["Wernicke"][0]) > 0.15  # labels do not collide


def test_a_dropped_window_leaves_a_gap_in_time(qt_app):
    from mne_rt.viz import ConnectivityPlot

    names, info = _info()
    w = ConnectivityPlot(ch_names=names, info=info)
    for t in (0.0, 0.5, 2.0):  # the window at 1.0 s and 1.5 s never arrived
        w.push({"x": dict(_snap([0.5, 0.5]), time=t)})
    x, _ = w._panels["x"]._curves[0].getData()
    np.testing.assert_allclose(x, [-2.0, -1.5, 0.0])
