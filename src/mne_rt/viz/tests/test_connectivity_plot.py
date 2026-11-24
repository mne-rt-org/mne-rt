"""ConnectivityPlot: the pairs a connectivity modality computes, live."""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.viz.connectivity_plot import circle_order, hemisphere, value_range

pytestmark = pytest.mark.usefixtures("_destroy_widgets")

PAIRS = [("C3", "C4"), ("F3", "F4")]
ROIS = ["insula-lh", "insula-rh", "superiortemporal-lh", "superiortemporal-rh", "Brain-Stem"]


def _snap(edges, labels=PAIRS, metric="coh", space="sensor", nodes=None, signed=True, t=None):
    edges = np.asarray(edges, dtype=float)
    snap = dict(
        value=float(np.nanmean(edges)),
        edges=edges,
        labels=labels,
        metric=metric,
        signed=signed,
        space=space,
        nodes=nodes,
    )
    if t is not None:
        snap["time"] = t
    return snap


def _plot(**kwargs):
    import mne

    from mne_rt._montage import make_standard_montage
    from mne_rt.viz import ConnectivityPlot

    names = ["F3", "F4", "C3", "C4", "O1", "O2"]
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage(make_standard_montage("colin27_1020"))
    return ConnectivityPlot(ch_names=names, info=info, **kwargs)


def test_each_modality_is_picked_in_the_side_panel(qt_app):
    w = _plot()
    w.push({"sensor_connectivity": _snap([0.9, 0.2])})
    qt_app.processEvents()
    w.push({"sensor_connectivity": _snap([0.8, 0.3]), "sensor_graph@beta": _snap([0.1, 0.2])})
    assert [w._cmb_mod.itemText(i) for i in range(w._cmb_mod.count())] == [
        "sensor_connectivity",
        "sensor_graph@beta",
    ]
    assert w._current == "sensor_connectivity"  # the first one is shown
    w._cmb_mod.setCurrentIndex(1)
    assert w._current == "sensor_graph@beta"


def test_sensor_pairs_are_drawn_on_mne_s_head(qt_app):
    """The head and sensors are mne.viz.plot_sensors', as on the topomaps."""
    w = _plot()
    w.push({"sensor_connectivity": _snap([0.9, 0.1])})
    qt_app.processEvents()
    ax = w._fig.axes[0]
    assert len(ax.collections) == 1  # plot_sensors' scatter of the sensors
    strong, weak = w._edge_lines
    assert strong.get_linewidth() > weak.get_linewidth()
    # Each edge joins its two sensors' dots.
    dots = ax.collections[0].get_offsets()
    names = list(w._info["ch_names"])
    xs, ys = strong.get_data()
    np.testing.assert_allclose([xs[0], ys[0]], dots[names.index("C3")])
    np.testing.assert_allclose([xs[1], ys[1]], dots[names.index("C4")])


def test_source_pairs_are_drawn_on_mne_connectivity_s_circle(qt_app):
    w = _plot()
    labels = [("insula-lh", "insula-rh")]
    w.push({"source_connectivity": _snap([0.4], labels, "plv", "source", ROIS)})
    qt_app.processEvents()
    ax = w._fig.axes[0]
    assert ax.name == "polar"  # plot_connectivity_circle's
    assert {t.get_text() for t in ax.texts} >= set(ROIS)  # every ROI, paired or not
    (edge,) = w._circle_edges
    assert edge in ax.patches


def test_circle_edges_change_with_their_values(qt_app):
    """Fixed at linewidth 3 on a [-1, 1] scale, imcoh's edges hardly changed."""
    w = _plot()
    labels = [("insula-lh", "insula-rh"), ("superiortemporal-lh", "superiortemporal-rh")]
    w.push({"src": _snap([0.25, 0.05], labels, "imcoh", "source", ROIS)})
    qt_app.processEvents()
    strong, weak = w._circle_edges
    assert strong.get_linewidth() > 2 * weak.get_linewidth()
    assert w._series["src"].color_range(60.0) == (-0.25, 0.25)  # the values on show
    w.push({"src": _snap([0.05, 0.25], labels, "imcoh", "source", ROIS)})
    qt_app.processEvents()
    assert w._circle_edges[0] is strong  # restyled, not redrawn
    assert strong.get_linewidth() < weak.get_linewidth()


def test_homologous_rois_face_each_other():
    assert hemisphere("insula-lh") == -1 and hemisphere("ctx-rh-insula") == 1
    assert hemisphere("Brain-Stem") == 0
    assert hemisphere("Broca") == -1  # from its atlas labels
    order = circle_order(ROIS)
    assert order == [
        "insula-lh",
        "superiortemporal-lh",
        "Brain-Stem",
        "superiortemporal-rh",
        "insula-rh",
    ]


def test_a_hidden_pair_leaves_the_trend_and_the_head(qt_app):
    w = _plot()
    w.push({"sensor_connectivity": _snap([0.9, 0.2])})
    qt_app.processEvents()
    w._pair_checks[1].setChecked(False)
    assert not w._curves[1].isVisible()
    assert not w._edge_lines[1].get_visible()
    assert w._series["sensor_connectivity"].visible == [True, False]
    legend = [label.text for _, label in w._legend.items]
    assert not any("F3" in text for text in legend)  # it leaves the legend too
    w._chk_value.setChecked(False)
    assert len(w._legend.items) == 1


def test_the_time_window_is_chosen_in_the_side_panel(qt_app):
    w = _plot()
    for t in np.arange(0, 100, 0.5):
        w.push({"x": _snap([0.5, 0.5], t=float(t))})
        qt_app.processEvents()
    w._cmb_window.setCurrentIndex(w._cmb_window.findData(30))
    lo, hi = w._trend.viewRange()[0]
    assert lo == pytest.approx(-30, abs=1) and hi == pytest.approx(0, abs=1)


def test_each_pair_keeps_its_history_and_a_dropped_window_leaves_a_gap(qt_app):
    w = _plot()
    for t, v in ((0.0, 0.1), (0.5, 0.2), (2.0, 0.3)):  # the windows at 1.0-1.5 s never came
        w.push({"x": _snap([v, 0.5], t=t)})
        qt_app.processEvents()
    x, y = w._curves[0].getData()
    np.testing.assert_allclose(x, [-2.0, -1.5, 0.0])
    np.testing.assert_allclose(y, [0.1, 0.2, 0.3])
    assert "0.3" in w._legend.items[0][1].text


def test_pausing_keeps_the_data_and_freezes_the_display(qt_app):
    w = _plot()
    w.push({"x": _snap([0.5, 0.5], t=0.0)})
    qt_app.processEvents()
    w._btn_pause.setChecked(True)
    w.push({"x": _snap([0.9, 0.5], t=0.5)})
    qt_app.processEvents()
    _, y = w._curves[0].getData()
    assert y[-1] == 0.5 and w._series["x"].view()[3][0] == 0.9
    w._btn_pause.setChecked(False)
    _, y = w._curves[0].getData()
    assert y[-1] == 0.9


@pytest.mark.parametrize(
    "metric, signed, expected",
    [("coh", True, (0.0, 1.0)), ("imcoh", True, (-1.0, 1.0)), ("imcoh", False, (0.0, 1.0))],
)
def test_the_colour_scale_is_fixed_by_the_metric(metric, signed, expected):
    assert value_range(metric, signed) == expected


def test_a_bounded_metric_s_colours_follow_the_values_within_its_bounds(qt_app):
    w = _plot()
    w.push({"x": _snap([0.3, 0.2], metric="coh", t=0.0)})
    qt_app.processEvents()
    assert w._series["x"].color_range(60.0) == (0.0, 0.3)
    w.push({"y": _snap([0.01, 0.02], metric="coh", t=0.0)})  # near 0: not noise at full scale
    assert w._series["y"].color_range(60.0) == (0.0, 0.1)
    w.push({"z": _snap([0.1, 0.2], metric="imcoh", t=0.0)})  # signed: centred on 0
    assert w._series["z"].color_range(60.0) == (-0.2, 0.2)


def test_an_open_ended_metric_scales_to_the_values_on_show(qt_app):
    """Not to every value seen: one outlier would dim the rest of the session."""
    w = _plot()
    assert value_range("gc", True) is None
    w.push({"x": _snap([5.0, 1.0], metric="gc", t=0.0)})
    qt_app.processEvents()
    series = w._series["x"]
    assert series.color_range(10.0)[1] == 5.0
    for t in np.arange(1, 30):  # the outlier leaves the 10 s window
        w.push({"x": _snap([1.0, 0.5], metric="gc", t=float(t))})
        qt_app.processEvents()
    assert series.color_range(10.0) == (0.0, 1.0)


def test_a_signed_metric_gone_negative_is_centred_on_zero(qt_app):
    w = _plot()
    w.push({"x": _snap([-0.6, 0.2], metric="pli2_unbiased", signed=True)})
    qt_app.processEvents()
    assert w._series["x"].color_range(60.0) == (-0.6, 0.6)


def _info_with(names, types, montage=True):
    import mne

    from mne_rt._montage import make_standard_montage

    info = mne.create_info(names, 250.0, types)
    if montage:
        info.set_montage(make_standard_montage("colin27_1020"), on_missing="ignore")
    return info


def test_edges_join_the_right_sensors_with_a_stim_channel_first(qt_app):
    """plot_sensors draws one channel type: matched by position to the whole
    info, a stim channel first shifted every edge one electrode over."""
    from mne_rt.viz import ConnectivityPlot

    names = ["STI", "F3", "F4", "C3", "C4", "O1"]
    w = ConnectivityPlot(ch_names=names, info=_info_with(names, ["stim"] + ["eeg"] * 5))
    w.push({"x": _snap([0.9, 0.1])})
    qt_app.processEvents()
    dots = w._fig.axes[0].collections[0].get_offsets()
    drawn = list(w._drawn_info["ch_names"])
    assert drawn == ["F3", "F4", "C3", "C4", "O1"]
    xs, ys = w._edge_lines[0].get_data()  # C3 - C4
    np.testing.assert_allclose([xs[0], ys[0]], dots[drawn.index("C3")])
    np.testing.assert_allclose([xs[1], ys[1]], dots[drawn.index("C4")])


def test_unplaced_channels_are_left_out_and_no_montage_still_draws(qt_app):
    from mne_rt.viz import ConnectivityPlot

    names = ["F3", "F4", "C3", "C4", "AUX1"]  # AUX1: no position in the montage
    w = ConnectivityPlot(ch_names=names, info=_info_with(names, "eeg"))
    assert "AUX1" not in w._drawn_info["ch_names"]

    bare = ConnectivityPlot(ch_names=names, info=_info_with(names, "eeg", montage=False))
    assert bare._drawn_info is None
    bare.push({"x": _snap([0.9, 0.1])})
    qt_app.processEvents()
    assert all(line is not None for line in bare._edge_lines)  # placed by a layout


def test_only_ch_names_are_drawn_and_bad_ones_are_outlined(qt_app):
    from mne_rt.viz import ConnectivityPlot

    names = ["F3", "F4", "C3", "C4", "O1", "O2"]
    info = _info_with(names, "eeg")
    info["bads"] = ["C3"]
    w = ConnectivityPlot(ch_names=names[:4], info=info)
    w.push({"x": _snap([0.9, 0.1])})
    qt_app.processEvents()
    assert list(w._drawn_info["ch_names"]) == names[:4]
    dots = w._fig.axes[0].collections[0]
    assert dots.get_picker() is None  # plot_sensors' click handler removed our names
    edge = dots.get_edgecolors()
    assert edge[names.index("C3")][0] > 0.8  # red
    assert edge[names.index("F3")][3] == 0  # none


def test_a_bad_snapshot_does_not_stop_the_others(qt_app):
    w = _plot()
    w.push({"broken": {"labels": PAIRS}, "x": _snap([0.4, 0.6])})
    qt_app.processEvents()
    assert "x" in w._series and "broken" not in w._series


def test_a_circle_without_rois_or_with_repeats_does_not_fail(qt_app):
    w = _plot()
    w.push({"src": _snap([0.4], [("a-lh", "a-lh")], "plv", "source", ["a-lh", "a-lh"])})
    qt_app.processEvents()
    assert w._series["src"].nodes == ["a-lh"]
    assert any("no ROIs" in t.get_text() for t in w._fig.axes[0].texts)


def test_pause_holds_while_the_controls_are_used(qt_app):
    """Changing the time window or a pair while paused showed the newest data."""
    w = _plot()
    for t, v in ((0.0, 0.2), (0.5, 0.3)):
        w.push({"x": _snap([v, 0.5], t=t)})
        qt_app.processEvents()
    w._btn_pause.setChecked(True)
    w.push({"x": _snap([0.9, 0.5], t=1.0)})
    qt_app.processEvents()
    w._cmb_window.setCurrentIndex(w._cmb_window.findData(10))
    w._pair_checks[1].setChecked(False)
    _, y = w._curves[0].getData()
    assert y[-1] == 0.3
    w._btn_pause.setChecked(False)
    _, y = w._curves[0].getData()
    assert y[-1] == 0.9


def test_the_head_panel_width_is_kept_across_modalities(qt_app):
    w = _plot()
    w.resize(1500, 640)
    w.show()
    w.push({"a": _snap([0.5, 0.5]), "b": _snap([0.5, 0.5])})
    qt_app.processEvents()
    w._splitter.setSizes([500, 700])
    chosen = w._splitter.sizes()
    w._cmb_mod.setCurrentIndex(1)
    w._cmb_mod.setCurrentIndex(0)
    assert w._splitter.sizes() == chosen


@pytest.mark.parametrize(
    "name, side",
    [
        ("ctx_lh_G_insular_short", -1),  # aparc.a2009s+aseg
        ("ctx_rh_S_central", 1),
        ("lh.V1", -1),
        ("Right-Putamen", 1),
        ("Brain-Stem", 0),
    ],
)
def test_the_hemisphere_is_read_from_any_atlas_naming(name, side):
    assert hemisphere(name) == side


def test_an_roi_named_without_a_hemisphere_takes_its_labels_one(qt_app):
    """'LangL' would sit at the bottom, in neither hemisphere."""
    nodes = ["parsopercularis-lh", "LangL", "insula-rh"]
    sides = {"LangL": -1}
    assert circle_order(nodes, sides) == ["parsopercularis-lh", "LangL", "insula-rh"]
    w = _plot()
    snap = _snap([0.4], [("LangL", "insula-rh")], "plv", "source", nodes)
    snap["sides"] = sides
    w.push({"src": snap})
    qt_app.processEvents()
    assert w._series["src"].sides == sides


def test_a_hidden_pair_leaves_the_colour_range(qt_app):
    w = _plot()
    w.push({"x": _snap([0.8, 0.1], metric="imcoh")})
    qt_app.processEvents()
    assert w._series["x"].color_range(60.0) == (-0.8, 0.8)
    w._pair_checks[0].setChecked(False)
    assert w._series["x"].color_range(60.0) == (-0.1, 0.1)


def test_a_click_on_a_legend_line_unticks_its_pair(qt_app):
    """pyqtgraph's own click hid the curve, and nothing else."""
    w = _plot()
    w.push({"x": _snap([0.8, 0.1])})
    qt_app.processEvents()
    sample = w._legend.items[0][0]

    class _Click:
        def accept(self):
            pass

    sample.mouseClickEvent(_Click())
    qt_app.processEvents()
    assert not w._pair_checks[0].isChecked()
    assert w._series["x"].visible == [False, True]
    assert not w._edge_lines[0].get_visible()


def test_a_modality_first_seen_while_paused_is_frozen_too(qt_app):
    w = _plot()
    w.push({"a": _snap([0.5, 0.5], t=0.0)})
    qt_app.processEvents()
    w._btn_pause.setChecked(True)
    w.push({"b": _snap([0.2, 0.5], t=0.5)})
    w.push({"b": _snap([0.9, 0.5], t=1.0)})
    w._cmb_mod.setCurrentIndex(1)
    _, y = w._curves[0].getData()
    assert y[-1] == 0.2
