"""The clickable scalp map shared by the viewers, and sensor positions."""

from __future__ import annotations

import pytest

from mne_rt._montage import make_standard_montage
from mne_rt.viz.tests._helpers import CH_NAMES, SFREQ, _bad_p3_info, _Click, _selector

pytestmark = pytest.mark.usefixtures("_destroy_widgets")


def test_sensor_positions_put_frontal_channels_on_top():
    from mne_rt.viz._sensor_selector import sensor_positions

    pos = dict(
        zip(["Fz", "Cz", "Oz"], sensor_positions(["Fz", "Cz", "Oz"], montage="standard_1020"))
    )
    assert pos["Fz"][1] < pos["Cz"][1] < pos["Oz"][1]  # 0 = frontal


def test_sensor_selector_respects_limits_and_signals(qt_app):
    w = _selector(selected=["Fp1"], max_selected=2)
    changes = []
    w.selectionChanged.connect(changes.append)
    assert w.toggle("F3") and w.selected == ["Fp1", "F3"]
    assert not w.toggle("F4")  # at the maximum
    assert w.toggle("Fp1") and w.selected == ["F3"]
    assert not w.toggle("F3")  # at the minimum of one
    assert not w.toggle("nope")
    assert changes == [["Fp1", "F3"], ["F3"]]
    w.set_selected(["P3", "nope"])  # no signal
    assert w.selected == ["P3"] and len(changes) == 2
    w.close()


def test_compare_evoked_and_tfr_follow_the_selector(qt_app, common_kw):
    from mne_rt.viz.compare_evoked import CompareEvoked
    from mne_rt.viz.tfr_plot import TFRPlot

    w = CompareEvoked(**common_kw, channels=CH_NAMES[:1])
    w._selector.toggle(CH_NAMES[3])
    assert w._disp_channels == [CH_NAMES[0], CH_NAMES[3]]
    assert set(w._ch_plots) == {CH_NAMES[0], CH_NAMES[3]}
    w.close()

    t = TFRPlot(**common_kw, channels=CH_NAMES[:1])
    for ch in CH_NAMES[1:]:
        t._selector.toggle(ch)
    assert len(t._display_chs) == 8 == t._selector._max_selected
    t.close()


def test_sensor_positions_place_bad_channels_too():
    """find_layout drops bads, which put them in the off-head fallback column."""
    from mne_rt.viz._sensor_selector import sensor_positions

    info = _bad_p3_info()
    pos = dict(zip(info["ch_names"], sensor_positions(info["ch_names"], info)))
    assert pos["P3"][1] > pos["C3"][1] > pos["F3"][1]  # behind C3, not at the top edge


def test_sensor_positions_tolerate_channels_without_a_position():
    """Two unplaced channels overlap at 0, which made find_layout refuse them all."""
    import mne

    from mne_rt.viz._sensor_selector import sensor_positions

    names = ["Fz", "Cz", "Oz", "HEOG", "VEOG"]
    info = mne.create_info(names, SFREQ, "eeg")
    info.set_montage(make_standard_montage("colin27_1020"), on_missing="ignore")
    pos = dict(zip(names, sensor_positions(names, info, montage=None)))
    assert pos["Fz"][1] < pos["Cz"][1] < pos["Oz"][1]  # a real layout, not the circle
    assert pos["HEOG"][0] == pos["VEOG"][0] == 0.02  # the edge column


def test_sensor_selector_right_click_reaches_the_handler(qt_app):
    """pyqtgraph's scatter emits clicks for the left button only, so "right
    click: bad" never fired from a real click."""
    from qtpy.QtCore import Qt

    from mne_rt.viz._sensor_selector import SensorSelector, sensor_positions

    names = ["Fz", "Cz", "Pz"]
    w = SensorSelector(names, sensor_positions(names))
    got = []
    w.channelRightClicked.connect(got.append)
    pos = w._scatter.points()[0].pos()
    # Offscreen the view has no real pixel size, so let the scatter say which
    # dot is under the click rather than assume it.
    expected = w._scatter.pointsAt(pos)[0].data()
    ev = _Click(Qt.MouseButton.RightButton, pos)
    w._scatter.mouseClickEvent(ev)
    w.close()
    assert ev.accepted and got == [expected]


def test_sensor_positions_keep_many_unplaced_channels_in_view():
    """A single column put the 21st unplaced channel and later below the view."""
    from mne_rt.viz._sensor_selector import sensor_positions

    placed = (
        "Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 Oz O2 Fpz AFz FCz CPz POz".split()
    )
    unplaced = [f"AUX{i}" for i in range(24)]
    names = placed + unplaced  # 25 placed: a layout, not the circle
    pos = dict(zip(names, sensor_positions(names)))
    assert pos["Fz"][1] < pos["Cz"][1] < pos["Oz"][1]
    assert abs(pos["C3"][1] - pos["C4"][1]) < 0.02  # left and right, not on a ring
    assert all(0.0 <= pos[ch][1] <= 1.0 for ch in unplaced)
