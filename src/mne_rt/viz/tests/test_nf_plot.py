"""NFPlot: the scrolling neurofeedback signal."""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.viz.tests._helpers import NF_MODALITIES, NF_SCALES, _reward_visible

pytestmark = pytest.mark.usefixtures("_destroy_widgets")


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
