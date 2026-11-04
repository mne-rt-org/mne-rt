"""RawPlot: scrolling raw signal viewer and its channel picker."""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.viz.tests._helpers import CH_NAMES, N_CH, SFREQ

pytestmark = pytest.mark.usefixtures("_destroy_widgets")
RNG = np.random.default_rng(42)


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

    def test_push_does_not_rescale_by_unit_mul(self, qt_app):
        """Data reaching RawPlot is already SI: RTStream converts a µV stream
        where it reads it (``_read_window``). Scaling again here by
        ``unit_mul`` would shrink it a million times, and in MNE ``unit_mul``
        never scales data anyway.
        """
        import mne

        from mne_rt.viz.raw_plot import RawPlot

        info = mne.create_info(CH_NAMES, SFREQ, "eeg")
        with info._unlock():
            for ch in info["chs"]:
                ch["unit_mul"] = -6  # FIFF_UNITM_MU
        w = RawPlot(CH_NAMES, sfreq=SFREQ, info=info)
        w.push(np.full((N_CH, 64), 50e-6))
        w._flush_data_queue()
        assert np.allclose(w._buf[:, -64:], 50e-6)
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


class TestRawPlotChannelPicker:
    def _plot(self, n_shown=3):
        from mne_rt.viz.raw_plot import RawPlot

        return RawPlot(CH_NAMES, sfreq=SFREQ, n_shown=n_shown)

    def test_hidden_channels_are_not_drawn(self, qt_app):
        w = self._plot()
        w.push(np.arange(N_CH, dtype=float)[:, None] * np.ones((N_CH, 50)) * 1e-6)
        w._flush_data_queue()
        w.show_channels([CH_NAMES[6], CH_NAMES[1]])
        assert w.shown_channels == [CH_NAMES[1], CH_NAMES[6]]  # channel order
        assert w._visible() == [1, 6]
        labels = [t[1] for t in w._pi.getAxis("left")._tickLevels[0]]
        assert sorted(labels) == [CH_NAMES[1], CH_NAMES[6]]
        # The rows are drawn from the shown channels' data, the third row is empty
        top = w._curves[0].getData()[1]
        assert np.allclose(top[-1] - 1.0, 1e-6 / w._scale)  # row offset 1, channel 1
        assert w._curves[2].getData()[0] is None or len(w._curves[2].getData()[0]) == 0
        assert w._ch_scroll.maximum() == 0
        w.close()

    def test_paging_runs_over_shown_channels_only(self, qt_app):
        w = self._plot(n_shown=2)
        w.show_channels(CH_NAMES[:5])
        assert w._ch_scroll.maximum() == 3
        w._set_page_start(10)
        assert w._visible() == [3, 4]
        w.show_channels([])
        assert w._visible() == [] and w.shown_channels == []
        w.show_channels(CH_NAMES)
        assert w._visible() == [0, 1]
        w.close()

    def test_right_click_on_a_sensor_marks_it_bad(self, qt_app):
        w = self._plot()
        w._selector.channelRightClicked.emit(CH_NAMES[2])
        assert w.bad_channels == [CH_NAMES[2]]
        assert w._selector._bads == {CH_NAMES[2]}
        w._toggle_bad_channel(2)  # the y-axis label click un-marks it, in both places
        assert w.bad_channels == [] and w._selector._bads == set()
        w.close()


def test_raw_plot_takes_integer_chunks_and_says_when_it_drops_one(qt_app, monkeypatch):
    """A chunk of another channel set was dropped without a word, which is how
    the viewer sat on "Waiting for data" when fed the wrong channels."""
    from mne_rt.viz import raw_plot

    seen = []
    monkeypatch.setattr(raw_plot.logger, "warning", lambda msg, *a, **k: seen.append(msg))
    w = raw_plot.RawPlot(["C3", "C4"], sfreq=100.0)
    w.push(np.ones((2, 10), dtype=np.int16))
    w._flush_data_queue()
    assert np.any(w._buf != 0)
    w.push(np.ones((3, 10)))
    w.push(np.ones((3, 10)))
    assert len(seen) == 1 and "dropping" in seen[0]
    w.close()


def test_raw_plot_picker_uses_the_standard_montage_without_info(qt_app):
    """With montage=None, a RawPlot without info put 10-20 channels on a circle."""
    from mne_rt.viz.raw_plot import RawPlot

    names = ["Fz", "Cz", "Pz", "Oz", "C3", "C4"]
    w = RawPlot(names, sfreq=SFREQ)
    pos = dict(zip(names, w._selector._positions))
    w.close()
    assert pos["Fz"][1] < pos["Cz"][1] < pos["Oz"][1]
    # The test a circle fails: C3 and C4 level with each other, either side of Cz.
    assert abs(pos["C3"][1] - pos["C4"][1]) < 0.02
    assert pos["C3"][0] < pos["Cz"][0] < pos["C4"][0]
