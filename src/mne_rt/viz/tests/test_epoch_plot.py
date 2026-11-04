"""EpochPlot: scrolling raw signal with epoch overlays."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pytest

from mne_rt.viz.tests._helpers import CH_NAMES, N_CH, SFREQ

pytestmark = pytest.mark.usefixtures("_destroy_widgets")
RNG = np.random.default_rng(42)


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
        w._process_pending()  # drain queue synchronously (timer-based in production)
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
        w._process_pending()  # drain queue synchronously
        assert len(w._triggers) == 1
        trig_abs, trig_code, epoch_id = w._triggers[0]
        assert trig_abs == 64
        assert trig_code == 1
        assert epoch_id == 0
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
        w.push(data)
        w.push_trigger(1)
        w.push_trigger(2)
        w._process_pending()  # drain queue so triggers land in _triggers
        assert len(w._triggers) == 2
        w._clear_triggers()
        assert len(w._triggers) == 0
        w.close()

    def test_epoch_ids_increment_and_survive_clear_triggers(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ)
        w.push_trigger(1)
        w.push_trigger(1)
        w._process_pending()
        ids = [t[2] for t in w._triggers]
        assert ids == [0, 1]
        # next_epoch_id keeps counting monotonically even after clearing
        # the (now orphaned) trigger records.
        w._clear_triggers()
        w.push_trigger(1)
        w._process_pending()
        assert w._triggers[0][2] == 2
        w.close()

    def test_toggle_bad_epoch_marks_and_unmarks(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ)
        w.push_trigger(1)
        w._process_pending()
        assert w.bad_epoch_ids == []
        assert not w.is_epoch_bad(0)

        w._toggle_bad_epoch(0)
        assert w.bad_epoch_ids == [0]
        assert w.is_epoch_bad(0)

        w._toggle_bad_epoch(0)
        assert w.bad_epoch_ids == []
        assert not w.is_epoch_bad(0)
        w.close()

    def test_bad_epoch_renders_in_red(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ, time_window=5.0)
        w.push_trigger(1)
        w._process_pending()
        assert len(w._visible_epoch_spans) == 1
        assert w._visible_epoch_spans[0][0] == 0

        w._toggle_bad_epoch(0)
        # After marking bad, the trigger line for epoch 0 is drawn in the
        # dedicated bad-epoch red, not the default event-code colour.
        trig_lines = [
            item
            for item in w._epoch_overlay_items
            if isinstance(item, pg.InfiniteLine) and item.angle == 90
        ]
        assert any(line.pen.color().name() == w._BAD_EPOCH_COLOR for line in trig_lines)
        w.close()

    def test_epoch_id_at_hit_tests_visible_spans(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ, tmin=-0.1, tmax=0.5, time_window=5.0)
        # Trigger fired "now" (at the right edge); pushing data afterwards
        # scrolls it left to the middle of the visible window, at x0 = 2.5 s.
        w.push_trigger(1)
        n_shift = int(2.5 * SFREQ)
        data = RNG.standard_normal((N_CH, n_shift)).astype(np.float32) * 1e-6
        w.push(data)
        w._process_pending()
        assert w._visible_epoch_spans[0] == (0, pytest.approx(2.4), pytest.approx(3.0))
        assert w._epoch_id_at(2.7) == 0
        assert w._epoch_id_at(4.9) is None
        w.close()

    def test_clear_bad_epochs_resets_marks_but_keeps_triggers(self, qt_app):
        from mne_rt.viz.epoch_plot import EpochPlot

        w = EpochPlot(CH_NAMES, sfreq=SFREQ)
        w.push_trigger(1)
        w.push_trigger(1)
        w._process_pending()
        w._toggle_bad_epoch(0)
        w._toggle_bad_epoch(1)
        assert w.bad_epoch_ids == [0, 1]

        w._clear_bad_epochs()
        assert w.bad_epoch_ids == []
        assert len(w._triggers) == 2
        w.close()


def test_epoch_clear_removes_the_trigger_overlays(qt_app):
    from mne_rt.viz.epoch_plot import EpochPlot

    w = EpochPlot(CH_NAMES, sfreq=SFREQ)
    w.push(np.zeros((N_CH, int(SFREQ))))
    w.push_trigger(1)
    w.push(np.zeros((N_CH, int(SFREQ))))
    w._process_pending()
    drawn = list(w._epoch_overlay_items)
    assert drawn
    w._clear()
    assert not any(item in w._pi.items for item in drawn)
    w.close()


def test_epoch_triggers_are_dropped_while_paused(qt_app):
    """Their data is dropped while paused, so they piled up at one stale index."""
    from mne_rt.viz.epoch_plot import EpochPlot

    w = EpochPlot(CH_NAMES, sfreq=SFREQ)
    w._toggle_pause(True)
    w.push_trigger(1)
    assert not w._pending
    w.close()
