"""CompareEvoked: per-channel condition comparison."""

from __future__ import annotations

import pytest

from mne_rt.viz.tests._helpers import CH_NAMES, N_CH, _make_epochs

pytestmark = pytest.mark.usefixtures("_destroy_widgets")


class TestCompareEvoked:
    def test_construct(self, qt_app, common_kw):
        from mne_rt.viz.compare_evoked import CompareEvoked

        w = CompareEvoked(**common_kw)
        assert w is not None
        w.close()

    def test_construct_with_channels(self, qt_app, common_kw):
        from mne_rt.viz.compare_evoked import CompareEvoked

        kw = {**common_kw, "channels": CH_NAMES[:3]}
        w = CompareEvoked(**kw)
        assert w._disp_channels == CH_NAMES[:3]
        w.close()

    def test_update_buffer(self, qt_app, common_kw):
        from mne_rt.viz.compare_evoked import CompareEvoked

        w = CompareEvoked(**common_kw)
        data, conds = _make_epochs(10)
        w.update(data, conds)
        qt_app.processEvents()
        assert w._n_per["left"] == 5
        assert w._n_per["right"] == 5
        w.close()

    def test_no_channel_cap(self, qt_app, common_kw):
        """All 8 channels selectable — no artificial cap."""
        from mne_rt.viz.compare_evoked import CompareEvoked

        kw = {**common_kw, "channels": CH_NAMES}  # all 8
        w = CompareEvoked(**kw)
        assert len(w._disp_channels) == N_CH
        w.close()
