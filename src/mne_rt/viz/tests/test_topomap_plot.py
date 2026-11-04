"""TopomapPlot: live band-power scalp maps."""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.viz.tests._helpers import N_TOPO, SFREQ, _bad_p3_info

pytestmark = pytest.mark.usefixtures("_destroy_widgets")
RNG = np.random.default_rng(42)


class TestTopomapPlot:
    def test_construct(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        assert w._sfreq == SFREQ
        assert len(w._bands) == len(w._visible_bands)
        w.close()

    def test_push_updates_last_data(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        data = RNG.standard_normal((N_TOPO, int(SFREQ))).astype(np.float32) * 1e-6
        w.push(data)
        qt_app.processEvents()
        assert w._last_data is not None
        assert w._last_data.shape == data.shape
        w.close()

    def test_pause_inhibits_update(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        w._paused = True
        data = RNG.standard_normal((N_TOPO, int(SFREQ))).astype(np.float32) * 1e-6
        w.push(data)
        qt_app.processEvents()
        assert w._last_data is None
        w.close()

    def test_custom_band_added(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        initial = len(w._bands)
        w._custom_flo.setValue(15.0)
        w._custom_fhi.setValue(25.0)
        w._add_custom_band()
        assert len(w._bands) == initial + 1
        assert "Custom 15.0–25.0 Hz" in w._bands
        assert "Custom 15.0–25.0 Hz" in w._visible_bands
        w.close()

    def test_toggle_band_visibility(self, qt_app, topo_info):
        from mne_rt.viz.topomap_plot import TopomapPlot

        w = TopomapPlot(topo_info, sfreq=SFREQ)
        first_band = list(w._bands.keys())[0]
        # Disable first band
        w._band_checks[first_band].setChecked(False)
        assert first_band not in w._visible_bands
        # Re-enable
        w._band_checks[first_band].setChecked(True)
        assert first_band in w._visible_bands
        w.close()


def test_topomap_interpolates_a_bad_channel(qt_app):
    """A broken electrode is drawn from its neighbours, whatever it records."""
    from mne_rt.viz.topomap_plot import TopomapPlot

    info = _bad_p3_info()
    pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    amp = 1 + 5 * (pos[:, 1].max() - pos[:, 1]) / np.ptp(pos[:, 1])  # alpha grows posteriorly
    t = np.arange(int(SFREQ)) / SFREQ
    data = amp[:, None] * np.sin(2 * np.pi * 10 * t) * 1e-5
    data[0] = RNG.standard_normal(len(t)) * 1e-3  # P3 is broken

    w = TopomapPlot(info, sfreq=SFREQ)
    assert list(w._plot_info["ch_names"][-1:]) == ["P3"]
    assert w._interp_matrix.shape == (1, len(info["ch_names"]) - 1)
    alpha = w._compute_band_powers(data)[next(b for b in w._bands if "8" in b)]
    ratio = alpha[-1] / np.median(alpha[:-1])
    assert ratio == pytest.approx(amp[0] ** 2 / np.median(amp[1:] ** 2), rel=0.1)
    w.push(data)  # draws without raising
    w.close()

    off = TopomapPlot(info, sfreq=SFREQ, interpolate_bads=False)
    assert off._interp_matrix is None and "P3" not in off._plot_info["ch_names"]
    off.close()


def test_topomap_without_any_position_builds_and_ignores_pushes(qt_app):
    """pick_info of no channel raised, so the window, and record_main with it,
    failed at setup."""
    import mne

    from mne_rt.viz.topomap_plot import TopomapPlot

    info = mne.create_info(["A", "B", "C"], SFREQ, "eeg")  # no montage: NaN positions
    w = TopomapPlot(info=info, sfreq=SFREQ)
    w.push(np.zeros((3, 64)))  # no-op, not a crash
    w.close()


def test_topomap_with_every_eeg_channel_bad_builds(qt_app, topo_info):
    """The fallback picks included the bads, which the interpolation added a
    second time: pick_info refused the duplicates."""
    from mne_rt.viz.topomap_plot import TopomapPlot

    info = topo_info.copy()
    info["bads"] = list(info["ch_names"])
    w = TopomapPlot(info=info, sfreq=SFREQ)
    topo_picks = list(w._topo_picks)
    w.close()
    assert topo_picks == []  # the fallback no longer draws bad channels as good


def test_topomap_takes_a_window_of_the_good_channels(qt_app, topo_info):
    """What mne-lsl's get_data returns by default; each caller had to pad it."""
    from mne_rt.viz.topomap_plot import TopomapPlot

    info = topo_info.copy()
    info["bads"] = [info["ch_names"][2]]
    w = TopomapPlot(info=info, sfreq=SFREQ)
    seen = []
    w._update_topomaps = lambda data: seen.append(data.shape[0])
    w.push(RNG.standard_normal((N_TOPO - 1, 64)))
    assert seen == [N_TOPO]
    w.close()


def test_topomap_smoothing_survives_a_nan(qt_app, topo_info):
    """One NaN band power blanked that channel for the rest of the session."""
    from mne_rt.viz.topomap_plot import TopomapPlot

    w = TopomapPlot(info=topo_info, sfreq=SFREQ, display_smoothing=0.5)
    shown = []
    band = next(iter(w._bands))
    w._compute_band_powers = lambda data: {band: data[:, 0].copy()}
    import mne

    monkey = mne.viz.plot_topomap
    try:
        mne.viz.plot_topomap = lambda powers, *a, **k: shown.append(np.array(powers))
        for first in (1.0, np.nan, 3.0):
            data = np.ones((N_TOPO, 4))
            data[0, 0] = first
            w._update_topomaps(data)
    finally:
        mne.viz.plot_topomap = monkey
    w.close()
    assert np.isnan(shown[1][0])  # drawn blank in the bad frame
    assert shown[2][0] == pytest.approx(0.5 * 3.0 + 0.5 * 1.0)  # and back after it


def test_topomap_smoothing_zero_raises(qt_app, topo_info):
    """0 froze the map on its first frame, as it did the brain."""
    from mne_rt.viz.topomap_plot import TopomapPlot

    with pytest.raises(ValueError, match="display_smoothing"):
        TopomapPlot(info=topo_info, sfreq=SFREQ, display_smoothing=0.0)


def test_topomap_colour_scale_ignores_a_nan_channel(qt_app, topo_info):
    """One NaN channel made the whole band's scale NaN."""
    from mne_rt.viz.topomap_plot import TopomapPlot

    w = TopomapPlot(info=topo_info, sfreq=SFREQ)
    band = next(iter(w._bands))
    powers = np.arange(1.0, N_TOPO + 1.0)
    powers[0] = np.nan
    w._compute_band_powers = lambda data: {band: powers}
    w._update_topomaps(np.zeros((N_TOPO, 4)))
    clim = w._clim_cache[band]
    w.close()
    assert np.all(np.isfinite(clim))
