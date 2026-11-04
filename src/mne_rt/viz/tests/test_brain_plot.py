"""Tests for mne_rt.viz.brain_plot.BrainPlot.

The module itself is always importable (pyvista/pyvistaqt imports are
guarded internally), so constant-consistency and the "missing optional
dependency" error path are tested unconditionally.  Tests that construct a
real ``BrainPlot`` instance require the ``viz`` extra (pyvista + pyvistaqt)
and are skipped when it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.viz import brain_plot as bp_module
from mne_rt.viz.brain_plot import BrainPlot

_VIZ_AVAILABLE = bp_module._pyvista_available and bp_module._pyvistaqt_available


_viz_required = pytest.mark.skipif(not _VIZ_AVAILABLE, reason="pyvista/pyvistaqt not installed")

# ------------------------------------------------------------------
# Module-level constant consistency (no pyvista required)
# ------------------------------------------------------------------


def test_every_display_mode_has_a_band():
    """The mode chooses what record_main computes, so each needs one."""
    assert set(bp_module._MODE_BANDS) == set(bp_module._DISPLAY_MODES)
    for band in bp_module._MODE_BANDS.values():
        assert band is None or band[0] < band[1]


def test_view_presets_have_position_focal_up():
    for preset in bp_module._VIEW_PRESETS.values():
        assert set(preset.keys()) == {"position", "focal", "up"}
        assert len(preset["position"]) == 3


def test_backgrounds_are_hex_pairs():
    for bot, top in bp_module._BACKGROUNDS.values():
        assert bot.startswith("#")
        assert top.startswith("#")


def test_default_background_is_valid_key():
    assert bp_module._DEFAULT_BG in bp_module._BACKGROUNDS


# ------------------------------------------------------------------
# Missing optional dependency
# ------------------------------------------------------------------


def test_raises_import_error_when_pyvista_unavailable(monkeypatch):
    monkeypatch.setattr(bp_module, "_pyvista_available", False)
    with pytest.raises(ImportError, match="pyvista"):
        BrainPlot(subjects_fs_dir="/tmp")


def test_raises_import_error_when_pyvistaqt_unavailable(monkeypatch):
    monkeypatch.setattr(bp_module, "_pyvistaqt_available", False)
    with pytest.raises(ImportError, match="pyvista"):
        BrainPlot(subjects_fs_dir="/tmp")


# ------------------------------------------------------------------
# Validation that fires before any pyvista/Qt object is constructed
# (only reachable when both optional deps are actually importable)
# ------------------------------------------------------------------


@_viz_required
def test_invalid_cmap_raises():
    with pytest.raises(ValueError, match="cmap"):
        BrainPlot(subjects_fs_dir="/tmp", cmap="not_a_cmap")


@_viz_required
def test_invalid_surf_raises():
    with pytest.raises(ValueError, match="surf"):
        BrainPlot(subjects_fs_dir="/tmp", surf="not_a_surf")


@_viz_required
@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_display_smoothing_outside_0_1_raises(value):
    """0 weighed every new frame by nothing, freezing the map on the first."""
    with pytest.raises(ValueError, match="display_smoothing"):
        BrainPlot(subjects_fs_dir="/tmp", display_smoothing=value)


# ------------------------------------------------------------------
# Full instantiation (requires viz extra + fsaverage; offscreen rendering)
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def brain(tmp_path_factory):
    if not _VIZ_AVAILABLE:
        pytest.skip("pyvista/pyvistaqt not installed")
    try:
        import pyvista as pv

        pv.OFF_SCREEN = True
        subjects_dir = tmp_path_factory.mktemp("brain_plot_subjects")
        instance = BrainPlot(subjects_fs_dir=str(subjects_dir), window_size=(300, 200))
        yield instance
        instance.plotter.close()
    except Exception as exc:
        pytest.skip(f"BrainPlot construction unavailable: {exc}")


def test_brain_surf_default(brain):
    assert brain.surf == "inflated"


def test_brain_display_mode_default(brain):
    assert brain.display_mode == bp_module._DISPLAY_MODES[0]


def test_brain_set_display_mode(brain):
    brain.set_display_mode("Alpha Power  (8–13 Hz)")
    assert brain.display_mode == "Alpha Power  (8–13 Hz)"
    brain.set_display_mode(bp_module._DISPLAY_MODES[0])


def test_brain_set_display_mode_invalid_raises(brain):
    with pytest.raises(ValueError, match="mode"):
        brain.set_display_mode("not_a_mode")


def test_brain_set_surface_invalid_raises(brain):
    with pytest.raises(ValueError, match="surf"):
        brain.set_surface("not_a_surf")


def test_brain_reset_activity_zeros_scalars(brain):
    brain._scalars_full[:] = 5.0
    brain.reset_activity()
    assert np.all(brain._scalars_full == 0.0)


def test_brain_update_from_arrays_updates_scalars(brain):
    n_src = 10242
    lh = np.ones(n_src)
    rh = np.full(n_src, 2.0)
    brain.update_from_arrays(lh, rh, deferred=True)
    n_lh = brain._n_lh
    assert np.allclose(brain._scalars_full[:n_lh], 1.0)
    assert np.allclose(brain._scalars_full[n_lh:], 2.0)
    brain.reset_activity()


def test_brain_custom_band_set_and_clear(brain):
    assert brain.custom_band is None
    brain.set_custom_band(13.0, 30.0)
    assert brain.custom_band == (13.0, 30.0) == brain.band
    brain.clear_custom_band()
    assert brain.custom_band is None and brain.band is None


def test_brain_band_follows_the_display_mode(brain):
    """The controls used to change only a label: nothing read them."""
    brain.set_display_mode("Alpha Power  (8–13 Hz)")
    assert brain.band == (8.0, 13.0)
    brain.set_display_mode("Source Activation")
    assert brain.band is None


def test_brain_set_band_picks_a_mode_or_a_custom_band(brain):
    brain.set_band((13.0, 30.0))
    assert brain.display_mode == "Beta Power   (13–30 Hz)" and brain.custom_band is None
    brain.set_band((9.0, 11.0))
    assert brain.custom_band == (9.0, 11.0)
    brain.set_band(None)
    assert brain.band is None


def test_brain_mode_change_restarts_the_smoothing(brain):
    """Or the next frames would show the previous band under the new label."""
    brain.update_from_arrays(np.ones(10242), np.ones(10242), deferred=True)
    assert brain._lh_ema is not None
    brain.set_display_mode("Theta Power  (4–8 Hz)")
    assert brain._lh_ema is None
    brain.set_display_mode(bp_module._DISPLAY_MODES[0])


def test_brain_draws_the_ico5_mesh(brain):
    """The full surface uploaded 16 times more data per frame for nothing."""
    assert brain._mesh.n_points == 2 * 10242


def test_brain_colours_hidden_hemisphere_and_threshold_as_cortex(brain):
    brain.update_from_arrays(np.full(10242, 0.5), np.full(10242, 0.5), deferred=True)
    base = (brain._base_rgb * 255).astype(np.uint8)
    n_lh = brain._n_lh
    assert not np.array_equal(brain._rgb[:n_lh], base[:n_lh])  # activity drawn
    brain._hemi_visible["lh"] = False
    brain._refresh_scalars(deferred=True)
    np.testing.assert_array_equal(brain._rgb[:n_lh], base[:n_lh])
    brain._hemi_visible["lh"] = True
    brain._threshold = 0.6  # above every value
    brain._refresh_scalars(deferred=True)
    np.testing.assert_array_equal(brain._rgb, base)
    brain._threshold = 0.0
    brain.reset_activity()


def test_brain_blanks_a_vertex_with_no_value_this_frame(brain):
    """Shown stale, a vertex without data looked like activity that was not there."""
    lh = np.full(10242, 0.5)
    brain.update_from_arrays(lh, lh, deferred=True)
    lh = lh.copy()
    lh[0] = np.nan
    brain.update_from_arrays(lh, np.full(10242, 0.5), deferred=True)
    assert np.isnan(brain._scalars_full[0])
    assert np.isfinite(brain._lh_ema[0])  # its average is kept for the next frame
    brain.reset_activity()


def test_to_display_places_values_by_vertex_number():
    from mne_rt.viz.brain_plot import N_ICO5, to_display

    lh, rh = to_display(np.array([1.0, 2.0, 3.0]), [np.array([0, 5]), np.array([7])])
    assert lh.shape == rh.shape == (N_ICO5,)
    assert lh[0] == 1.0 and lh[5] == 2.0 and rh[7] == 3.0
    assert np.isnan(lh[1]) and np.isnan(rh[0])


def test_brain_menu_follows_a_band_set_from_code(brain):
    """record_main sets the band after construction; the menu kept saying
    "Source Activation", and choosing it then changed nothing."""
    brain.set_band((8.0, 13.0))
    assert brain._mode_combo.currentText() == "Alpha Power  (8–13 Hz)"
    brain.set_band(None)
    assert brain._mode_combo.currentText() == "Source Activation"


def test_brain_clear_custom_band_goes_back_to_the_mode(brain):
    """Clear always fell back to activation, whatever the menu said."""
    brain.set_display_mode("Beta Power   (13–30 Hz)")
    brain.set_custom_band(15.0, 20.0)
    assert brain.band == (15.0, 20.0)
    brain.clear_custom_band()
    assert brain.band == (13.0, 30.0)
    brain.set_display_mode(bp_module._DISPLAY_MODES[0])
