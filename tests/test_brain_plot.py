"""Tests for mne_rt.viz.brain_plot.BrainPlot.

The module itself is always importable (pyvista/pyvistaqt imports are
guarded internally), so constant-consistency and the "missing optional
dependency" error path are tested unconditionally.  Tests that construct a
real ``BrainPlot`` instance require the ``viz`` extra (pyvista + pyvistaqt)
and are skipped when it is not installed.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from mne_rt.viz import brain_plot as bp_module
from mne_rt.viz.brain_plot import BrainPlot

_VIZ_AVAILABLE = bp_module._pyvista_available and bp_module._pyvistaqt_available
_viz_required = pytest.mark.skipif(not _VIZ_AVAILABLE, reason="pyvista/pyvistaqt not installed")

# ------------------------------------------------------------------
# Module-level constant consistency (no pyvista required)
# ------------------------------------------------------------------


def test_display_modes_have_clim_hints():
    for mode in bp_module._DISPLAY_MODES:
        assert mode in bp_module._DISPLAY_CLIM_HINTS


def test_clim_hints_are_two_tuples():
    for hint in bp_module._DISPLAY_CLIM_HINTS.values():
        assert len(hint) == 2
        assert hint[0] < hint[1]


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
    brain._custom_lo_spin.setValue(13.0)
    brain._custom_hi_spin.setValue(30.0)
    brain._set_custom_band()
    assert brain.custom_band == (13.0, 30.0)
    brain._clear_custom_band()
    assert brain.custom_band is None
