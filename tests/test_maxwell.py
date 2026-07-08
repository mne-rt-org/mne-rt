"""Tests for RTMaxwellFilter (real-time SSS / tSSS Maxwell filtering)."""

import numpy as np
import pytest

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def meg_info():
    """Return MEG info from MNE sample data, or skip if unavailable."""
    pytest.importorskip("mne")
    try:
        import mne

        data_path = mne.datasets.sample.data_path(download=False, verbose=False)
        fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
        if not fname.exists():
            pytest.skip("MNE sample data not downloaded")
        raw = mne.io.read_raw_fif(fname, preload=False, verbose=False)
        return raw.info
    except Exception:
        pytest.skip("MNE sample data not available")


# ------------------------------------------------------------------
# Group A — interface tests (no MEG data needed)
# ------------------------------------------------------------------


def test_invalid_int_order():
    from mne_rt.tools import RTMaxwellFilter

    with pytest.raises(ValueError):
        RTMaxwellFilter(int_order=0)


def test_invalid_ext_order():
    from mne_rt.tools import RTMaxwellFilter

    with pytest.raises(ValueError):
        RTMaxwellFilter(ext_order=-1)


def test_invalid_st_correlation():
    from mne_rt.tools import RTMaxwellFilter

    with pytest.raises(ValueError):
        RTMaxwellFilter(st_correlation=1.5)


def test_mode_sss():
    from mne_rt.tools import RTMaxwellFilter

    assert RTMaxwellFilter().mode == "sss"


def test_mode_tsss():
    from mne_rt.tools import RTMaxwellFilter

    assert RTMaxwellFilter(st_duration=10.0).mode == "tsss"


def test_transform_before_fit():
    from mne_rt.tools import RTMaxwellFilter

    filt = RTMaxwellFilter()
    data = RNG.standard_normal((306, 250))
    with pytest.raises(RuntimeError):
        filt.transform(data)


def test_sss_projector_before_fit():
    from mne_rt.tools import RTMaxwellFilter

    filt = RTMaxwellFilter()
    with pytest.raises(RuntimeError):
        _ = filt.sss_projector


def test_repr_unfitted():
    from mne_rt.tools import RTMaxwellFilter

    filt = RTMaxwellFilter()
    r = repr(filt)
    assert "RTMaxwellFilter" in r
    assert "not fitted" in r


def test_repr_fitted_state(meg_info):
    from mne_rt.tools import RTMaxwellFilter

    filt = RTMaxwellFilter()
    filt.fit(meg_info)
    r = repr(filt)
    assert "fitted" in r


def test_default_params():
    from mne_rt.tools import RTMaxwellFilter

    filt = RTMaxwellFilter()
    assert filt.int_order == 8
    assert filt.ext_order == 3
    assert filt.st_duration is None
    assert filt.st_correlation == 0.98
    assert filt.st_update_interval == 1
    assert filt.calibration is None
    assert filt.cross_talk is None
    assert filt.coord_frame == "head"
    assert filt.regularize == "in"
    assert filt.mag_scale == 100.0


# ------------------------------------------------------------------
# Group B — integration tests (skip if MNE sample data absent)
# ------------------------------------------------------------------


def test_fit_sss_sets_fitted(meg_info):
    from mne_rt.tools import RTMaxwellFilter

    filt = RTMaxwellFilter()
    filt.fit(meg_info)
    assert filt._fitted is True


def test_sss_projector_shape(meg_info):
    from mne_rt.tools import RTMaxwellFilter

    filt = RTMaxwellFilter()
    filt.fit(meg_info)
    P = filt.sss_projector
    assert isinstance(P, np.ndarray)
    assert P.ndim == 2


def test_transform_shape(meg_info):
    import mne

    from mne_rt.tools import RTMaxwellFilter

    filt = RTMaxwellFilter()
    filt.fit(meg_info)

    n_ch = len(meg_info["ch_names"])
    data = RNG.standard_normal((n_ch, 250))
    out = filt.transform(data)
    assert out.shape == data.shape
