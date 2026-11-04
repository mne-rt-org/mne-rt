"""Tests for ASRDenoiser (Artifact Subspace Reconstruction)."""

import numpy as np
import pytest

RNG = np.random.default_rng(42)
N_CH = 16
SFREQ = 256.0
N_T = int(SFREQ * 4)


@pytest.fixture()
def asr():
    from mne_rt.tools import ASRDenoiser

    a = ASRDenoiser(cutoff=3.0)
    data = RNG.standard_normal((N_CH, N_T)) * 1e-6
    a.fit(data, SFREQ)
    return a


# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------


def test_invalid_cutoff():
    from mne_rt.tools import ASRDenoiser

    with pytest.raises(ValueError):
        ASRDenoiser(cutoff=-1)


def test_invalid_dropout_fraction():
    from mne_rt.tools import ASRDenoiser

    with pytest.raises(ValueError):
        ASRDenoiser(max_dropout_fraction=1.1)


def test_invalid_window_overlap():
    from mne_rt.tools import ASRDenoiser

    with pytest.raises(ValueError):
        ASRDenoiser(window_overlap=1.0)


# ------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------


def test_fit_returns_self():
    from mne_rt.tools import ASRDenoiser

    a = ASRDenoiser(cutoff=3.0)
    data = RNG.standard_normal((N_CH, N_T)) * 1e-6
    result = a.fit(data, SFREQ)
    assert result is a


def test_thresholds_shape(asr):
    assert asr.thresholds.shape == (N_CH,)


def test_eigenvectors_shape(asr):
    assert asr.eigenvectors.shape == (N_CH, N_CH)


# ------------------------------------------------------------------
# Pre-fit guards
# ------------------------------------------------------------------


def test_thresholds_before_fit():
    from mne_rt.tools import ASRDenoiser

    a = ASRDenoiser()
    with pytest.raises(RuntimeError):
        _ = a.thresholds


def test_transform_before_fit():
    from mne_rt.tools import ASRDenoiser

    a = ASRDenoiser()
    data = RNG.standard_normal((N_CH, N_T)) * 1e-6
    with pytest.raises(RuntimeError):
        a.transform(data)


# ------------------------------------------------------------------
# Transform
# ------------------------------------------------------------------


def test_transform_clean_data_noop(asr):
    """Transform of low-amplitude clean data returns same shape, close to input."""
    clean = RNG.standard_normal((N_CH, N_T)) * 1e-7
    out = asr.transform(clean)
    assert out.shape == clean.shape
    np.testing.assert_allclose(out, clean, rtol=1e-5, atol=1e-15)


def test_transform_removes_artifact(asr):
    """A large Gaussian spike injected into a window is attenuated after transform."""
    rng_local = np.random.default_rng(123)
    amplitude = 1e-6

    # Build test window: clean background + large burst in the first half
    data = rng_local.standard_normal((N_CH, N_T)) * amplitude
    half = N_T // 2
    spike = rng_local.standard_normal((N_CH, half)) * (amplitude * 50)
    data[:, :half] += spike

    out = asr.transform(data)
    assert out.max() < data.max()


def test_transform_output_shape(asr):
    data = RNG.standard_normal((N_CH, N_T)) * 1e-6
    out = asr.transform(data)
    assert out.shape == data.shape


# ------------------------------------------------------------------
# Repr & properties
# ------------------------------------------------------------------


def test_repr(asr):
    r = repr(asr)
    assert "ASRDenoiser" in r
    assert "fitted" in r


def test_thresholds_all_positive(asr):
    assert np.all(asr.thresholds > 0)
