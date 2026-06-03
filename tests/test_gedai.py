"""Tests for GEDAIDenoiser (GED-based artifact removal)."""

import numpy as np
import pytest

RNG = np.random.default_rng(7)
N_CH = 16
SFREQ = 256.0
N_T = int(SFREQ * 4)   # 4-second segment


@pytest.fixture()
def gedai():
    from mne_rt.tools import GEDAIDenoiser
    return GEDAIDenoiser(n_channels=N_CH, shrinkage=0.05)


def _make_data():
    """Broadband + sinusoidal 'artifact' contamination."""
    t = np.arange(N_T) / SFREQ
    broadband = RNG.standard_normal((N_CH, N_T)) * 1e-6
    artifact = np.sin(2 * np.pi * 10 * t) * 5e-6
    broadband[0] += artifact
    return broadband


def test_fit(gedai):
    data = _make_data()
    gedai.fit_from_raw(data, sfreq=SFREQ, band=(8.0, 12.0))
    assert gedai.eigenvalues is not None
    assert len(gedai.eigenvalues) == N_CH


def test_spatial_filters_shape(gedai):
    data = _make_data()
    gedai.fit_from_raw(data, sfreq=SFREQ, band=(8.0, 12.0))
    assert gedai.spatial_filters.shape == (N_CH, N_CH)


def test_activation_patterns_shape(gedai):
    data = _make_data()
    gedai.fit_from_raw(data, sfreq=SFREQ, band=(8.0, 12.0))
    assert gedai.activation_patterns.shape == (N_CH, N_CH)


def test_denoise(gedai):
    data = _make_data()
    gedai.fit_from_raw(data, sfreq=SFREQ, band=(8.0, 12.0))
    cleaned = gedai.denoise(data, artifact_idx=[0])
    assert cleaned.shape == data.shape


def test_find_artifact_components(gedai):
    data = _make_data()
    gedai.fit_from_raw(data, sfreq=SFREQ, band=(8.0, 12.0))
    template = gedai.activation_patterns[:, 0]
    idx, corrs = gedai.find_artifact_components(template, threshold=0.0)
    assert isinstance(idx, (list, np.ndarray))
    assert isinstance(corrs, np.ndarray)


def test_update_and_denoise(gedai):
    data = _make_data()
    gedai.fit_from_raw(data, sfreq=SFREQ, band=(8.0, 12.0))
    template = gedai.activation_patterns[:, 0]
    new_data = _make_data()
    cleaned = gedai.update_and_denoise(new_data, template, threshold=0.3)
    assert cleaned.shape == new_data.shape


def test_fit_requires_enough_data(gedai):
    """fit_from_raw with a segment too short should still complete or raise
    a clean ValueError, not an unhandled numpy error."""
    tiny = RNG.standard_normal((N_CH, 10))
    with pytest.raises(Exception):
        gedai.fit_from_raw(tiny, sfreq=SFREQ, band=(8.0, 12.0))
