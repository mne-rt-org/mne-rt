"""Tests for AdaptiveLMSFilter (real-time LMS artifact removal)."""

from __future__ import annotations

import numpy as np
import pytest

from mne_rt.tools.lms import AdaptiveLMSFilter

RNG = np.random.default_rng(0)
N_CH = 6
N_T = 500

# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------


def test_invalid_mu_raises():
    with pytest.raises(ValueError, match="mu"):
        AdaptiveLMSFilter(mu=0.0)


def test_negative_mu_raises():
    with pytest.raises(ValueError, match="mu"):
        AdaptiveLMSFilter(mu=-0.01)


def test_invalid_n_taps_raises():
    with pytest.raises(ValueError, match="n_taps"):
        AdaptiveLMSFilter(n_taps=0)


def test_default_params():
    filt = AdaptiveLMSFilter()
    assert filt.ref_ch_idx == 0
    assert filt.n_taps == 5
    assert filt.mu == pytest.approx(0.01)
    assert filt.weights_ is None


# ------------------------------------------------------------------
# fit (no-op)
# ------------------------------------------------------------------


def test_fit_returns_self():
    filt = AdaptiveLMSFilter()
    assert filt.fit() is filt
    assert filt.fit(raw_info="ignored", extra=1) is filt


# ------------------------------------------------------------------
# transform
# ------------------------------------------------------------------


def test_transform_shape_preserved():
    filt = AdaptiveLMSFilter()
    data = RNG.standard_normal((N_CH, N_T))
    out = filt.transform(data)
    assert out.shape == data.shape


def test_transform_sets_weights_after_first_call():
    filt = AdaptiveLMSFilter(n_taps=5)
    assert filt.weights_ is None
    data = RNG.standard_normal((N_CH, N_T))
    filt.transform(data)
    assert filt.weights_ is not None
    assert filt.weights_.shape == (N_CH, 5)


def test_reference_channel_passes_through_unchanged():
    filt = AdaptiveLMSFilter(ref_ch_idx=2)
    data = RNG.standard_normal((N_CH, N_T))
    out = filt.transform(data)
    np.testing.assert_array_equal(out[2], data[2])


def test_weights_persist_across_calls():
    """Successive transform() calls continue adapting rather than resetting."""
    filt = AdaptiveLMSFilter(n_taps=3)
    data1 = RNG.standard_normal((N_CH, N_T))
    filt.transform(data1)
    weights_after_first = filt.weights_.copy()

    data2 = RNG.standard_normal((N_CH, N_T))
    filt.transform(data2)
    # Weights should have kept adapting, not reset to zero
    assert not np.allclose(filt.weights_, weights_after_first)
    assert not np.allclose(filt.weights_, np.zeros_like(filt.weights_))


def test_attenuates_correlated_artifact():
    """A channel that is a scaled copy of the reference should be attenuated
    after LMS adaptation converges."""
    rng = np.random.default_rng(1)
    n_times = 2000
    ref = rng.standard_normal(n_times)
    data = np.zeros((2, n_times))
    data[0] = ref
    data[1] = 0.8 * ref + 0.01 * rng.standard_normal(n_times)

    filt = AdaptiveLMSFilter(ref_ch_idx=0, n_taps=1, mu=0.05)
    cleaned = filt.transform(data)

    tail = slice(n_times // 2, None)
    assert np.std(cleaned[1, tail]) < np.std(data[1, tail]) * 0.5


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------


def test_reset_clears_weights():
    filt = AdaptiveLMSFilter()
    data = RNG.standard_normal((N_CH, N_T))
    filt.transform(data)
    assert filt.weights_ is not None
    filt.reset()
    assert filt.weights_ is None


def test_reset_then_transform_restarts_adaptation():
    filt = AdaptiveLMSFilter(n_taps=3)
    data = RNG.standard_normal((N_CH, N_T))
    filt.transform(data)
    filt.reset()
    filt.transform(data)
    # Should behave the same as a freshly constructed filter given the same input
    fresh = AdaptiveLMSFilter(n_taps=3)
    fresh.transform(data)
    np.testing.assert_allclose(filt.weights_, fresh.weights_)


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------


def test_repr_unfitted():
    filt = AdaptiveLMSFilter()
    r = repr(filt)
    assert "AdaptiveLMSFilter" in r
    assert "adapted=False" in r


def test_repr_after_transform():
    filt = AdaptiveLMSFilter()
    data = RNG.standard_normal((N_CH, N_T))
    filt.transform(data)
    r = repr(filt)
    assert "adapted=True" in r
