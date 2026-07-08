"""Tests for simulate_nf_session."""

import numpy as np
import pytest

from mne_rt.tools.simulation import simulate_nf_session

# ------------------------------------------------------------------
# Basic output shape and types
# ------------------------------------------------------------------


def test_output_types():
    import mne

    raw, nf_state = simulate_nf_session(duration=2.0, sfreq=256.0, n_channels=64, rng_seed=0)
    assert isinstance(raw, mne.io.BaseRaw)
    assert isinstance(nf_state, np.ndarray)


def test_output_n_samples():
    raw, nf_state = simulate_nf_session(duration=4.0, sfreq=256.0, n_channels=64, rng_seed=0)
    expected_samples = int(4.0 * 256.0)
    assert raw.n_times == expected_samples
    assert nf_state.shape[0] == expected_samples


def test_output_n_channels():
    raw, _ = simulate_nf_session(duration=2.0, sfreq=256.0, n_channels=64, rng_seed=0)
    assert len(raw.ch_names) == 64


def test_sfreq_preserved():
    raw, _ = simulate_nf_session(duration=2.0, sfreq=512.0, n_channels=64, rng_seed=0)
    assert raw.info["sfreq"] == pytest.approx(512.0)


# ------------------------------------------------------------------
# nf_state properties
# ------------------------------------------------------------------


def test_nf_state_is_bool():
    _, nf_state = simulate_nf_session(duration=4.0, sfreq=256.0, n_channels=64, rng_seed=0)
    assert nf_state.dtype == bool


def test_nf_state_fraction():
    """nf_state True fraction should be close to nf_epoch_fraction."""
    frac = 0.4
    _, nf_state = simulate_nf_session(
        duration=10.0, sfreq=256.0, n_channels=64, nf_epoch_fraction=frac, rng_seed=0
    )
    observed = nf_state.sum() / len(nf_state)
    assert abs(observed - frac) < 0.05


def test_nf_state_zero_fraction():
    _, nf_state = simulate_nf_session(
        duration=4.0, sfreq=256.0, n_channels=64, nf_epoch_fraction=0.0, rng_seed=0
    )
    assert nf_state.sum() == 0


def test_nf_state_full_fraction():
    _, nf_state = simulate_nf_session(
        duration=4.0, sfreq=256.0, n_channels=64, nf_epoch_fraction=1.0, rng_seed=0
    )
    assert nf_state.all()


# ------------------------------------------------------------------
# Alpha reactivity
# ------------------------------------------------------------------


def test_alpha_reactivity_reduces_power():
    """Alpha power should be lower during NF-state epochs when reactivity=True."""
    raw_react, nf_state = simulate_nf_session(
        duration=20.0,
        sfreq=256.0,
        n_channels=64,
        alpha_reactivity=True,
        nf_epoch_fraction=0.5,
        rng_seed=42,
    )
    raw_no_react, _ = simulate_nf_session(
        duration=20.0,
        sfreq=256.0,
        n_channels=64,
        alpha_reactivity=False,
        nf_epoch_fraction=0.5,
        rng_seed=42,
    )
    data_react = raw_react.get_data()
    data_no = raw_no_react.get_data()
    # During NF-state samples, variance should be lower with reactivity
    nf_on = nf_state == True  # noqa: E712
    var_react_on = np.var(data_react[:, nf_on])
    var_no_on = np.var(data_no[:, nf_on])
    # Alpha reactivity reduces oscillation amplitude → lower variance during NF state
    assert var_react_on < var_no_on


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------


def test_rng_seed_reproducibility():
    kwargs = dict(duration=4.0, sfreq=256.0, n_channels=64, rng_seed=7)
    raw1, ns1 = simulate_nf_session(**kwargs)
    raw2, ns2 = simulate_nf_session(**kwargs)
    np.testing.assert_array_equal(raw1.get_data(), raw2.get_data())
    np.testing.assert_array_equal(ns1, ns2)


def test_different_seeds_give_different_data():
    kwargs = dict(duration=4.0, sfreq=256.0, n_channels=64)
    raw1, _ = simulate_nf_session(**kwargs, rng_seed=1)
    raw2, _ = simulate_nf_session(**kwargs, rng_seed=2)
    assert not np.allclose(raw1.get_data(), raw2.get_data())


# ------------------------------------------------------------------
# Data sanity
# ------------------------------------------------------------------


def test_data_has_no_nan_or_inf():
    raw, _ = simulate_nf_session(duration=4.0, sfreq=256.0, n_channels=64, rng_seed=0)
    data = raw.get_data()
    assert np.all(np.isfinite(data))


def test_data_amplitude_reasonable():
    """Peak amplitude should be in a physiologically plausible range (< 1 mV)."""
    raw, _ = simulate_nf_session(duration=4.0, sfreq=256.0, n_channels=64, rng_seed=0)
    data = raw.get_data()
    assert np.max(np.abs(data)) < 1e-3  # < 1 mV
