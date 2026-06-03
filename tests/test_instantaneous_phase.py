"""Tests for compute_instantaneous_phase."""

import math

import numpy as np
import pytest

from mne_rt.tools.tools import compute_instantaneous_phase


SFREQ = 512.0
DURATION = 2.0  # seconds — long enough for sosfiltfilt edge handling
N_SAMPLES = int(SFREQ * DURATION)
N_CHANNELS = 8


def _sine_data(freq_hz, sfreq=SFREQ, n_samples=N_SAMPLES, n_channels=N_CHANNELS, amplitude=1.0):
    """Pure sinusoid at freq_hz, same signal on all channels."""
    t = np.arange(n_samples) / sfreq
    signal = amplitude * np.sin(2 * math.pi * freq_hz * t)
    return np.tile(signal, (n_channels, 1))


# ------------------------------------------------------------------
# Phase is in [-π, π]
# ------------------------------------------------------------------

def test_phase_range_random():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((N_CHANNELS, N_SAMPLES))
    phase, _ = compute_instantaneous_phase(data, SFREQ, (8.0, 13.0))
    assert -math.pi <= phase <= math.pi


def test_phase_range_pure_sine():
    data = _sine_data(10.0)
    phase, _ = compute_instantaneous_phase(data, SFREQ, (8.0, 13.0))
    assert -math.pi <= phase <= math.pi


# ------------------------------------------------------------------
# Amplitude is non-negative
# ------------------------------------------------------------------

def test_amplitude_non_negative_random():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((N_CHANNELS, N_SAMPLES))
    _, amplitude = compute_instantaneous_phase(data, SFREQ, (8.0, 13.0))
    assert amplitude >= 0.0


def test_amplitude_non_negative_sine():
    data = _sine_data(10.0)
    _, amplitude = compute_instantaneous_phase(data, SFREQ, (8.0, 13.0))
    assert amplitude >= 0.0


# ------------------------------------------------------------------
# Amplitude scales with input amplitude
# ------------------------------------------------------------------

def test_amplitude_proportional_to_input():
    data_1 = _sine_data(10.0, amplitude=1.0)
    data_10 = _sine_data(10.0, amplitude=10.0)
    _, amp_1 = compute_instantaneous_phase(data_1, SFREQ, (8.0, 13.0))
    _, amp_10 = compute_instantaneous_phase(data_10, SFREQ, (8.0, 13.0))
    assert amp_10 == pytest.approx(amp_1 * 10.0, rel=0.05)


# ------------------------------------------------------------------
# Out-of-band signal produces near-zero amplitude
# ------------------------------------------------------------------

def test_out_of_band_signal_low_amplitude():
    """Signal at 1 Hz should have very low amplitude when filtered to 8–13 Hz."""
    data = _sine_data(1.0, amplitude=100.0)
    _, amp = compute_instantaneous_phase(data, SFREQ, (8.0, 13.0))
    # After bandpass filtering, out-of-band content should be much attenuated
    assert amp < 10.0


# ------------------------------------------------------------------
# channel_indices parameter
# ------------------------------------------------------------------

def test_channel_indices_single_channel():
    data = _sine_data(10.0)
    phase_all, amp_all = compute_instantaneous_phase(data, SFREQ, (8.0, 13.0))
    phase_idx, amp_idx = compute_instantaneous_phase(
        data, SFREQ, (8.0, 13.0), channel_indices=[0]
    )
    # Single channel = same signal as channel-mean for identical channels
    assert phase_all == pytest.approx(phase_idx, abs=1e-6)
    assert amp_all == pytest.approx(amp_idx, abs=1e-6)


def test_channel_indices_subset():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((N_CHANNELS, N_SAMPLES))
    # Should not raise with a subset of indices
    phase, amp = compute_instantaneous_phase(
        data, SFREQ, (8.0, 13.0), channel_indices=[0, 1, 2]
    )
    assert -math.pi <= phase <= math.pi
    assert amp >= 0.0


def test_channel_indices_none_uses_all():
    """channel_indices=None and channel_indices=[all] should give same result."""
    data = _sine_data(10.0)
    phase_none, amp_none = compute_instantaneous_phase(
        data, SFREQ, (8.0, 13.0), channel_indices=None
    )
    phase_all, amp_all = compute_instantaneous_phase(
        data, SFREQ, (8.0, 13.0), channel_indices=list(range(N_CHANNELS))
    )
    assert phase_none == pytest.approx(phase_all, abs=1e-10)
    assert amp_none == pytest.approx(amp_all, abs=1e-10)


# ------------------------------------------------------------------
# Return types
# ------------------------------------------------------------------

def test_return_types():
    data = _sine_data(10.0)
    phase, amp = compute_instantaneous_phase(data, SFREQ, (8.0, 13.0))
    assert isinstance(phase, float)
    assert isinstance(amp, float)
