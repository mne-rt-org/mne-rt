"""Tests for mne_rt.tools.tools (spectral/utility helpers)."""

import math
from pathlib import Path

import numpy as np
import pytest

from mne_rt.tools.tools import (
    butter_bandpass,
    compute_bandpower,
    compute_fft,
    compute_instantaneous_phase,
    get_canonical_freqs,
    get_params,
    log_degree_barrier,
    remove_blinks_lms,
    timed,
    weight_to_degree_map,
)

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
    phase_idx, amp_idx = compute_instantaneous_phase(data, SFREQ, (8.0, 13.0), channel_indices=[0])
    # Single channel = same signal as channel-mean for identical channels
    assert phase_all == pytest.approx(phase_idx, abs=1e-6)
    assert amp_all == pytest.approx(amp_idx, abs=1e-6)


def test_channel_indices_subset():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((N_CHANNELS, N_SAMPLES))
    # Should not raise with a subset of indices
    phase, amp = compute_instantaneous_phase(data, SFREQ, (8.0, 13.0), channel_indices=[0, 1, 2])
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


# ------------------------------------------------------------------
# timed
# ------------------------------------------------------------------


def test_timed_returns_value_and_elapsed():
    @timed
    def add(a, b):
        return a + b

    value, elapsed = add(2, 3)
    assert value == 5
    assert elapsed >= 0.0


def test_timed_preserves_exceptions():
    @timed
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        boom()


# ------------------------------------------------------------------
# get_canonical_freqs
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("delta", [0.5, 4]),
        ("theta", [4, 8]),
        ("alpha", [8, 13]),
        ("beta", [15, 30]),
        ("gamma", [30, 80]),
        ("smr", [12, 15]),
    ],
)
def test_get_canonical_freqs_known_bands(name, expected):
    assert get_canonical_freqs(name) == expected


def test_get_canonical_freqs_unknown_raises():
    with pytest.raises(ValueError, match="not defined"):
        get_canonical_freqs("not_a_band")


# ------------------------------------------------------------------
# get_params
# ------------------------------------------------------------------


@pytest.fixture()
def config_file():
    return str(Path(__file__).parent.parent / "src" / "mne_rt" / "config_methods.yml")


def test_get_params_returns_defaults(config_file):
    params = get_params(config_file, "sensor_power", {})
    assert params["frange"] == [8, 13]
    assert params["method"] == "welch"


def test_get_params_none_overrides(config_file):
    params = get_params(config_file, "sensor_power", None)
    assert params["method"] == "welch"


def test_get_params_unknown_modality_raises(config_file):
    with pytest.raises(ValueError, match="Unknown modality"):
        get_params(config_file, "not_a_modality", {})


def test_get_params_unknown_override_key_raises(config_file):
    with pytest.raises(ValueError, match="Unknown method"):
        get_params(config_file, "sensor_power", {"not_a_param": {}})


def test_get_params_does_not_mutate_config_between_calls(config_file):
    """Overrides on one call must not leak defaults into a later call."""
    params1 = get_params(config_file, "sensor_power", {})
    params1["frange"][0] = 999
    params2 = get_params(config_file, "sensor_power", {})
    assert params2["frange"] == [8, 13]


# ------------------------------------------------------------------
# compute_bandpower
# ------------------------------------------------------------------

_BP_SFREQ = 256.0
_BP_N_SAMPLES = int(_BP_SFREQ * 2)


def _alpha_data(n_ch=4, n_samp=_BP_N_SAMPLES, sfreq=_BP_SFREQ, freq=10.0):
    t = np.arange(n_samp) / sfreq
    return np.tile(np.sin(2 * np.pi * freq * t), (n_ch, 1))


@pytest.mark.parametrize("method", ["fft", "periodogram", "welch", "multitaper"])
def test_compute_bandpower_methods(method):
    data = _alpha_data()
    bp = compute_bandpower(data, _BP_SFREQ, (8.0, 13.0), method=method)
    assert bp.shape == (4,)
    assert np.all(np.isfinite(bp))


def test_compute_bandpower_relative_in_unit_interval():
    data = _alpha_data()
    bp = compute_bandpower(data, _BP_SFREQ, (8.0, 13.0), method="welch", relative=True)
    assert np.all(bp >= 0.0) and np.all(bp <= 1.0 + 1e-9)


def test_compute_bandpower_alpha_dominant():
    """A pure 10 Hz sine should show far more power in alpha than in delta."""
    data = _alpha_data(freq=10.0)
    alpha_bp = compute_bandpower(data, _BP_SFREQ, (8.0, 13.0), method="welch", relative=False)
    delta_bp = compute_bandpower(data, _BP_SFREQ, (0.5, 4.0), method="welch", relative=False)
    assert np.all(alpha_bp > delta_bp)


def test_compute_bandpower_unsupported_method_raises():
    data = _alpha_data()
    with pytest.raises(ValueError, match="Unsupported method"):
        compute_bandpower(data, _BP_SFREQ, (8.0, 13.0), method="not_a_method")


# ------------------------------------------------------------------
# compute_fft
# ------------------------------------------------------------------


def test_compute_fft_shapes():
    fft_window, freq_band, freq_band_idxs, frequencies = compute_fft(
        sfreq=256.0, winsize=1.0, freq_range=(8.0, 13.0)
    )
    assert fft_window.shape == (256,)
    assert len(freq_band) == len(freq_band_idxs)
    assert np.all(frequencies[freq_band_idxs] == freq_band)


def test_compute_fft_freq_band_within_range():
    _, freq_band, _, _ = compute_fft(sfreq=256.0, winsize=1.0, freq_range=(8.0, 13.0))
    assert np.all(freq_band >= 8.0) and np.all(freq_band <= 13.0)


# ------------------------------------------------------------------
# butter_bandpass
# ------------------------------------------------------------------


def test_butter_bandpass_sos_shape():
    sos = butter_bandpass(l_freq=8.0, h_freq=13.0, sfreq=256.0, order=4)
    assert sos.ndim == 2
    assert sos.shape[1] == 6  # second-order-sections format


def test_butter_bandpass_filters_out_of_band_signal():
    from scipy.signal import sosfiltfilt

    sfreq = 256.0
    t = np.arange(int(sfreq * 2)) / sfreq
    # Out-of-band 1 Hz signal should be strongly attenuated by an 8-13 Hz filter
    signal = np.sin(2 * np.pi * 1.0 * t)
    sos = butter_bandpass(l_freq=8.0, h_freq=13.0, sfreq=sfreq, order=4)
    filtered = sosfiltfilt(sos, signal)
    assert np.std(filtered) < np.std(signal) * 0.1


# ------------------------------------------------------------------
# remove_blinks_lms
# ------------------------------------------------------------------


def test_remove_blinks_lms_shape_preserved():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 500))
    cleaned = remove_blinks_lms(data, ref_ch_idx=0)
    assert cleaned.shape == data.shape


def test_remove_blinks_lms_reference_channel_unchanged():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 500))
    cleaned = remove_blinks_lms(data, ref_ch_idx=0)
    np.testing.assert_array_equal(cleaned[0], data[0])


def test_remove_blinks_lms_attenuates_correlated_artifact():
    """A channel that is a scaled copy of the reference should be strongly
    attenuated after LMS adaptation."""
    rng = np.random.default_rng(1)
    n_times = 2000
    ref = rng.standard_normal(n_times)
    data = np.zeros((2, n_times))
    data[0] = ref
    data[1] = 0.8 * ref + 0.01 * rng.standard_normal(n_times)
    cleaned = remove_blinks_lms(data, ref_ch_idx=0, n_taps=1, mu=0.05)
    # Late-adaptation residual should be much smaller than the original artifact
    tail = slice(n_times // 2, None)
    assert np.std(cleaned[1, tail]) < np.std(data[1, tail]) * 0.5


# ------------------------------------------------------------------
# weight_to_degree_map
# ------------------------------------------------------------------


def test_weight_to_degree_map_shapes():
    n_nodes = 5
    n_edges = n_nodes * (n_nodes - 1) // 2
    k, kt = weight_to_degree_map(n_nodes)
    w = np.ones(n_edges)
    d = k(w)
    assert d.shape == (n_nodes,)
    w_back = kt(d)
    assert w_back.shape == (n_edges,)


def test_weight_to_degree_map_unit_weights_give_complete_graph_degree():
    """With all edge weights = 1, every node's degree equals n_nodes - 1
    (a fully-connected unweighted graph)."""
    n_nodes = 4
    n_edges = n_nodes * (n_nodes - 1) // 2
    k, _ = weight_to_degree_map(n_nodes)
    d = k(np.ones(n_edges))
    np.testing.assert_allclose(d, np.full(n_nodes, n_nodes - 1))


# ------------------------------------------------------------------
# log_degree_barrier (2-node analytic fast path — no pyunlocbox needed)
# ------------------------------------------------------------------


def test_log_degree_barrier_two_node_shape():
    signals = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    mat = log_degree_barrier(signals, dist_type="euclidean", alpha=1.0, beta=0.1)
    assert mat.shape == (2, 2)


def test_log_degree_barrier_two_node_symmetric_zero_diagonal():
    signals = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    mat = log_degree_barrier(signals, dist_type="euclidean", alpha=1.0, beta=0.1)
    assert mat[0, 0] == 0.0 and mat[1, 1] == 0.0
    assert mat[0, 1] == pytest.approx(mat[1, 0])


def test_log_degree_barrier_two_node_nonnegative():
    signals = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    mat = log_degree_barrier(signals, dist_type="euclidean", alpha=1.0, beta=0.1)
    assert np.all(mat >= 0.0)
