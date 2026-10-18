"""Tests for AdaptiveMultibandGEDAIDenoiser (real-time adaptive multiband GEDAI)."""

import numpy as np
import pytest

# gedai is an opt-in extra (`mne-rt[gedai]`); every test here exercises it.
pytest.importorskip("gedai")

RNG = np.random.default_rng(11)
N_CH = 8
SFREQ = 256.0
N_T_BASELINE = int(SFREQ * 8)  # 8-second baseline
CHUNK_SAMPLES = int(SFREQ * 0.1)  # 100 ms real-time chunk


@pytest.fixture()
def denoiser():
    gedai = pytest.importorskip("gedai")  # noqa: F841 — skip cleanly if not installed
    from mne_rt.tools import AdaptiveMultibandGEDAIDenoiser

    return AdaptiveMultibandGEDAIDenoiser(
        n_channels=N_CH, sfreq=SFREQ, wavelet_level=3, cycles_per_wavelet=4
    )


def _make_baseline():
    t = np.arange(N_T_BASELINE) / SFREQ
    broadband = RNG.standard_normal((N_CH, N_T_BASELINE)) * 1e-6
    artifact = np.sin(2 * np.pi * 10 * t) * 5e-6
    broadband[0] += artifact
    return broadband


def _make_chunk():
    return RNG.standard_normal((N_CH, CHUNK_SAMPLES)) * 1e-6


def test_fit(denoiser):
    denoiser.fit(_make_baseline())
    assert denoiser.n_bands == 4  # wavelet_level=3 -> 4 bands
    assert len(denoiser.eigenvalues) == denoiser.n_bands
    for eigvals in denoiser.eigenvalues:
        assert len(eigvals) == N_CH


def test_band_frequencies_cover_spectrum(denoiser):
    denoiser.fit(_make_baseline())
    freqs = denoiser.band_frequencies
    assert len(freqs) == denoiser.n_bands
    assert freqs[0][1] == pytest.approx(SFREQ / 2)  # finest band upper edge
    assert freqs[-1][0] == 0.0  # coarsest band is a lowpass


def test_time_constants_scale_with_band(denoiser):
    """Lower-frequency (coarser) bands should have longer time constants."""
    denoiser.fit(_make_baseline())
    taus = denoiser.band_time_constants
    assert len(taus) == denoiser.n_bands
    # band 0 is finest/highest frequency, last band is coarsest/lowest frequency
    assert taus[0] < taus[-1]


def test_spatial_filters_and_patterns_shape(denoiser):
    denoiser.fit(_make_baseline())
    for W, A in zip(denoiser.spatial_filters, denoiser.activation_patterns):
        assert W.shape == (N_CH, N_CH)
        assert A.shape == (N_CH, N_CH)


def test_update_and_denoise_shape(denoiser):
    denoiser.fit(_make_baseline())
    chunk = _make_chunk()
    cleaned = denoiser.update_and_denoise(chunk)
    assert cleaned.shape == chunk.shape


def test_update_and_denoise_with_template(denoiser):
    denoiser.fit(_make_baseline())
    template = denoiser.activation_patterns[0][:, 0]
    chunk = _make_chunk()
    cleaned = denoiser.update_and_denoise(chunk, template_map=template, threshold=0.0)
    assert cleaned.shape == chunk.shape


def test_streaming_updates_are_stateful(denoiser):
    """Calling update_and_denoise repeatedly should not error and should keep
    adapting the per-band covariance (filters may change across calls)."""
    denoiser.fit(_make_baseline())
    W0 = [W.copy() for W in denoiser.spatial_filters]
    for _ in range(5):
        denoiser.update_and_denoise(_make_chunk())
    W1 = denoiser.spatial_filters
    assert any(not np.allclose(a, b) for a, b in zip(W0, W1))


def test_fit_with_leadfield():
    pytest.importorskip("gedai")
    from mne_rt.tools import AdaptiveMultibandGEDAIDenoiser

    denoiser = AdaptiveMultibandGEDAIDenoiser(n_channels=N_CH, sfreq=SFREQ, wavelet_level=2)
    leadfield = RNG.standard_normal((N_CH, N_CH * 2))
    denoiser.fit(_make_baseline(), leadfield=leadfield)
    cleaned = denoiser.update_and_denoise(_make_chunk())
    assert cleaned.shape == (N_CH, CHUNK_SAMPLES)


def test_requires_fit_before_use():
    pytest.importorskip("gedai")
    from mne_rt.tools import AdaptiveMultibandGEDAIDenoiser

    denoiser = AdaptiveMultibandGEDAIDenoiser(n_channels=N_CH, sfreq=SFREQ, wavelet_level=2)
    with pytest.raises(RuntimeError):
        denoiser.update_and_denoise(_make_chunk())


def test_wrong_channel_count_raises(denoiser):
    denoiser.fit(_make_baseline())
    with pytest.raises(ValueError):
        denoiser.update_and_denoise(RNG.standard_normal((N_CH + 1, CHUNK_SAMPLES)))


@pytest.mark.parametrize("broadband_pass", [False, True])
@pytest.mark.parametrize("prefilter_mode", ["reference", "lowcut", "none"])
def test_no_rejection_matches_delayed_pygedai_reconstruction(
    denoiser, broadband_pass, prefilter_mode
):
    from gedai.wavelet.transform import _apply_wavelet_highpass_prefilter, _modwt_haar_single_band

    rng = np.random.default_rng(40)
    data = rng.normal(size=(N_CH, 16000))
    original = data.copy()
    baseline_samples = 4096
    denoiser.broadband_pass = broadband_pass
    denoiser.prefilter_mode = prefilter_mode
    denoiser.fit(data[:, :baseline_samples])
    output = np.concatenate(
        [
            denoiser.update_and_denoise(data[:, i : i + 64], n_artifact=0, refit=False)
            for i in range(baseline_samples, data.shape[1], 64)
        ],
        axis=1,
    )
    expected = data
    if prefilter_mode == "reference":
        expected = _apply_wavelet_highpass_prefilter(expected, SFREQ, 0.1, engine="numpy")
    if prefilter_mode == "lowcut" or (prefilter_mode == "reference" and broadband_pass):
        expected = _apply_wavelet_highpass_prefilter(
            expected, SFREQ, denoiser.wavelet_low_cutoff, engine="numpy"
        )
        if prefilter_mode == "reference":
            expected = _apply_wavelet_highpass_prefilter(expected, SFREQ, 0.1, engine="numpy")
    expected = sum(
        _modwt_haar_single_band(expected.T, denoiser.wavelet_level, i, engine="numpy")
        for i, keep in enumerate(denoiser.retained_bands)
        if keep
    )
    first, last = baseline_samples + 1024, data.shape[1] - denoiser.delay_samples - 1024
    offset = denoiser.delay_samples - baseline_samples
    np.testing.assert_allclose(
        output[:, first + offset : last + offset], expected[:, first:last], atol=1e-12
    )
    assert denoiser.latency == denoiser.delay_samples / SFREQ
    np.testing.assert_array_equal(data, original)


def test_leadfield_rejects_high_not_low_eigenvalue(denoiser):
    rng = np.random.default_rng(41)
    baseline = rng.normal(size=(N_CH, 4096))
    baseline[0] *= 100
    data = rng.normal(size=(N_CH, 1024))
    data[0] *= 100
    denoiser.broadband_pass = False
    denoiser.fit(baseline, leadfield=np.eye(N_CH))
    band = denoiser._bands[0]
    band["W"], band["A"], band["eigenvalues"] = denoiser._solve_ged(
        np.diag([10000.0] + [1.0] * (N_CH - 1))
    )
    cleaned = data - denoiser._process_band(band, data, None, 0.7, 1, False, False)
    assert np.mean(cleaned[0] ** 2) < 0.01 * np.mean(data[0] ** 2)
    assert np.mean((cleaned[1:] - data[1:]) ** 2) < 0.1


def test_covariance_does_not_depend_on_chunk_partition(denoiser):
    import copy

    rng = np.random.default_rng(42)
    denoiser.fit(rng.normal(size=(N_CH, 2048)))
    samplewise = copy.deepcopy(denoiser)
    data = rng.normal(size=(N_CH, 501)) + np.linspace(0, 3, 501)
    denoiser.update_and_denoise(data, n_artifact=0, refit=False)
    for i in range(data.shape[1]):
        samplewise.update_and_denoise(data[:, i : i + 1], n_artifact=0, refit=False)
    for batch_band, sample_band in zip(denoiser._bands, samplewise._bands):
        np.testing.assert_allclose(batch_band["R"], sample_band["R"], atol=1e-12)


def test_refit_can_be_requested_after_covariance_update(denoiser):
    denoiser.fit(_make_baseline())
    denoiser.update_and_denoise(_make_baseline() * np.arange(1, N_CH + 1)[:, None], refit=False)
    before = [values.copy() for values in denoiser.eigenvalues]
    denoiser.update_and_denoise(_make_chunk(), update_covariance=False, refit=True)
    assert any(not np.allclose(a, b, atol=0) for a, b in zip(before, denoiser.eigenvalues))


def test_baseline_sensai_calibrates_each_band_once(denoiser, monkeypatch):
    from mne_rt.tools import adaptive_gedai

    calls = []

    def optimize(epochs, reference_cov, **kwargs):
        calls.append((epochs.copy(), reference_cov.copy(), kwargs))
        return 5.0, 0.0, []

    monkeypatch.setattr(adaptive_gedai, "_sensai_optimize", optimize)
    denoiser.fit(_make_baseline(), leadfield=np.eye(N_CH))
    fitted_bands = [denoiser._broadband] + [b for b in denoiser._bands if not b["ignore"]]
    assert len(calls) == len(fitted_bands)
    for band, (epochs, reference, options) in zip(fitted_bands, calls):
        assert epochs.shape[1] == N_CH
        assert epochs.shape[2] >= N_CH
        np.testing.assert_array_equal(reference, denoiser._reference_cov)
        assert options["bounds"][1] == 12.0
        assert options["bounds"][0] in (-6.0, -4.0, 0.0)
        assert options["noise_multiplier"] == 3.0
        assert options["engine"] == "numpy"
        assert band["threshold"] == 5.0
    thresholds = denoiser.eigenvalue_thresholds
    denoiser.update_and_denoise(_make_chunk())
    assert denoiser.eigenvalue_thresholds == thresholds
    assert len(calls) == len(fitted_bands)


@pytest.mark.parametrize("reject_all", [False, True])
def test_auto_uses_threshold_instead_of_fixed_count(denoiser, reject_all):
    denoiser.fit(_make_baseline())
    band = denoiser._bands[0]
    band["threshold"] = 1.0 if reject_all else np.inf
    band["sensitivity"] = 105.0
    data = _make_chunk()
    output = data - denoiser._process_band(band, data, None, 0.7, "auto", False, False)
    expected = np.zeros_like(data) if reject_all else data
    np.testing.assert_allclose(output, expected, atol=1e-16)


def test_sensai_removes_artifact_with_known_reference():
    from mne_rt.tools import AdaptiveMultibandGEDAIDenoiser

    rng = np.random.default_rng(44)
    gain = np.diag(np.sqrt([16, 8, 4, 1, 0.5, 0.25, 0.125, 0.0625]))
    clean = gain @ rng.normal(size=(N_CH, 16000))
    mixture = clean.copy()
    mixture[-1] += 20 * rng.normal(size=16000)
    denoiser = AdaptiveMultibandGEDAIDenoiser(N_CH, SFREQ, wavelet_level=3)
    denoiser.fit(mixture[:, :4096], leadfield=gain)
    output = np.concatenate(
        [denoiser.update_and_denoise(mixture[:, i : i + 32]) for i in range(4096, 16000, 32)],
        axis=1,
    )
    first, last = 5120, 16000 - denoiser.delay_samples
    offset = denoiser.delay_samples - 4096
    error = np.mean((output[:, first + offset : last + offset] - clean[:, first:last]) ** 2)
    raw_error = np.mean((mixture[:, first:last] - clean[:, first:last]) ** 2)
    assert error < 0.1 * raw_error


def test_sensai_calibration_preserves_pygedai_units(denoiser, monkeypatch):
    import copy

    from mne_rt.tools import adaptive_gedai

    calls = []

    def optimize(epochs, reference_cov, **kwargs):
        calls.append(epochs.copy())
        return 5.0, 0.0, []

    monkeypatch.setattr(adaptive_gedai, "_sensai_optimize", optimize)
    baseline = _make_baseline()
    denoiser.broadband_pass = False
    scaled = copy.deepcopy(denoiser)
    denoiser.fit(baseline, leadfield=np.eye(N_CH))
    scaled.fit(baseline * 1e6, leadfield=np.eye(N_CH))
    np.testing.assert_allclose(
        scaled.eigenvalue_thresholds,
        denoiser.eigenvalue_thresholds,
        rtol=1e-5,
    )
    # Each fit makes the same number of SENSAI calls, so pair the halves. Splitting
    # at n_bands only works while broadband_pass is False (one extra call each).
    half = len(calls) // 2
    for volts, microvolts in zip(calls[:half], calls[half:]):
        np.testing.assert_allclose(volts * 1e6, microvolts, atol=1e-12)


def test_band_filter_fixed_count_still_rejects_smallest(denoiser):
    denoiser.fit(_make_baseline())
    band = denoiser._bands[0]
    data = _make_chunk()
    noise = denoiser._process_band(band, data, None, 0.7, 1, False, False)
    expected = band["A"][:, -1:] @ (band["W"][:, -1:].T @ data)
    np.testing.assert_allclose(noise, expected)


def test_template_selection_overrides_auto(denoiser):
    denoiser.fit(_make_baseline())
    band = denoiser._bands[0]
    band["threshold"] = np.inf
    data = _make_chunk()
    noise = denoiser._process_band(band, data, band["A"][:, 0], 0.999999, "auto", False, False)
    expected = band["A"][:, :1] @ (band["W"][:, :1].T @ data)
    np.testing.assert_allclose(noise, expected)


def test_parallel_and_serial_outputs_match(denoiser):
    import copy

    denoiser.fit(_make_baseline())
    parallel = copy.deepcopy(denoiser)
    parallel.n_jobs = 2
    for _ in range(3):
        data = _make_chunk()
        np.testing.assert_allclose(
            denoiser.update_and_denoise(data),
            parallel.update_and_denoise(data),
            atol=1e-15,
        )


def test_zero_level_keeps_only_preprocessing_delay():
    from mne_rt.tools import AdaptiveMultibandGEDAIDenoiser

    model = AdaptiveMultibandGEDAIDenoiser(N_CH, SFREQ, wavelet_level=0)
    model.fit(_make_baseline())
    assert model.n_bands == 1
    data = _make_chunk()
    assert model.delay_samples == (len(model._prefilter_kernel) - 1) // 2
    np.testing.assert_array_equal(model._filter_kernels[0], np.ones(1))
    assert model.update_and_denoise(data, n_artifact=0).shape == data.shape


def test_actual_haar_kernels_share_one_delay(denoiser):
    denoiser.fit(_make_baseline())
    assert len({len(kernel) for kernel in denoiser._filter_kernels}) == 1
    for kernel in denoiser._filter_kernels:
        np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-15)
    np.testing.assert_allclose(
        sum(denoiser._filter_kernels),
        np.eye(1, len(denoiser._filter_kernels[0]), len(denoiser._filter_kernels[0]) // 2)[0],
        atol=1e-15,
    )


@pytest.mark.parametrize("sfreq, max_epoch", [(200, 15.36), (256, 12.0)])
def test_six_cycle_epochs_and_ignored_residual_match_pygedai(sfreq, max_epoch):
    from gedai.gedai.adaptive import _compute_wavelet_parameters
    from gedai.wavelet.transform import compute_wavelet_level

    from mne_rt.tools import AdaptiveMultibandGEDAIDenoiser

    rng = np.random.default_rng(61)
    data = rng.normal(size=(4, int(20 * sfreq)))
    model = AdaptiveMultibandGEDAIDenoiser(4, sfreq, broadband_pass=False)
    model.fit(data, np.diag([4.0, 3.0, 2.0, 1.0]))
    assert model.cycles_per_wavelet == 6
    expected_level = compute_wavelet_level(
        sfreq, lowcut_hz=0.5, n_times=data.shape[1], cycles_per_wavelet=6
    )
    assert model.wavelet_level == expected_level
    expected = _compute_wavelet_parameters(sfreq, expected_level, 6, 4, data.shape[1])
    for band, parameter, keep, kernel in zip(
        model._bands, expected, model.retained_bands, model._filter_kernels
    ):
        assert band["epoch_samples"] == parameter["n_samples"] + parameter["n_samples"] % 2
        assert keep == (parameter["fmax"] > 0.5)
        if not keep:
            np.testing.assert_array_equal(kernel, 0)
            assert np.isinf(band["threshold"])
    assert max(t for t, keep in zip(model.band_time_constants, model.retained_bands) if keep) == (
        pytest.approx(max_epoch)
    )


def test_dynamic_cutoff_uses_baseline_sensitivity(denoiser):
    from gedai.sensai.sensai import _sensai_to_eigen

    denoiser.fit(_make_baseline())
    band = denoiser._bands[0]
    sensitivity = band["sensitivity"]
    baseline_threshold = band["threshold"]
    denoiser._process_band(band, _make_chunk(), None, 0.7, "auto", True, True)
    expected = _sensai_to_eigen(sensitivity, np.abs(band["eigenvalues"]), percentile=98)
    assert band["effective_threshold"] == pytest.approx(expected)
    assert band["sensitivity"] == sensitivity
    assert band["threshold"] == baseline_threshold


@pytest.mark.parametrize("sfreq, radius", [(200, 127), (256, 127), (500, 255)])
@pytest.mark.parametrize("cycles", [2, 4, 6])
@pytest.mark.parametrize("mode", ["lowcut", "none"])
def test_shortened_haar_delay_and_epochs(sfreq, radius, cycles, mode):
    from mne_rt.tools import AdaptiveMultibandGEDAIDenoiser

    data = np.random.default_rng(71).normal(size=(4, 60 * sfreq))
    model = AdaptiveMultibandGEDAIDenoiser(
        4,
        sfreq,
        cycles_per_wavelet=cycles,
        wavelet_low_cutoff=1,
        prefilter_mode=mode,
    ).fit(data, np.eye(4))
    assert model.delay_samples == radius * (2 if mode == "lowcut" else 1)
    active = [b for b in model._bands if not b["ignore"]]
    assert active[-1]["tau"] == pytest.approx(cycles / active[-1]["fmin"])
    assert all(b["fmax"] > 1 for b in active)
    if mode == "none":
        np.testing.assert_array_equal(model._prefilter_kernel, [1])
    result = model.update_and_denoise(data[:, :25])
    assert np.isfinite(result).all()


def test_invalid_prefilter_mode():
    from mne_rt.tools import AdaptiveMultibandGEDAIDenoiser

    with pytest.raises(ValueError, match="prefilter_mode"):
        AdaptiveMultibandGEDAIDenoiser(N_CH, SFREQ, prefilter_mode="invalid")


def test_causal_window_processes_unique_samples_and_aligns_timestamps():
    import copy

    from mne_rt.tools.adaptive_gedai import (
        AdaptiveMultibandGEDAIDenoiser,
        _CausalGEDAIWindow,
    )

    rng = np.random.default_rng(711)
    baseline = rng.normal(size=(4, 1000))
    data = rng.normal(size=(4, 1000))
    model = AdaptiveMultibandGEDAIDenoiser(
        4,
        100,
        prefilter_mode="none",
        broadband_pass=False,
        wavelet_low_cutoff=1,
    ).fit(baseline, leadfield=np.eye(4))
    direct = copy.deepcopy(model)
    expected = np.concatenate(
        [direct.update_and_denoise(data[:, i : i + 5]) for i in range(0, 1000, 5)], axis=1
    )
    window = _CausalGEDAIWindow(model, 120)
    times = 100 + np.arange(1000) / 100
    emitted = 0
    for start in range(0, 841, 60):
        end = start + 120
        result = window.update(data[:, start:end], times[start:end])
        if end < 120 + model.delay_samples:
            assert result is None
            continue
        clean, raw, source_times = result
        np.testing.assert_allclose(clean, expected[:, end - 120 : end], atol=1e-12)
        source_end = end - model.delay_samples
        np.testing.assert_array_equal(raw, data[:, source_end - 120 : source_end])
        np.testing.assert_array_equal(source_times, times[source_end - 120 : source_end])
        assert window._pending_times.size == model.delay_samples
        assert window._times.size == 120
        assert window.update(data[:, start:end], times[start:end]) is None
        emitted += 1
    assert emitted > 5


def test_causal_window_rejects_gaps_and_invalid_timestamps():
    from mne_rt.tools.adaptive_gedai import (
        AdaptiveMultibandGEDAIDenoiser,
        _CausalGEDAIWindow,
    )

    model = AdaptiveMultibandGEDAIDenoiser(
        4,
        100,
        prefilter_mode="none",
        broadband_pass=False,
    ).fit(np.random.default_rng(12).normal(size=(4, 1000)), leadfield=np.eye(4))
    window = _CausalGEDAIWindow(model, 20)
    data = np.ones((4, 10))
    assert window.update(data, np.zeros(10)) is None
    with pytest.raises(ValueError, match="finite timestamp"):
        window.update(data, np.full(10, np.nan))
    with pytest.raises(ValueError, match="strictly increasing"):
        window.update(data, np.ones(10))
    window.update(data, 100 + np.arange(10) / 100)
    with pytest.raises(RuntimeError, match="missing input samples"):
        window.update(data, 101 + np.arange(10) / 100)


def test_ged_projection_matches_inverse_and_preserves_reference_metric():
    from scipy import linalg

    from mne_rt.tools import AdaptiveMultibandGEDAIDenoiser

    rng = np.random.default_rng(73)
    model = AdaptiveMultibandGEDAIDenoiser(4, 200)
    gain = rng.normal(size=(4, 6))
    model._reference_cov = gain @ gain.T + 0.1 * np.eye(4)
    model._reference_is_leadfield = True
    data = rng.normal(size=(4, 100))
    W, A, eigenvalues = model._solve_ged(np.cov(data))
    band = {"ignore": False, "W": W, "A": A, "eigenvalues": eigenvalues}
    np.testing.assert_allclose(A, linalg.inv(W.T), atol=1e-12)
    for count in range(5):
        noise = model._process_band(band, data, None, 0.7, count, False, False)
        expected = model._reference_cov @ W[:, :count] @ W[:, :count].T @ data
        np.testing.assert_allclose(noise, expected, atol=1e-12)
        cleaned = data - noise
        projection = np.eye(4) - A[:, :count] @ W[:, :count].T
        np.testing.assert_allclose(projection @ projection, projection, atol=1e-12)
        energy_before = np.sum(data * linalg.solve(model._reference_cov, data))
        energy_after = np.sum(cleaned * linalg.solve(model._reference_cov, cleaned))
        assert energy_after <= energy_before * (1 + 1e-12)
        if count == 0:
            np.testing.assert_array_equal(cleaned, data)
        elif count == 4:
            np.testing.assert_allclose(cleaned, 0, atol=1e-12)


def test_future_chunks_cannot_change_previous_outputs(denoiser):
    import copy

    denoiser.fit(_make_baseline())
    second = copy.deepcopy(denoiser)
    prefix = _make_chunk()
    out = denoiser.update_and_denoise(prefix)
    saved = out.copy()
    np.testing.assert_array_equal(out, second.update_and_denoise(prefix))
    denoiser.update_and_denoise(_make_chunk() * 100)
    second.update_and_denoise(_make_chunk() * -100)
    np.testing.assert_array_equal(out, saved)


@pytest.mark.parametrize(
    "bad_data",
    [
        np.ones(N_CH),
        np.empty((N_CH, 0)),
        np.full((N_CH, 10), np.nan),
        np.full((N_CH, 10), np.inf),
    ],
)
def test_invalid_data_rejected(denoiser, bad_data):
    with pytest.raises(ValueError):
        denoiser.fit(bad_data)
    denoiser.fit(_make_baseline())
    with pytest.raises(ValueError):
        denoiser.update_and_denoise(bad_data)


@pytest.mark.parametrize("value", [-1, 1.5, "invalid"])
def test_invalid_rejection_count_raises_before_updating(denoiser, value):
    denoiser.fit(_make_baseline())
    means = [b["mean"].copy() for b in denoiser._bands]
    with pytest.raises(ValueError, match="n_artifact"):
        denoiser.update_and_denoise(_make_chunk(), n_artifact=value)
    for before, band in zip(means, denoiser._bands):
        np.testing.assert_array_equal(before, band["mean"])
