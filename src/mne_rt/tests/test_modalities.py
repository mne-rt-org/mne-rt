"""Unit tests for NF modality computations with synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pytest

import mne_rt

# Synthetic EEG-like data: 64 channels, 500 Hz, 1 second
RNG = np.random.default_rng(42)
SFREQ = 500.0
N_CHANNELS = 64
N_TIMES = int(SFREQ)  # 1 second
DATA = RNG.standard_normal((N_CHANNELS, N_TIMES)).astype(np.float64)

# Helpers for SCP / PAF / connectivity_ratio (synthetic alpha-band signal)

_SFREQ_ALPHA = 256.0
_DURATION_ALPHA = 2.0
_N_SAMPLES_ALPHA = int(_SFREQ_ALPHA * _DURATION_ALPHA)
_N_CHANNELS_ALPHA = 4
_RNG_ALPHA = np.random.default_rng(42)


def _make_alpha_eeg(n_ch=_N_CHANNELS_ALPHA, n_samp=_N_SAMPLES_ALPHA, sfreq=_SFREQ_ALPHA, freq=10.0):
    """Return a synthetic EEG-like array with a sinusoidal alpha component."""
    t = np.arange(n_samp) / sfreq
    signal = np.sin(2 * np.pi * freq * t)
    noise = _RNG_ALPHA.standard_normal((n_ch, n_samp)) * 0.1
    return signal[np.newaxis, :] + noise


def _make_dummy_modality_mixin(sfreq=_SFREQ_ALPHA, data=None, ch_names=None, data_type="eeg"):
    """Build a minimal ModalityMixin-like object for testing prep/compute."""
    import mne

    from mne_rt.modalities import ModalityMixin

    class _Dummy(ModalityMixin):
        pass

    obj = _Dummy()
    n_ch = _N_CHANNELS_ALPHA if data is None else data.shape[0]
    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(n_ch)]

    obj.rec_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    obj._sfreq = sfreq
    obj.data_type = data_type
    obj.winsize = _DURATION_ALPHA
    obj.picks = None
    obj.params = {}
    obj.raw_baseline = None
    return obj


@pytest.fixture()
def nf_obj(tmp_path):
    """Minimal RTStream-like object with only the mixin methods wired up."""
    from pathlib import Path

    import mne

    from mne_rt.modalities import ModalityMixin

    config_file = Path(mne_rt.__file__).parent / "config_methods.yml"

    # Minimal mne.Info-like object with the channel names
    ch_names = [f"EEG{i:03d}" for i in range(N_CHANNELS)]
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")

    class _Fake(ModalityMixin):
        def __init__(self):
            self._sfreq = SFREQ
            self.data_type = "eeg"
            self.config_file = str(config_file)
            self._modality_params = {}
            self.rec_info = info
            self.winsize = N_TIMES / SFREQ  # 1 s, matching DATA

        @property
        def modality_params(self):
            return self._modality_params

        @modality_params.setter
        def modality_params(self, v):
            self._modality_params = v or {}

    return _Fake()


def _call_modality(nf_obj, mod_name, data):
    from mne_rt.tools import get_params

    nf_obj.params = get_params(nf_obj.config_file, mod_name, {})
    prep_fn = getattr(nf_obj, f"_{mod_name}_prep", None)
    precomp = prep_fn() if callable(prep_fn) else {}
    fn = getattr(nf_obj, f"_{mod_name}")
    result, delay = fn(data, **precomp)
    return result, delay


def test_sensor_power(nf_obj):
    val, delay = _call_modality(nf_obj, "sensor_power", DATA)
    assert isinstance(val, float)
    assert val >= 0
    assert delay >= 0


def test_band_ratio(nf_obj):
    val, delay = _call_modality(nf_obj, "band_ratio", DATA)
    assert isinstance(val, float)
    assert delay >= 0


def test_entropy(nf_obj):
    val, delay = _call_modality(nf_obj, "entropy", DATA)
    assert isinstance(val, float)
    assert delay >= 0


def test_hjorth(nf_obj):
    val, delay = _call_modality(nf_obj, "hjorth", DATA)
    assert isinstance(val, float)
    assert delay >= 0


def test_spectral_centroid(nf_obj):
    val, delay = _call_modality(nf_obj, "spectral_centroid", DATA)
    assert isinstance(val, float)
    assert val > 0  # centroid should be positive frequency


def test_instantaneous_phase(nf_obj):
    """Regression: shipped in ``config_methods.yml`` and documented as a
    modality, but ``ModalityMixin`` had no ``_instantaneous_phase``, so
    ``record_main(modality=["instantaneous_phase"])`` raised
    ``NotImplementedError``. This goes through the real config -> prep ->
    compute path, which is where the gap was."""
    val, delay = _call_modality(nf_obj, "instantaneous_phase", DATA)
    assert isinstance(val, float)
    assert -np.pi <= val <= np.pi
    assert delay >= 0


def test_instantaneous_phase_tracks_a_known_oscillation(nf_obj):
    """Phase of a pure sine is reproducible and shifts with the sine's own."""
    from mne_rt.tools import get_params

    nf_obj.params = get_params(nf_obj.config_file, "instantaneous_phase", {})
    precomp = nf_obj._instantaneous_phase_prep()
    t = np.arange(N_TIMES) / SFREQ
    base = np.tile(np.sin(2 * np.pi * 10.0 * t), (N_CHANNELS, 1))
    shifted = np.tile(np.sin(2 * np.pi * 10.0 * t + np.pi / 2), (N_CHANNELS, 1))

    val_a, _ = nf_obj._instantaneous_phase(base, **precomp)
    val_b, _ = nf_obj._instantaneous_phase(base.copy(), **precomp)
    val_c, _ = nf_obj._instantaneous_phase(shifted, **precomp)

    assert val_a == pytest.approx(val_b)  # deterministic
    assert val_a != pytest.approx(val_c)  # sensitive to the input phase


def test_erd_ers_needs_baseline(nf_obj):
    """erd_ers_prep requires baseline_power — check the error is meaningful."""
    with pytest.raises((AttributeError, RuntimeError)):
        _call_modality(nf_obj, "erd_ers", DATA)


def test_laterality_needs_channels(nf_obj):
    """laterality_prep auto-detects L/R channels from 10-20 names.
    With generic EEG channel names it may return 0 — that is acceptable."""
    val, delay = _call_modality(nf_obj, "laterality", DATA)
    assert isinstance(val, float)


# decode — RTDecode-backed "decode" modality


def _fit_decoder_for(nf_obj, n_channels=N_CHANNELS, n_times=N_TIMES, n_components=2):
    from mne_rt import RTDecode

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, n_channels, n_times))
    y = np.array([0, 1] * 10)
    return RTDecode(info=nf_obj.rec_info, n_components=n_components).fit(X, y, verbose=False)


class TestDecodeModality:
    def test_decode_without_decoder_raises(self, nf_obj):
        with pytest.raises(RuntimeError, match="set_decoder"):
            _call_modality(nf_obj, "decode", DATA)

    def test_decode_with_unfitted_decoder_raises(self, nf_obj):
        from mne_rt import RTDecode

        nf_obj.decoder = RTDecode(info=nf_obj.rec_info, n_components=2)
        with pytest.raises(RuntimeError, match="fit"):
            _call_modality(nf_obj, "decode", DATA)

    def test_decode_proba_output(self, nf_obj):
        nf_obj.decoder = _fit_decoder_for(nf_obj)

        val, delay = _call_modality(nf_obj, "decode", DATA)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0
        assert delay >= 0

    def test_decode_class_index_out_of_range_raises(self, nf_obj):
        nf_obj.decoder = _fit_decoder_for(nf_obj)  # binary decoder: classes_ = [0, 1]

        nf_obj.params = {"class_index": 5}
        with pytest.raises(ValueError, match="class_index"):
            nf_obj._decode_prep()

    def test_decode_channel_count_mismatch_raises(self, nf_obj):
        # Decoder fit on half as many channels as the session provides.
        nf_obj.decoder = _fit_decoder_for(nf_obj, n_channels=N_CHANNELS // 2)

        nf_obj.params = {"class_index": 1}
        with pytest.raises(ValueError, match="channels"):
            nf_obj._decode_prep()

    def test_decode_without_predict_proba_raises(self, nf_obj):
        from sklearn.svm import LinearSVC

        from mne_rt import RTDecode

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, N_CHANNELS, N_TIMES))
        y = np.array([0, 1] * 10)
        nf_obj.decoder = RTDecode(info=nf_obj.rec_info, estimator=LinearSVC(), n_components=2).fit(
            X, y, verbose=False
        )

        nf_obj.params = {"class_index": 1}
        with pytest.raises(AttributeError, match="predict_proba"):
            nf_obj._decode_prep()


# SCP — Slow Cortical Potentials


class TestSCPModality:
    def _obj(self, highpass=0.0, lowpass=1.0, reference="mean"):
        obj = _make_dummy_modality_mixin()
        obj.params = {"highpass": highpass, "lowpass": lowpass, "reference": reference}
        return obj

    def test_prep_returns_dict(self):
        obj = self._obj()
        prep = obj._scp_prep()
        assert "sos_lp" in prep
        assert "sos_hp" in prep
        assert "reference" in prep

    def test_prep_no_highpass(self):
        obj = self._obj(highpass=0.0)
        prep = obj._scp_prep()
        assert prep["sos_hp"] is None

    def test_prep_with_highpass(self):
        obj = self._obj(highpass=0.1, lowpass=1.0)
        prep = obj._scp_prep()
        assert prep["sos_hp"] is not None

    def test_compute_returns_float(self):
        obj = self._obj()
        data = _make_alpha_eeg()
        prep = obj._scp_prep()
        val, dt = obj._scp(data, **prep)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_compute_with_median_reference(self):
        obj = self._obj(reference="median")
        data = _make_alpha_eeg()
        prep = obj._scp_prep()
        val, dt = obj._scp(data, **prep)
        assert isinstance(val, float)

    def test_scp_suppresses_alpha(self):
        """SCP (LP at 1 Hz) should have near-zero mean for a 10 Hz signal."""
        obj = self._obj(highpass=0.0, lowpass=1.0)
        data = _make_alpha_eeg(freq=10.0)  # pure alpha
        prep = obj._scp_prep()
        val, _ = obj._scp(data, **prep)
        # Low-pass filtered 10 Hz sine → near zero mean
        assert abs(val) < 1.0

    def test_scp_captures_dc_shift(self):
        """SCP should detect a DC offset in the signal."""
        obj = self._obj()
        data = _make_alpha_eeg() + 5.0  # add large DC offset
        prep = obj._scp_prep()
        val, _ = obj._scp(data, **prep)
        assert abs(val) > 1.0


# PAF — Peak Alpha Frequency tracking


class TestPAFModality:
    def _obj(self, frange=(7.0, 13.0), method="welch", smoothing=0.85):
        obj = _make_dummy_modality_mixin()
        obj.params = {"frange": frange, "method": method, "smoothing": smoothing}
        return obj

    def test_prep_returns_dict(self):
        obj = self._obj()
        prep = obj._peak_alpha_freq_prep()
        assert "sfreq" in prep
        assert "frange" in prep
        assert "_paf_state" in prep
        assert isinstance(prep["_paf_state"], list)
        assert len(prep["_paf_state"]) == 1

    def test_initial_paf_in_frange(self):
        obj = self._obj(frange=(8.0, 12.0))
        prep = obj._peak_alpha_freq_prep()
        paf0 = prep["_paf_state"][0]
        assert 8.0 <= paf0 <= 12.0

    def test_compute_returns_float(self):
        obj = self._obj()
        data = _make_alpha_eeg(freq=10.0)
        prep = obj._peak_alpha_freq_prep()
        val, dt = obj._peak_alpha_freq(data, **prep)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_paf_within_frange(self):
        obj = self._obj(frange=(8.0, 13.0))
        data = _make_alpha_eeg(freq=10.0)
        prep = obj._peak_alpha_freq_prep()
        val, _ = obj._peak_alpha_freq(data, **prep)
        # With EMA smoothing the estimate may not be exactly at 10 Hz but should be in range
        assert 7.0 <= val <= 14.0

    def test_paf_state_updated_across_calls(self):
        obj = self._obj(smoothing=0.0)  # no smoothing → pure instantaneous
        data = _make_alpha_eeg(freq=10.0)
        prep = obj._peak_alpha_freq_prep()
        obj._peak_alpha_freq(data, **prep)
        state_after = prep["_paf_state"][0]
        # State should have been updated (or at least checked)
        assert isinstance(state_after, float)

    def test_paf_ema_smoothing(self):
        """High smoothing should make PAF change slowly between windows."""
        obj_smooth = self._obj(smoothing=0.99)
        obj_instant = self._obj(smoothing=0.0)
        data = _make_alpha_eeg(freq=10.0)
        prep_s = obj_smooth._peak_alpha_freq_prep()
        prep_i = obj_instant._peak_alpha_freq_prep()
        # Set a very different initial state
        prep_s["_paf_state"][0] = 8.0
        prep_i["_paf_state"][0] = 8.0
        # Apply two windows
        for _ in range(2):
            val_s, _ = obj_smooth._peak_alpha_freq(data, **prep_s)
            val_i, _ = obj_instant._peak_alpha_freq(data, **prep_i)
        # Smoothed should change less from initial
        assert abs(val_s - 8.0) < abs(val_i - 8.0)


# Connectivity Ratio


class TestConnectivityRatioModality:
    def _obj(self):
        ch_names = ["C3", "C4", "F3", "F4"]
        obj = _make_dummy_modality_mixin(ch_names=ch_names)
        obj.params = {
            "frange": [8, 13],
            "channels_num": ["C3", "C4"],
            "channels_den": ["F3", "F4"],
            "method": "coh",
            "mode": "cwt_morlet",
        }
        return obj

    def test_prep_returns_dict(self):
        obj = self._obj()
        prep = obj._connectivity_ratio_prep()
        assert "indices_num" in prep
        assert "indices_den" in prep
        assert "freqs" in prep
        assert "fmin" in prep
        assert "fmax" in prep

    def test_compute_returns_float(self):
        obj = self._obj()
        data = _make_alpha_eeg(n_ch=4, n_samp=int(_SFREQ_ALPHA * 2))
        prep = obj._connectivity_ratio_prep()
        val, dt = obj._connectivity_ratio(data, **prep)
        assert isinstance(val, float)
        assert np.isfinite(val)
        assert val >= 0.0  # ratio of coherences is non-negative

    def test_missing_channel_raises(self):
        ch_names = ["C3", "C4", "F3", "F4"]
        obj = _make_dummy_modality_mixin(ch_names=ch_names)
        obj.params = {
            "frange": [8, 13],
            "channels_num": ["C3", "Cz"],  # Cz not in ch_names
            "channels_den": ["F3", "F4"],
            "method": "coh",
            "mode": "cwt_morlet",
        }
        with pytest.raises(ValueError):
            obj._connectivity_ratio_prep()


# Connectivity: imcoh and n_cycles


def _connectivity_channels(nf_obj):
    """First two real channel names, as the config's [ [a,b], [c,d] ] pair form."""
    names = nf_obj.rec_info["ch_names"]
    return [[names[0], names[1]], [names[2], names[3]]]


def test_sensor_connectivity_coh(nf_obj):
    from mne_rt.tools import get_params

    nf_obj.params = get_params(nf_obj.config_file, "sensor_connectivity", {})
    nf_obj.params["channels"] = _connectivity_channels(nf_obj)
    prep = nf_obj._sensor_connectivity_prep()
    val, delay = nf_obj._sensor_connectivity(DATA, **prep)
    assert isinstance(val, float)
    assert np.isfinite(val)
    assert delay >= 0


def test_sensor_connectivity_imcoh_runs(nf_obj):
    """imcoh used to raise KeyError from spectral_connectivity_time."""
    from mne_rt.tools import get_params

    nf_obj.params = get_params(nf_obj.config_file, "sensor_connectivity", {})
    nf_obj.params["channels"] = _connectivity_channels(nf_obj)
    nf_obj.params["method"] = "imcoh"
    prep = nf_obj._sensor_connectivity_prep()
    assert prep["method"] == "cohy"  # remapped
    assert prep["take_imag"] is True
    val, _ = nf_obj._sensor_connectivity(DATA, **prep)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_sensor_connectivity_imcoh_is_real_valued(nf_obj):
    """The imaginary part must be extracted, not left complex."""
    from mne_rt.tools import get_params

    nf_obj.params = get_params(nf_obj.config_file, "sensor_connectivity", {})
    nf_obj.params["channels"] = _connectivity_channels(nf_obj)
    nf_obj.params["method"] = "imcoh"
    prep = nf_obj._sensor_connectivity_prep()
    val, _ = nf_obj._sensor_connectivity(DATA, **prep)
    assert not isinstance(val, complex)
    assert abs(val) <= 1.0 + 1e-9


def test_sensor_connectivity_imcoh_detects_phase_lag(nf_obj):
    """Guards against silently returning 0.

    ``get_data(output="dense")`` allocates a real array and discards the
    imaginary part, so a naive implementation yields exactly 0.0 for every
    window.  Two channels with a genuine non-zero phase lag must produce a
    non-zero imaginary coherence.
    """
    from mne_rt.tools import get_params

    n = N_TIMES
    t = np.arange(n) / SFREQ
    rng = np.random.default_rng(11)
    lagged = np.zeros((N_CHANNELS, n))
    lagged[0] = np.sin(2 * np.pi * 10 * t) + 0.3 * rng.standard_normal(n)
    lagged[1] = np.sin(2 * np.pi * 10 * t + 1.2) + 0.3 * rng.standard_normal(n)
    lagged[2:] = 0.3 * rng.standard_normal((N_CHANNELS - 2, n))

    names = nf_obj.rec_info["ch_names"]
    nf_obj.params = get_params(nf_obj.config_file, "sensor_connectivity", {})
    nf_obj.params["channels"] = [[names[0], names[1]]]
    nf_obj.params["method"] = "imcoh"
    prep = nf_obj._sensor_connectivity_prep()
    val, _ = nf_obj._sensor_connectivity(lagged, **prep)
    assert abs(val) > 0.05, f"imcoh collapsed to {val!r} — imaginary part lost?"


def test_sensor_connectivity_imcoh_near_zero_for_zero_lag(nf_obj):
    """The complement: zero-lag (volume-conduction-like) coupling gives ~0."""
    from mne_rt.tools import get_params

    n = N_TIMES
    t = np.arange(n) / SFREQ
    rng = np.random.default_rng(12)
    common = np.sin(2 * np.pi * 10 * t)
    synced = np.zeros((N_CHANNELS, n))
    synced[0] = common + 0.05 * rng.standard_normal(n)
    synced[1] = common + 0.05 * rng.standard_normal(n)
    synced[2:] = 0.3 * rng.standard_normal((N_CHANNELS - 2, n))

    names = nf_obj.rec_info["ch_names"]
    nf_obj.params = get_params(nf_obj.config_file, "sensor_connectivity", {})
    nf_obj.params["channels"] = [[names[0], names[1]]]
    nf_obj.params["method"] = "imcoh"
    prep = nf_obj._sensor_connectivity_prep()
    val, _ = nf_obj._sensor_connectivity(synced, **prep)
    assert abs(val) < 0.2


def test_sensor_connectivity_theta_rejected_at_default_n_cycles(nf_obj):
    """5 cycles at 4 Hz needs 1.25 s; the 1 s window must be refused up front."""
    from mne_rt.tools import get_params

    nf_obj.params = get_params(nf_obj.config_file, "sensor_connectivity", {})
    nf_obj.params["channels"] = _connectivity_channels(nf_obj)
    nf_obj.params["frange"] = [4, 8]
    with pytest.raises(ValueError, match="winsize"):
        nf_obj._sensor_connectivity_prep()


def test_sensor_connectivity_theta_works_with_auto_n_cycles(nf_obj):
    from mne_rt.tools import get_params

    nf_obj.params = get_params(nf_obj.config_file, "sensor_connectivity", {})
    nf_obj.params["channels"] = _connectivity_channels(nf_obj)
    nf_obj.params["frange"] = [4, 8]
    nf_obj.params["n_cycles"] = "auto"
    prep = nf_obj._sensor_connectivity_prep()
    val, _ = nf_obj._sensor_connectivity(DATA, **prep)
    assert np.isfinite(val)


def test_connectivity_ratio_imcoh_runs():
    """connectivity_ratio shares the same imcoh remapping."""
    from mne_rt.tools import get_params

    data = _make_alpha_eeg()
    obj = _make_dummy_modality_mixin(data=data, ch_names=["C3", "C4", "F3", "F4"])
    config_file = str(Path(mne_rt.__file__).parent / "config_methods.yml")
    obj.params = get_params(config_file, "connectivity_ratio", {})
    obj.params["method"] = "imcoh"
    prep = obj._connectivity_ratio_prep()
    assert prep["method"] == "cohy"
    assert prep["take_imag"] is True
    val, _ = obj._connectivity_ratio(data, **prep)
    assert isinstance(val, float)
    assert np.isfinite(val)


# Source modalities: inverse-operator lookup


def test_get_inverse_operator_prefers_in_memory(nf_obj):
    sentinel = object()
    nf_obj.inv = sentinel
    assert nf_obj._get_inverse_operator() is sentinel


def test_get_inverse_operator_missing_raises_actionable_error(nf_obj, tmp_path):
    """Previously this raised AttributeError on the never-assigned `self.visit`."""
    nf_obj.inv = None
    nf_obj.subject_id = "01"
    nf_obj.session = "01"
    nf_obj.subject_dir = tmp_path
    (tmp_path / "inv").mkdir()
    with pytest.raises(RuntimeError, match="record_baseline"):
        nf_obj._get_inverse_operator()


def test_get_inverse_operator_reads_bids_stem(nf_obj, tmp_path, monkeypatch):
    """The read path must match the stem compute_inv_operator() writes."""
    import mne_rt.modalities as modalities

    nf_obj.inv = None
    nf_obj.subject_id = "P01"
    nf_obj.session = "02"
    nf_obj.subject_dir = tmp_path
    (tmp_path / "inv").mkdir()
    expected = tmp_path / "inv" / "sub-P01_ses-02_task-baseline_inv.fif"
    expected.touch()

    seen = {}

    def _fake_read(fname):
        seen["fname"] = fname
        return "inverse-operator"

    monkeypatch.setattr(modalities, "read_inverse_operator", _fake_read)
    assert nf_obj._get_inverse_operator() == "inverse-operator"
    assert str(seen["fname"]) == str(expected)


# ROI-kernel caching


class _FakeROI:
    def __init__(self, name):
        self.name = name


class _CountingModel:
    """Stands in for SourceModel, counting how often the kernel is built."""

    def __init__(self):
        self.n_builds = 0

    def roi_kernel(self, rois, mri_resolution):
        self.n_builds += 1
        # Includes members, so two same-named ROIs over different labels give
        # different kernels -- as they would in reality.
        return ("kernel", tuple((r.name, getattr(r, "members", ())) for r in rois), mri_resolution)


def test_roi_kernel_is_built_once_per_distinct_request():
    """Several bands of one measure ask for an identical kernel.

    Building it is the expensive part of source preparation, so it must be
    shared rather than repeated per instance.
    """
    from mne_rt.modalities import ModalityMixin

    obj = ModalityMixin()
    model = _CountingModel()
    rois = [_FakeROI("broca"), _FakeROI("wernicke")]

    first = obj._get_roi_kernel(model, rois, True)
    second = obj._get_roi_kernel(model, list(rois), True)
    assert second is first
    assert model.n_builds == 1

    # A different ROI set, or a different resolution, is a different kernel.
    obj._get_roi_kernel(model, [_FakeROI("broca")], True)
    obj._get_roi_kernel(model, rois, False)
    assert model.n_builds == 3


def test_roi_kernel_cache_is_per_model():
    from mne_rt.modalities import ModalityMixin

    obj = ModalityMixin()
    rois = [_FakeROI("broca")]
    model_a, model_b = _CountingModel(), _CountingModel()
    obj._get_roi_kernel(model_a, rois, True)
    obj._get_roi_kernel(model_b, rois, True)
    assert model_a.n_builds == model_b.n_builds == 1


def test_roi_kernel_cache_distinguishes_same_named_rois():
    """Two ROIs may share a name but cover different labels.

    Keying on the name alone would hand the second instance the first one's
    kernel, silently measuring the wrong region.
    """
    from mne_rt.modalities import ModalityMixin
    from mne_rt.source import ROI

    obj = ModalityMixin()
    model = _CountingModel()
    left = [ROI(name="target", members=("Left-Hippocampus",), kind="volume")]
    right = [ROI(name="target", members=("Right-Hippocampus",), kind="volume")]

    k_left = obj._get_roi_kernel(model, left, True)
    k_right = obj._get_roi_kernel(model, right, True)
    assert model.n_builds == 2
    assert k_left != k_right


def test_roi_kernel_cache_holds_its_model_alive():
    """The cache must not key on an address a later model could reuse."""
    from mne_rt.modalities import ModalityMixin

    obj = ModalityMixin()
    rois = [_FakeROI("broca")]
    model = _CountingModel()
    obj._get_roi_kernel(model, rois, True)
    # A cache keyed by id() could match a new model that reused the address.
    assert any(model is key[0] for key in obj._roi_kernels)


def test_cfc_says_what_to_install_when_pactools_cannot_import(monkeypatch):
    import sys

    from mne_rt.modalities import ModalityMixin

    monkeypatch.setitem(sys.modules, "pactools", None)
    with pytest.raises(ImportError, match=r"scipy<2"):
        ModalityMixin._cfc_sensor_prep(object.__new__(ModalityMixin))


# Per-pair values for the connectivity viewer


def test_edge_values_are_the_float_they_were():
    """Everything downstream (z-score, EMA, protocols, saving) sees a float."""
    from mne_rt.modalities import EdgeValues

    v = EdgeValues(0.5, [0.25, 0.75], [("C3", "C4"), ("F3", "F4")], "coh", space="sensor")
    assert v == 0.5 and isinstance(v, float)
    assert type(v * 2) is float and type(float(v)) is float
    np.testing.assert_array_equal(v.edges, [0.25, 0.75])
    assert v.labels == [("C3", "C4"), ("F3", "F4")]


def test_sensor_connectivity_reports_its_pairs(nf_obj):
    from mne_rt.tools import get_params

    nf_obj.params = get_params(nf_obj.config_file, "sensor_connectivity", {})
    nf_obj.params["channels"] = _connectivity_channels(nf_obj)
    prep = nf_obj._sensor_connectivity_prep()
    val, _ = nf_obj._sensor_connectivity(DATA, **prep)
    assert val.labels == [tuple(p) for p in nf_obj.params["channels"]]
    assert val.edges.shape == (2,)
    assert float(val) == pytest.approx(val.edges.mean())
    assert val.space == "sensor" and val.metric == nf_obj.params["method"]


def test_sensor_graph_uses_the_configured_pairs_and_their_weights(nf_obj, monkeypatch):
    """It paired the first pair's channels with the second's (C3-F3, C4-F4 for
    [[C3, C4], [F3, F4]]), and graph_matrix[array([i, j])] took two whole rows:
    the value averaged 2 x n_channels weights rather than the configured edges."""
    from mne_rt import modalities
    from mne_rt.tools import get_params

    names = nf_obj.rec_info["ch_names"]
    n = len(names)
    graph = np.arange(n * n, dtype=float).reshape(n, n)
    monkeypatch.setattr(modalities, "learn_graph", lambda *a, **k: graph)
    nf_obj.params = get_params(nf_obj.config_file, "sensor_graph", {})
    nf_obj.params["channels"] = [[names[0], names[1]], [names[2], names[3]]]
    prep = nf_obj._sensor_graph_prep()
    assert prep["labels"] == [(names[0], names[1]), (names[2], names[3])]
    val, _ = nf_obj._sensor_graph(DATA, **prep)
    np.testing.assert_array_equal(val.edges, [graph[0, 1], graph[2, 3]])
    assert float(val) == pytest.approx((graph[0, 1] + graph[2, 3]) / 2)


def test_connectivity_ratio_reports_both_pairs(nf_obj):
    from mne_rt.tools import get_params

    names = nf_obj.rec_info["ch_names"]
    nf_obj.params = get_params(nf_obj.config_file, "connectivity_ratio", {})
    nf_obj.params["channels_num"] = [names[0], names[1]]
    nf_obj.params["channels_den"] = [names[2], names[3]]
    prep = nf_obj._connectivity_ratio_prep()
    val, _ = nf_obj._connectivity_ratio(DATA, **prep)
    num, den = val.edges
    assert float(val) == pytest.approx(num / (den + 1e-30))
    assert val.labels == [(names[0], names[1]), (names[2], names[3])]


def test_edge_values_survive_copy_and_pickle():
    import copy
    import pickle

    from mne_rt.modalities import EdgeValues

    v = EdgeValues(0.5, [0.25, 0.75], [("C3", "C4"), ("F3", "F4")], "coh", space="sensor")
    for w in (copy.copy(v), copy.deepcopy(v), pickle.loads(pickle.dumps(v))):
        assert w == 0.5 and w.labels == v.labels
        np.testing.assert_array_equal(w.edges, v.edges)


def test_sensor_graph_explains_the_pair_format(nf_obj):
    from mne_rt.tools import get_params

    names = nf_obj.rec_info["ch_names"]
    nf_obj.params = get_params(nf_obj.config_file, "sensor_graph", {})
    nf_obj.params["channels"] = [names[:3], names[3:6]]  # the old two-lists form
    with pytest.raises(ValueError, match=r"\[ch_A, ch_B\] pairs"):
        nf_obj._sensor_graph_prep()


def test_graph_learning_solves_graphs_of_more_than_two_nodes():
    """The solver path raised on any graph of 3+ nodes: sensor_graph could not
    run on a real recording. Two signals sharing a rhythm get the strongest edge."""
    from mne_rt.tools import log_degree_barrier

    rng = np.random.default_rng(0)
    t = np.arange(250) / 250.0
    signals = rng.standard_normal((6, 250))
    shared = 3 * np.sin(2 * np.pi * 10 * t)
    signals[1] += shared
    signals[4] += shared
    g = log_degree_barrier(signals, dist_type="sqeuclidean", alpha=1, beta=1)
    assert g.shape == (6, 6) and np.allclose(g, g.T)
    assert g.min() >= 0 and g.max() == pytest.approx(1.0)
    assert g[1, 4] == pytest.approx(g.max())


@pytest.fixture()
def fake_mne_connectivity(monkeypatch):
    """Install a stand-in mne_connectivity; the learner lookup is cached, so reset it."""
    import types

    from mne_rt.tools import tools

    def install(**attrs):
        monkeypatch.setitem(sys.modules, "mne_connectivity", types.SimpleNamespace(**attrs))
        tools._mne_learn_graph.cache_clear()

    yield install
    tools._mne_learn_graph.cache_clear()


def test_learn_graph_uses_mne_connectivity_s_learner(fake_mne_connectivity):
    """MNE-Connectivity's learn_graph when it is there, scaled to its strongest edge."""
    from mne_rt.tools import learn_graph

    calls = []

    class _Con:
        def __init__(self, dense):
            self._dense = dense

        def get_data(self, output):
            assert output == "dense"
            return self._dense

    def fake(signals, **kwargs):
        calls.append(kwargs)
        n = signals.shape[0]
        dense = np.full((n, n), 2.0)
        np.fill_diagonal(dense, 0.0)
        dense[0, 1] = dense[1, 0] = 4.0
        return _Con(dense)

    fake_mne_connectivity(learn_graph=fake)
    g = learn_graph(np.zeros((3, 10)), dist_type="sqeuclidean", alpha=1.0, beta=0.5)
    assert calls == [
        {"alpha": 1.0, "beta": 0.5, "metric": "sqeuclidean", "max_iter": 10000, "verbose": False}
    ]
    assert g.max() == 1.0 and g[0, 1] == 1.0 and g[0, 2] == 0.5


def test_learn_graph_falls_back_to_mne_rt_s_own(fake_mne_connectivity):
    from mne_rt.tools import tools

    fake_mne_connectivity()  # an mne-connectivity without learn_graph
    signals = np.random.default_rng(0).standard_normal((4, 100))
    np.testing.assert_allclose(
        tools.learn_graph(signals),
        tools.log_degree_barrier(signals, dist_type="sqeuclidean", alpha=1.0, beta=1.0),
    )


def test_a_window_of_bad_data_gives_no_graph():
    """Not a graph of zeros, which would read as a measured value."""
    from mne_rt.tools import learn_graph

    signals = np.random.default_rng(0).standard_normal((4, 100))
    signals[2, 50] = np.nan
    assert np.all(np.isnan(learn_graph(signals)))


def test_mne_rt_s_learner_matches_mne_connectivity_s():
    """Both scale distances by their mean: with the same alpha and beta they
    give the same graph (by the max, ours was about three times as dense)."""
    mne_connectivity = pytest.importorskip("mne_connectivity")
    if not hasattr(mne_connectivity, "learn_graph"):
        pytest.skip("mne-connectivity < 0.10 has no learn_graph")
    from scipy.signal import sosfiltfilt

    from mne_rt.tools import butter_bandpass, log_degree_barrier

    sos = butter_bandpass(8, 13, 250.0, order=5)
    x = sosfiltfilt(sos, np.random.default_rng(0).standard_normal((20, 250)))
    ours = log_degree_barrier(x, dist_type="sqeuclidean", alpha=1.0, beta=1.0)
    con = mne_connectivity.learn_graph(x, alpha=1.0, beta=1.0, max_iter=10000, verbose=False)
    theirs = np.asarray(con.get_data(output="dense")).reshape(20, 20)
    theirs = theirs / theirs.max()
    assert np.corrcoef(ours.ravel(), theirs.ravel())[0, 1] > 0.99
    assert np.abs(ours - theirs).max() < 0.05


def test_graph_modalities_check_alpha_and_beta_before_the_session(nf_obj):
    from mne_rt.tools import get_params

    names = nf_obj.rec_info["ch_names"]
    nf_obj.params = get_params(nf_obj.config_file, "sensor_graph", {})
    nf_obj.params["channels"] = [[names[0], names[1]]]
    nf_obj.params["alpha"] = 0
    with pytest.raises(ValueError, match="alpha must be > 0"):
        nf_obj._sensor_graph_prep()


def test_graph_modalities_need_three_nodes(nf_obj, monkeypatch):
    """Over two nodes a learned graph is one edge whose weight does not depend
    on the data (source_graph's default config had two ROIs: 1.0 every window)."""
    import types

    from mne_rt.tools import get_params

    nf_obj.params = get_params(nf_obj.config_file, "source_graph", {})
    rois = [types.SimpleNamespace(name=n) for n in ("a-lh", "a-rh")]
    monkeypatch.setattr(type(nf_obj), "_source_prep_common", lambda self, **k: {"rois": rois})
    nf_obj.params["pair"] = ["a-lh", "a-rh"]
    with pytest.raises(ValueError, match="at least three"):
        nf_obj._source_graph_prep()

    names = nf_obj.rec_info["ch_names"]
    monkeypatch.setattr(type(nf_obj), "_acquired_ch_names", lambda self, **k: names[:2])
    nf_obj.params = get_params(nf_obj.config_file, "sensor_graph", {})
    nf_obj.params["channels"] = [[names[0], names[1]]]
    with pytest.raises(ValueError, match="at least three"):
        nf_obj._sensor_graph_prep()
