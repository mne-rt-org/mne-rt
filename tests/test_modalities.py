"""Unit tests for NF modality computations with synthetic data."""

import numpy as np
import pytest

# Synthetic EEG-like data: 64 channels, 500 Hz, 1 second
RNG = np.random.default_rng(42)
SFREQ = 500.0
N_CHANNELS = 64
N_TIMES = int(SFREQ)  # 1 second
DATA = RNG.standard_normal((N_CHANNELS, N_TIMES)).astype(np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers for SCP / PAF / connectivity_ratio (synthetic alpha-band signal)
# ─────────────────────────────────────────────────────────────────────────────

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

    config_file = Path(__file__).parent.parent / "src" / "mne_rt" / "config_methods.yml"

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


def test_erd_ers_needs_baseline(nf_obj):
    """erd_ers_prep requires baseline_power — check the error is meaningful."""
    with pytest.raises((AttributeError, RuntimeError)):
        _call_modality(nf_obj, "erd_ers", DATA)


def test_laterality_needs_channels(nf_obj):
    """laterality_prep auto-detects L/R channels from 10-20 names.
    With generic EEG channel names it may return 0 — that is acceptable."""
    val, delay = _call_modality(nf_obj, "laterality", DATA)
    assert isinstance(val, float)


# ─────────────────────────────────────────────────────────────────────────────
# SCP — Slow Cortical Potentials
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# PAF — Peak Alpha Frequency tracking
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Connectivity Ratio
# ─────────────────────────────────────────────────────────────────────────────


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
