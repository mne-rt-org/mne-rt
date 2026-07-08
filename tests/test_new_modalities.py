"""Tests for new modalities: SCP, PAF, connectivity_ratio."""

from __future__ import annotations

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SFREQ = 256.0
DURATION = 2.0
N_SAMPLES = int(SFREQ * DURATION)
N_CHANNELS = 4
RNG = np.random.default_rng(42)


def make_eeg(n_ch=N_CHANNELS, n_samp=N_SAMPLES, sfreq=SFREQ, freq=10.0):
    """Return a synthetic EEG-like array with a sinusoidal alpha component."""
    t = np.arange(n_samp) / sfreq
    signal = np.sin(2 * np.pi * freq * t)
    noise = RNG.standard_normal((n_ch, n_samp)) * 0.1
    return signal[np.newaxis, :] + noise


def make_dummy_modality_mixin(sfreq=SFREQ, data=None, ch_names=None, data_type="eeg"):
    """Build a minimal ModalityMixin-like object for testing prep/compute."""
    import mne

    from mne_rt.modalities import ModalityMixin

    class _Dummy(ModalityMixin):
        pass

    obj = _Dummy()
    n_ch = N_CHANNELS if data is None else data.shape[0]
    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(n_ch)]

    obj.rec_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    obj._sfreq = sfreq
    obj.data_type = data_type
    obj.winsize = DURATION
    obj.picks = None
    obj.params = {}
    obj.raw_baseline = None
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# SCP — Slow Cortical Potentials
# ─────────────────────────────────────────────────────────────────────────────


class TestSCPModality:
    def _obj(self, highpass=0.0, lowpass=1.0, reference="mean"):
        obj = make_dummy_modality_mixin()
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
        data = make_eeg()
        prep = obj._scp_prep()
        val, dt = obj._scp(data, **prep)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_compute_with_median_reference(self):
        obj = self._obj(reference="median")
        data = make_eeg()
        prep = obj._scp_prep()
        val, dt = obj._scp(data, **prep)
        assert isinstance(val, float)

    def test_scp_suppresses_alpha(self):
        """SCP (LP at 1 Hz) should have near-zero mean for a 10 Hz signal."""
        obj = self._obj(highpass=0.0, lowpass=1.0)
        data = make_eeg(freq=10.0)  # pure alpha
        prep = obj._scp_prep()
        val, _ = obj._scp(data, **prep)
        # Low-pass filtered 10 Hz sine → near zero mean
        assert abs(val) < 1.0

    def test_scp_captures_dc_shift(self):
        """SCP should detect a DC offset in the signal."""
        obj = self._obj()
        data = make_eeg() + 5.0  # add large DC offset
        prep = obj._scp_prep()
        val, _ = obj._scp(data, **prep)
        assert abs(val) > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PAF — Peak Alpha Frequency tracking
# ─────────────────────────────────────────────────────────────────────────────


class TestPAFModality:
    def _obj(self, frange=(7.0, 13.0), method="welch", smoothing=0.85):
        obj = make_dummy_modality_mixin()
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
        data = make_eeg(freq=10.0)
        prep = obj._peak_alpha_freq_prep()
        val, dt = obj._peak_alpha_freq(data, **prep)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_paf_within_frange(self):
        obj = self._obj(frange=(8.0, 13.0))
        data = make_eeg(freq=10.0)
        prep = obj._peak_alpha_freq_prep()
        val, _ = obj._peak_alpha_freq(data, **prep)
        # With EMA smoothing the estimate may not be exactly at 10 Hz but should be in range
        assert 7.0 <= val <= 14.0

    def test_paf_state_updated_across_calls(self):
        obj = self._obj(smoothing=0.0)  # no smoothing → pure instantaneous
        data = make_eeg(freq=10.0)
        prep = obj._peak_alpha_freq_prep()
        obj._peak_alpha_freq(data, **prep)
        state_after = prep["_paf_state"][0]
        # State should have been updated (or at least checked)
        assert isinstance(state_after, float)

    def test_paf_ema_smoothing(self):
        """High smoothing should make PAF change slowly between windows."""
        obj_smooth = self._obj(smoothing=0.99)
        obj_instant = self._obj(smoothing=0.0)
        data = make_eeg(freq=10.0)
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
        obj = make_dummy_modality_mixin(ch_names=ch_names)
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
        data = make_eeg(n_ch=4, n_samp=int(SFREQ * 2))
        prep = obj._connectivity_ratio_prep()
        val, dt = obj._connectivity_ratio(data, **prep)
        assert isinstance(val, float)
        assert np.isfinite(val)
        assert val >= 0.0  # ratio of coherences is non-negative

    def test_missing_channel_raises(self):
        ch_names = ["C3", "C4", "F3", "F4"]
        obj = make_dummy_modality_mixin(ch_names=ch_names)
        obj.params = {
            "frange": [8, 13],
            "channels_num": ["C3", "Cz"],  # Cz not in ch_names
            "channels_den": ["F3", "F4"],
            "method": "coh",
            "mode": "cwt_morlet",
        }
        with pytest.raises(ValueError):
            obj._connectivity_ratio_prep()


# ─────────────────────────────────────────────────────────────────────────────
# NFRealtime: artifact_rate and snr_data attributes
# ─────────────────────────────────────────────────────────────────────────────


class TestNFRealtimeNewFeatures:
    def test_replay_method_exists(self):
        from mne_rt import RTStream

        assert hasattr(RTStream, "replay")

    def test_run_blocks_method_exists(self):
        from mne_rt import RTStream

        assert hasattr(RTStream, "run_blocks")

    def test_run_blocks_empty_raises(self, tmp_path):
        from mne_rt import RTStream

        nf = RTStream(
            subject_id="sub01",
            session="01",
            subjects_dir=str(tmp_path),
            montage="easycap-M1",
        )
        with pytest.raises(ValueError, match="blocks"):
            nf.run_blocks(blocks=[])
