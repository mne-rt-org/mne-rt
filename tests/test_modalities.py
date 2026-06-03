"""Unit tests for NF modality computations with synthetic data."""

import numpy as np
import pytest

# Synthetic EEG-like data: 64 channels, 500 Hz, 1 second
RNG = np.random.default_rng(42)
SFREQ = 500.0
N_CHANNELS = 64
N_TIMES = int(SFREQ)  # 1 second
DATA = RNG.standard_normal((N_CHANNELS, N_TIMES)).astype(np.float64)


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
    assert val > 0   # centroid should be positive frequency


def test_erd_ers_needs_baseline(nf_obj):
    """erd_ers_prep requires baseline_power — check the error is meaningful."""
    with pytest.raises((AttributeError, RuntimeError)):
        _call_modality(nf_obj, "erd_ers", DATA)


def test_laterality_needs_channels(nf_obj):
    """laterality_prep auto-detects L/R channels from 10-20 names.
    With generic EEG channel names it may return 0 — that is acceptable."""
    val, delay = _call_modality(nf_obj, "laterality", DATA)
    assert isinstance(val, float)
