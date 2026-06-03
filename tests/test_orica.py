"""Tests for the ORICA online ICA implementation."""

import numpy as np
import pytest

RNG = np.random.default_rng(0)
N_CH = 8
N_T = 256


@pytest.fixture()
def orica():
    from mne_rt.tools import ORICA
    o = ORICA(
        n_channels=N_CH,
        block_size=64,
        learning_rate=0.01,
        online_whitening=True,
    )
    return o


def test_orica_partial_fit(orica):
    data = RNG.standard_normal((N_CH, N_T))
    orica.partial_fit(data)
    assert orica.W is not None
    assert orica.W.shape == (N_CH, N_CH)


def test_orica_transform(orica):
    data = RNG.standard_normal((N_CH, N_T))
    orica.partial_fit(data)
    S = orica.transform(data)
    assert S.shape == (N_CH, N_T)


def test_orica_inverse_transform(orica):
    data = RNG.standard_normal((N_CH, N_T)).astype(np.float32)
    orica.partial_fit(data)
    S = orica.transform(data)
    recon = orica.inverse_transform(S)
    assert recon.shape == data.shape


def test_orica_denoise(orica):
    data = RNG.standard_normal((N_CH, N_T))
    orica.partial_fit(data)
    cleaned = orica.denoise(data, artifact_idx=[0])
    assert cleaned.shape == data.shape


def test_orica_mixing_matrix(orica):
    data = RNG.standard_normal((N_CH, N_T))
    orica.partial_fit(data)
    M = orica._get_mixing_matrix()
    assert M.shape == (N_CH, N_CH)


def test_orica_find_blink_ic_shape(orica):
    data = RNG.standard_normal((N_CH, N_T))
    orica.partial_fit(data)
    template = RNG.standard_normal(N_CH)
    idxs, corrs = orica.find_blink_ic(template, threshold=0.1)
    assert isinstance(idxs, (list, np.ndarray))
