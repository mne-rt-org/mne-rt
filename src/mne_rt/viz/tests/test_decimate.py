"""Peak decimation keeps what a pixel column can show."""

import numpy as np
import pytest

from mne_rt.viz._decimate import peak_decimate


def test_short_traces_are_returned_unchanged():
    x = np.arange(10.0)
    y = np.sin(x)
    xd, yd = peak_decimate(x, y, n_bins=5)
    assert xd is x and yd is y


def test_a_one_sample_spike_keeps_its_full_height():
    x = np.arange(5000.0)
    y = np.zeros(5000)
    y[2345] = 7.0
    y[3456] = -3.0
    xd, yd = peak_decimate(x, y, n_bins=200)
    assert xd.shape == yd.shape == (400,)
    assert yd.max() == 7.0 and yd.min() == -3.0


@pytest.mark.parametrize("n, n_bins", [(5000, 1100), (5000, 1700), (10000, 2200), (1003, 100)])
def test_every_sample_is_drawn_from_the_first_to_the_newest(n, n_bins):
    """Equal bins dropped the n % n_bins samples that did not fill one: at 1700
    px a 10 s window started at 3.2 s, its left third blank."""
    x = np.arange(float(n))
    y = np.arange(float(n))
    xd, yd = peak_decimate(x, y, n_bins=n_bins)
    assert xd[0] == x[0] and yd[0] == y[0]
    assert xd[-1] == x[-1] and yd[-1] == y[-1]
    assert np.all(np.diff(xd) >= 0)


def test_several_traces_are_decimated_together():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 5000)
    y = rng.standard_normal((20, 5000))
    xd, yd = peak_decimate(x, y, n_bins=500)
    assert yd.shape == (20, 1000)
    np.testing.assert_array_equal(yd.max(axis=1), y.max(axis=1))
    np.testing.assert_array_equal(yd.min(axis=1), y.min(axis=1))
