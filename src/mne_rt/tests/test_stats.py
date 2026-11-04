"""Tests for the scale-free standardisation helpers."""

import math

import pytest

from mne_rt._stats import (
    DEGENERATE_REL_TOL,
    ema_variance,
    usable_std,
    welford_std,
    zscore,
)

# Realistic (mean, std) pairs per the display-scale table in rt_stream.py.
NATIVE_SCALES = [
    pytest.param(3.2e-13, 7.9e-15, id="sensor_power-eeg"),
    pytest.param(4.1e-25, 9.0e-27, id="sensor_power-meg"),
    pytest.param(2.0e-17, 5.0e-19, id="source_graph"),
    pytest.param(1.0, 0.25, id="o1-control"),
]


# ------------------------------------------------------------------
# usable_std
# ------------------------------------------------------------------


@pytest.mark.parametrize("mean, std", NATIVE_SCALES)
def test_realistic_spreads_are_kept_untouched(mean, std):
    """The whole point: a real spread must survive, whatever its magnitude."""
    assert usable_std(std, mean) == std


@pytest.mark.parametrize("mean, std", NATIVE_SCALES)
def test_zscore_is_one_standard_deviation_out(mean, std):
    assert zscore(mean + std, mean, std) == pytest.approx(1.0)
    assert zscore(mean, mean, std) == 0.0


def test_constant_signal_is_degenerate():
    """A signal with no spread has no z-score; 0.0 is the honest answer."""
    assert usable_std(0.0, 3.2e-13) == 0.0
    assert zscore(5.0, 3.2e-13, 0.0) == 0.0


def test_cancellation_is_caught():
    """std so far below the mean that (value - mean) is rounding noise."""
    mean = 3.2e-13
    assert usable_std(mean * 1e-13, mean) == 0.0  # below rel tol
    assert usable_std(mean * 1e-10, mean) != 0.0  # comfortably above it


def test_centred_feature_is_never_spuriously_degenerate():
    """With mean ~0 the scale reference is the std itself (e.g. laterality)."""
    assert usable_std(1e-30, 0.0) == 1e-30


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_non_finite_or_negative_std_is_degenerate(bad):
    assert usable_std(bad, 1.0) == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_explicit_min_std_still_rejects_non_finite(bad):
    """A NaN must not reach a reward magnitude via the back-compat branch."""
    assert usable_std(bad, 1.0, min_std=1e-6) == 0.0


def test_rel_tol_is_above_float64_eps():
    """Leaves ~4 significant digits in (value - mean) when the test just passes."""
    assert DEGENERATE_REL_TOL > 2.220446049250313e-16 * 1e3


# ------------------------------------------------------------------
# min_std: the backward-compatibility branch
# ------------------------------------------------------------------


def test_explicit_min_std_reproduces_the_old_floor():
    """A caller who chose a floor deliberately must get exactly it back."""
    assert usable_std(7.9e-15, 3.2e-13, min_std=1e-6) == 1e-6
    assert usable_std(2.0, 1.0, min_std=1e-6) == 2.0


def test_explicit_min_std_overrides_the_degeneracy_rule():
    """Even a constant signal returns the floor when one was requested."""
    assert usable_std(0.0, 1.0, min_std=1e-6) == 1e-6
    assert zscore(2.0, 1.0, 0.0, min_std=1e-6) == pytest.approx(1e6)


# ------------------------------------------------------------------
# welford_std / ema_variance
# ------------------------------------------------------------------


def test_welford_std_needs_more_samples_than_ddof():
    assert welford_std(0.0, 1) == 0.0
    assert welford_std(-1e-30, 5) == 0.0  # rounding drove m2 negative
    assert welford_std(12.0, 4) == pytest.approx(2.0)


def test_welford_std_uses_sample_variance():
    # values [1, 2, 3]: m2 = 2, sample var = 1.0, population var = 2/3
    assert welford_std(2.0, 3) == pytest.approx(1.0)
    assert welford_std(2.0, 3, ddof=0) == pytest.approx(math.sqrt(2 / 3))


def test_ema_variance_converges_to_the_true_variance():
    """Tracks a variance, not a mean absolute deviation.

    Averaging ``abs(delta)`` instead — as the previous implementation did —
    converges to ~0.798 sigma for Gaussian input, inflating z by ~25%.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    alpha, mean, var = 0.02, 0.0, 1.0
    for x in rng.normal(0.0, 1.0, 20000):
        delta = x - mean
        var = ema_variance(var, delta, alpha)
        mean += alpha * delta
    assert math.sqrt(var) == pytest.approx(1.0, rel=0.05)


def test_ema_variance_is_unbiased():
    """Fixed point is the true variance, not (1 - alpha) times it."""
    for alpha, sigma in [(0.05, 1.0), (0.5, 1.0), (0.2, 3.0)]:
        var = sigma**2
        for i in range(4000):
            # +/- sigma alternating has exactly variance sigma**2 about 0
            var = ema_variance(var, sigma * (-1) ** i, alpha)
        assert math.sqrt(var) == pytest.approx(sigma, rel=1e-6)


@pytest.mark.parametrize("k", [1e-24, 1e-14, 1.0, 1e6])
def test_ema_variance_is_scale_equivariant(k):
    """Scaling the data by k scales the standard deviation by k."""
    var = 1.0 * k**2
    for i in range(50):
        var = ema_variance(var, 0.3 * k * (-1) ** i, 0.1)
    ref = 1.0
    for i in range(50):
        ref = ema_variance(ref, 0.3 * (-1) ** i, 0.1)
    assert math.sqrt(var) == pytest.approx(math.sqrt(ref) * k, rel=1e-9)
