"""Tests for ZScoreProtocol."""

import numpy as np
import pytest

from mne_rt.protocols import ZScoreProtocol

# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------


def test_invalid_direction():
    with pytest.raises(ValueError):
        ZScoreProtocol(direction="flat")


def test_invalid_warmup_zero():
    with pytest.raises(ValueError):
        ZScoreProtocol(warmup_windows=0)


def test_invalid_smoothing():
    with pytest.raises(ValueError):
        ZScoreProtocol(smoothing=1.0)


def test_invalid_min_std():
    with pytest.raises(ValueError):
        ZScoreProtocol(min_std=0.0)


def test_invalid_zscore_threshold():
    with pytest.raises(ValueError):
        ZScoreProtocol(zscore_threshold=-0.5)


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------


def test_defaults():
    proto = ZScoreProtocol()
    assert proto.direction == "up"
    assert proto.warmup_windows == 20
    assert proto.smoothing == 0.0
    assert proto.min_std is None
    assert proto.zscore_threshold == 0.5
    assert proto.n_evaluated == 0
    assert proto.zscore == 0.0
    assert proto.mean_ == 0.0


# ------------------------------------------------------------------
# Warmup suppresses reward
# ------------------------------------------------------------------


def test_warmup_suppresses_reward():
    proto = ZScoreProtocol(warmup_windows=10)
    for i in range(10):
        crossed, mag = proto.evaluate(float(i * 100))
        assert not crossed
        assert mag == 0.0


# ------------------------------------------------------------------
# Post-warmup upward crossing
# ------------------------------------------------------------------


def test_up_crossing_after_warmup():
    proto = ZScoreProtocol(direction="up", warmup_windows=10, zscore_threshold=0.5)
    # Warmup with moderate values
    for _ in range(10):
        proto.evaluate(1.0)
    # Inject a large positive spike — should cross z > 0.5
    crossed, mag = proto.evaluate(1000.0)
    assert crossed
    assert mag > 0.5


def test_up_no_crossing_negative_spike():
    proto = ZScoreProtocol(direction="up", warmup_windows=10, zscore_threshold=0.5)
    for _ in range(10):
        proto.evaluate(100.0)
    # Large negative spike should not cross upward threshold
    crossed, mag = proto.evaluate(-1000.0)
    assert not crossed


# ------------------------------------------------------------------
# Downward crossing
# ------------------------------------------------------------------


def test_down_crossing_after_warmup():
    proto = ZScoreProtocol(direction="down", warmup_windows=10, zscore_threshold=0.5)
    for _ in range(10):
        proto.evaluate(1.0)
    crossed, mag = proto.evaluate(-1000.0)
    assert crossed
    assert mag > 0.5


# ------------------------------------------------------------------
# Running statistics accuracy
# ------------------------------------------------------------------


def test_running_mean_converges():
    proto = ZScoreProtocol(warmup_windows=1)
    values = [2.0, 4.0, 6.0, 8.0, 10.0]
    for v in values:
        proto.evaluate(v)
    # Welford mean should be close to arithmetic mean
    assert abs(proto.mean_ - np.mean(values)) < 1e-9


def test_running_std_converges():
    proto = ZScoreProtocol(warmup_windows=1)
    values = list(range(1, 101))
    for v in values:
        proto.evaluate(float(v))
    assert abs(proto.std_ - np.std(values, ddof=1)) < 0.01


# ------------------------------------------------------------------
# magnitude
# ------------------------------------------------------------------


def test_magnitude_equals_abs_zscore_when_crossed():
    proto = ZScoreProtocol(direction="up", warmup_windows=5, zscore_threshold=0.0)
    for i in range(5):
        proto.evaluate(float(i))
    crossed, mag = proto.evaluate(1000.0)
    if crossed:
        assert abs(mag - abs(proto.zscore)) < 1e-9


# ------------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------------


def test_smoothing_does_not_crash():
    proto = ZScoreProtocol(smoothing=0.7, warmup_windows=5)
    for i in range(15):
        proto.evaluate(float(i))
    assert proto.n_evaluated == 15


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------


def test_reset_clears_state():
    proto = ZScoreProtocol(warmup_windows=5)
    for i in range(10):
        proto.evaluate(float(i))
    proto.reset()
    assert proto.n_evaluated == 0
    assert proto.zscore == 0.0
    assert proto.mean_ == 0.0


def test_reset_preserves_params():
    proto = ZScoreProtocol(direction="down", warmup_windows=15, zscore_threshold=1.0)
    proto.reset()
    assert proto.direction == "down"
    assert proto.warmup_windows == 15
    assert proto.zscore_threshold == 1.0


# ------------------------------------------------------------------
# current_threshold
# ------------------------------------------------------------------


def test_current_threshold_none_during_warmup():
    proto = ZScoreProtocol(warmup_windows=10)
    for i in range(9):
        proto.evaluate(float(i))
        assert proto.current_threshold is None


def test_current_threshold_after_warmup_up():
    proto = ZScoreProtocol(direction="up", warmup_windows=5, zscore_threshold=1.0)
    for _ in range(5):
        proto.evaluate(1.0)
    expected = proto.mean_ + 1.0 * proto.std_
    assert proto.current_threshold == pytest.approx(expected)


def test_current_threshold_after_warmup_down():
    proto = ZScoreProtocol(direction="down", warmup_windows=5, zscore_threshold=1.0)
    for _ in range(5):
        proto.evaluate(1.0)
    expected = proto.mean_ - 1.0 * proto.std_
    assert proto.current_threshold == pytest.approx(expected)


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------


def test_repr():
    proto = ZScoreProtocol()
    r = repr(proto)
    assert "ZScoreProtocol" in r
    assert "direction" in r


# ------------------------------------------------------------------
# Scale-free standardisation (the property that was missing)
# ------------------------------------------------------------------

# (mean, std) at the native magnitude of real modalities. See the display-scale
# table in rt_stream.py.
_NATIVE_SCALES = [
    pytest.param(3.2e-13, 7.9e-15, id="sensor_power-eeg"),
    pytest.param(4.1e-25, 9.0e-27, id="sensor_power-meg"),
    pytest.param(2.0e-17, 5.0e-19, id="source_graph"),
    pytest.param(1.0, 0.25, id="o1-control"),
]


@pytest.mark.parametrize("loc, scale", _NATIVE_SCALES)
def test_zscore_has_unit_variance_at_native_scale(loc, scale):
    """z must have mean 0 and std 1 whatever the feature's magnitude.

    This is the assertion whose absence let an absolute 1e-6 floor stand: on
    band power (~1e-13) it replaced the real spread and every z came out ~1e-8.
    """
    rng = np.random.default_rng(0)
    warmup = 200
    proto = ZScoreProtocol(warmup_windows=warmup, zscore_threshold=0.0)
    z = []
    for v in rng.normal(loc, scale, 2000):
        proto.evaluate(float(v))
        z.append(proto.zscore)
    z = np.asarray(z[warmup:])

    assert np.std(z, ddof=1) == pytest.approx(1.0, rel=0.1)
    assert abs(np.mean(z)) < 0.1


@pytest.mark.parametrize("k", [1e-24, 1e-14, 1.0, 1e6])
def test_zscore_is_scale_invariant(k):
    """Scaling the whole signal by k must not change a single output.

    Stronger than the unit-variance test, and it will catch the next absolute
    constant anyone introduces anywhere in this path.
    """
    rng = np.random.default_rng(1)
    xs = rng.normal(3.0, 0.7, 500)

    def run(scale):
        proto = ZScoreProtocol(warmup_windows=50, zscore_threshold=0.5)
        return [proto.evaluate(float(x) * scale) for x in xs]

    base, scaled = run(1.0), run(k)
    assert [c for c, _ in scaled] == [c for c, _ in base]
    np.testing.assert_allclose([m for _, m in scaled], [m for _, m in base], rtol=1e-9)


def test_constant_signal_yields_no_zscore_and_no_reward():
    """No spread means no z-score; 0.0 rather than a fabricated crossing."""
    proto = ZScoreProtocol(warmup_windows=3, zscore_threshold=0.5)
    for _ in range(10):
        crossed, magnitude = proto.evaluate(4.2)
    assert proto.zscore == 0.0
    assert crossed is False
    assert magnitude == 0.0
    assert proto.std_ == 0.0
