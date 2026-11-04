"""Tests for PercentileProtocol."""

import numpy as np
import pytest

from mne_rt.protocols import PercentileProtocol

# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------


def test_invalid_percentile_zero():
    with pytest.raises(ValueError):
        PercentileProtocol(percentile=0.0)


def test_invalid_percentile_hundred():
    with pytest.raises(ValueError):
        PercentileProtocol(percentile=100.0)


def test_invalid_direction():
    with pytest.raises(ValueError):
        PercentileProtocol(direction="sideways")


def test_invalid_history_len():
    with pytest.raises(ValueError):
        PercentileProtocol(history_len=1)


def test_invalid_smoothing():
    with pytest.raises(ValueError):
        PercentileProtocol(smoothing=1.0)


# ------------------------------------------------------------------
# Default constructor
# ------------------------------------------------------------------


def test_defaults():
    proto = PercentileProtocol()
    assert proto.percentile == 75.0
    assert proto.direction == "up"
    assert proto.history_len == 100
    assert proto.n_evaluated == 0
    assert np.isnan(proto.current_threshold)
    assert proto.hit_rate == 0.0


# ------------------------------------------------------------------
# Single-sample behaviour (degenerate buffer)
# ------------------------------------------------------------------


def test_single_sample_no_reward():
    proto = PercentileProtocol()
    crossed, mag = proto.evaluate(1.0)
    assert not crossed
    assert mag == 0.0
    assert np.isnan(proto.current_threshold)


# ------------------------------------------------------------------
# Threshold adapts to history
# ------------------------------------------------------------------


def test_threshold_updates():
    proto = PercentileProtocol(percentile=50.0)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        proto.evaluate(v)
    # 50th percentile of [1,2,3,4,5] ≈ 3.0
    assert abs(proto.current_threshold - 3.0) < 0.5


# ------------------------------------------------------------------
# Upward crossing
# ------------------------------------------------------------------


def test_up_crossing_high_value():
    proto = PercentileProtocol(percentile=50.0, direction="up")
    # Seed history with low values
    for _ in range(10):
        proto.evaluate(0.1)
    # A large spike should cross the 50th percentile
    crossed, mag = proto.evaluate(100.0)
    assert crossed
    assert mag > 0.0


def test_up_no_crossing_low_value():
    proto = PercentileProtocol(percentile=50.0, direction="up")
    for v in np.linspace(10, 20, 20):
        proto.evaluate(v)
    # A very small value should not cross the 50th percentile upward
    crossed, mag = proto.evaluate(0.001)
    assert not crossed


# ------------------------------------------------------------------
# Downward crossing
# ------------------------------------------------------------------


def test_down_crossing_low_value():
    proto = PercentileProtocol(percentile=25.0, direction="down")
    for _ in range(10):
        proto.evaluate(10.0)
    # Very small value crosses downward below 25th percentile
    crossed, mag = proto.evaluate(0.001)
    assert crossed


def test_down_no_crossing_high_value():
    proto = PercentileProtocol(percentile=25.0, direction="down")
    for v in np.linspace(1, 10, 10):
        proto.evaluate(v)
    crossed, mag = proto.evaluate(1000.0)
    assert not crossed


# ------------------------------------------------------------------
# hit_rate
# ------------------------------------------------------------------


def test_hit_rate_range():
    proto = PercentileProtocol(percentile=50.0)
    for v in range(20):
        proto.evaluate(float(v))
    assert 0.0 <= proto.hit_rate <= 1.0


def test_hit_rate_approximately_correct():
    # With p=50 and "up" direction, ~50% of values should cross
    proto = PercentileProtocol(percentile=50.0, direction="up", history_len=200)
    rng = np.random.default_rng(0)
    for _ in range(300):
        proto.evaluate(rng.uniform(0, 1))
    assert 0.2 < proto.hit_rate < 0.8


# ------------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------------


def test_smoothing_does_not_crash():
    proto = PercentileProtocol(smoothing=0.5)
    for i in range(20):
        proto.evaluate(float(i))
    assert proto.n_evaluated == 20


# ------------------------------------------------------------------
# rolling buffer cap
# ------------------------------------------------------------------


def test_history_len_limits_buffer():
    proto = PercentileProtocol(history_len=5)
    for i in range(20):
        proto.evaluate(float(i))
    # current_threshold should reflect only the last 5 values
    assert 15.0 <= proto.current_threshold <= 19.0


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------


def test_reset_clears_state():
    proto = PercentileProtocol()
    for i in range(10):
        proto.evaluate(float(i))
    proto.reset()
    assert proto.n_evaluated == 0
    assert np.isnan(proto.current_threshold)
    assert proto.hit_rate == 0.0


def test_reset_preserves_params():
    proto = PercentileProtocol(percentile=60.0, direction="down", history_len=50)
    proto.reset()
    assert proto.percentile == 60.0
    assert proto.direction == "down"
    assert proto.history_len == 50


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------


def test_repr():
    proto = PercentileProtocol()
    r = repr(proto)
    assert "PercentileProtocol" in r
    assert "percentile" in r
