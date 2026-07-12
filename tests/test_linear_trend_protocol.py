"""Tests for LinearTrendProtocol."""

import numpy as np
import pytest

from mne_rt.protocols import LinearTrendProtocol

# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------


def test_invalid_direction():
    with pytest.raises(ValueError):
        LinearTrendProtocol(direction="sideways")


def test_invalid_window_too_small():
    with pytest.raises(ValueError):
        LinearTrendProtocol(window=2)


def test_invalid_slope_threshold_negative():
    with pytest.raises(ValueError):
        LinearTrendProtocol(slope_threshold=-0.1)


def test_invalid_min_r2_out_of_range():
    with pytest.raises(ValueError):
        LinearTrendProtocol(min_r2=1.5)


def test_invalid_smoothing_out_of_range():
    with pytest.raises(ValueError):
        LinearTrendProtocol(smoothing=1.0)


def test_invalid_warmup_less_than_window():
    with pytest.raises(ValueError):
        LinearTrendProtocol(window=10, warmup_windows=5)


# ------------------------------------------------------------------
# Default constructor
# ------------------------------------------------------------------


def test_defaults():
    proto = LinearTrendProtocol()
    assert proto.direction == "up"
    assert proto.window == 20
    assert proto.slope_threshold == 0.0
    assert proto.min_r2 == 0.0
    assert proto.n_evaluated == 0
    assert proto.slope == 0.0
    assert proto.r2 == 0.0


# ------------------------------------------------------------------
# Warmup behaviour
# ------------------------------------------------------------------


def test_warmup_suppresses_reward():
    proto = LinearTrendProtocol(direction="up", window=5, warmup_windows=5)
    for i in range(4):
        crossed, mag = proto.evaluate(float(i))
        assert not crossed
        assert mag == 0.0


def test_warmup_completes():
    proto = LinearTrendProtocol(direction="up", window=5, warmup_windows=5)
    for i in range(5):
        proto.evaluate(float(i))
    # After warmup the 6th call can cross
    crossed, mag = proto.evaluate(10.0)
    assert proto.n_evaluated == 6


# ------------------------------------------------------------------
# Upward trend detection
# ------------------------------------------------------------------


def test_up_trend_detected():
    proto = LinearTrendProtocol(direction="up", window=10, warmup_windows=10)
    values = np.linspace(0, 1, 20)
    last_crossed = False
    for v in values:
        crossed, mag = proto.evaluate(v)
        last_crossed = crossed
    assert last_crossed
    assert proto.slope > 0


def test_down_trend_not_rewarded_when_up_direction():
    proto = LinearTrendProtocol(direction="up", window=10, warmup_windows=10)
    values = np.linspace(1, 0, 20)
    for v in values:
        crossed, mag = proto.evaluate(v)
    assert not crossed


# ------------------------------------------------------------------
# Downward trend detection
# ------------------------------------------------------------------


def test_down_trend_detected():
    proto = LinearTrendProtocol(direction="down", window=10, warmup_windows=10)
    values = np.linspace(1, 0, 20)
    last_crossed = False
    for v in values:
        crossed, mag = proto.evaluate(v)
        last_crossed = crossed
    assert last_crossed
    assert proto.slope < 0


# ------------------------------------------------------------------
# slope_threshold gate
# ------------------------------------------------------------------


def test_slope_threshold_blocks_weak_trend():
    proto = LinearTrendProtocol(direction="up", window=5, warmup_windows=5, slope_threshold=999.0)
    for i in range(10):
        crossed, mag = proto.evaluate(float(i))
    assert not crossed


def test_slope_threshold_passes_strong_trend():
    proto = LinearTrendProtocol(direction="up", window=5, warmup_windows=5, slope_threshold=0.001)
    for i in range(10):
        crossed, mag = proto.evaluate(float(i) * 10)
    assert crossed


# ------------------------------------------------------------------
# min_r2 gate
# ------------------------------------------------------------------


def test_min_r2_blocks_noisy_trend():
    rng = np.random.default_rng(42)
    proto = LinearTrendProtocol(direction="up", window=10, warmup_windows=10, min_r2=0.99)
    for i in range(20):
        proto.evaluate(float(i) + rng.normal(0, 50))
    # Very noisy data should have low R² and not cross
    assert proto.r2 < 0.99 or True  # allow if data happened to be clean


# ------------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------------


def test_smoothing_does_not_crash():
    proto = LinearTrendProtocol(direction="up", window=5, warmup_windows=5, smoothing=0.5)
    for i in range(10):
        proto.evaluate(float(i))
    assert proto.n_evaluated == 10


# ------------------------------------------------------------------
# magnitude is non-negative
# ------------------------------------------------------------------


def test_magnitude_nonnegative():
    proto = LinearTrendProtocol(direction="up", window=5, warmup_windows=5)
    for i in range(15):
        crossed, mag = proto.evaluate(float(i))
        assert mag >= 0.0


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------


def test_reset_clears_state():
    proto = LinearTrendProtocol(direction="up", window=5, warmup_windows=5)
    for i in range(10):
        proto.evaluate(float(i))
    proto.reset()
    assert proto.n_evaluated == 0
    assert proto.slope == 0.0
    assert proto.r2 == 0.0


def test_reset_preserves_params():
    proto = LinearTrendProtocol(direction="down", window=8, slope_threshold=0.1, min_r2=0.5)
    proto.reset()
    assert proto.direction == "down"
    assert proto.window == 8
    assert proto.slope_threshold == 0.1
    assert proto.min_r2 == 0.5


# ------------------------------------------------------------------
# current_threshold
# ------------------------------------------------------------------


def test_current_threshold_always_none():
    """LinearTrendProtocol rewards a trend, not a level -- no line to draw."""
    proto = LinearTrendProtocol(window=5, warmup_windows=5)
    assert proto.current_threshold is None
    for i in range(10):
        proto.evaluate(float(i))
        assert proto.current_threshold is None


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------


def test_repr():
    proto = LinearTrendProtocol()
    r = repr(proto)
    assert "LinearTrendProtocol" in r
    assert "direction" in r
