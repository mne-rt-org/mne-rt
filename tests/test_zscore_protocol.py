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
    assert proto.min_std == 1e-6
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
    proto = ZScoreProtocol(warmup_windows=1, min_std=1e-12)
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
