"""Tests for ThresholdProtocol."""

import pytest

from mne_rt.protocols import ThresholdProtocol

# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------


def test_invalid_direction():
    with pytest.raises(ValueError):
        ThresholdProtocol(direction="sideways")


def test_invalid_adapt_rate():
    with pytest.raises(ValueError):
        ThresholdProtocol(adapt_rate=0.0)


def test_invalid_target_hit_rate():
    with pytest.raises(ValueError):
        ThresholdProtocol(target_hit_rate=1.5)


def test_invalid_smoothing():
    with pytest.raises(ValueError):
        ThresholdProtocol(smoothing=1.0)


def test_invalid_history_len():
    with pytest.raises(ValueError):
        ThresholdProtocol(history_len=4)


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------


def test_defaults():
    proto = ThresholdProtocol()
    assert proto.threshold == 0.0
    assert proto.direction == "up"
    assert proto.adaptive is False
    assert proto.n_evaluated == 0
    assert proto.hit_rate == 0.0


# ------------------------------------------------------------------
# Basic crossing behaviour
# ------------------------------------------------------------------


def test_up_crossing():
    proto = ThresholdProtocol(threshold=1.0, direction="up")
    crossed, mag = proto.evaluate(2.0)
    assert crossed
    assert mag > 0.0


def test_up_no_crossing():
    proto = ThresholdProtocol(threshold=1.0, direction="up")
    crossed, mag = proto.evaluate(0.5)
    assert not crossed
    assert mag == 0.0


def test_down_crossing():
    proto = ThresholdProtocol(threshold=1.0, direction="down")
    crossed, mag = proto.evaluate(0.5)
    assert crossed
    assert mag > 0.0


def test_down_no_crossing():
    proto = ThresholdProtocol(threshold=1.0, direction="down")
    crossed, mag = proto.evaluate(2.0)
    assert not crossed


# ------------------------------------------------------------------
# Threshold get/set
# ------------------------------------------------------------------


def test_threshold_setter():
    proto = ThresholdProtocol(threshold=1.0)
    proto.threshold = 5.0
    assert proto.threshold == 5.0
    assert proto.current_threshold == 5.0


# ------------------------------------------------------------------
# current_threshold
# ------------------------------------------------------------------


def test_current_threshold_matches_threshold():
    proto = ThresholdProtocol(threshold=2.5)
    assert proto.current_threshold == proto.threshold == 2.5


def test_current_threshold_tracks_adaptive_updates():
    proto = ThresholdProtocol(
        threshold=0.0, direction="up", adaptive=True, target_hit_rate=0.5, history_len=10
    )
    for _ in range(20):
        proto.evaluate(100.0)  # always crosses -> hit_rate > target -> threshold rises
    assert proto.current_threshold == proto.threshold
    assert proto.current_threshold > 0.0


# ------------------------------------------------------------------
# Adaptive mode
# ------------------------------------------------------------------


def test_adaptive_threshold_rises_with_high_hit_rate():
    proto = ThresholdProtocol(
        threshold=0.0, direction="up", adaptive=True, target_hit_rate=0.5, history_len=10
    )
    for _ in range(15):
        proto.evaluate(100.0)
    assert proto.threshold > 0.0


def test_non_adaptive_threshold_stays_fixed():
    proto = ThresholdProtocol(threshold=1.0, direction="up", adaptive=False)
    for _ in range(20):
        proto.evaluate(100.0)
    assert proto.threshold == 1.0


# ------------------------------------------------------------------
# hit_rate
# ------------------------------------------------------------------


def test_hit_rate_all_hits():
    proto = ThresholdProtocol(threshold=0.0, direction="up")
    for _ in range(5):
        proto.evaluate(1.0)
    assert proto.hit_rate == pytest.approx(1.0)


def test_hit_rate_no_hits():
    proto = ThresholdProtocol(threshold=10.0, direction="up")
    for _ in range(5):
        proto.evaluate(1.0)
    assert proto.hit_rate == pytest.approx(0.0)


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------


def test_reset_clears_history_preserves_threshold():
    proto = ThresholdProtocol(threshold=1.0, direction="up", adaptive=True)
    for _ in range(15):
        proto.evaluate(100.0)
    threshold_before_reset = proto.threshold
    proto.reset()
    assert proto.n_evaluated == 0
    assert proto.hit_rate == 0.0
    assert proto.threshold == threshold_before_reset


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------


def test_repr():
    proto = ThresholdProtocol(threshold=1.0)
    r = repr(proto)
    assert "ThresholdProtocol" in r
    assert "threshold" in r
