"""Tests for UpDownStaircaseProtocol."""

import numpy as np
import pytest

from mne_rt.protocols.staircase import UpDownStaircaseProtocol

# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------


def test_invalid_direction():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, direction="sideways")


def test_invalid_n_up_zero():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, n_up=0)


def test_invalid_n_down_zero():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, n_down=0)


def test_invalid_step_size_zero():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, step_size=0.0)


def test_invalid_step_size_negative():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, step_size=-0.1)


def test_invalid_step_factor_zero():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, step_factor=0.0)


def test_invalid_step_factor_above_one():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, step_factor=1.1)


def test_invalid_min_step_zero():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, min_step=0.0)


def test_invalid_smoothing_one():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, smoothing=1.0)


def test_invalid_smoothing_above_one():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, smoothing=1.5)


def test_invalid_max_reversals_zero():
    with pytest.raises(ValueError):
        UpDownStaircaseProtocol(0.5, max_reversals=0)


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------


def test_defaults():
    proto = UpDownStaircaseProtocol(initial_threshold=1.0)
    assert proto.threshold == 1.0
    assert proto.direction == "up"
    assert proto.n_up == 1
    assert proto.n_down == 2
    assert proto.n_reversals == 0
    assert proto.reversal_thresholds == []
    assert proto._n_evaluated == 0


# ------------------------------------------------------------------
# Basic up crossing
# ------------------------------------------------------------------


def test_up_crossing():
    proto = UpDownStaircaseProtocol(initial_threshold=0.5, direction="up")
    crossed, mag = proto.evaluate(1.0)
    assert crossed
    assert mag == pytest.approx(0.5)


def test_up_no_crossing():
    proto = UpDownStaircaseProtocol(initial_threshold=0.5, direction="up")
    crossed, mag = proto.evaluate(0.3)
    assert not crossed
    assert mag == 0.0


def test_down_crossing():
    proto = UpDownStaircaseProtocol(initial_threshold=0.5, direction="down")
    crossed, mag = proto.evaluate(0.2)
    assert crossed
    assert mag == pytest.approx(0.3)


def test_down_no_crossing():
    proto = UpDownStaircaseProtocol(initial_threshold=0.5, direction="down")
    crossed, mag = proto.evaluate(0.8)
    assert not crossed
    assert mag == 0.0


# ------------------------------------------------------------------
# Threshold adjustment: n_up=1 rule
# ------------------------------------------------------------------


def test_threshold_increases_after_success():
    """1-up: single success → threshold increases."""
    proto = UpDownStaircaseProtocol(initial_threshold=0.5, direction="up", n_up=1, step_size=0.1)
    proto.evaluate(1.0)  # success
    assert proto.threshold == pytest.approx(0.6)


def test_threshold_decreases_after_two_failures():
    """2-down: two consecutive failures → threshold decreases."""
    proto = UpDownStaircaseProtocol(initial_threshold=0.5, direction="up", n_down=2, step_size=0.1)
    proto.evaluate(0.0)  # failure 1
    assert proto.threshold == pytest.approx(0.5)  # not yet
    proto.evaluate(0.0)  # failure 2
    assert proto.threshold == pytest.approx(0.4)


# ------------------------------------------------------------------
# Reversal counting
# ------------------------------------------------------------------


def test_reversal_detected():
    """Success then failure should trigger a reversal."""
    proto = UpDownStaircaseProtocol(
        initial_threshold=0.5, direction="up", n_up=1, n_down=1, step_size=0.05
    )
    proto.evaluate(1.0)  # success → threshold goes up
    assert proto.n_reversals == 0
    proto.evaluate(0.0)  # failure → direction flip = reversal
    assert proto.n_reversals == 1


def test_reversal_thresholds_recorded():
    proto = UpDownStaircaseProtocol(
        initial_threshold=0.5, direction="up", n_up=1, n_down=1, step_size=0.05
    )
    proto.evaluate(1.0)  # success
    threshold_before = proto.threshold
    proto.evaluate(0.0)  # failure → reversal
    assert len(proto.reversal_thresholds) == 1
    assert proto.reversal_thresholds[0] == pytest.approx(threshold_before)


# ------------------------------------------------------------------
# Step size halving
# ------------------------------------------------------------------


def test_step_halving_after_n_reversals():
    proto = UpDownStaircaseProtocol(
        initial_threshold=0.5,
        direction="up",
        n_up=1,
        n_down=1,
        step_size=0.1,
        step_factor=0.5,
        n_reversals_before_halving=4,
        min_step=1e-6,
    )
    # Alternate success/failure to generate reversals
    for _ in range(10):
        proto.evaluate(1.0)  # success
        proto.evaluate(0.0)  # failure
    # After ≥4 reversals, step should have been halved
    assert proto._step_size < 0.1


# ------------------------------------------------------------------
# max_reversals freezes threshold
# ------------------------------------------------------------------


def test_max_reversals_freezes_threshold():
    proto = UpDownStaircaseProtocol(
        initial_threshold=0.5,
        direction="up",
        n_up=1,
        n_down=1,
        step_size=0.05,
        max_reversals=2,
    )
    # Drive reversals
    for _ in range(6):
        proto.evaluate(1.0)
        proto.evaluate(0.0)
    assert proto.n_reversals >= 2
    frozen_threshold = proto.threshold
    # Extra evaluations should not change threshold
    proto.evaluate(1.0)
    proto.evaluate(0.0)
    assert proto.threshold == frozen_threshold


# ------------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------------


def test_smoothing_does_not_crash():
    proto = UpDownStaircaseProtocol(initial_threshold=0.5, smoothing=0.5)
    for i in range(20):
        proto.evaluate(float(i % 2))
    assert proto._n_evaluated == 20


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------


def test_reset_restores_initial_state():
    proto = UpDownStaircaseProtocol(initial_threshold=0.5, step_size=0.05)
    for _ in range(20):
        proto.evaluate(1.0)
        proto.evaluate(0.0)
    proto.reset()
    assert proto.threshold == pytest.approx(0.5)
    assert proto.n_reversals == 0
    assert proto.reversal_thresholds == []
    assert proto._n_evaluated == 0
    assert proto._step_size == pytest.approx(0.05)


def test_reset_preserves_params():
    proto = UpDownStaircaseProtocol(initial_threshold=2.0, direction="down", n_up=2, n_down=3)
    proto.reset()
    assert proto.direction == "down"
    assert proto.n_up == 2
    assert proto.n_down == 3


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------


def test_repr():
    proto = UpDownStaircaseProtocol(initial_threshold=0.5)
    r = repr(proto)
    assert "UpDownStaircaseProtocol" in r
    assert "threshold" in r
    assert "n_reversals" in r
