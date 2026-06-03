"""Tests for MultiBandProtocol."""

import math

import numpy as np
import pytest

from mne_rt.protocols import ThresholdProtocol, ZScoreProtocol
from mne_rt.protocols.multiband import MultiBandProtocol


def _up_proto(threshold=0.5):
    """ThresholdProtocol that rewards when value > threshold."""
    return ThresholdProtocol(threshold=threshold, direction="up")


def _down_proto(threshold=0.5):
    """ThresholdProtocol that rewards when value < threshold."""
    return ThresholdProtocol(threshold=threshold, direction="down")


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

def test_defaults():
    proto = MultiBandProtocol(_up_proto(), _down_proto())
    assert proto.require_both is True
    assert proto.up_label == "up_band"
    assert proto.down_label == "down_band"
    assert proto.n_evaluated == 0


# ------------------------------------------------------------------
# AND logic (require_both=True)
# ------------------------------------------------------------------

def test_and_logic_both_cross():
    proto = MultiBandProtocol(_up_proto(0.5), _down_proto(0.5), require_both=True)
    crossed, mag = proto.evaluate(1.0, 0.0)  # up > 0.5, down < 0.5
    assert crossed


def test_and_logic_only_up_crosses():
    proto = MultiBandProtocol(_up_proto(0.5), _down_proto(0.5), require_both=True)
    crossed, mag = proto.evaluate(1.0, 0.8)  # up > 0.5, down NOT < 0.5
    assert not crossed
    assert mag == 0.0


def test_and_logic_only_down_crosses():
    proto = MultiBandProtocol(_up_proto(0.5), _down_proto(0.5), require_both=True)
    crossed, mag = proto.evaluate(0.2, 0.1)  # up NOT > 0.5, down < 0.5
    assert not crossed
    assert mag == 0.0


def test_and_logic_neither_crosses():
    proto = MultiBandProtocol(_up_proto(0.5), _down_proto(0.5), require_both=True)
    crossed, mag = proto.evaluate(0.2, 0.8)
    assert not crossed
    assert mag == 0.0


# ------------------------------------------------------------------
# OR logic (require_both=False)
# ------------------------------------------------------------------

def test_or_logic_both_cross():
    proto = MultiBandProtocol(_up_proto(0.5), _down_proto(0.5), require_both=False)
    crossed, _ = proto.evaluate(1.0, 0.0)
    assert crossed


def test_or_logic_only_up_crosses():
    proto = MultiBandProtocol(_up_proto(0.5), _down_proto(0.5), require_both=False)
    crossed, _ = proto.evaluate(1.0, 0.8)
    assert crossed


def test_or_logic_only_down_crosses():
    proto = MultiBandProtocol(_up_proto(0.5), _down_proto(0.5), require_both=False)
    crossed, _ = proto.evaluate(0.2, 0.1)
    assert crossed


def test_or_logic_neither_crosses():
    proto = MultiBandProtocol(_up_proto(0.5), _down_proto(0.5), require_both=False)
    crossed, mag = proto.evaluate(0.2, 0.8)
    assert not crossed
    assert mag == 0.0


# ------------------------------------------------------------------
# magnitude — geometric mean when both > 0
# ------------------------------------------------------------------

def test_magnitude_geometric_mean():
    up = ThresholdProtocol(threshold=0.0, direction="up")   # always crosses
    down = ThresholdProtocol(threshold=2.0, direction="down")  # always crosses
    proto = MultiBandProtocol(up, down, require_both=True)
    # up_value=1.0 → mag_up=1.0; down_value=1.0 → mag_down=1.0
    crossed, mag = proto.evaluate(1.0, 1.0)
    assert crossed
    assert mag == pytest.approx(math.sqrt(1.0 * 1.0))


def test_magnitude_geometric_mean_unequal():
    up = ThresholdProtocol(threshold=0.0, direction="up")
    down = ThresholdProtocol(threshold=2.0, direction="down")
    proto = MultiBandProtocol(up, down, require_both=True)
    # up_value=4.0 → mag_up=4.0; down_value=1.5 → mag_down=0.5
    crossed, mag = proto.evaluate(4.0, 1.5)
    assert crossed
    assert mag == pytest.approx(math.sqrt(4.0 * 0.5))


# ------------------------------------------------------------------
# magnitude — arithmetic mean fallback when one is zero
# ------------------------------------------------------------------

def test_magnitude_arithmetic_fallback_up_zero():
    """up mag=0 (not crossed) but OR logic → fallback to arithmetic mean."""
    up = ThresholdProtocol(threshold=0.5, direction="up")    # mag_up=0 for value=0.2
    down = ThresholdProtocol(threshold=2.0, direction="down") # mag_down > 0
    proto = MultiBandProtocol(up, down, require_both=False)
    # Only down crosses: mag_up=0, mag_down=1.0
    crossed, mag = proto.evaluate(0.2, 1.0)
    assert crossed
    assert mag == pytest.approx((0.0 + 1.0) / 2.0)


# ------------------------------------------------------------------
# n_evaluated counter
# ------------------------------------------------------------------

def test_n_evaluated_increments():
    proto = MultiBandProtocol(_up_proto(), _down_proto())
    for i in range(10):
        proto.evaluate(float(i), float(i))
    assert proto.n_evaluated == 10


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------

def test_reset_counter():
    proto = MultiBandProtocol(_up_proto(), _down_proto())
    for _ in range(5):
        proto.evaluate(1.0, 0.0)
    proto.reset()
    assert proto.n_evaluated == 0


def test_reset_calls_inner_protocols():
    up = ZScoreProtocol(warmup_windows=5)
    down = ZScoreProtocol(warmup_windows=5)
    proto = MultiBandProtocol(up, down)
    for i in range(10):
        proto.evaluate(float(i), float(-i))
    proto.reset()
    assert up.n_evaluated == 0
    assert down.n_evaluated == 0


# ------------------------------------------------------------------
# Custom labels
# ------------------------------------------------------------------

def test_custom_labels():
    proto = MultiBandProtocol(
        _up_proto(), _down_proto(), up_label="alpha", down_label="theta"
    )
    assert proto.up_label == "alpha"
    assert proto.down_label == "theta"


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------

def test_repr():
    proto = MultiBandProtocol(
        _up_proto(), _down_proto(), up_label="alpha", down_label="theta"
    )
    r = repr(proto)
    assert "MultiBandProtocol" in r
    assert "alpha" in r
    assert "theta" in r
