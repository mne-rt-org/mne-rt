"""Tests for ShamProtocol."""

import numpy as np
import pytest

from mne_rt.protocols import ThresholdProtocol, ZScoreProtocol
from mne_rt.protocols.sham import ShamProtocol

# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------


def test_invalid_sham_rate_negative():
    inner = ThresholdProtocol()
    with pytest.raises(ValueError):
        ShamProtocol(inner, sham_rate=-0.1)


def test_invalid_sham_rate_above_one():
    inner = ThresholdProtocol()
    with pytest.raises(ValueError):
        ShamProtocol(inner, sham_rate=1.1)


def test_invalid_buffer_len_zero():
    inner = ThresholdProtocol()
    with pytest.raises(ValueError):
        ShamProtocol(inner, buffer_len=0)


def test_invalid_buffer_len_negative():
    inner = ThresholdProtocol()
    with pytest.raises(ValueError):
        ShamProtocol(inner, buffer_len=-5)


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------


def test_defaults():
    inner = ThresholdProtocol()
    proto = ShamProtocol(inner)
    assert proto.sham_rate == 0.5
    assert proto.buffer_len == 60
    assert proto.n_real == 0
    assert proto.n_sham == 0
    assert proto.sham_log == []
    assert proto.sham_fraction == 0.0


# ------------------------------------------------------------------
# sham_rate extremes
# ------------------------------------------------------------------


def test_sham_rate_zero_never_shams():
    inner = ThresholdProtocol(threshold=0.0, direction="up")
    proto = ShamProtocol(inner, sham_rate=0.0, rng_seed=0)
    for i in range(50):
        proto.evaluate(float(i))
    assert proto.n_sham == 0
    assert proto.n_real == 50
    assert all(not s for s in proto.sham_log)


def test_sham_rate_one_always_shams():
    inner = ThresholdProtocol(threshold=0.0, direction="up")
    proto = ShamProtocol(inner, sham_rate=1.0, rng_seed=0)
    for i in range(50):
        proto.evaluate(float(i))
    assert proto.n_sham == 50
    assert proto.n_real == 0
    assert all(s for s in proto.sham_log)


# ------------------------------------------------------------------
# Approximate sham rate behaviour
# ------------------------------------------------------------------


def test_sham_rate_approximate():
    """With rng_seed, sham fraction should be close to sham_rate."""
    inner = ThresholdProtocol(threshold=0.0, direction="up")
    proto = ShamProtocol(inner, sham_rate=0.4, rng_seed=42)
    n = 500
    for i in range(n):
        proto.evaluate(float(i))
    assert abs(proto.sham_fraction - 0.4) < 0.1


# ------------------------------------------------------------------
# Inner protocol state always advances
# ------------------------------------------------------------------


def test_inner_state_advances_on_sham():
    """Inner ZScoreProtocol should accumulate evaluations even on sham windows."""
    inner = ZScoreProtocol(warmup_windows=5)
    proto = ShamProtocol(inner, sham_rate=1.0, rng_seed=0)
    for i in range(20):
        proto.evaluate(float(i))
    # All 20 should have reached inner (even though all output was sham)
    assert inner.n_evaluated == 20


# ------------------------------------------------------------------
# sham_log length
# ------------------------------------------------------------------


def test_sham_log_length():
    inner = ThresholdProtocol()
    proto = ShamProtocol(inner, rng_seed=7)
    for _ in range(30):
        proto.evaluate(1.0)
    assert len(proto.sham_log) == 30


# ------------------------------------------------------------------
# n_real + n_sham == total evaluations
# ------------------------------------------------------------------


def test_counts_sum_to_total():
    inner = ThresholdProtocol()
    proto = ShamProtocol(inner, sham_rate=0.3, rng_seed=99)
    n = 100
    for i in range(n):
        proto.evaluate(float(i))
    assert proto.n_real + proto.n_sham == n


# ------------------------------------------------------------------
# sham_fraction property
# ------------------------------------------------------------------


def test_sham_fraction_equals_n_sham_over_total():
    inner = ThresholdProtocol()
    proto = ShamProtocol(inner, sham_rate=0.5, rng_seed=1)
    for i in range(40):
        proto.evaluate(float(i))
    expected = proto.n_sham / (proto.n_real + proto.n_sham)
    assert abs(proto.sham_fraction - expected) < 1e-12


# ------------------------------------------------------------------
# Reproducibility with rng_seed
# ------------------------------------------------------------------


def test_rng_seed_reproducibility():
    def run(seed):
        inner = ThresholdProtocol(threshold=0.5)
        proto = ShamProtocol(inner, sham_rate=0.5, rng_seed=seed)
        results = []
        for i in range(30):
            results.append(proto.evaluate(float(i)))
        return results, proto.sham_log[:]

    r1, log1 = run(42)
    r2, log2 = run(42)
    assert log1 == log2
    assert r1 == r2


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------


def test_reset_clears_state():
    inner = ThresholdProtocol()
    proto = ShamProtocol(inner, sham_rate=0.5, rng_seed=0)
    for i in range(20):
        proto.evaluate(float(i))
    proto.reset()
    assert proto.n_real == 0
    assert proto.n_sham == 0
    assert proto.sham_log == []


def test_reset_calls_inner_reset():
    inner = ZScoreProtocol(warmup_windows=5)
    proto = ShamProtocol(inner, rng_seed=0)
    for i in range(10):
        proto.evaluate(float(i))
    proto.reset()
    assert inner.n_evaluated == 0


def test_reset_preserves_params():
    inner = ThresholdProtocol()
    proto = ShamProtocol(inner, sham_rate=0.3, buffer_len=20)
    proto.reset()
    assert proto.sham_rate == 0.3
    assert proto.buffer_len == 20


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------


def test_repr():
    inner = ThresholdProtocol()
    proto = ShamProtocol(inner, sham_rate=0.5)
    r = repr(proto)
    assert "ShamProtocol" in r
    assert "sham_rate" in r
