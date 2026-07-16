"""Tests for OperantProtocol."""

from __future__ import annotations

import time

import pytest

from mne_rt.protocols import (
    LinearTrendProtocol,
    OperantProtocol,
    PercentileProtocol,
    ThresholdProtocol,
)

# ─────────────────────────────────────────────────────────────────────────────
# OperantProtocol
# ─────────────────────────────────────────────────────────────────────────────


class TestOperantProtocol:
    def _make_base(self):
        # ThresholdProtocol with threshold=0 always rewards positive values
        return ThresholdProtocol(threshold=0.0, direction="up")

    def test_invalid_schedule(self):
        with pytest.raises(ValueError, match="schedule"):
            OperantProtocol(base_protocol=self._make_base(), schedule="XX")

    def test_invalid_ratio(self):
        with pytest.raises(ValueError, match="ratio"):
            OperantProtocol(base_protocol=self._make_base(), schedule="FR", ratio=0)

    def test_invalid_interval(self):
        with pytest.raises(ValueError, match="interval"):
            OperantProtocol(base_protocol=self._make_base(), schedule="FI", interval=-1)

    def test_fr_delivers_every_nth_hit(self):
        base = ThresholdProtocol(threshold=0.0, direction="up")
        p = OperantProtocol(base_protocol=base, schedule="FR", ratio=3)
        rewards = []
        for v in [1.0] * 9:
            crossed, mag = p.evaluate(v)
            rewards.append(crossed)
        # Should reward at hits 3, 6, 9
        reward_count = sum(rewards)
        assert reward_count == 3

    def test_vr_long_run_hit_rate(self):
        base = ThresholdProtocol(threshold=0.0, direction="up")
        p = OperantProtocol(base_protocol=base, schedule="VR", ratio=5, rng_seed=42)
        n = 500
        n_rewards = sum(1 for _ in range(n) if p.evaluate(1.0)[0])
        # Expected ~n/ratio ± tolerance
        assert abs(n_rewards - n / 5) < n / 5 * 0.4

    def test_fi_rewards_first_hit_after_interval(self):
        base = ThresholdProtocol(threshold=0.0, direction="up")
        p = OperantProtocol(base_protocol=base, schedule="FI", interval=0.05)
        # First hit before interval → no reward
        crossed1, _ = p.evaluate(1.0)
        # Wait for interval to elapse
        time.sleep(0.06)
        # Next hit after interval → reward
        crossed2, _ = p.evaluate(1.0)
        assert crossed2

    def test_vi_rewards_eventually(self):
        base = ThresholdProtocol(threshold=0.0, direction="up")
        p = OperantProtocol(base_protocol=base, schedule="VI", interval=0.01, rng_seed=0)
        # First call starts the clock and draws the interval
        p.evaluate(1.0)
        # Sleep long enough to outlast any reasonable exponential(0.01) draw
        time.sleep(0.15)
        crossed, _ = p.evaluate(1.0)
        assert crossed

    def test_reset(self):
        base = ThresholdProtocol(threshold=0.0, direction="up")
        p = OperantProtocol(base_protocol=base, schedule="FR", ratio=3)
        for _ in range(5):
            p.evaluate(1.0)
        p.reset()
        assert p.n_hits == 0
        assert p.n_rewards == 0

    def test_repr(self):
        p = OperantProtocol(base_protocol=self._make_base(), schedule="FR")
        assert "OperantProtocol" in repr(p)

    def test_reward_rate(self):
        base = ThresholdProtocol(threshold=0.0, direction="up")
        p = OperantProtocol(base_protocol=base, schedule="FR", ratio=2)
        for _ in range(10):
            p.evaluate(1.0)
        assert p.reward_rate == pytest.approx(0.5, abs=0.1)

    def test_current_threshold_passes_through_threshold_attr(self):
        base = ThresholdProtocol(threshold=2.5, direction="up")
        p = OperantProtocol(base_protocol=base, schedule="FR", ratio=2)
        assert p.current_threshold == 2.5

    def test_current_threshold_passes_through_current_threshold_property(self):
        # PercentileProtocol only exposes `current_threshold`, not `.threshold`.
        base = PercentileProtocol(percentile=75.0)
        p = OperantProtocol(base_protocol=base, schedule="FR", ratio=2)
        p.evaluate(1.0)
        p.evaluate(2.0)
        assert p.current_threshold == base.current_threshold

    def test_current_threshold_none_when_unavailable(self):
        base = LinearTrendProtocol()
        p = OperantProtocol(base_protocol=base, schedule="FR", ratio=2)
        assert p.current_threshold is None
