"""Tests for RLProtocol."""

from __future__ import annotations

import pytest

from mne_rt.protocols import RLProtocol

# ─────────────────────────────────────────────────────────────────────────────
# RLProtocol
# ─────────────────────────────────────────────────────────────────────────────


class TestRLProtocol:
    def test_default_init(self):
        p = RLProtocol()
        assert p.direction == "up"
        assert p.target_hit_rate == pytest.approx(0.70)
        assert p.n_evaluated == 0
        assert p.n_explored == 0

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            RLProtocol(direction="sideways")

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="lr"):
            RLProtocol(lr=0.0)

    def test_invalid_target_hit_rate(self):
        with pytest.raises(ValueError, match="target_hit_rate"):
            RLProtocol(target_hit_rate=1.5)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            RLProtocol(epsilon=1.5)

    def test_warmup_returns_no_reward(self):
        p = RLProtocol(warmup_windows=10, epsilon=0.0)
        for _ in range(10):
            crossed, mag = p.evaluate(1.0)
            assert not crossed
            assert mag == pytest.approx(0.0)
        assert p.n_evaluated == 10

    def test_reward_after_warmup(self):
        p = RLProtocol(
            initial_threshold=0.0,
            direction="up",
            warmup_windows=5,
            epsilon=0.0,
            lr=1e-9,  # near-zero lr → threshold barely moves
        )
        for _ in range(5):
            p.evaluate(1.0)
        # threshold starts at 0.0, value=1.0 → should cross
        crossed, mag = p.evaluate(1.0)
        assert crossed
        assert mag > 0.0

    def test_threshold_increases_when_hit_rate_high(self):
        p = RLProtocol(
            initial_threshold=0.0,
            direction="up",
            target_hit_rate=0.5,
            lr=0.5,
            epsilon=0.0,
            warmup_windows=5,
            history_len=10,
        )
        # Drive many successes
        for _ in range(5):
            p.evaluate(100.0)  # warmup, always crosses
        thresh_before = p.threshold
        for _ in range(10):
            p.evaluate(100.0)  # all successes → hit_rate > target → threshold rises
        assert p.threshold > thresh_before

    def test_reset(self):
        p = RLProtocol(warmup_windows=3)
        for _ in range(5):
            p.evaluate(1.0)
        p.reset()
        assert p.n_evaluated == 0
        assert p.n_explored == 0

    def test_repr(self):
        p = RLProtocol()
        r = repr(p)
        assert "RLProtocol" in r

    def test_direction_down(self):
        p = RLProtocol(
            initial_threshold=10.0,
            direction="down",
            warmup_windows=3,
            epsilon=0.0,
            lr=1e-9,
        )
        for _ in range(3):
            p.evaluate(0.0)
        crossed, mag = p.evaluate(0.0)  # 0.0 < 10.0 → should cross for "down"
        assert crossed

    def test_current_threshold_matches_threshold(self):
        p = RLProtocol(initial_threshold=1.5)
        assert p.current_threshold == p.threshold == 1.5
