"""Tests for RLProtocol, OperantProtocol, and TransferProtocol."""
from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mne_rt.protocols import (
    RLProtocol, OperantProtocol, TransferProtocol,
    ZScoreProtocol, ThresholdProtocol,
)


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


# ─────────────────────────────────────────────────────────────────────────────
# TransferProtocol
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def beh_json(tmp_path):
    """Write a minimal MNE-RT beh JSON and return its path."""
    data = {
        "meta": {"modalities": ["sensor_power"]},
        "data": {"sensor_power": [0.1, 0.2, 0.3, 0.4, 0.5]},
    }
    path = tmp_path / "beh.json"
    path.write_text(json.dumps(data))
    return path


class TestTransferProtocol:
    def test_loads_prior(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power")
        assert p.n_prior == 5
        assert p.prior_mean == pytest.approx(np.mean([0.1, 0.2, 0.3, 0.4, 0.5]))
        assert p.prior_std > 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TransferProtocol(fname=tmp_path / "nonexistent.json", modality="x")

    def test_missing_modality_raises(self, beh_json):
        with pytest.raises(KeyError):
            TransferProtocol(fname=beh_json, modality="no_such_modality")

    def test_evaluate_uses_prior(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power",
                             zscore_threshold=0.5)
        # Prior mean ~0.3; a very high value should produce a reward
        crossed, mag = p.evaluate(100.0)
        assert crossed
        assert mag > 0

    def test_low_value_not_crossed(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power",
                             direction="up", zscore_threshold=0.5)
        crossed, mag = p.evaluate(-100.0)
        assert not crossed
        assert mag == 0.0

    def test_direction_down(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power",
                             direction="down", zscore_threshold=0.5)
        crossed, mag = p.evaluate(-100.0)
        assert crossed

    def test_reset_restores_prior(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power")
        for _ in range(10):
            p.evaluate(1.0)
        mean_after = p.mean_
        p.reset()
        assert p.mean_ == pytest.approx(p.prior_mean, rel=1e-6)

    def test_invalid_direction(self, beh_json):
        with pytest.raises(ValueError, match="direction"):
            TransferProtocol(fname=beh_json, modality="sensor_power",
                             direction="sideways")

    def test_repr(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power")
        assert "TransferProtocol" in repr(p)

    def test_n_evaluated_increments(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power")
        for _ in range(5):
            p.evaluate(0.3)
        assert p.n_evaluated == 5
