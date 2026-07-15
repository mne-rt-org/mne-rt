"""Tests for TransferProtocol."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mne_rt.protocols import TransferProtocol

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
        p = TransferProtocol(fname=beh_json, modality="sensor_power", zscore_threshold=0.5)
        # Prior mean ~0.3; a very high value should produce a reward
        crossed, mag = p.evaluate(100.0)
        assert crossed
        assert mag > 0

    def test_low_value_not_crossed(self, beh_json):
        p = TransferProtocol(
            fname=beh_json, modality="sensor_power", direction="up", zscore_threshold=0.5
        )
        crossed, mag = p.evaluate(-100.0)
        assert not crossed
        assert mag == 0.0

    def test_direction_down(self, beh_json):
        p = TransferProtocol(
            fname=beh_json, modality="sensor_power", direction="down", zscore_threshold=0.5
        )
        crossed, mag = p.evaluate(-100.0)
        assert crossed

    def test_reset_restores_prior(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power")
        for _ in range(10):
            p.evaluate(1.0)
        p.reset()
        assert p.mean_ == pytest.approx(p.prior_mean, rel=1e-6)

    def test_invalid_direction(self, beh_json):
        with pytest.raises(ValueError, match="direction"):
            TransferProtocol(fname=beh_json, modality="sensor_power", direction="sideways")

    def test_current_threshold_defined_before_any_evaluation(self, beh_json):
        # Unlike ZScoreProtocol, no warmup is needed — prior stats seed it.
        p = TransferProtocol(fname=beh_json, modality="sensor_power", zscore_threshold=0.5)
        expected = p.mean_ + 0.5 * p.std_
        assert p.current_threshold == pytest.approx(expected)

    def test_current_threshold_direction_down(self, beh_json):
        p = TransferProtocol(
            fname=beh_json, modality="sensor_power", direction="down", zscore_threshold=0.5
        )
        expected = p.mean_ - 0.5 * p.std_
        assert p.current_threshold == pytest.approx(expected)

    def test_repr(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power")
        assert "TransferProtocol" in repr(p)

    def test_n_evaluated_increments(self, beh_json):
        p = TransferProtocol(fname=beh_json, modality="sensor_power")
        for _ in range(5):
            p.evaluate(0.3)
        assert p.n_evaluated == 5
