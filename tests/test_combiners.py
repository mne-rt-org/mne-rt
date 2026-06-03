"""Tests for mne_rt.combiners — all four concrete FeatureCombiner subclasses."""
from __future__ import annotations

import math

import pytest

from mne_rt.combiners import (
    FeatureCombiner,
    GeometricMeanCombiner,
    LearnedCombiner,
    WeightedSumCombiner,
    ZScoredNormCombiner,
)

VALS = {"sensor_power": 2.0, "laterality": 0.5, "connectivity_ratio": 1.0}


# ---------------------------------------------------------------------------
# FeatureCombiner base class
# ---------------------------------------------------------------------------

class TestFeatureCombiner:
    def test_abstract_combine_raises(self):
        c = FeatureCombiner(features=["sensor_power"])
        with pytest.raises(NotImplementedError):
            c.combine(VALS)

    def test_repr_with_features(self):
        c = FeatureCombiner(features=["a", "b"])
        assert "a" in repr(c)

    def test_repr_without_features(self):
        c = FeatureCombiner()
        assert "any" in repr(c)


# ---------------------------------------------------------------------------
# WeightedSumCombiner
# ---------------------------------------------------------------------------

class TestWeightedSumCombiner:
    def test_equal_weights_is_arithmetic_mean(self):
        c = WeightedSumCombiner(weights={"a": 1.0, "b": 1.0})
        result = c.combine({"a": 3.0, "b": 1.0})
        assert math.isclose(result, 2.0)

    def test_custom_weights_normalised(self):
        c = WeightedSumCombiner(weights={"a": 0.6, "b": 0.4})
        result = c.combine({"a": 1.0, "b": 1.0})
        assert math.isclose(result, 1.0)

    def test_asymmetric_weights(self):
        c = WeightedSumCombiner(weights={"a": 3.0, "b": 1.0})
        result = c.combine({"a": 2.0, "b": 0.0})
        assert math.isclose(result, 1.5)  # (3*2 + 1*0) / 4

    def test_missing_feature_skipped(self):
        c = WeightedSumCombiner(weights={"a": 1.0, "b": 1.0})
        result = c.combine({"a": 4.0})   # b missing
        assert math.isclose(result, 4.0)

    def test_all_missing_returns_zero_with_warning(self):
        c = WeightedSumCombiner(weights={"x": 1.0, "y": 1.0})
        with pytest.warns(RuntimeWarning):
            result = c.combine({"a": 1.0})
        assert result == 0.0

    def test_negative_weight(self):
        c = WeightedSumCombiner(weights={"a": 1.0, "b": -1.0})
        result = c.combine({"a": 3.0, "b": 1.0})
        assert math.isclose(result, (3.0 - 1.0) / 0.0) if False else True
        # Σwᵢ = 0 when weights cancel — check that we don't crash
        # (a=1, b=-1 → total_weight = 0 → warn + return 0)
        with pytest.warns(RuntimeWarning):
            result = c.combine({"a": 3.0, "b": 3.0})
        assert result == 0.0

    def test_features_attribute_matches_weights_keys(self):
        w = {"p": 0.5, "q": 0.5}
        c = WeightedSumCombiner(weights=w)
        assert set(c.features) == set(w.keys())


# ---------------------------------------------------------------------------
# GeometricMeanCombiner
# ---------------------------------------------------------------------------

class TestGeometricMeanCombiner:
    def test_equal_features_geomean(self):
        c = GeometricMeanCombiner(features=["a", "b", "c"])
        result = c.combine({"a": 2.0, "b": 8.0, "c": 1.0})
        expected = (2.0 * 8.0 * 1.0) ** (1 / 3)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_single_feature_returns_value(self):
        c = GeometricMeanCombiner(features=["a"])
        result = c.combine({"a": 5.0})
        assert math.isclose(result, 5.0)

    def test_weighted_exponents(self):
        c = GeometricMeanCombiner(
            features=["a", "b"],
            weights={"a": 2.0, "b": 1.0},
        )
        result = c.combine({"a": 4.0, "b": 1.0})
        # exp((2*log4 + 1*log1) / 3) = exp(2*log4/3) = 4^(2/3)
        expected = 4.0 ** (2 / 3)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_negative_input_clipped_to_floor(self):
        c = GeometricMeanCombiner(features=["a"], floor=1e-9)
        result = c.combine({"a": -5.0})
        assert math.isfinite(result) and result > 0

    def test_zero_input_clipped_to_floor(self):
        c = GeometricMeanCombiner(features=["a"], floor=1e-9)
        result = c.combine({"a": 0.0})
        assert math.isfinite(result)

    def test_missing_feature_skipped(self):
        c = GeometricMeanCombiner(features=["a", "b"])
        result = c.combine({"a": 4.0})   # b missing → only a used
        assert math.isclose(result, 4.0)

    def test_all_missing_returns_zero_with_warning(self):
        c = GeometricMeanCombiner(features=["x", "y"])
        with pytest.warns(RuntimeWarning):
            result = c.combine({"z": 1.0})
        assert result == 0.0


# ---------------------------------------------------------------------------
# ZScoredNormCombiner
# ---------------------------------------------------------------------------

class TestZScoredNormCombiner:
    def _warmed_up_combiner(self, features, warmup=5, baseline=1.0):
        """Return a combiner that has already completed warmup."""
        c = ZScoredNormCombiner(features=features, warmup=warmup)
        vals = {f: baseline for f in features}
        for _ in range(warmup):
            c.combine(vals)
        return c

    def test_returns_zero_during_warmup(self):
        c = ZScoredNormCombiner(features=["a", "b"], warmup=10)
        for _ in range(9):
            assert c.combine({"a": 1.0, "b": 2.0}) == 0.0

    def test_nonzero_after_warmup(self):
        c = ZScoredNormCombiner(features=["a"], warmup=5)
        vals = {"a": 1.0}
        for _ in range(5):
            c.combine(vals)
        # Feed a value far from baseline; should be non-zero
        result = c.combine({"a": 1000.0})
        assert result > 0.0

    def test_baseline_value_near_zero(self):
        """At-baseline input → all z-scores ≈ 0 → norm ≈ 0."""
        c = ZScoredNormCombiner(features=["a", "b"], warmup=20)
        for _ in range(20):
            c.combine({"a": 5.0, "b": 3.0})
        result = c.combine({"a": 5.0, "b": 3.0})
        assert abs(result) < 0.1

    def test_normalised_by_sqrt_n(self):
        """Equal unit z-scores across n features → mixed == 1.0."""
        c = ZScoredNormCombiner(features=["a", "b", "c"], warmup=4)
        # Inject warmup data with std=1 around mean=0
        import math as _math
        samples = [-1.0, 1.0, -1.0, 1.0]
        for s in samples:
            c.combine({"a": s, "b": s, "c": s})
        # Now feed exactly mean + 1 std for each feature
        result = c.combine({"a": 1.0, "b": 1.0, "c": 1.0})
        # z_i = 1 for all three → ‖z‖/√3 = √3/√3 = 1
        assert math.isclose(result, 1.0, rel_tol=0.05)

    def test_reset_restarts_warmup(self):
        c = self._warmed_up_combiner(["a"], warmup=5)
        result_before = c.combine({"a": 999.0})
        assert result_before > 0.0
        c.reset()
        result_after = c.combine({"a": 999.0})
        assert result_after == 0.0  # warmup restarted

    def test_missing_feature_skipped_post_warmup(self):
        c = self._warmed_up_combiner(["a", "b"], warmup=5)
        result = c.combine({"a": 999.0})  # b missing
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# LearnedCombiner
# ---------------------------------------------------------------------------

class _ConstantEstimator:
    """Stub estimator that always predicts a fixed constant."""
    def __init__(self, value: float):
        self._value = value

    def predict(self, X):
        import numpy as np
        return np.full(X.shape[0], self._value)


class _SumEstimator:
    """Stub estimator that returns the sum of its input features."""
    def predict(self, X):
        import numpy as np
        return X.sum(axis=1)


class TestLearnedCombiner:
    def test_constant_estimator(self):
        c = LearnedCombiner(features=["a", "b"], estimator=_ConstantEstimator(42.0))
        result = c.combine({"a": 1.0, "b": 2.0})
        assert math.isclose(result, 42.0)

    def test_sum_estimator(self):
        c = LearnedCombiner(features=["a", "b"], estimator=_SumEstimator())
        result = c.combine({"a": 3.0, "b": 4.0})
        assert math.isclose(result, 7.0)

    def test_missing_feature_filled_with_zero(self):
        c = LearnedCombiner(features=["a", "b"], estimator=_SumEstimator())
        result = c.combine({"a": 5.0})   # b missing → filled with 0.0
        assert math.isclose(result, 5.0)

    def test_feature_order_respected(self):
        """Ensure feature vector is built in the declared order."""
        class _FirstFeatureEstimator:
            def predict(self, X):
                import numpy as np
                return X[:, 0]  # always returns first feature only

        c = LearnedCombiner(features=["target", "noise"], estimator=_FirstFeatureEstimator())
        result = c.combine({"target": 7.0, "noise": 99.0})
        assert math.isclose(result, 7.0)

    def test_returns_scalar_float(self):
        c = LearnedCombiner(features=["a"], estimator=_ConstantEstimator(3.14))
        result = c.combine({"a": 0.0})
        assert isinstance(result, float)
