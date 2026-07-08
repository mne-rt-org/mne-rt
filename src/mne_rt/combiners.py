"""Feature combiners for multi-modality real-time M/EEG processing.

When MNE-RT extracts several feature values in parallel (e.g. ``sensor_power``,
``laterality``, ``connectivity_ratio``), each produces its own numeric value
per window.  A :class:`FeatureCombiner` reduces those N values to a single
*mixed* output score that can be passed to a protocol or displayed as one trace.

Pipeline position::

    feature extraction  →  FeatureCombiner.combine()  →  Protocol  →  display

Quick examples::

    from mne_rt.combiners import WeightedSumCombiner, ZScoredNormCombiner

    # Weighted blend: 60 % alpha power, 40 % laterality
    combiner = WeightedSumCombiner(
        weights={"sensor_power": 0.6, "laterality": 0.4}
    )
    mixed = combiner.combine({"sensor_power": 1.5, "laterality": 0.3})

    # Unit-free deviation score: how far are we from baseline across all features?
    combiner = ZScoredNormCombiner(
        features=["sensor_power", "laterality", "connectivity_ratio"],
        warmup=30,
    )
    mixed = combiner.combine({"sensor_power": 1.5, "laterality": 0.3,
                               "connectivity_ratio": 0.7})

Classes
-------
FeatureCombiner
    Abstract base class.  All combiners implement :meth:`combine`.
WeightedSumCombiner
    Weighted linear combination of feature values.
GeometricMeanCombiner
    Geometric mean (suitable when features are positive ratios or powers).
ZScoredNormCombiner
    Per-feature z-score normalisation followed by Euclidean norm.
LearnedCombiner
    Data-driven combination via a fitted sklearn-compatible estimator.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class FeatureCombiner:
    """Abstract base class for multi-feature NF combiners.

    Subclass this and implement :meth:`combine` to define a custom mixing
    strategy.  All combiners share the same one-method interface so they can
    be swapped in without changing the surrounding pipeline code.

    Parameters
    ----------
    features : list of str | None, default None
        Ordered list of modality names this combiner expects.  If ``None``
        the combiner accepts any keys present in the dict passed to
        :meth:`combine`.  Subclasses may require this to be set at
        construction time (e.g. :class:`WeightedSumCombiner`).

    Notes
    -----
    The combiner receives a snapshot dict ``{modality_name: float}`` once per
    analysis window, immediately after the EMA smoothing step inside
    :meth:`~mne_rt.RTStream.record_main`.  The returned scalar replaces the
    per-modality values for protocol evaluation and display when a combiner is
    active.
    """

    def __init__(self, features: Optional[list[str]] = None) -> None:
        self.features = features

    def combine(self, values: dict[str, float]) -> float:
        """Reduce a dict of per-modality NF values to one scalar.

        Parameters
        ----------
        values : dict[str, float]
            Mapping of ``{modality_name: current_value}`` for all active
            modalities in the current window.

        Returns
        -------
        mixed : float
            Single combined NF value passed downstream to the protocol and
            display.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement combine().")

    def __repr__(self) -> str:
        feat = self.features or "any"
        return f"{type(self).__name__}(features={feat})"


# ---------------------------------------------------------------------------
# Concrete combiners
# ---------------------------------------------------------------------------


class WeightedSumCombiner(FeatureCombiner):
    """Weighted linear combination of feature values.

    Computes the weight-normalised linear blend::

        mixed = Σ(wᵢ · xᵢ) / Σ(wᵢ)

    where the sum runs only over features present in *values*.  Normalising by
    the sum of active weights means the result is unaffected by how many
    features are missing in a given window.

    Parameters
    ----------
    weights : dict[str, float]
        Mapping of ``{modality_name: weight}``.  Weights do not need to sum
        to 1 — they are normalised internally.  Negative weights are allowed
        (e.g. to subtract one feature from another).  Features absent from
        *values* at call time are silently skipped.

    Notes
    -----
    Returns ``0.0`` if none of the specified features are present in *values*,
    with a :mod:`warnings` message.

    Examples
    --------
    Alpha-power minus frontal asymmetry::

        from mne_rt.combiners import WeightedSumCombiner

        combiner = WeightedSumCombiner(
            weights={"sensor_power": 0.6, "laterality": 0.4}
        )
        mixed = combiner.combine({"sensor_power": 1.5, "laterality": 0.3})
        # mixed ≈ 0.6*1.5/1.0 + 0.4*0.3/1.0 = 1.02

    Suppressing one feature (negative weight)::

        combiner = WeightedSumCombiner(
            weights={"sensor_power": 1.0, "entropy": -0.5}
        )
    """

    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__(features=list(weights.keys()))
        self.weights = weights

    def combine(self, values: dict[str, float]) -> float:
        """Return the normalised weighted sum of available feature values."""
        total_weight = 0.0
        weighted_sum = 0.0
        for feat, w in self.weights.items():
            if feat in values:
                weighted_sum += w * values[feat]
                total_weight += w

        if total_weight == 0.0:
            warnings.warn(
                f"{type(self).__name__}: none of the specified features "
                f"({list(self.weights)}) were present in values — returning 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0

        return weighted_sum / total_weight


class GeometricMeanCombiner(FeatureCombiner):
    """Geometric mean of (positive) feature values.

    Computes the weighted geometric mean::

        mixed = exp( Σ(wᵢ · log(max(xᵢ, floor))) / Σ(wᵢ) )

    with uniform weights ``wᵢ = 1`` when *weights* is ``None``.  Input values
    are clipped to *floor* before the log transform so that zero or negative
    inputs do not cause ``NaN`` or ``−inf``.

    Best suited for features that are inherently positive and multiplicative,
    such as band-power ratios, coherence values, or connectivity measures.

    Parameters
    ----------
    features : list of str
        Ordered modality names to include.
    weights : dict[str, float] | None, default None
        Optional per-feature exponents in the weighted geometric mean.
        ``None`` applies equal weighting (all exponents = 1).
    floor : float, default 1e-9
        Minimum value each input is clipped to before ``log``.

    Examples
    --------
    Equal-weight geometric mean of three power features::

        from mne_rt.combiners import GeometricMeanCombiner

        combiner = GeometricMeanCombiner(
            features=["sensor_power", "band_ratio", "individual_peak_power"]
        )
        mixed = combiner.combine({
            "sensor_power": 2.0,
            "band_ratio": 0.5,
            "individual_peak_power": 1.0,
        })
        # mixed = (2.0 * 0.5 * 1.0) ** (1/3) ≈ 1.0

    Weighted exponents (emphasise sensor_power)::

        combiner = GeometricMeanCombiner(
            features=["sensor_power", "band_ratio"],
            weights={"sensor_power": 2.0, "band_ratio": 1.0},
        )
    """

    def __init__(
        self,
        features: list[str],
        weights: Optional[dict[str, float]] = None,
        floor: float = 1e-9,
    ) -> None:
        super().__init__(features=features)
        self.weights = weights
        self.floor = floor

    def combine(self, values: dict[str, float]) -> float:
        """Return the weighted geometric mean of available feature values."""
        log_sum = 0.0
        weight_sum = 0.0

        for feat in self.features:
            if feat not in values:
                continue
            w = self.weights.get(feat, 1.0) if self.weights else 1.0
            x = max(values[feat], self.floor)
            log_sum += w * math.log(x)
            weight_sum += w

        if weight_sum == 0.0:
            warnings.warn(
                f"{type(self).__name__}: none of the specified features "
                f"({self.features}) were present in values — returning 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0

        return math.exp(log_sum / weight_sum)


class ZScoredNormCombiner(FeatureCombiner):
    """Euclidean norm after online z-score normalisation of each feature.

    Each feature stream is independently normalised using the mean and standard
    deviation estimated from the first *warmup* windows::

        zᵢ = (xᵢ − μᵢ) / σᵢ
        mixed = ‖z‖ / √n  =  sqrt(Σ zᵢ²) / sqrt(n)

    Dividing by ``√n`` keeps the output near **1** when all features hover at
    their baseline mean, and it grows when *any* feature deviates — making it
    a natural "how different from baseline are we?" score.

    Statistics are **frozen** after warmup (no drift tracking).  To re-fit
    (e.g. between blocks), call :meth:`reset`.

    Parameters
    ----------
    features : list of str
        Modality names to include.
    warmup : int, default 30
        Number of windows collected before statistics are fixed and the
        combiner starts producing non-zero output.  Returns ``0.0`` during
        the warmup phase.

    Notes
    -----
    If a feature's standard deviation is effectively zero (constant signal),
    it is floored at ``1e-9`` to prevent division-by-zero.

    Examples
    --------
    ::

        from mne_rt.combiners import ZScoredNormCombiner

        combiner = ZScoredNormCombiner(
            features=["sensor_power", "laterality", "connectivity_ratio"],
            warmup=30,
        )
        for window_vals in session_data:         # first 30 calls return 0.0
            mixed = combiner.combine(window_vals)
    """

    def __init__(self, features: list[str], warmup: int = 30) -> None:
        super().__init__(features=features)
        self.warmup = warmup
        self._buf: dict[str, list[float]] = {f: [] for f in features}
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}
        self._warmed_up: bool = False

    def reset(self) -> None:
        """Clear collected statistics and restart the warmup phase.

        Useful when called between NF blocks so the combiner re-fits its
        baseline to the new block's distribution.
        """
        self._buf = {f: [] for f in self.features}
        self._mean.clear()
        self._std.clear()
        self._warmed_up = False

    def combine(self, values: dict[str, float]) -> float:
        """Return the z-scored Euclidean norm, or ``0.0`` during warmup."""
        if not self._warmed_up:
            # Accumulate warmup buffer
            for feat in self.features:
                if feat in values:
                    self._buf[feat].append(values[feat])

            # Check whether all features have enough samples
            ready = all(len(self._buf[f]) >= self.warmup for f in self.features)
            if ready:
                for feat in self.features:
                    buf = self._buf[feat]
                    mu = sum(buf) / len(buf)
                    variance = sum((x - mu) ** 2 for x in buf) / len(buf)
                    self._mean[feat] = mu
                    self._std[feat] = max(math.sqrt(variance), 1e-9)
                self._buf.clear()
                self._warmed_up = True
            else:
                return 0.0

        # Post-warmup: z-score each present feature then return normalised norm
        z_scores = [
            (values[f] - self._mean[f]) / self._std[f]
            for f in self.features
            if f in values and f in self._mean
        ]
        if not z_scores:
            return 0.0

        norm = math.sqrt(sum(z**2 for z in z_scores))
        return norm / math.sqrt(len(z_scores))


class LearnedCombiner(FeatureCombiner):
    """Data-driven combination via a fitted sklearn-compatible estimator.

    Assembles the feature vector ``[x₁, x₂, …, xₙ]`` (in *features* order),
    calls ``estimator.predict([[x₁, …, xₙ]])``, and returns the scalar result.
    Any ``sklearn``-style regressor works out of the box.  For classifiers,
    wrap ``predict_proba`` in a small adapter so the interface matches.

    The model must be fitted **offline** — on resting-state recordings, prior
    session data, or a dedicated calibration block — before passing it here.

    Typical estimator choices:

    * ``sklearn.linear_model.Ridge`` — regularised linear projection; low
      variance, directly interpretable weights.
    * ``sklearn.cross_decomposition.PLSRegression`` — finds the latent
      direction in feature space most correlated with a target (e.g. tinnitus
      severity).
    * ``sklearn.svm.SVR`` — non-linear kernel regression; higher capacity but
      needs more calibration data and may overfit.

    Parameters
    ----------
    features : list of str
        Ordered modality names.  The feature vector fed to *estimator* is
        built in this exact order; missing features are filled with ``0.0``.
    estimator : fitted sklearn-compatible estimator
        Must expose a ``predict(X)`` method where ``X`` has shape ``(1, n)``.

    Examples
    --------
    Offline fit, then real-time use::

        from sklearn.linear_model import Ridge
        from mne_rt.combiners import LearnedCombiner

        # --- offline (calibration session) ---
        X_cal = ...  # shape (n_windows, n_features)
        y_cal = ...  # target scores, shape (n_windows,)
        model = Ridge(alpha=1.0).fit(X_cal, y_cal)

        # --- real-time session ---
        combiner = LearnedCombiner(
            features=["sensor_power", "laterality", "connectivity_ratio"],
            estimator=model,
        )
        mixed = combiner.combine({
            "sensor_power": 1.2,
            "laterality": 0.4,
            "connectivity_ratio": 0.7,
        })
    """

    def __init__(self, features: list[str], estimator: Any) -> None:
        super().__init__(features=features)
        self.estimator = estimator

    def combine(self, values: dict[str, float]) -> float:
        """Return the estimator's prediction for the current feature vector."""
        import numpy as np

        x = np.array(
            [[values.get(f, 0.0) for f in self.features]],
            dtype=float,
        )
        result = self.estimator.predict(x)
        return float(np.ravel(result)[0])
