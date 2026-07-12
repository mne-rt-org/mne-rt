"""Z-score-based feedback reward protocol for MNE-RT.

This module provides :class:`ZScoreProtocol`, a stateful protocol that
normalises incoming NF values against a running baseline mean and standard
deviation and returns a reward proportional to the resulting z-score.

Classes
-------
ZScoreProtocol
    Rolling z-score normaliser with configurable direction and warmup.
"""

from __future__ import annotations

import collections
from typing import Optional

import numpy as np


class ZScoreProtocol:
    """Z-score feedback protocol with rolling baseline normalisation.

    Normalises each incoming NF value against the running mean and standard
    deviation accumulated since initialisation (or the last :meth:`reset`).
    A reward is issued when the z-score magnitude exceeds
    ``zscore_threshold`` in the requested ``direction``.

    During the first ``warmup_windows`` calls to :meth:`evaluate` the
    baseline statistics are accumulating; ``crossed`` is always ``False``
    and ``magnitude`` is always ``0.0`` until warmup completes.

    Parameters
    ----------
    direction : {"up", "down"}
        "up"  -> reward when z-score >  ``zscore_threshold``.
        "down" -> reward when z-score < -``zscore_threshold``.
    warmup_windows : int
        Number of initial evaluations used solely to seed the baseline
        statistics before any reward can be issued. Default is 20.
    smoothing : float
        EMA smoothing coefficient applied to the raw input before
        z-scoring.  Must be in ``[0, 1)``.  ``0.0`` disables smoothing.
        Applied as: ``smoothed = (1 - smoothing) * new + smoothing * prev``.
        Default is 0.0.
    min_std : float
        Floor applied to the running standard deviation to prevent
        division by zero or near-zero blowup. Default is 1e-6.
    zscore_threshold : float
        Minimum absolute z-score required to issue a reward. Default is 0.5.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.

    Notes
    -----
    The running mean and variance are updated with Welford's online algorithm,
    which is numerically stable and requires O(1) memory per update.

    Examples
    --------
    Reward upward alpha-power deviations beyond half a standard deviation::

        proto = ZScoreProtocol(direction="up", zscore_threshold=0.5)
        for value in nf_stream:
            crossed, magnitude = proto.evaluate(value)
            if crossed:
                send_reward(magnitude)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        direction: str = "up",
        warmup_windows: int = 20,
        smoothing: float = 0.0,
        min_std: float = 1e-6,
        zscore_threshold: float = 0.5,
    ) -> None:
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        if warmup_windows < 1:
            raise ValueError(f"warmup_windows must be >= 1, got {warmup_windows}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        if min_std <= 0.0:
            raise ValueError(f"min_std must be > 0, got {min_std}")
        if zscore_threshold < 0.0:
            raise ValueError(f"zscore_threshold must be >= 0, got {zscore_threshold}")

        self.direction: str = direction
        self.warmup_windows: int = warmup_windows
        self.smoothing: float = smoothing
        self.min_std: float = min_std
        self.zscore_threshold: float = zscore_threshold

        self._n_evaluated: int = 0
        self._smoothed: Optional[float] = None
        self._zscore: float = 0.0

        # Welford online mean/variance accumulators
        self._welford_mean: float = 0.0
        self._welford_m2: float = 0.0

    def evaluate(self, value: float) -> tuple[bool, float]:
        """Evaluate one NF value and return (crossed, magnitude).

        Updates the running baseline with the (optionally smoothed) input,
        computes the z-score, and determines whether the reward criterion
        is met.

        Parameters
        ----------
        value : float
            Current NF feature value.

        Returns
        -------
        crossed : bool
            True if the z-score criterion is met in the target direction
            and warmup has completed.
        magnitude : float
            Absolute value of the z-score when ``crossed`` is True;
            ``0.0`` otherwise.

        Notes
        -----
        During warmup (``n_evaluated < warmup_windows``) this method always
        returns ``(False, 0.0)`` while still accumulating baseline statistics.
        """
        if self.smoothing > 0.0:
            if self._smoothed is None:
                self._smoothed = float(value)
            else:
                self._smoothed = (1.0 - self.smoothing) * value + self.smoothing * self._smoothed
        else:
            self._smoothed = float(value)

        smoothed = self._smoothed
        self._n_evaluated += 1

        # Welford's online update
        delta = smoothed - self._welford_mean
        self._welford_mean += delta / self._n_evaluated
        delta2 = smoothed - self._welford_mean
        self._welford_m2 += delta * delta2

        if self._n_evaluated < 2:
            std = self.min_std
        else:
            std = max(
                float(np.sqrt(self._welford_m2 / (self._n_evaluated - 1))),
                self.min_std,
            )

        self._zscore = (smoothed - self._welford_mean) / std

        if self._n_evaluated <= self.warmup_windows:
            return False, 0.0

        if self.direction == "up":
            crossed = self._zscore > self.zscore_threshold
        else:
            crossed = self._zscore < -self.zscore_threshold

        magnitude = abs(self._zscore) if crossed else 0.0
        return crossed, magnitude

    def reset(self) -> None:
        """Reset all state.

        Clears the running baseline statistics, smoothed value, z-score,
        and evaluation counter.  All constructor parameters are preserved.
        """
        self._n_evaluated = 0
        self._smoothed = None
        self._zscore = 0.0
        self._welford_mean = 0.0
        self._welford_m2 = 0.0

    @property
    def n_evaluated(self) -> int:
        """Total number of values evaluated since init or last reset."""
        return self._n_evaluated

    @property
    def zscore(self) -> float:
        """Z-score computed during the most recent :meth:`evaluate` call.

        Returns ``0.0`` before any evaluation.
        """
        return self._zscore

    @property
    def mean_(self) -> float:
        """Current running baseline mean.

        Returns ``0.0`` before any evaluation.
        """
        return self._welford_mean

    @property
    def std_(self) -> float:
        """Current running baseline standard deviation.

        Returns ``min_std`` before at least two evaluations.
        """
        if self._n_evaluated < 2:
            return self.min_std
        return max(
            float(np.sqrt(self._welford_m2 / (self._n_evaluated - 1))),
            self.min_std,
        )

    @property
    def current_threshold(self) -> Optional[float]:
        """Current reward boundary in raw NF-signal units, or ``None``.

        ``ZScoreProtocol`` rewards on a *relative* criterion
        (``zscore_threshold`` standard deviations from the running mean)
        rather than a fixed level, so this converts that criterion back to
        the signal's native units as ``mean_ ± zscore_threshold * std_``
        (``+`` for ``direction="up"``, ``-`` for ``"down"``) for display by
        :class:`~mne_rt.viz.NFPlot`.  Returns ``None`` during warmup, before
        the baseline mean/std are meaningful.
        """
        if self._n_evaluated < self.warmup_windows:
            return None
        sign = 1.0 if self.direction == "up" else -1.0
        return self.mean_ + sign * self.zscore_threshold * self.std_

    def __repr__(self) -> str:
        return (
            f"ZScoreProtocol("
            f"direction={self.direction!r}, "
            f"warmup_windows={self.warmup_windows}, "
            f"zscore={self._zscore:.4g}, "
            f"mean={self.mean_:.4g}, "
            f"std={self.std_:.4g}, "
            f"n_evaluated={self._n_evaluated})"
        )
