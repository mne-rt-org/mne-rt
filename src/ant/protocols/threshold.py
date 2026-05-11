"""Threshold-based neurofeedback reward protocol for the ANT package.

This module provides :class:`ThresholdProtocol`, a lightweight stateful object
that converts a continuous NF feature value into a binary reward signal with an
optional adaptive threshold mechanism.

Classes
-------
ThresholdProtocol
    Threshold comparator with optional EMA smoothing and adaptive threshold.
"""
from __future__ import annotations

import collections
from typing import Optional

import numpy as np


class ThresholdProtocol:
    """Threshold-based NF reward protocol with optional adaptive threshold.

    Converts a continuous NF feature value into a binary reward signal.
    Optionally adapts the threshold over time to maintain a target success rate.

    Parameters
    ----------
    threshold : float, default 0.0
        Initial decision threshold.
    direction : {"up", "down"}, default "up"
        "up"  -> reward when value > threshold (e.g., enhance alpha).
        "down" -> reward when value < threshold (e.g., suppress beta).
    adaptive : bool, default False
        If True, slowly adjust threshold to keep hit_rate near target_hit_rate.
        Uses an exponential moving average of recent successes.
    adapt_rate : float, default 0.05
        Step size for threshold adaptation (in units of the NF signal's
        running standard deviation). Larger = faster adaptation.
    target_hit_rate : float, default 0.70
        Desired proportion of windows where the threshold is crossed.
        The adaptive mechanism pushes threshold toward this rate.
    smoothing : float, default 0.0
        EMA smoothing factor for the input value before thresholding.
        0.0 = no smoothing; 0.1 = light smoothing; 0.5 = heavy smoothing.
        Applied as: smoothed = (1 - smoothing) * new + smoothing * prev
    history_len : int, default 50
        Number of recent evaluations used to estimate the running hit rate
        and running standard deviation (for adaptive scaling).

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.

    Examples
    --------
    Basic usage — reward when alpha power exceeds a fixed threshold::

        proto = ThresholdProtocol(threshold=0.5, direction="up")
        crossed, magnitude = proto.evaluate(0.8)

    Adaptive threshold that targets a 70 % success rate::

        proto = ThresholdProtocol(
            threshold=0.0,
            direction="up",
            adaptive=True,
            target_hit_rate=0.70,
        )
        for value in nf_stream:
            crossed, magnitude = proto.evaluate(value)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        direction: str = "up",
        adaptive: bool = False,
        adapt_rate: float = 0.05,
        target_hit_rate: float = 0.70,
        smoothing: float = 0.0,
        history_len: int = 50,
    ) -> None:
        if direction not in ("up", "down"):
            raise ValueError(
                f"direction must be 'up' or 'down', got {direction!r}"
            )
        if adapt_rate <= 0:
            raise ValueError(
                f"adapt_rate must be > 0, got {adapt_rate}"
            )
        if not (0 < target_hit_rate < 1):
            raise ValueError(
                f"target_hit_rate must be in (0, 1), got {target_hit_rate}"
            )
        if not (0 <= smoothing < 1):
            raise ValueError(
                f"smoothing must be in [0, 1), got {smoothing}"
            )
        if history_len < 5:
            raise ValueError(
                f"history_len must be >= 5, got {history_len}"
            )

        self._threshold: float = float(threshold)
        self.direction: str = direction
        self.adaptive: bool = adaptive
        self.adapt_rate: float = adapt_rate
        self.target_hit_rate: float = target_hit_rate
        self.smoothing: float = smoothing

        self._history: collections.deque[bool] = collections.deque(
            maxlen=history_len
        )
        self._values_history: collections.deque[float] = collections.deque(
            maxlen=history_len
        )
        self._smoothed: Optional[float] = None
        self._n_evaluated: int = 0

    def evaluate(self, value: float) -> tuple[bool, float]:
        """Evaluate one NF value and return (success, reward_magnitude).

        Steps:

        1. Apply EMA smoothing if smoothing > 0.
        2. Compare smoothed value against current threshold.
        3. Record hit/miss in history.
        4. If adaptive, update threshold.
        5. Return (crossed, magnitude).

        magnitude is 0.0 when the threshold was not crossed; a positive float
        proportional to the distance from the threshold (normalised by the
        running standard deviation) when crossed.

        Parameters
        ----------
        value : float
            Current NF feature value.

        Returns
        -------
        crossed : bool
            True if the threshold was crossed in the target direction.
        magnitude : float
            Non-negative reward magnitude. 0 when not crossed;
            ``abs(smoothed - threshold) / (running_std + eps)`` when crossed.
        """
        if self.smoothing > 0.0:
            if self._smoothed is None:
                self._smoothed = value
            else:
                self._smoothed = (
                    (1.0 - self.smoothing) * value
                    + self.smoothing * self._smoothed
                )
        else:
            self._smoothed = float(value)

        smoothed = self._smoothed

        if self.direction == "up":
            crossed = smoothed > self._threshold
        else:
            crossed = smoothed < self._threshold

        self._history.append(crossed)
        self._values_history.append(smoothed)
        self._n_evaluated += 1

        running_std = (
            float(np.std(list(self._values_history)))
            if len(self._values_history) > 1
            else 1.0
        ) or 1.0

        if self.adaptive and len(self._history) >= 10:
            step = self.adapt_rate * running_std
            if self.direction == "up":
                self._threshold += step * (self.hit_rate - self.target_hit_rate)
            else:
                self._threshold -= step * (self.hit_rate - self.target_hit_rate)

        if crossed:
            magnitude = abs(smoothed - self._threshold) / (running_std + 1e-6)
        else:
            magnitude = 0.0

        return crossed, magnitude

    def reset(self) -> None:
        """Reset hit history and smoothed value, keep threshold.

        Clears ``_history``, ``_values_history``, and ``_smoothed``, and
        resets the evaluation counter to zero.  The current threshold value
        is preserved.
        """
        self._history.clear()
        self._values_history.clear()
        self._smoothed = None
        self._n_evaluated = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of recent windows that crossed the threshold (0–1).

        Returns 0.0 when no evaluations have been recorded yet.
        """
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def threshold(self) -> float:
        """Current threshold value."""
        return self._threshold

    @threshold.setter
    def threshold(self, val: float) -> None:
        """Set threshold directly.

        Parameters
        ----------
        val : float
            New threshold value.
        """
        self._threshold = float(val)

    @property
    def n_evaluated(self) -> int:
        """Total number of values evaluated since init or last reset."""
        return self._n_evaluated

    def __repr__(self) -> str:
        return (
            f"ThresholdProtocol("
            f"threshold={self._threshold:.4g}, "
            f"direction={self.direction!r}, "
            f"adaptive={self.adaptive}, "
            f"hit_rate={self.hit_rate:.2f}, "
            f"n_evaluated={self._n_evaluated})"
        )
