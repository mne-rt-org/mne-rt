"""Linear-trend neurofeedback protocol for the ANT package.

This module provides :class:`LinearTrendProtocol`, a stateful protocol that
rewards continuous improvement by detecting a statistically significant
upward (or downward) trend in the recent NF signal history using ordinary
least-squares regression.

Classes
-------
LinearTrendProtocol
    OLS-based trend detector with configurable window, slope threshold,
    and minimum goodness-of-fit (R²).
"""
from __future__ import annotations

import collections
from typing import Optional

import numpy as np


class LinearTrendProtocol:
    """Reward protocol that detects a statistically significant NF trend.

    Instead of rewarding for exceeding a fixed threshold at a single time
    point, this protocol fits an ordinary least-squares (OLS) line through
    the last ``window`` NF values and issues a reward when:

    1. The fitted slope is in the target ``direction`` **and** exceeds
       ``slope_threshold`` in absolute value, **and**
    2. The regression goodness-of-fit R² ≥ ``min_r2`` (optional quality
       gate — set to 0.0 to disable).

    This is particularly useful in clinical neurofeedback where participants
    may not reach a threshold in every window but should be encouraged for
    sustained directional change across multiple windows.

    Parameters
    ----------
    direction : {"up", "down"}
        "up"  -> reward when the slope is positive (signal trending upward).
        "down" -> reward when the slope is negative (signal trending downward).
        Default is ``"up"``.
    window : int
        Number of most-recent NF values used for each regression.  Must be
        ≥ 3.  Larger values give more stable estimates but react more slowly.
        Default is 20.
    slope_threshold : float
        Minimum absolute slope (in NF-signal units per sample) required to
        issue a reward.  Set to 0.0 to reward any trend in the right direction.
        Default is 0.0.
    min_r2 : float
        Minimum coefficient of determination (R²) of the OLS fit required
        before a reward is issued.  Values in ``[0.0, 1.0]``.  Set to 0.0
        to disable the quality gate.  Default is 0.0.
    warmup_windows : int
        Evaluations needed to fill the history buffer before rewards can be
        issued.  Must be ≥ ``window``.  Defaults to ``window``.
    smoothing : float
        EMA smoothing coefficient applied to the raw input before adding to
        history.  ``0.0`` disables smoothing.  Default is 0.0.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.

    Notes
    -----
    The OLS slope and R² are computed analytically (no external libraries
    needed) in O(window) time per evaluation.

    ``magnitude`` returned by :meth:`evaluate` is the absolute slope divided
    by the running standard deviation of the history buffer, giving a
    dimensionless measure of trend strength.

    Examples
    --------
    Reward sustained alpha-power increase over the last 20 windows::

        proto = LinearTrendProtocol(direction="up", window=20)
        for value in nf_stream:
            crossed, magnitude = proto.evaluate(value)
            if crossed:
                send_reward(magnitude)

    Require a clear trend (R² ≥ 0.5) with a non-trivial slope::

        proto = LinearTrendProtocol(
            direction="up",
            window=15,
            slope_threshold=0.01,
            min_r2=0.5,
        )

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        direction: str = "up",
        window: int = 20,
        slope_threshold: float = 0.0,
        min_r2: float = 0.0,
        warmup_windows: Optional[int] = None,
        smoothing: float = 0.0,
    ) -> None:
        if direction not in ("up", "down"):
            raise ValueError(
                f"direction must be 'up' or 'down', got {direction!r}"
            )
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if slope_threshold < 0.0:
            raise ValueError(
                f"slope_threshold must be >= 0, got {slope_threshold}"
            )
        if not (0.0 <= min_r2 <= 1.0):
            raise ValueError(
                f"min_r2 must be in [0.0, 1.0], got {min_r2}"
            )
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(
                f"smoothing must be in [0, 1), got {smoothing}"
            )

        _warmup = warmup_windows if warmup_windows is not None else window
        if _warmup < window:
            raise ValueError(
                f"warmup_windows ({_warmup}) must be >= window ({window})"
            )

        self.direction = direction
        self.window = window
        self.slope_threshold = slope_threshold
        self.min_r2 = min_r2
        self.warmup_windows = _warmup
        self.smoothing = smoothing

        self._history: collections.deque[float] = collections.deque(maxlen=window)
        self._n_evaluated: int = 0
        self._smoothed: Optional[float] = None
        self._slope: float = 0.0
        self._r2: float = 0.0

    def evaluate(self, value: float) -> tuple[bool, float]:
        """Evaluate one NF value and return ``(crossed, magnitude)``.

        Parameters
        ----------
        value : float
            Current NF feature value.

        Returns
        -------
        crossed : bool
            ``True`` when all of the following hold:
            * warmup is complete,
            * the OLS slope is in the target direction and ≥ ``slope_threshold``,
            * R² ≥ ``min_r2``.
        magnitude : float
            Absolute slope normalised by the history standard deviation
            (dimensionless trend strength).  ``0.0`` when not crossed.
        """
        if self.smoothing > 0.0:
            if self._smoothed is None:
                self._smoothed = float(value)
            else:
                self._smoothed = (
                    (1.0 - self.smoothing) * value + self.smoothing * self._smoothed
                )
        else:
            self._smoothed = float(value)

        self._history.append(self._smoothed)
        self._n_evaluated += 1

        if self._n_evaluated < self.warmup_windows:
            return False, 0.0

        y = np.array(self._history, dtype=np.float64)
        n = len(y)
        x = np.arange(n, dtype=np.float64)

        x_mean = (n - 1) / 2.0
        y_mean = y.mean()
        sxx = np.sum((x - x_mean) ** 2)
        sxy = np.sum((x - x_mean) * (y - y_mean))

        slope = sxy / sxx if sxx > 0 else 0.0
        self._slope = slope

        # R²
        y_pred = y_mean + slope * (x - x_mean)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        self._r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        std = float(np.std(y)) or 1.0

        if self.direction == "up":
            crossed = slope >= self.slope_threshold and self._r2 >= self.min_r2
        else:
            crossed = slope <= -self.slope_threshold and self._r2 >= self.min_r2

        magnitude = abs(slope) / std if crossed else 0.0
        return crossed, magnitude

    def reset(self) -> None:
        """Clear history and counters, preserving all constructor parameters."""
        self._history.clear()
        self._n_evaluated = 0
        self._smoothed = None
        self._slope = 0.0
        self._r2 = 0.0

    @property
    def slope(self) -> float:
        """OLS slope from the most recent :meth:`evaluate` call (0.0 before warmup)."""
        return self._slope

    @property
    def r2(self) -> float:
        """R² from the most recent :meth:`evaluate` call (0.0 before warmup)."""
        return self._r2

    @property
    def n_evaluated(self) -> int:
        """Total number of values evaluated since init or last :meth:`reset`."""
        return self._n_evaluated

    def __repr__(self) -> str:
        return (
            f"LinearTrendProtocol("
            f"direction={self.direction!r}, "
            f"window={self.window}, "
            f"slope={self._slope:.4g}, "
            f"r2={self._r2:.3f}, "
            f"n_evaluated={self._n_evaluated})"
        )
