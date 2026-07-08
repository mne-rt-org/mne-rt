"""Up-down adaptive staircase threshold protocol for MNE-RT.

This module provides :class:`UpDownStaircaseProtocol`, a classic psychophysics
adaptive procedure that converges the NF reward threshold to a target
performance level determined by the ``n_up`` / ``n_down`` rule.

Classes
-------
UpDownStaircaseProtocol
    Up-down adaptive staircase threshold protocol.

References
----------
Levitt, H. (1971). Transformed up-down methods in psychoacoustics.
Journal of the Acoustical Society of America, 49(2B), 467–477.

García-Pérez, M. A. (1998). Forced-choice staircases with fixed step sizes:
Asymptotic and small-sample properties. Vision Research, 38(12), 1861–1881.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class UpDownStaircaseProtocol:
    """Up-down adaptive staircase threshold protocol.

    Adjusts the reward threshold after each window based on consecutive
    success/failure runs, converging the difficulty to a target performance
    level determined by the ratio n_up/n_down.

    Parameters
    ----------
    initial_threshold : float
        Starting threshold.
    direction : {"up", "down"}, default "up"
        "up" → reward when value > threshold; "down" → reward when
        value < threshold.
    n_up : int, default 1
        Consecutive successes needed to increase difficulty (tighten
        threshold).
    n_down : int, default 2
        Consecutive failures needed to decrease difficulty (loosen
        threshold).  The rule (n_up=1, n_down=2) converges to
        approximately 70.7 % success rate (Levitt, 1971).
    step_size : float, default 0.05
        Initial threshold step size per reversal.
    step_factor : float, default 0.5
        Multiplicative factor applied to ``step_size`` after
        ``n_reversals_before_halving`` reversals (standard Levitt
        procedure to zoom in on the threshold).
    n_reversals_before_halving : int, default 4
        Number of reversals before ``step_size`` is multiplied by
        ``step_factor``.
    min_step : float, default 1e-4
        Floor for ``step_size`` to prevent it collapsing to zero.
    smoothing : float, default 0.0
        EMA smoothing coefficient for the input value before
        thresholding.  Must be in ``[0, 1)``.
    max_reversals : int | None, default None
        If set, stop adapting after this many reversals (threshold
        freezes at its current value).

    Attributes
    ----------
    threshold : float
        Current threshold.
    n_reversals : int
        Number of reversals so far.
    reversal_thresholds : list[float]
        Threshold value recorded at each reversal point.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.

    Examples
    --------
    1-up/2-down staircase targeting ~70.7 % success rate::

        from mne_rt.protocols.staircase import UpDownStaircaseProtocol

        proto = UpDownStaircaseProtocol(
            initial_threshold=0.5,
            direction="up",
            n_up=1,
            n_down=2,
            step_size=0.05,
        )
        for value in nf_stream:
            crossed, magnitude = proto.evaluate(value)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        initial_threshold: float,
        direction: str = "up",
        n_up: int = 1,
        n_down: int = 2,
        step_size: float = 0.05,
        step_factor: float = 0.5,
        n_reversals_before_halving: int = 4,
        min_step: float = 1e-4,
        smoothing: float = 0.0,
        max_reversals: Optional[int] = None,
    ) -> None:
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        if n_up < 1:
            raise ValueError(f"n_up must be >= 1, got {n_up}")
        if n_down < 1:
            raise ValueError(f"n_down must be >= 1, got {n_down}")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")
        if not (0 < step_factor <= 1):
            raise ValueError(f"step_factor must be in (0, 1], got {step_factor}")
        if min_step <= 0:
            raise ValueError(f"min_step must be > 0, got {min_step}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        if max_reversals is not None and max_reversals < 1:
            raise ValueError(f"max_reversals must be >= 1 or None, got {max_reversals}")

        self._initial_threshold: float = float(initial_threshold)
        self.threshold: float = float(initial_threshold)
        self.direction: str = direction
        self.n_up: int = n_up
        self.n_down: int = n_down
        self._initial_step_size: float = float(step_size)
        self._step_size: float = float(step_size)
        self.step_factor: float = step_factor
        self.n_reversals_before_halving: int = n_reversals_before_halving
        self.min_step: float = float(min_step)
        self.smoothing: float = smoothing
        self.max_reversals: Optional[int] = max_reversals

        self.n_reversals: int = 0
        self.reversal_thresholds: list[float] = []

        self._consecutive_up: int = 0
        self._consecutive_down: int = 0
        # "up" = last change was threshold increase; "down" = decrease; None = start
        self._last_direction: Optional[str] = None
        self._n_evaluated: int = 0
        self._smoothed: Optional[float] = None

    def evaluate(self, value: float) -> tuple[bool, float]:
        """Evaluate one NF value and update the staircase.

        Applies EMA smoothing, compares against the current threshold,
        updates consecutive-run counters, and adjusts the threshold when
        an up or down rule is triggered.  A reversal is recorded whenever
        the direction of threshold change flips.

        Parameters
        ----------
        value : float
            Current NF feature value.

        Returns
        -------
        crossed : bool
            True if the threshold was crossed in the target direction.
        magnitude : float
            Distance from the threshold (absolute difference) when
            ``crossed`` is True; ``0.0`` otherwise.
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

        if self.direction == "up":
            crossed = smoothed > self.threshold
        else:
            crossed = smoothed < self.threshold

        magnitude = abs(smoothed - self.threshold) if crossed else 0.0

        frozen = self.max_reversals is not None and self.n_reversals >= self.max_reversals

        if not frozen:
            self._update_staircase(crossed)

        return crossed, magnitude

    def _update_staircase(self, crossed: bool) -> None:
        """Apply the up-down rule and update the threshold."""
        if crossed:
            self._consecutive_up += 1
            self._consecutive_down = 0
            if self._consecutive_up >= self.n_up:
                self._consecutive_up = 0
                self._apply_step(increase_difficulty=True)
        else:
            self._consecutive_down += 1
            self._consecutive_up = 0
            if self._consecutive_down >= self.n_down:
                self._consecutive_down = 0
                self._apply_step(increase_difficulty=False)

    def _apply_step(self, increase_difficulty: bool) -> None:
        """Shift the threshold by the current step size and handle reversals."""
        # direction of this change in threshold-space
        if self.direction == "up":
            # Increasing difficulty means raising the threshold
            change_dir = "up" if increase_difficulty else "down"
        else:
            # Increasing difficulty means lowering the threshold
            change_dir = "down" if increase_difficulty else "up"

        # Detect reversal before applying step
        if self._last_direction is not None and change_dir != self._last_direction:
            self.n_reversals += 1
            self.reversal_thresholds.append(self.threshold)
            # Halve step size after n_reversals_before_halving reversals
            if (
                self.n_reversals >= self.n_reversals_before_halving
                and self._step_size * self.step_factor >= self.min_step
            ):
                self._step_size = max(self._step_size * self.step_factor, self.min_step)

        self._last_direction = change_dir

        if change_dir == "up":
            self.threshold += self._step_size
        else:
            self.threshold -= self._step_size

    def reset(self) -> None:
        """Reset all state to initial conditions.

        Restores the threshold to ``initial_threshold``, clears reversal
        history, and resets all counters.  The ``step_size`` is restored
        to its value from construction (stored as ``_initial_step_size``).
        """
        self.threshold = self._initial_threshold
        self._step_size = self._initial_step_size
        self.n_reversals = 0
        self.reversal_thresholds = []
        self._consecutive_up = 0
        self._consecutive_down = 0
        self._last_direction = None
        self._n_evaluated = 0
        self._smoothed = None

    def __repr__(self) -> str:
        return (
            f"UpDownStaircaseProtocol("
            f"threshold={self.threshold:.4g}, "
            f"direction={self.direction!r}, "
            f"n_up={self.n_up}, "
            f"n_down={self.n_down}, "
            f"step_size={self._step_size:.4g}, "
            f"n_reversals={self.n_reversals}, "
            f"n_evaluated={self._n_evaluated})"
        )
