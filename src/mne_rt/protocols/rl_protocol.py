"""Reinforcement-learning adaptive threshold protocol for MNE-RT.

This module provides :class:`RLProtocol`, an adaptive feedback protocol
that maintains a target hit rate by adjusting the threshold using a proper RL
update rule with epsilon-greedy exploration.

Classes
-------
RLProtocol
    RL-adaptive threshold protocol with epsilon-greedy exploration.
"""
from __future__ import annotations

import collections
from typing import Optional

import numpy as np


class RLProtocol:
    """Adaptive NF protocol with reinforcement-learning threshold updates.

    Adjusts the decision threshold after every evaluation to maintain a target
    hit rate using the update rule::

        threshold += lr * (hit_rate - target_hit_rate) * running_std

    Unlike :class:`~mne_rt.protocols.ThresholdProtocol` (which also has an
    adaptive mode), this protocol tracks a rolling hit rate in a fixed-length
    window, scales updates by the running standard deviation of recent values,
    and optionally applies epsilon-greedy exploration: with probability
    ``epsilon`` a reward is given regardless of the threshold.  Exploration
    trials do **not** count toward the hit-rate used for threshold updates.

    During the first ``warmup_windows`` calls to :meth:`evaluate` the
    threshold is frozen and ``crossed`` is always ``False``.

    Parameters
    ----------
    direction : {"up", "down"}
        "up"  -> reward when value > threshold (e.g., enhance alpha power).
        "down" -> reward when value < threshold (e.g., suppress beta power).
        Default is "up".
    initial_threshold : float
        Starting decision threshold.  Default is 0.0.
    target_hit_rate : float
        Desired proportion of non-exploration windows that cross the
        threshold.  Must be strictly in ``(0, 1)``.  Default is 0.70.
    lr : float
        Learning rate for threshold updates.  Must be > 0.  Default is 0.05.
    epsilon : float
        Exploration probability.  On each call to :meth:`evaluate`,
        ``epsilon`` is the chance of giving a reward regardless of threshold.
        Must be in ``[0, 1)``.  Default is 0.05.
    smoothing : float
        EMA smoothing coefficient applied to the raw input before
        thresholding.  Must be in ``[0, 1)``.  ``0.0`` disables smoothing.
        Applied as: ``smoothed = (1 - smoothing) * new + smoothing * prev``.
        Default is 0.0.
    history_len : int
        Rolling-window length for hit-rate and running-std estimation.
        Must be >= 10.  Default is 50.
    warmup_windows : int
        Number of initial evaluations used solely to seed the rolling
        statistics before any reward can be issued or any threshold update
        is applied.  Must be >= 1.  Default is 20.
    rng_seed : int | None
        Seed for the NumPy random generator used for epsilon draws.
        Default is None (non-deterministic).

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.

    Notes
    -----
    The update rule is direction-aware: when ``direction="up"`` a higher
    threshold raises difficulty; when ``direction="down"`` a lower threshold
    raises difficulty.  The sign of the update is therefore flipped for
    "down" protocols.

    Examples
    --------
    RL-adaptive alpha-up protocol targeting 70 % hit rate::

        proto = RLProtocol(
            direction="up",
            initial_threshold=0.5,
            target_hit_rate=0.70,
            lr=0.05,
            epsilon=0.05,
        )
        for value in nf_stream:
            crossed, magnitude = proto.evaluate(value)
            if crossed:
                send_reward(magnitude)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        direction: str = "up",
        initial_threshold: float = 0.0,
        target_hit_rate: float = 0.70,
        lr: float = 0.05,
        epsilon: float = 0.05,
        smoothing: float = 0.0,
        history_len: int = 50,
        warmup_windows: int = 20,
        rng_seed: Optional[int] = None,
    ) -> None:
        if direction not in ("up", "down"):
            raise ValueError(
                f"direction must be 'up' or 'down', got {direction!r}"
            )
        if not (0.0 < target_hit_rate < 1.0):
            raise ValueError(
                f"target_hit_rate must be in (0, 1), got {target_hit_rate}"
            )
        if lr <= 0.0:
            raise ValueError(
                f"lr must be > 0, got {lr}"
            )
        if not (0.0 <= epsilon < 1.0):
            raise ValueError(
                f"epsilon must be in [0, 1), got {epsilon}"
            )
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(
                f"smoothing must be in [0, 1), got {smoothing}"
            )
        if history_len < 10:
            raise ValueError(
                f"history_len must be >= 10, got {history_len}"
            )
        if warmup_windows < 1:
            raise ValueError(
                f"warmup_windows must be >= 1, got {warmup_windows}"
            )

        self.direction: str = direction
        self.initial_threshold: float = float(initial_threshold)
        self.target_hit_rate: float = target_hit_rate
        self.lr: float = lr
        self.epsilon: float = epsilon
        self.smoothing: float = smoothing
        self.history_len: int = history_len
        self.warmup_windows: int = warmup_windows

        self._rng = np.random.default_rng(rng_seed)

        self._threshold: float = float(initial_threshold)
        self._n_evaluated: int = 0
        self._n_explored: int = 0
        self._smoothed: Optional[float] = None

        # Rolling window for hit/miss of non-exploration evaluations
        self._hit_history: collections.deque[bool] = collections.deque(
            maxlen=history_len
        )
        # Rolling window for raw smoothed values (for running std)
        self._value_history: collections.deque[float] = collections.deque(
            maxlen=history_len
        )

    def evaluate(self, value: float) -> tuple[bool, float]:
        """Evaluate one NF value and return (crossed, magnitude).

        Applies optional EMA smoothing, checks the current threshold,
        draws for epsilon-greedy exploration, updates the rolling hit history
        (exploration draws excluded), and then applies the RL threshold update.
        Warmup period suppresses all rewards and threshold updates.

        Parameters
        ----------
        value : float
            Current NF feature value.

        Returns
        -------
        crossed : bool
            True if a reward is issued.  May be True due to exploration
            even when the threshold was not crossed.  Always False during
            warmup.
        magnitude : float
            Absolute distance from the current threshold, normalised by the
            running standard deviation.  ``0.0`` when not rewarded.

        Notes
        -----
        Exploration trials (where the reward is given due to epsilon-greedy)
        are counted in :attr:`n_explored` but are not recorded in the hit
        history used for the threshold-update rule.
        """
        # --- EMA smoothing ---------------------------------------------------
        if self.smoothing > 0.0:
            if self._smoothed is None:
                self._smoothed = float(value)
            else:
                self._smoothed = (
                    (1.0 - self.smoothing) * value
                    + self.smoothing * self._smoothed
                )
        else:
            self._smoothed = float(value)

        smoothed = self._smoothed
        self._n_evaluated += 1
        self._value_history.append(smoothed)

        # Running std from rolling window (floor at 1e-6)
        running_std: float
        if len(self._value_history) > 1:
            running_std = max(
                float(np.std(list(self._value_history))), 1e-6
            )
        else:
            running_std = 1e-6

        # Warmup: accumulate statistics, issue no rewards, update no threshold
        if self._n_evaluated <= self.warmup_windows:
            return False, 0.0

        # --- Threshold crossing check ----------------------------------------
        if self.direction == "up":
            real_crossed = smoothed > self._threshold
        else:
            real_crossed = smoothed < self._threshold

        # --- Epsilon-greedy exploration ---------------------------------------
        is_explore = bool(self._rng.random() < self.epsilon)

        if is_explore:
            self._n_explored += 1
            crossed = True
            # Exploration does NOT count toward hit rate for threshold update
        else:
            crossed = real_crossed
            # Record in hit history only for non-exploration windows
            self._hit_history.append(real_crossed)

            # --- RL threshold update -----------------------------------------
            if self._hit_history:
                current_hit_rate = sum(self._hit_history) / len(self._hit_history)
                update = self.lr * (current_hit_rate - self.target_hit_rate) * running_std
                # "up" direction: raising threshold increases difficulty
                # "down" direction: lowering threshold increases difficulty
                if self.direction == "up":
                    self._threshold += update
                else:
                    self._threshold -= update

        # --- Magnitude -------------------------------------------------------
        magnitude = (
            abs(smoothed - self._threshold) / (running_std + 1e-6)
            if crossed
            else 0.0
        )
        return crossed, magnitude

    def reset(self) -> None:
        """Reset all adaptive state to initial conditions.

        Restores the threshold to ``initial_threshold``, clears the rolling
        histories, resets counters and the smoothed value.  All constructor
        parameters (``lr``, ``epsilon``, ``target_hit_rate``, etc.) are
        preserved.
        """
        self._threshold = self.initial_threshold
        self._n_evaluated = 0
        self._n_explored = 0
        self._smoothed = None
        self._hit_history.clear()
        self._value_history.clear()

    @property
    def hit_rate(self) -> float:
        """Rolling hit rate over non-exploration evaluations (0–1).

        Returns 0.0 before any non-exploration evaluations are recorded.
        """
        if not self._hit_history:
            return 0.0
        return sum(self._hit_history) / len(self._hit_history)

    @property
    def threshold(self) -> float:
        """Current decision threshold."""
        return self._threshold

    @property
    def n_evaluated(self) -> int:
        """Total number of evaluations since init or last :meth:`reset`."""
        return self._n_evaluated

    @property
    def n_explored(self) -> int:
        """Number of exploration trials (epsilon draws) since init or reset."""
        return self._n_explored

    def __repr__(self) -> str:
        return (
            f"RLProtocol("
            f"direction={self.direction!r}, "
            f"threshold={self._threshold:.4g}, "
            f"target_hit_rate={self.target_hit_rate}, "
            f"hit_rate={self.hit_rate:.2f}, "
            f"lr={self.lr}, "
            f"epsilon={self.epsilon}, "
            f"n_evaluated={self._n_evaluated}, "
            f"n_explored={self._n_explored})"
        )
