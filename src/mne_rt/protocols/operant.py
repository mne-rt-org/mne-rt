"""Operant conditioning reinforcement-schedule wrapper for MNE-RT.

This module provides :class:`OperantProtocol`, which wraps any inner
protocol and gates its reward output through a classical operant conditioning
schedule (fixed ratio, variable ratio, fixed interval, or variable interval).

Classes
-------
OperantProtocol
    Operant conditioning schedule wrapper for any NF protocol.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). Schedules of reinforcement.
Appleton-Century-Crofts.

Skinner, B. F. (1938). The behavior of organisms: An experimental analysis.
Appleton-Century-Crofts.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

_VALID_SCHEDULES = ("FR", "VR", "FI", "VI")


class OperantProtocol:
    """Operant conditioning reinforcement-schedule wrapper.

    Wraps any inner NF protocol and applies a classical operant conditioning
    schedule to determine whether a reward is actually delivered.  The inner
    protocol is always evaluated so that its internal state (running statistics,
    adaptive threshold, etc.) advances correctly; the schedule only decides
    whether to pass the reward through to the caller.

    .. rubric:: Supported schedules

    ``"FR"`` (Fixed Ratio)
        Deliver reward on every ``ratio``-th hit (hit = inner protocol
        returned ``crossed=True``).
    ``"VR"`` (Variable Ratio)
        Deliver reward with probability ``1 / ratio`` on each hit (geometric
        distribution; same average number of hits per reward as FR).
    ``"FI"`` (Fixed Interval)
        Deliver reward for the **first** hit that occurs after ``interval``
        seconds have elapsed since the last reward.
    ``"VI"`` (Variable Interval)
        Deliver reward for the **first** hit that occurs after a random
        interval sampled from an exponential distribution with mean
        ``interval`` seconds.

    Parameters
    ----------
    base_protocol : any protocol with .evaluate(value) -> (bool, float)
        The inner NF protocol whose output is filtered by the schedule.
    schedule : {"FR", "VR", "FI", "VI"}
        Reinforcement schedule identifier.
    ratio : int
        Required for ``"FR"`` and ``"VR"`` schedules.  For FR: reward is
        delivered on every ``ratio``-th hit.  For VR: reward probability per
        hit is ``1 / ratio``.  Must be >= 1.  Default is 5.
    interval : float
        Required for ``"FI"`` and ``"VI"`` schedules, in seconds.  For FI:
        minimum fixed interval between rewards.  For VI: mean interval
        (exponential distribution).  Must be > 0.  Default is 30.0.
    rng_seed : int | None
        Seed for the NumPy random generator used by ``"VR"`` and ``"VI"``
        schedules.  Default is None (non-deterministic).

    Raises
    ------
    ValueError
        If ``schedule`` is not one of the valid identifiers, ``ratio < 1``,
        or ``interval <= 0``.

    Notes
    -----
    The internal clock is started on the **first** call to :meth:`evaluate`
    using :func:`time.monotonic`, not at construction time.  This avoids
    artificially burning interval time between object creation and the start
    of the session.

    ``reset()`` resets all internal counters, hit/reward tallies, and the
    clock.  It also calls ``base_protocol.reset()`` if that method exists.

    Examples
    --------
    Fixed-ratio schedule (reward every 5th hit)::

        from mne_rt.protocols import ZScoreProtocol
        from mne_rt.protocols.operant import OperantProtocol

        inner = ZScoreProtocol(direction="up")
        proto = OperantProtocol(inner, schedule="FR", ratio=5)
        for value in nf_stream:
            crossed, magnitude = proto.evaluate(value)
            if crossed:
                deliver_reward(magnitude)

    Variable-interval schedule (reward at most once per ~30 s on average)::

        proto = OperantProtocol(inner, schedule="VI", interval=30.0, rng_seed=0)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        base_protocol: Any,
        schedule: str = "FR",
        ratio: int = 5,
        interval: float = 30.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        if schedule not in _VALID_SCHEDULES:
            raise ValueError(f"schedule must be one of {_VALID_SCHEDULES}, got {schedule!r}")
        if ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {ratio}")
        if interval <= 0.0:
            raise ValueError(f"interval must be > 0, got {interval}")

        self.base_protocol: Any = base_protocol
        self.schedule: str = schedule
        self.ratio: int = int(ratio)
        self.interval: float = float(interval)

        self._rng = np.random.default_rng(rng_seed)

        self._n_hits: int = 0
        self._n_rewards: int = 0

        # Counter for FR: hits since last reward
        self._fr_counter: int = 0

        # Timing state for FI / VI
        self._clock_start: Optional[float] = None
        self._next_interval_end: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, value: float) -> tuple[bool, float]:
        """Evaluate one NF value and apply the reinforcement schedule.

        Delegates to the inner protocol unconditionally, then applies the
        schedule logic to decide whether to pass the reward through.

        Parameters
        ----------
        value : float
            Current NF feature value.

        Returns
        -------
        crossed : bool
            True if the schedule releases a reward on this evaluation.
        magnitude : float
            Reward magnitude from the inner protocol when ``crossed`` is
            True; ``0.0`` otherwise.

        Notes
        -----
        For interval schedules (``"FI"``, ``"VI"``) the clock is initialised
        on the first call.
        """
        # Start the clock on first call
        if self._clock_start is None:
            self._clock_start = time.monotonic()
            self._next_interval_end = self._clock_start + self._draw_interval()

        base_crossed, base_magnitude = self.base_protocol.evaluate(value)

        if not base_crossed:
            return False, 0.0

        # Base protocol issued a hit — run schedule logic
        self._n_hits += 1
        reward = self._apply_schedule()

        if reward:
            self._n_rewards += 1
            return True, base_magnitude
        return False, 0.0

    def reset(self) -> None:
        """Reset all internal state including counters and the clock.

        Also calls ``base_protocol.reset()`` if that method exists.
        All constructor parameters are preserved.
        """
        if hasattr(self.base_protocol, "reset"):
            self.base_protocol.reset()

        self._n_hits = 0
        self._n_rewards = 0
        self._fr_counter = 0
        self._clock_start = None
        self._next_interval_end = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_hits(self) -> int:
        """Total hits (crossed=True from base protocol) since init or reset."""
        return self._n_hits

    @property
    def n_rewards(self) -> int:
        """Total rewards delivered by the schedule since init or reset."""
        return self._n_rewards

    @property
    def reward_rate(self) -> float:
        """Fraction of hits that resulted in a reward (0–1).

        Returns 0.0 before any hits are recorded.
        """
        if self._n_hits == 0:
            return 0.0
        return self._n_rewards / self._n_hits

    @property
    def current_threshold(self) -> Optional[float]:
        """Pass-through threshold from the base protocol, if available.

        Every built-in protocol exposes ``current_threshold``; this checks
        that first and falls back to a plain ``threshold`` attribute for
        third-party base protocols that only implement the older name.
        Returns ``None`` if neither is available.
        """
        if hasattr(self.base_protocol, "current_threshold"):
            return self.base_protocol.current_threshold
        return getattr(self.base_protocol, "threshold", None)

    # ------------------------------------------------------------------
    # Schedule implementation
    # ------------------------------------------------------------------

    def _apply_schedule(self) -> bool:
        """Apply the active schedule and return True if a reward is released."""
        if self.schedule == "FR":
            return self._schedule_fr()
        elif self.schedule == "VR":
            return self._schedule_vr()
        elif self.schedule == "FI":
            return self._schedule_fi()
        else:  # "VI"
            return self._schedule_vi()

    def _schedule_fr(self) -> bool:
        """Fixed Ratio: reward every ratio-th hit."""
        self._fr_counter += 1
        if self._fr_counter >= self.ratio:
            self._fr_counter = 0
            return True
        return False

    def _schedule_vr(self) -> bool:
        """Variable Ratio: each hit rewarded with probability 1/ratio."""
        return bool(self._rng.random() < (1.0 / self.ratio))

    def _schedule_fi(self) -> bool:
        """Fixed Interval: reward first hit after the fixed interval expires."""
        now = time.monotonic()
        if now >= self._next_interval_end:
            self._next_interval_end = now + self.interval
            return True
        return False

    def _schedule_vi(self) -> bool:
        """Variable Interval: reward first hit after a random interval expires."""
        now = time.monotonic()
        if now >= self._next_interval_end:
            self._next_interval_end = now + self._draw_interval()
            return True
        return False

    def _draw_interval(self) -> float:
        """Draw a random interval for VI schedule (exponential with mean=interval)."""
        if self.schedule == "VI":
            return float(self._rng.exponential(scale=self.interval))
        return self.interval

    def __repr__(self) -> str:
        return (
            f"OperantProtocol("
            f"schedule={self.schedule!r}, "
            f"ratio={self.ratio}, "
            f"interval={self.interval}, "
            f"n_hits={self._n_hits}, "
            f"n_rewards={self._n_rewards}, "
            f"reward_rate={self.reward_rate:.2f}, "
            f"base_protocol={self.base_protocol!r})"
        )
