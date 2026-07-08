"""Percentile-based feedback reward protocol for MNE-RT.

This module provides :class:`PercentileProtocol`, a stateful protocol that
rewards the participant when the current NF value crosses the Nth percentile
of a rolling history buffer.

Classes
-------
PercentileProtocol
    Rolling-percentile threshold comparator with configurable direction.
"""

from __future__ import annotations

import collections
from typing import Optional

import numpy as np


class PercentileProtocol:
    """Percentile-based NF reward protocol with rolling history.

    Maintains a fixed-length circular buffer of recent (optionally smoothed)
    NF values.  At each call to :meth:`evaluate` the Nth percentile of that
    buffer is computed and used as the dynamic reward threshold.

    A reward is issued when the current value exceeds (``"up"``) or falls
    below (``"down"``) that threshold.

    Parameters
    ----------
    percentile : float
        Target percentile in the range ``(0, 100)`` used to derive the
        dynamic threshold from the history buffer.  Default is 75.0.
    direction : {"up", "down"}
        "up"  -> reward when value > percentile threshold.
        "down" -> reward when value < percentile threshold.
        Default is "up".
    history_len : int
        Maximum number of recent values retained in the rolling buffer.
        Must be >= 2.  Default is 100.
    smoothing : float
        EMA smoothing coefficient applied to the raw input before
        comparison.  Must be in ``[0, 1)``.  ``0.0`` disables smoothing.
        Applied as: ``smoothed = (1 - smoothing) * new + smoothing * prev``.
        Default is 0.0.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.

    Notes
    -----
    The ``current_threshold`` is ``nan`` until at least two values have been
    added to the history buffer (``numpy.percentile`` requires at least one
    element, but a single-element buffer is degenerate).

    Examples
    --------
    Reward when alpha power exceeds the 75th percentile of recent history::

        proto = PercentileProtocol(percentile=75.0, direction="up")
        for value in nf_stream:
            crossed, magnitude = proto.evaluate(value)
            if crossed:
                send_reward(magnitude)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        percentile: float = 75.0,
        direction: str = "up",
        history_len: int = 100,
        smoothing: float = 0.0,
    ) -> None:
        if not (0.0 < percentile < 100.0):
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        if history_len < 2:
            raise ValueError(f"history_len must be >= 2, got {history_len}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")

        self.percentile: float = percentile
        self.direction: str = direction
        self.history_len: int = history_len
        self.smoothing: float = smoothing

        self._history: collections.deque[float] = collections.deque(maxlen=history_len)
        self._hits: collections.deque[bool] = collections.deque(maxlen=history_len)
        self._smoothed: Optional[float] = None
        self._n_evaluated: int = 0
        self._current_threshold: float = float("nan")

    def evaluate(self, value: float) -> tuple[bool, float]:
        """Evaluate one NF value and return (crossed, magnitude).

        Appends the (optionally smoothed) value to the rolling buffer,
        recomputes the percentile threshold, and tests the crossing condition.

        Parameters
        ----------
        value : float
            Current NF feature value.

        Returns
        -------
        crossed : bool
            True if the current value is on the reward side of the
            percentile threshold.
        magnitude : float
            Absolute distance between the current value and the threshold
            when ``crossed`` is True; ``0.0`` otherwise.  Returns ``0.0``
            when the history buffer contains fewer than two entries.
        """
        if self.smoothing > 0.0:
            if self._smoothed is None:
                self._smoothed = float(value)
            else:
                self._smoothed = (1.0 - self.smoothing) * value + self.smoothing * self._smoothed
        else:
            self._smoothed = float(value)

        smoothed = self._smoothed
        self._history.append(smoothed)
        self._n_evaluated += 1

        if len(self._history) < 2:
            self._current_threshold = float("nan")
            self._hits.append(False)
            return False, 0.0

        self._current_threshold = float(np.percentile(list(self._history), self.percentile))

        if self.direction == "up":
            crossed = smoothed > self._current_threshold
        else:
            crossed = smoothed < self._current_threshold

        self._hits.append(crossed)

        magnitude = abs(smoothed - self._current_threshold) if crossed else 0.0
        return crossed, magnitude

    def reset(self) -> None:
        """Reset all state.

        Clears the rolling history buffer, hit log, smoothed value, and
        evaluation counter.  All constructor parameters are preserved.
        """
        self._history.clear()
        self._hits.clear()
        self._smoothed = None
        self._n_evaluated = 0
        self._current_threshold = float("nan")

    @property
    def n_evaluated(self) -> int:
        """Total number of values evaluated since init or last reset."""
        return self._n_evaluated

    @property
    def current_threshold(self) -> float:
        """Percentile threshold computed during the last :meth:`evaluate` call.

        Returns ``nan`` until at least two values have been accumulated.
        """
        return self._current_threshold

    @property
    def hit_rate(self) -> float:
        """Fraction of recent evaluations that crossed the threshold (0–1).

        Computed over the rolling window defined by ``history_len``.
        Returns ``0.0`` when no evaluations have been recorded yet.
        """
        if not self._hits:
            return 0.0
        return sum(self._hits) / len(self._hits)

    def __repr__(self) -> str:
        return (
            f"PercentileProtocol("
            f"percentile={self.percentile}, "
            f"direction={self.direction!r}, "
            f"current_threshold={self._current_threshold:.4g}, "
            f"hit_rate={self.hit_rate:.2f}, "
            f"n_evaluated={self._n_evaluated})"
        )
