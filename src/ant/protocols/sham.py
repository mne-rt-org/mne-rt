"""Double-blind sham feedback wrapper protocol for the ANT package.

This module provides :class:`ShamProtocol`, a stateful wrapper that intercepts
real neurofeedback rewards and substitutes randomly-drawn historical values on a
configurable fraction of windows, enabling within-session double-blind designs
for neurofeedback RCTs.

Classes
-------
ShamProtocol
    Wraps any NF protocol with sham (double-blind) feedback.

References
----------
Thibaut, A., et al. (2018). Sham feedback in neurofeedback research.
Brain Stimulation, 11(3), 459–460.

Zander, T. O., et al. (2016). Towards adaptive classification for BCI.
Journal of Neural Engineering, 13(2), 026005.
"""
from __future__ import annotations

import collections
from typing import Any, Optional

import numpy as np


class ShamProtocol:
    """Wraps any NF protocol with sham (double-blind) feedback.

    In real-time NF, feedback is issued on every incoming window (e.g. every
    1 second). ShamProtocol intercepts those feedback values and on
    ``sham_rate`` fraction of windows replaces the real reward with a randomly
    shuffled historical value drawn from a rolling buffer, creating
    placebo/sham feedback windows indistinguishable from real ones.

    This enables within-session double-blind designs for neurofeedback RCTs
    without a separate sham session.

    Parameters
    ----------
    inner : any protocol with .evaluate(value) -> (bool, float)
        The real protocol to wrap (ThresholdProtocol, ZScoreProtocol, etc.).
    sham_rate : float, default 0.5
        Fraction of windows that receive sham feedback (0–1).
        0.0 = never sham; 1.0 = always sham.
    buffer_len : int, default 60
        Number of historical real-reward values to keep in the sham pool.
    rng_seed : int | None, default None
        Random seed for reproducibility.

    Attributes
    ----------
    n_real : int
        Number of real feedback windows so far.
    n_sham : int
        Number of sham feedback windows so far.
    sham_log : list[bool]
        Per-window sham flag (True = was sham).

    Raises
    ------
    ValueError
        If ``sham_rate`` is not in ``[0, 1]`` or ``buffer_len < 1``.

    Examples
    --------
    Wrap an existing protocol so 50 % of windows receive sham feedback::

        from ant.protocols import ZScoreProtocol
        from ant.protocols.sham import ShamProtocol

        inner = ZScoreProtocol(direction="up")
        proto = ShamProtocol(inner, sham_rate=0.5, rng_seed=42)
        for value in nf_stream:
            crossed, magnitude = proto.evaluate(value)
            # On sham windows, (crossed, magnitude) comes from a historical draw.

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        inner: Any,
        sham_rate: float = 0.5,
        buffer_len: int = 60,
        rng_seed: Optional[int] = None,
    ) -> None:
        if not (0.0 <= sham_rate <= 1.0):
            raise ValueError(
                f"sham_rate must be in [0, 1], got {sham_rate}"
            )
        if buffer_len < 1:
            raise ValueError(
                f"buffer_len must be >= 1, got {buffer_len}"
            )

        self.inner = inner
        self.sham_rate: float = sham_rate
        self.buffer_len: int = buffer_len

        self._rng = np.random.default_rng(rng_seed)
        # Buffer seeded with zeros; fills up from real evaluations.
        self._buffer: collections.deque[tuple[bool, float]] = collections.deque(
            [(False, 0.0)] * buffer_len, maxlen=buffer_len
        )

        self.n_real: int = 0
        self.n_sham: int = 0
        self.sham_log: list[bool] = []

    def evaluate(self, value: float) -> tuple[bool, float]:
        """Evaluate one NF value and return (crossed, magnitude).

        Always delegates to the inner protocol first so that its state
        (running statistics, adaptive threshold, etc.) advances correctly.
        On ``sham_rate`` fraction of calls the real result is silently
        discarded and a randomly-drawn historical value is returned instead.

        Parameters
        ----------
        value : float
            Current NF feature value.

        Returns
        -------
        crossed : bool
            True if the criterion was met.  On sham windows this value
            was drawn from the historical buffer, not the current signal.
        magnitude : float
            Non-negative reward magnitude.  On sham windows this value
            was drawn from the historical buffer.
        """
        real_crossed, real_magnitude = self.inner.evaluate(value)
        self._buffer.append((real_crossed, real_magnitude))

        is_sham = bool(self._rng.random() < self.sham_rate)
        self.sham_log.append(is_sham)

        if is_sham:
            self.n_sham += 1
            idx = int(self._rng.integers(0, len(self._buffer)))
            crossed, magnitude = list(self._buffer)[idx]
        else:
            self.n_real += 1
            crossed, magnitude = real_crossed, real_magnitude

        return crossed, magnitude

    def reset(self) -> None:
        """Reset sham counters, log, and buffer; also resets the inner protocol.

        The inner protocol's own ``reset()`` is called if it provides one.
        Buffer is re-seeded with zeros.
        """
        if hasattr(self.inner, "reset"):
            self.inner.reset()
        self._buffer = collections.deque(
            [(False, 0.0)] * self.buffer_len, maxlen=self.buffer_len
        )
        self.n_real = 0
        self.n_sham = 0
        self.sham_log = []

    @property
    def sham_fraction(self) -> float:
        """Observed sham fraction so far (0–1).

        Returns ``0.0`` when no evaluations have been recorded.
        """
        total = self.n_real + self.n_sham
        return self.n_sham / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"ShamProtocol("
            f"inner={self.inner!r}, "
            f"sham_rate={self.sham_rate}, "
            f"n_real={self.n_real}, "
            f"n_sham={self.n_sham}, "
            f"sham_fraction={self.sham_fraction:.2f})"
        )
