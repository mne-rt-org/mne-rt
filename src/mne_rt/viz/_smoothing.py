"""Display smoothing shared by the live viewers."""

from __future__ import annotations

import numpy as np


def check_smoothing(value: float, name: str = "display_smoothing") -> float:
    """``value`` as a float, if it is a usable :func:`ema_step` weight.

    0 weighs every new frame by nothing, so the display would never move from
    its first frame.
    """
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}.")
    return float(value)


def ema_step(state: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """One exponential-moving-average step that a non-finite value cannot poison.

    Where ``x`` is not finite the average is left as it was, and where the
    average is not finite it restarts from ``x``. A plain blend would turn a
    value NaN once and keep it NaN for good, blanking that part of the display
    until it was reset.

    Parameters
    ----------
    state : ndarray
        The current average.
    x : ndarray
        The new values, same shape.
    alpha : float
        Weight of the new values, in ``(0, 1]``.

    Returns
    -------
    ndarray
        The new average.
    """
    x_ok = np.isfinite(x)
    with np.errstate(invalid="ignore", over="ignore"):
        out = np.where(np.isfinite(state), alpha * x + (1.0 - alpha) * state, x)
    return np.where(x_ok, out, state)
