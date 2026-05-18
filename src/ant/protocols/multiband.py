"""Multi-band simultaneous reward protocol for the ANT package.

This module provides :class:`MultiBandProtocol`, which wraps two inner
protocols (one for up-regulation, one for down-regulation) and issues a
combined reward only when both criteria are simultaneously satisfied —
e.g., alpha↑ + theta↓ for focus training or SMR↑ + theta↓ for ADHD NF.

Classes
-------
MultiBandProtocol
    Simultaneous two-band reward protocol.

References
----------
Sterman, M. B., & Egner, T. (2006). Foundation and practice of neurofeedback
for the treatment of epilepsy. Applied Psychophysiology and Biofeedback,
31(1), 21–35.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class MultiBandProtocol:
    """Reward protocol for simultaneous two-band control.

    Wraps two inner protocols (one for up-regulation, one for
    down-regulation) and issues a combined reward only when BOTH criteria
    are met simultaneously (or either, if ``require_both=False``).

    The combined ``magnitude`` is the geometric mean of the two individual
    magnitudes to ensure both bands contribute equally.  When one magnitude
    is zero the arithmetic mean is used as a fallback so that partial
    rewards are still numerically meaningful.

    Parameters
    ----------
    protocol_up : protocol with .evaluate(value) -> (bool, float)
        Protocol applied to the up-regulation value (e.g., alpha power).
    protocol_down : protocol with .evaluate(value) -> (bool, float)
        Protocol applied to the down-regulation value (e.g., theta power).
    require_both : bool, default True
        If True, both criteria must be met for a reward (AND logic).
        If False, either criterion suffices (OR logic).
    up_label : str, default "up_band"
        Human-readable label for the up-regulation band (used in
        ``__repr__`` and logging).
    down_label : str, default "down_band"
        Human-readable label for the down-regulation band (used in
        ``__repr__`` and logging).

    Notes
    -----
    Call ``evaluate(up_value, down_value)`` with TWO positional arguments —
    one from each modality/band.  Configure ``NFRealtime`` with
    ``modality=["sensor_power_alpha", "sensor_power_theta"]`` (or similar)
    and unpack the two returned values before each call.

    Examples
    --------
    Alpha-up / theta-down simultaneous reward::

        from ant.protocols import ZScoreProtocol
        from ant.protocols.multiband import MultiBandProtocol

        alpha_proto = ZScoreProtocol(direction="up")
        theta_proto = ZScoreProtocol(direction="down")

        proto = MultiBandProtocol(
            protocol_up=alpha_proto,
            protocol_down=theta_proto,
            up_label="alpha",
            down_label="theta",
        )
        for alpha_val, theta_val in zip(alpha_stream, theta_stream):
            crossed, magnitude = proto.evaluate(alpha_val, theta_val)
            if crossed:
                send_reward(magnitude)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        protocol_up: Any,
        protocol_down: Any,
        require_both: bool = True,
        up_label: str = "up_band",
        down_label: str = "down_band",
    ) -> None:
        self.protocol_up = protocol_up
        self.protocol_down = protocol_down
        self.require_both: bool = require_both
        self.up_label: str = up_label
        self.down_label: str = down_label

        self._n_evaluated: int = 0

    def evaluate(self, up_value: float, down_value: float) -> tuple[bool, float]:
        """Evaluate one pair of NF values and return (crossed, magnitude).

        Delegates to both inner protocols, then combines the results
        according to ``require_both``.  The combined magnitude is the
        geometric mean of the two individual magnitudes; when one is zero
        the arithmetic mean is used as fallback.

        Parameters
        ----------
        up_value : float
            Current NF feature value for the up-regulation band.
        down_value : float
            Current NF feature value for the down-regulation band.

        Returns
        -------
        crossed : bool
            True if the combined criterion is met.
        magnitude : float
            Non-negative combined reward magnitude.
        """
        crossed_up, mag_up = self.protocol_up.evaluate(up_value)
        crossed_down, mag_down = self.protocol_down.evaluate(down_value)

        self._n_evaluated += 1

        if self.require_both:
            crossed = crossed_up and crossed_down
        else:
            crossed = crossed_up or crossed_down

        if not crossed:
            return False, 0.0

        if mag_up > 0.0 and mag_down > 0.0:
            magnitude = float(np.sqrt(mag_up * mag_down))
        else:
            magnitude = (mag_up + mag_down) / 2.0

        return crossed, magnitude

    def reset(self) -> None:
        """Reset both inner protocols and the evaluation counter.

        Calls ``reset()`` on each inner protocol if that method exists.
        """
        if hasattr(self.protocol_up, "reset"):
            self.protocol_up.reset()
        if hasattr(self.protocol_down, "reset"):
            self.protocol_down.reset()
        self._n_evaluated = 0

    @property
    def n_evaluated(self) -> int:
        """Total number of value-pair evaluations since init or last reset."""
        return self._n_evaluated

    def __repr__(self) -> str:
        return (
            f"MultiBandProtocol("
            f"up={self.up_label!r}→{self.protocol_up!r}, "
            f"down={self.down_label!r}→{self.protocol_down!r}, "
            f"require_both={self.require_both}, "
            f"n_evaluated={self._n_evaluated})"
        )
