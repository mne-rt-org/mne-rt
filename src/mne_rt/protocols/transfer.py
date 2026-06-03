"""Cross-session transfer protocol for MNE-RT.

This module provides :class:`TransferProtocol`, which loads baseline
statistics from a previous session's BIDS behavioural JSON file and uses them
to seed a Welford online-mean/variance tracker.  This eliminates the warmup
period of :class:`~ant.protocols.ZScoreProtocol` and provides consistent
cross-day normalisation for longitudinal NF studies.

Classes
-------
TransferProtocol
    Z-score protocol seeded with baseline statistics from a prior session.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np


class TransferProtocol:
    """Cross-session transfer NF protocol seeded from a prior-session file.

    Loads a previous session's NF feature time-series from a BIDS
    behavioural JSON file (``beh/*.json``), computes the prior mean and
    standard deviation, and initialises Welford's online algorithm as if
    those samples had already been observed.  Subsequent calls to
    :meth:`evaluate` continue updating the running statistics, but because
    the prior already contributes, no warmup period is needed.

    The JSON file must follow ANT's BIDS behavioural format::

        {
            "meta": {"modalities": ["sensor_power"], ...},
            "data": {"sensor_power": [0.1, 0.2, ...]}
        }

    Parameters
    ----------
    fname : str | Path
        Path to a prior-session BIDS behavioural JSON file.
    modality : str
        Key inside ``data`` from which to read the prior time-series
        (e.g., ``"sensor_power"``).
    direction : {"up", "down"}, default "up"
        "up"  -> reward when z-score >  ``zscore_threshold``.
        "down" -> reward when z-score < -``zscore_threshold``.
    zscore_threshold : float, default 0.5
        Minimum absolute z-score required to issue a reward.  Must be >= 0.
    adapt_rate : float, default 0.0
        Controls how quickly the running statistics adapt to the new session.
        Set to ``0.0`` to freeze the prior statistics (pure transfer);
        higher values let the statistics drift toward the current session.
        Implemented as an EMA blend between the Welford update and the
        frozen prior: the effective weight of each new sample is
        ``adapt_rate`` (must be in ``[0, 1)``).
    smoothing : float, default 0.0
        EMA smoothing coefficient applied to the raw input before z-scoring.
        Must be in ``[0, 1)``.  ``0.0`` disables smoothing.

    Raises
    ------
    FileNotFoundError
        If ``fname`` does not point to an existing file.
    KeyError
        If the ``"data"`` key is absent from the JSON or ``modality`` is not
        found inside ``data``.
    ValueError
        If any numerical parameter is outside its valid range, or if the
        prior data array has fewer than 2 elements (insufficient for std).

    Notes
    -----
    The Welford accumulator is initialised with::

        n    = len(prior_data)
        mean = mean(prior_data)
        M2   = var(prior_data, ddof=1) * (n - 1)

    so the running ``std_`` on the first call to :meth:`evaluate` equals the
    prior standard deviation.  When ``adapt_rate > 0`` each new sample
    contributes weight ``adapt_rate`` to the mean and M2, while the existing
    accumulator retains weight ``1 - adapt_rate``.

    Examples
    --------
    Seed today's session from yesterday's baseline::

        from mne_rt.protocols.transfer import TransferProtocol

        proto = TransferProtocol(
            fname="sub-01/ses-01/beh/sub-01_ses-01_task-nf_beh.json",
            modality="sensor_power",
            direction="up",
            zscore_threshold=0.5,
        )
        for value in nf_stream:
            crossed, magnitude = proto.evaluate(value)
            if crossed:
                send_reward(magnitude)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        fname: Union[str, Path],
        modality: str,
        direction: str = "up",
        zscore_threshold: float = 0.5,
        adapt_rate: float = 0.0,
        smoothing: float = 0.0,
    ) -> None:
        # --- Parameter validation --------------------------------------------
        if direction not in ("up", "down"):
            raise ValueError(
                f"direction must be 'up' or 'down', got {direction!r}"
            )
        if zscore_threshold < 0.0:
            raise ValueError(
                f"zscore_threshold must be >= 0, got {zscore_threshold}"
            )
        if not (0.0 <= adapt_rate < 1.0):
            raise ValueError(
                f"adapt_rate must be in [0, 1), got {adapt_rate}"
            )
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(
                f"smoothing must be in [0, 1), got {smoothing}"
            )

        # --- Load prior data from file ---------------------------------------
        fname = Path(fname)
        if not fname.exists():
            raise FileNotFoundError(
                f"Prior-session file not found: {fname}"
            )
        with fname.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        if "data" not in payload:
            raise KeyError(
                f"The file {fname} does not contain a top-level 'data' key. "
                "Expected ANT BIDS JSON format: "
                '{"meta": {...}, "data": {"<modality>": [...]}}'
            )
        data_section = payload["data"]
        if modality not in data_section:
            available = list(data_section.keys())
            raise KeyError(
                f"Modality {modality!r} not found in {fname}. "
                f"Available modalities: {available}"
            )

        prior_data = np.asarray(data_section[modality], dtype=float)
        if prior_data.ndim != 1 or len(prior_data) < 2:
            raise ValueError(
                f"Prior data for modality {modality!r} must be a 1-D array "
                f"with at least 2 elements; got shape {prior_data.shape}."
            )

        # --- Store constructor parameters ------------------------------------
        self.fname: Path = fname
        self.modality: str = modality
        self.direction: str = direction
        self.zscore_threshold: float = zscore_threshold
        self.adapt_rate: float = adapt_rate
        self.smoothing: float = smoothing

        # --- Compute and cache prior statistics ------------------------------
        self._prior_data: np.ndarray = prior_data
        self._prior_mean: float = float(np.mean(prior_data))
        self._prior_std: float = float(np.std(prior_data, ddof=1))
        self._n_prior: int = len(prior_data)

        # --- Initialise Welford accumulators from prior ----------------------
        self._n_evaluated: int = 0
        self._smoothed: Optional[float] = None
        self._zscore: float = 0.0

        self._welford_n: int = self._n_prior
        self._welford_mean: float = self._prior_mean
        # M2 = sample_var * (n - 1)
        self._welford_m2: float = self._prior_std ** 2 * (self._n_prior - 1)

    def evaluate(self, value: float) -> tuple[bool, float]:
        """Evaluate one NF value and return (crossed, magnitude).

        Updates the running baseline statistics (seeded from the prior) with
        the (optionally smoothed) input, computes the z-score against the
        running mean and standard deviation, and determines whether the
        reward criterion is met.

        Parameters
        ----------
        value : float
            Current NF feature value.

        Returns
        -------
        crossed : bool
            True if the z-score criterion is met in the target direction.
        magnitude : float
            Absolute value of the z-score when ``crossed`` is True;
            ``0.0`` otherwise.

        Notes
        -----
        When ``adapt_rate = 0.0`` the running statistics are frozen at the
        prior values: every evaluation is z-scored against the prior
        mean/std.  When ``adapt_rate > 0`` new samples gradually shift the
        running statistics toward the current session distribution.
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

        # --- Welford update --------------------------------------------------
        if self.adapt_rate > 0.0:
            # Weighted Welford: new sample has effective weight adapt_rate
            # relative to the accumulated mean.
            # Equivalent to treating each new observation as contributing
            # adapt_rate / (adapt_rate + existing_weight) fractionally.
            old_mean = self._welford_mean
            old_m2 = self._welford_m2
            old_n = self._welford_n

            # Standard Welford step (unweighted count)
            new_n = old_n + 1
            delta = smoothed - old_mean
            new_mean = old_mean + (self.adapt_rate * delta) / max(new_n, 1)
            delta2 = smoothed - new_mean
            new_m2 = old_m2 + self.adapt_rate * delta * delta2

            self._welford_n = new_n
            self._welford_mean = new_mean
            self._welford_m2 = max(new_m2, 0.0)
        else:
            # Frozen prior: no state update; z-score is purely against prior
            pass

        # --- Compute std and z-score -----------------------------------------
        total_n = self._welford_n
        if total_n >= 2:
            std = max(
                float(np.sqrt(self._welford_m2 / (total_n - 1))),
                1e-6,
            )
        else:
            std = max(self._prior_std, 1e-6)

        self._zscore = (smoothed - self._welford_mean) / std

        # --- Reward criterion ------------------------------------------------
        if self.direction == "up":
            crossed = self._zscore > self.zscore_threshold
        else:
            crossed = self._zscore < -self.zscore_threshold

        magnitude = abs(self._zscore) if crossed else 0.0
        return crossed, magnitude

    def reset(self) -> None:
        """Reset to the prior statistics without re-reading the file.

        Restores the Welford accumulators to the values computed from the
        prior data during ``__init__``, clears the smoothed state, and
        resets the evaluation counter.  All constructor parameters are
        preserved.
        """
        self._n_evaluated = 0
        self._smoothed = None
        self._zscore = 0.0

        self._welford_n = self._n_prior
        self._welford_mean = self._prior_mean
        self._welford_m2 = self._prior_std ** 2 * (self._n_prior - 1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def prior_mean(self) -> float:
        """Mean of the prior session data (read-only)."""
        return self._prior_mean

    @property
    def prior_std(self) -> float:
        """Standard deviation of the prior session data (read-only)."""
        return self._prior_std

    @property
    def n_prior(self) -> int:
        """Number of samples in the prior session data (read-only)."""
        return self._n_prior

    @property
    def n_evaluated(self) -> int:
        """Number of values evaluated since init or last :meth:`reset`."""
        return self._n_evaluated

    @property
    def zscore(self) -> float:
        """Z-score computed during the most recent :meth:`evaluate` call.

        Returns ``0.0`` before any evaluation.
        """
        return self._zscore

    @property
    def mean_(self) -> float:
        """Current running mean (prior-seeded).

        Before any adaptations this equals :attr:`prior_mean`.
        """
        return self._welford_mean

    @property
    def std_(self) -> float:
        """Current running standard deviation (prior-seeded).

        Before any adaptations this equals :attr:`prior_std`.
        """
        total_n = self._welford_n
        if total_n >= 2:
            return max(
                float(np.sqrt(self._welford_m2 / (total_n - 1))),
                1e-6,
            )
        return max(self._prior_std, 1e-6)

    def __repr__(self) -> str:
        return (
            f"TransferProtocol("
            f"fname={str(self.fname)!r}, "
            f"modality={self.modality!r}, "
            f"direction={self.direction!r}, "
            f"prior_mean={self._prior_mean:.4g}, "
            f"prior_std={self._prior_std:.4g}, "
            f"n_prior={self._n_prior}, "
            f"zscore={self._zscore:.4g}, "
            f"mean_={self.mean_:.4g}, "
            f"std_={self.std_:.4g}, "
            f"n_evaluated={self._n_evaluated})"
        )
