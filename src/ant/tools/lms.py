"""Adaptive LMS filter class for real-time artifact removal.

Classes
-------
AdaptiveLMSFilter
    Widrow-Hoff LMS adaptive filter with persistent weight state.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class AdaptiveLMSFilter:
    """Adaptive LMS filter for real-time EOG / ECG artifact removal.

    Implements the Widrow–Hoff Least Mean Squares algorithm
    :footcite:p:`Widrow1960` to regress out a reference artifact channel
    (e.g. a frontal EOG electrode or an ECG lead) from all other M/EEG
    channels.  Filter weights are updated online — no calibration baseline
    is required.

    Parameters
    ----------
    ref_ch_idx : int, default 0
        Index of the reference (artifact) channel in the data array.
    n_taps : int, default 5
        Number of tapped-delay filter coefficients (filter order).
        Larger values capture longer temporal autocorrelation in the
        artifact but increase computation.
    mu : float, default 0.01
        LMS step size (learning rate).  Must satisfy
        :math:`0 < \\mu < 2 / (n_{\\mathrm{taps}} \\cdot P_{\\mathrm{ref}})`,
        where :math:`P_{\\mathrm{ref}}` is the reference-channel power.
        Values between 0.001 and 0.05 are typically stable.

    Attributes
    ----------
    weights_ : ndarray, shape (n_channels, n_taps) or None
        Current filter weights.  ``None`` until the first call to
        :meth:`transform`.  Weights persist across successive chunk
        calls to enable online adaptation.

    Raises
    ------
    ValueError
        If ``mu <= 0`` or ``n_taps < 1``.

    Notes
    -----
    No :meth:`fit` call is required — the filter adapts from the first
    sample.  Use ``artifact_correction="lms"`` in :class:`~ant.NFRealtime`
    to enable during recording, or call :meth:`transform` directly.

    See :ref:`denoising-lms` for the full mathematical background.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    Apply to a single data chunk:

    >>> filt = AdaptiveLMSFilter(ref_ch_idx=0, mu=0.01)
    >>> clean = filt.transform(data)   # data: (n_channels, n_times)

    Maintain weight state across consecutive real-time chunks:

    >>> filt = AdaptiveLMSFilter()
    >>> for chunk in stream:
    ...     clean_chunk = filt.transform(chunk)
    """

    def __init__(
        self,
        ref_ch_idx: int = 0,
        n_taps: int = 5,
        mu: float = 0.01,
    ) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        if n_taps < 1:
            raise ValueError(f"n_taps must be >= 1, got {n_taps}")
        self.ref_ch_idx = int(ref_ch_idx)
        self.n_taps = int(n_taps)
        self.mu = float(mu)
        self.weights_: Optional[np.ndarray] = None

    def fit(self, raw_info=None, **kwargs) -> "AdaptiveLMSFilter":
        """No-op: LMS requires no calibration baseline.

        Provided for API consistency with other artifact correctors.
        Returns ``self``.
        """
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the adaptive LMS filter to a data chunk.

        The reference channel (``ref_ch_idx``) drives artifact cancellation
        for all other channels.  Filter weights are updated in-place so that
        successive calls continue adapting rather than restarting from zero.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Raw M/EEG data chunk.

        Returns
        -------
        cleaned : ndarray, shape (n_channels, n_times)
            Artifact-attenuated data (reference channel is passed through
            unchanged).
        """
        n_channels, n_times = data.shape

        if self.weights_ is None:
            self.weights_ = np.zeros((n_channels, self.n_taps))

        ref = data[self.ref_ch_idx]

        # Tapped-delay matrix: column k = ref delayed by k samples
        X = np.zeros((n_times, self.n_taps))
        for k in range(self.n_taps):
            X[k:, k] = ref[:n_times - k]

        cleaned = data.copy()
        W = self.weights_

        for t in range(n_times):
            x = X[t]
            y = W @ x
            e = data[:, t] - y
            cleaned[:, t] = e
            W += self.mu * np.outer(e, x)

        # Reference channel passes through unchanged
        cleaned[self.ref_ch_idx] = data[self.ref_ch_idx]
        self.weights_ = W
        return cleaned

    def reset(self) -> None:
        """Reset filter weights to zero (restart adaptation)."""
        self.weights_ = None

    def __repr__(self) -> str:
        adapted = self.weights_ is not None
        return (
            f"AdaptiveLMSFilter(ref_ch_idx={self.ref_ch_idx}, "
            f"n_taps={self.n_taps}, mu={self.mu}, adapted={adapted})"
        )
