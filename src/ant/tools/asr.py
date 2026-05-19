"""Artifact Subspace Reconstruction (ASR) for real-time EEG denoising.

Classes
-------
ASRDenoiser
    Calibrate-once, apply-online artifact suppressor for streaming EEG.

References
----------
Mullen, T. R., et al. (2015). Real-time neuroimaging and cognitive monitoring
using wearable dry EEG. *IEEE Trans. Biomed. Eng.*, 62(11), 2553–2567.
https://doi.org/10.1109/TBME.2015.2481482

de Cheveigné, A., & Arzounian, D. (2018). Robust detrending, referencing,
outlier detection, and inpainting for multichannel data. *NeuroImage*, 172,
903–912. https://doi.org/10.1016/j.neuroimage.2018.01.035
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg


class ASRDenoiser:
    """Artifact Subspace Reconstruction (ASR) for streaming EEG.

    ASR :footcite:p:`mullen2015real,de2018robust` separates *clean*
    from *artifactual* activity by learning the statistics of a resting-state
    baseline and projecting out components that deviate beyond a threshold.

    **Calibration** (:meth:`fit`)
        Segment the baseline into overlapping windows of length
        :math:`T_{\\mathrm{win}}`, compute the sample covariance per window,
        discard the fraction ``max_dropout_fraction`` with the highest total
        power (likely artifact), and eigendecompose the mean clean covariance:

        .. math::

            \\mathbf{C}_0 = \\mathbf{U}\\,\\mathrm{diag}(d_1,\\dots,d_p)\\,\\mathbf{U}^\\top.

        Per-component RMS thresholds are then set as:

        .. math::

            T_k = \\sigma \\cdot \\sqrt{d_k}, \\qquad k = 1,\\dots,p,

        where :math:`\\sigma` = ``cutoff`` (default 5).

    **Cleaning** (:meth:`transform`)
        For each incoming window :math:`\\mathbf{X} \\in \\mathbb{R}^{p \\times n}`:

        1. Project to calibration component space:
           :math:`\\mathbf{Z} = \\mathbf{U}^\\top\\mathbf{X}`.
        2. Compute per-component RMS:
           :math:`r_k = \\sqrt{\\operatorname{mean}(Z_k^2)}`.
        3. Zero components that exceed the threshold
           (:math:`r_k > T_k`).
        4. Back-project:
           :math:`\\hat{\\mathbf{X}} = \\mathbf{U}\\mathbf{Z}`.

    Parameters
    ----------
    cutoff : float, default 5.0
        Rejection threshold in multiples of the clean-data per-component RMS.
        Lower values (3–4) are more aggressive; higher values (8–10) are more
        conservative.  Mullen et al. (2015) recommend 5.
    max_dropout_fraction : float, default 0.1
        Fraction of calibration windows with the highest total power to
        discard before estimating clean statistics.  ``0.1`` retains the
        90 % "cleanest" windows.
    window_overlap : float, default 0.5
        Fractional overlap between consecutive calibration windows (0–1).
        ``0.5`` halves the effective hop size, doubling the number of windows.

    Attributes
    ----------
    cutoff : float
    max_dropout_fraction : float
    window_overlap : float

    Raises
    ------
    ValueError
        If constructor arguments are out of valid range.
    RuntimeError
        If :meth:`transform` is called before :meth:`fit`.

    See Also
    --------
    ant.tools.GEDAIDenoiser : GED-based spatial filter (requires forward model).
    ant.tools.ORICA : Online recursive ICA for adaptive unmixing.
    ant.NFRealtime.fit_asr : Fit from a live session baseline.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> asr = ASRDenoiser(cutoff=5.0)
    >>> asr.fit(baseline_data, sfreq=250.0)
    >>> clean = asr.transform(noisy_window)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        max_dropout_fraction: float = 0.1,
        window_overlap: float = 0.5,
    ) -> None:
        if cutoff <= 0:
            raise ValueError("`cutoff` must be a positive float.")
        if not 0.0 <= max_dropout_fraction < 1.0:
            raise ValueError("`max_dropout_fraction` must be in [0, 1).")
        if not 0.0 <= window_overlap < 1.0:
            raise ValueError("`window_overlap` must be in [0, 1).")

        self.cutoff = cutoff
        self.max_dropout_fraction = max_dropout_fraction
        self.window_overlap = window_overlap

        self._U: Optional[np.ndarray] = None   # calibration eigenvectors
        self._T: Optional[np.ndarray] = None   # per-component RMS thresholds
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        sfreq: float,
        window_len: float = 1.0,
    ) -> "ASRDenoiser":
        """Calibrate from a clean resting-state baseline.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Baseline M/EEG recording.  Eyes-closed resting state is
            recommended to minimise ocular and muscular contamination.
        sfreq : float
            Sampling frequency in Hz.
        window_len : float, default 1.0
            Calibration window length in seconds.  Shorter windows yield
            more estimates with higher variance; 0.5–1.0 s is typical.

        Returns
        -------
        self : ASRDenoiser

        Raises
        ------
        ValueError
            If fewer samples are available than one window, or all windows
            are discarded as outliers.
        """
        n_ch, n_samp = data.shape
        win_samp = int(window_len * sfreq)
        if win_samp < 2:
            raise ValueError(
                f"window_len={window_len} s at {sfreq} Hz → {win_samp} samples: "
                "too short for covariance estimation."
            )
        if n_samp < win_samp:
            raise ValueError(
                f"Data has {n_samp} samples but window_len needs {win_samp}."
            )

        hop = max(int(win_samp * (1.0 - self.window_overlap)), 1)
        X = data - data.mean(axis=1, keepdims=True)

        # Per-window sample covariance matrices
        covs = np.stack(
            [(X[:, s:s + win_samp] @ X[:, s:s + win_samp].T) / win_samp
             for s in range(0, n_samp - win_samp + 1, hop)]
        )

        # Discard windows with highest total power (likely artifact)
        if self.max_dropout_fraction > 0.0 and len(covs) > 1:
            power = np.array([np.trace(c) for c in covs])
            keep = power <= np.quantile(power, 1.0 - self.max_dropout_fraction)
            clean_covs = covs[keep] if keep.any() else covs
        else:
            clean_covs = covs

        # Mean clean covariance + small diagonal regularisation
        C0 = clean_covs.mean(axis=0)
        C0 += np.eye(n_ch) * (1e-8 * np.trace(C0) / n_ch)

        d, U = linalg.eigh(C0)
        order = np.argsort(d)[::-1]   # descending eigenvalue order
        self._U = U[:, order]
        self._T = self.cutoff * np.sqrt(np.maximum(d[order], 0.0))
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Remove artifact-dominated components from a data window.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Raw EEG window to clean.

        Returns
        -------
        clean : ndarray, shape (n_channels, n_samples)
            Cleaned data.  Returned unchanged if no component exceeds its
            threshold (no-op when data is already clean).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._check_fitted()
        mean = data.mean(axis=1, keepdims=True)
        Z = self._U.T @ (data - mean)           # project to component space
        rms = np.sqrt(np.mean(Z ** 2, axis=1))  # per-component RMS
        bad = rms > self._T
        if not bad.any():
            return data
        Z[bad] = 0.0
        return self._U @ Z + mean               # back-project + restore mean

    # ------------------------------------------------------------------
    # Properties & helpers
    # ------------------------------------------------------------------

    @property
    def thresholds(self) -> np.ndarray:
        """Per-component RMS thresholds, shape (n_channels,)."""
        self._check_fitted()
        return self._T

    @property
    def eigenvectors(self) -> np.ndarray:
        """Calibration eigenvectors :math:`\\mathbf{U}`, shape (n_ch, n_ch)."""
        self._check_fitted()
        return self._U

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before using this method.")

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "not fitted"
        return (
            f"ASRDenoiser(cutoff={self.cutoff}, "
            f"max_dropout_fraction={self.max_dropout_fraction}, "
            f"state={state!r})"
        )
