"""Riemannian Potato — online artifact detection for streaming M/EEG.

Detects artifactual data windows by checking whether the covariance matrix
of each incoming chunk is too far from the geometric mean of clean data on
the symmetric positive-definite (SPD) matrix manifold.

Classes
-------
RiemannianPotatoDetector
    Online Riemannian artifact detector with calibration-based baseline.

References
----------
Barachant, A. & Congedo, M. (2014). A Plug&Play P300 BCI Using Information
Geometry. *arXiv:1409.0107*.

Barthélemy, Q., Mayaud, L., Ojeda, D., & Congedo, M. (2019). The Riemannian
potato field: a tool for online signal quality index of EEG.
*IRBM*, 40(1), 15–26. https://doi.org/10.1016/j.irbm.2018.09.001
"""
from __future__ import annotations

from typing import Union

import numpy as np

from ant._logging import logger


class RiemannianPotatoDetector:
    """Online EEG/MEG artifact detector based on the Riemannian Potato.

    The *Riemannian Potato* detects artifactual epochs by measuring how far
    each incoming covariance matrix is from the geometric mean of a clean
    calibration set — distance is computed in the Riemannian metric on the
    manifold of symmetric positive-definite (SPD) matrices.  A window is
    declared artifactual when this distance (expressed as a z-score) exceeds
    ``threshold``.

    Unlike threshold-based or ICA-based methods, this approach requires no
    reference channel and is robust to gradual amplitude drift (because the
    geometric mean adapts to the signal statistics during calibration rather
    than using a fixed absolute threshold).

    Parameters
    ----------
    threshold : float, default 3.0
        Z-score cutoff (Riemannian distance units).  Windows with a
        distance z-score above this value are flagged as artifacts.
        Typical range: 2.5–4.0.  Lower = more aggressive rejection.
    estimator : str, default "oas"
        Covariance estimator passed to
        :class:`pyriemann.estimation.Covariances`.
        ``"oas"`` (Oracle Approximating Shrinkage) is robust at low
        sample-to-channel ratios; ``"scm"`` gives the raw sample covariance.
    metric : str, default "riemann"
        Riemannian metric used for the potato.  ``"riemann"`` (affine-invariant)
        is the canonical choice; ``"logeuclid"`` is faster but less robust.
    verbose : bool | str | None, default None
        Verbosity level.  See :func:`~ant._logging.set_log_level`.

    Attributes
    ----------
    is_fitted_ : bool
        ``True`` after :meth:`fit` has been called successfully.
    n_channels_ : int
        Number of channels seen during :meth:`fit`.
    n_calibration_windows_ : int
        Number of clean windows used to fit the potato.

    Raises
    ------
    ImportError
        If ``pyriemann`` is not installed.
    RuntimeError
        If :meth:`detect` is called before :meth:`fit`.

    Examples
    --------
    Calibrate on 60 s of clean data, then detect online::

        detector = RiemannianPotatoDetector(threshold=3.0)
        detector.fit(clean_windows)            # shape (n_windows, n_ch, n_samples)

        while streaming:
            window = stream.get_data(1.0)      # shape (n_ch, n_samples)
            is_clean, z_score = detector.detect(window)
            if is_clean:
                process_for_nf(window)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        threshold: float = 3.0,
        estimator: str = "oas",
        metric: str = "riemann",
        verbose: Union[bool, str, None] = None,
    ) -> None:
        self._check_pyriemann()
        from ant._logging import set_log_level
        set_log_level(verbose)

        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")

        self.threshold = threshold
        self.estimator = estimator
        self.metric = metric

        self._potato = None
        self._cov_estimator = None
        self.is_fitted_: bool = False
        self.n_channels_: int = 0
        self.n_calibration_windows_: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_pyriemann() -> None:
        try:
            import pyriemann  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RiemannianPotatoDetector requires pyriemann.\n"
                "Install it with:  pip install pyriemann"
            ) from exc

    def _make_estimators(self) -> None:
        from pyriemann.clustering import Potato
        from pyriemann.estimation import Covariances
        self._cov_estimator = Covariances(estimator=self.estimator)
        self._potato = Potato(
            metric=self.metric,
            threshold=self.threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, windows: np.ndarray) -> "RiemannianPotatoDetector":
        """Calibrate the potato on clean data windows.

        Computes covariance matrices for each window and fits the geometric
        mean and variance used as the potato centre and spread.

        Parameters
        ----------
        windows : ndarray, shape (n_windows, n_channels, n_samples)
            Calibration data.  Should contain **clean** (artifact-free)
            segments.  At least 10 windows are recommended; 30–60 is typical
            for a 1-second window length at standard EEG sampling rates.

        Returns
        -------
        self : RiemannianPotatoDetector

        Raises
        ------
        ValueError
            If ``windows.ndim != 3`` or fewer than 2 windows are provided.
        """
        if windows.ndim != 3:
            raise ValueError(
                f"windows must be 3-D (n_windows, n_ch, n_samples), "
                f"got shape {windows.shape}"
            )
        if len(windows) < 2:
            raise ValueError("At least 2 calibration windows are required.")

        self._make_estimators()
        covs = self._cov_estimator.fit_transform(windows)
        self._potato.fit(covs)

        self.is_fitted_ = True
        self.n_channels_ = windows.shape[1]
        self.n_calibration_windows_ = len(windows)
        logger.info(
            "RiemannianPotatoDetector fitted on %d windows (%d channels).",
            self.n_calibration_windows_,
            self.n_channels_,
        )
        return self

    def detect(self, window: np.ndarray) -> tuple[bool, float]:
        """Test a single window for artifacts.

        Parameters
        ----------
        window : ndarray, shape (n_channels, n_samples)
            One data window to evaluate.

        Returns
        -------
        is_clean : bool
            ``True`` when the window is inside the potato (clean);
            ``False`` when it is outside (potential artifact).
        z_score : float
            Riemannian distance z-score.  Higher = more anomalous.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If the channel count does not match the calibration set.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Call fit() with clean calibration data before detect()."
            )
        if window.ndim != 2 or window.shape[0] != self.n_channels_:
            raise ValueError(
                f"Expected window shape ({self.n_channels_}, n_samples), "
                f"got {window.shape}."
            )

        cov = self._cov_estimator.transform(window[np.newaxis])
        label = float(self._potato.predict(cov)[0])
        z_score = float(self._potato.transform(cov)[0])

        is_clean = (label == 1.0)
        return is_clean, z_score

    def __repr__(self) -> str:
        status = (
            f"fitted on {self.n_calibration_windows_} windows, "
            f"{self.n_channels_} channels"
            if self.is_fitted_
            else "not fitted"
        )
        return (
            f"RiemannianPotatoDetector("
            f"threshold={self.threshold}, "
            f"metric={self.metric!r}, "
            f"{status})"
        )
