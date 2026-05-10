"""GEneralized Decomposition for Artifact Identification (GEDAI).

Removes artifacts from EEG/MEG by solving a generalized eigenvalue problem
that finds spatial filters maximising signal in a target band relative to
broadband activity.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.signal import sosfiltfilt

from ant.tools.tools import butter_bandpass


class GEDAIDenoiser:
    """Artifact removal via Generalised Eigendecomposition-based Artifact Identification (GEDAI).

    Solves the generalised eigenvalue problem (GEP):

    .. math::

        \\mathbf{C}\\,\\mathbf{w} = \\lambda\\,\\mathbf{R}\\,\\mathbf{w}

    where :math:`\\mathbf{C}` is the signal covariance and :math:`\\mathbf{R}`
    is a reference covariance.  Two modes are supported:

    * **Leadfield mode** (recommended, true GEDAI):
      :math:`\\mathbf{R} = \\mathbf{L}\\mathbf{L}^\\top`, where
      :math:`\\mathbf{L}` is the EEG forward/leadfield gain matrix.
      Components with *large* :math:`\\lambda` are well-explained by the
      brain's theoretical source model and are kept; components with *small*
      :math:`\\lambda` are not leadfield-aligned and are treated as artifacts.
      Use :meth:`fit_from_leadfield` to fit in this mode.

    * **Band-filter mode** (Cohen-style GED):
      :math:`\\mathbf{C} = \\mathbf{R}_{\\mathrm{band}}` and
      :math:`\\mathbf{R} = \\mathbf{R}_{\\mathrm{broad}}` (broadband EEG
      covariance, Tikhonov-regularised).  Components with large :math:`\\lambda`
      maximise the target-band-to-broadband ratio.
      Use :meth:`fit` or :meth:`fit_from_raw` for this mode.

    In both modes the unmixing matrix :math:`\\mathbf{W}` (spatial filters)
    and activation patterns :math:`\\mathbf{A} = (\\mathbf{W}^\\top)^+` are
    stored after fitting.  Denoising zeroes the selected artifact columns of
    :math:`\\mathbf{A}` and reconstructs clean sensor data as
    :math:`\\hat{\\mathbf{x}} = \\mathbf{A}_{\\mathrm{clean}}\\,\\mathbf{W}^\\top\\mathbf{x}`.

    Parameters
    ----------
    n_channels : int
        Number of EEG/MEG channels.
    shrinkage : float, default 0.01
        Tikhonov regularisation strength applied to the reference covariance
        before solving the GEP.  Prevents ill-conditioning when the
        covariance matrix is rank-deficient.

    References
    ----------
    Ros, T., Férat, V., Huang, Y., et al. (2025). Return of the GEDAI:
    Unsupervised EEG Denoising based on Leadfield Filtering. *bioRxiv*.
    https://doi.org/10.1101/2025.10.04.680449

    Cohen, M. X. (2022). A tutorial on generalized eigendecomposition for
    denoising, contrast enhancement, and dimension reduction in multichannel
    electrophysiology. *NeuroImage*, 247, 118809.
    """

    def __init__(self, n_channels: int, shrinkage: float = 0.01) -> None:
        self.n_channels = n_channels
        self.shrinkage = shrinkage

        self._W: np.ndarray | None = None        # spatial filters, shape (n_ch, n_ch)
        self._A: np.ndarray | None = None        # activation patterns = pinv(W.T)
        self._eigenvalues: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data_broadband: np.ndarray,
        data_band: np.ndarray,
    ) -> "GEDAIDenoiser":
        """Estimate spatial filters from baseline data.

        Parameters
        ----------
        data_broadband : ndarray, shape (n_channels, n_samples)
            Broadband (or noise-reference) baseline data.
        data_band : ndarray, shape (n_channels, n_samples)
            Band-filtered baseline data (the signal of interest).

        Returns
        -------
        self
        """
        if data_broadband.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data_broadband.shape[0]}"
            )

        n_samples = data_broadband.shape[1]

        Xb = data_broadband - data_broadband.mean(axis=1, keepdims=True)
        Xs = data_band - data_band.mean(axis=1, keepdims=True)

        R_broad = (Xb @ Xb.T) / n_samples
        R_band  = (Xs @ Xs.T) / n_samples

        # Regularise broadband covariance
        reg = self.shrinkage * np.trace(R_broad) / self.n_channels
        R_broad += reg * np.eye(self.n_channels)

        eigenvalues, W = linalg.eigh(R_band, R_broad)

        # Sort descending: largest λ = most band-specific
        order = np.argsort(eigenvalues)[::-1]
        self._eigenvalues = eigenvalues[order]
        self._W = W[:, order]
        self._A = np.linalg.pinv(self._W.T)

        return self

    def fit_from_leadfield(
        self,
        data: np.ndarray,
        leadfield: np.ndarray,
    ) -> "GEDAIDenoiser":
        """Fit using the forward/leadfield matrix as the reference covariance.

        This is the **true GEDAI** mode described in Ros et al. (2025).
        The reference covariance is constructed as
        :math:`\\mathbf{R} = \\mathbf{L}\\mathbf{L}^\\top` where
        :math:`\\mathbf{L}` is the leadfield gain matrix.  Components
        whose eigenvalues are large are well-aligned with the theoretical
        brain source model and are treated as signal; components with small
        eigenvalues are artifact candidates.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Broadband baseline EEG/MEG recording.
        leadfield : ndarray, shape (n_channels, n_sources)
            Forward solution gain matrix :math:`\\mathbf{L}` — the
            ``fwd['sol']['data']`` array from an MNE forward solution.

        Returns
        -------
        self
        """
        if data.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[0]}"
            )
        if leadfield.shape[0] != self.n_channels:
            raise ValueError(
                f"Leadfield row count ({leadfield.shape[0]}) must equal n_channels ({self.n_channels})"
            )

        n_samples = data.shape[1]
        Xb = data - data.mean(axis=1, keepdims=True)
        C = (Xb @ Xb.T) / n_samples

        # Leadfield-based reference: R = L @ L.T (normalised)
        R_lead = leadfield @ leadfield.T
        R_lead /= np.trace(R_lead) / self.n_channels  # scale to unit average power

        # Tikhonov regularisation
        reg = self.shrinkage * np.trace(R_lead) / self.n_channels
        R_lead += reg * np.eye(self.n_channels)

        eigenvalues, W = linalg.eigh(C, R_lead)

        # Descending order: largest λ = most leadfield-aligned = brain signal
        order = np.argsort(eigenvalues)[::-1]
        self._eigenvalues = eigenvalues[order]
        self._W = W[:, order]
        self._A = np.linalg.pinv(self._W.T)

        return self

    def fit_from_raw(
        self,
        data: np.ndarray,
        sfreq: float,
        band: tuple[float, float],
    ) -> "GEDAIDenoiser":
        """Convenience wrapper: bandpass-filter ``data`` internally then call ``fit``.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Baseline recording (broadband).
        sfreq : float
            Sampling frequency in Hz.
        band : (low, high)
            Target frequency band in Hz.

        Returns
        -------
        self
        """
        sos = butter_bandpass(band[0], band[1], sfreq, order=5)
        data_band = sosfiltfilt(sos, data)
        return self.fit(data, data_band)

    # ------------------------------------------------------------------
    # Transform / reconstruct
    # ------------------------------------------------------------------

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Project data into component space.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)

        Returns
        -------
        components : ndarray, shape (n_channels, n_samples)
        """
        self._check_fitted()
        return self._W.T @ data

    def inverse_transform(self, components: np.ndarray) -> np.ndarray:
        """Reconstruct sensor-space data from components.

        Parameters
        ----------
        components : ndarray, shape (n_channels, n_samples)

        Returns
        -------
        data : ndarray, shape (n_channels, n_samples)
        """
        self._check_fitted()
        return self._A @ components

    # ------------------------------------------------------------------
    # Artifact identification and removal
    # ------------------------------------------------------------------

    def find_artifact_components(
        self,
        template_map: np.ndarray,
        threshold: float = 0.7,
    ) -> tuple[list[int], np.ndarray]:
        """Identify artifact components by correlation with a spatial template.

        Parameters
        ----------
        template_map : ndarray, shape (n_channels,)
            Known topography of the artifact (e.g., blink, ECG).
        threshold : float
            Absolute-correlation threshold.

        Returns
        -------
        artifact_idx : list of int
        corrs : ndarray, shape (n_channels,)
        """
        self._check_fitted()
        corrs = np.array(
            [
                np.corrcoef(self._A[:, i], template_map)[0, 1]
                for i in range(self._A.shape[1])
            ]
        )
        artifact_idx = np.where(np.abs(corrs) > threshold)[0].tolist()
        return artifact_idx, corrs

    def find_noise_components(self, n_noise: int = 1) -> list[int]:
        """Return indices of the ``n_noise`` components with smallest eigenvalues.

        These components capture the least band-specific activity and are
        candidates for broadband noise.

        Parameters
        ----------
        n_noise : int
            Number of components to flag.

        Returns
        -------
        list of int
        """
        self._check_fitted()
        return list(range(len(self._eigenvalues) - n_noise, len(self._eigenvalues)))

    def denoise(
        self,
        data: np.ndarray,
        artifact_idx: list[int],
    ) -> np.ndarray:
        """Suppress artifact components and reconstruct sensor data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
        artifact_idx : list of int
            Component indices to zero out.

        Returns
        -------
        data_clean : ndarray, shape (n_channels, n_samples)
        """
        components = self.transform(data)
        if artifact_idx:
            components[artifact_idx, :] = 0.0
        return self.inverse_transform(components)

    def update_and_denoise(
        self,
        data: np.ndarray,
        template_map: np.ndarray,
        threshold: float = 0.7,
    ) -> np.ndarray:
        """Identify blink/artifact components then denoise in one call.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
        template_map : ndarray, shape (n_channels,)
        threshold : float

        Returns
        -------
        data_clean : ndarray, shape (n_channels, n_samples)
        """
        artifact_idx, _ = self.find_artifact_components(template_map, threshold)
        return self.denoise(data, artifact_idx)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self) -> np.ndarray:
        """Sorted eigenvalues (descending) from the GED."""
        self._check_fitted()
        return self._eigenvalues

    @property
    def spatial_filters(self) -> np.ndarray:
        """Spatial filters W, shape (n_channels, n_channels)."""
        self._check_fitted()
        return self._W

    @property
    def activation_patterns(self) -> np.ndarray:
        """Activation patterns A = pinv(W.T), shape (n_channels, n_channels)."""
        self._check_fitted()
        return self._A

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._W is None:
            raise RuntimeError("Call fit() or fit_from_raw() before using this method.")
