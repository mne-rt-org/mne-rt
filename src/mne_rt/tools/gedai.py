"""GEneralized Decomposition for Artifact Identification (GEDAI).

Removes artifacts from EEG/MEG by solving a generalized eigenvalue problem
that finds spatial filters maximising signal in a target band relative to
broadband activity.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.signal import sosfiltfilt

from mne_rt.tools.tools import butter_bandpass


class GEDAIDenoiser:
    """Artifact removal via Generalised Eigendecomposition-based Artifact Identification (GEDAI).

    Solves the generalised eigenvalue problem (GEP) ``C @ w = lambda * R @ w``
    where ``C`` is the signal covariance and ``R`` is a reference covariance.
    Two modes are supported:

    * **Leadfield mode** (recommended, true GEDAI):
      ``R = L @ L.T``, where ``L`` is the EEG forward/leadfield gain matrix.
      Because ``lambda = (w.T @ C @ w) / (w.T @ R @ w)``, components with
      *large* ``lambda`` carry sensor power that the brain's theoretical
      source model cannot explain and are treated as artifacts; components
      with *small* ``lambda`` are leadfield-aligned and are kept as brain
      signal. Use :meth:`fit_from_leadfield` to fit in this mode.

    * **Band-filter mode** (Cohen-style GED):
      ``C = R_band`` and ``R = R_broad`` (broadband EEG covariance,
      Tikhonov-regularised). Components with large ``lambda`` maximise the
      target-band-to-broadband ratio, while components with small ``lambda``
      capture broadband noise. Use :meth:`fit` or :meth:`fit_from_raw` for
      this mode.

    In both modes the unmixing matrix ``W`` (spatial filters) and activation
    patterns ``A = pinv(W.T)`` are stored after fitting. Denoising zeroes the
    selected artifact columns of ``A`` and reconstructs clean sensor data as
    ``x_hat = A_clean @ W.T @ x``.

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

        self._W: np.ndarray | None = None  # spatial filters, shape (n_ch, n_ch)
        self._A: np.ndarray | None = None  # activation patterns = pinv(W.T)
        self._eigenvalues: np.ndarray | None = None
        self._reference_is_leadfield: bool = False
        self._eigenvalue_threshold: float = np.inf
        self._sensitivity: float = 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data_broadband: np.ndarray,
        data_band: np.ndarray,
    ) -> "GEDAIDenoiser":
        """Estimate spatial filters from baseline data (band-filter mode).

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
            raise ValueError(f"Expected {self.n_channels} channels, got {data_broadband.shape[0]}")

        n_samples = data_broadband.shape[1]

        Xb = data_broadband - data_broadband.mean(axis=1, keepdims=True)
        Xs = data_band - data_band.mean(axis=1, keepdims=True)

        R_broad = (Xb @ Xb.T) / n_samples
        R_band = (Xs @ Xs.T) / n_samples

        # Regularise broadband covariance
        reg = self.shrinkage * np.trace(R_broad) / self.n_channels
        R_broad += reg * np.eye(self.n_channels)

        eigenvalues, W = linalg.eigh(R_band, R_broad)

        # Sort descending: largest λ = most band-specific
        order = np.argsort(eigenvalues)[::-1]
        self._eigenvalues = eigenvalues[order]
        self._W = W[:, order]
        self._A = np.linalg.pinv(self._W.T)
        self._reference_is_leadfield = False
        self._eigenvalue_threshold = np.inf
        self._sensitivity = 0.0

        return self

    def fit_from_leadfield(
        self,
        data: np.ndarray,
        leadfield: np.ndarray,
        sfreq: float = 250.0,
        noise_multiplier: float = 3.0,
    ) -> "GEDAIDenoiser":
        """Fit using the forward/leadfield matrix as the reference covariance.

        This is the **true GEDAI** mode described in Ros et al. (2025).
        The reference covariance is constructed as ``R = L @ L.T`` where
        ``L`` is the leadfield gain matrix. Because the GEP solves
        ``C @ w = lambda * R @ w``, components whose eigenvalues are *large*
        have excess sensor variance relative to the theoretical brain source
        model and are treated as artifacts; components with *small*
        eigenvalues are leadfield-aligned and are kept as brain signal.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Broadband baseline EEG/MEG recording.
        leadfield : ndarray, shape (n_channels, n_sources)
            Forward solution gain matrix ``L`` — the
            ``fwd['sol']['data']`` array from an MNE forward solution.
        sfreq : float, default 250.0
            Sampling frequency in Hz (used to size 1-second SENSAI epochs
            when estimating the automatic eigenvalue threshold if ``gedai``
            is installed).
        noise_multiplier : float, default 3.0
            Noise penalty weight passed to SENSAI optimization when available.

        Returns
        -------
        self
        """
        if data.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {data.shape[0]}")
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

        # Descending order: largest λ = highest sensor-to-leadfield ratio = artifact
        order = np.argsort(eigenvalues)[::-1]
        self._eigenvalues = eigenvalues[order]
        self._W = W[:, order]
        self._A = np.linalg.pinv(self._W.T)
        self._reference_is_leadfield = True
        self._eigenvalue_threshold = np.inf
        self._sensitivity = 0.0

        # Estimate automatic SENSAI eigenvalue cutoff when gedai is available
        try:
            from gedai.sensai.sensai import (
                _precompute_gevd,
                _sensai_optimize,
                _sensai_to_eigen,
            )

            ep_len = max(2, min(int(round(sfreq)), n_samples // 2))
            step = max(1, ep_len // 2)
            if n_samples >= ep_len * 2 and float(np.mean(Xb**2)) > 0:
                epochs = np.stack(
                    [Xb[:, s : s + ep_len] for s in range(0, n_samples - ep_len + 1, step)]
                )
                if len(epochs) >= 2:
                    vals, vecs = _precompute_gevd(epochs, R_lead, engine="numpy")
                    _, sens, _ = _sensai_optimize(
                        epochs,
                        R_lead,
                        n_pc=min(3, self.n_channels),
                        noise_multiplier=noise_multiplier,
                        epochs_eigenvalues=vals,
                        bounds=(-4.0, 12.0),
                        all_eval=vals,
                        all_evec=vecs,
                        percentile=98,
                        signal_type="eeg",
                        engine="numpy",
                    )
                    self._sensitivity = float(sens)
                    self._eigenvalue_threshold = float(
                        _sensai_to_eigen(
                            self._sensitivity, np.abs(self._eigenvalues), percentile=98
                        )
                    )
        except Exception:
            pass

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
            [np.corrcoef(self._A[:, i], template_map)[0, 1] for i in range(self._A.shape[1])]
        )
        artifact_idx = np.where(np.abs(corrs) > threshold)[0].tolist()
        return artifact_idx, corrs

    def find_noise_components(self, n_noise: int | str = 1) -> list[int]:
        """Return indices of artifact/noise components according to the fitted mode.

        In **leadfield mode** (:meth:`fit_from_leadfield`), eigenvalues measure
        the ratio of observed sensor variance to theoretical brain variance
        (``C @ w = lambda * R_lead @ w``), so components with the *largest*
        eigenvalues are artifact candidates. Passing ``n_noise="auto"`` selects
        all components whose eigenvalue meets or exceeds the SENSAI threshold
        (:attr:`eigenvalue_threshold`); passing an integer returns the top
        ``n_noise`` components ``[0, ..., n_noise - 1]``.

        In **band-filter mode** (:meth:`fit` / :meth:`fit_from_raw`),
        components with the *smallest* eigenvalues capture the least
        band-specific activity and are returned as broadband noise candidates
        ``[n_channels - n_noise, ..., n_channels - 1]``.

        Parameters
        ----------
        n_noise : int | "auto", default 1
            Number of components to flag, or ``"auto"`` to use the fitted
            SENSAI eigenvalue threshold in leadfield mode.

        Returns
        -------
        list of int
        """
        self._check_fitted()
        n_total = len(self._eigenvalues)
        if n_noise == "auto":
            if self._reference_is_leadfield and np.isfinite(self._eigenvalue_threshold):
                return np.flatnonzero(
                    np.abs(self._eigenvalues) >= self._eigenvalue_threshold
                ).tolist()
            n_noise = 1
        n = max(0, min(int(n_noise), n_total))
        if self._reference_is_leadfield:
            return list(range(n))
        return list(range(n_total - n, n_total))

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
        template_map: np.ndarray | None = None,
        threshold: float = 0.7,
        n_noise: int | str = "auto",
    ) -> np.ndarray:
        """Identify artifact components then denoise in one call.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
        template_map : ndarray, shape (n_channels,), optional
            If provided, components whose activation pattern correlates with
            ``template_map`` above ``threshold`` are suppressed. If ``None``,
            falls back to :meth:`find_noise_components` with ``n_noise``.
        threshold : float, default 0.7
            Absolute-correlation threshold used when ``template_map`` is given.
        n_noise : int | "auto", default "auto"
            Component selection count or ``"auto"`` passed to
            :meth:`find_noise_components` when ``template_map`` is ``None``.

        Returns
        -------
        data_clean : ndarray, shape (n_channels, n_samples)
        """
        if template_map is not None:
            artifact_idx, _ = self.find_artifact_components(template_map, threshold)
        else:
            artifact_idx = self.find_noise_components(n_noise=n_noise)
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
    def eigenvalue_threshold(self) -> float:
        """SENSAI eigenvalue cutoff estimated during :meth:`fit_from_leadfield`."""
        self._check_fitted()
        return self._eigenvalue_threshold

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
            raise RuntimeError(
                "Call fit(), fit_from_raw(), or fit_from_leadfield() before using this method."
            )
