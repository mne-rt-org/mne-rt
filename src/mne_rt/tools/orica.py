"""Online Recursive ICA (ORICA) for real-time EEG artifact removal."""

from __future__ import annotations

import numpy as np


class ORICA:
    """Online Recursive ICA (ORICA) for EEG data.

    Parameters
    ----------
    n_channels : int
        Number of input EEG channels.
    learning_rate : float
        Learning rate for ICA updates.
    block_size : int
        Size of blocks for online updates.
    online_whitening : bool
        If True, perform online whitening. If False, assumes input is already whitened.
    calibrate_pca : bool
        If True, estimate whitening matrix from the first block and fix it.
        If False, update recursively.
    forgetfac : float
        Forgetting factor for online covariance (0 < forgetfac <= 1).
        Values < 1 allow adaptation to nonstationary signals.
    nonlinearity : str
        Nonlinearity: ``"tanh"``, ``"pow3"``, or ``"gauss"``.
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_channels: int,
        learning_rate: float = 0.1,
        block_size: int = 256,
        online_whitening: bool = True,
        calibrate_pca: bool = False,
        forgetfac: float = 1.0,
        nonlinearity: str = "tanh",
        random_state: int | None = None,
    ) -> None:
        self.n_channels = n_channels
        self.learning_rate = learning_rate
        self.block_size = block_size
        self.online_whitening = online_whitening
        self.calibrate_pca = calibrate_pca
        self.forgetfac = forgetfac
        self.nonlinearity = nonlinearity

        rng = np.random.default_rng(random_state)
        self.W, _ = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))

        self.mean_ = np.zeros((n_channels, 1))
        self.cov_ = np.eye(n_channels)
        self.whitening_ = np.eye(n_channels)
        self.whitening_inv_ = np.eye(n_channels)  # cached inverse of whitening matrix
        self._mixing_matrix: np.ndarray | None = None  # cached pinv(W)
        self._calibrated = False

    # ------------------------------------------------------------------
    # Nonlinearities
    # ------------------------------------------------------------------

    def _nonlinear_func(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.nonlinearity == "tanh":
            gY = np.tanh(Y)
            gprime = 1.0 - gY**2
        elif self.nonlinearity == "pow3":
            gY = Y**3
            gprime = 3 * Y**2
        elif self.nonlinearity == "gauss":
            gY = Y * np.exp(-0.5 * Y**2)
            gprime = (1 - Y**2) * np.exp(-0.5 * Y**2)
        else:
            raise ValueError(f"Unknown nonlinearity {self.nonlinearity!r}")
        return gY, gprime

    # ------------------------------------------------------------------
    # Whitening
    # ------------------------------------------------------------------

    def _update_whitening(self, X: np.ndarray) -> np.ndarray:
        """Update whitening matrix with forgetting factor and return whitened data."""
        self.mean_ = self.forgetfac * self.mean_ + (1 - self.forgetfac) * X.mean(
            axis=1, keepdims=True
        )
        Xc = X - self.mean_

        cov_block = (Xc @ Xc.T) / X.shape[1]
        self.cov_ = self.forgetfac * self.cov_ + (1 - self.forgetfac) * cov_block

        d, E = np.linalg.eigh(self.cov_)
        d_safe = np.maximum(d, 1e-10)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))
        D_sqrt = np.diag(np.sqrt(d_safe))

        self.whitening_ = E @ D_inv_sqrt @ E.T
        self.whitening_inv_ = E @ D_sqrt @ E.T  # exact inverse of whitening_

        return self.whitening_ @ Xc

    def _apply_whitening(self, X: np.ndarray) -> np.ndarray:
        """Apply current whitening without updating statistics."""
        return self.whitening_ @ (X - self.mean_)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def partial_fit(self, X: np.ndarray) -> "ORICA":
        """Update the unmixing matrix with a new block of EEG data.

        Parameters
        ----------
        X : ndarray, shape (n_channels, n_samples)
            EEG data block.

        Returns
        -------
        self
        """
        if X.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {X.shape[0]}")

        if self.online_whitening:
            if self.calibrate_pca and not self._calibrated:
                Xw = self._update_whitening(X)
                self._calibrated = True
            elif self.calibrate_pca:
                Xw = self._apply_whitening(X)
            else:
                Xw = self._update_whitening(X)
        else:
            Xw = X - X.mean(axis=1, keepdims=True)

        Y = self.W @ Xw
        gY, gprime = self._nonlinear_func(Y)

        N = X.shape[1]
        dW = (np.eye(self.n_channels) - np.mean(gprime, axis=1)[:, None]) @ self.W + (
            gY @ Y.T
        ) / N @ self.W
        self.W += self.learning_rate * dW
        self.W, _ = np.linalg.qr(self.W)

        # Invalidate mixing-matrix cache
        self._mixing_matrix = None
        return self

    # ------------------------------------------------------------------
    # Transform / reconstruct
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data to source space without updating W.

        Parameters
        ----------
        X : ndarray, shape (n_channels, n_samples)

        Returns
        -------
        S : ndarray, shape (n_channels, n_samples)
        """
        Xw = (
            self._apply_whitening(X) if self.online_whitening else X - X.mean(axis=1, keepdims=True)
        )
        return self.W @ Xw

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return estimated sources.

        Equivalent to ``partial_fit(X)`` followed by ``transform(X)``.
        """
        self.partial_fit(X)
        return self.transform(X)

    def inverse_transform(self, S: np.ndarray) -> np.ndarray:
        """Reconstruct sensor-space data from source signals.

        Applies the full inverse pipeline:
        ``X_rec = whitening_inv_ @ pinv(W) @ S + mean_``

        Parameters
        ----------
        S : ndarray, shape (n_channels, n_samples)
            Source signals to reconstruct.

        Returns
        -------
        X_rec : ndarray, shape (n_channels, n_samples)
        """
        A = self._get_mixing_matrix()
        Xw_rec = A @ S
        if self.online_whitening:
            return self.whitening_inv_ @ Xw_rec + self.mean_
        return Xw_rec + self.mean_

    # ------------------------------------------------------------------
    # Artifact identification and removal
    # ------------------------------------------------------------------

    def find_blink_ic(
        self, template_map: np.ndarray, threshold: float = 0.7
    ) -> tuple[list[int], np.ndarray]:
        """Identify ICs corresponding to blinks by template correlation.

        Parameters
        ----------
        template_map : ndarray, shape (n_channels,)
            Spatial topography of a typical blink.
        threshold : float
            Absolute-correlation threshold.

        Returns
        -------
        blink_idx : list of int
        corrs : ndarray, shape (n_channels,)
        """
        A = self._get_mixing_matrix()
        corrs = np.array([np.corrcoef(A[:, ic], template_map)[0, 1] for ic in range(A.shape[1])])
        blink_idx = np.where(np.abs(corrs) > threshold)[0].tolist()
        return blink_idx, corrs

    def denoise(self, X: np.ndarray, artifact_idx: list[int]) -> np.ndarray:
        """Remove specific ICs and reconstruct sensor data.

        Parameters
        ----------
        X : ndarray, shape (n_channels, n_samples)
            Raw EEG data.
        artifact_idx : list of int
            Component indices to suppress.

        Returns
        -------
        X_clean : ndarray, shape (n_channels, n_samples)
        """
        S = self.transform(X)
        S_clean = S.copy()
        if artifact_idx:
            S_clean[artifact_idx, :] = 0.0
        return self.inverse_transform(S_clean)

    def update_and_denoise(
        self,
        X: np.ndarray,
        template_map: np.ndarray,
        threshold: float = 0.7,
    ) -> np.ndarray:
        """Fit one block, detect blinks by template, and return cleaned data.

        Parameters
        ----------
        X : ndarray, shape (n_channels, n_samples)
        template_map : ndarray, shape (n_channels,)
        threshold : float

        Returns
        -------
        X_clean : ndarray, shape (n_channels, n_samples)
        """
        self.partial_fit(X)
        blink_idx, _ = self.find_blink_ic(template_map, threshold)
        return self.denoise(X, blink_idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mixing_matrix(self) -> np.ndarray:
        """Return cached pinv(W), recomputing only when W changed."""
        if self._mixing_matrix is None:
            self._mixing_matrix = np.linalg.pinv(self.W)
        return self._mixing_matrix
