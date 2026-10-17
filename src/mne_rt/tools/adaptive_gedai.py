"""Delayed, streaming Haar adaptive multiband GEDAI.

The FIR kernels are obtained from pyGEDAI's actual Haar MODWT reconstruction,
not an IIR approximation. Symmetric band kernels are shifted to one common
causal delay; the output sums only retained, cleaned bands. Bands at or below
the low cutoff never contribute a residual to the output.

Calibration reuses pyGEDAI's epoch sizing and SENSAI optimization, including
an optional broadband pass. Streaming uses finite trailing covariance epochs
and recalculates eigenvalue cutoffs from the fitted SENSAI sensitivity.
Unlike offline pyGEDAI's overlapping, cosine-weighted epoch reconstructions,
streaming uses one trailing-epoch spatial projection per acquired chunk.
Thus the uncleaned Haar bands match after delay alignment, but denoised
output is not guaranteed identical to the noncausal offline algorithm.

By default, the 0.1 Hz input prefilter and, in broadband mode, low-cut and broadband
0.1 Hz prefilters also use delayed Haar reconstruction. Their delays add
to the common band delay. Consumers must account for ``delay_samples``;
output starts with delayed calibration history, not current live samples.

License notice
--------------
mne-rt itself is BSD-3-Clause, but this module depends on the ``gedai``
package, which is distributed under the PolyForm Noncommercial License 1.0.0
(https://polyformproject.org/licenses/noncommercial/1.0.0). That license
restricts use to noncommercial purposes, and the underlying GEDAI algorithm
is patent-pending with no commercial patent grant included. Commercial use
of :class:`AdaptiveMultibandGEDAIDenoiser` is not covered by mne-rt's
permissive license -- see the project README's "License" section. Note that
``gedai`` is installed as a mandatory mne-rt dependency even though only
this module (not :class:`~mne_rt.tools.GEDAIDenoiser`) imports it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from scipy import linalg
from scipy.signal import lfilter

try:
    from gedai.gedai.adaptive import _compute_wavelet_parameters
    from gedai.gedai.gedai import _clean_continuous_dual_stream
    from gedai.sensai.sensai import _precompute_gevd, _sensai_optimize, _sensai_to_eigen
    from gedai.wavelet.transform import (
        _apply_wavelet_highpass_prefilter,
        _modwt_haar_single_band,
        compute_wavelet_level,
    )
except ImportError as exc:  # pragma: no cover - exercised only without gedai installed
    raise ImportError(
        "AdaptiveMultibandGEDAIDenoiser requires the 'gedai' package "
        "(pip install gedai) for its wavelet band layout and SENSAI calibration."
    ) from exc

from mne_rt._logging import logger


class AdaptiveMultibandGEDAIDenoiser:
    """Streaming adaptive multiband GEDAI.

    Parameters
    ----------
    n_channels : int
        Number of EEG/MEG channels.
    sfreq : float
        Sampling frequency in Hz.
    wavelet_level : int | "auto", default "auto"
        Number of wavelet decomposition levels (``n_bands = wavelet_level +
        1``). If ``"auto"``, uses :func:`gedai.wavelet.transform.compute_wavelet_level`.
    cycles_per_wavelet : int, default 6
        Cycles at each band's lower frequency used to size covariance epochs.
        The lowest approximation uses half its upper edge, as in pyGEDAI.
    shrinkage : float, default 0.05
        Reference covariance shrinkage: ``(1-s) * R + s * mean(diag(R)) * I``.
    filter_order : int | None, default None
        Deprecated compatibility argument. Haar kernels have no Butterworth
        order. Passing this argument logs a warning; it has no effect.
    n_jobs : int, default 1
        Number of threads used to denoise bands in parallel (``joblib``,
        ``prefer="threads"``). Use ``-1`` for all cores.
    wavelet_low_cutoff : float, default 0.5
        Bands whose upper edge is at or below this frequency (Hz) are
        discarded. This is a dyadic wavelet cutoff, not a brick-wall filter.
        Also controls the broadband prefilter and automatic level selection.
    broadband_pass : bool, default True
        Calibrate and apply a broadband GEDAI pass before wavelet cleaning,
        matching pyGEDAI's default. The broadband covariance epoch is one
        second (or enough samples for the channel count).
    prefilter_mode : {"reference", "lowcut", "none"}, default "reference"
        ``"reference"`` preserves the pyGEDAI prefilter cascade. ``"lowcut"``
        uses only one Haar highpass at ``wavelet_low_cutoff``, even without a
        broadband pass. ``"none"`` skips standalone prefilters; sub-cutoff
        wavelet bands are still excluded. The latter modes reduce delay but
        change calibration and broadband cleaning, so require a new ``fit``.

    Notes
    -----
    ``fit`` calibrates an eigenvalue threshold in each band with pyGEDAI's
    EEG SENSAI objective (up to three reference principal components, noise
    weight 3, frequency-dependent sensitivity bounds). The fitted sensitivity
    stays fixed; each local eigenvalue spectrum determines the current
    cutoff. Baseline and live data must use the same units and sensor
    reference as the supplied leadfield.

    The finite trailing covariance history is bounded by the pyGEDAI epoch
    size, not an EMA with an unbounded tail. All emitted samples lag their
    input timestamps by ``latency`` seconds, excluding computation time.
    This can be several seconds even though no acquisition chunks are held
    waiting for a processing window. Offline calibration may use future
    samples within the baseline, but live filtering never reads future chunks.

    References
    ----------
    Ros, T., Férat, V., Huang, Y., et al. (2025). Return of the GEDAI:
    Unsupervised EEG Denoising based on Leadfield Filtering. *bioRxiv*.
    https://doi.org/10.1101/2025.10.04.680449
    """

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        wavelet_level: int | str = "auto",
        cycles_per_wavelet: int = 6,
        shrinkage: float = 0.05,
        filter_order: int | None = None,
        n_jobs: int = 1,
        wavelet_low_cutoff: float = 0.5,
        broadband_pass: bool = True,
        prefilter_mode: Literal["reference", "lowcut", "none"] = "reference",
    ) -> None:
        self.n_channels = int(n_channels)
        self.sfreq = float(sfreq)
        self.cycles_per_wavelet = int(cycles_per_wavelet)
        self.shrinkage = float(shrinkage)
        self.filter_order = filter_order
        self.n_jobs = int(n_jobs)
        self.wavelet_low_cutoff = float(wavelet_low_cutoff)
        self.broadband_pass = bool(broadband_pass)
        if prefilter_mode not in ("reference", "lowcut", "none"):
            raise ValueError("prefilter_mode must be 'reference', 'lowcut', or 'none'")
        self.prefilter_mode = prefilter_mode
        if self.n_channels < 1 or not np.isfinite(self.sfreq) or self.sfreq <= 0:
            raise ValueError("n_channels and sfreq must be positive")
        if self.cycles_per_wavelet < 1:
            raise ValueError("cycles_per_wavelet must be positive")
        if (
            not np.isfinite(self.wavelet_low_cutoff)
            or not 0 <= self.wavelet_low_cutoff < self.sfreq / 2
        ):
            raise ValueError("wavelet_low_cutoff must be in [0, sfreq / 2)")
        if not np.isfinite(self.shrinkage) or not 0 <= self.shrinkage <= 1:
            raise ValueError("shrinkage must be between 0 and 1")
        if filter_order is not None:
            logger.warning(
                "filter_order is deprecated and has no effect: causal GEDAI now "
                "uses delayed Haar reconstruction rather than Butterworth filters."
            )
        self._requested_wavelet_level = wavelet_level

        if wavelet_level == "auto":
            wavelet_level = compute_wavelet_level(
                self.sfreq,
                lowcut_hz=self.wavelet_low_cutoff or 0.5,
                cycles_per_wavelet=self.cycles_per_wavelet,
            )
        self.wavelet_level = int(wavelet_level)
        if self.wavelet_level < 0:
            raise ValueError("wavelet_level must be non-negative")

        self._bands: list[dict] | None = None
        self._reference_cov: np.ndarray | None = None
        self._reference_is_leadfield = False
        self._filter_kernels: list[np.ndarray] = []
        self._filter_zi: list[np.ndarray] = []
        self._prefilter_kernel = np.ones(1)
        self._prefilter_zi = np.empty((self.n_channels, 0))
        self._broadband: dict | None = None
        self._delay_samples = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self, data: np.ndarray, leadfield: np.ndarray | None = None
    ) -> "AdaptiveMultibandGEDAIDenoiser":
        """Fit per-band spatial filters from baseline data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Broadband baseline recording.
        leadfield : ndarray, shape (n_channels, n_sources), optional
            Forward/leadfield gain matrix. If given, the reference covariance
            is ``L @ L.T`` (true GEDAI / leadfield mode); otherwise the
            broadband covariance of ``data`` is used (band-filter mode).
            For automatic selection without a leadfield, this baseline must
            be representative of the signal covariance to preserve.

        Returns
        -------
        self
        """
        data = self._validate_data(data)
        n_samples = data.shape[1]
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to fit AdaptiveMultibandGEDAIDenoiser")

        if leadfield is not None:
            leadfield = np.asarray(leadfield, dtype=float)
            if (
                leadfield.ndim != 2
                or leadfield.shape[0] != self.n_channels
                or leadfield.shape[1] == 0
                or not np.isfinite(leadfield).all()
            ):
                raise ValueError(
                    f"Leadfield must be finite with shape ({self.n_channels}, n_sources)"
                )
            R_ref = leadfield @ leadfield.T
            ref_power = np.trace(R_ref) / self.n_channels
            if ref_power <= 0:
                raise ValueError("Leadfield must have nonzero power")
            self._reference_is_leadfield = True
        else:
            Xb = data - data.mean(axis=1, keepdims=True)
            R_ref = (Xb @ Xb.T) / n_samples
            self._reference_is_leadfield = False
        reg = self.shrinkage * np.trace(R_ref) / self.n_channels
        if np.trace(R_ref) <= 0:
            raise ValueError("Baseline must have nonzero variance in band-filter mode")
        R_ref = (1 - self.shrinkage) * R_ref + reg * np.eye(self.n_channels)
        self._reference_cov = R_ref

        if self._requested_wavelet_level == "auto":
            self.wavelet_level = compute_wavelet_level(
                self.sfreq,
                lowcut_hz=self.wavelet_low_cutoff or 0.5,
                n_times=n_samples,
                cycles_per_wavelet=self.cycles_per_wavelet,
            )
        parameters = _compute_wavelet_parameters(
            self.sfreq,
            self.wavelet_level,
            self.cycles_per_wavelet,
            self.n_channels,
            n_samples,
        )
        if self.prefilter_mode == "reference":
            cutoffs = [0.1]
            if self.broadband_pass:
                cutoffs.extend([self.wavelet_low_cutoff, 0.1])
        elif self.prefilter_mode == "lowcut":
            cutoffs = [self.wavelet_low_cutoff]
        else:
            cutoffs = []
        self._prefilter_kernel = np.ones(1)
        baseline = data.copy()
        for cutoff in cutoffs:
            self._prefilter_kernel = np.convolve(
                self._prefilter_kernel, self._highpass_kernel(cutoff, n_samples)
            )
            baseline = _apply_wavelet_highpass_prefilter(
                baseline, self.sfreq, lowcut_hz=cutoff, engine="numpy"
            )
        self._prefilter_zi = self._initial_state(self._prefilter_kernel, data[:, 0])
        streamed_baseline, self._prefilter_zi = lfilter(
            self._prefilter_kernel, [1.0], data, axis=1, zi=self._prefilter_zi
        )
        prefilter_delay = (len(self._prefilter_kernel) - 1) // 2
        self._broadband = None
        if self.broadband_pass:
            bb_samples = max(2, round(self.sfreq), min(self.n_channels, n_samples))
            self._broadband = self._init_band(
                -1,
                0.0,
                self.sfreq / 2,
                baseline,
                bb_samples,
                (-4.0, 12.0),
                overlap=0.5,
            )
            baseline, _ = _clean_continuous_dual_stream(
                baseline,
                self.sfreq,
                self._reference_cov,
                bb_samples / self.sfreq,
                threshold=self._broadband["threshold"],
                T1=(105 - self._broadband["sensitivity"]) / 100,
                percentile=98,
                engine="numpy",
            )
            # Seed causal states at the same delayed time reached by the prefilter.
            self._seed_history(self._broadband, streamed_baseline[:, :1])
            bb_stream = np.empty_like(streamed_baseline)
            step = max(1, round(0.05 * self.sfreq))
            for start in range(0, n_samples, step):
                chunk = streamed_baseline[:, start : start + step]
                bb_stream[:, start : start + step] = chunk - self._process_band(
                    self._broadband, chunk, None, 0.7, "auto", True, True
                )
            streamed_baseline = bb_stream

        active = [p["fmax"] > self.wavelet_low_cutoff for p in parameters]
        supports = [
            2 ** (min(i + 1, self.wavelet_level)) - 1 if keep else 0
            for i, keep in enumerate(active)
        ]
        band_delay = max(supports)
        self._delay_samples = prefilter_delay + band_delay
        self._filter_kernels = []
        baseline_bands = []
        for i, keep in enumerate(active):
            if keep:
                kernel = self._haar_kernel(self.wavelet_level, i)
                trim = (len(kernel) - 1) // 2 - supports[i]
                if trim:
                    kernel = kernel[trim:-trim]
                kernel = np.pad(kernel, (band_delay - supports[i],) * 2)
                values = _modwt_haar_single_band(baseline.T, self.wavelet_level, i, engine="numpy")
            else:
                kernel = np.zeros(2 * band_delay + 1)
                values = np.zeros_like(baseline)
            self._filter_kernels.append(kernel)
            baseline_bands.append(values)
        self._filter_zi = [
            self._initial_state(kernel, streamed_baseline[:, 0]) for kernel in self._filter_kernels
        ]
        self._bands = [
            self._init_band(
                i,
                p["fmin"],
                p["fmax"],
                values,
                p["n_samples"] + p["n_samples"] % 2,
                (
                    -6.0 if self.wavelet_low_cutoff <= (p["fmin"] + p["fmax"]) / 2 <= 60 else 0.0,
                    12.0,
                ),
                ignore=not keep,
            )
            for i, (p, values, keep) in enumerate(zip(parameters, baseline_bands, active))
        ]
        delayed_bands = self._split_bands(streamed_baseline)
        for band, values in zip(self._bands, delayed_bands):
            if not band["ignore"]:
                self._seed_history(band, values)
        logger.info(
            "Causal Haar GEDAI: %d retained bands, %.3f s output delay, "
            "%d cycles, %.3f Hz low cutoff.",
            sum(active),
            self.latency,
            self.cycles_per_wavelet,
            self.wavelet_low_cutoff,
        )
        return self

    @staticmethod
    def _haar_kernel(level: int, band_index: int) -> np.ndarray:
        support = 2**level - 1
        impulse = np.zeros((2 * support + 1, 1))
        impulse[support, 0] = 1
        return _modwt_haar_single_band(impulse, level, band_index, engine="numpy")[0]

    def _highpass_kernel(self, cutoff: float, n_samples: int) -> np.ndarray:
        if cutoff <= 0:
            return np.ones(1)
        level = max(1, int(np.ceil(np.log2(self.sfreq / cutoff))) - 1)
        if level > int(np.floor(np.log2(n_samples))):
            logger.warning(
                "Baseline too short to resolve the %.3f Hz Haar prefilter; "
                "this prefilter is not applied, matching pyGEDAI.",
                cutoff,
            )
            return np.ones(1)
        kernel = -self._haar_kernel(level, level)
        kernel[len(kernel) // 2] += 1
        return kernel

    @staticmethod
    def _initial_state(kernel: np.ndarray, first_sample: np.ndarray) -> np.ndarray:
        return first_sample[:, None] * np.cumsum(kernel[:0:-1])[::-1][None, :]

    def _split_bands(self, data: np.ndarray) -> list[np.ndarray]:
        bands = []
        for i, kernel in enumerate(self._filter_kernels):
            values, self._filter_zi[i] = lfilter(kernel, [1.0], data, axis=1, zi=self._filter_zi[i])
            bands.append(values)
        return bands

    def _init_band(
        self,
        band_idx: int,
        fmin: float,
        fmax: float,
        band_data: np.ndarray,
        epoch_samples: int,
        bounds: tuple[float, float],
        ignore: bool = False,
        overlap: float = 0.0,
    ) -> dict:
        center_freq = max(float(np.sqrt(fmin * fmax)) if fmin > 0 else fmax / 2, 0.01)
        tau = epoch_samples / self.sfreq

        n_samples = band_data.shape[1]
        mean = band_data.mean(axis=1)
        Xb = band_data - mean[:, None]
        R_band = (Xb @ Xb.T) / n_samples

        W, A, eigenvalues = self._solve_ged(R_band)
        eigen_threshold, sensitivity = (
            (np.inf, 0.0)
            if ignore
            else self._fit_sensai(band_data, fmin, fmax, epoch_samples, bounds, overlap)
        )

        return {
            "index": band_idx,
            "fmin": fmin,
            "fmax": fmax,
            "center_freq": center_freq,
            "tau": tau,
            "epoch_samples": epoch_samples,
            "history": band_data[:, -epoch_samples:].copy(),
            "ignore": ignore,
            "mean": mean,
            "R": R_band,
            "W": W,
            "A": A,
            "eigenvalues": eigenvalues,
            "threshold": eigen_threshold,
            "effective_threshold": eigen_threshold,
            "sensitivity": sensitivity,
        }

    def _fit_sensai(
        self,
        data: np.ndarray,
        fmin: float,
        fmax: float,
        epoch_samples: int,
        bounds: tuple[float, float],
        overlap: float = 0.0,
    ) -> tuple[float, float]:
        centered = data - data.mean(axis=1, keepdims=True)
        power = float(np.mean(centered**2))
        if power == 0:
            logger.warning(
                "No baseline variance in GEDAI band %.3f-%.3f Hz; "
                "automatic rejection is disabled for this band until refitted.",
                fmin,
                fmax,
            )
            return np.inf, 0.0
        epoch_samples = min(data.shape[1], epoch_samples)
        step = max(1, round(epoch_samples * (1 - overlap)))
        epochs = np.stack(
            [
                data[:, start : start + epoch_samples]
                for start in range(0, data.shape[1] - epoch_samples + 1, step)
            ]
        )
        values, vectors = _precompute_gevd(epochs, self._reference_cov, engine="numpy")
        eigen_threshold, sensitivity, _ = _sensai_optimize(
            epochs,
            self._reference_cov,
            n_pc=min(3, self.n_channels),
            noise_multiplier=3.0,
            epochs_eigenvalues=values,
            bounds=bounds,
            all_eval=values,
            all_evec=vectors,
            percentile=98,
            signal_type="eeg",
            engine="numpy",
        )
        return float(eigen_threshold), float(sensitivity)

    def _seed_history(self, band: dict, data: np.ndarray) -> None:
        history = data[:, -band["epoch_samples"] :].copy()
        band["history"] = history
        band["mean"] = history.mean(axis=1)
        centered = history - band["mean"][:, None]
        band["R"] = centered @ centered.T / max(1, history.shape[1] - 1)
        band["W"], band["A"], band["eigenvalues"] = self._solve_ged(band["R"])

    def _solve_ged(self, R_band: np.ndarray):
        eigenvalues, W = linalg.eigh(R_band, self._reference_cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        W = W[:, order]
        A = self._reference_cov @ W
        return W, A, eigenvalues

    # ------------------------------------------------------------------
    # Streaming denoise
    # ------------------------------------------------------------------

    def update_and_denoise(
        self,
        data: np.ndarray,
        template_map: np.ndarray | None = None,
        threshold: float = 0.7,
        n_artifact: int | str = "auto",
        update_covariance: bool = True,
        refit: bool = True,
    ) -> np.ndarray:
        """Denoise one real-time chunk, updating per-band covariances as it goes.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            New chunk of broadband sensor data.
        template_map : ndarray, shape (n_channels,), optional
            Known artifact topography (e.g. blink). If given, components
            whose activation pattern correlates with it above ``threshold``
            (absolute correlation) are suppressed in every band.
        threshold : float, default 0.7
            Absolute-correlation threshold used with ``template_map``.
        n_artifact : int | "auto", default "auto"
            When ``template_map`` is ``None``, ``"auto"`` rejects eigenvalues
            above each band's current SENSAI threshold, calculated from the
            baseline-fitted sensitivity and the local eigenvalue spectrum. With no
            leadfield, the baseline broadband covariance is the signal
            reference instead of a theoretical brain model.
            An integer overrides SENSAI with a fixed count. Leadfield mode suppresses the
            largest eigenvalues (excess power relative to the brain model);
            band-filter mode suppresses the smallest (least band-specific).
            Zero disables component rejection, not prefiltering, output delay,
            or sub-cutoff band removal.
            A positive fixed count can remove activity from clean recordings.
        update_covariance : bool, default True
            Whether to update each band's trailing-epoch covariance with this chunk.
        refit : bool, default True
            Whether to re-solve the GED for each band after updating its
            covariance. Set to ``False`` to only update covariances (e.g. to
            re-solve less often than every chunk) while still denoising with
            the previous spatial filters.

        Returns
        -------
        data_clean : ndarray, shape (n_channels, n_samples)
            Output sample ``t`` corresponds to input sample
            ``t - delay_samples``, including delayed calibration history at startup.
        """
        self._check_fitted()
        data = self._validate_data(data)
        if n_artifact != "auto" and (
            not isinstance(n_artifact, (int, np.integer)) or n_artifact < 0
        ):
            raise ValueError("n_artifact must be 'auto' or a non-negative integer")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if template_map is not None:
            template_map = np.asarray(template_map, dtype=float)
            if (
                template_map.shape != (self.n_channels,)
                or not np.isfinite(template_map).all()
                or np.ptp(template_map) == 0
            ):
                raise ValueError("template_map must be finite, nonconstant, and match n_channels")

        data, self._prefilter_zi = lfilter(
            self._prefilter_kernel, [1.0], data, axis=1, zi=self._prefilter_zi
        )
        if self._broadband is not None:
            data = data - self._process_band(
                self._broadband,
                data,
                template_map,
                threshold,
                n_artifact,
                update_covariance,
                refit,
            )
        band_data = self._split_bands(data)
        args = (template_map, threshold, n_artifact, update_covariance, refit)
        if self.n_jobs != 1 and len(self._bands) > 1:
            noise_bands = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._process_band)(band, values, *args)
                for band, values in zip(self._bands, band_data)
            )
        else:
            noise_bands = [
                self._process_band(band, values, *args)
                for band, values in zip(self._bands, band_data)
            ]
        return np.sum(band_data, axis=0) - np.sum(noise_bands, axis=0)

    def _process_band(
        self,
        band: dict,
        band_data: np.ndarray,
        template_map: np.ndarray | None,
        threshold: float,
        n_artifact: int | str,
        update_covariance: bool,
        refit: bool,
    ) -> np.ndarray:
        if band["ignore"]:
            return np.zeros_like(band_data)

        if update_covariance:
            history = np.concatenate([band["history"], band_data], axis=1)
            history = history[:, -band["epoch_samples"] :]
            band["history"] = history
            band["mean"] = history.mean(axis=1)
            centered = history - band["mean"][:, None]
            band["R"] = centered @ centered.T / max(1, history.shape[1] - 1)
        if refit:
            band["W"], band["A"], band["eigenvalues"] = self._solve_ged(band["R"])

        W, A = band["W"], band["A"]
        if template_map is not None:
            template = template_map - template_map.mean()
            patterns = A - A.mean(axis=0, keepdims=True)
            norms = np.linalg.norm(patterns, axis=0) * np.linalg.norm(template)
            corrs = np.divide(template @ patterns, norms, out=np.zeros(A.shape[1]), where=norms > 0)
            artifact_idx = np.where(np.abs(corrs) > threshold)[0]
        elif n_artifact == "auto":
            positive = np.abs(band["eigenvalues"])
            if np.isfinite(band["threshold"]) and np.any(positive > 0):
                band["effective_threshold"] = float(
                    _sensai_to_eigen(band["sensitivity"], positive, percentile=98)
                )
            else:
                band["effective_threshold"] = np.inf
            artifact_idx = np.flatnonzero(positive >= band["effective_threshold"])
        else:
            assert isinstance(n_artifact, (int, np.integer))
            n = min(n_artifact, A.shape[1])
            artifact_idx = (
                np.arange(n)
                if self._reference_is_leadfield
                else np.arange(A.shape[1] - n, A.shape[1])
            )

        return A[:, artifact_idx] @ (W[:, artifact_idx].T @ band_data)

    def _validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] != self.n_channels or data.shape[1] == 0:
            raise ValueError(f"Expected data with shape ({self.n_channels}, n_samples > 0)")
        if not np.isfinite(data).all():
            raise ValueError("Data must contain only finite values")
        return data

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_bands(self) -> int:
        """Number of frequency bands."""
        self._check_fitted()
        return len(self._bands)

    @property
    def band_frequencies(self) -> list[tuple[float, float]]:
        """List of ``(fmin, fmax)`` per band."""
        self._check_fitted()
        return [(b["fmin"], b["fmax"]) for b in self._bands]

    @property
    def band_time_constants(self) -> list[float]:
        """Trailing covariance epoch durations (seconds), including ignored bands."""
        self._check_fitted()
        return [b["tau"] for b in self._bands]

    @property
    def retained_bands(self) -> list[bool]:
        """Whether each band contributes to the reconstruction."""
        self._check_fitted()
        return [not b["ignore"] for b in self._bands]

    @property
    def latency(self) -> float:
        """Fixed FIR reconstruction delay in seconds; not compute time."""
        self._check_fitted()
        return self._delay_samples / self.sfreq

    @property
    def delay_samples(self) -> int:
        """Align emitted sample ``t`` to source sample ``t - delay_samples``."""
        self._check_fitted()
        return self._delay_samples

    @property
    def eigenvalues(self) -> list[np.ndarray]:
        """Sorted eigenvalues (descending) per band."""
        self._check_fitted()
        return [b["eigenvalues"] for b in self._bands]

    @property
    def eigenvalue_thresholds(self) -> list[float]:
        """Baseline SENSAI eigenvalue cutoffs; live cutoffs adapt to each spectrum."""
        self._check_fitted()
        return [b["threshold"] for b in self._bands]

    @property
    def current_eigenvalue_thresholds(self) -> list[float]:
        """Most recently used per-band automatic cutoffs."""
        self._check_fitted()
        return [b["effective_threshold"] for b in self._bands]

    @property
    def spatial_filters(self) -> list[np.ndarray]:
        """Spatial filters ``W`` per band, each shape (n_channels, n_channels)."""
        self._check_fitted()
        return [b["W"] for b in self._bands]

    @property
    def activation_patterns(self) -> list[np.ndarray]:
        """Activation patterns ``A`` per band, each shape (n_channels, n_channels)."""
        self._check_fitted()
        return [b["A"] for b in self._bands]

    def _check_fitted(self) -> None:
        if self._bands is None:
            raise RuntimeError("Call fit() before using AdaptiveMultibandGEDAIDenoiser.")


class _CausalGEDAIWindow:
    """Process unique timestamped samples, then assemble delayed analysis windows."""

    def __init__(self, model: AdaptiveMultibandGEDAIDenoiser, window_samples: int):
        self.model = model
        self.window_samples = window_samples
        self._chunk_samples = max(1, round(0.05 * model.sfreq))
        self._warmup = model.delay_samples
        self._last_timestamp: float | None = None
        self._pending_raw = np.empty((model.n_channels, 0))
        self._pending_times = np.empty(0)
        self._clean = np.empty((model.n_channels, 0))
        self._raw = np.empty((model.n_channels, 0))
        self._times = np.empty(0)

    def update(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        data = self.model._validate_data(data)
        timestamps = np.asarray(timestamps, dtype=float)
        if timestamps.shape != (data.shape[1],) or not np.isfinite(timestamps).all():
            raise ValueError("Causal GEDAI requires one finite timestamp per input sample")
        valid = np.flatnonzero(timestamps > 0)
        if not valid.size:
            logger.debug("Causal GEDAI is waiting for initialized stream timestamps.")
            return None
        data, timestamps = data[:, valid[0] :], timestamps[valid[0] :]
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Causal GEDAI input timestamps must be strictly increasing")
        if self._last_timestamp is not None:
            first = np.searchsorted(timestamps, self._last_timestamp, side="right")
            data, timestamps = data[:, first:], timestamps[first:]
        if not timestamps.size:
            return None
        previous = (
            timestamps if self._last_timestamp is None else np.r_[self._last_timestamp, timestamps]
        )
        if np.any(np.diff(previous) > 1.5 / self.model.sfreq):
            raise RuntimeError(
                "Causal GEDAI detected missing input samples. Increase the stream "
                "buffer/analysis window or reduce processing load, then refit and restart."
            )

        for start in range(0, timestamps.size, self._chunk_samples):
            stop = start + self._chunk_samples
            chunk, times = data[:, start:stop], timestamps[start:stop]
            cleaned = self.model.update_and_denoise(chunk)
            self._pending_raw = np.concatenate([self._pending_raw, chunk], axis=1)
            self._pending_times = np.r_[self._pending_times, times]
            skip = min(self._warmup, chunk.shape[1])
            self._warmup -= skip
            ready = chunk.shape[1] - skip
            if ready:
                self._clean = np.concatenate([self._clean, cleaned[:, skip:]], axis=1)
                self._raw = np.concatenate([self._raw, self._pending_raw[:, :ready]], axis=1)
                self._times = np.r_[self._times, self._pending_times[:ready]]
                self._pending_raw = self._pending_raw[:, ready:]
                self._pending_times = self._pending_times[ready:]
                self._clean = self._clean[:, -self.window_samples :]
                self._raw = self._raw[:, -self.window_samples :]
                self._times = self._times[-self.window_samples :]
        self._last_timestamp = float(timestamps[-1])
        if self._times.size < self.window_samples:
            return None
        return self._clean.copy(), self._raw.copy(), self._times.copy()
