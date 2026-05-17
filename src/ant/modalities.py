"""ModalityMixin — real-time feature extraction for all NF modalities.

Mixed into :class:`NFRealtime`. Assumes the following attributes exist on
``self`` at call time:

- ``rec_info``      — MNE :class:`~mne.Info`
- ``data_type``     — ``"eeg"`` | ``"meg"``
- ``_sfreq``        — sampling frequency (Hz)
- ``winsize``       — analysis window length (s)
- ``picks``         — channel selection (may be ``None``)
- ``params``        — current modality parameter dict (set before each prep call)
- ``subject_fs_id`` — FreeSurfer subject identifier
- ``subjects_fs_dir``
- ``subject_dir``   — :class:`~pathlib.Path` to subject data folder
- ``visit``         — visit number
- ``raw_baseline``  — baseline :class:`~mne.io.RawArray` (source modalities)
- ``fwd``, ``noise_cov`` — forward solution / covariance (LCMV)
- ``_prepare_raw_array(data)`` — wraps data in ``RawArray``, applies EEG ref if needed

New modalities added:
- ``erd_ers``          — event-related desynchronisation/synchronisation (%)
- ``laterality``       — inter-hemispheric power asymmetry index
- ``hjorth``           — mean of Hjorth mobility and complexity (no FFT)
- ``spectral_centroid``— frequency-weighted centre-of-mass of the PSD within a band
"""
from __future__ import annotations

from typing import Optional
from warnings import warn

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt, welch
from pactools import Comodulogram

import mne
from mne import read_labels_from_annot
from mne.minimum_norm import apply_inverse_raw, read_inverse_operator
from mne.beamformer import apply_lcmv_raw, make_lcmv
from mne_connectivity import spectral_connectivity_time
from mne_features.univariate import (
    compute_app_entropy,
    compute_samp_entropy,
    compute_spect_entropy,
    compute_svd_entropy,
    
)

from ant._logging import logger
from ant.tools import (
    butter_bandpass,
    compute_bandpower,
    compute_fft,
    estimate_aperiodic_component,
    log_degree_barrier,
    timed,
)


class ModalityMixin:
    """Feature-extraction engine for all ANT NF modalities.

    :class:`ModalityMixin` is mixed into :class:`~ant.NFRealtime` and provides
    the **prep / compute** pair for every modality:

    * **Prep** (``_<modality>_prep()``) — runs *once* before the main loop.
      Returns a ``dict`` of pre-computed artefacts (filter coefficients, index
      arrays, connectivity indices, …) that are passed as keyword arguments to
      the compute step.
    * **Compute** (``_<modality>(data, **prep_kwargs)``) — runs *every window*
      inside a thread-pool worker.  Decorated with :func:`~ant.tools.timed` so
      it returns ``(value, elapsed_seconds)``.

    Supported modalities (20 total)
    --------------------------------

    **Sensor-space power & time-domain**

    ``sensor_power``, ``band_ratio``, ``erd_ers``, ``laterality``,
    ``laterality_erd_ers``, ``hjorth``, ``spectral_centroid``, ``argmax_freq``,
    ``individual_peak_power``, ``entropy``, ``instantaneous_phase``,
    ``scp``, ``peak_alpha_freq``

    **Sensor-space connectivity & graph**

    ``sensor_connectivity``, ``connectivity_ratio``, ``cfc_sensor``,
    ``sensor_graph``

    **Source-space**

    ``source_power``, ``source_connectivity``, ``source_graph``

    Notes
    -----
    All methods are private (single-underscore prefix) and are called
    internally by :meth:`~ant.NFRealtime.record_main`.  To extend ANT with a
    custom modality, sub-class :class:`~ant.NFRealtime` and add a matching
    ``_<name>_prep`` / ``_<name>`` pair following the same pattern.
    """

    # ------------------------------------------------------------------
    # Prep methods  (run once before the main loop, return kwargs dict)
    # ------------------------------------------------------------------

    def _sensor_power_prep(self) -> dict:
        return {
            "sfreq": self.rec_info["sfreq"],
            "frange": self.params["frange"],
            "method": self.params["method"],
            "relative": self.params["relative"],
        }

    def _band_ratio_prep(self) -> dict:
        return {
            "sfreq": self.rec_info["sfreq"],
            "frange_1": self.params["frange_1"],
            "frange_2": self.params["frange_2"],
            "method": self.params["method"],
        }

    def _individual_peak_power_prep(self) -> dict:
        _, peak_params_ = estimate_aperiodic_component(
            raw_baseline=self.raw_baseline,
            picks=self.picks,
            method=self.params["method"],
        )
        candidates = [
            p[0] for p in peak_params_
            if self.params["frange"][0] < p[0] < self.params["frange"][1]
        ]
        if len(candidates) == 1:
            cf = candidates[0]
        else:
            cf = (self.params["frange"][0] + self.params["frange"][1]) / 2.0
            warn(
                "individual_peak_power: center frequency defaulted to mid-range "
                f"({cf:.1f} Hz); found {len(candidates)} peak(s) in band.",
                UserWarning,
                stacklevel=2,
            )
        return {"sfreq": self._sfreq, "freq_var": 2.0, "cf": cf}

    def _entropy_prep(self) -> dict:
        sos = butter_bandpass(
            self.params["frange"][0], self.params["frange"][1],
            self._sfreq, order=5,
        )
        return {
            "sos": sos,
            "method": self.params["method"],
            "psd_method": self.params["psd_method"],
        }

    def _argmax_freq_prep(self) -> dict:
        if not hasattr(self, "raw_baseline"):
            raise RuntimeError(
                "Baseline recording must be completed before using 'argmax_freq'."
            )
        ap_params, _ = estimate_aperiodic_component(
            raw_baseline=self.raw_baseline,
            picks=self.picks,
            method=self.params["method"],
        )
        n_samples = int(self.winsize * self._sfreq)
        fft_window = np.hanning(n_samples)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self._sfreq)
        mask = (freqs >= self.params["frange"][0]) & (freqs <= self.params["frange"][1])
        freqs_band = freqs[mask]
        ap_model = (10 ** ap_params[0]) / (freqs_band ** ap_params[1])

        def _gaussian(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
            return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        return {
            "fft_window": fft_window,
            "ap_model": ap_model,
            "gaussian": _gaussian,
        }

    def _source_power_prep(self) -> dict:
        fft_window, _, freq_band_idxs, _ = compute_fft(
            sfreq=self._sfreq,
            winsize=self.winsize,
            freq_range=self.params["frange"],
        )
        bls = read_labels_from_annot(
            subject=self.subject_fs_id,
            parc=self.params["atlas"],
            subjects_dir=self.subjects_fs_dir,
        )
        brain_label = bls[[bl.name for bl in bls].index(self.params["brain_label"])]

        method = self.params["method"]
        if method in ("MNE", "dSPM", "sLORETA", "eLORETA"):
            inverse_operator = read_inverse_operator(
                fname=self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif"
            )
        elif method == "LCMV":
            inverse_operator = make_lcmv(
                self.rec_info, self.fwd, self.noise_cov,
                reg=0.05, pick_ori="max-power",
                weight_norm="unit-noise-gain", rank=None,
            )
        else:
            raise ValueError(
                f"Unknown source method: {method!r}. "
                "Expected one of 'MNE', 'dSPM', 'sLORETA', 'eLORETA', 'LCMV'."
            )

        return {
            "fft_window": fft_window,
            "freq_band_idxs": freq_band_idxs,
            "brain_label": brain_label,
            "inverse_operator": inverse_operator,
            "method": method,
        }

    def _sensor_connectivity_prep(self) -> dict:
        ch_names = self.rec_info["ch_names"]
        chs = self.params["channels"]
        indices = tuple(
            np.array([ch_names.index(ch1), ch_names.index(ch2)])
            for ch1, ch2 in zip(chs[0], chs[1])
        )
        freqs = np.linspace(self.params["frange"][0], self.params["frange"][1], 6)
        return {
            "indices": indices,
            "freqs": freqs,
            "fmin": self.params["frange"][0],
            "fmax": self.params["frange"][1],
            "mode": self.params["mode"],
            "method": self.params["method"],
        }

    def _source_connectivity_prep(self) -> dict:
        lbl1, lbl2 = self.params["brain_label_1"], self.params["brain_label_2"]
        if not lbl1.endswith("-lh"):
            raise ValueError(f"brain_label_1 must end with '-lh', got {lbl1!r}.")
        if not lbl2.endswith("-rh"):
            raise ValueError(f"brain_label_2 must end with '-rh', got {lbl2!r}.")

        bls = read_labels_from_annot(
            subject=self.subject_fs_id,
            parc=self.params["atlas"],
            subjects_dir=self.subjects_fs_dir,
        )
        bl_names = [bl.name for bl in bls]
        merged_label = (
            bls[bl_names.index(lbl1)] + bls[bl_names.index(lbl2)]
        )
        inverse_operator = read_inverse_operator(
            fname=self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif"
        )
        freqs = np.linspace(self.params["frange"][0], self.params["frange"][1], 6)
        return {
            "merged_label": merged_label,
            "inverse_operator": inverse_operator,
            "freqs": freqs,
        }

    def _sensor_graph_prep(self) -> dict:
        ch_names = self.rec_info["ch_names"]
        chs = self.params["channels"]
        indices = tuple(
            np.array([ch_names.index(ch1), ch_names.index(ch2)])
            for ch1, ch2 in zip(chs[0], chs[1])
        )
        sos = butter_bandpass(
            self.params["frange"][0], self.params["frange"][1],
            self._sfreq, order=5,
        )
        return {
            "indices": indices,
            "sos": sos,
            "dist_type": self.params["dist_type"],
            "alpha": self.params["alpha"],
            "beta": self.params["beta"],
        }

    def _source_graph_prep(self) -> dict:
        bls = read_labels_from_annot(
            subject=self.subject_fs_id,
            parc=self.params["atlas"],
            subjects_dir=self.subjects_fs_dir,
        )
        bl_names = [bl.name for bl in bls]
        bl_idxs = (
            bl_names.index(self.params["brain_label_1"]),
            bl_names.index(self.params["brain_label_2"]),
        )
        inverse_operator = read_inverse_operator(
            fname=self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif"
        )
        sos = butter_bandpass(
            self.params["frange"][0], self.params["frange"][1],
            self._sfreq, order=5,
        )
        return {
            "bls": bls,
            "bl_idxs": bl_idxs,
            "inverse_operator": inverse_operator,
            "sos": sos,
        }

    def _cfc_sensor_prep(self) -> dict:
        comod = Comodulogram(
            fs=self._sfreq,
            low_fq_range=np.linspace(
                self.params["frange_1"][0], self.params["frange_1"][1], 5
            ),
            high_fq_range=np.linspace(
                self.params["frange_2"][0], self.params["frange_2"][1], 5
            ),
            method=self.params["method"],
            n_surrogates=0,
        )
        return {"comod": comod}

    # ------------------------------------------------------------------
    # Feature-extraction methods  (decorated with @timed)
    # ------------------------------------------------------------------

    @timed
    def _sensor_power(
        self,
        data: np.ndarray,
        sfreq: float,
        frange: tuple,
        method: str = "welch",
        relative: bool = False,
    ) -> float:
        """Mean band power across channels at sensor level."""
        bp = compute_bandpower(data, sfreq, frange, method=method, relative=relative)
        return float(bp.mean())

    @timed
    def _band_ratio(
        self,
        data: np.ndarray,
        sfreq: float,
        frange_1: tuple,
        frange_2: tuple,
        method: str = "welch",
    ) -> float:
        """Power ratio between two frequency bands."""
        bp1 = compute_bandpower(data, sfreq, tuple(frange_1), method=method, relative=False)
        bp2 = compute_bandpower(data, sfreq, tuple(frange_2), method=method, relative=False)
        return float(bp1.mean() / (bp2.mean() + 1e-30))

    @timed
    def _individual_peak_power(
        self,
        data: np.ndarray,
        sfreq: float,
        freq_var: float,
        cf: float,
    ) -> float:
        """Band power in a narrow window around the individual peak frequency."""
        bp = compute_bandpower(
            data, sfreq, (cf - freq_var, cf + freq_var),
            method="welch", relative=False,
        )
        return float(bp.mean())

    @timed
    def _entropy(
        self,
        data: np.ndarray,
        sos: np.ndarray,
        method: str,
        psd_method: Optional[str] = None,
    ) -> float:
        """Entropy of band-filtered M/EEG signals."""
        data_filt = sosfiltfilt(sos, data)
        if method == "AppEn":
            ents = compute_app_entropy(data_filt)
        elif method == "SampEn":
            ents = compute_samp_entropy(data_filt)
        elif method == "Spectral":
            ents = compute_spect_entropy(
                sfreq=self._sfreq, data=data_filt, psd_method=psd_method
            )
        elif method == "SVD":
            ents = compute_svd_entropy(data_filt)
        else:
            raise ValueError(
                f"Unknown entropy method: {method!r}. "
                "Expected one of 'AppEn', 'SampEn', 'Spectral', 'SVD'."
            )
        return float(ents.mean() - 2)

    @timed
    def _argmax_freq(
        self,
        data: np.ndarray,
        fft_window: np.ndarray,
        ap_model: np.ndarray,
        gaussian,
    ) -> float:
        """Individual peak frequency via aperiodic subtraction + Gaussian fit."""
        data_win = data * fft_window
        fftval = np.abs(np.fft.rfft(data_win, axis=1) / data.shape[-1])
        freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / self._sfreq)

        mask = (freqs >= self.params["frange"][0]) & (freqs <= self.params["frange"][1])
        freqs_band = freqs[mask]
        periodic_power = np.mean(np.square(fftval[:, mask]), axis=0) - ap_model

        p0 = [periodic_power.max(), freqs_band[np.argmax(periodic_power)], 1.0]
        try:
            popt, _ = curve_fit(gaussian, freqs_band, periodic_power, p0=p0)
            return float(popt[1])
        except RuntimeError:
            warn(
                "argmax_freq: Gaussian fit failed; returning 0 Hz.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0

    @timed
    def _source_power(
        self,
        data: np.ndarray,
        fft_window: np.ndarray,
        freq_band_idxs: np.ndarray,
        brain_label,
        inverse_operator,
        method: str,
    ) -> float:
        """Source-level band power in a brain label."""
        raw_data = self._prepare_raw_array(data)

        if method in ("MNE", "dSPM", "sLORETA", "eLORETA"):
            stc_data = apply_inverse_raw(
                raw_data, inverse_operator,
                lambda2=1.0 / 9, method=method,
                pick_ori="normal", label=brain_label,
            ).data
        else:
            stc_data = apply_lcmv_raw(raw_data, inverse_operator).data

        stc_data = stc_data * fft_window
        fft_val = np.abs(np.fft.rfft(stc_data, axis=1) / stc_data.shape[-1])
        return float(np.mean(np.square(fft_val[:, freq_band_idxs])))

    @timed
    def _sensor_connectivity(
        self,
        data: np.ndarray,
        indices: tuple,
        freqs: np.ndarray,
        fmin: float,
        fmax: float,
        mode: str,
        method: str,
    ) -> float:
        """Sensor-level spectral connectivity between channel pairs."""
        con = spectral_connectivity_time(
            data=data[np.newaxis, :],
            freqs=freqs,
            indices=indices,
            average=False,
            sfreq=self._sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            mode=mode,
            method=method,
            n_cycles=5,
        )
        return float(np.squeeze(con.get_data(output="dense"))[indices].mean())

    @timed
    def _source_connectivity(
        self,
        data: np.ndarray,
        merged_label,
        inverse_operator,
        freqs: np.ndarray,
    ) -> float:
        """Source-level connectivity between two brain labels."""
        raw_data = self._prepare_raw_array(data)
        stcs = apply_inverse_raw(
            raw_data, inverse_operator,
            lambda2=1.0 / 9, pick_ori="normal", label=merged_label,
        )
        con = spectral_connectivity_time(
            data=np.array([[stcs.lh_data.mean(axis=0), stcs.rh_data.mean(axis=0)]]),
            freqs=freqs,
            indices=None,
            average=False,
            sfreq=self._sfreq,
            fmin=self.params["frange"][0],
            fmax=self.params["frange"][1],
            faverage=True,
            mode=self.params["mode"],
            method=self.params["method"],
            n_cycles=5,
        )
        return float(np.squeeze(con.get_data(output="dense"))[1][0])

    @timed
    def _sensor_graph(
        self,
        data: np.ndarray,
        indices: tuple,
        sos: np.ndarray,
        dist_type: str,
        alpha: float,
        beta: float,
    ) -> float:
        """Graph-theoretic connectivity from sensor-space M/EEG."""
        data_filt = sosfiltfilt(sos, data)
        graph_matrix = log_degree_barrier(
            data_filt, dist_type=dist_type, alpha=alpha, beta=beta
        )
        return float(np.mean([graph_matrix[idxs] for idxs in indices]) - 0.025)

    @timed
    def _source_graph(
        self,
        data: np.ndarray,
        bls: list,
        bl_idxs: tuple,
        inverse_operator,
        sos: np.ndarray,
    ) -> float:
        """Graph-theoretic connectivity from source-space M/EEG."""
        raw_data = self._prepare_raw_array(data)
        stcs = apply_inverse_raw(
            raw_data, inverse_operator,
            lambda2=1.0 / 9, pick_ori="normal",
        )
        tcs = stcs.extract_label_time_course(
            bls,
            src=inverse_operator["src"],
            mode="mean_flip",
            allow_empty=True,
        )
        tcs_filt = sosfiltfilt(sos, tcs)
        graph_matrix = log_degree_barrier(
            tcs_filt,
            dist_type=self.params["dist_type"],
            alpha=self.params["alpha"],
            beta=self.params["beta"],
        )
        return float(graph_matrix[bl_idxs[0], bl_idxs[1]])

    @timed
    def _cfc_sensor(self, data: np.ndarray, comod) -> float:
        """Cross-frequency coupling (modulation index) at sensor level."""
        comod.fit(data)
        return float(comod.comod_.mean())

    # ------------------------------------------------------------------
    # ERD/ERS
    # ------------------------------------------------------------------

    def _erd_ers_prep(self) -> dict:
        if not hasattr(self, "raw_baseline") or self.raw_baseline is None:
            raise RuntimeError(
                "erd_ers requires a completed baseline recording. "
                "Call record_baseline() first."
            )
        baseline_power = compute_bandpower(
            self.raw_baseline.get_data(),
            sfreq=self._sfreq,
            band=tuple(self.params["frange"]),
            method=self.params["method"],
            relative=False,
        ).mean()
        return {
            "sfreq": self._sfreq,
            "frange": self.params["frange"],
            "method": self.params["method"],
            "baseline_power": float(baseline_power),
        }

    @timed
    def _erd_ers(
        self,
        data: np.ndarray,
        sfreq: float,
        frange: tuple,
        method: str,
        baseline_power: float,
    ) -> float:
        """Event-related desynchronisation / synchronisation (%).

        Positive values = synchronisation (ERS); negative = desynchronisation (ERD).
        """
        current_power = compute_bandpower(data, sfreq, tuple(frange), method=method, relative=False).mean()
        return float((current_power - baseline_power) / (baseline_power + 1e-300) * 100.0)

    # ------------------------------------------------------------------
    # Laterality
    # ------------------------------------------------------------------

    def _laterality_prep(self) -> dict:
        ch_names = self.rec_info["ch_names"]

        def _is_left(name: str) -> bool:
            # 10-20 convention: trailing odd digit → left hemisphere
            for i in range(len(name) - 1, -1, -1):
                if name[i].isdigit():
                    return int(name[i]) % 2 == 1
            return False

        def _is_right(name: str) -> bool:
            for i in range(len(name) - 1, -1, -1):
                if name[i].isdigit():
                    return int(name[i]) % 2 == 0
            return False

        lh_idx = [i for i, ch in enumerate(ch_names) if _is_left(ch)]
        rh_idx = [i for i, ch in enumerate(ch_names) if _is_right(ch)]

        if not lh_idx or not rh_idx:
            warn(
                "laterality: could not auto-detect left/right channels from names; "
                "splitting by index instead.",
                UserWarning,
                stacklevel=2,
            )
            mid = len(ch_names) // 2
            lh_idx = list(range(mid))
            rh_idx = list(range(mid, len(ch_names)))

        return {
            "sfreq": self._sfreq,
            "frange": self.params["frange"],
            "method": self.params["method"],
            "lh_idx": lh_idx,
            "rh_idx": rh_idx,
        }

    @timed
    def _laterality(
        self,
        data: np.ndarray,
        sfreq: float,
        frange: tuple,
        method: str,
        lh_idx: list,
        rh_idx: list,
    ) -> float:
        """Inter-hemispheric power asymmetry: log(P_right) − log(P_left).

        Positive → right dominance; negative → left dominance.
        """
        lh_power = compute_bandpower(data[lh_idx], sfreq, tuple(frange), method=method, relative=False).mean()
        rh_power = compute_bandpower(data[rh_idx], sfreq, tuple(frange), method=method, relative=False).mean()
        return float(np.log(rh_power + 1e-300) - np.log(lh_power + 1e-300))

    # ------------------------------------------------------------------
    # Hjorth parameters
    # ------------------------------------------------------------------

    def _hjorth_prep(self) -> dict:
        sos = butter_bandpass(
            self.params["frange"][0], self.params["frange"][1],
            self._sfreq, order=5,
        )
        return {"sos": sos}

    @timed
    def _hjorth(self, data: np.ndarray, sos: np.ndarray) -> float:
        """Mean of Hjorth mobility and complexity across channels.

        Mobility ≈ dominant frequency proxy; complexity ≈ signal irregularity.
        No FFT required — pure time-domain.
        """
        x = sosfiltfilt(sos, data)   # shape (n_ch, n_samples)
        dx = np.diff(x, axis=1)
        ddx = np.diff(dx, axis=1)

        var_x  = np.var(x,   axis=1) + 1e-300
        var_dx = np.var(dx,  axis=1) + 1e-300
        var_ddx = np.var(ddx, axis=1) + 1e-300

        mobility   = np.sqrt(var_dx / var_x)
        mobility_d = np.sqrt(var_ddx / var_dx)
        complexity = mobility_d / mobility

        return float(0.5 * (mobility.mean() + complexity.mean()))

    # ------------------------------------------------------------------
    # Spectral centroid
    # ------------------------------------------------------------------

    def _spectral_centroid_prep(self) -> dict:
        return {
            "sfreq": self._sfreq,
            "frange": self.params["frange"],
        }

    @timed
    def _spectral_centroid(
        self,
        data: np.ndarray,
        sfreq: float,
        frange: tuple,
    ) -> float:
        """Frequency-weighted centre-of-mass of the PSD within a band (Hz).

        High centroid → activity shifted towards the upper edge of the band
        (useful for tracking alpha-peak drift or SMR centre-frequency).
        """
        n_samples = data.shape[1]
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
        mask = (freqs >= frange[0]) & (freqs <= frange[1])
        freqs_band = freqs[mask]

        psd = np.abs(np.fft.rfft(data, axis=1)) ** 2   # shape (n_ch, n_freqs)
        psd_band = psd[:, mask]

        total = psd_band.sum(axis=1, keepdims=True) + 1e-300
        centroid_per_ch = (psd_band * freqs_band[np.newaxis, :]).sum(axis=1) / total.squeeze()
        return float(centroid_per_ch.mean())

    # ------------------------------------------------------------------
    # ERD/ERS laterality index
    # ------------------------------------------------------------------

    def _laterality_erd_ers_prep(self) -> dict:
        """Prep: detect hemispheric channel indices + compute baseline powers."""
        # requires raw_baseline
        if not hasattr(self, "raw_baseline") or self.raw_baseline is None:
            raise RuntimeError(
                "laterality_erd_ers requires a completed baseline recording. "
                "Call record_baseline() first."
            )
        ch_names = self.rec_info["ch_names"]

        def _is_left(name):
            for i in range(len(name)-1, -1, -1):
                if name[i].isdigit():
                    return int(name[i]) % 2 == 1
            return False
        def _is_right(name):
            for i in range(len(name)-1, -1, -1):
                if name[i].isdigit():
                    return int(name[i]) % 2 == 0
            return False

        lh_idx = [i for i, ch in enumerate(ch_names) if _is_left(ch)]
        rh_idx = [i for i, ch in enumerate(ch_names) if _is_right(ch)]
        if not lh_idx or not rh_idx:
            warn("laterality_erd_ers: hemispheric auto-detection failed; splitting by index.", UserWarning, stacklevel=2)
            mid = len(ch_names) // 2
            lh_idx = list(range(mid))
            rh_idx = list(range(mid, len(ch_names)))

        baseline_data = self.raw_baseline.get_data()
        frange = tuple(self.params["frange"])
        method = self.params["method"]
        baseline_lh = float(compute_bandpower(baseline_data[lh_idx], self._sfreq, frange, method=method, relative=False).mean())
        baseline_rh = float(compute_bandpower(baseline_data[rh_idx], self._sfreq, frange, method=method, relative=False).mean())

        return {
            "sfreq": self._sfreq,
            "frange": frange,
            "method": method,
            "lh_idx": lh_idx,
            "rh_idx": rh_idx,
            "baseline_lh": baseline_lh,
            "baseline_rh": baseline_rh,
        }

    @timed
    def _laterality_erd_ers(
        self,
        data: np.ndarray,
        sfreq: float,
        frange: tuple,
        method: str,
        lh_idx: list,
        rh_idx: list,
        baseline_lh: float,
        baseline_rh: float,
    ) -> float:
        """Baseline-normalised inter-hemispheric ERD/ERS asymmetry (%).

        Computes the ERD/ERS ratio for each hemisphere separately
        (normalised by its own baseline power) and returns the signed
        difference:

            feature = ERD_ERS_right − ERD_ERS_left

        * Positive → right hemisphere more activated (ERS) or less suppressed.
        * Negative → left hemisphere more activated (or right more suppressed).

        Motor imagery example: right-hand imagery produces left-hemisphere
        alpha ERD, so the feature becomes strongly negative during the task
        and recovers toward zero at rest.
        """
        lh_now = compute_bandpower(data[lh_idx], sfreq, frange, method=method, relative=False).mean()
        rh_now = compute_bandpower(data[rh_idx], sfreq, frange, method=method, relative=False).mean()
        erd_lh = (lh_now - baseline_lh) / (baseline_lh + 1e-300) * 100.0
        erd_rh = (rh_now - baseline_rh) / (baseline_rh + 1e-300) * 100.0
        return float(erd_rh - erd_lh)

    # ------------------------------------------------------------------
    # Slow Cortical Potentials (SCP)
    # ------------------------------------------------------------------

    def _scp_prep(self) -> dict:
        """Prep: build SOS low-pass (and optional high-pass) Butterworth filters."""
        sfreq = self.rec_info["sfreq"]
        lowpass = self.params["lowpass"]
        highpass = self.params.get("highpass", 0.0)
        reference = self.params.get("reference", "mean")

        nyq = sfreq / 2.0
        sos_lp = butter(4, lowpass / nyq, btype="low", output="sos")

        sos_hp = None
        if highpass > 0.0:
            sos_hp = butter(4, highpass / nyq, btype="high", output="sos")

        return {
            "sos_lp": sos_lp,
            "sos_hp": sos_hp,
            "reference": reference,
        }

    @timed
    def _scp(
        self,
        data: np.ndarray,
        sos_lp: np.ndarray,
        sos_hp,
        reference: str,
    ) -> float:
        """Slow Cortical Potential: mean amplitude of the DC-coupled slow signal.

        Applies a low-pass (and optional high-pass) zero-phase Butterworth
        filter to extract the slow envelope, then collapses channels via
        mean or median and returns the temporal mean of the resulting signal.

        Positive SCP → cortical deactivation; negative SCP → activation.
        """
        sig = data.copy()

        # Optional high-pass first (removes very slow drifts if DC not coupled)
        if sos_hp is not None:
            sig = sosfiltfilt(sos_hp, sig)

        # Low-pass to extract the slow cortical potential
        sig = sosfiltfilt(sos_lp, sig)

        # Collapse channels
        if reference == "median":
            channel_summary = np.median(sig, axis=0)   # shape: (n_samples,)
        else:
            channel_summary = np.mean(sig, axis=0)     # shape: (n_samples,)

        return float(channel_summary.mean())

    # ------------------------------------------------------------------
    # Peak Alpha Frequency (PAF) tracker
    # ------------------------------------------------------------------

    def _peak_alpha_freq_prep(self) -> dict:
        """Prep: initialise EMA state for the real-time PAF tracker."""
        sfreq = self.rec_info["sfreq"]
        frange = self.params["frange"]
        method = self.params.get("method", "welch")
        smoothing = self.params.get("smoothing", 0.85)

        # Compute initial PAF from baseline if available; else use band midpoint
        if hasattr(self, "raw_baseline") and self.raw_baseline is not None:
            baseline_data = self.raw_baseline.get_data()   # (n_ch, n_samples)
            mean_sig = baseline_data.mean(axis=0)           # (n_samples,)
            if method == "welch":
                freqs_bl, psd_bl = welch(mean_sig, fs=sfreq, nperseg=min(256, mean_sig.shape[-1]))
            else:
                n = mean_sig.shape[-1]
                fft_vals = np.abs(np.fft.rfft(mean_sig)) ** 2
                freqs_bl = np.fft.rfftfreq(n, d=1.0 / sfreq)
                psd_bl = fft_vals
            mask_bl = (freqs_bl >= frange[0]) & (freqs_bl <= frange[1])
            if mask_bl.any():
                initial_paf = float(freqs_bl[mask_bl][np.argmax(psd_bl[mask_bl])])
            else:
                initial_paf = float((frange[0] + frange[1]) / 2.0)
        else:
            initial_paf = float((frange[0] + frange[1]) / 2.0)

        return {
            "sfreq": float(sfreq),
            "frange": list(frange),
            "method": method,
            "smoothing": float(smoothing),
            "_paf_state": [initial_paf],   # mutable reference cell for EMA state
        }

    @timed
    def _peak_alpha_freq(
        self,
        data: np.ndarray,
        sfreq: float,
        frange: list,
        method: str,
        smoothing: float,
        _paf_state: list,
    ) -> float:
        """Real-time peak alpha frequency (PAF) with exponential smoothing.

        Computes the PSD of the current window (averaged across channels),
        finds the dominant peak within *frange*, and updates an exponential
        moving average (EMA) to suppress frame-to-frame jitter.

        Returns the EMA-smoothed PAF in Hz.
        """
        # Average across channels to get a single time series
        mean_sig = data.mean(axis=0)   # shape: (n_samples,)

        # Compute PSD
        if method == "welch":
            freqs, psd = welch(mean_sig, fs=sfreq, nperseg=min(256, mean_sig.shape[-1]))
        else:
            n = mean_sig.shape[-1]
            psd = np.abs(np.fft.rfft(mean_sig)) ** 2
            freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)

        # Find peak within frange
        mask = (freqs >= frange[0]) & (freqs <= frange[1])
        if mask.any():
            peak_freq = float(freqs[mask][np.argmax(psd[mask])])
        else:
            peak_freq = float(_paf_state[0])   # fallback: keep current estimate

        # EMA update — mutate the state cell so state persists across windows
        new_paf = (1.0 - smoothing) * peak_freq + smoothing * _paf_state[0]
        _paf_state[0] = new_paf

        return float(new_paf)

    # ------------------------------------------------------------------
    # Connectivity Ratio
    # ------------------------------------------------------------------

    def _connectivity_ratio_prep(self) -> dict:
        """Prep: build connectivity indices for numerator and denominator pairs."""
        ch_names = self.rec_info["ch_names"]

        def _pair_to_indices(pair):
            a, b = pair[0], pair[1]
            if a not in ch_names or b not in ch_names:
                missing = [c for c in [a, b] if c not in ch_names]
                raise ValueError(
                    f"connectivity_ratio: channels {missing} not found in recording. "
                    f"Available: {ch_names}"
                )
            return (np.array([ch_names.index(a)]), np.array([ch_names.index(b)]))

        indices_num = _pair_to_indices(self.params["channels_num"])
        indices_den = _pair_to_indices(self.params["channels_den"])

        freqs = np.linspace(self.params["frange"][0], self.params["frange"][1], 6)

        return {
            "indices_num": indices_num,
            "indices_den": indices_den,
            "freqs": freqs,
            "fmin": float(self.params["frange"][0]),
            "fmax": float(self.params["frange"][1]),
            "mode": self.params["mode"],
            "method": self.params["method"],
        }

    @timed
    def _connectivity_ratio(
        self,
        data: np.ndarray,
        indices_num: tuple,
        indices_den: tuple,
        freqs: np.ndarray,
        fmin: float,
        fmax: float,
        mode: str,
        method: str,
    ) -> float:
        """Ratio of functional connectivity between two channel pairs (or groups).

        Useful for laterality of connectivity, e.g. ipsilateral / contralateral.
        Returns conn_pair1 / conn_pair2.
        """
        # Numerator connectivity
        con_num = spectral_connectivity_time(
            data=data[np.newaxis, :],
            freqs=freqs,
            indices=indices_num,
            average=False,
            sfreq=self._sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            mode=mode,
            method=method,
            n_cycles=5,
        )
        conn_num = float(np.squeeze(con_num.get_data(output="dense"))[indices_num].mean())

        # Denominator connectivity
        con_den = spectral_connectivity_time(
            data=data[np.newaxis, :],
            freqs=freqs,
            indices=indices_den,
            average=False,
            sfreq=self._sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            mode=mode,
            method=method,
            n_cycles=5,
        )
        conn_den = float(np.squeeze(con_den.get_data(output="dense"))[indices_den].mean())

        return float(conn_num / (conn_den + 1e-30))
