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
from scipy.signal import sosfiltfilt
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
    """Feature-extraction prep and compute methods for every NF modality."""

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
