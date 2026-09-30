"""ModalityMixin — real-time feature extraction for all modalities.

Mixed into :class:`RTStream`. Assumes the following attributes exist on
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
- ``decode``           — fitted :class:`~mne_rt.RTDecode` classifier queried
  once per window (``self.decoder``, set via
  :meth:`~mne_rt.RTStream.set_decoder`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from warnings import warn

import mne
import numpy as np
from mne import read_labels_from_annot
from mne.beamformer import apply_lcmv_raw, make_lcmv
from mne.minimum_norm import apply_inverse_raw, read_inverse_operator
from mne_connectivity import spectral_connectivity_time
from mne_features.univariate import (
    compute_app_entropy,
    compute_samp_entropy,
    compute_spect_entropy,
    compute_svd_entropy,
)
from pactools import Comodulogram
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt, welch

from mne_rt._logging import logger
from mne_rt.source import SourceModel, resolve_rois
from mne_rt.tools import (
    butter_bandpass,
    compute_bandpower,
    compute_fft,
    compute_instantaneous_phase,
    estimate_aperiodic_component,
    log_degree_barrier,
    resolve_connectivity_method,
    resolve_n_cycles,
    timed,
)

#: Inverse solutions the source-space modalities accept.
_SOURCE_METHODS = frozenset({"MNE", "dSPM", "sLORETA", "eLORETA", "LCMV"})

#: Connectivity metrics that are meaningless on a magnitude (phase-free) source
#: estimate, because they are defined in terms of relative phase.
_PHASE_SENSITIVE_METRICS = frozenset({"imcoh", "cohy", "plv", "ciplv", "pli", "wpli"})


def model_supports_phase(model) -> bool:
    """Whether a source model preserves phase.

    A free-orientation estimate combines three orientations by norm, giving a
    non-negative magnitude with no phase — so any phase-based connectivity
    measure computed on it is meaningless.  Fixed-orientation solutions (LCMV
    with ``pick_ori="max-power"``, or a surface minimum-norm estimate with
    ``pick_ori="normal"``) keep the signed time course.
    """
    return bool(model.supports_kernel)


def _connectivity_values(con, take_imag: bool) -> np.ndarray:
    """Per-pair connectivity values from a ``spectral_connectivity_time`` result.

    Uses the default *raveled* output rather than ``output="dense"``: the dense
    form allocates a real-valued matrix and silently discards the imaginary part
    (``ComplexWarning``), which would zero out every ``imcoh`` value.

    Parameters
    ----------
    con : instance of SpectralConnectivity
            Result of :func:`~mne_connectivity.spectral_connectivity_time`,
            computed with explicit ``indices`` and ``faverage=True``.
    take_imag : bool
            Take the imaginary part (imaginary coherence) rather than the value
            as returned.

    Returns
    -------
    values : ndarray, shape (n_pairs,)
            One real value per requested connection, averaged over epochs and
            the (already band-averaged) frequency axis.
    """
    values = np.asarray(con.get_data())  # (n_epochs, n_pairs, n_freqs)
    values = values.reshape(values.shape[0], values.shape[1], -1).mean(axis=(0, 2))
    if take_imag:
        return np.imag(values)
    if np.iscomplexobj(values):
        # e.g. "cohy" requested directly — report the real (co-coherence) part.
        return np.real(values)
    return values


class ModalityMixin:
    """Feature-extraction engine for all MNE-RT NF modalities.

    :class:`ModalityMixin` is mixed into :class:`~mne_rt.RTStream` and provides
    the **prep / compute** pair for every modality:

    * **Prep** (``_<modality>_prep()``) — runs *once* before the main loop.
      Returns a ``dict`` of pre-computed artefacts (filter coefficients, index
      arrays, connectivity indices, …) that are passed as keyword arguments to
      the compute step.
    * **Compute** (``_<modality>(data, **prep_kwargs)``) — runs *every window*
      inside a thread-pool worker.  Decorated with :func:`~ant.tools.timed` so
      it returns ``(value, elapsed_seconds)``.

    .. rubric:: Supported modalities (21 total)


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

    **Decoding**

    ``decode`` — single-trial classification via a fitted
    :class:`~mne_rt.RTDecode` instance (see
    :meth:`~mne_rt.RTStream.set_decoder`)

    Notes
    -----
    All methods are private (single-underscore prefix) and are called
    internally by :meth:`~mne_rt.RTStream.record_main`.  To extend MNE-RT with a
    custom modality, sub-class :class:`~mne_rt.RTStream` and add a matching
    ``_<name>_prep`` / ``_<name>`` pair following the same pattern.
    """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _acquired_ch_names(self) -> list:
        """Channels present in the array ``stream.get_data()`` returns.

        This is **not** ``rec_info["ch_names"]``: ``get_data()`` defaults to
        ``exclude="bads"``, so a channel marked bad — at connect time, or later
        by :class:`~mne_rt.BadChannelDetector` during the baseline — is absent
        from the array while still listed in the info.

        Everything that indexes the acquired array by position, or wraps it in
        a :class:`~mne.io.RawArray`, must use this list. Using
        ``rec_info["ch_names"]`` instead shifts every position once a channel is
        bad, which is what made :meth:`record_baseline` raise
        ``len(data) does not match len(info["ch_names"])`` and made the
        connectivity modalities silently compute on the wrong channel pairs.

        Bads are deliberately left in ``rec_info``: the forward model and
        beamformer exclude them by themselves, and
        :class:`~mne_rt.tools.RTMaxwellFilter` needs them marked in order to
        reconstruct them through the SSS expansion.
        """
        bads = set(self.rec_info["bads"])
        names = [ch for ch in self.rec_info["ch_names"] if ch not in bads]
        picks = getattr(self, "picks", None)
        if picks is None or isinstance(picks, str):
            # A channel-type string is resolved by the stream itself; the
            # ordering it returns still follows rec_info.
            return names
        wanted = set(picks)
        return [ch for ch in names if ch in wanted]

    def _get_source_model(self, *, method: str, atlas: str):
        """Return a :class:`~mne_rt.SourceModel`, building it at most once.

        Memoised on ``(method, atlas)`` so that several source modalities — for
        example the same connectivity measure in two frequency bands — share one
        forward model and one beamformer instead of each paying to build their own.
        """
        cache = getattr(self, "_source_models", None)
        if cache is None:
            cache = self._source_models = {}
        key = (str(method), str(atlas))
        if key not in cache:
            # `from_stream` needs `src`/`fwd`/`data_cov`, which record_baseline no
            # longer builds. Note compute_inv_operator() clears this cache, so it
            # has to run before the entry is written, not after.
            self._ensure_head_model(f"Source modality {method!r}")
            cache = self._source_models
            cache[key] = SourceModel.from_stream(self, method=method, atlas=atlas)
            logger.info("Built %r for method=%s atlas=%s", cache[key], method, atlas)
        return cache[key]

    def _get_roi_kernel(self, model, rois, mri_resolution: bool):
        """Return the sensor → ROI operator, building it at most once.

        Several instances of one modality — the same ROI pairs in three
        frequency bands, say — ask for an identical kernel, and building it is
        the expensive part of source preparation for a volume atlas at
        ``mri_resolution=True``.

        The model is keyed by identity rather than by ``id()``: the cache holds
        a reference, so an entry cannot outlive its model and be matched by a
        later object that happens to reuse the address. :class:`~mne_rt.ROI` is
        a frozen dataclass, so keying on the ROIs themselves distinguishes
        same-named regions defined over different labels.
        """
        cache = getattr(self, "_roi_kernels", None)
        if cache is None:
            cache = self._roi_kernels = {}
        key = (model, tuple(rois), bool(mri_resolution))
        if key not in cache:
            cache[key] = model.roi_kernel(rois, mri_resolution=mri_resolution)
        return cache[key]

    def _roi_names(self, *legacy_keys: str) -> list:
        """ROI spec from ``rois``, falling back to the deprecated label keys."""
        rois = self.params.get("rois")
        if rois:
            return list(rois) if not isinstance(rois, (str, dict)) else [rois]

        legacy = [self.params[key] for key in legacy_keys if self.params.get(key)]
        if not legacy:
            raise ValueError(
                "No ROIs configured. Set 'rois' for this modality, e.g. "
                "rois: ['Broca', 'Wernicke']. Use mne_rt.list_rois() to see what "
                "the chosen atlas provides."
            )
        warn(
            f"{legacy_keys} is deprecated; use 'rois' (and 'pairs' where a modality "
            "needs specific ROI pairs) instead. Support will be removed in a future "
            "release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return legacy

    def _source_prep_common(self, *, inverse_method: str) -> dict:
        """Resolve ROIs and build the cached sensor → ROI operator."""
        # `atlas: null` in the config means "use the session's", which is what
        # makes RTStream(source_space="volume") pick a volumetric atlas without
        # every modality having to restate it.
        atlas = self.params.get("atlas") or getattr(self, "source_atlas", None) or "aparc"
        model = self._get_source_model(method=inverse_method, atlas=atlas)
        rois = resolve_rois(
            self._roi_names("brain_label", "brain_label_1", "brain_label_2"),
            atlas=atlas,
            subject=self.subject_fs_id,
            subjects_dir=self.subjects_fs_dir,
        )
        mri_resolution = bool(self.params.get("mri_resolution", True))

        if model.supports_kernel:
            return {
                "model": model,
                "rois": rois,
                "kernel": self._get_roi_kernel(model, rois, mri_resolution),
                # The operator's columns follow the forward model's channels, which
                # exclude bads and any channel without a digitised position, so the
                # data must be selected/reordered to match.
                "ch_picks": model.channel_picks(self._acquired_ch_names()),
                "label_operator": None,
            }

        warn(
            f"{model!r} cannot use the cached-kernel path, so every window will run a "
            "full source estimate — roughly two orders of magnitude slower, and likely "
            "too slow to keep up with the acquisition loop. Use method='LCMV', or a "
            "surface source space with a minimum-norm method.",
            RuntimeWarning,
            stacklevel=3,
        )
        return {
            "model": model,
            "rois": rois,
            "kernel": None,
            "ch_picks": None,
            "label_operator": model.label_operator(rois, mri_resolution=mri_resolution),
        }

    @staticmethod
    def _roi_time_courses(model, data, kernel, ch_picks, label_operator) -> np.ndarray:
        """ROI time courses ``(n_roi, n_times)`` for one analysis window."""
        return model.apply(data, kernel=kernel, ch_picks=ch_picks, label_operator=label_operator)

    def _ensure_head_model(self, reason: str, *, need_inverse: bool = False) -> None:
        """Build the forward/inverse model if nothing has built it yet.

        :meth:`~mne_rt.RTStream.record_baseline` used to do this
        unconditionally, which meant every baseline — including sensor-space
        ones that never look at a source — fetched the ``fsaverage`` anatomy
        over the network. It is built here instead, on the first request that
        actually needs it.

        Parameters
        ----------
        reason : str
            What asked for it, quoted in the error when no baseline exists.
        need_inverse : bool, default False
            Require a minimum-norm inverse operator as well as the forward
            model.  A beamformer needs only the latter, and
            ``compute_inv_operator(make_inverse=False)`` deliberately produces
            no inverse, so callers that dereference :attr:`inv` have to say so
            — otherwise this returns satisfied and they fail later with nothing
            to explain it.
        """
        _have_src = getattr(self, "src", None) is not None
        _have_inv = getattr(self, "inv", None) is not None
        if _have_src and (_have_inv or not need_inverse):
            return
        if getattr(self, "raw_baseline", None) is None:
            raise RuntimeError(
                f"{reason} needs a head model, which needs a baseline recording. "
                "Run record_baseline() first, or call compute_inv_operator() "
                "yourself if you have a baseline from elsewhere."
            )
        logger.info("Building the head model for %s …", reason)
        self.compute_inv_operator(make_inverse=True)

    def _get_inverse_operator(self):
        """Return the fitted inverse operator, from memory or from disk.

        :meth:`~mne_rt.RTStream.compute_inv_operator` leaves the operator on the
        instance and also writes it under ``inv/``; prefer the in-memory copy and
        fall back to the file so a session restarted against an existing subject
        directory still works.
        """
        inv = getattr(self, "inv", None)
        if inv is not None:
            return inv

        stem = f"sub-{self.subject_id}_ses-{self.session}_task-baseline"
        fname = self.subject_dir / "inv" / f"{stem}_inv.fif"
        if Path(fname).is_file():
            return read_inverse_operator(fname=fname)

        # Nothing on disk either, so build it now -- raising here would be a
        # failure the caller can do nothing about except run the very step this
        # method exists to perform.
        self._ensure_head_model("A source-space modality", need_inverse=True)
        inv = getattr(self, "inv", None)
        if inv is None:
            raise RuntimeError(
                "No inverse operator available for a source-space modality, and "
                "building one produced none. Call compute_inv_operator() with "
                f"make_inverse=True. Looked for {fname}."
            )
        return inv

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
            p[0] for p in peak_params_ if self.params["frange"][0] < p[0] < self.params["frange"][1]
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
            self.params["frange"][0],
            self.params["frange"][1],
            self._sfreq,
            order=5,
        )
        return {
            "sos": sos,
            "method": self.params["method"],
            "psd_method": self.params["psd_method"],
        }

    def _argmax_freq_prep(self) -> dict:
        if not hasattr(self, "raw_baseline"):
            raise RuntimeError("Baseline recording must be completed before using 'argmax_freq'.")
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
            return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

        return {
            "fft_window": fft_window,
            "ap_model": ap_model,
            "gaussian": _gaussian,
            "mask": mask,
            "freqs_band": freqs_band,
        }

    def _source_power_prep(self) -> dict:
        method = self.params["method"]
        if method not in _SOURCE_METHODS:
            raise ValueError(
                f"Unknown source method: {method!r}. Expected one of {sorted(_SOURCE_METHODS)}."
            )
        fft_window, _, freq_band_idxs, _ = compute_fft(
            sfreq=self._sfreq,
            winsize=self.winsize,
            freq_range=self.params["frange"],
        )
        return {
            **self._source_prep_common(inverse_method=method),
            "fft_window": fft_window,
            "freq_band_idxs": freq_band_idxs,
        }

    def _sensor_connectivity_prep(self) -> dict:
        ch_names = self._acquired_ch_names()  # data excludes bads; rec_info does not
        chs = self.params["channels"]
        # `channels` is a list of [ch_A, ch_B] pairs; build mne-connectivity's
        # (seeds, targets) form.  The previous zip(chs[0], chs[1]) happened to
        # give the same answer for exactly two pairs, but raised IndexError for
        # one pair and silently ignored the third onwards.
        missing = sorted({c for pair in chs for c in pair} - set(ch_names))
        if missing:
            raise ValueError(
                f"sensor_connectivity: channels {missing} not found in recording. "
                f"Available: {ch_names}"
            )
        seeds = np.array([ch_names.index(pair[0]) for pair in chs])
        targets = np.array([ch_names.index(pair[1]) for pair in chs])
        indices = (seeds, targets)
        fmin, fmax = self.params["frange"]
        freqs = np.linspace(fmin, fmax, 6)
        method, take_imag = resolve_connectivity_method(self.params["method"])
        n_cycles = resolve_n_cycles(freqs, self.params.get("n_cycles", 5), self.winsize)
        return {
            "indices": indices,
            "freqs": freqs,
            "fmin": fmin,
            "fmax": fmax,
            "mode": self.params["mode"],
            "method": method,
            "take_imag": take_imag,
            "n_cycles": n_cycles,
        }

    def _source_connectivity_prep(self) -> dict:
        inverse_method = self.params.get("inverse_method", "dSPM")
        if inverse_method not in _SOURCE_METHODS:
            raise ValueError(
                f"Unknown inverse_method: {inverse_method!r}. "
                f"Expected one of {sorted(_SOURCE_METHODS)}."
            )
        common = self._source_prep_common(inverse_method=inverse_method)
        rois = common["rois"]
        roi_names = [roi.name for roi in rois]

        pairs = self.params.get("pairs")
        if not pairs:
            if len(roi_names) != 2:
                raise ValueError(
                    f"source_connectivity needs 'pairs' when more than two ROIs are "
                    f"configured; got {len(roi_names)} ROIs {roi_names} and no pairs. "
                    'Example: pairs: [["Broca", "Wernicke"]]'
                )
            pairs = [roi_names]  # two ROIs, one obvious connection

        seeds, targets = [], []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Each entry of 'pairs' must name two ROIs, got {pair!r}.")
            for name, bucket in zip(pair, (seeds, targets)):
                if name not in roi_names:
                    raise ValueError(
                        f"Pair member {name!r} is not among the configured ROIs {roi_names}."
                    )
                bucket.append(roi_names.index(name))
        indices = (np.array(seeds), np.array(targets))

        if not model_supports_phase(common["model"]):
            method_name = self.params["method"]
            if method_name in _PHASE_SENSITIVE_METRICS:
                raise ValueError(
                    f"Connectivity metric {method_name!r} depends on phase, but "
                    f"{common['model']!r} produces a magnitude (a free-orientation "
                    "estimate combines three orientations by norm). Use "
                    "inverse_method='LCMV', or a surface source space."
                )

        fmin, fmax = self.params["frange"]
        freqs = np.linspace(fmin, fmax, int(self.params.get("n_freqs", 6)))
        method, take_imag = resolve_connectivity_method(self.params["method"])
        n_cycles = resolve_n_cycles(freqs, self.params.get("n_cycles", 5), self.winsize)
        return {
            **common,
            "indices": indices,
            "freqs": freqs,
            "fmin": fmin,
            "fmax": fmax,
            "mode": self.params["mode"],
            "method": method,
            "take_imag": take_imag,
            "n_cycles": n_cycles,
            "signed": bool(self.params.get("signed", False)),
        }

    def _sensor_graph_prep(self) -> dict:
        ch_names = self._acquired_ch_names()  # data excludes bads; rec_info does not
        chs = self.params["channels"]
        indices = tuple(
            np.array([ch_names.index(ch1), ch_names.index(ch2)]) for ch1, ch2 in zip(chs[0], chs[1])
        )
        sos = butter_bandpass(
            self.params["frange"][0],
            self.params["frange"][1],
            self._sfreq,
            order=5,
        )
        return {
            "indices": indices,
            "sos": sos,
            "dist_type": self.params["dist_type"],
            "alpha": self.params["alpha"],
            "beta": self.params["beta"],
        }

    def _source_graph_prep(self) -> dict:
        inverse_method = self.params.get("inverse_method", "dSPM")
        if inverse_method not in _SOURCE_METHODS:
            raise ValueError(
                f"Unknown inverse_method: {inverse_method!r}. "
                f"Expected one of {sorted(_SOURCE_METHODS)}."
            )
        common = self._source_prep_common(inverse_method=inverse_method)
        roi_names = [roi.name for roi in common["rois"]]
        if len(roi_names) < 2:
            raise ValueError(
                f"source_graph learns a graph over ROIs and needs at least two; got {roi_names}."
            )

        pair = self.params.get("pair")
        if pair is None:
            pairs = self.params.get("pairs")
            pair = pairs[0] if pairs else roi_names[:2]
        missing = [name for name in pair if name not in roi_names]
        if missing:
            raise ValueError(f"Edge member(s) {missing} are not among the ROIs {roi_names}.")
        edge = (roi_names.index(pair[0]), roi_names.index(pair[1]))

        sos = butter_bandpass(
            self.params["frange"][0],
            self.params["frange"][1],
            self._sfreq,
            order=5,
        )
        return {
            **common,
            "edge": edge,
            "sos": sos,
            "dist_type": self.params["dist_type"],
            "alpha": self.params["alpha"],
            "beta": self.params["beta"],
        }

    def _cfc_sensor_prep(self) -> dict:
        comod = Comodulogram(
            fs=self._sfreq,
            low_fq_range=np.linspace(self.params["frange_1"][0], self.params["frange_1"][1], 5),
            high_fq_range=np.linspace(self.params["frange_2"][0], self.params["frange_2"][1], 5),
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
            data,
            sfreq,
            (cf - freq_var, cf + freq_var),
            method="welch",
            relative=False,
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
            ents = compute_spect_entropy(sfreq=self._sfreq, data=data_filt, psd_method=psd_method)
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
        mask: np.ndarray,
        freqs_band: np.ndarray,
    ) -> float:
        """Individual peak frequency via aperiodic subtraction + Gaussian fit."""
        data_win = data * fft_window
        fftval = np.abs(np.fft.rfft(data_win, axis=1) / data.shape[-1])
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
        model,
        rois: list,
        kernel,
        ch_picks,
        label_operator,
        fft_window: np.ndarray,
        freq_band_idxs: np.ndarray,
    ) -> float:
        """Band power averaged over the configured source-space ROIs."""
        tcs = self._roi_time_courses(model, data, kernel, ch_picks, label_operator)
        tcs = tcs * fft_window
        fft_val = np.abs(np.fft.rfft(tcs, axis=1) / tcs.shape[-1])
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
        take_imag: bool,
        n_cycles: np.ndarray,
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
            n_cycles=n_cycles,
        )
        return float(_connectivity_values(con, take_imag).mean())

    @timed
    def _source_connectivity(
        self,
        data: np.ndarray,
        model,
        rois: list,
        kernel,
        ch_picks,
        label_operator,
        indices: tuple,
        freqs: np.ndarray,
        fmin: float,
        fmax: float,
        mode: str,
        method: str,
        take_imag: bool,
        n_cycles: np.ndarray,
        signed: bool,
    ) -> float:
        """Connectivity between the configured source-space ROI pairs."""
        tcs = self._roi_time_courses(model, data, kernel, ch_picks, label_operator)
        con = spectral_connectivity_time(
            data=tcs[np.newaxis],
            freqs=freqs,
            indices=indices,
            average=False,
            sfreq=self._sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            mode=mode,
            method=method,
            n_cycles=n_cycles,
        )
        values = _connectivity_values(con, take_imag)
        if not signed:
            # imcoh and friends are signed: the sign encodes which region leads.
            # For a reward signal the magnitude is usually what is wanted.
            values = np.abs(values)
        return float(values.mean())

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
        graph_matrix = log_degree_barrier(data_filt, dist_type=dist_type, alpha=alpha, beta=beta)
        return float(np.mean([graph_matrix[idxs] for idxs in indices]) - 0.025)

    @timed
    def _source_graph(
        self,
        data: np.ndarray,
        model,
        rois: list,
        kernel,
        ch_picks,
        label_operator,
        edge: tuple,
        sos: np.ndarray,
        dist_type: str,
        alpha: float,
        beta: float,
    ) -> float:
        """Learned graph edge weight between two source-space ROIs."""
        tcs = self._roi_time_courses(model, data, kernel, ch_picks, label_operator)
        tcs_filt = sosfiltfilt(sos, tcs)
        graph_matrix = log_degree_barrier(
            tcs_filt,
            dist_type=dist_type,
            alpha=alpha,
            beta=beta,
        )
        return float(graph_matrix[edge[0], edge[1]])

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
                "erd_ers requires a completed baseline recording. Call record_baseline() first."
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
        current_power = compute_bandpower(
            data, sfreq, tuple(frange), method=method, relative=False
        ).mean()
        return float((current_power - baseline_power) / (baseline_power + 1e-300) * 100.0)

    # ------------------------------------------------------------------
    # Laterality
    # ------------------------------------------------------------------

    def _laterality_prep(self) -> dict:
        ch_names = self._acquired_ch_names()  # data excludes bads; rec_info does not

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
        lh_power = compute_bandpower(
            data[lh_idx], sfreq, tuple(frange), method=method, relative=False
        ).mean()
        rh_power = compute_bandpower(
            data[rh_idx], sfreq, tuple(frange), method=method, relative=False
        ).mean()
        return float(np.log(rh_power + 1e-300) - np.log(lh_power + 1e-300))

    # ------------------------------------------------------------------
    # Hjorth parameters
    # ------------------------------------------------------------------

    def _hjorth_prep(self) -> dict:
        sos = butter_bandpass(
            self.params["frange"][0],
            self.params["frange"][1],
            self._sfreq,
            order=5,
        )
        return {"sos": sos}

    @timed
    def _hjorth(self, data: np.ndarray, sos: np.ndarray) -> float:
        """Mean of Hjorth mobility and complexity across channels.

        Mobility ≈ dominant frequency proxy; complexity ≈ signal irregularity.
        No FFT required — pure time-domain.
        """
        x = sosfiltfilt(sos, data)  # shape (n_ch, n_samples)
        dx = np.diff(x, axis=1)
        ddx = np.diff(dx, axis=1)

        var_x = np.var(x, axis=1) + 1e-300
        var_dx = np.var(dx, axis=1) + 1e-300
        var_ddx = np.var(ddx, axis=1) + 1e-300

        mobility = np.sqrt(var_dx / var_x)
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

        psd = np.abs(np.fft.rfft(data, axis=1)) ** 2  # shape (n_ch, n_freqs)
        psd_band = psd[:, mask]

        total = psd_band.sum(axis=1, keepdims=True) + 1e-300
        centroid_per_ch = (psd_band * freqs_band[np.newaxis, :]).sum(axis=1) / total.squeeze()
        return float(centroid_per_ch.mean())

    # ------------------------------------------------------------------
    # Instantaneous phase
    # ------------------------------------------------------------------

    def _instantaneous_phase_prep(self) -> dict:
        return {
            "sfreq": self._sfreq,
            "frange": self.params["frange"],
            "channel_indices": self.params.get("channel_indices"),
        }

    @timed
    def _instantaneous_phase(
        self,
        data: np.ndarray,
        sfreq: float,
        frange: tuple,
        channel_indices,
    ) -> float:
        """Instantaneous phase of the analytic signal at the window's last sample.

        Returned in radians in ``[-pi, pi]``, for phase-triggered stimulation and
        phase-alignment protocols. The amplitude envelope that
        :func:`~mne_rt.tools.compute_instantaneous_phase` also returns is
        discarded here, since a modality yields one value per window; gate on
        ``sensor_power`` in the same band if you need an amplitude criterion.
        """
        phase, _amplitude = compute_instantaneous_phase(
            data, sfreq, frange, channel_indices=channel_indices
        )
        return float(phase)

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
        ch_names = self._acquired_ch_names()  # data excludes bads; rec_info does not

        def _is_left(name):
            for i in range(len(name) - 1, -1, -1):
                if name[i].isdigit():
                    return int(name[i]) % 2 == 1
            return False

        def _is_right(name):
            for i in range(len(name) - 1, -1, -1):
                if name[i].isdigit():
                    return int(name[i]) % 2 == 0
            return False

        lh_idx = [i for i, ch in enumerate(ch_names) if _is_left(ch)]
        rh_idx = [i for i, ch in enumerate(ch_names) if _is_right(ch)]
        if not lh_idx or not rh_idx:
            warn(
                "laterality_erd_ers: hemispheric auto-detection failed; splitting by index.",
                UserWarning,
                stacklevel=2,
            )
            mid = len(ch_names) // 2
            lh_idx = list(range(mid))
            rh_idx = list(range(mid, len(ch_names)))

        baseline_data = self.raw_baseline.get_data()
        frange = tuple(self.params["frange"])
        method = self.params["method"]
        baseline_lh = float(
            compute_bandpower(
                baseline_data[lh_idx], self._sfreq, frange, method=method, relative=False
            ).mean()
        )
        baseline_rh = float(
            compute_bandpower(
                baseline_data[rh_idx], self._sfreq, frange, method=method, relative=False
            ).mean()
        )

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
        lh_now = compute_bandpower(
            data[lh_idx], sfreq, frange, method=method, relative=False
        ).mean()
        rh_now = compute_bandpower(
            data[rh_idx], sfreq, frange, method=method, relative=False
        ).mean()
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
            channel_summary = np.median(sig, axis=0)  # shape: (n_samples,)
        else:
            channel_summary = np.mean(sig, axis=0)  # shape: (n_samples,)

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
            baseline_data = self.raw_baseline.get_data()  # (n_ch, n_samples)
            mean_sig = baseline_data.mean(axis=0)  # (n_samples,)
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
            "_paf_state": [initial_paf],  # mutable reference cell for EMA state
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
        mean_sig = data.mean(axis=0)  # shape: (n_samples,)

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
            peak_freq = float(_paf_state[0])  # fallback: keep current estimate

        # EMA update — mutate the state cell so state persists across windows
        new_paf = (1.0 - smoothing) * peak_freq + smoothing * _paf_state[0]
        _paf_state[0] = new_paf

        return float(new_paf)

    # ------------------------------------------------------------------
    # Connectivity Ratio
    # ------------------------------------------------------------------

    def _connectivity_ratio_prep(self) -> dict:
        """Prep: build connectivity indices for numerator and denominator pairs."""
        ch_names = self._acquired_ch_names()  # data excludes bads; rec_info does not

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
        method, take_imag = resolve_connectivity_method(self.params["method"])
        n_cycles = resolve_n_cycles(freqs, self.params.get("n_cycles", 5), self.winsize)

        return {
            "indices_num": indices_num,
            "indices_den": indices_den,
            "freqs": freqs,
            "fmin": float(self.params["frange"][0]),
            "fmax": float(self.params["frange"][1]),
            "mode": self.params["mode"],
            "method": method,
            "take_imag": take_imag,
            "n_cycles": n_cycles,
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
        take_imag: bool,
        n_cycles: np.ndarray,
    ) -> float:
        """Ratio of functional connectivity between two channel pairs (or groups).

        Useful for laterality of connectivity, e.g. ipsilateral / contralateral.
        Returns conn_pair1 / conn_pair2.
        """

        def _connectivity(indices):
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
                n_cycles=n_cycles,
            )
            return float(_connectivity_values(con, take_imag).mean())

        conn_num = _connectivity(indices_num)
        conn_den = _connectivity(indices_den)

        return float(conn_num / (conn_den + 1e-30))

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_prep(self) -> dict:
        decoder = getattr(self, "decoder", None)
        if decoder is None:
            raise RuntimeError(
                "The 'decode' modality requires a fitted RTDecode instance. "
                "Call set_decoder(RTDecode(...).fit(X, y)) before record_main()."
            )
        if not decoder.fitted:
            raise RuntimeError(
                "The RTDecode instance passed to set_decoder() must be "
                "fit() on calibration epochs before record_main()."
            )
        if not decoder._supports_proba:
            raise AttributeError(
                f"The 'decode' modality requires predict_proba(); the estimator "
                f"{type(decoder.estimator).__name__} attached via set_decoder() "
                "does not implement it. Pass an estimator that does, e.g. "
                "LogisticRegression or SVC(probability=True)."
            )

        class_index = self.params["class_index"]
        n_classes = len(decoder.classes_)
        if not (0 <= class_index < n_classes):
            raise ValueError(
                f"`class_index={class_index}` is out of range for the fitted "
                f"decoder, which has {n_classes} classes "
                f"(decoder.classes_={list(decoder.classes_)!r})."
            )

        picks = getattr(self, "picks", None)
        n_expected = len(picks) if picks is not None else len(self.rec_info["ch_names"])
        if decoder.n_channels_ != n_expected:
            raise ValueError(
                f"The fitted decoder expects {decoder.n_channels_} channels but "
                f"the current session provides {n_expected} (picks={picks!r}). "
                "Re-fit RTDecode on calibration data matching this session's "
                "channel selection before record_main()."
            )

        return {"decoder": decoder, "class_index": class_index}

    @timed
    def _decode(self, data: np.ndarray, decoder, class_index: int) -> float:
        """Probability of ``decoder.classes_[class_index]`` for one window."""
        return float(decoder.predict_proba(data)[class_index])
