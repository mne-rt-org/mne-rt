"""Core session class for the MNE-RT.

This module provides :class:`RTStream`, the top-level object that
orchestrates LSL streaming, artifact rejection, feature extraction, and
real-time M/EEG signal processing, feature extraction, and visualisation.

Typical workflow
----------------
::

    nf = RTStream(subject_id="sub01", session="01",
                    subjects_dir="/data/subjects", montage="easycap-M1")
    nf.connect_to_lsl()
    nf.record_baseline(baseline_duration=120)
    nf.record_main(duration=600, modality=["sensor_power", "erd_ers"])

Classes
-------
RTStream
    Main session controller — inherits all feature-extraction methods from
    :class:`~mne_rt.modalities.ModalityMixin`.
"""
from __future__ import annotations

import datetime
import json
import queue as _queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Optional, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from pyqtgraph.Qt import QtWidgets

import mne
from mne import (
    Report,
    write_cov,
    write_forward_solution,
)
from mne.channels import get_builtin_montages, read_dig_captrak
from mne.io import RawArray
from mne.minimum_norm import (
    apply_inverse_raw,
    write_inverse_operator,
)
from mne_lsl.lsl import local_clock
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream

from mne_rt.modalities import ModalityMixin
from mne_rt._logging import logger, set_log_level, verbose
from mne_rt.tools import (
    _compute_inv_operator,
    create_blink_template,
    get_params,
    plot_glass_brain,
    remove_blinks_lms,
)
from mne_rt.tools.asr import ASRDenoiser
from mne_rt.tools.gedai import GEDAIDenoiser
from mne_rt.tools.maxwell import RTMaxwellFilter
from mne_rt.tools.orica import ORICA
from mne_rt.viz import BrainPlot, NFPlot, RawPlot, TopomapPlot

# Package root — resolves correctly both in editable installs and installed wheels
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent  # src/ant → src → repo root


def _make_demo_raw_fif(
    sfreq: float = 256.0,
    duration: float = 300.0,
    n_channels: int = 64,
) -> Path:
    """Generate a synthetic 64-ch EEG FIF file for mock LSL streaming.

    Used as the default mock source when the bundled BrainVision sample file
    is not present (i.e. non-editable installs).  The file is written to a
    temporary directory and its path returned.
    """
    import tempfile

    import mne

    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(
        ch_names=montage.ch_names[:n_channels],
        sfreq=sfreq,
        ch_types="eeg",
    )
    info.set_montage(montage)

    rng = np.random.default_rng(42)
    n_samples = int(duration * sfreq)
    t = np.arange(n_samples) / sfreq
    # Alpha (~10 Hz) + broadband noise
    alpha = 1e-6 * np.sin(2 * np.pi * 10.0 * t)
    data = rng.standard_normal((n_channels, n_samples)) * 5e-7 + alpha

    raw = mne.io.RawArray(data, info, verbose=False)
    tmp_dir = Path(tempfile.mkdtemp(prefix="ant_mock_"))
    fif_path = tmp_dir / "demo-raw.fif"
    raw.save(fif_path, overwrite=True, verbose=False)
    return fif_path


class RTStream(ModalityMixin):
    """Real-time Real-time M/EEG session controller.

    Orchestrates LSL streaming, optional artifact rejection, parallel
    feature extraction, and real-time visualisation for a complete
    neurofeedback session.  Inherits all feature-extraction methods from
    :class:`~mne_rt.modalities.ModalityMixin`.

    Parameters
    ----------
    subject_id : str
        Unique subject identifier (non-empty string).  Used as the BIDS
        subject label (e.g. ``"sub01"`` → folder ``sub-sub01/``).
    session : str
        BIDS session label (e.g. ``"01"``, ``"pre"``, ``"week1"``).  Used
        to name output files and directories following the BIDS convention
        ``sub-<ID>_ses-<session>_task-<task>``.
    subjects_dir : str
        Root directory that holds one sub-folder per subject.
    montage : str | None
        EEG montage — a MNE built-in name (e.g. ``"easycap-M1"``), a
        path to a ``.bvct`` CapTrak file, or ``None`` for MEG.
    data_type : {"eeg", "meg"}, default "eeg"
        Recording modality.  Controls channel selection and forward model.
    mri : bool, default False
        If ``True``, individual MRI anatomy is used for source localisation
        instead of the ``fsaverage`` template.
    subject_fs_id : str, default "fsaverage"
        FreeSurfer subject identifier.  Use ``"fsaverage"`` for template-
        based source localisation.
    subjects_fs_dir : str | None, default None
        FreeSurfer subjects directory.  Required when
        ``subject_fs_id != "fsaverage"`` or when ``show_brain_activation``
        is requested.
    bandpass_freq : tuple(float, float) | None, default None
        Online band-pass filter applied to the LSL stream before feature
        extraction, as ``(l_freq, h_freq)`` in Hz.  ``None`` disables
        band-pass filtering.  Example: ``(1.0, 40.0)`` for a standard
        EEG band-pass.
    notch_freq : float | list[float] | None, default None
        One or more frequencies (Hz) to suppress with an IIR notch filter.
        ``None`` disables notch filtering.  Example: ``50`` or ``[50, 100]``
        for 50 Hz power-line interference and its harmonic.
    artifact_correction : {False, "lms", "orica", "gedai", "asr", "maxwell"}, default False
        Real-time artifact correction strategy applied sample-by-sample
        inside the acquisition loop.

        * ``False``       — no correction
        * ``"lms"``       — adaptive LMS regression on a frontal reference
          channel; fast and lightweight (EEG only).
          (:class:`~mne_rt.tools.AdaptiveLMSFilter`)
        * ``"orica"``     — online recursive ICA; blind source separation
          with continuous weight updates.
          (:class:`~mne_rt.tools.ORICA`)
        * ``"gedai"``     — generalised eigendecomposition artefact
          isolation; call :meth:`fit_gedai` after :meth:`record_baseline`.
          (:class:`~mne_rt.tools.GEDAIDenoiser`)
        * ``"asr"``       — Artifact Subspace Reconstruction; call
          :meth:`fit_asr` after :meth:`record_baseline`.
          (:class:`~mne_rt.tools.ASRDenoiser`)
        * ``"maxwell"``   — Signal Space Separation / tSSS for MEG;
          call :meth:`fit_maxwell` before :meth:`record_main` (MEG only).
          (:class:`~mne_rt.tools.RTMaxwellFilter`)

    save_nf_signal : bool, default True
        Save extracted feature time-series as JSON.
    config_file : str | None, default None
        Path to a YAML configuration file.  ``None`` uses the bundled
        default (``config_methods.yml``).
    verbose : bool | str | None, default None
        Verbosity level.  Mirrors MNE's convention:
        ``True``/``"INFO"`` → informational, ``False``/``"WARNING"`` →
        warnings only, ``"DEBUG"`` → all messages.

    Raises
    ------
    ValueError
        If any constructor argument fails validation.

    See Also
    --------
    ant.modalities.ModalityMixin : All supported NF feature methods.
    mne_rt.viz.NFPlot : Scrolling real-time NF signal display.
    mne_rt.viz.RawPlot : Scrolling raw M/EEG channel viewer (bad-channel / bad-segment marking).
    mne_rt.viz.EpochPlot : Scrolling raw viewer with trigger/epoch overlays.
    mne_rt.viz.BrainPlot : 3D brain activation display.

    Notes
    -----
    **Typical workflow**::

        nf = RTStream("sub01", session="01",
                        subjects_dir="/data/subjects",
                        montage="easycap-M1")
        nf.connect_to_lsl()
        nf.record_baseline(baseline_duration=120)
        nf.record_main(duration=600, modality=["sensor_power", "erd_ers"])

    The main main loop runs M/EEG acquisition in a background daemon thread
    and drives all visualisation windows (StreamViewer, signal plot,
    brain plot) from the Qt event loop on the main thread via a 33 ms pump
    timer, ensuring all three windows are truly parallel and non-blocking.

    .. versionadded:: 1.0.0
    """

    _VALID_ARTIFACT_METHODS = {False, "orica", "lms", "gedai", "asr", "maxwell"}
    _VALID_DATA_TYPES = {"eeg", "meg"}

    _SENSOR_POWER_SCALE: dict[str, float] = {"eeg": 1e-12, "meg": 1e-24}

    def __init__(
        self,
        subject_id: str,
        session: str,
        subjects_dir: str,
        montage: Optional[str],
        data_type: str = "eeg",
        mri: bool = False,
        subject_fs_id: str = "fsaverage",
        subjects_fs_dir: Optional[str] = None,
        bandpass_freq: Optional[tuple] = None,
        notch_freq: Union[float, list, None] = None,
        artifact_correction: Union[bool, str] = False,
        save_nf_signal: bool = True,
        config_file: Optional[str] = None,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        if not subject_id or not isinstance(subject_id, str):
            raise ValueError("`subject_id` must be a non-empty string.")
        if not session or not isinstance(session, str):
            raise ValueError("`session` must be a non-empty string (BIDS session label).")
        if data_type not in self._VALID_DATA_TYPES:
            raise ValueError(
                f"`data_type` must be one of {self._VALID_DATA_TYPES}, got {data_type!r}."
            )
        if montage is not None and not (
            montage in get_builtin_montages()
            or (montage.endswith(".bvct") and Path(montage).is_file())
        ):
            raise ValueError(
                "`montage` must be a built-in name, a valid '.bvct' path, or None."
            )
        if not isinstance(mri, bool):
            raise ValueError("`mri` must be a boolean.")
        if not subject_fs_id or not isinstance(subject_fs_id, str):
            raise ValueError("`subject_fs_id` must be a non-empty string.")
        if subjects_fs_dir is not None and not Path(subjects_fs_dir).is_dir():
            raise ValueError("`subjects_fs_dir` must be None or an existing directory.")
        if bandpass_freq is not None:
            if (
                not (hasattr(bandpass_freq, "__len__") and len(bandpass_freq) == 2)
                or not all(isinstance(f, (int, float)) and f > 0 for f in bandpass_freq)
                or bandpass_freq[0] >= bandpass_freq[1]
            ):
                raise ValueError(
                    "`bandpass_freq` must be a (l_freq, h_freq) tuple with 0 < l_freq < h_freq."
                )
        if notch_freq is not None:
            _nf = notch_freq if isinstance(notch_freq, list) else [notch_freq]
            if not all(isinstance(f, (int, float)) and f > 0 for f in _nf):
                raise ValueError("`notch_freq` must be a positive float or list of positive floats.")
        if artifact_correction not in self._VALID_ARTIFACT_METHODS:
            raise ValueError(
                f"`artifact_correction` must be one of "
                f"{self._VALID_ARTIFACT_METHODS}, got {artifact_correction!r}."
            )
        if artifact_correction == "lms" and data_type == "meg":
            raise ValueError("LMS artifact correction is only supported for EEG.")
        if artifact_correction == "maxwell" and data_type != "meg":
            raise ValueError("Maxwell filtering is only supported for MEG data.")
        if not isinstance(save_nf_signal, bool):
            raise ValueError("`save_nf_signal` must be a boolean.")

        if config_file is None:
            # Prefer the bundled copy inside the package; fall back to repo root
            # for backward-compat with in-tree development without pip install.
            _bundled = _PKG_DIR / "config_methods.yml"
            config = _bundled if _bundled.is_file() else _REPO_ROOT / "config_methods.yml"
        elif config_file.endswith(".yml") and Path(config_file).is_file():
            config = Path(config_file)
        else:
            raise ValueError("`config_file` must be None or a valid .yml file path.")

        self.subject_id = subject_id
        self.session = session
        self.subjects_dir = subjects_dir
        self.montage = montage
        self.data_type = data_type
        self.mri = mri
        self.subject_fs_id = subject_fs_id
        self.subjects_fs_dir = subjects_fs_dir
        self.bandpass_freq = bandpass_freq
        self.notch_freq = notch_freq
        self.artifact_correction = artifact_correction
        self.save_nf_signal = save_nf_signal
        self.config_file = config
        self.verbose = verbose
        self.subject_dir = (
            Path(subjects_dir) / f"sub-{subject_id}" / f"ses-{session}"
        )

        set_log_level(verbose)

    # ------------------------------------------------------------------
    # LSL connection
    # ------------------------------------------------------------------

    @verbose
    def connect_to_lsl(
        self,
        chunk_size: int = 10,
        mock_lsl: bool = False,
        fname: Optional[str] = None,
        n_repeat: Union[int, float] = np.inf,
        bufsize_baseline: int = 4,
        bufsize_main: int = 3,
        acquisition_delay: float = 0.001,
        timeout: float = 15.0,
        stream_name: Optional[str] = None,
        stream_source_id: Optional[str] = None,
        pick_types: Optional[str] = None,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Connect to an LSL M/EEG stream.

        Parameters
        ----------
        chunk_size : int, default 10
            Samples per chunk for mock streaming.
        mock_lsl : bool, default False
            Stream a pre-recorded file instead of live hardware.
            Requires ``fname`` or uses the bundled sample file.
        fname : str | None, default None
            Path to any MNE-readable recording for mock streaming
            (.fif, .vhdr, .edf, .bdf, .set, …).
            ``None`` uses the bundled sample recording.
        n_repeat : int | float, default ``np.inf``
            How many times to loop the mock recording.
        bufsize_baseline : int, default 4
            LSL buffer size in seconds for baseline sessions.
        bufsize_main : int, default 3
            LSL buffer size in seconds for main sessions.
        acquisition_delay : float, default 0.001
            Seconds between acquisition polling attempts.
        timeout : float, default 15.0
            Maximum wait time in seconds for the LSL connection.
        stream_name : str | None, default None
            Connect by stream name (e.g. ``neuromag2lsl`` for MEG devices).
        stream_source_id : str | None, default None
            Connect by stream source ID.
        pick_types : str | None, default None
            Channel type to keep (e.g. ``"eeg"``, ``"mag"``).
            ``None`` keeps all available channels.
        verbose : bool | str | None, default None
            Override the instance-level verbosity for this call.

        Raises
        ------
        RuntimeError
            If no LSL stream matching the given criteria is found within
            ``timeout`` seconds.

        Notes
        -----
        All public methods of :class:`mne_lsl.stream.StreamLSL` are
        also exposed directly on the :class:`RTStream` instance after
        connection.

        Examples
        --------
        Connect to a live EEG amplifier::

            nf.connect_to_lsl()

        Simulate from any MNE-readable file::

            nf.connect_to_lsl(mock_lsl=True, fname="path/to/data.fif")
            nf.connect_to_lsl(mock_lsl=True, fname="path/to/data.edf")
        """
        if hasattr(self, "stream") and getattr(self.stream, "connected", False):
            self.stream.disconnect()

        if getattr(self, "_mock_player", None) is not None:
            try:
                self._mock_player.stop()
            except Exception:
                pass
            self._mock_player = None

        if mock_lsl and fname is None:
            _bundled = _REPO_ROOT / "data" / "sample" / "sample_data.vhdr"
            if _bundled.is_file():
                fname = _bundled
            else:
                # Not an editable install — generate a synthetic EEG recording
                fname = _make_demo_raw_fif()

        self.bufsize = bufsize_main

        if self.montage is not None and Path(str(self.montage)).is_file():
            self.montage = read_dig_captrak(self.montage)

        self.source_id = uuid.uuid4().hex
        if mock_lsl:
            self._mock_player = Player(
                fname,
                chunk_size=chunk_size,
                n_repeat=n_repeat,
                source_id=self.source_id,
            )
            self._mock_player.start()
            time.sleep(3.0)  # let LSL multicast initialize on first use and player advertise

        if stream_name is not None:
            stream = Stream(bufsize=self.bufsize, name=stream_name)
        elif stream_source_id is not None:
            stream = Stream(bufsize=self.bufsize, source_id=stream_source_id)
        else:
            stream = Stream(bufsize=self.bufsize, source_id=self.source_id)

        stream.connect(acquisition_delay=acquisition_delay, timeout=timeout)

        if self.montage is not None:
            stream.set_montage(self.montage, on_missing="warn")
        if pick_types is not None:
            stream.pick(pick_types)

        stream.set_meas_date(
            datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)
        )
        self.stream = stream
        self.sfreq = stream.info["sfreq"]
        self.rec_info = stream.info
        self.rec_info["subject_info"] = {"his_id": self.subject_id}

        # Expose stream methods directly on self
        for name in dir(self.stream):
            if not name.startswith("__"):
                attr = getattr(self.stream, name)
                if callable(attr):
                    setattr(self, name, attr)

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self, include_delays: bool = False) -> None:
        """Create BIDS-aligned session subdirectories under ``subject_dir``."""
        for sub in ("eeg", "beh", "inv", "reports"):
            (self.subject_dir / sub).mkdir(parents=True, exist_ok=True)
        if include_delays:
            (self.subject_dir / "delays").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @verbose
    def record_baseline(
        self,
        baseline_duration: float,
        winsize: float = 3.0,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Record a resting-state baseline segment.

        Collects ``baseline_duration`` seconds of M/EEG, stores it as
        :attr:`raw_baseline`, saves it to disk, and computes the inverse
        operator (stored as :attr:`inv`).

        Parameters
        ----------
        baseline_duration : float
            Total recording duration in seconds.
        winsize : float, default 3.0
            Duration in seconds of each data fetch chunk.
        verbose : bool | str | None, default None
            Override the instance-level verbosity for this call.

        Notes
        -----
        Output files written under ``<subjects_dir>/sub-<ID>/ses-<session>/``:

        * ``eeg/sub-<ID>_ses-<session>_task-baseline_eeg.fif``
        * ``inv/sub-<ID>_ses-<session>_task-baseline_inv.fif``
        * ``inv/sub-<ID>_ses-<session>_task-baseline_fwd.fif``
        * ``inv/sub-<ID>_ses-<session>_task-baseline_cov.fif``

        Examples
        --------
        >>> nf.record_baseline(baseline_duration=120)
        """
        self.baseline_duration = baseline_duration
        logger.info("Recording baseline (%.0f s) …", baseline_duration)

        t_start = local_clock()
        chunks: list[np.ndarray] = []
        while local_clock() < t_start + baseline_duration:
            chunks.append(self.stream.get_data(winsize)[0])
            time.sleep(winsize)

        data = np.concatenate(chunks, axis=1)
        raw_baseline = RawArray(data, self.rec_info)

        self._ensure_dirs()
        _stem = f"sub-{self.subject_id}_ses-{self.session}"
        raw_baseline.save(
            self.subject_dir / "eeg" / f"{_stem}_task-baseline_eeg.fif",
            overwrite=True,
        )
        self.raw_baseline = raw_baseline
        self.compute_inv_operator()

    @verbose
    def record_main(
        self,
        duration: float,
        modality: Union[str, list[str]] = "sensor_power",
        picks: Optional[Union[str, list[str]]] = None,
        winsize: float = 1.0,
        estimate_delays: bool = False,
        modality_params: Optional[dict[str, Any]] = None,
        show_raw_signal: bool = True,
        show_nf_signal: bool = True,
        time_window: float = 10.0,
        show_topo: bool = False,
        topo_bands: Optional[dict] = None,
        show_brain_activation: bool = False,
        brain_surf: str = "inflated",
        brain_mode: str = "power",
        brain_freq_range: tuple[float, float] = (8.0, 13.0),
        zscore_normalize: bool = False,
        zscore_warmup: int = 10,
        zscore_alpha: float = 0.0,
        osc_sender: Optional[Any] = None,
        lsl_sender: Optional[Any] = None,
        protocol: Optional[Any] = None,
        save_raw: bool = False,
        ref_channel: str = "Fp1",
        signal_smoothing: float = 0.25,
        display_smoothing: float = 0.3,
        topo_display_smoothing: float = 1.0,
        brain_display_smoothing: float = 0.3,
        track_artifact_rate: bool = True,
        artifact_threshold_uv: float = 100.0,
        track_snr: bool = False,
        snr_frange: Optional[tuple] = None,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Stream M/EEG, extract neural features, and drive NF visualisation.

        This is the main closed-loop entry point.  It:

        1. Prepares each requested modality (calls ``_<modality>_prep``).
        2. Opens the selected visualisation windows (StreamViewer, signal
           plot, brain plot) on the main thread.
        3. Starts a background daemon thread that continuously fetches data,
           runs artifact correction, and computes features in parallel via
           a thread pool.
        4. Drives all windows from the Qt event loop via a 33 ms pump timer.
        5. Saves raw data and feature time-series to disk when finished.

        Parameters
        ----------
        duration : float
            Total recording length in seconds.
        modality : str | list of str, default "sensor_power"
            NF feature(s) to extract.  Must match keys in
            ``config_methods.yml``.  Multiple modalities are extracted in
            parallel.  Available modalities:
            ``"sensor_power"``, ``"band_ratio"``, ``"erd_ers"``,
            ``"laterality"``, ``"laterality_erd_ers"``, ``"hjorth"``,
            ``"spectral_centroid"``, ``"argmax_freq"``,
            ``"individual_peak_power"``, ``"entropy"``,
            ``"instantaneous_phase"``, ``"scp"``, ``"peak_alpha_freq"``,
            ``"sensor_connectivity"``, ``"cfc_sensor"``, ``"sensor_graph"``,
            ``"connectivity_ratio"``,
            ``"source_power"``, ``"source_connectivity"``, ``"source_graph"``.
        picks : str | list of str | None, default None
            Channel selection passed to the LSL stream.  ``None`` uses all
            available channels.  Must be ``None`` for source-space modalities.
        winsize : float, default 1.0
            Analysis window length in seconds.
        estimate_delays : bool, default False
            Measure and save per-step timing (acquisition, artifact
            correction, feature extraction).
        modality_params : dict | None, default None
            Per-modality parameter overrides.  Keys are modality names;
            values are dicts of ``{parameter: new_value}`` pairs that
            override the config-file defaults.
        show_raw_signal : bool, default True
            Show the :class:`~mne_rt.viz.RawPlot` scrolling raw M/EEG viewer.
        show_nf_signal : bool, default True
            Show the :class:`~mne_rt.viz.NFPlot` real-time NF monitor.
        time_window : float, default 10.0
            Visible time range in seconds for the signal plot.
        show_topo : bool, default False
            Show the :class:`~mne_rt.viz.TopomapPlot` real-time scalp topomap
            display.  Requires the montage to be set on the channel info.
        topo_bands : dict | None, default None
            Frequency bands to show in the topomap as
            ``{label: (f_low, f_high)}``.  ``None`` uses the default
            δ/θ/α/β/γ bands.
        show_brain_activation : bool, default False
            Show the :class:`~mne_rt.viz.BrainPlot` 3D brain activation
            display (requires ``subjects_fs_dir`` and a fitted inverse
            operator).
        brain_surf : {"inflated", "pial", "white", "sphere"}, default "pial"
            Cortical surface geometry for the brain display.
            ``"pial"`` shows the true cortical folding; ``"inflated"``
            unfolds gyri/sulci for easier label inspection.
        brain_mode : {"power", "activation"}, default "power"
            Source-space display mode.  ``"power"`` shows mean squared
            amplitude; ``"activation"`` shows mean amplitude.
        brain_freq_range : (float, float), default (8.0, 13.0)
            Frequency band in Hz used to band-pass the data before computing
            source power for the brain display.
        zscore_normalize : bool, default False
            Apply online z-score normalisation to each NF feature value
            before storing and displaying it.  During the first
            ``zscore_warmup`` windows the raw value is passed through
            unchanged; once warmup completes, each value is normalised as
            ``z = (x − μ) / σ`` where μ and σ are estimated from the
            warmup windows.  If ``zscore_alpha > 0`` the statistics are
            updated after every window with an exponential moving average
            so the normaliser slowly tracks drift.
        zscore_warmup : int, default 10
            Number of windows to collect before activating normalisation.
            The mean and standard deviation of these windows are used as
            the initial statistics.
        zscore_alpha : float, default 0.0
            EMA forgetting factor for updating μ and σ each window.
            ``0.0`` freezes statistics after warmup (recommended for most
            NF protocols).  Values in [0.01, 0.1] add slow adaptation.
        osc_sender : OSCSender | None, default None
            If provided, each computed NF value is also broadcast over OSC
            to the configured host/port after every update cycle.
            See :class:`~mne_rt.osc.OSCSender`.
        lsl_sender : LSLSender | None, default None
            If provided, each computed NF value is pushed into an LSL stream
            outlet after every update cycle.  Faster and more reliable than
            OSC for same-machine feedback delivery.
            See :class:`~mne_rt.lsl_output.LSLSender`.
        protocol : Protocol instance | dict | None, default None
            Real-time NF reward protocol evaluated on every analysis window.
            Pass a single Protocol instance (e.g.
            :class:`~mne_rt.protocols.ThresholdProtocol`) to apply it to the
            first modality, or a ``{modality_name: protocol}`` dict to
            apply different protocols to different modalities.  On each
            window the protocol's ``evaluate(value)`` method is called and
            ``(crossed, magnitude)`` is recorded.  Results are accessible via
            :attr:`reward_data` after the session.
        save_raw : bool, default False
            Persist the raw pre-correction M/EEG acquired during the main
            session to ``raw/<stem>-raw.fif``.  Off by default because FIF
            files can be large; enable when the raw continuous signal is
            needed for offline re-analysis or provenance.
        signal_smoothing : float, default 0.25
            Exponential moving average (EMA) factor applied to each NF feature
            value before it is stored and displayed.  Controls the trade-off
            between signal smoothness and responsiveness:

            * ``1.0`` — no smoothing; raw per-window estimate passed through.
            * ``0.5`` — moderate smoothing; each value is 50 % new + 50 % history.
            * ``0.1`` — heavy smoothing; very slow response to rapid changes.

            The EMA is applied after z-score normalisation (if enabled) and
            before protocol evaluation, so protocols see the smoothed value.
        display_smoothing : float, default 0.3
            Additional EMA factor applied **only** inside the live signal
            plot.  Does not affect stored ``nf_data`` or protocol evaluation.
            Lower values give a smoother, slower-reacting display curve;
            ``1.0`` disables this extra layer and shows the already
            ``signal_smoothing``-filtered values directly.
        topo_display_smoothing : float, default 1.0
            EMA factor for the :class:`~mne_rt.viz.TopomapPlot` band-power maps.
            ``1.0`` (default) disables smoothing so transient artifacts remain
            visible for operator monitoring.  Lower values progressively
            smooth the spatial maps across consecutive windows.
        brain_display_smoothing : float, default 0.3
            EMA factor for the :class:`~mne_rt.viz.BrainPlot` per-vertex
            activation arrays.  Blends consecutive frames so the cortical
            map transitions smoothly.  ``1.0`` disables smoothing.
        ref_channel : str, default "Fp1"
            Reference channel used for LMS artifact correction
            (``artifact_correction="lms"`` only).  Ignored for all other
            correction methods.
        track_artifact_rate : bool, default True
            If ``True``, count windows whose peak-to-peak amplitude exceeds
            ``artifact_threshold_uv`` and store the fraction as
            :attr:`artifact_rate` at the end of the session.
        artifact_threshold_uv : float, default 100.0
            Peak-to-peak amplitude threshold in µV used to classify a window
            as artifactual when ``track_artifact_rate=True``.
        track_snr : bool, default False
            If ``True``, compute a per-window signal-to-noise ratio (band
            power in ``snr_frange`` divided by broadband noise power, in dB)
            and store the resulting time-series as :attr:`snr_data`.
        snr_frange : tuple(float, float) | None, default None
            Frequency band ``(f_low, f_high)`` in Hz used as the "signal"
            band when ``track_snr=True``.  ``None`` defaults to the alpha
            band ``(8.0, 13.0)``.
        verbose : bool | str | None, default None
            Override the instance-level verbosity for this call.

        Raises
        ------
        NotImplementedError
            If a requested ``modality`` is not implemented.
        ValueError
            If a source-space modality is requested together with non-``None``
            ``picks``.
        RuntimeError
            If ``show_brain_activation=True`` but ``subjects_fs_dir`` is
            not set, or if ``artifact_correction="gedai"`` but
            :meth:`fit_gedai` has not been called.

        Notes
        -----
        Output files written under ``<subjects_dir>/sub-<ID>/ses-<session>/``:

        * ``beh/sub-<ID>_ses-<session>_task-neurofeedback_beh.json`` — NF
          feature time-series plus session metadata
        * ``delays/sub-<ID>_ses-<session>_task-neurofeedback_delays.json``
          — per-step timing (only when ``estimate_delays=True``)
        * ``eeg/sub-<ID>_ses-<session>_task-neurofeedback_eeg.fif`` —
          pre-correction M/EEG (only when ``save_raw=True``)

        Examples
        --------
        Single-modality alpha-power NF with brain activation::

            nf.record_main(
                duration=300,
                modality="sensor_power",
                show_brain_activation=True,
            )

        Multi-modality session with custom parameters::

            nf.record_main(
                duration=600,
                modality=["sensor_power", "erd_ers", "laterality"],
                modality_params={"sensor_power": {"frange": [10, 12]}},
                show_nf_signal=True,
            )

        .. versionadded:: 1.0.0
        """
        self.duration = duration
        self.modality = modality
        self.picks = picks
        self.modality_params = modality_params
        self.winsize = winsize
        self.window_size_s = int(winsize * self.rec_info["sfreq"])
        self.estimate_delays = estimate_delays
        self._sfreq = self.rec_info["sfreq"]
        self.show_nf_signal = show_nf_signal
        if zscore_alpha < 0.0 or zscore_alpha >= 1.0:
            raise ValueError("`zscore_alpha` must be in [0, 1).")
        if zscore_warmup < 2:
            raise ValueError("`zscore_warmup` must be ≥ 2.")
        if not (0.0 < signal_smoothing <= 1.0):
            raise ValueError("`signal_smoothing` must be in (0, 1].")

        self._session_start_time = datetime.datetime.now(datetime.timezone.utc)
        self._session_stem = f"sub-{self.subject_id}_ses-{self.session}"
        self._ensure_dirs(include_delays=estimate_delays)

        # Artifact correction setup
        ref_ch_idx: Optional[int] = None
        if self.artifact_correction == "lms":
            ref_ch_idx = self.rec_info["ch_names"].index(ref_channel)
        elif self.artifact_correction == "orica":
            self.run_orica(n_channels=len(self.rec_info["ch_names"]), forgetfac=0.99)
        elif self.artifact_correction == "gedai":
            if not hasattr(self, "gedai") or self.gedai is None:
                raise RuntimeError(
                    "Call fit_gedai() after baseline recording before starting a gedai session."
                )
        elif self.artifact_correction == "asr":
            if not hasattr(self, "asr") or self.asr is None:
                raise RuntimeError(
                    "Call fit_asr() after baseline recording before starting an ASR session."
                )
        elif self.artifact_correction == "maxwell":
            if not hasattr(self, "maxwell_filter") or self.maxwell_filter is None:
                raise RuntimeError(
                    "Call fit_maxwell() before record_main() to initialise Maxwell filtering."
                )

        # Modality preparation
        mods = [modality] if isinstance(modality, str) else list(modality)
        self._mods = mods
        self.executor = ThreadPoolExecutor(max_workers=len(mods))
        self.mod_params_dict = {
            mod: get_params(self.config_file, mod, self.modality_params)
            for mod in mods
        }
        precomps: list[dict] = []
        nf_fns: list = []
        for mod in mods:
            self.params = self.mod_params_dict[mod]
            fn = getattr(self, f"_{mod}", None)
            if not callable(fn):
                raise NotImplementedError(f"Modality '{mod}' is not implemented.")
            if "source" in mod and picks is not None:
                raise ValueError("'picks' must be None for source-space modalities.")
            prep = getattr(self, f"_{mod}_prep", None)
            precomps.append(prep() if callable(prep) else {})
            nf_fns.append(fn)

        # ---- Visualization setup (main thread) ----

        scales_dict = {
            "sensor_power":         self._SENSOR_POWER_SCALE[self.data_type],
            "band_ratio":           4.0,
            "source_power":         3e-2,
            "sensor_connectivity":  1.0,
            "source_connectivity":  1.0,
            "connectivity_ratio":   4.0,
            "sensor_graph":         0.05,
            "source_graph":         2e-17,
            "entropy":              3.0,
            "argmax_freq":          8.0,
            "peak_alpha_freq":      13.0,
            "individual_peak_power": self._SENSOR_POWER_SCALE[self.data_type],
            "cfc_sensor":           1.0,
            "erd_ers":              50.0,
            "laterality":           2.0,
            "hjorth":               5.0,
            "spectral_centroid":    5.0,
            "scp":                  50e-6,
        }

        nf_plot: Optional[NFPlot] = None
        raw_plot: Optional[RawPlot] = None
        topo_plot: Optional[TopomapPlot] = None
        brain_plot: Optional[BrainPlot] = None

        needs_qt = show_nf_signal or show_brain_activation or show_raw_signal or show_topo
        app: Optional[QtWidgets.QApplication] = None

        if needs_qt:
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        if show_nf_signal:
            nf_plot = NFPlot(
                modalities=mods,
                scales_dict=scales_dict,
                sfreq=30.0,   # pump timer rate drives display at 30 fps
                time_window=time_window,
                display_smoothing=display_smoothing,
            )
            nf_plot.show()

        if show_topo:
            topo_plot = TopomapPlot(
                info=self.rec_info,
                sfreq=self._sfreq,
                bands=topo_bands,
                display_smoothing=topo_display_smoothing,
            )
            topo_plot.show()

        if show_brain_activation:
            if self.subjects_fs_dir is None:
                raise ValueError(
                    "subjects_fs_dir must be set to use brain activation display."
                )
            brain_plot = BrainPlot(
                subjects_fs_dir=self.subjects_fs_dir,
                clim=[0, 0.6],
                surf=brain_surf,
                display_smoothing=brain_display_smoothing,
            )

        if show_raw_signal:
            raw_plot = RawPlot(
                ch_names=self.rec_info["ch_names"],
                sfreq=self._sfreq,
                info=self.rec_info,
            )
            raw_plot.show()

        if self.bandpass_freq is not None:
            self.stream.filter(
                l_freq=self.bandpass_freq[0], h_freq=self.bandpass_freq[1]
            )
        if self.notch_freq is not None:
            _freqs = (
                self.notch_freq
                if isinstance(self.notch_freq, list)
                else [self.notch_freq]
            )
            for _f in _freqs:
                self.stream.notch_filter(freqs=_f)

        # ---- Thread-safe queues between acquisition thread and UI ----

        # small caps: drop stale frames rather than accumulate backlog
        nf_queue: _queue.Queue = _queue.Queue(maxsize=4)
        topo_queue: Optional[_queue.Queue] = (
            _queue.Queue(maxsize=2) if topo_plot is not None else None
        )
        brain_queue: Optional[_queue.Queue] = (
            _queue.Queue(maxsize=2) if brain_plot is not None else None
        )
        raw_queue: Optional[_queue.Queue] = (
            _queue.Queue(maxsize=4) if raw_plot is not None else None
        )
        done_event = threading.Event()
        nf_data: dict[str, list] = {m: [] for m in mods}
        _ema: dict[str, float] = {}  # EMA state, seeded on first window

        # Build protocol map: {modality_name: protocol_instance}
        if protocol is None:
            _proto_map: dict[str, Any] = {}
        elif isinstance(protocol, dict):
            _proto_map = dict(protocol)
        else:
            _proto_map = {mods[0]: protocol}
        reward_data: dict[str, list] = {m: [] for m in _proto_map}

        # Delay accumulators live in the thread, assigned to self when done
        _acq_delays:     list[float] = []
        _art_delays:     list[float] = []
        _meth_delays:    dict[str, list] = {m: [] for m in mods}
        _plot_delays:    list[float] = []  # only used when viz runs inline

        # ---- Artifact rate tracking ----
        _n_total_windows: list[int] = [0]
        _n_artifact_windows: list[int] = [0]
        _artifact_threshold_raw = artifact_threshold_uv * 1e-6  # V for EEG, T for MEG

        # ---- SNR tracking ----
        _snr_data: list[float] = []
        _snr_frange = snr_frange if snr_frange is not None else (
            tuple(self.mod_params_dict[mods[0]].get("frange", [8, 13]))
            if mods else (8.0, 13.0)
        )

        # ---- Z-score normalisation state ----
        _z_buf:   dict[str, list] = {m: [] for m in mods}
        _z_mean:  dict[str, float] = {}
        _z_std:   dict[str, float] = {}

        def _apply_zscore(mod: str, val: float) -> float:
            if not zscore_normalize:
                return val
            buf = _z_buf[mod]
            buf.append(float(val))
            n = len(buf)
            if n < zscore_warmup:
                return val                          # pass-through during warmup
            if n == zscore_warmup:
                arr = np.array(buf, dtype=float)
                _z_mean[mod] = float(arr.mean())
                _z_std[mod]  = max(float(arr.std()), 1e-6)
            elif zscore_alpha > 0.0:
                m_prev = _z_mean[mod]
                _z_mean[mod] += zscore_alpha * (val - m_prev)
                _z_std[mod]   = max(
                    _z_std[mod] * (1.0 - zscore_alpha)
                    + zscore_alpha * abs(val - m_prev),
                    1e-6,
                )
            return (val - _z_mean[mod]) / _z_std[mod]

        # ---- Shared artifact-correction helper ----

        def _correct(data: np.ndarray) -> np.ndarray:
            if self.artifact_correction == "lms":
                art_tic = time.time()
                data = remove_blinks_lms(data, ref_ch_idx=ref_ch_idx, n_taps=5, mu=0.01)
                if estimate_delays:
                    _art_delays.append(time.time() - art_tic)
            elif self.artifact_correction == "orica":
                art_tic = time.time()
                data = self.orica.denoise(data, self.orica.find_blink_ic(self.blink_template, threshold=0.4)[0])
                if estimate_delays:
                    _art_delays.append(time.time() - art_tic)
            elif self.artifact_correction == "gedai":
                art_tic = time.time()
                data = self.gedai.update_and_denoise(data, self.blink_template, threshold=0.7)
                if estimate_delays:
                    _art_delays.append(time.time() - art_tic)
            elif self.artifact_correction == "asr":
                art_tic = time.time()
                data = self.asr.transform(data)
                if estimate_delays:
                    _art_delays.append(time.time() - art_tic)
            elif self.artifact_correction == "maxwell":
                art_tic = time.time()
                data = self.maxwell_filter.transform(data)
                if estimate_delays:
                    _art_delays.append(time.time() - art_tic)
            return data

        def _push_topo(window: np.ndarray) -> None:
            """Enqueue raw data window for the topomap display."""
            if topo_queue is None:
                return
            try:
                topo_queue.put_nowait(window)
            except _queue.Full:
                pass

        def _push_brain(window: np.ndarray) -> None:
            """Submit inverse computation to thread pool; enqueue scalars for UI."""
            if brain_queue is None or brain_queue.full():
                return
            if not hasattr(self, "inv") or self.inv is None:
                return

            def _compute_and_enqueue(w: np.ndarray) -> None:
                raw_d = self._prepare_raw_array(w)
                if brain_mode == "power":
                    raw_d.filter(
                        l_freq=brain_freq_range[0], h_freq=brain_freq_range[1],
                        fir_design="firwin", verbose=False,
                    )
                stc = apply_inverse_raw(raw_d, self.inv, lambda2=1.0 / 9, pick_ori="normal")
                if brain_mode == "power":
                    lh = np.mean(stc.lh_data ** 2, axis=1)
                    rh = np.mean(stc.rh_data ** 2, axis=1)
                else:
                    lh = np.abs(stc.lh_data.mean(axis=1))
                    rh = np.abs(stc.rh_data.mean(axis=1))
                # Normalise to [0, 1] relative to the 98th percentile so the
                # BrainPlot clim [0, 0.6] gives meaningful spatial contrast.
                p98 = float(np.percentile(np.concatenate([lh, rh]), 98)) or 1.0
                lh = lh / p98
                rh = rh / p98
                try:
                    brain_queue.put_nowait((lh, rh))
                except _queue.Full:
                    pass

            self.executor.submit(_compute_and_enqueue, window.copy())

        # ---- Acquisition thread ----

        def _acquire() -> None:
            t_start = local_clock()
            _raw_chunks: list[np.ndarray] = []
            # 50 % overlap: advance by half a window each step so consecutive
            # NF estimates share data → smooth, correlated curve updates.
            _hop = max(1, self.window_size_s // 2)

            while local_clock() < t_start + duration:
                # Block until half a window of new samples has arrived, then
                # fetch the latest winsize seconds (50 % overlap with prev window).
                # While waiting, flush any accumulated raw samples to the display
                # queue at ~30 fps so RawPlot scrolls smoothly.
                _t_hop = local_clock()
                _hop_dur = _hop / self._sfreq
                while local_clock() < _t_hop + _hop_dur:
                    if local_clock() >= t_start + duration:
                        break
                    if raw_queue is not None:
                        n_avail = self.stream.n_new_samples
                        if n_avail >= max(1, int(0.033 * self._sfreq)):
                            try:
                                raw_chunk = self.stream.get_data(
                                    n_avail / self._sfreq
                                )[0]
                                if raw_chunk.shape[1] > 0:
                                    raw_queue.put_nowait(raw_chunk)
                            except (_queue.Full, Exception):
                                pass
                    time.sleep(0.005)

                tic = time.time()
                data = self.stream.get_data(winsize, picks=picks)[0]
                if estimate_delays:
                    _acq_delays.append(time.time() - tic)
                if data.shape[1] != self.window_size_s:
                    continue

                _n_total_windows[0] += 1
                if track_artifact_rate:
                    if np.any(np.abs(data) > _artifact_threshold_raw):
                        _n_artifact_windows[0] += 1

                _raw_chunks.append(data.copy())
                data = _correct(data)

                if track_snr:
                    from mne_rt.tools import compute_bandpower
                    _sig = compute_bandpower(data, self._sfreq, _snr_frange, method="welch")
                    _all = compute_bandpower(data, self._sfreq, (0.5, self._sfreq / 2 - 1), method="welch")
                    _noise = _all.mean() - _sig.mean()
                    _snr_data.append(float(10.0 * np.log10(_sig.mean() / (abs(_noise) + 1e-300))))

                futures = [
                    self.executor.submit(nf_fns[i], data, **precomps[i])
                    for i in range(len(mods))
                ]
                for m, fut in zip(mods, futures):
                    nf_val, m_delay = fut.result()
                    nf_val = _apply_zscore(m, float(nf_val))
                    # EMA smoothing: seed on first window, then blend
                    if m not in _ema:
                        _ema[m] = nf_val
                    else:
                        nf_val = signal_smoothing * nf_val + (1.0 - signal_smoothing) * _ema[m]
                        _ema[m] = nf_val
                    nf_data[m].append(nf_val)
                    if m in _proto_map:
                        _crossed, _mag = _proto_map[m].evaluate(nf_val)
                        reward_data[m].append(_mag if _crossed else 0.0)
                    if estimate_delays:
                        _meth_delays[m].append(m_delay)

                _vals = [nf_data[m][-1] for m in mods]
                try:
                    nf_queue.put_nowait(_vals)
                except _queue.Full:
                    pass

                if osc_sender is not None:
                    try:
                        osc_sender.send_all(mods, _vals)
                    except Exception:
                        pass

                if lsl_sender is not None:
                    try:
                        lsl_sender.push(mods, _vals)
                    except Exception:
                        pass

                _push_topo(data)
                _push_brain(data)

            done_event.set()
            self._raw_chunks = _raw_chunks

        # ---- Start acquisition thread ----

        acq_thread = threading.Thread(target=_acquire, daemon=True)
        acq_thread.start()

        # ---- Event loop or blocking join ----

        if needs_qt:
            from qtpy.QtCore import QTimer

            # Interpolation state: [prev_vals, curr_vals, step_index]
            # Linearly interpolates between consecutive NF estimates so the
            # display ramps smoothly. With 50 % overlap the acquisition produces
            # values every winsize/2 s, so we ramp over winsize/2 s worth of
            # 30-fps ticks.
            _interp: list = [[], [], [0]]
            _n_steps: int = max(1, int(30 * winsize // 2))

            def _pump_signal() -> None:
                """Fast timer (~30 fps) — signal plot only.

                Linearly interpolates between the two most-recent NF estimates
                so the trace ramps smoothly over one window period.
                """
                try:
                    while not nf_queue.empty():
                        new_vals = nf_queue.get_nowait()
                        _interp[0] = _interp[1] if _interp[1] else new_vals
                        _interp[1] = new_vals
                        _interp[2][0] = 0
                except Exception:
                    pass

                if nf_plot is not None and _interp[1]:
                    step = _interp[2][0]
                    if _interp[0] and step < _n_steps:
                        alpha = step / _n_steps
                        vals = [
                            _interp[0][i] * (1 - alpha) + _interp[1][i] * alpha
                            for i in range(len(_interp[1]))
                        ]
                    else:
                        vals = _interp[1]
                    nf_plot.push(vals)
                    _interp[2][0] += 1

                if done_event.is_set():
                    signal_timer.stop()
                    if topo_timer is not None:
                        topo_timer.stop()
                    if brain_timer is not None:
                        brain_timer.stop()
                    if raw_timer is not None:
                        raw_timer.stop()
                    QTimer.singleShot(600, app.quit)

            signal_timer = QTimer()
            signal_timer.setInterval(33)   # ~30 fps
            signal_timer.timeout.connect(_pump_signal)
            signal_timer.start()

            # Topomap timer (~5 fps) — matplotlib redraws are slower than
            # pyqtgraph so we use a separate slower timer.
            topo_timer: Optional[QTimer] = None
            if topo_plot is not None:
                def _pump_topo_qt() -> None:
                    topo_data = None
                    try:
                        while not topo_queue.empty():
                            topo_data = topo_queue.get_nowait()
                    except Exception:
                        pass
                    if topo_data is not None:
                        topo_plot.push(topo_data)

                topo_timer = QTimer()
                topo_timer.setInterval(200)  # 5 fps
                topo_timer.timeout.connect(_pump_topo_qt)
                topo_timer.start()

            # Separate slow timer for the brain plot so its render never
            # blocks the signal pump.  Scalars are updated without an
            # immediate render (deferred=True); a single render() is issued
            # at the end of the callback so the signal timer can fire first.
            brain_timer: Optional[QTimer] = None
            if brain_plot is not None:
                def _pump_brain() -> None:
                    """Slow timer (~5 fps) — brain render only."""
                    try:
                        updated = False
                        while not brain_queue.empty():
                            lh, rh = brain_queue.get_nowait()
                            brain_plot.update_from_arrays(
                                lh, rh, mode=brain_mode, deferred=True
                            )
                            updated = True
                        if updated:
                            brain_plot.plotter.render()
                            brain_plot.write_frame_if_recording()
                    except Exception:
                        pass

                brain_timer = QTimer()
                brain_timer.setInterval(200)  # 5 fps — brain doesn't need more
                brain_timer.timeout.connect(_pump_brain)
                brain_timer.start()

            raw_timer: Optional[QTimer] = None
            if raw_plot is not None:
                def _pump_raw() -> None:
                    try:
                        while not raw_queue.empty():
                            chunk = raw_queue.get_nowait()
                            raw_plot.push(chunk)
                    except Exception:
                        pass

                raw_timer = QTimer()
                raw_timer.setInterval(33)  # ~30 fps
                raw_timer.timeout.connect(_pump_raw)
                raw_timer.start()

            app.exec()          # blocks here; drives all three windows in parallel

        else:
            acq_thread.join()   # headless: block until acquisition finishes

        acq_thread.join(timeout=10)

        # ---- Persist results ----

        self.nf_data = nf_data
        self.reward_data = reward_data
        if estimate_delays:
            self.acq_delays      = _acq_delays
            self.artifact_delays = _art_delays
            self.method_delays   = _meth_delays
        n_tot = _n_total_windows[0]
        self.artifact_rate = (
            _n_artifact_windows[0] / n_tot if n_tot > 0 else 0.0
        )
        self.n_artifact_windows = _n_artifact_windows[0]
        self.n_total_windows = n_tot
        self.snr_data = _snr_data if track_snr else []

        saved_files = self.save(
            nf_data=self.save_nf_signal,
            acq_delay=True,
            artifact_delay=True,
            method_delay=True,
            raw_data=save_raw,
            format="json",
        )
        for kind, path in saved_files.items():
            logger.info("Saved %s → %s", kind, path)

        if signal_plot is not None:
            signal_plot.close()

    # ------------------------------------------------------------------
    # Offline replay
    # ------------------------------------------------------------------

    @verbose
    def replay(
        self,
        fname: str,
        modality: Union[str, list[str]] = "sensor_power",
        duration: Optional[float] = None,
        winsize: float = 1.0,
        verbose: Union[bool, str, None] = None,
        **record_main_kwargs,
    ) -> None:
        """Replay a saved recording as a mock LSL session.

        Loads a pre-recorded M/EEG file (any MNE-readable format), streams it
        via :class:`~mne_lsl.player.PlayerLSL` at its native sampling rate,
        and passes the live stream through :meth:`record_main` — exercising
        the full real-time pipeline (artifact correction, feature extraction,
        protocol evaluation) without live hardware.

        Useful for:

        * Offline parameter tuning (test different protocols on the same data).
        * Verifying pipeline latency and modality behaviour.
        * Reproducing a session with modified parameters.

        Parameters
        ----------
        fname : str
            Path to any MNE-readable recording
            (``.fif``, ``.vhdr``, ``.edf``, ``.bdf``, ``.set``, …).
        modality : str | list of str, default "sensor_power"
            NF modality(ies) to extract.
        duration : float | None, default None
            Duration of replay in seconds.  ``None`` infers the full recording
            length from the file.
        winsize : float, default 1.0
            Analysis window length in seconds.
        verbose : bool | str | None, default None
            Override instance verbosity for this call.
        **record_main_kwargs
            Additional keyword arguments forwarded to :meth:`record_main`
            (e.g. ``protocol``, ``modality_params``, ``track_snr``).

        Examples
        --------
        Replay a saved EEG file and run a ZScore protocol offline::

            from mne_rt.protocols import ZScoreProtocol
            nf.replay(
                "sub-01/ses-01/eeg/sub-01_ses-01_task-neurofeedback_eeg.fif",
                modality="sensor_power",
                protocol=ZScoreProtocol(),
                show_nf_signal=False,
                show_raw_signal=False,
            )
            print(f"Artifact rate: {nf.artifact_rate:.1%}")

        .. versionadded:: 1.0.0
        """
        if duration is None:
            _raw_probe = mne.io.read_raw(fname, preload=False, verbose=False)
            duration = float(_raw_probe.times[-1])
            del _raw_probe

        self.connect_to_lsl(
            mock_lsl=True,
            fname=fname,
            n_repeat=1,
            verbose=verbose,
        )
        self.record_main(
            duration=duration,
            modality=modality,
            winsize=winsize,
            verbose=verbose,
            **record_main_kwargs,
        )

    # ------------------------------------------------------------------
    # Multi-run session
    # ------------------------------------------------------------------

    @verbose
    def run_blocks(
        self,
        blocks: list[dict],
        rest_duration: float = 30.0,
        verbose: Union[bool, str, None] = None,
    ) -> list[dict]:
        """Run multiple NF blocks separated by rest periods.

        Each block calls :meth:`record_main` with the parameters given in the
        corresponding dict.  Blocks are separated by ``rest_duration`` seconds
        of silence (no acquisition, no feedback).

        Parameters
        ----------
        blocks : list of dict
            Each dict is passed as keyword arguments to :meth:`record_main`.
            The key ``"rest"`` (optional) overrides ``rest_duration`` for the
            pause *after* that block.  All other keys must be valid
            :meth:`record_main` parameters.

            Minimal example::

                blocks = [
                    {"duration": 120, "modality": "sensor_power"},
                    {"duration": 120, "modality": "sensor_power", "rest": 60},
                    {"duration": 120, "modality": "sensor_power"},
                ]

        rest_duration : float, default 30.0
            Default inter-block rest period in seconds.
        verbose : bool | str | None, default None
            Override instance verbosity for this call.

        Returns
        -------
        all_nf_data : list of dict
            One dict per block, each matching :attr:`nf_data` from that block's
            :meth:`record_main` call.  Also stored as :attr:`block_nf_data`.

        Notes
        -----
        :meth:`save` is called internally at the end of *each* block by
        :meth:`record_main`.  The returned list lets you inspect per-block
        feature time-series without re-loading JSON files.

        Examples
        --------
        Three 2-minute NF runs separated by 30-second rests::

            nf.connect_to_lsl(mock_lsl=True, fname="recording.fif")
            nf.record_baseline(baseline_duration=60)
            results = nf.run_blocks(
                blocks=[
                    {"duration": 120, "modality": "sensor_power",
                     "show_nf_signal": False, "show_raw_signal": False},
                    {"duration": 120, "modality": "sensor_power",
                     "show_nf_signal": False, "show_raw_signal": False},
                    {"duration": 120, "modality": "sensor_power",
                     "show_nf_signal": False, "show_raw_signal": False},
                ],
                rest_duration=30.0,
            )
            for i, block_data in enumerate(results):
                vals = block_data["sensor_power"]
                print(f"Block {i+1}: {len(vals)} windows, mean={sum(vals)/len(vals):.4f}")

        .. versionadded:: 1.0.0
        """
        if not blocks:
            raise ValueError("`blocks` must be a non-empty list of dicts.")

        all_nf_data: list[dict] = []
        all_artifact_rates: list[float] = []
        all_snr_data: list[list] = []

        for i, block in enumerate(blocks):
            block_kwargs = dict(block)
            post_rest = float(block_kwargs.pop("rest", rest_duration))

            logger.info(
                "run_blocks: starting block %d/%d (duration=%.0f s)",
                i + 1, len(blocks),
                block_kwargs.get("duration", 0),
            )
            self.record_main(verbose=verbose, **block_kwargs)
            all_nf_data.append(dict(self.nf_data))
            all_artifact_rates.append(self.artifact_rate)
            all_snr_data.append(list(self.snr_data))

            if i < len(blocks) - 1 and post_rest > 0:
                logger.info(
                    "run_blocks: rest period %.0f s …", post_rest
                )
                time.sleep(post_rest)

        self.block_nf_data = all_nf_data
        self.block_artifact_rates = all_artifact_rates
        self.block_snr_data = all_snr_data
        return all_nf_data

    # ------------------------------------------------------------------
    # Modality-params property
    # ------------------------------------------------------------------

    @property
    def modality_params(self) -> dict:
        """Per-modality parameter overrides applied during the NF session.

        A flat or nested dict that maps modality keys (e.g. ``"sensor_power"``)
        to keyword arguments forwarded to the corresponding feature extractor.
        ``None`` is accepted on assignment and normalised to ``{}``.

        Examples
        --------
        >>> nf.modality_params = {"sensor_power": {"frange": [10, 12]},
        ...                       "erd_ers": {"frange": [8, 13]}}
        """
        return self._modality_params

    @modality_params.setter
    def modality_params(self, params: Optional[dict]) -> None:
        if params is not None and not isinstance(params, dict):
            raise ValueError("`modality_params` must be a dict or None.")
        self._modality_params = params or {}

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _prepare_raw_array(self, data: np.ndarray) -> RawArray:
        """Wrap data in a RawArray; set average EEG reference for EEG data."""
        raw = RawArray(data, self.rec_info, verbose=False)
        if self.data_type == "eeg":
            raw.set_eeg_reference("average", projection=True)
        return raw

    def get_blink_template(
        self,
        max_iter: int = 800,
        method: str = "infomax",
        n_components: Union[int, float, None] = 0.95,
        fit_params: Optional[dict] = None,
        random_state: Optional[int] = 0,
        iclabel_threshold: float = 0.5,
    ) -> None:
        """Identify the eye-blink ICA component from baseline EEG.

        Sets ``self.blink_template`` (ndarray) — the spatial template vector
        for the blink component, used by artifact correction during the main
        session.

        Parameters
        ----------
        max_iter : int, default 800
            Maximum number of ICA fitting iterations.
        method : str, default "infomax"
            ICA algorithm passed to :class:`mne.preprocessing.ICA`.
            Common choices: ``"infomax"``, ``"fastica"``, ``"picard"``.
        n_components : int | float | None, default 0.95
            Number of PCA components before ICA.  A float in (0, 1) retains
            enough components to explain that fraction of variance.  ``None``
            uses all channels.  Falls back to ``5`` if the initial fit fails.
        fit_params : dict | None, default None
            Extra keyword arguments forwarded to the ICA solver.  ``None``
            defaults to ``{"extended": True}`` for infomax (recommended).
        random_state : int | None, default 0
            Seed for reproducible ICA solutions.
        iclabel_threshold : float, default 0.5
            Minimum ICLabel probability for a component to be classified as
            ``"eye blink"``.  Lower values are more permissive; higher values
            require stronger confidence.
        """
        self.blink_template = create_blink_template(
            self.raw_baseline,
            max_iter=max_iter,
            method=method,
            n_components=n_components,
            fit_params=fit_params,
            random_state=random_state,
            iclabel_threshold=iclabel_threshold,
        )

    def run_orica(
        self,
        n_channels: int,
        learning_rate: float = 0.1,
        block_size: int = 256,
        online_whitening: bool = True,
        calibrate_pca: bool = False,
        forgetfac: float = 1.0,
        nonlinearity: str = "tanh",
        random_state: Optional[int] = None,
    ) -> None:
        """Initialise an Online Recursive ICA (ORICA) instance.

        ORICA is a streaming, adaptive ICA algorithm that updates its
        unmixing matrix incrementally as each new EEG block arrives —
        without ever storing the full recording.  It is the preferred
        real-time alternative to offline ICA when the signal statistics
        change over time (non-stationarity).

        Parameters
        ----------
        n_channels : int
            Number of EEG/MEG channels.  Must match the channel count
            of the data passed to each subsequent :meth:`~ant.tools.ORICA.partial_fit`
            or :meth:`~ant.tools.ORICA.fit_transform` call.
        learning_rate : float, default 0.1
            Step size for the online natural-gradient update of the
            unmixing matrix W.  Larger values adapt faster but may
            oscillate; values in [0.01, 0.2] are typically stable.
        block_size : int, default 256
            Number of samples per update block.  Smaller blocks give
            finer temporal resolution at the cost of noisier gradient
            estimates.  Must be ≥ ``n_channels``.
        online_whitening : bool, default True
            If ``True``, a recursive PCA whitening step is applied to
            each block before the ICA update, keeping the algorithm
            numerically stable as signal variance drifts.
        calibrate_pca : bool, default False
            If ``True``, run a batch PCA on the first block to
            initialise the whitening matrix before switching to
            online updates.  Recommended when starting cold with no
            prior covariance estimate.
        forgetfac : float, default 1.0
            Exponential forgetting factor for the online covariance
            estimate (1.0 = no forgetting; 0.99 → slowly decaying
            influence of older samples).  Values < 1 help track
            gradual changes in the mixing matrix.
        nonlinearity : str, default "tanh"
            Score function used in the natural-gradient ICA update.
            ``"tanh"`` works well for super-Gaussian sources (spikes,
            blinks); ``"logcosh"`` is a smooth approximation with
            similar properties.
        random_state : int | None, default None
            Seed for reproducible initialisation of the unmixing matrix W.
            ``None`` uses a random seed.

        See Also
        --------
        ant.tools.ORICA : The underlying ORICA implementation.
        RTStream.get_blink_template : Compute a blink spatial template
            to guide component identification.

        Notes
        -----
        :meth:`run_orica` is called automatically inside
        :meth:`record_baseline` when ``artifact_correction="orica"``
        is set on the :class:`RTStream` instance.  Call it manually
        only if you need to tune the ORICA hyperparameters.
        """
        self.orica = ORICA(
            n_channels=n_channels,
            learning_rate=learning_rate,
            block_size=block_size,
            online_whitening=online_whitening,
            calibrate_pca=calibrate_pca,
            forgetfac=forgetfac,
            nonlinearity=nonlinearity,
            random_state=random_state,
        )

    @verbose
    def fit_gedai(
        self,
        band: tuple[float, float] = (8.0, 13.0),
        shrinkage: float = 0.01,
        use_leadfield: bool = True,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Fit a :class:`~mne_rt.tools.GEDAIDenoiser` from the recorded baseline.

        Must be called after :meth:`record_baseline` and before
        :meth:`record_main` when ``artifact_correction="gedai"``.

        Parameters
        ----------
        band : tuple of float, default (8.0, 13.0)
            Target frequency band ``(low_Hz, high_Hz)`` used when fitting
            in band-filter mode (``use_leadfield=False``).  Ignored when
            ``use_leadfield=True``.
        shrinkage : float, default 0.01
            Tikhonov regularisation strength applied to the reference
            covariance before solving the generalised eigenvalue problem.
            Larger values improve numerical stability at the cost of
            slightly less discriminative spatial filters.
        use_leadfield : bool, default True
            If ``True`` and a forward solution is available
            (i.e., :meth:`compute_inv_operator` has been called), fit in
            **leadfield mode** — the true GEDAI algorithm (Ros et al., 2025).
            The forward gain matrix :math:`\\mathbf{L}` is used as the
            reference covariance :math:`\\mathbf{R} = \\mathbf{L}\\mathbf{L}^\\top`,
            so components that best explain the theoretical brain-source
            model are kept and non-leadfield-aligned components (artifacts)
            are removed.

            If ``False``, or if no forward solution is available, falls back
            to band-filter mode: the GEP is solved between the band-filtered
            and broadband EEG covariances (Cohen, 2022).
        verbose : bool | str | None, default None
            Override the instance-level verbosity for this call.

        Raises
        ------
        RuntimeError
            If :meth:`record_baseline` has not been called yet.

        See Also
        --------
        ant.tools.GEDAIDenoiser : The underlying GED denoiser class.
        RTStream.compute_inv_operator : Computes the forward solution
            required for leadfield mode.

        References
        ----------
        Ros, T., Férat, V., Huang, Y., et al. (2025). Return of the GEDAI:
        Unsupervised EEG Denoising based on Leadfield Filtering. *bioRxiv*.
        https://doi.org/10.1101/2025.10.04.680449

        Examples
        --------
        Leadfield mode (recommended — requires forward solution):

        >>> nf.record_baseline(baseline_duration=120)
        >>> nf.compute_inv_operator()          # builds the forward model
        >>> nf.fit_gedai(use_leadfield=True)
        >>> nf.record_main(duration=600, artifact_correction="gedai")

        Band-filter mode (no MRI required):

        >>> nf.record_baseline(baseline_duration=120)
        >>> nf.fit_gedai(band=(8, 13), use_leadfield=False)
        >>> nf.record_main(duration=600, artifact_correction="gedai")
        """
        if not hasattr(self, "raw_baseline") or self.raw_baseline is None:
            raise RuntimeError("Run record_baseline() before calling fit_gedai().")

        n_channels = len(self.rec_info["ch_names"])
        self.gedai = GEDAIDenoiser(n_channels=n_channels, shrinkage=shrinkage)

        if use_leadfield and hasattr(self, "fwd") and self.fwd is not None:
            L = self.fwd["sol"]["data"]  # shape (n_ch, n_sources)
            self.gedai.fit_from_leadfield(
                data=self.raw_baseline.get_data(),
                leadfield=L,
            )
            logger.info("GEDAI fitted in leadfield mode (Ros et al., 2025).")
        else:
            if use_leadfield:
                logger.warning(
                    "use_leadfield=True but no forward solution found. "
                    "Falling back to band-filter mode. "
                    "Call compute_inv_operator() first to enable leadfield mode."
                )
            self.gedai.fit_from_raw(
                data=self.raw_baseline.get_data(),
                sfreq=self._sfreq,
                band=band,
            )
            logger.info("GEDAI fitted in band-filter mode.")

    @verbose
    def fit_asr(
        self,
        cutoff: float = 5.0,
        window_len: float = 1.0,
        max_dropout_fraction: float = 0.1,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Fit an :class:`~mne_rt.tools.ASRDenoiser` from the recorded baseline.

        Must be called after :meth:`record_baseline` and before
        :meth:`record_main` when ``artifact_correction="asr"``.

        Parameters
        ----------
        cutoff : float, default 5.0
            Rejection threshold in standard deviations above the clean-data
            RMS per component.  Lower values (e.g. 3) are more aggressive;
            higher values (e.g. 10) are more conservative.
            The Mullen et al. (2015) default is 5.0.
        window_len : float, default 1.0
            Calibration window length in seconds.  Shorter windows give more
            estimates but with higher variance.  Recommend 0.5–1.0 s.
        max_dropout_fraction : float, default 0.1
            Fraction of calibration windows with the highest total power
            to discard before estimating clean statistics.  ``0.1`` keeps
            the 90 % cleanest windows.
        verbose : bool | str | None, default None
            Override the instance-level verbosity for this call.

        Raises
        ------
        RuntimeError
            If :meth:`record_baseline` has not been called yet.

        See Also
        --------
        ant.tools.ASRDenoiser : The underlying ASR implementation.
        RTStream.record_baseline : Records and stores the baseline segment.

        References
        ----------
        Mullen, T. R., et al. (2015). Real-Time Neuroimaging and Cognitive
        Monitoring Using Wearable Dry EEG. *IEEE Trans. Biomed. Eng.*,
        62(11), 2553–2567.

        Examples
        --------
        >>> nf.record_baseline(baseline_duration=120)
        >>> nf.fit_asr(cutoff=5.0)
        >>> nf.record_main(duration=600, artifact_correction="asr")
        """
        if not hasattr(self, "raw_baseline") or self.raw_baseline is None:
            raise RuntimeError("Run record_baseline() before calling fit_asr().")

        self.asr = ASRDenoiser(
            cutoff=cutoff,
            max_dropout_fraction=max_dropout_fraction,
        )
        self.asr.fit(
            self.raw_baseline.get_data(),
            sfreq=float(self.raw_baseline.info["sfreq"]),
            window_len=window_len,
        )
        logger.info("ASR fitted (cutoff=%.1f) from baseline.", cutoff)

    @verbose
    def fit_maxwell(
        self,
        int_order: int = 8,
        ext_order: int = 3,
        origin: Union[str, tuple] = "auto",
        st_duration: Optional[float] = None,
        st_correlation: float = 0.98,
        st_update_interval: int = 1,
        calibration: Optional[str] = None,
        cross_talk: Optional[str] = None,
        coord_frame: str = "head",
        regularize: Optional[str] = "in",
        mag_scale: float = 100.0,
        empty_room_raw: Optional[Any] = None,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Prepare real-time Maxwell filtering (SSS / tSSS) for MEG data.

        Computes the Signal Space Separation (SSS) projection operator once
        from sensor geometry.  No baseline recording is required — the SSS
        basis is entirely geometric.  Call this method at any point before
        :meth:`record_main` when ``artifact_correction="maxwell"``.

        Parameters
        ----------
        int_order : int, default 8
            Internal spherical-harmonic expansion order.
            ``8`` → 80 internal moments (MNE default, adequate for all
            standard MEG systems).
        ext_order : int, default 3
            External expansion order (``3`` → 16 external moments).
        origin : array-like of shape (3,) | "auto", default "auto"
            SSS expansion origin in metres (head frame).  ``"auto"`` places
            it at the geometric centre of the sensor array.
        st_duration : float | None, default None
            Temporal SSS (tSSS) buffer duration in seconds.

            * ``None`` — spatial SSS only.  The pre-computed projector is
              applied per chunk via a single matrix multiply.
            * ``float`` — tSSS mode.  A rolling buffer of ``st_duration``
              seconds feeds MNE's full tSSS every ``st_update_interval``
              chunks for temporal interference suppression.  The spatial SSS
              projector is always applied first (zero-latency stage 1).

            Typical values: 10 s for persistent shielding leakage, 1–4 s
            for moving subjects.
        st_correlation : float, default 0.98
            Minimum inside-outside correlation for tSSS suppression.
        st_update_interval : int, default 1
            Apply tSSS every *N* chunks.  Increase to reduce CPU load
            when ``winsize`` is small.
        calibration : str | None, default None
            Path to the fine-calibration ``.dat`` file.  Strongly
            recommended for Elekta/MEGIN systems — it corrects sensor
            position and orientation errors.
        cross_talk : str | None, default None
            Path to the cross-talk ``.fif`` file.  Compensates for flux
            leakage between adjacent sensors.
        coord_frame : {"head", "meg"}, default "head"
            Coordinate frame for the spherical-harmonic expansion.
        regularize : {"in", None}, default "in"
            Internal-moment regularisation passed to MNE.  ``"in"``
            (Tikhonov) is recommended for most datasets.
        mag_scale : float, default 100.0
            Magnetometer/gradiometer balance factor in the SSS decomposition.
        empty_room_raw : mne.io.Raw | None, default None
            Empty-room recording (shielded room, no subject).  When
            provided, the SSS operator is extracted via **system
            identification** so that noise-informed regularisation from the
            empty room is incorporated into the cached matrix.  This
            improves suppression of spatially correlated sensor noise and
            is equivalent to passing ``noise_cov`` to
            :func:`~mne.preprocessing.maxwell_filter`.

            When ``None``, geometric regularisation only is used via
            :func:`~mne.preprocessing.compute_maxwell_basis` (faster,
            adequate when shielding is good).
        verbose : bool | str | None, default None
            Override instance-level verbosity.

        Raises
        ------
        RuntimeError
            If no LSL stream has been connected yet (``connect_to_lsl()``
            must be called first so that ``self.rec_info`` is populated).
        ValueError
            If this instance was initialised with ``data_type="eeg"``
            (Maxwell filtering is MEG-only).

        Notes
        -----
        See :class:`~mne_rt.tools.RTMaxwellFilter` for the underlying filter
        class and :meth:`fit_asr` for the EEG alternative (ASR).

        Unlike :meth:`fit_asr` and :meth:`fit_gedai`, **no baseline
        recording is needed**.  The SSS operator depends only on sensor
        positions, not on brain signal statistics.  Simply call::

            nf.connect_to_lsl()
            nf.fit_maxwell()          # operator ready in seconds
            nf.record_main(duration=600, artifact_correction="maxwell")

        If a fine-calibration file is available, passing it via
        ``calibration`` typically reduces the RMS noise floor by 10–20 %
        compared to the uncalibrated SSS.

        References
        ----------
        Taulu, S., Kajola, M., & Simola, J. (2004). Suppression of
        interference and artifacts by the Signal Space Separation Method.
        *Brain Topogr.*, 16(4), 269–275.

        Taulu, S., & Simola, J. (2006). Spatiotemporal signal space
        separation method for rejecting nearby interference in MEG
        measurements. *Phys. Med. Biol.*, 51(7), 1759–1768.

        Examples
        --------
        SSS-only (fastest, no latency):

        >>> nf.connect_to_lsl()
        >>> nf.fit_maxwell()
        >>> nf.record_main(duration=600, artifact_correction="maxwell")

        tSSS with fine calibration and empty room:

        >>> import mne
        >>> er_raw = mne.io.read_raw_fif("empty_room.fif", preload=True)
        >>> nf.connect_to_lsl()
        >>> nf.fit_maxwell(
        ...     st_duration=10.0,
        ...     calibration="sss_cal.dat",
        ...     cross_talk="ct_sparse.fif",
        ...     empty_room_raw=er_raw,
        ... )
        >>> nf.record_main(duration=600, artifact_correction="maxwell")
        """
        if not hasattr(self, "rec_info") or self.rec_info is None:
            raise RuntimeError(
                "connect_to_lsl() must be called before fit_maxwell() "
                "so that sensor info is available."
            )

        self.maxwell_filter = RTMaxwellFilter(
            int_order=int_order,
            ext_order=ext_order,
            origin=origin,
            st_duration=st_duration,
            st_correlation=st_correlation,
            st_update_interval=st_update_interval,
            calibration=calibration,
            cross_talk=cross_talk,
            coord_frame=coord_frame,
            regularize=regularize,
            mag_scale=mag_scale,
        )
        self.maxwell_filter.fit(self.rec_info, empty_room_raw=empty_room_raw)
        logger.info(
            "Maxwell filter ready: mode=%s, n_internal=%d.",
            self.maxwell_filter.mode,
            self.maxwell_filter.n_use_in,
        )

    def compute_inv_operator(
        self,
        loose: float = 0.2,
        depth: float = 0.8,
        noise_cov_method: str = "ad_hoc",
        reg: float = 0.1,
    ) -> None:
        """Compute and save the inverse operator for source localisation.

        Wraps MNE's forward-solution and inverse-operator pipeline.
        Results are saved to ``<subjects_dir>/<subject_id>/inv/``.

        Parameters
        ----------
        loose : float, default 0.2
            Orientation constraint for cortical source dipoles.
            ``0`` = fixed (normal to surface), ``1`` = fully free,
            ``0.2`` = loose (recommended — allows slight tangential
            component while favouring surface-normal currents).
        depth : float | None, default 0.8
            Depth-weighting exponent to compensate for the MNE bias
            towards superficial sources.  ``None`` disables depth
            weighting; ``0.8`` is the MNE default.  Higher values
            suppress surface bias more aggressively.
        noise_cov_method : str, default "ad_hoc"
            How to estimate the noise covariance used by the inverse
            operator.

            * ``"ad_hoc"`` *(default)* — :func:`mne.make_ad_hoc_cov`
              creates a diagonal covariance from standard sensor-noise
              floors (1 µV for EEG, 20 fT for MEG grads, 200 fT/cm for
              MEG mags).  Recommended when no dedicated noise recording
              is available: the resting-state baseline contains brain
              signal, not pure noise, so fitting an empirical covariance
              on it conflates signal and noise.
            * ``"empirical"`` — sample covariance from the baseline raw.
            * ``"shrunk"`` — Ledoit-Wolf shrinkage; more stable when
              n_channels ≈ n_samples.
            * ``"diagonal_fixed"`` — diagonal regularisation.
        reg : float, default 0.1
            Per-channel-type regularisation added to the noise covariance
            diagonal via :func:`mne.cov.regularize`.  Applied only when
            ``noise_cov_method != "ad_hoc"``.  Set to ``0`` to skip.
            Helps numerical stability when the baseline recording is short
            relative to the number of channels.

        Raises
        ------
        RuntimeError
            If :meth:`record_baseline` has not been called yet.

        See Also
        --------
        mne.make_inverse_operator : Underlying MNE function.
        mne.compute_raw_covariance : Noise covariance estimation.
        """
        self._ensure_dirs()
        self.inv, self.fwd, self.noise_cov = _compute_inv_operator(
            self.raw_baseline,
            subject_fs_id=self.subject_fs_id,
            subjects_fs_dir=self.subjects_fs_dir,
            data_type=self.data_type,
            loose=loose,
            depth=depth,
            noise_cov_method=noise_cov_method,
            reg=reg,
        )
        inv_dir = self.subject_dir / "inv"
        _stem = f"sub-{self.subject_id}_ses-{self.session}_task-baseline"
        write_inverse_operator(
            fname=inv_dir / f"{_stem}_inv.fif",
            inv=self.inv,
            overwrite=True,
        )
        write_forward_solution(
            fname=inv_dir / f"{_stem}_fwd.fif",
            fwd=self.fwd,
            overwrite=True,
        )
        write_cov(
            fname=inv_dir / f"{_stem}_cov.fif",
            cov=self.noise_cov,
            overwrite=True,
        )

    # ------------------------------------------------------------------
    # LSL stream viewer
    # ------------------------------------------------------------------

    def open_stream_viewer(self, bufsize: float = 0.2) -> None:
        """Open the mne-lsl StreamViewer for raw M/EEG monitoring.

        Parameters
        ----------
        bufsize : float, default 0.2
            Display window size (s).
        """
        import subprocess
        import sys

        # StreamViewer.start() calls sys.exit(app.exec_()), which would block
        # the main thread and prevent the acquisition thread from ever starting.
        # Run it in a separate process so the main loop continues uninterrupted.
        code = (
            "from mne_lsl.stream_viewer import StreamViewer; "
            f"StreamViewer(stream_name={self.stream.name!r}).start({bufsize})"
        )
        subprocess.Popen([sys.executable, "-c", code])
        time.sleep(0.5)  # give the viewer process time to connect to the stream

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def save(
        self,
        nf_data: bool = True,
        acq_delay: bool = True,
        artifact_delay: bool = True,
        method_delay: bool = True,
        raw_data: bool = False,
        bids_tsv: bool = False,
        format: str = "json",
        delay_include_trace: bool = False,
    ) -> dict[str, Path]:
        """Save session outputs and disconnect the LSL stream.

        All output files share the stem
        ``sub-<ID>_ses-<session>`` set at the
        start of :meth:`record_main`, so multiple runs in a day never
        overwrite each other.

        Parameters
        ----------
        nf_data : bool, default True
            Save feature time-series as
            ``beh/<stem>_task-neurofeedback_beh.json``.  The JSON contains a
            ``"meta"`` block (subject, modalities, sfreq, duration, artifact
            correction, artifact rate, SNR, start/end timestamps) and a
            ``"data"`` block with per-modality value lists.  When
            ``track_snr=True`` was passed to :meth:`record_main`, the
            per-window SNR series is included in ``"data"`` under the key
            ``"snr_db"``.  Reward magnitudes delivered by the protocol are
            saved under ``"reward_<modality>"`` keys (one per modality).
        acq_delay : bool, default True
            Include acquisition-loop timing in the delays file (only
            written when ``estimate_delays=True`` was set in
            :meth:`record_main`).
        artifact_delay : bool, default True
            Include artifact-correction timing in the delays file.
        method_delay : bool, default True
            Include per-modality feature-extraction timing in the delays
            file.
        raw_data : bool, default False
            Save the pre-correction M/EEG acquired during the main
            session as ``eeg/<stem>_task-neurofeedback_eeg.fif``.
        bids_tsv : bool, default False
            Additionally write a BIDS-compliant tab-separated values file
            ``beh/<stem>_task-neurofeedback_beh.tsv`` alongside the JSON.
            Columns are: one per modality, ``reward_<modality>`` per
            modality, and ``snr_db`` when available.  Each row is one
            analysis window.  This file passes a BIDS validator and can be
            loaded directly by EEGLAB, Fieldtrip, or any TSV reader.
        format : str, default "json"
            Serialisation format for NF data and delays.  Currently only
            ``"json"`` is supported.
        delay_include_trace : bool, default False
            Embed the full per-window delay trace (ms) in the delays JSON
            alongside the summary statistics.  Can be large for long sessions.

        Returns
        -------
        saved : dict[str, Path]
            Maps output type (``"nf_data"``, ``"nf_tsv"``, ``"delays"``,
            ``"raw"``) to the saved file path.  Only keys for files actually
            written are included.

        Notes
        -----
        Disconnects the LSL stream as a side effect — the stream is not
        needed after the session ends.
        """
        if hasattr(self, "stream") and getattr(self.stream, "connected", False):
            self.stream.disconnect()
        if hasattr(self, "_mock_player"):
            try:
                self._mock_player.stop()
            except Exception:
                pass

        self._ensure_dirs()

        stem = getattr(
            self, "_session_stem",
            f"sub-{self.subject_id}_ses-{self.session}",
        )
        saved: dict[str, Path] = {}

        # ── helpers ──────────────────────────────────────────────────────

        def _summarize(vals: list, include_trace: bool = False) -> dict:
            """Convert a list of wall-clock seconds to a ms summary dict."""
            if not vals:
                return {"n": 0}
            arr = np.asarray(vals, dtype=float) * 1000.0
            out: dict = {
                "mean_ms": round(float(arr.mean()), 4),
                "std_ms":  round(float(arr.std()), 4),
                "min_ms":  round(float(arr.min()), 4),
                "max_ms":  round(float(arr.max()), 4),
                "p95_ms":  round(float(np.percentile(arr, 95)), 4),
                "n":       int(len(arr)),
            }
            if include_trace:
                out["trace_ms"] = [round(v, 4) for v in arr.tolist()]
            return out

        def _ser(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            return v

        # ── feature time-series ───────────────────────────────────────

        if nf_data and hasattr(self, "nf_data"):
            start_iso = (
                self._session_start_time.isoformat()
                if hasattr(self, "_session_start_time") else None
            )
            payload: dict = {
                "meta": {
                    "subject_id":          self.subject_id,
                    "session":             self.session,
                    "data_type":           self.data_type,
                    "modalities":          list(self.nf_data.keys()),
                    "sfreq_hz":            float(getattr(self, "_sfreq", 0)),
                    "winsize_s":           float(getattr(self, "winsize", 0)),
                    "duration_s":          float(getattr(self, "duration", 0)),
                    "n_windows":           {m: len(v) for m, v in self.nf_data.items()},
                    "artifact_correction": str(self.artifact_correction),
                    "artifact_rate":       getattr(self, "artifact_rate", None),
                    "start_time":          start_iso,
                    "end_time":            datetime.datetime.now(
                                               datetime.timezone.utc
                                           ).isoformat(),
                },
                "data": {
                    m: [_ser(v) for v in vals]
                    for m, vals in self.nf_data.items()
                },
            }
            snr = getattr(self, "snr_data", [])
            if snr:
                payload["data"]["snr_db"] = [_ser(v) for v in snr]
            reward = getattr(self, "reward_data", {})
            for m, vals in reward.items():
                if vals:
                    payload["data"][f"reward_{m}"] = [_ser(v) for v in vals]
            p = self.subject_dir / "beh" / f"{stem}_task-neurofeedback_beh.json"
            with open(p, "w") as fh:
                json.dump(payload, fh, indent=2)
            saved["nf_data"] = p

            if bids_tsv:
                cols = list(self.nf_data.keys())
                for m in reward:
                    if reward[m]:
                        cols.append(f"reward_{m}")
                if snr:
                    cols.append("snr_db")
                n_rows = max(len(self.nf_data[m]) for m in self.nf_data) if self.nf_data else 0
                tsv_p = self.subject_dir / "beh" / f"{stem}_task-neurofeedback_beh.tsv"
                combined = {**self.nf_data,
                            **{f"reward_{m}": v for m, v in reward.items() if v},
                            **({"snr_db": snr} if snr else {})}
                with open(tsv_p, "w") as fh:
                    fh.write("\t".join(cols) + "\n")
                    for i in range(n_rows):
                        row = []
                        for col in cols:
                            vals_col = combined.get(col, [])
                            row.append(str(_ser(vals_col[i])) if i < len(vals_col) else "n/a")
                        fh.write("\t".join(row) + "\n")
                saved["nf_tsv"] = tsv_p

        # ── Timing / delay statistics ────────────────────────────────────

        if getattr(self, "estimate_delays", False) and (
            acq_delay or artifact_delay or method_delay
        ):
            delays_payload: dict = {}
            if acq_delay and hasattr(self, "acq_delays"):
                delays_payload["acquisition"] = _summarize(
                    self.acq_delays, delay_include_trace
                )
            if artifact_delay and hasattr(self, "artifact_delays"):
                delays_payload["artifact_correction"] = _summarize(
                    self.artifact_delays, delay_include_trace
                )
            if method_delay and hasattr(self, "method_delays"):
                delays_payload["methods"] = {
                    m: _summarize([float(v) for v in vals], delay_include_trace)
                    for m, vals in self.method_delays.items()
                }
            p = self.subject_dir / "delays" / f"{stem}_task-neurofeedback_delays.json"
            with open(p, "w") as fh:
                json.dump(delays_payload, fh, indent=2)
            saved["delays"] = p

        # ── Raw M/EEG (pre-correction) ───────────────────────────────────

        if raw_data:
            chunks = getattr(self, "_raw_chunks", None)
            if chunks:
                raw_all = np.concatenate(chunks, axis=1)
                raw_nf = RawArray(raw_all, self.rec_info, verbose=False)
                p = self.subject_dir / "eeg" / f"{stem}_task-neurofeedback_eeg.fif"
                raw_nf.save(p, overwrite=True, verbose=False)
                saved["raw"] = p
            else:
                warn(
                    "save(raw_data=True) but no raw chunks were accumulated. "
                    "Did record_main() complete successfully?",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return saved

    @classmethod
    def load_nf_data(cls, path: Union[str, Path]) -> dict:
        """Load a saved NF data file produced by :meth:`save`.

        Parameters
        ----------
        path : str | Path
            Path to a ``*_nf.json`` file.

        Returns
        -------
        payload : dict
            Dict with two keys:

            * ``"meta"`` — session metadata: subject ID, session
              type, modalities, sfreq, winsize, duration, n_windows,
              artifact correction, start/end timestamps.
            * ``"data"`` — ``{modality: [values, …]}`` per-modality lists.

        Examples
        --------
        >>> d = RTStream.load_nf_data("subjects/sub-sub01/ses-01/beh/sub-sub01_ses-01_task-neurofeedback_beh.json")
        >>> import numpy as np
        >>> alpha = np.array(d["data"]["sensor_power"])
        >>> print(f"Mean alpha power: {alpha.mean():.3e}")
        >>> print(f"Session sfreq: {d['meta']['sfreq_hz']} Hz")
        """
        with open(Path(path)) as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def create_report(
        self,
        overwrite: bool = True,
        include_psd: bool = True,
        include_nf_signal: bool = True,
        open_browser: bool = False,
    ) -> Path:
        """Generate an HTML MNE report for the session.

        Produces a self-contained HTML file containing baseline recording
        info, sensor layouts, optional PSD, feature time-series, and
        brain-label diagrams for source-space modalities.

        Parameters
        ----------
        overwrite : bool, default True
            Overwrite an existing report file with the same name.
        include_psd : bool, default True
            If ``True``, add a baseline power-spectral-density plot
            (1–40 Hz) to the report.
        include_nf_signal : bool, default True
            If ``True`` and :attr:`nf_data` is populated (i.e.,
            :meth:`record_main` has been run), add a time-series plot
            of the NF feature values for each modality.
        open_browser : bool, default False
            If ``True``, open the saved report in the default web browser
            immediately after saving.

        Returns
        -------
        report_path : pathlib.Path
            Full path of the saved HTML report file.

        Raises
        ------
        RuntimeError
            If :meth:`record_baseline` has not been called yet.
        """
        modalities = (
            [self.modality] if isinstance(self.modality, str) else list(self.modality)
        )
        report = Report(title=f"Neurofeedback Session — {', '.join(modalities)}")

        # ── Baseline recording ────────────────────────────────────────────
        report.add_raw(
            self.raw_baseline,
            title="Baseline recording",
            psd=False,
            butterfly=False,
        )

        # ── Baseline PSD ─────────────────────────────────────────────────
        if include_psd:
            fig_psd, ax_psd = plt.subplots(figsize=(10, 4))
            self.raw_baseline.compute_psd(fmax=40.0).plot(axes=ax_psd, show=False)
            ax_psd.set_title("Baseline PSD (1–40 Hz)")
            report.add_figure(fig=fig_psd, title="Baseline PSD")
            plt.close(fig_psd)

        # ── Sensor layouts & brain labels per modality ────────────────────

        '''
        source_modalities = {"source_power", "source_connectivity", "source_graph"}

        for mod in modalities:
            params = self.mod_params_dict.get(mod, {})
            if mod not in source_modalities:
                bads = self.picks if self.picks is not None else self.rec_info["ch_names"]
                self.rec_info["bads"].extend(bads)

                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122, projection="3d")
                mne.viz.plot_sensors(info=self.rec_info, kind="topomap", axes=ax1, show=False)
                mne.viz.plot_sensors(info=self.rec_info, kind="3d", axes=ax2, show=False)
                ax2.axis("off")
                self.rec_info["bads"] = []
                report.add_figure(fig=fig, title=f"Sensors — {mod}")
                plt.close(fig)
            else:
                if mod == "source_power":
                    fig_brain = plot_glass_brain(bl1=params.get("brain_label"))
                else:
                    fig_brain = plot_glass_brain(
                        bl1=params.get("brain_label_1"),
                        bl2=params.get("brain_label_2"),
                    )
                report.add_figure(fig=fig_brain, title=f"Brain labels — {mod}")
                plt.close(fig_brain)
        '''

        # ── signal time-series ─────────────────────────────────────────
        if include_nf_signal and hasattr(self, "nf_data") and self.nf_data:
            fig_nf, axes_nf = plt.subplots(
                len(modalities), 1,
                figsize=(12, 2.5 * len(modalities)),
                sharex=True,
                squeeze=False,
            )
            for i, mod in enumerate(modalities):
                vals = self.nf_data.get(mod, [])
                ax = axes_nf[i, 0]
                ax.plot(vals, lw=1.5)
                ax.set_ylabel(mod, fontsize=9)
                ax.grid(True, alpha=0.3)
            axes_nf[-1, 0].set_xlabel("Window index")
            fig_nf.suptitle("feature time-series", fontsize=11)
            fig_nf.tight_layout()
            report.add_figure(fig=fig_nf, title="signal")
            plt.close(fig_nf)

        # ── Summary table ─────────────────────────────────────────────────
        n_windows = max(
            (len(v) for v in self.nf_data.values() if isinstance(v, list)),
            default=0,
        ) if hasattr(self, "nf_data") else 0
        summary_html = (
            "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
            f"<tr><th>Subject</th><td>{self.subject_id}</td></tr>"
            f"<tr><th>Session</th><td>{self.session}</td></tr>"
            f"<tr><th>Modalities</th><td>{', '.join(modalities)}</td></tr>"
            f"<tr><th>NF windows</th><td>{n_windows}</td></tr>"
            f"<tr><th>Artifact correction</th><td>{self.artifact_correction}</td></tr>"
            "</table>"
        )
        report.add_html(html=summary_html, title="Session summary")

        fname = f"sub-{self.subject_id}_ses-{self.session}_report.html"
        report_path = self.subject_dir / "reports" / fname
        report.save(report_path, overwrite=overwrite, open_browser=open_browser)
        return report_path

    def __del__(self) -> None:
        """Stop the mock player and disconnect the stream on garbage collection."""
        if getattr(self, "_mock_player", None) is not None:
            try:
                self._mock_player.stop()
            except Exception:
                pass
        if getattr(self, "stream", None) is not None:
            try:
                if getattr(self.stream, "connected", False):
                    self.stream.disconnect()
            except Exception:
                pass
