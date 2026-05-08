"""Core session class for the Advanced Neurofeedback Toolbox (ANT).

This module provides :class:`NFRealtime`, the top-level object that
orchestrates LSL streaming, artifact rejection, feature extraction, and
real-time visualisation for an M/EEG neurofeedback session.

Typical workflow
----------------
::

    nf = NFRealtime(subject_id="sub01", visit=1, session="main",
                    subjects_dir="/data/subjects", montage="easycap-M1")
    nf.connect_to_lsl()
    nf.record_baseline(baseline_duration=120)
    nf.record_main(duration=600, modality=["sensor_power", "erd_ers"])

Classes
-------
NFRealtime
    Main session controller — inherits all feature-extraction methods from
    :class:`~ant.modalities.ModalityMixin`.
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
    read_labels_from_annot,
    write_cov,
    write_forward_solution,
)
from mne.channels import get_builtin_montages, read_dig_captrak
from mne.io import RawArray
from mne.minimum_norm import (
    apply_inverse_raw,
    read_inverse_operator,
    write_inverse_operator,
)
from mne_lsl.lsl import local_clock
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream

from ant.modalities import ModalityMixin
from ant._logging import logger, set_log_level, verbose
from ant.tools import (
    _compute_inv_operator,
    create_blink_template,
    get_params,
    plot_glass_brain,
    remove_blinks_lms,
)
from ant.tools.gedai import GEDAIDenoiser
from ant.tools.orica import ORICA
from ant.viz import BrainPlot, NFSignalPlot

# Package root — resolves correctly both in editable installs and installed wheels
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent  # src/ant → src → repo root


class NFRealtime(ModalityMixin):
    """Real-time M/EEG neurofeedback session controller.

    Orchestrates LSL streaming, optional artifact rejection, parallel
    feature extraction, and real-time visualisation for a complete
    neurofeedback session.  Inherits all feature-extraction methods from
    :class:`~ant.modalities.ModalityMixin`.

    Parameters
    ----------
    subject_id : str
        Unique subject identifier (non-empty string).
    visit : int
        Visit number (≥ 1).  Used to name saved files.
    session : {"baseline", "main"}
        Session type.  Baseline sessions record resting-state data;
        main sessions run the closed-loop NF loop.
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
    filtering : bool, default False
        Apply online band-pass filtering during the main session.
    l_freq : float, default 1.0
        Low cut-off frequency in Hz (used only when ``filtering=True``).
    h_freq : float, default 40.0
        High cut-off frequency in Hz (used only when ``filtering=True``).
    artifact_correction : {False, "orica", "lms", "gedai"}, default False
        Real-time artifact correction strategy.

        * ``False``     — no correction
        * ``"lms"``     — adaptive LMS regression on a frontal reference
          channel (EEG only)
        * ``"orica"``   — online recursive ICA
          (:class:`~ant.tools.ORICA`)
        * ``"gedai"``   — GED-based decomposition
          (:class:`~ant.tools.GEDAIDenoiser`); requires
          :meth:`fit_gedai` to be called after baseline.

    ref_channel : str, default "Fp1"
        Reference channel for LMS artifact correction.
    save_raw : bool, default True
        Save raw data to ``<subjects_dir>/<subject_id>/raw/``.
    save_nf_signal : bool, default True
        Save extracted NF feature time-series as JSON.
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
    ant.viz.NFSignalPlot : Scrolling NF signal display.
    ant.viz.BrainPlot : 3-D brain activation display.

    Notes
    -----
    **Typical workflow**::

        nf = NFRealtime("sub01", visit=1, session="main",
                        subjects_dir="/data/subjects",
                        montage="easycap-M1")
        nf.connect_to_lsl()
        nf.record_baseline(baseline_duration=120)
        nf.record_main(duration=600, modality=["sensor_power", "erd_ers"])

    The main NF loop runs M/EEG acquisition in a background daemon thread
    and drives all visualisation windows (StreamViewer, NF signal plot,
    brain plot) from the Qt event loop on the main thread via a 33 ms pump
    timer, ensuring all three windows are truly parallel and non-blocking.

    .. versionadded:: 1.0.0
    """

    VALID_SESSIONS = {"baseline", "main"}
    VALID_ARTIFACT_METHODS = {False, "orica", "lms", "gedai"}
    VALID_DATA_TYPES = {"eeg", "meg"}

    _SENSOR_POWER_SCALE: dict[str, float] = {"eeg": 1e-12, "meg": 1e-24}

    def __init__(
        self,
        subject_id: str,
        visit: int,
        session: str,
        subjects_dir: str,
        montage: Optional[str],
        data_type: str = "eeg",
        mri: bool = False,
        subject_fs_id: str = "fsaverage",
        subjects_fs_dir: Optional[str] = None,
        filtering: bool = False,
        l_freq: float = 1.0,
        h_freq: float = 40.0,
        artifact_correction: Union[bool, str] = False,
        ref_channel: str = "Fp1",
        save_raw: bool = True,
        save_nf_signal: bool = True,
        config_file: Optional[str] = None,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        if not subject_id or not isinstance(subject_id, str):
            raise ValueError("`subject_id` must be a non-empty string.")
        if not isinstance(visit, int) or visit < 1:
            raise ValueError("`visit` must be a positive integer (≥ 1).")
        if session not in self.VALID_SESSIONS:
            raise ValueError(
                f"`session` must be one of {self.VALID_SESSIONS}, got {session!r}."
            )
        if data_type not in self.VALID_DATA_TYPES:
            raise ValueError(
                f"`data_type` must be one of {self.VALID_DATA_TYPES}, got {data_type!r}."
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
        if not isinstance(filtering, bool):
            raise ValueError("`filtering` must be a boolean.")
        if artifact_correction not in self.VALID_ARTIFACT_METHODS:
            raise ValueError(
                f"`artifact_correction` must be one of "
                f"{self.VALID_ARTIFACT_METHODS}, got {artifact_correction!r}."
            )
        if artifact_correction == "lms" and data_type == "meg":
            raise ValueError("LMS artifact correction is only supported for EEG.")
        if not isinstance(save_raw, bool):
            raise ValueError("`save_raw` must be a boolean.")
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
        self.visit = visit
        self.session = session
        self.subjects_dir = subjects_dir
        self.montage = montage
        self.data_type = data_type
        self.mri = mri
        self.subject_fs_id = subject_fs_id
        self.subjects_fs_dir = subjects_fs_dir
        self.filtering = filtering
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.artifact_correction = artifact_correction
        self.ref_channel = ref_channel
        self.save_raw = save_raw
        self.save_nf_signal = save_nf_signal
        self.config_file = config
        self.verbose = verbose

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
        timeout: float = 5.0,
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
            File path for mock streaming.  ``None`` uses the bundled
            sample recording at ``data/sample/sample_data.vhdr``.
        n_repeat : int | float, default ``np.inf``
            How many times to loop the mock recording.
        bufsize_baseline : int, default 4
            LSL buffer size in seconds for baseline sessions.
        bufsize_main : int, default 3
            LSL buffer size in seconds for main sessions.
        acquisition_delay : float, default 0.001
            Seconds between acquisition polling attempts.
        timeout : float, default 5.0
            Maximum wait time in seconds for the LSL connection.
        stream_name : str | None, default None
            Connect by stream name (e.g. ``"neuromag2lsl"`` for MEG devices).
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
        also exposed directly on the :class:`NFRealtime` instance after
        connection.

        Examples
        --------
        Connect to a live EEG amplifier::

            nf.connect_to_lsl()

        Simulate from a BrainVision file::

            nf.connect_to_lsl(mock_lsl=True, fname="path/to/data.vhdr")
        """
        self.subject_dir = Path(self.subjects_dir) / self.subject_id
        self.subject_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self, "stream") and getattr(self.stream, "connected", False):
            self.stream.disconnect()

        if mock_lsl and fname is None:
            fname = _REPO_ROOT / "data" / "sample" / "sample_data.vhdr"

        self.bufsize = bufsize_baseline if self.session == "baseline" else bufsize_main

        if self.montage is not None and Path(str(self.montage)).is_file():
            self.montage = read_dig_captrak(self.montage)

        self.source_id = uuid.uuid4().hex
        if mock_lsl:
            Player(
                fname,
                chunk_size=chunk_size,
                n_repeat=n_repeat,
                source_id=self.source_id,
            ).start()

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
        Output files written to ``<subjects_dir>/<subject_id>/``:

        * ``baseline/visit_<N>-raw.fif`` — baseline raw recording
        * ``inv/visit_<N>-inv.fif`` — inverse operator
        * ``inv/visit_<N>-fwd.fif`` — forward solution
        * ``inv/visit_<N>-cov.fif`` — noise covariance

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

        baseline_dir = self.subject_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        raw_baseline.save(baseline_dir / f"visit_{self.visit}-raw.fif", overwrite=True)
        self.raw_baseline = raw_baseline

        (self.subject_dir / "inv").mkdir(parents=True, exist_ok=True)
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
        show_brain_activation: bool = False,
        brain_mode: str = "power",
        brain_freq_range: tuple[float, float] = (8.0, 13.0),
        use_ring_buffer: bool = False,
        osc_sender: Optional[Any] = None,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Stream M/EEG, extract neural features, and drive NF visualisation.

        This is the main closed-loop entry point.  It:

        1. Prepares each requested modality (calls ``_<modality>_prep``).
        2. Opens the selected visualisation windows (StreamViewer, NF signal
           plot, brain plot) on the main thread.
        3. Starts a background daemon thread that continuously fetches data,
           runs artifact correction, and computes NF features in parallel via
           a thread pool.
        4. Drives all windows from the Qt event loop via a 33 ms pump timer.
        5. Saves raw data and NF feature time-series to disk when finished.

        Parameters
        ----------
        duration : float
            Total recording length in seconds.
        modality : str | list of str, default "sensor_power"
            NF feature(s) to extract.  Must match keys in
            ``config_methods.yml``.  Multiple modalities are extracted in
            parallel.  Available modalities:
            ``"sensor_power"``, ``"band_ratio"``, ``"source_power"``,
            ``"sensor_connectivity"``, ``"source_connectivity"``,
            ``"sensor_graph"``, ``"source_graph"``, ``"entropy"``,
            ``"argmax_freq"``, ``"individual_peak_power"``,
            ``"cfc_sensor"``, ``"erd_ers"``, ``"laterality"``,
            ``"hjorth"``, ``"spectral_centroid"``.
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
            Open the mne-lsl :class:`~mne_lsl.stream_viewer.StreamViewer`
            for live raw signal inspection.
        show_nf_signal : bool, default True
            Show the :class:`~ant.viz.NFSignalPlot` real-time NF monitor.
        time_window : float, default 10.0
            Visible time range in seconds for the NF signal plot.
        show_brain_activation : bool, default False
            Show the :class:`~ant.viz.BrainPlot` 3-D brain activation
            display (requires ``subjects_fs_dir`` and a fitted inverse
            operator).
        brain_mode : {"power", "activation"}, default "power"
            Source-space display mode.  ``"power"`` shows mean squared
            amplitude; ``"activation"`` shows mean amplitude.
        brain_freq_range : (float, float), default (8.0, 13.0)
            Frequency band in Hz used to band-pass the data before computing
            source power for the brain display.
        use_ring_buffer : bool, default False
            Use a sliding ring-buffer acquisition loop (50 % overlap) instead
            of discrete fixed-length window pulls.  Increases update rate at
            the cost of higher CPU load.
        osc_sender : OSCSender | None, default None
            If provided, each computed NF value is also broadcast over OSC
            to the configured host/port after every update cycle.
            See :class:`~ant.osc.OSCSender`.
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
        Output files written to ``<subjects_dir>/<subject_id>/``:

        * ``raw/visit_<N>-raw.fif`` — full raw NF recording
        * ``nf/visit_<N>-nf.json`` — NF feature time-series (if
          ``save_nf_signal=True``)
        * ``delays/visit_<N>-delays.json`` — timing breakdown (if
          ``estimate_delays=True``)

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
        self.use_ring_buffer = use_ring_buffer

        # Artifact correction setup
        ref_ch_idx: Optional[int] = None
        if self.artifact_correction == "lms":
            ref_ch_idx = self.rec_info["ch_names"].index(self.ref_channel)
        elif self.artifact_correction == "orica":
            self.run_orica(n_channels=len(self.rec_info["ch_names"]), forgetfac=0.99)
        elif self.artifact_correction == "gedai":
            if not hasattr(self, "gedai") or self.gedai is None:
                raise RuntimeError(
                    "Call fit_gedai() after baseline recording before starting a gedai session."
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
            "sensor_graph":         0.05,
            "source_graph":         2e-17,
            "entropy":              3.0,
            "argmax_freq":          8.0,
            "individual_peak_power": self._SENSOR_POWER_SCALE[self.data_type],
            "cfc_sensor":           1.0,
            "erd_ers":              50.0,
            "laterality":           2.0,
            "hjorth":               5.0,
            "spectral_centroid":    5.0,
        }

        signal_plot: Optional[NFSignalPlot] = None
        brain_plot: Optional[BrainPlot] = None

        needs_qt = show_nf_signal or show_brain_activation or show_raw_signal
        app: Optional[QtWidgets.QApplication] = None

        if needs_qt:
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        if show_nf_signal:
            signal_plot = NFSignalPlot(
                modalities=mods,
                scales_dict=scales_dict,
                sfreq=30.0,   # pump timer rate drives display at 30 fps
                time_window=time_window,
            )
            signal_plot.show()

        if show_brain_activation:
            if self.subjects_fs_dir is None:
                raise ValueError(
                    "subjects_fs_dir must be set to use brain activation display."
                )
            brain_plot = BrainPlot(
                subjects_fs_dir=self.subjects_fs_dir,
                clim=[0, 0.6],
            )

        if show_raw_signal:
            self.open_stream_viewer()

        if self.filtering:
            self.stream.filter(l_freq=self.l_freq, h_freq=self.h_freq)

        # ---- Thread-safe queues between acquisition thread and UI ----

        # small caps: drop stale frames rather than accumulate backlog
        nf_queue: _queue.Queue = _queue.Queue(maxsize=4)
        brain_queue: Optional[_queue.Queue] = (
            _queue.Queue(maxsize=2) if brain_plot is not None else None
        )
        done_event = threading.Event()
        nf_data: dict[str, list] = {m: [] for m in mods}

        # Delay accumulators live in the thread, assigned to self when done
        _acq_delays:     list[float] = []
        _art_delays:     list[float] = []
        _meth_delays:    dict[str, list] = {m: [] for m in mods}
        _plot_delays:    list[float] = []  # only used when viz runs inline

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
            return data

        def _push_brain(window: np.ndarray) -> None:
            """Compute source scalars in acquisition thread, enqueue for UI."""
            if brain_queue is None:
                return
            raw_d = self._prepare_raw_array(window)
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
                lh = stc.lh_data.mean(axis=1)
                rh = stc.rh_data.mean(axis=1)
            try:
                brain_queue.put_nowait((lh, rh))
            except _queue.Full:
                pass  # UI is slower than acquisition; drop this frame

        # ---- Acquisition thread ----

        def _acquire() -> None:
            t_start = local_clock()

            if not use_ring_buffer:
                # Fixed-window loop
                while local_clock() < t_start + duration:
                    tic = time.time()
                    data = self.stream.get_data(winsize, picks=picks)[0]
                    if estimate_delays:
                        _acq_delays.append(time.time() - tic)
                    if data.shape[1] != self.window_size_s:
                        continue

                    data = _correct(data)

                    futures = [
                        self.executor.submit(nf_fns[i], data, **precomps[i])
                        for i in range(len(mods))
                    ]
                    for m, fut in zip(mods, futures):
                        nf_val, m_delay = fut.result()
                        nf_data[m].append(nf_val)
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

                    _push_brain(data)

            else:
                # Sliding ring-buffer loop
                n_ch = len(self.rec_info["ch_names"])
                ring = np.zeros((n_ch, 0), dtype=np.float32)
                fetch_secs = max(winsize * 4, 2.0)
                hop = max(self.window_size_s // 2, 1)
                max_buf = int(fetch_secs * self._sfreq) + self.window_size_s

                while local_clock() < t_start + duration:
                    tic = time.time()
                    fetched = self.stream.get_data(fetch_secs, picks=picks)[0]
                    if estimate_delays:
                        _acq_delays.append(time.time() - tic)

                    if fetched is None or fetched.size == 0:
                        time.sleep(0.001)
                        continue

                    ring = np.concatenate((ring, fetched), axis=1)
                    if ring.shape[1] > max_buf:
                        ring = ring[:, -max_buf:]

                    while ring.shape[1] >= self.window_size_s:
                        window = ring[:, : self.window_size_s].copy()
                        window = _correct(window)

                        for i, mod in enumerate(mods):
                            meth_tic = time.time()
                            nf_val, m_delay = nf_fns[i](window, **precomps[i])
                            nf_data[mod].append(nf_val)
                            if estimate_delays:
                                _meth_delays[mod].append(
                                    {"measured": time.time() - meth_tic, "reported": m_delay}
                                )

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

                        _push_brain(window)
                        ring = ring[:, hop:]

            done_event.set()

        # ---- Start acquisition thread ----

        acq_thread = threading.Thread(target=_acquire, daemon=True)
        acq_thread.start()

        # ---- Event loop or blocking join ----

        if needs_qt:
            from PyQt6.QtCore import QTimer

            # Interpolation state: [prev_vals, curr_vals, step_index]
            # Linearly interpolates between consecutive NF estimates so the
            # display ramps smoothly rather than stepping every winsize seconds.
            _interp: list = [[], [], [0]]
            _n_steps: int = max(1, int(30 * winsize))

            def _pump_signal() -> None:
                """Fast timer (~30 fps) — NF signal plot only.

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

                if signal_plot is not None and _interp[1]:
                    step = _interp[2][0]
                    if _interp[0] and step < _n_steps:
                        alpha = step / _n_steps
                        vals = [
                            _interp[0][i] * (1 - alpha) + _interp[1][i] * alpha
                            for i in range(len(_interp[1]))
                        ]
                    else:
                        vals = _interp[1]
                    signal_plot.push(vals)
                    _interp[2][0] += 1

                if done_event.is_set():
                    signal_timer.stop()
                    if brain_timer is not None:
                        brain_timer.stop()
                    QTimer.singleShot(600, app.quit)

            signal_timer = QTimer()
            signal_timer.setInterval(33)   # ~30 fps
            signal_timer.timeout.connect(_pump_signal)
            signal_timer.start()

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
                    except Exception:
                        pass

                brain_timer = QTimer()
                brain_timer.setInterval(200)  # 5 fps — brain doesn't need more
                brain_timer.timeout.connect(_pump_brain)
                brain_timer.start()

            app.exec()          # blocks here; drives all three windows in parallel

        else:
            acq_thread.join()   # headless: block until acquisition finishes

        acq_thread.join(timeout=10)

        # ---- Persist results ----

        self.nf_data = nf_data
        if estimate_delays:
            self.acq_delays      = _acq_delays
            self.artifact_delays = _art_delays
            self.method_delays   = _meth_delays
            self.save(nf_data=True, acq_delay=True, artifact_delay=True,
                      method_delay=True, format="json")
        else:
            self.save(nf_data=True, acq_delay=False, artifact_delay=False,
                      method_delay=False, format="json")

        if signal_plot is not None:
            signal_plot.close()

    # ------------------------------------------------------------------
    # Modality-params property
    # ------------------------------------------------------------------

    @property
    def modality_params(self) -> dict:
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
        self, max_iter: int = 800, method: str = "infomax"
    ) -> None:
        """Identify the eye-blink ICA component from baseline EEG.

        Sets ``self.blink_template`` (ndarray) — the spatial template vector
        for the blink component, used by artifact correction during the main
        session.
        """
        self.blink_template = create_blink_template(
            self.raw_baseline, max_iter=max_iter, method=method
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

        Parameters
        ----------
        n_channels : int
        learning_rate : float, default 0.1
        block_size : int, default 256
        online_whitening : bool, default True
        calibrate_pca : bool, default False
        forgetfac : float, default 1.0
        nonlinearity : str, default "tanh"
        random_state : int | None, default None
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
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Fit a :class:`~ant.tools.GEDAIDenoiser` from the recorded baseline.

        Must be called after :meth:`record_baseline` and before
        :meth:`record_main` when ``artifact_correction="gedai"``.

        Parameters
        ----------
        band : tuple of float, default (8.0, 13.0)
            Target frequency band ``(low_Hz, high_Hz)`` for the GED.
            The band-filtered baseline covariance is used as the
            signal matrix; the broadband baseline covariance is the
            reference.
        shrinkage : float, default 0.01
            Tikhonov regularisation strength applied to the broadband
            covariance before solving the generalised eigenvalue problem.
            Larger values improve numerical stability at the cost of
            slightly less discriminative filters.
        verbose : bool | str | None, default None
            Override the instance-level verbosity for this call.

        Raises
        ------
        RuntimeError
            If :meth:`record_baseline` has not been called yet.

        See Also
        --------
        ant.tools.GEDAIDenoiser : The underlying GED denoiser class.

        Examples
        --------
        >>> nf.record_baseline(baseline_duration=120)
        >>> nf.fit_gedai(band=(8, 13))
        >>> nf.record_main(duration=600, artifact_correction="gedai")
        """
        if not hasattr(self, "raw_baseline") or self.raw_baseline is None:
            raise RuntimeError("Run record_baseline() before calling fit_gedai().")

        n_channels = len(self.rec_info["ch_names"])
        self.gedai = GEDAIDenoiser(n_channels=n_channels, shrinkage=shrinkage)
        self.gedai.fit_from_raw(
            data=self.raw_baseline.get_data(),
            sfreq=self._sfreq,
            band=band,
        )

    def compute_inv_operator(self) -> None:
        """Compute and save the inverse operator for source localisation.

        Saves inverse operator, forward solution and noise covariance to
        ``<subject_dir>/inv/``.
        """
        self.inv, self.fwd, self.noise_cov = _compute_inv_operator(
            self.raw_baseline,
            subject_fs_id=self.subject_fs_id,
            subjects_fs_dir=self.subjects_fs_dir,
            data_type=self.data_type,
        )
        inv_dir = self.subject_dir / "inv"
        write_inverse_operator(
            fname=inv_dir / f"visit_{self.visit}-inv.fif",
            inv=self.inv,
            overwrite=True,
        )
        write_forward_solution(
            fname=inv_dir / f"visit_{self.visit}-fwd.fif",
            fwd=self.fwd,
            overwrite=True,
        )
        write_cov(
            fname=inv_dir / f"visit_{self.visit}-cov.fif",
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
        # Run it in a separate process so the NF loop continues uninterrupted.
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
        format: str = "json",
    ) -> None:
        """Save session data and disconnect the LSL stream.

        Parameters
        ----------
        nf_data : bool, default True
        acq_delay : bool, default True
        artifact_delay : bool, default True
        method_delay : bool, default True
        raw_data : bool, default False
            Not yet implemented.
        format : str, default "json"
        """
        if self.stream.connected:
            self.stream.disconnect()
        for folder in ("neurofeedback", "delays", "main", "reports"):
            (self.subject_dir / folder).mkdir(parents=True, exist_ok=True)

        if format == "json":

            def _write(path: Path, obj: Any) -> None:
                with open(path, "w") as fh:
                    json.dump(obj, fh)

            base = self.subject_dir
            if nf_data:
                _write(
                    base / "neurofeedback" / f"nf_data_visit_{self.visit}.json",
                    self.nf_data,
                )
            if acq_delay and hasattr(self, "acq_delays"):
                _write(
                    base / "delays" / f"acq_delay_visit_{self.visit}.json",
                    self.acq_delays,
                )
            if artifact_delay and hasattr(self, "artifact_delays"):
                _write(
                    base / "delays" / f"artifact_delay_visit_{self.visit}.json",
                    self.artifact_delays,
                )
            if method_delay and hasattr(self, "method_delays"):
                _write(
                    base / "delays" / f"method_delay_visit_{self.visit}.json",
                    self.method_delays,
                )

        if raw_data:
            raise NotImplementedError("Saving raw M/EEG data is not yet implemented.")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def create_report(self, overwrite: bool = True) -> None:
        """Generate an HTML MNE report for the session.

        Parameters
        ----------
        overwrite : bool, default True
        """
        modalities = (
            [self.modality] if isinstance(self.modality, str) else list(self.modality)
        )
        report = Report(title=f"Neurofeedback Session — {', '.join(modalities)}")
        report.add_raw(self.raw_baseline, title="Baseline recording", psd=False, butterfly=False)

        source_modalities = {
            "source_power", "source_connectivity", "source_graph",
        }

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
            else:
                if mod == "source_power":
                    fig_brain = plot_glass_brain(bl1=params.get("brain_label"))
                else:
                    fig_brain = plot_glass_brain(
                        bl1=params.get("brain_label_1"),
                        bl2=params.get("brain_label_2"),
                    )
                report.add_figure(fig=fig_brain, title=f"Brain labels — {mod}")

        fname = (
            f"subject_{self.subject_id}_visit_{self.visit}"
            f"_modality_{'_'.join(modalities)}.html"
        )
        report.save(self.subject_dir / "reports" / fname, overwrite=overwrite)
