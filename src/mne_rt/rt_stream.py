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
import math
import queue as _queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union
from warnings import warn

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import (
    Report,
    compute_raw_covariance,
    write_cov,
    write_forward_solution,
    write_source_spaces,
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
from pyqtgraph.Qt import QtWidgets

from mne_rt._logging import logger, set_log_level, verbose
from mne_rt._naming import MODALITY_SEP, osc_address_name, parse_modality, split_modality
from mne_rt._stats import ema_variance, zscore
from mne_rt.combiners import GeometricMeanCombiner, ZScoredNormCombiner
from mne_rt.decoding import RTDecode
from mne_rt.modalities import ModalityMixin
from mne_rt.tools import (
    _compute_inv_operator,
    create_blink_template,
    get_params,
    remove_blinks_lms,
)
from mne_rt.tools.asr import ASRDenoiser
from mne_rt.tools.bids_io import _build_bids_stem, describe_nf_columns, write_nf_beh_tsv
from mne_rt.tools.gedai import GEDAIDenoiser
from mne_rt.tools.maxwell import RTMaxwellFilter
from mne_rt.tools.orica import ORICA
from mne_rt.viz import BrainPlot, NFPlot, RawPlot, TopomapPlot

# The report's axis labels come from the same tables the live NF window uses, so
# the two cannot describe the same modality differently.
from mne_rt.viz.nf_plot import _label_for, _unit_for

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


class ArrayStream:
    """Simulate a live LSL stream from an in-memory numpy array.

    Duck-types the subset of :class:`mne_lsl.stream.StreamLSL`'s public
    interface that :class:`RTStream` relies on (``get_data``,
    ``n_new_samples``, ``connected``, ``info``, ``disconnect``, ``pick``,
    ``filter``, ``notch_filter``, ``set_montage``, ``set_meas_date``), so an
    :class:`RTStream` session can be driven by a plain numpy array instead of
    live hardware or a recorded file replayed over LSL. Returned by
    :meth:`RTStream.connect_to_array` — not normally instantiated directly.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        The full recording to stream.
    info : mne.Info
        Channel/sampling-rate metadata; ``info["nchan"]`` must equal
        ``data.shape[0]``.
    bufsize : float, default 3.0
        Ring-buffer size in seconds, mirroring ``StreamLSL(bufsize=...)``.
    chunk_size : int, default 10
        Number of samples released into the buffer per acquisition tick,
        mirroring :class:`mne_lsl.player.PlayerLSL`'s ``chunk_size``.
    n_repeat : int | float, default 1
        Number of times to loop over ``data`` once exhausted. Use
        ``np.inf`` for an open-ended session.

    Notes
    -----
    Because the entire recording is already available in memory,
    :meth:`filter` and :meth:`notch_filter` apply a single zero-phase FIR
    pass over the whole array (via :func:`mne.filter.filter_data` /
    :func:`mne.filter.notch_filter`) rather than a causal online filter —
    there is no "future data" constraint to respect as there would be for a
    genuinely live stream. If called while already streaming (as
    :meth:`~mne_rt.RTStream.record_main` does), samples already pushed into
    the ring buffer stay unfiltered until they age out — the buffer is fully
    refreshed, and the transient gone, within one ``bufsize`` window.

    If ``n_repeat`` is exhausted before the caller stops requesting data,
    streaming simply stops advancing: :attr:`n_new_samples` stays at 0 and
    :meth:`get_data` keeps returning the final buffered window.
    """

    def __init__(
        self,
        data: np.ndarray,
        info: mne.Info,
        *,
        bufsize: float = 3.0,
        chunk_size: int = 10,
        n_repeat: Union[int, float] = 1,
    ) -> None:
        self._source = np.ascontiguousarray(data, dtype=np.float64)
        self._info = info.copy()
        self._bufsize = bufsize
        self._chunk_size = chunk_size
        self._n_repeat = n_repeat

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._buffer = np.empty((0, 0))
        self._timestamps = np.empty(0)
        self._n_new_samples = 0

    def __repr__(self) -> str:
        state = "connected" if self._connected else "disconnected"
        return f"<ArrayStream | {self._info['nchan']} channels, {state}>"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> "ArrayStream":
        """Start the simulated real-time acquisition thread."""
        if self._connected:
            self.disconnect()
        sfreq = self._info["sfreq"]
        n_buf = max(1, int(np.ceil(self._bufsize * sfreq)))
        self._buffer = np.zeros((self._info["nchan"], n_buf))
        self._timestamps = np.zeros(n_buf)
        self._n_new_samples = 0
        self._stop_event.clear()
        self._connected = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        return self

    def disconnect(self) -> "ArrayStream":
        """Stop the acquisition thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._connected = False
        return self

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def info(self) -> mne.Info:
        if not self._connected:
            raise RuntimeError(
                "The stream information is available once connected. "
                "Please connect to the stream first."
            )
        return self._info

    @property
    def n_new_samples(self) -> int:
        """Number of new samples available since the last :meth:`get_data` call."""
        return self._n_new_samples

    # ------------------------------------------------------------------
    # Acquisition thread
    # ------------------------------------------------------------------

    def _stream_loop(self) -> None:
        sfreq = self._info["sfreq"]
        chunk_dur = self._chunk_size / sfreq
        n_total = self._source.shape[1]
        cursor = 0
        n_done = 0
        while not self._stop_event.is_set():
            tic = time.time()
            with self._lock:
                source = self._source
            end = min(cursor + self._chunk_size, n_total)
            chunk = source[:, cursor:end]
            self._push(chunk)
            cursor = end
            if cursor >= n_total:
                n_done += 1
                if n_done >= self._n_repeat:
                    break
                cursor = 0
            self._stop_event.wait(max(0.0, chunk_dur - (time.time() - tic)))

    def _push(self, chunk: np.ndarray) -> None:
        k = chunk.shape[1]
        if k == 0:
            return
        # `local_clock`, not `time.time`: a real LSL stream timestamps its samples
        # on this clock, and window onsets are derived from those timestamps. The
        # two are ~1.8e9 seconds apart (seconds-since-boot against the Unix
        # epoch), so mixing them would put every onset computed under this test
        # double on a different time base from the live one.
        now = local_clock()
        with self._lock:
            n_buf = self._buffer.shape[1]
            if k >= n_buf:
                self._buffer[:] = chunk[:, -n_buf:]
                self._timestamps[:] = now
            else:
                self._buffer[:, :-k] = self._buffer[:, k:]
                self._buffer[:, -k:] = chunk
                self._timestamps[:-k] = self._timestamps[k:]
                self._timestamps[-k:] = now
            self._n_new_samples = min(self._n_new_samples + k, n_buf)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_data(
        self,
        winsize: Optional[float] = None,
        picks: Optional[Union[str, list]] = None,
        exclude: Union[str, list, tuple] = "bads",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve the latest data from the buffer.

        Mirrors :meth:`mne_lsl.stream.StreamLSL.get_data`: resets
        :attr:`n_new_samples` to 0 on every call.
        """
        if not self._connected:
            raise RuntimeError(
                "The Stream is not connected. Please connect to the stream before "
                "retrieving data from the buffer."
            )
        sfreq = self._info["sfreq"]
        idx = self._resolve_picks(picks, exclude)
        with self._lock:
            n_buf = self._buffer.shape[1]
            n_samples = n_buf if winsize is None else min(n_buf, int(np.ceil(winsize * sfreq)))
            data = self._buffer[idx, -n_samples:].copy()
            ts = self._timestamps[-n_samples:].copy()
            self._n_new_samples = 0
        return data, ts

    def _resolve_picks(self, picks: Any, exclude: Any = ()) -> np.ndarray:
        ch_names = self._info["ch_names"]
        exclude_names = set(self._info["bads"]) if exclude == "bads" else set(exclude or ())
        if picks is None:
            return np.array(
                [i for i, ch in enumerate(ch_names) if ch not in exclude_names], dtype=int
            )
        if isinstance(picks, (str, int, np.integer)):
            picks = [picks]
        idx: list[int] = []
        for p in picks:
            if isinstance(p, str) and p in ch_names:
                if p not in exclude_names:
                    idx.append(ch_names.index(p))
            elif isinstance(p, str):
                idx.extend(mne.pick_types(self._info, exclude=exclude, **{p: True}).tolist())
            else:
                idx.append(int(p))
        return np.array(idx, dtype=int)

    # ------------------------------------------------------------------
    # In-place channel / signal operations
    # ------------------------------------------------------------------

    def set_montage(self, montage: Any, on_missing: str = "warn") -> "ArrayStream":
        self._info.set_montage(montage, on_missing=on_missing, verbose=False)
        return self

    def set_meas_date(self, meas_date: Any) -> "ArrayStream":
        self._info.set_meas_date(meas_date)
        return self

    def pick(self, picks: Any = None, exclude: Any = ()) -> "ArrayStream":
        """Restrict the stream to a subset of channels, in-place.

        Must be called before :meth:`connect` — picking after the
        acquisition thread has started would change the channel count out
        from under it mid-stream.
        """
        if self._connected:
            raise RuntimeError("pick() must be called before connect().")
        idx = self._resolve_picks(picks, exclude)
        self._source = self._source[idx]
        self._info = mne.pick_info(self._info, idx, copy=True, verbose=False)
        return self

    def filter(self, l_freq: Optional[float], h_freq: Optional[float]) -> "ArrayStream":
        """Zero-phase FIR band-pass over the entire underlying array.

        Computed outside the buffer lock so the acquisition thread is only
        blocked for the final (near-instant) reference swap, not the full
        filtering pass.
        """
        filtered = mne.filter.filter_data(
            self._source, self._info["sfreq"], l_freq, h_freq, verbose=False
        )
        with self._lock:
            self._source = filtered
        return self

    def notch_filter(self, freqs: Union[float, list]) -> "ArrayStream":
        """Zero-phase FIR notch filter over the entire underlying array.

        See :meth:`filter` for why the computation happens outside the lock.
        """
        filtered = mne.filter.notch_filter(self._source, self._info["sfreq"], freqs, verbose=False)
        with self._lock:
            self._source = filtered
        return self


class MarkerArrayStream:
    """Replay a fixed schedule of event markers as an irregular LSL-like stream.

    The marker counterpart to :class:`ArrayStream`: it stands in for the LSL
    outlet a stimulus program such as PsychoPy publishes, so a gated
    neurofeedback session can be driven and tested without any LSL networking.
    Returned by :meth:`RTStream.connect_marker_array`.

    :class:`ArrayStream` cannot serve here. A marker stream is *irregularly*
    sampled (``sfreq == 0``), and that class assumes a rate throughout: its
    replay thread divides by ``sfreq``, its ring buffer is sized
    ``ceil(bufsize * sfreq)`` samples, and ``get_data`` would ask for
    ``ceil(winsize * sfreq) == 0`` samples and return the entire buffer through
    a ``[-0:]`` slice.

    Timestamps come from :func:`~mne_lsl.lsl.local_clock`, the same clock the
    acquisition loop reads window onsets on, so a marker and a window can be
    compared directly.

    Parameters
    ----------
    codes : array-like of int
        Marker codes to publish, in order.
    onsets : array-like of float
        When to publish each code, in seconds after :meth:`connect`.  Must be
        the same length as ``codes`` and non-decreasing.
    bufsize : int, default 200
        Ring-buffer size, in markers.  Irregular streams are buffered by count,
        not by duration.
    ch_name : str, default "markers"
        Name of the single marker channel.

    See Also
    --------
    ArrayStream : The signal-side equivalent.
    RTStream.connect_marker_array : Attaches one of these to a session.
    RTStream.connect_marker_stream : The live LSL equivalent.

    .. versionadded:: 1.2.0
    """

    def __init__(
        self,
        codes,
        onsets,
        *,
        bufsize: int = 200,
        ch_name: str = "markers",
    ) -> None:
        codes = np.asarray(codes, dtype=np.int64).ravel()
        onsets = np.asarray(onsets, dtype=np.float64).ravel()
        if codes.size != onsets.size:
            raise ValueError(
                f"`codes` and `onsets` must be the same length; got {codes.size} and {onsets.size}."
            )
        if np.any(np.diff(onsets) < 0):
            raise ValueError("`onsets` must be non-decreasing.")
        if bufsize <= 0:
            raise ValueError(
                f"`bufsize` is a number of markers and must be positive; got {bufsize}."
            )

        self._codes = codes
        self._onsets = onsets
        self._bufsize = int(bufsize)
        self._ch_name = str(ch_name)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        # Preallocated like mne-lsl's own ring buffers, zeros included: a
        # timestamp of 0.0 marks a slot no marker has reached yet.
        self._buffer = np.zeros((self._bufsize, 1))
        self._timestamps = np.zeros(self._bufsize)
        self._n_new_samples = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MarkerArrayStream | {self._codes.size} markers | connected={self._connected}>"

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, acquisition_delay: float = 0.005) -> "MarkerArrayStream":
        """Start publishing the schedule on a background thread."""
        if self._connected:
            return self
        with self._lock:
            self._buffer = np.zeros((self._bufsize, 1))
            self._timestamps = np.zeros(self._bufsize)
            self._n_new_samples = 0
        self._stop_event.clear()
        self._connected = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        return self

    def disconnect(self) -> "MarkerArrayStream":
        """Stop publishing."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._connected = False
        return self

    @property
    def connected(self) -> bool:
        """Whether the stream is publishing."""
        return self._connected

    @property
    def info(self) -> dict:
        """Minimal stream description, mirroring the keys read off a real stream."""
        return {"nchan": 1, "sfreq": 0.0, "ch_names": [self._ch_name]}

    @property
    def n_new_samples(self) -> int:
        """Markers pushed since the last :meth:`get_data` call."""
        return self._n_new_samples

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _stream_loop(self) -> None:
        t0 = local_clock()
        i = 0
        while not self._stop_event.is_set() and i < self._codes.size:
            if local_clock() - t0 >= self._onsets[i]:
                self._push(int(self._codes[i]))
                i += 1
                continue
            time.sleep(0.002)

    def _push(self, code: int) -> None:
        """Append one marker to the ring buffer, stamped on the LSL clock."""
        now = local_clock()
        with self._lock:
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1, 0] = code
            self._timestamps[:-1] = self._timestamps[1:]
            self._timestamps[-1] = now
            self._n_new_samples = min(self._n_new_samples + 1, self._bufsize)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_data(
        self,
        winsize: Optional[int] = None,
        picks: Optional[Union[str, list]] = None,
        exclude: Union[str, list, tuple] = "bads",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve the latest markers from the buffer.

        Mirrors :meth:`mne_lsl.stream.StreamLSL.get_data` on an irregularly
        sampled stream: ``winsize`` is a number of samples rather than a
        duration, ``None`` returns the whole buffer, and
        :attr:`n_new_samples` is reset on every call.
        """
        if not self._connected:
            raise RuntimeError(
                "The Stream is not connected. Please connect to the stream before "
                "retrieving data from the buffer."
            )
        with self._lock:
            n = self._bufsize if winsize is None else min(self._bufsize, max(0, int(winsize)))
            data = self._buffer[self._bufsize - n :, :].T.copy()
            ts = self._timestamps[self._bufsize - n :].copy()
            self._n_new_samples = 0
        return data, ts


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
    source_space : {"surface", "volume"}, default "surface"
        Geometry of the source space built by
        :meth:`compute_inv_operator`.  ``"surface"`` is the cortical-surface
        model used historically.  ``"volume"`` builds a volumetric grid, which
        is required for **subcortical** ROIs (hippocampus, amygdala,
        thalamus) — these do not exist on the cortical surface at all.
    source_atlas : str | None, default None
        Atlas used to resolve ROI names.  ``None`` selects a sensible default
        for ``source_space``: ``"aparc"`` (a surface annotation) for
        ``"surface"``, and ``"aparc+aseg"`` (a volumetric ``.mgz``) for
        ``"volume"``.  Note ``aparc+aseg`` contains cortical parcels *and*
        subcortical structures, so one volume source space covers both.
    source_pos : float, default 5.0
        Grid spacing in mm for a volumetric source space.
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
        source_space: str = "surface",
        source_atlas: Optional[str] = None,
        source_pos: float = 5.0,
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
            raise ValueError("`montage` must be a built-in name, a valid '.bvct' path, or None.")
        if not isinstance(mri, bool):
            raise ValueError("`mri` must be a boolean.")
        if not subject_fs_id or not isinstance(subject_fs_id, str):
            raise ValueError("`subject_fs_id` must be a non-empty string.")
        if subjects_fs_dir is not None and not Path(subjects_fs_dir).is_dir():
            raise ValueError("`subjects_fs_dir` must be None or an existing directory.")
        if source_space not in ("surface", "volume"):
            raise ValueError(f"`source_space` must be 'surface' or 'volume', got {source_space!r}.")
        if not isinstance(source_pos, (int, float)) or source_pos <= 0:
            raise ValueError("`source_pos` must be a positive number (grid spacing in mm).")
        if source_atlas is None:
            # A volume source space needs a volumetric atlas; the surface
            # default ("aparc") is an annot and has no .mgz counterpart.
            source_atlas = "aparc+aseg" if source_space == "volume" else "aparc"
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
                raise ValueError(
                    "`notch_freq` must be a positive float or list of positive floats."
                )
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
        self.source_space = source_space
        self.source_atlas = source_atlas
        self.source_pos = source_pos
        self.src = None
        self.data_cov = None
        self.bandpass_freq = bandpass_freq
        self.notch_freq = notch_freq
        self.artifact_correction = artifact_correction
        self.save_nf_signal = save_nf_signal
        self.config_file = config
        self.verbose = verbose
        self.subject_dir = Path(subjects_dir) / f"sub-{subject_id}" / f"ses-{session}"

        set_log_level(verbose)

    # ------------------------------------------------------------------
    # LSL connection
    # ------------------------------------------------------------------

    def _teardown_session_streams(self) -> None:
        """Disconnect the acquisition stream, if one is connected."""
        if hasattr(self, "stream") and getattr(self.stream, "connected", False):
            self.stream.disconnect()
        # mne-lsl empties the filter chain on disconnect, so a stream that is
        # reconnected by hand starts unfiltered; leaving the flag set would run
        # the next session with no band-pass and no notch, silently.
        self._filters_applied = False

    def _teardown_existing_stream(self) -> None:
        """Disconnect any previously-connected stream/mock player, if present."""
        self._teardown_session_streams()
        # The marker stream is timestamped against the signal stream, so it
        # cannot outlive it: keeping it would silently match markers to a
        # different recording.
        self._teardown_marker_stream(announce=True)
        # Belongs to the stream just torn down; leaving it set would make the
        # clocksync check in connect_marker_stream answer for the wrong stream.
        self._processing_flags = None
        if getattr(self, "_mock_player", None) is not None:
            try:
                self._mock_player.stop()
            except Exception:
                pass
            self._mock_player = None

    def _acquired_info(self) -> Any:
        """``rec_info`` restricted to :meth:`_acquired_ch_names`."""
        names = self._acquired_ch_names()
        if names == list(self.rec_info["ch_names"]):
            return self.rec_info
        keep = [self.rec_info["ch_names"].index(ch) for ch in names]
        return mne.pick_info(self.rec_info, keep, verbose=False)

    def _finalize_stream(self, stream: Any) -> None:
        """Store the connected stream and expose its public methods on self."""
        self.stream = stream
        # A fresh stream carries no filters, whatever the last one had.
        self._filters_applied = False
        self.sfreq = stream.info["sfreq"]
        self.rec_info = stream.info
        self.rec_info["subject_info"] = {"his_id": self.subject_id}

        # Expose stream methods directly on self
        for name in dir(self.stream):
            if not name.startswith("__"):
                attr = getattr(self.stream, name)
                if callable(attr):
                    setattr(self, name, attr)

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
        processing_flags: Union[str, Sequence[str], None] = None,
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
        processing_flags : str | sequence of str | None, default None
            Forwarded to :meth:`mne_lsl.stream.StreamLSL.connect`.  Pass
            ``"all"`` to enable ``clocksync``, which is required when a marker
            stream published from another machine has to be aligned against
            this one — see :meth:`connect_marker_stream`.
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
        self._teardown_existing_stream()

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

        stream.connect(
            acquisition_delay=acquisition_delay,
            processing_flags=processing_flags,
            timeout=timeout,
        )
        # Recorded because StreamLSL does not expose the flags it connected
        # with, and connect_marker_stream needs to know whether the two streams
        # share a clock.
        self._processing_flags = processing_flags

        if self.montage is not None:
            stream.set_montage(self.montage, on_missing="warn")
        if pick_types is not None:
            stream.pick(pick_types)

        stream.set_meas_date(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc))
        self._finalize_stream(stream)

    @verbose
    def connect_to_array(
        self,
        data: np.ndarray,
        info: mne.Info,
        chunk_size: int = 10,
        n_repeat: Union[int, float] = 1,
        bufsize_main: float = 3.0,
        pick_types: Optional[str] = None,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Connect to a plain numpy array instead of an LSL stream.

        Drives the exact same acquisition/feature-extraction pipeline as
        :meth:`connect_to_lsl`, backed by :class:`ArrayStream` — no LSL
        networking or recorded file is required. Useful for offline
        analysis, unit tests, and demos, e.g. feeding synthetic data or an
        already-loaded recording straight through :meth:`record_baseline` /
        :meth:`record_main`.

        Parameters
        ----------
        data : array of shape (n_channels, n_samples)
            The full recording to stream.
        info : mne.Info
            Channel/sampling-rate metadata; ``info["nchan"]`` must equal
            ``data.shape[0]``.
        chunk_size : int, default 10
            Samples released into the buffer per acquisition tick.
        n_repeat : int | float, default 1
            Number of times to loop over ``data`` once exhausted. Use
            ``np.inf`` to loop indefinitely for open-ended sessions.
        bufsize_main : float, default 3.0
            Ring-buffer size in seconds.
        pick_types : str | None, default None
            Channel type to keep (e.g. ``"eeg"``, ``"mag"``).
            ``None`` keeps all available channels.
        verbose : bool | str | None, default None
            Override the instance-level verbosity for this call.

        Raises
        ------
        ValueError
            If ``data`` is not 2D, or its channel count does not match
            ``info["nchan"]``.

        Notes
        -----
        All public methods of :class:`ArrayStream` are also exposed
        directly on the :class:`RTStream` instance after connection — a
        narrower surface than :meth:`connect_to_lsl` exposes, since
        :class:`ArrayStream` has no LSL-specific extras (e.g. ``name``,
        used by :meth:`open_stream_viewer`, which is unavailable for
        array-backed sessions).

        See Also
        --------
        connect_to_lsl : Connect to a live or mock-replayed LSL stream.

        Examples
        --------
        Stream a synthetic recording end-to-end without any LSL infra::

            info = mne.create_info(ch_names, sfreq=256.0, ch_types="eeg")
            nf.connect_to_array(data, info)
            nf.record_baseline(baseline_duration=60)
            nf.record_main(duration=300, modality="sensor_power")
        """
        self._teardown_existing_stream()

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(
                f"`data` must be a 2D (n_channels, n_samples) array, got shape {data.shape}."
            )
        if data.shape[0] != info["nchan"]:
            raise ValueError(
                f"`data` has {data.shape[0]} channels but `info` describes "
                f"{info['nchan']} channels."
            )

        self.bufsize = bufsize_main

        if self.montage is not None and Path(str(self.montage)).is_file():
            self.montage = read_dig_captrak(self.montage)

        self.source_id = uuid.uuid4().hex
        stream = ArrayStream(
            data, info, bufsize=self.bufsize, chunk_size=chunk_size, n_repeat=n_repeat
        )

        # Apply montage/picks/meas-date before starting the acquisition thread
        # so ArrayStream.pick()'s channel-count change can never race with it.
        if self.montage is not None:
            stream.set_montage(self.montage, on_missing="warn")
        if pick_types is not None:
            stream.pick(pick_types)
        stream.set_meas_date(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc))

        stream.connect()
        self._finalize_stream(stream)

    # ------------------------------------------------------------------
    # Marker stream
    # ------------------------------------------------------------------

    @verbose
    def connect_marker_stream(
        self,
        stream_name: Optional[str] = None,
        source_id: Optional[str] = None,
        marker_id: Optional[dict] = None,
        bufsize: int = 200,
        ch_name: Optional[str] = None,
        timeout: float = 10.0,
        acquisition_delay: float = 0.005,
        processing_flags: Union[str, Sequence[str], None] = "all",
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Connect the LSL stream that carries the experiment's event markers.

        This is the inbound counterpart to :class:`~mne_rt.LSLSender`: a
        stimulus program such as PsychoPy publishes trial markers on their own
        LSL outlet, and connecting it here lets
        :meth:`record_main` tag every analysis window with the condition that
        was running, and optionally mute feedback outside chosen conditions.

        The marker stream is kept separate from the M/EEG stream in
        :attr:`marker_stream`; it is never mixed into the signal path.

        Parameters
        ----------
        stream_name : str | None, default None
            Connect by stream name.
        source_id : str | None, default None
            Connect by source ID.  May be given instead of, or together with,
            ``stream_name``; together they must identify exactly one stream.
        marker_id : dict | None, default None
            Condition label to marker code, e.g. ``{"speech": 1, "rest": 2}``.
            Used to label windows and to interpret ``gate_conditions``.  With
            ``None`` the codes are their own labels.
        bufsize : int, default 200
            Buffer size **in markers**.  A marker outlet is irregularly
            sampled, so a duration would carry no meaning.
        ch_name : str | None, default None
            Name of the channel carrying the codes.  ``None`` uses the stream's
            first channel.
        timeout : float, default 10.0
            LSL connection timeout in seconds.
        acquisition_delay : float, default 0.005
            Seconds between acquisition polling attempts.
        processing_flags : str | sequence of str | None, default "all"
            Forwarded to :meth:`mne_lsl.stream.StreamLSL.connect`.  The default
            enables ``clocksync`` — see Notes.
        verbose : bool | str | None, default None

        Raises
        ------
        RuntimeError
            If no matching stream is found, or if the outlet publishes strings.

        See Also
        --------
        connect_marker_array : Offline equivalent, for tests and demos.
        RTEpochs.connect_to_lsl : The same idea for event-triggered epochs.

        Notes
        -----
        **Clock synchronisation.**  Two LSL streams share a time base only once
        their clocks are synchronised.  Without it, markers published from a
        second machine sit at an arbitrary offset from the M/EEG data and every
        one of them lands in the wrong window, with nothing to report it.  This
        method therefore defaults to ``processing_flags="all"``, and warns if
        the M/EEG stream was connected without the same treatment — pass
        ``processing_flags="all"`` to :meth:`connect_to_lsl` as well.

        **Publishing the markers.**  The outlet must be **numeric**; mne-lsl
        refuses string streams, and ``channel_format="string"`` is what most
        PsychoPy marker examples use.  It should label its channel, since an
        unlabelled outlet is exposed as ``"0"``.  A minimal publisher::

            from mne_lsl.lsl import StreamInfo, StreamOutlet

            sinfo = StreamInfo(
                "psychopy_markers", "Markers", 1, 0.0, "int32", "psychopy_uid"
            )
            sinfo.set_channel_names(["markers"])
            outlet = StreamOutlet(sinfo)
            outlet.push_sample([1])          # a "speech" trial

        Examples
        --------
        Tag every window with the running condition, and reward only during
        speech attempts::

            nf.connect_to_lsl(stream_name="ANT", processing_flags="all")
            nf.connect_marker_stream(
                stream_name="psychopy_markers",
                marker_id={"speech": 1, "rest": 2},
            )
            nf.record_main(duration=300, gate_conditions=["speech"])
        """
        if stream_name is None and source_id is None:
            raise ValueError("Give `stream_name` or `source_id` to identify the marker stream.")
        if int(bufsize) != bufsize or bufsize <= 0:
            raise ValueError(
                f"`bufsize` is a number of markers and must be a positive integer; got {bufsize!r}."
            )

        self._teardown_marker_stream()

        logger.info("Connecting marker stream (name=%r, source_id=%r) …", stream_name, source_id)
        stream = Stream(bufsize=int(bufsize), name=stream_name, source_id=source_id)
        try:
            stream.connect(
                acquisition_delay=acquisition_delay,
                processing_flags=processing_flags,
                timeout=timeout,
            )
        except RuntimeError as exc:
            if "string LSL streams" not in str(exc):
                raise
            raise RuntimeError(
                "The marker stream publishes strings, which mne-lsl cannot read. "
                "Publish the codes from a numeric outlet instead, e.g. "
                "channel_format='int32' — note that PsychoPy's marker examples "
                "commonly default to channel_format='string'."
            ) from exc

        self._attach_marker_stream(stream, marker_id=marker_id, ch_name=ch_name)

        # Only meaningful against a live LSL signal stream: an array stream is
        # local by construction and has no clock to synchronise with.
        _signal = getattr(self, "stream", None)
        if (
            processing_flags is not None
            and _signal is not None
            and not isinstance(_signal, (ArrayStream, MarkerArrayStream))
            and getattr(self, "_processing_flags", None) is None
        ):
            logger.warning(
                "The marker stream is clock-synchronised but the M/EEG stream is not. "
                "If the two are published from different machines their clocks differ "
                "by an arbitrary offset, and every marker will be matched to the wrong "
                "window without any error. Pass processing_flags='all' to "
                "connect_to_lsl() as well, or run both on one machine."
            )

    @verbose
    def connect_marker_array(
        self,
        codes,
        onsets,
        marker_id: Optional[dict] = None,
        bufsize: int = 200,
        ch_name: str = "markers",
        verbose: Union[bool, str, None] = None,
    ) -> None:
        """Attach a fixed marker schedule instead of a live LSL outlet.

        The marker counterpart to :meth:`connect_to_array`: drives the exact
        same gating and window-tagging path as :meth:`connect_marker_stream`
        with no LSL networking, which is what makes gated sessions testable and
        demonstrable offline.

        Parameters
        ----------
        codes : array-like of int
            Marker codes to publish, in order.
        onsets : array-like of float
            When to publish each code, in seconds after this call.
        marker_id : dict | None, default None
            Condition label to marker code.  See :meth:`connect_marker_stream`.
        bufsize : int, default 200
            Buffer size in markers.
        ch_name : str, default "markers"
            Name of the marker channel.
        verbose : bool | str | None, default None

        Examples
        --------
        Alternate 10 s speech and rest blocks::

            nf.connect_marker_array(
                codes=[1, 2] * 5,
                onsets=[10.0 * i for i in range(10)],
                marker_id={"speech": 1, "rest": 2},
            )
        """
        self._teardown_marker_stream()
        stream = MarkerArrayStream(codes, onsets, bufsize=bufsize, ch_name=ch_name)
        stream.connect()
        self._attach_marker_stream(stream, marker_id=marker_id, ch_name=ch_name)

    def _attach_marker_stream(self, stream: Any, *, marker_id: Optional[dict], ch_name) -> None:
        """Store a connected marker stream and resolve its code channel.

        Deliberately not :meth:`_finalize_stream`: that copies every non-dunder
        callable of the stream onto ``self``, so a second pass would rebind
        ``self.get_data``, ``self.disconnect`` and ``self._push`` to the marker
        stream and quietly cut the signal path.
        """
        ch_names = list(stream.info["ch_names"])
        if ch_name is None:
            idx = 0
        elif ch_name in ch_names:
            idx = ch_names.index(ch_name)
        else:
            stream.disconnect()
            raise ValueError(
                f"Marker channel {ch_name!r} is not in the marker stream, which publishes "
                f"{ch_names}. An LSL outlet that does not label its channels is exposed "
                "by mne-lsl as '0', '1', … — either label the channel in the publisher, "
                "or pass the name it actually has."
            )

        self.marker_stream = stream
        self._marker_ch_idx = idx
        self.marker_id = dict(marker_id) if marker_id else {}
        # code -> label. Codes without a label are their own label, so an
        # unmapped marker is still recorded rather than silently dropped.
        self._marker_labels = {int(v): str(k) for k, v in self.marker_id.items()}
        logger.info(
            "Marker stream connected — channel %r of %s, %d labelled code(s).",
            ch_names[idx],
            ch_names,
            len(self._marker_labels),
        )

    def _teardown_marker_stream(self, *, announce: bool = False) -> None:
        """Disconnect any previously-connected marker stream.

        ``announce`` when the caller is reconnecting the *signal* stream: the
        marker stream is timestamped against it and cannot survive, and losing
        one silently would mean a session that looks gated but never gates.
        """
        stream = getattr(self, "marker_stream", None)
        if stream is not None:
            if announce:
                logger.warning(
                    "Reconnecting the M/EEG stream also drops the marker stream, whose "
                    "timestamps only mean anything against the recording they were taken "
                    "with. Call connect_marker_stream() again afterwards."
                )
            try:
                stream.disconnect()
            except Exception:
                pass
        self.marker_stream = None

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
        :attr:`raw_baseline`, and saves it to disk.

        The head model is **not** built here.  For ``fsaverage`` that means
        downloading roughly 700 MB of anatomy, which a sensor-space session
        never needs; it is built on first use instead, by a source-space
        modality or by the brain-activation display.  Call
        :meth:`compute_inv_operator` directly to build it eagerly, or to pass
        non-default arguments.

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

        The ``inv/`` files listed under :meth:`compute_inv_operator` appear
        once something asks for the head model.

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
        raw_baseline = RawArray(data, self._acquired_info())

        self._ensure_dirs()
        _stem = f"sub-{self.subject_id}_ses-{self.session}"
        raw_baseline.save(
            self.subject_dir / "eeg" / f"{_stem}_task-baseline_eeg.fif",
            overwrite=True,
        )
        self.raw_baseline = raw_baseline

    def set_decoder(self, decoder: RTDecode) -> None:
        """Attach a fitted decoder for the ``"decode"`` modality.

        Parameters
        ----------
        decoder : instance of RTDecode
            Must already be fit (see :meth:`~mne_rt.RTDecode.fit`) on
            labelled calibration epochs before being attached here.

        See Also
        --------
        mne_rt.RTDecode : Fit-offline / predict-online single-trial decoder.

        Examples
        --------
        >>> decoder = RTDecode(info=epochs_info).fit(X_cal, y_cal)  # doctest: +SKIP
        >>> nf.set_decoder(decoder)  # doctest: +SKIP
        >>> nf.record_main(modality=["decode"])  # doctest: +SKIP
        """
        if not isinstance(decoder, RTDecode):
            raise TypeError(
                f"`decoder` must be an instance of RTDecode, got {type(decoder).__name__}."
            )
        if not decoder.fitted:
            raise RuntimeError(
                "`decoder` must be fit() on calibration epochs before set_decoder()."
            )
        self.decoder = decoder

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
        time_window: float = 30.0,
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
        combiner: Optional[Any] = None,
        combined_name: str = "combined",
        gate_conditions: Optional[list[str]] = None,
        save_raw: bool = False,
        save_tsv: bool = True,
        run: Optional[int] = None,
        disconnect: bool = True,
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
            ``"source_power"``, ``"source_connectivity"``, ``"source_graph"``,
            ``"decode"`` (requires :meth:`set_decoder` beforehand).

            A name may carry an **instance label** after an ``"@"`` —
            ``"source_connectivity@theta"`` — so the same measure can run
            several times over with different parameters (see
            ``modality_params``).  The label distinguishes the instances
            everywhere they appear: plot traces, protocol keys, combiner
            feature names, OSC addresses, LSL channels and saved columns.
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

            A key may name either a base modality or a specific instance.  A
            base entry applies to every instance of it, and an instance's own
            entry takes precedence — so shared settings are written once::

                modality=["source_connectivity@theta",
                          "source_connectivity@alpha"],
                modality_params={
                    "source_connectivity": {"method": "imcoh"},
                    "source_connectivity@theta": {"frange": [4, 8]},
                    "source_connectivity@alpha": {"frange": [8, 13]},
                },
        show_raw_signal : bool, default True
            Show the :class:`~mne_rt.viz.RawPlot` scrolling raw M/EEG viewer.
        show_nf_signal : bool, default True
            Show the :class:`~mne_rt.viz.NFPlot` real-time NF monitor.
        time_window : float, default 30.0
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

            The estimate is scale-free: σ is used exactly as observed, whatever
            the feature's units, so band power (~1e-13 V²/Hz) and a connectivity
            index (~1) both come out around 1. A feature with no spread at all
            has no z-score and yields ``0.0``. When this is enabled the plot's
            display scales become 1.0, since the traces are no longer in native
            units.

            .. versionchanged:: 1.2.0
                Previously σ was floored at ``1e-6``, which for power-like
                features replaced the real spread and made z-scores several
                orders of magnitude too small.
        zscore_warmup : int, default 10
            Number of windows to collect before activating normalisation.
            The mean and standard deviation of these windows are used as
            the initial statistics.

            Longer is better: σ estimated from a handful of windows carries
            its own sampling error, and the first analysis window of a run
            tends to read low (the acquisition buffer is still filling), which
            inflates σ and shrinks every later z-score. Measured on synthetic
            EEG band power, the spread of the resulting trace was ~0.5 at
            ``zscore_warmup=8`` and ~1.0 at 60.
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
            apply different protocols to different modalities.  Keys must
            name active modalities exactly; where a modality runs as several
            instances, each needs its own protocol *object*, since protocols
            accumulate state and one shared between instances would be fed
            several different values per window.  On each
            window the protocol's ``evaluate(value)`` method is called and
            ``(crossed, magnitude)`` is recorded.  Results are accessible via
            :attr:`reward_data` after the session.  When ``show_nf_signal=True``,
            each protocol's ``current_threshold`` (fixed or adaptive) is also
            drawn live as a dashed horizontal line on the corresponding
            :class:`~mne_rt.viz.NFPlot` subplot; modalities without a
            protocol, or protocols with no single-level threshold (e.g.
            :class:`~mne_rt.protocols.LinearTrendProtocol`), simply show no
            line.
        combiner : FeatureCombiner | None, default None
            Reduce the per-modality values to a single scalar once per window,
            after z-scoring and EMA smoothing.  The result is appended as an
            extra trace named ``combined_name`` and is treated as a modality
            throughout: it is plotted, can be driven by its own protocol, is
            broadcast over OSC/LSL, and is saved alongside the others.  The
            per-modality traces are kept, not replaced.

            Combining features whose units differ by orders of magnitude (band
            power in V²/Hz against a dimensionless laterality index, say)
            requires ``zscore_normalize=True``; otherwise the largest-scale
            feature dominates, and a warning is issued.
            See :mod:`mne_rt.combiners`.
        gate_conditions : list of str | None, default None
            Restrict feedback to windows running one of these conditions, as
            labelled by the marker stream attached with
            :meth:`connect_marker_stream`.  ``None`` — the default — leaves
            feedback running continuously, which is the behaviour of earlier
            versions.  A window outside the gate still computes its features
            and is saved, but no protocol is evaluated for it and nothing is
            sent over OSC or LSL; see Notes.
        combined_name : str, default "combined"
            Name of the combined trace.  Must not collide with a modality name.
        save_raw : bool, default False
            Persist the raw pre-correction M/EEG acquired during the main
            session to ``raw/<stem>-raw.fif``.  Off by default because FIF
            files can be large; enable when the raw continuous signal is
            needed for offline re-analysis or provenance.
        run : int | None, default None
            BIDS ``run-`` index for the output filenames, 1-based.  ``None``
            omits the entity, which is the behaviour of earlier versions.
            :meth:`run_blocks` sets it per block, so one block no longer
            overwrites the previous one's files.
        disconnect : bool, default True
            Disconnect the stream once the session has been saved.  ``False``
            keeps it open for a following block; see :meth:`run_blocks`.
        save_tsv : bool, default True
            Also write the per-window values as a BIDS ``_beh.tsv`` table with a
            ``_beh.json`` sidecar, alongside the session JSON.  The table carries
            an ``onset`` and ``duration`` column per window, so the trace can be
            aligned against a stimulus log without assuming a regular grid.
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

        * ``beh/<stem>_beh.json`` — NF feature time-series plus session metadata
        * ``beh/<stem>_beh.tsv`` — the same as a BIDS behavioural table
        * ``delays/<stem>_delays.json`` — per-step timing (only when
          ``estimate_delays=True``)
        * ``eeg/<stem>_eeg.fif`` — pre-correction M/EEG (only when
          ``save_raw=True``)

        where ``<stem>`` is ``sub-<ID>_ses-<session>_task-neurofeedback``, with
        ``_run-<NN>`` appended when ``run`` is given.

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
        # ceil, matching what `Stream.get_data(winsize)` actually returns. With
        # int() the two disagreed by one sample whenever winsize*sfreq was not a
        # whole number, and the length check below then discarded every window —
        # a silently empty session.
        self.window_size_s = int(np.ceil(winsize * self.rec_info["sfreq"]))
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
        # `run-` comes after `task-` in BIDS, so the whole stem is built in one
        # place rather than assembled from a prefix at each output path.
        self._session_stem = _build_bids_stem(
            self.subject_id,
            self.session,
            "neurofeedback",
            None if run is None else f"{int(run):02d}",
        )
        self._ensure_dirs(include_delays=estimate_delays)

        # Artifact correction setup
        ref_ch_idx: Optional[int] = None
        if self.artifact_correction == "lms":
            # indexes the acquired window, which excludes bads
            ref_ch_idx = self._acquired_ch_names().index(ref_channel)
        elif self.artifact_correction == "orica":
            self.run_orica(n_channels=len(self._acquired_ch_names()), forgetfac=0.99)
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
        #
        # A name may carry an instance label ("source_connectivity@theta"), so the
        # same measure can run several times over with different parameters.  The
        # *base* drives config lookup and method dispatch; the *full name* keys
        # every piece of per-instance state and every output channel.  A plain
        # name is just an instance with an empty label, so nothing below changes
        # behaviour for a session that does not use labels.
        specs = [parse_modality(m) for m in ([modality] if isinstance(modality, str) else modality)]
        mods = [s.name for s in specs]
        _dupes = sorted({m for m in mods if mods.count(m) > 1})
        if _dupes:
            raise ValueError(
                f"Modality name(s) {_dupes} listed more than once. Give each instance a "
                f"distinct label — e.g. {_dupes[0] + MODALITY_SEP + 'alpha'!r} and "
                f"{_dupes[0] + MODALITY_SEP + 'theta'!r} — so their values, smoothing, "
                "z-score state and protocols stay separate."
            )
        self._mods = mods
        self._mod_specs = specs

        # Validate protocol keys here, before the thread pool and the Qt windows
        # are created: raising once those exist would leave orphaned widgets on
        # screen and a live executor, since record_main has no teardown path.
        if isinstance(protocol, dict):
            _valid_keys = mods + ([combined_name] if combiner is not None else [])
            for _key in protocol:
                if _key in _valid_keys:
                    continue
                # Protocols carry state -- ZScoreProtocol's running mean and
                # variance, StaircaseProtocol's step, PercentileProtocol's
                # buffer.  Fanning one object across N instances would feed it
                # N different values per window and corrupt all of it, so a
                # base name that has instances is refused rather than expanded.
                _insts = sorted({s.name for s in specs if s.is_instanced and s.base == _key})
                if _insts:
                    raise ValueError(
                        f"Protocol key {_key!r} names a base modality with {len(_insts)} "
                        f"active instance(s) ({', '.join(_insts)}). Protocols are stateful, "
                        "so give each instance its own protocol object rather than sharing "
                        "one between them."
                    )
                raise ValueError(
                    f"Protocol key {_key!r} does not name an active modality. "
                    f"Valid keys are {_valid_keys}."
                )

        if gate_conditions is not None:
            if getattr(self, "marker_stream", None) is None:
                raise RuntimeError(
                    "`gate_conditions` needs a marker stream to gate on. Call "
                    "connect_marker_stream() (or connect_marker_array()) first."
                )
            gate_conditions = list(gate_conditions)
            if not gate_conditions:
                raise ValueError(
                    "`gate_conditions` is empty, which would mute the entire session. "
                    "Pass None to leave feedback ungated."
                )
            _known = set(getattr(self, "marker_id", {}) or {})
            _unknown = [c for c in gate_conditions if c not in _known]
            if _known and _unknown:
                raise ValueError(
                    f"Condition(s) {_unknown} are not in the marker stream's marker_id "
                    f"map, whose labels are {sorted(_known)}. A condition that never "
                    "matches would mute the whole session silently."
                )

        # Snapshot the marker buffer *before* the expensive setup below.
        # Feature preparation can build forward and inverse operators, and the
        # Qt windows take a moment more; a marker arriving inside that gap
        # belongs to this run, so the boundary has to be drawn here rather than
        # once the acquisition thread finally starts.
        _marker_start_hwm = 0.0
        _marker_start_condition: Optional[str] = None
        if getattr(self, "marker_stream", None) is not None:
            try:
                _m_data, _m_ts = self.marker_stream.get_data()
                _m_labels = getattr(self, "_marker_labels", {}) or {}
                _m_ch = int(getattr(self, "_marker_ch_idx", 0))
                for _i in range(_m_ts.size):
                    _t_m = float(_m_ts[_i])
                    if _t_m <= 0.0:
                        continue
                    if _t_m >= _marker_start_hwm:
                        _marker_start_hwm = _t_m
                        _code = int(round(float(_m_data[_m_ch, _i])))
                        # Carried forward as the starting condition, so a
                        # paradigm that announced the block before the run began
                        # is not treated as having said nothing.
                        _marker_start_condition = _m_labels.get(_code, str(_code))
            except Exception:
                logger.warning("Could not read the marker stream at startup.", exc_info=True)

        self.executor = ThreadPoolExecutor(max_workers=len(mods))
        self.mod_params_dict = {
            s.name: get_params(self.config_file, s.base, self.modality_params, instance=s.name)
            for s in specs
        }
        precomps: list[dict] = []
        nf_fns: list = []
        for s in specs:
            self.params = self.mod_params_dict[s.name]
            fn = getattr(self, f"_{s.base}", None)
            if not callable(fn):
                raise NotImplementedError(f"Modality '{s.base}' is not implemented.")
            # Test the base, not the full name: "sensor_power@source" is a sensor
            # modality whose label happens to read "source".
            if "source" in s.base and picks is not None:
                raise ValueError("'picks' must be None for source-space modalities.")
            prep = getattr(self, f"_{s.base}_prep", None)
            precomps.append(prep() if callable(prep) else {})
            nf_fns.append(fn)
        # Leaving this pointing at the last instance's params would quietly hand
        # them to anything that read it at window time.
        self.params = None

        # ---- Visualization setup (main thread) ----

        scales_dict = {
            "sensor_power": self._SENSOR_POWER_SCALE[self.data_type],
            "band_ratio": 4.0,
            "source_power": 3e-2,
            "sensor_connectivity": 1.0,
            "source_connectivity": 1.0,
            "connectivity_ratio": 4.0,
            "sensor_graph": 0.05,
            "source_graph": 2e-17,
            "entropy": 3.0,
            "argmax_freq": 8.0,
            "peak_alpha_freq": 13.0,
            "individual_peak_power": self._SENSOR_POWER_SCALE[self.data_type],
            "cfc_sensor": 1.0,
            "erd_ers": 50.0,
            "laterality": 2.0,
            "hjorth": 5.0,
            "spectral_centroid": 5.0,
            "scp": 50e-6,
            "decode": 1.0,
            "instantaneous_phase": 3.2,  # radians, |phase| <= pi
            "laterality_erd_ers": 50.0,
        }
        # NFPlot indexes scales_dict with a bare [], so every *displayed* name
        # needs an entry: an instance inherits its base modality's scale unless
        # the caller gave it one of its own.
        for s in specs:
            if s.name not in scales_dict and s.base in scales_dict:
                scales_dict[s.name] = scales_dict[s.base]
        if zscore_normalize:
            # The traces are no longer in native units — a z-score sits around 1
            # whatever the feature. Dividing one by a 1e-12 band-power scale
            # would put it twelve orders off-screen. During the warmup windows
            # the raw value is still passed through, so the trace sits flat near
            # the axis origin until normalisation engages.
            scales_dict = dict.fromkeys(scales_dict, 1.0)
        # NFPlot looks every displayed modality up in scales_dict, so the
        # combined trace needs an entry of its own.
        if combiner is not None:
            if not callable(getattr(combiner, "combine", None)):
                raise TypeError(
                    "`combiner` must expose a combine(values) method "
                    f"(see mne_rt.FeatureCombiner); got {type(combiner).__name__}."
                )
            # The combined trace shares namespaces with the modalities, the
            # per-modality scales and labels, and the columns save() writes.
            _reserved = set(mods) | {"snr_db"} | set(scales_dict)
            if combined_name in _reserved or combined_name.startswith("reward_"):
                raise ValueError(
                    f"`combined_name` {combined_name!r} is already in use — it collides with "
                    "an active modality, a built-in modality name (whose display scale and "
                    "axis label it would inherit), or a reserved output column "
                    "('snr_db', 'reward_*'). Choose another."
                )
            expected = getattr(combiner, "features", None)
            missing = [f for f in (expected or []) if f not in mods]
            if missing:
                # A feature naming a base modality that is only present as
                # instances is the likely mistake; say which ones it could mean.
                hint = ""
                _cands = sorted({s.name for s in specs if s.is_instanced and s.base in missing})
                if _cands:
                    hint = f" Did you mean {_cands}?"
                raise ValueError(
                    f"{type(combiner).__name__} expects feature(s) {missing}, which are not "
                    f"among the active modalities {mods}.{hint}"
                )
            if isinstance(combiner, ZScoredNormCombiner):
                pass  # normalises each feature itself; nothing to advise
            elif isinstance(combiner, GeometricMeanCombiner):
                if zscore_normalize:
                    warn(
                        "GeometricMeanCombiner takes the log of each value, so it needs "
                        "positive inputs, but z-scores are negative in roughly half of all "
                        "windows. Each negative feature is dropped from that window's "
                        "product (or clipped, if you set `floor`), so the combined trace "
                        "is a geometric mean over a different subset of features from one "
                        "window to the next — a discontinuous signal that is hard to "
                        "interpret and harder to reward — and 0.0 whenever every feature "
                        "is negative. Use ZScoredNormCombiner to mix features of different "
                        "scale, or keep zscore_normalize=False and feed "
                        "GeometricMeanCombiner strictly positive features.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            elif not zscore_normalize:
                warn(
                    f"{type(combiner).__name__} blends its features in their native units. "
                    "If they differ in scale — band power is ~1e-11 V²/Hz while a laterality "
                    "index is ~1 — the largest one dominates the combined value. Pass "
                    "zscore_normalize=True, or use ZScoredNormCombiner, if that applies "
                    "to the modalities you are combining.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Each record_main() is a fresh session, and record_main rebuilds its own
            # z-score state per call; reset the combiner's so the two normalisation
            # layers agree across the blocks of a run_blocks() sequence.
            if callable(getattr(combiner, "reset", None)):
                combiner.reset()

        # The combined trace is a modality as far as the plot, the protocols, the
        # feedback outputs and the saved data are concerned.
        _display_mods = mods + [combined_name] if combiner is not None else list(mods)

        # OSC addresses are built once here rather than per window.  The
        # separator is mapped to something an OSC receiver will tolerate, which
        # can in principle make two names collide — and the send path cannot
        # report that, since a failure there must not kill acquisition.
        _osc_names = [osc_address_name(m) for m in _display_mods]
        _send_warned = [False, False]  # OSC, LSL — log the first failure only
        if osc_sender is not None:
            if len(set(_osc_names)) != len(_osc_names):
                _clash = sorted({n for n in _osc_names if _osc_names.count(n) > 1})
                raise ValueError(
                    f"Modality names collide once sanitised for OSC: {_clash}. "
                    "Rename an instance label so the addresses stay distinct."
                )
            # Say so rather than silently publishing to a different address than
            # the name would suggest.
            _renamed = {m: n for m, n in zip(_display_mods, _osc_names) if m != n}
            if _renamed:
                warn(
                    "These names are not valid in an OSC address and are sent to a "
                    f"sanitised one instead: {_renamed}. Subscribe to the address on "
                    "the right.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if combiner is not None:
            # A z-scored blend sits around 1; a blend of raw features keeps the
            # scale of its inputs, and a fixed 1.0 would flatten it to zero on
            # the plot. Take the largest constituent scale as the closest guess.
            _default_scale = 1.0 if zscore_normalize else max(scales_dict.get(m, 1.0) for m in mods)
            scales_dict.setdefault(combined_name, _default_scale)

        nf_plot: Optional[NFPlot] = None
        raw_plot: Optional[RawPlot] = None
        topo_plot: Optional[TopomapPlot] = None
        brain_plot: Optional[BrainPlot] = None

        if show_brain_activation:
            # Before any window is shown: `_push_brain` returns early without an
            # inverse operator, so the display would stay empty otherwise -- and
            # on a cold cache this fetches the anatomy, which should not happen
            # behind windows that are up but not yet painted.
            self._ensure_head_model("The brain-activation display", need_inverse=True)

        needs_qt = show_nf_signal or show_brain_activation or show_raw_signal or show_topo
        app: Optional[QtWidgets.QApplication] = None

        if needs_qt:
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        if show_nf_signal:
            nf_plot = NFPlot(
                modalities=_display_mods,
                scales_dict=scales_dict,
                sfreq=30.0,  # pump timer rate drives display at 30 fps
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
                raise ValueError("subjects_fs_dir must be set to use brain activation display.")
            brain_plot = BrainPlot(
                subjects_fs_dir=self.subjects_fs_dir,
                clim=[0, 0.6],
                surf=brain_surf,
                display_smoothing=brain_display_smoothing,
            )

        if show_raw_signal:
            raw_plot = RawPlot(
                ch_names=self._acquired_ch_names(),
                sfreq=self._sfreq,
                info=self._acquired_info(),
            )
            raw_plot.show()

        # Once per connection, not once per call: mne-lsl *appends* to the
        # stream's filter chain and ArrayStream rewrites its source array in
        # place, so a second record_main() on the same stream would run the data
        # through a second band-pass, a third through a third.
        if not getattr(self, "_filters_applied", False):
            if self.bandpass_freq is not None:
                self.stream.filter(l_freq=self.bandpass_freq[0], h_freq=self.bandpass_freq[1])
            if self.notch_freq is not None:
                _freqs = self.notch_freq if isinstance(self.notch_freq, list) else [self.notch_freq]
                for _f in _freqs:
                    self.stream.notch_filter(freqs=_f)
            self._filters_applied = True

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
        # Set when record_main is finishing, so an acquisition thread that
        # outlives the join -- the Qt windows closed early, say -- stops instead
        # of running on into the next block, where it would steal samples from
        # its successor and submit into a pool that no longer exists.
        stop_event = threading.Event()
        nf_data: dict[str, list] = {m: [] for m in _display_mods}
        _ema: dict[str, float] = {}  # EMA state, seeded on first window

        # Build protocol map: {modality_name: protocol_instance}
        if protocol is None:
            _proto_map: dict[str, Any] = {}
        elif isinstance(protocol, dict):
            _proto_map = dict(protocol)  # keys already validated above
        else:
            _proto_map = {mods[0]: protocol}
        reward_data: dict[str, list] = {m: [] for m in _proto_map}

        # Delay accumulators live in the thread, assigned to self when done
        _acq_delays: list[float] = []
        _art_delays: list[float] = []
        _meth_delays: dict[str, list] = {s.name: [] for s in specs}
        _plot_delays: list[float] = []  # only used when viz runs inline

        # ---- Artifact rate tracking ----
        _n_total_windows: list[int] = [0]
        _n_artifact_windows: list[int] = [0]
        _artifact_threshold_raw = artifact_threshold_uv * 1e-6  # V for EEG, T for MEG

        # ---- SNR tracking ----
        _snr_data: list[float] = []

        # ---- Per-window timing ----
        # The hop is a busy-wait that re-reads the clock *after* the previous
        # window's feature computation, so realised onsets drift from the
        # nominal index*hop grid — several seconds over a long run. Record the
        # real onset instead of letting readers infer it.
        # ---- Marker state ----
        # `_marker_latch` is a running variable rather than something recomputed
        # per window from `_markers`: a window whose onset is NaN has no interval
        # to search, and recomputing would un-latch it.
        _marker_stream = getattr(self, "marker_stream", None)
        _marker_labels = dict(getattr(self, "_marker_labels", {}) or {})
        _marker_ch = int(getattr(self, "_marker_ch_idx", 0))
        _markers: list[tuple[float, int]] = []  # (lsl timestamp, code)
        # Condition established before the run started, used for windows that
        # precede the first marker of this run.
        _marker_latch: list[Optional[str]] = [_marker_start_condition]
        _marker_hwm = [0.0]  # timestamp high-water mark; 0.0 also skips zero-fill
        _marker_hwm_n = [0]  # samples already consumed *at* the mark
        _marker_warned = [False, False, False]  # poll failed, overflow, >1 per window
        _prev_win_end = [0.0]  # so each marker is attributed to exactly one window
        _n_gated = [0]
        _win_conditions: list[Optional[str]] = []
        _win_marker_counts: list[int] = []
        _win_gated: list[int] = []

        def _poll_markers() -> None:
            """Drain new markers into `_markers` and advance the latch.

            Idempotent, so it is safe to call as often as convenient: a stream's
            ``get_data`` returns the *whole* buffer every time, and the
            high-water mark is what stops a marker being consumed twice.
            """
            if _marker_stream is None:
                return
            try:
                # Before get_data, which resets the counter to zero.
                n_new = int(getattr(_marker_stream, "n_new_samples", 0))
                mdata, mts = _marker_stream.get_data()
            except Exception:
                # A raise here would skip done_event.set() at the end of the
                # loop body and hang the Qt event loop forever.
                if not _marker_warned[0]:
                    _marker_warned[0] = True
                    logger.warning(
                        "Reading the marker stream failed; continuing without markers.",
                        exc_info=True,
                    )
                return
            if n_new >= mts.size and mts.size and not _marker_warned[1]:
                _marker_warned[1] = True
                logger.warning(
                    "The marker buffer (%d) filled between polls, so markers were "
                    "overwritten before they could be read. Increase `bufsize` on "
                    "connect_marker_stream().",
                    mts.size,
                )
            mark, seen_at_mark = _marker_hwm[0], _marker_hwm_n[0]
            n_at_mark = 0
            for i in range(mts.size):
                ts = float(mts[i])
                if ts <= 0.0:  # preallocated, never written
                    continue
                if ts < mark:
                    continue
                if ts == mark:
                    # Equal timestamps are legitimate (a pushed chunk, or the
                    # `monotize` flag), so dedupe by count rather than by value.
                    n_at_mark += 1
                    if n_at_mark <= seen_at_mark:
                        continue
                code = int(round(float(mdata[_marker_ch, i])))
                _markers.append((ts, code))
                _marker_latch[0] = _marker_labels.get(code, str(code))
                if ts > _marker_hwm[0]:
                    _marker_hwm[0] = ts
                    _marker_hwm_n[0] = 1
                elif ts == _marker_hwm[0]:
                    _marker_hwm_n[0] += 1

        # `_marker_start_hwm` was taken at the top of record_main, before the
        # feature preparation and the Qt windows: anything already in the buffer
        # then predates this run — a previous block, or the paradigm warming up.
        # Seeding the mark rather than draining here is what stops a marker that
        # arrives *during* setup from being consumed and thrown away.
        _marker_hwm[0] = float(_marker_start_hwm)
        _marker_latch[0] = _marker_start_condition

        _win_onsets: list[float] = []  # absolute, LSL clock
        _win_durations: list[float] = []
        _n_short: list[int] = [0]  # windows dropped for being the wrong length
        _short_warned: list[bool] = [False]
        _snr_frange = (
            snr_frange
            if snr_frange is not None
            else (
                tuple(self.mod_params_dict[mods[0]].get("frange", [8, 13])) if mods else (8.0, 13.0)
            )
        )

        # ---- Z-score normalisation state ----
        # The warmup buffer is dropped once its statistics are taken; `_z_n` is
        # the window counter, kept separately because the combiner warm-gate
        # below reads it after the buffer is gone. Variance rather than standard
        # deviation, so the adaptive update composes correctly.
        _z_buf: dict[str, list] = {s.name: [] for s in specs}
        _z_n: dict[str, int] = {s.name: 0 for s in specs}
        _z_mean: dict[str, float] = {}
        _z_var: dict[str, float] = {}

        def _apply_zscore(mod: str, val: float) -> float:
            if not zscore_normalize:
                return val
            val = float(val)
            _z_n[mod] += 1
            n = _z_n[mod]
            if n < zscore_warmup:
                _z_buf[mod].append(val)
                return val  # pass-through during warmup
            if n == zscore_warmup:
                buf = _z_buf[mod]
                buf.append(val)
                arr = np.array(buf, dtype=float)
                _z_mean[mod] = float(arr.mean())
                # ddof=1: these statistics are estimated from a finite baseline
                # and used to standardise future, unseen values.
                _z_var[mod] = float(arr.var(ddof=1))
                _z_buf[mod] = []  # dead from here on; `_z_n` carries the count
            elif zscore_alpha > 0.0:
                # One `delta`, taken before the mean moves, so mean and variance
                # advance on the same step.
                delta = val - _z_mean[mod]
                _z_var[mod] = ema_variance(_z_var[mod], delta, zscore_alpha)
                _z_mean[mod] += zscore_alpha * delta
            # No floor: a degenerate spread yields 0.0 rather than a value
            # divided by an arbitrary constant. See mne_rt._stats.
            return zscore(val, _z_mean[mod], math.sqrt(_z_var[mod]))

        # ---- Shared artifact-correction helper ----

        def _correct(data: np.ndarray) -> np.ndarray:
            if self.artifact_correction == "lms":
                art_tic = time.time()
                data = remove_blinks_lms(data, ref_ch_idx=ref_ch_idx, n_taps=5, mu=0.01)
                if estimate_delays:
                    _art_delays.append(time.time() - art_tic)
            elif self.artifact_correction == "orica":
                art_tic = time.time()
                data = self.orica.denoise(
                    data, self.orica.find_blink_ic(self.blink_template, threshold=0.4)[0]
                )
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
                        l_freq=brain_freq_range[0],
                        h_freq=brain_freq_range[1],
                        fir_design="firwin",
                        verbose=False,
                    )
                stc = apply_inverse_raw(raw_d, self.inv, lambda2=1.0 / 9, pick_ori="normal")
                if brain_mode == "power":
                    lh = np.mean(stc.lh_data**2, axis=1)
                    rh = np.mean(stc.rh_data**2, axis=1)
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
            # `done_event` is the only way the Qt event loop learns the run is
            # over (see `_pump_signal`), so anything raised in here would leave
            # record_main spinning forever with nothing saved.
            try:
                t_start = local_clock()
                _raw_chunks: list[np.ndarray] = []
                # 50 % overlap: advance by half a window each step so consecutive
                # NF estimates share data → smooth, correlated curve updates.
                _hop = max(1, self.window_size_s // 2)

                while local_clock() < t_start + duration and not stop_event.is_set():
                    # Block until half a window of new samples has arrived, then
                    # fetch the latest winsize seconds (50 % overlap with prev window).
                    # While waiting, flush any accumulated raw samples to the display
                    # queue at ~30 fps so RawPlot scrolls smoothly.
                    _t_hop = local_clock()
                    _hop_dur = _hop / self._sfreq
                    while local_clock() < _t_hop + _hop_dur:
                        if local_clock() >= t_start + duration or stop_event.is_set():
                            break
                        if raw_queue is not None:
                            n_avail = self.stream.n_new_samples
                            if n_avail >= max(1, int(0.033 * self._sfreq)):
                                try:
                                    raw_chunk = self.stream.get_data(n_avail / self._sfreq)[0]
                                    if raw_chunk.shape[1] > 0:
                                        raw_queue.put_nowait(raw_chunk)
                                except (_queue.Full, Exception):
                                    pass
                        # Outside the raw_queue guard: markers must be drained even
                        # with no viewer open, or the ring buffer overflows.
                        _poll_markers()
                        time.sleep(0.005)

                    tic = time.time()
                    data, _ts = self.stream.get_data(winsize, picks=picks)
                    if estimate_delays:
                        _acq_delays.append(time.time() - tic)
                    if data.shape[1] != self.window_size_s:
                        # Now that window_size_s uses ceil this should only happen
                        # when the buffer cannot supply a whole window — winsize
                        # larger than bufsize, or the very start of a session.
                        _n_short[0] += 1
                        if not _short_warned[0]:
                            _short_warned[0] = True
                            logger.warning(
                                "Analysis window has %d samples, expected %d — dropping it. "
                                "If this repeats, winsize (%.3f s) likely exceeds the stream "
                                "buffer (%.3f s).",
                                data.shape[1],
                                self.window_size_s,
                                winsize,
                                float(getattr(self, "bufsize", 0) or 0),
                            )
                        continue

                    # The timestamp buffer is preallocated with zeros, so `_ts[0]` can
                    # be a literal 0.0 rather than a clock value early in a session.
                    # Derive the onset from the tail, which the hop wait guarantees is
                    # real, and fall back to the wall clock if even that is unfilled.
                    _t_end = float(_ts[-1]) if _ts.size else 0.0
                    _win_dur = data.shape[1] / self._sfreq
                    if _t_end > 0.0:
                        _onset = _t_end - (data.shape[1] - 1) / self._sfreq
                    else:
                        # Still on the preallocated zeros. Substituting this host's
                        # local_clock() would put one window on a different clock
                        # from the rest whenever the sender's differs, quietly
                        # shifting every session-relative onset. Unknown is honest.
                        _onset = float("nan")
                    _win_onsets.append(_onset)
                    _win_durations.append(_win_dur)

                    # Poll again now that the window's end is known: the last poll in
                    # the hop wait can be 5 ms stale, and a marker arriving in that
                    # gap belongs to this window, not the next one.
                    _poll_markers()
                    # From the newest marker at or before this window's end, not
                    # from the free-running latch: the poll above also consumes
                    # markers that arrived after `_t_end` (the M/EEG buffer lags
                    # on chunked devices), and letting one of those label this
                    # window would open the gate a window early and disagree with
                    # `n_markers`. A window with an unknown end has no interval
                    # to search, so it keeps the latch.
                    if _t_end > 0.0:
                        _condition = _marker_start_condition
                        for _t_m, _c_m in reversed(_markers):
                            if _t_m <= _t_end:
                                _condition = _marker_labels.get(_c_m, str(_c_m))
                                break
                    else:
                        _condition = _marker_latch[0]
                    # Count over `(previous window end, this window end]`, not
                    # over the window itself. Windows overlap by half, so a
                    # contained marker would be counted twice; this partitions
                    # the run, and the column sums to the number received.
                    # `_t_end` rather than `_onset + _win_dur`, which overshoots
                    # by one sample.
                    if _t_end > 0.0:
                        _lo = _prev_win_end[0] if _prev_win_end[0] > 0.0 else _onset
                        _n_win_markers = sum(1 for _t_m, _ in _markers if _lo < _t_m <= _t_end)
                        _prev_win_end[0] = _t_end
                    else:
                        _n_win_markers = 0
                    if _n_win_markers > 1 and not _marker_warned[2]:
                        _marker_warned[2] = True
                        logger.warning(
                            "%d markers arrived within one analysis window. The trial "
                            "rate is outrunning the neurofeedback cadence, so only the "
                            "last one labels the window.",
                            _n_win_markers,
                        )
                    _in_gate = gate_conditions is None or _condition in gate_conditions
                    if not _in_gate:
                        _n_gated[0] += 1
                    if _marker_stream is not None:
                        # Only when a marker stream is attached: appending
                        # placeholders otherwise would add three all-empty
                        # columns to every existing user's table.
                        _win_conditions.append(_condition)
                        _win_marker_counts.append(_n_win_markers)
                        _win_gated.append(0 if _in_gate else 1)

                    _n_total_windows[0] += 1
                    if track_artifact_rate:
                        if np.any(np.abs(data) > _artifact_threshold_raw):
                            _n_artifact_windows[0] += 1

                    _raw_chunks.append(data.copy())
                    data = _correct(data)

                    if track_snr:
                        from mne_rt.tools import compute_bandpower

                        _sig = compute_bandpower(data, self._sfreq, _snr_frange, method="welch")
                        _all = compute_bandpower(
                            data, self._sfreq, (0.5, self._sfreq / 2 - 1), method="welch"
                        )
                        _noise = _all.mean() - _sig.mean()
                        _snr_data.append(
                            float(10.0 * np.log10(_sig.mean() / (abs(_noise) + 1e-300)))
                        )

                    futures = [
                        self.executor.submit(nf_fns[i], data, **precomps[i])
                        for i in range(len(specs))
                    ]
                    _crossed_map: dict = {}
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
                            # `_apply_zscore` passes the raw value through until this
                            # modality has `zscore_warmup` windows behind it, so
                            # evaluating during that period rewards native-unit values
                            # against a z-score-calibrated threshold. The gate is
                            # per-modality: `all(...)` would block one band's protocol
                            # because another band had not warmed yet.
                            _m_warm = not zscore_normalize or _z_n[m] >= zscore_warmup
                            if _in_gate and _m_warm:
                                _crossed, _mag = _proto_map[m].evaluate(nf_val)
                                reward_data[m].append(_mag if _crossed else 0.0)
                                _crossed_map[m] = _crossed
                            else:
                                # Not evaluating is what "do not update" means for a
                                # stateful protocol; the placeholder keeps this list
                                # the same length as nf_data[m].
                                reward_data[m].append(0.0)
                        if estimate_delays:
                            _meth_delays[m].append(m_delay)

                    if combiner is not None:
                        # _apply_zscore passes values through unchanged until each
                        # modality has `zscore_warmup` windows behind it. Combining
                        # during that period would mix native units — the very thing
                        # zscore_normalize=True is meant to prevent — and the combined
                        # trace would then jump by orders of magnitude the moment
                        # normalisation engaged, fitting any attached protocol's
                        # baseline on meaningless numbers. Hold at 0.0 until every
                        # feature is normalised, as ZScoredNormCombiner does.
                        _warm = not zscore_normalize or all(_z_n[m] >= zscore_warmup for m in mods)
                        if _warm:
                            try:
                                _mixed = float(combiner.combine({m: nf_data[m][-1] for m in mods}))
                            except Exception:
                                # User-supplied code; a raise here would skip
                                # done_event.set() below and hang the Qt loop.
                                logger.exception(
                                    "%s.combine() failed; using 0.0 for this window.",
                                    type(combiner).__name__,
                                )
                                _mixed = 0.0
                        else:
                            _mixed = 0.0
                        nf_data[combined_name].append(_mixed)
                        if _warm and _in_gate and combined_name in _proto_map:
                            _crossed, _mag = _proto_map[combined_name].evaluate(_mixed)
                            reward_data[combined_name].append(_mag if _crossed else 0.0)
                            _crossed_map[combined_name] = _crossed
                        elif combined_name in _proto_map:
                            reward_data[combined_name].append(0.0)

                    _vals = [nf_data[m][-1] for m in _display_mods]
                    _threshs = [
                        getattr(_proto_map[m], "current_threshold", None)
                        if m in _proto_map
                        else None
                        for m in _display_mods
                    ]
                    _rewards = [_crossed_map.get(m) for m in _display_mods]
                    try:
                        nf_queue.put_nowait((_vals, _threshs, _rewards))
                    except _queue.Full:
                        pass

                    if osc_sender is not None and _in_gate:
                        try:
                            osc_sender.send_all(_osc_names, _vals)
                        except Exception:
                            # Feedback delivery must never take the acquisition
                            # thread down, but silence for a whole session is worse:
                            # report the first failure, then stay quiet.
                            if not _send_warned[0]:
                                _send_warned[0] = True
                                logger.warning(
                                    "OSC send failed; continuing without OSC feedback.",
                                    exc_info=True,
                                )

                    if lsl_sender is not None and _in_gate:
                        try:
                            lsl_sender.push(_display_mods, _vals)
                        except Exception:
                            if not _send_warned[1]:
                                _send_warned[1] = True
                                logger.warning(
                                    "LSL push failed; continuing without LSL feedback.",
                                    exc_info=True,
                                )

                    _push_topo(data)
                    _push_brain(data)

            except BaseException:
                logger.exception("Acquisition failed; ending the run.")
                raise
            finally:
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
            # Threshold lines and reward zones update once per analysis
            # window (not per display frame), so the latest snapshot is
            # just held as-is rather than ramped like the value trace.
            _latest_threshs: list = [None] * len(_display_mods)
            _latest_rewards: list = [None] * len(_display_mods)

            def _pump_signal() -> None:
                """Fast timer (~30 fps) — signal plot only.

                Linearly interpolates between the two most-recent NF estimates
                so the trace ramps smoothly over one window period.
                """
                try:
                    while not nf_queue.empty():
                        new_vals, new_threshs, new_rewards = nf_queue.get_nowait()
                        _interp[0] = _interp[1] if _interp[1] else new_vals
                        _interp[1] = new_vals
                        _interp[2][0] = 0
                        _latest_threshs[:] = new_threshs
                        _latest_rewards[:] = new_rewards
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
                    nf_plot.push(vals, thresholds=_latest_threshs, rewards=_latest_rewards)
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
            signal_timer.setInterval(33)  # ~30 fps
            signal_timer.timeout.connect(_pump_signal)
            signal_timer.start()
            if nf_plot is not None:
                # Stop pumping the instant this window closes -- otherwise
                # the timer keeps firing into a closed widget while other
                # plot windows stay open, which crashes Qt/pyqtgraph.
                nf_plot.closed.connect(signal_timer.stop)

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
                topo_plot.closed.connect(topo_timer.stop)

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
                            brain_plot.update_from_arrays(lh, rh, mode=brain_mode, deferred=True)
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
                raw_plot.closed.connect(raw_timer.stop)

            app.exec()  # blocks here; drives all three windows in parallel

        else:
            acq_thread.join()  # headless: block until acquisition finishes

        # Ask first, then wait: the loop runs to `t_start + duration` regardless
        # of whether the windows are still open, so a bounded join alone can
        # return with the thread still live.
        stop_event.set()
        acq_thread.join(timeout=10)
        if acq_thread.is_alive():
            logger.warning(
                "The acquisition thread did not stop within 10 s; the session is being "
                "saved without it."
            )
        # After the join, so the brain-plot task submitted from the acquisition
        # thread cannot still be running `apply_inverse_raw` while the session
        # is serialised below. Left in place rather than set to None, because a
        # thread that outlived the join would then submit into `None`.
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True)

        # ---- Persist results ----

        self.nf_data = nf_data
        self.reward_data = reward_data
        self._zscore_normalize = zscore_normalize
        # The protocol objects themselves, not just their rewards: `create_report`
        # needs `current_threshold` to say what the rewards were judged against,
        # and it is unrecoverable once this frame goes away.
        self._protocols = _proto_map
        self._combined_name = combined_name if combiner is not None else None
        # Absolute LSL-clock onsets; `save()` converts to session-relative.
        self.window_onsets = _win_onsets
        self.window_durations = _win_durations
        self.n_short_windows = _n_short[0]
        self.window_conditions = _win_conditions
        self.window_marker_counts = _win_marker_counts
        self.window_gated = _win_gated
        self.markers = list(_markers)
        self.gate_conditions = gate_conditions
        self.n_gated_windows = _n_gated[0]

        # A gate that never opened looks exactly like a working session from the
        # outside: features are computed and saved, the plot updates, and the
        # subject simply never receives anything. Say so.
        if (
            gate_conditions is not None
            and _n_total_windows[0] > 0
            and _n_gated[0] == _n_total_windows[0]
        ):
            logger.warning(
                "Every one of the %d windows was outside gate_conditions=%s, so no "
                "feedback was delivered for the entire run. %d marker(s) arrived. "
                "Check that the paradigm publishes the expected codes, and that both "
                "streams are clock-synchronised if they come from different machines.",
                _n_total_windows[0],
                gate_conditions,
                len(_markers),
            )
        if estimate_delays:
            self.acq_delays = _acq_delays
            self.artifact_delays = _art_delays
            self.method_delays = _meth_delays
        n_tot = _n_total_windows[0]
        self.artifact_rate = _n_artifact_windows[0] / n_tot if n_tot > 0 else 0.0
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
            # The TSV and its sidecar are what an analysis pipeline reads; the
            # JSON is the full record. Writing only the JSON meant the per-window
            # table never reached disk in the default flow.
            bids_tsv=save_tsv,
            disconnect=disconnect,
        )
        for kind, path in saved_files.items():
            logger.info("Saved %s → %s", kind, path)

        # All of them, not just the neurofeedback trace: leaving the others open
        # means a second block stacks a fresh set of windows on top of the last.
        for _w in (nf_plot, raw_plot, topo_plot):
            if _w is not None:
                try:
                    _w.close()
                except Exception:
                    logger.debug("Closing %s failed.", type(_w).__name__, exc_info=True)
        if brain_plot is not None:
            # Not a QWidget: it wraps a PyVista plotter and has no close() of
            # its own, so a bare `.close()` would raise into a swallowed
            # exception and leave the window on screen.
            try:
                brain_plot.stop_recording()
            except Exception:
                logger.debug("Stopping the brain recording failed.", exc_info=True)
            try:
                brain_plot.plotter.close()
            except Exception:
                logger.debug("Closing the brain plotter failed.", exc_info=True)

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
        :meth:`record_main`.  Each block is written under its own BIDS
        ``run-`` entity — ``run-01``, ``run-02``, … — so one block no longer
        overwrites the previous one's files.  Pass ``"run"`` in a block dict to
        override the index.

        The stream stays connected between blocks.  Only the final block
        disconnects it, because :meth:`save`'s teardown also stops the mock
        player, which reconnecting the stream alone cannot undo.

        The returned list carries each block's feature time-series.
        :attr:`block_results` carries the rest — rewards, window onsets and
        durations, dropped-window counts and artifact rates — which are
        otherwise overwritten by the following block.

        Protocol objects are **not** reset between blocks: a
        :class:`~mne_rt.protocols.ZScoreProtocol` passed to two blocks carries
        its running statistics from one into the next.  A combiner *is* reset.
        If you want a protocol to start fresh, pass a new instance per block.

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
        all_results: list[dict] = []

        for i, block in enumerate(blocks):
            block_kwargs = dict(block)
            post_rest = float(block_kwargs.pop("rest", rest_duration))
            _last = i == len(blocks) - 1

            logger.info(
                "run_blocks: starting block %d/%d (duration=%.0f s)",
                i + 1,
                len(blocks),
                block_kwargs.get("duration", 0),
            )
            # Popped before the call rather than inside it: relying on argument
            # evaluation order to run these before `**block_kwargs` is expanded
            # would be a duplicate-keyword TypeError waiting to happen.
            # Each block writes its own files instead of overwriting the previous
            # block's, and the stream stays up between blocks -- `save()` stops
            # the mock player, which reconnecting the stream alone cannot undo.
            block_run = block_kwargs.pop("run", i + 1)
            block_disconnect = block_kwargs.pop("disconnect", _last)
            self.record_main(
                verbose=verbose,
                run=block_run,
                disconnect=block_disconnect,
                **block_kwargs,
            )
            # Everything record_main rebinds, not the three that happened to be
            # captured before: the rest were overwritten by the next block and
            # lost.
            all_nf_data.append(dict(self.nf_data))
            all_artifact_rates.append(self.artifact_rate)
            all_snr_data.append(list(self.snr_data))
            all_results.append(
                {
                    "run": block_run,
                    "nf_data": dict(self.nf_data),
                    "reward_data": {k: list(v) for k, v in self.reward_data.items()},
                    "window_onsets": list(getattr(self, "window_onsets", [])),
                    "window_durations": list(getattr(self, "window_durations", [])),
                    "n_short_windows": getattr(self, "n_short_windows", 0),
                    "artifact_rate": self.artifact_rate,
                    "n_total_windows": getattr(self, "n_total_windows", 0),
                    "n_artifact_windows": getattr(self, "n_artifact_windows", 0),
                    "snr_data": list(self.snr_data),
                    # Present only with estimate_delays=True, and overwritten by
                    # the next block like everything else here.
                    "acq_delays": list(getattr(self, "acq_delays", [])),
                    "artifact_delays": list(getattr(self, "artifact_delays", [])),
                    "method_delays": {
                        k: list(v) for k, v in getattr(self, "method_delays", {}).items()
                    },
                }
            )

            if not _last and post_rest > 0:
                logger.info("run_blocks: rest period %.0f s …", post_rest)
                time.sleep(post_rest)

        self.block_nf_data = all_nf_data
        self.block_artifact_rates = all_artifact_rates
        self.block_snr_data = all_snr_data
        self.block_results = all_results
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
        raw = RawArray(data, self._acquired_info(), verbose=False)
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

        n_channels = len(self._acquired_ch_names())
        self.gedai = GEDAIDenoiser(n_channels=n_channels, shrinkage=shrinkage)

        if use_leadfield:
            # The forward model is built on demand, so without this the
            # leadfield branch below would never be taken and GEDAI would fall
            # back to a different algorithm with only a log line to say so.
            self._ensure_head_model("GEDAI leadfield mode")

        if use_leadfield and getattr(self, "fwd", None) is not None:
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
                "connect_to_lsl() or connect_to_array() must be called before "
                "fit_maxwell() so that sensor info is available."
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
        *,
        src_type: Optional[str] = None,
        volume_labels: Optional[list] = None,
        data_cov: Optional[Any] = None,
        data_cov_method: str = "empirical",
        make_inverse: bool = True,
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
        src_type : {"surface", "volume"} | None, default None
            Overrides the session's ``source_space`` for this call.
        volume_labels : list of str | None, default None
            Restrict a volumetric source space to these atlas labels.
            Strongly recommended: a whole-brain 5 mm grid is ~14 600 source
            points, while a typical ROI set totals a few hundred — the
            difference between a real-time-capable pipeline and one that is
            not.
        data_cov : instance of Covariance | None, default None
            Data covariance for beamforming.  ``None`` estimates it from the
            baseline recording.  This is a *separate* quantity from
            ``noise_cov``: :func:`mne.beamformer.make_lcmv` adapts its spatial
            filter to the data covariance, and passing a noise covariance in
            its place yields a filter that is not adaptive at all.
        data_cov_method : str, default "empirical"
            Estimator for the data covariance when it is computed here.
        make_inverse : bool, default True
            Build a minimum-norm inverse operator.  Set ``False`` for a
            beamformer-only session.

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
        src_type = src_type if src_type is not None else self.source_space
        self.inv, self.fwd, self.noise_cov, self.src, _raw_fwd = _compute_inv_operator(
            self.raw_baseline,
            subject_fs_id=self.subject_fs_id,
            subjects_fs_dir=self.subjects_fs_dir,
            data_type=self.data_type,
            loose=loose,
            depth=depth,
            noise_cov_method=noise_cov_method,
            reg=reg,
            src_type=src_type,
            vol_pos=self.source_pos,
            vol_atlas=self.source_atlas if src_type == "volume" else "aparc+aseg",
            volume_labels=volume_labels,
            make_inverse=make_inverse,
        )

        # Data covariance — what a beamformer actually adapts to.  Estimated on
        # the same Raw the forward model was built from, so the projectors match.
        if data_cov is not None:
            self.data_cov = data_cov
        else:
            # Estimate on `_raw_fwd`, the recording the forward model and noise
            # covariance were built from — same channels, same average-reference
            # projection. Using self.raw_baseline here would hand make_lcmv a
            # projected info together with an unprojected covariance.
            self.data_cov = compute_raw_covariance(_raw_fwd, method=data_cov_method, verbose=False)

        # The cached SourceModels were built from the previous baseline's forward
        # model and covariances; a new baseline invalidates them.
        if getattr(self, "_source_models", None):
            logger.info(
                "New baseline: discarding %d cached source model(s).", len(self._source_models)
            )
        self._source_models = {}
        # The ROI kernels were derived from those models, so they go too.
        self._roi_kernels = {}

        inv_dir = self.subject_dir / "inv"
        _stem = f"sub-{self.subject_id}_ses-{self.session}_task-baseline"
        if self.inv is not None:
            write_inverse_operator(
                fname=inv_dir / f"{_stem}_inv.fif",
                inv=self.inv,
                overwrite=True,
            )
        write_cov(
            # must end in _cov.fif to satisfy MNE's naming check
            fname=inv_dir / f"{_stem}_desc-data_cov.fif",
            cov=self.data_cov,
            overwrite=True,
        )
        # The source space is needed to map atlas labels onto source estimates
        # and cannot be recovered from a beamformer, so persist it too.
        write_source_spaces(
            fname=inv_dir / f"{_stem}_src.fif",
            src=self.src,
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

        Raises
        ------
        RuntimeError
            If the session was connected via :meth:`connect_to_array` — the
            StreamViewer connects over real LSL, which an array-backed
            session never publishes to.
        """
        if isinstance(self.stream, ArrayStream):
            raise RuntimeError(
                "open_stream_viewer() requires a live/mock LSL stream "
                "(connect_to_lsl); it is not available for sessions "
                "connected via connect_to_array()."
            )

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

    def _session_windows(self, *, with_t0: bool = False):
        """The per-window table: onsets, durations, and the marker columns.

        The counterpart of :meth:`_session_meta`, and shared for the same
        reason: :meth:`save` writes this into the JSON and the TSV, and
        :meth:`create_report` renders the column dictionary from it.  Building
        it twice is how the two come to describe different tables.

        Kept out of ``data``: everything there is treated as a modality — by
        ``meta["modalities"]``, ``meta["n_windows"]``, the TSV columns and
        ``TransferProtocol`` alike.
        """
        nf_data = getattr(self, "nf_data", {}) or {}
        # Truncated to the rows already snapshotted into `data`: the acquisition
        # thread is a daemon and keeps appending if the Qt window is closed
        # early, which would otherwise leave onset rows with no features beside
        # them.
        n_rows = max((len(v) for v in nf_data.values()), default=0)

        def _take(name):
            return list(getattr(self, name, []) or [])[:n_rows]

        onsets = _take("window_onsets")
        finite = [t for t in onsets if np.isfinite(t)]
        t0 = float(finite[0]) if finite else None
        windows: dict = {}
        if t0 is not None:
            windows = {
                # Session-relative, so the first window is exactly 0.0.
                # `None` where the stream had not timestamped the window yet.
                "onset": [None if not np.isfinite(t) else float(t - t0) for t in onsets],
                "duration": [float(d) for d in _take("window_durations")],
                # Absolute, for alignment against a stimulus log without
                # having to reapply the offset.
                "onset_lsl": [None if not np.isfinite(t) else float(t) for t in onsets],
            }
        # Independent of onset finiteness: a session whose onsets are all
        # unknown still ran its gate, and losing that record would hide why
        # the subject received nothing.
        conditions = _take("window_conditions")
        if conditions:
            windows["condition"] = conditions
            windows["n_markers"] = [int(v) for v in _take("window_marker_counts")]
            windows["gated"] = [int(v) for v in _take("window_gated")]
        return (windows, t0) if with_t0 else windows

    def _session_meta(self, *, start_iso: Optional[str] = None) -> dict:
        """Describe the session that just ran.

        The single source of truth for the ``meta`` block: :meth:`save` writes it
        into the JSON and :meth:`create_report` renders it into the HTML.  They
        used to be written independently, which is how a report can end up
        claiming something the saved record contradicts.

        Every field is read defensively, so this is also callable after a
        baseline-only session, where most of them do not exist yet.
        """
        nf_data = getattr(self, "nf_data", {}) or {}
        if start_iso is None and hasattr(self, "_session_start_time"):
            start_iso = self._session_start_time.isoformat()
        return {
            "subject_id": self.subject_id,
            "session": self.session,
            "data_type": self.data_type,
            "modalities": list(nf_data.keys()),
            # Instance name -> base modality, so a reader can group the
            # bands of one measure without knowing the separator.
            "modality_bases": {s.name: s.base for s in getattr(self, "_mod_specs", []) or []},
            "sfreq_hz": float(getattr(self, "_sfreq", 0)),
            "winsize_s": float(getattr(self, "winsize", 0)),
            "duration_s": float(getattr(self, "duration", 0)),
            "n_windows": {m: len(v) for m, v in nf_data.items()},
            "artifact_correction": str(self.artifact_correction),
            "artifact_rate": getattr(self, "artifact_rate", None),
            "hop_s": float(getattr(self, "winsize", 0)) / 2.0,
            "zscore_normalize": bool(getattr(self, "_zscore_normalize", False)),
            # Unconditional: when every window is dropped this is the
            # only durable record of why the session came out empty.
            "n_short_windows": int(getattr(self, "n_short_windows", 0)),
            # Same argument as n_short_windows: when the gate never
            # opened, this is the only durable record of why the subject
            # received nothing.
            "gate_conditions": getattr(self, "gate_conditions", None),
            "n_gated_windows": int(getattr(self, "n_gated_windows", 0)),
            "marker_id": dict(getattr(self, "marker_id", {}) or {}),
            "start_time": start_iso,
            "end_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

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
        disconnect: bool = True,
    ) -> dict[str, Path]:
        """Save session outputs and, by default, disconnect the LSL stream.

        All output files share the stem set at the start of
        :meth:`record_main` — see Notes for its shape. Two sessions saved
        without a ``run`` index **do** overwrite each other; pass ``run=`` to
        :meth:`record_main`, as :meth:`run_blocks` does.

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
        Output filenames are built from the BIDS stem
        ``sub-<ID>_ses-<session>_task-neurofeedback``, plus a ``_run-<NN>``
        entity when :meth:`record_main` was given one.

        Disconnects the LSL stream as a side effect, because the stream is not
        normally needed once the session ends.  Pass ``disconnect=False`` to
        keep it — :meth:`run_blocks` does, since a block is not the end of the
        session.  Note the mock player is *stopped* rather than paused, and
        :meth:`connect_to_lsl` builds a new one, so a disconnected mock session
        cannot be resumed by reconnecting the stream alone.
        """
        if disconnect:
            self._teardown_session_streams()
            self._teardown_marker_stream()
            if getattr(self, "_mock_player", None) is not None:
                try:
                    self._mock_player.stop()
                except Exception:
                    pass

        self._ensure_dirs()

        # Includes the `task-` and any `run-` entity, in BIDS order. The
        # fallback has to build the same shape, since it is what a caller who
        # invokes save() without record_main() gets.
        stem = getattr(
            self,
            "_session_stem",
            _build_bids_stem(self.subject_id, self.session, "neurofeedback", None),
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
                "std_ms": round(float(arr.std()), 4),
                "min_ms": round(float(arr.min()), 4),
                "max_ms": round(float(arr.max()), 4),
                "p95_ms": round(float(np.percentile(arr, 95)), 4),
                "n": int(len(arr)),
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
                if hasattr(self, "_session_start_time")
                else None
            )
            payload: dict = {
                "meta": self._session_meta(start_iso=start_iso),
                "data": {m: [_ser(v) for v in vals] for m, vals in self.nf_data.items()},
            }
            # Per-window timing, kept out of `data`: everything there is treated
            # as a modality — by meta["modalities"], meta["n_windows"], the TSV
            # columns and TransferProtocol alike.
            # Truncated to the rows already snapshotted into `data` above: the
            # acquisition thread is a daemon and keeps appending if the Qt window
            # is closed early, which would otherwise leave onset rows with no
            # feature values beside them.
            _windows, _t0 = self._session_windows(with_t0=True)
            if _t0 is not None:
                payload["meta"]["t0_lsl"] = _t0
                # The acquisition stream's own LSL clock. With a remote sender and
                # no clocksync this is the sender's, not this host's.
                payload["meta"]["clock"] = "lsl_stream_clock"
            if _windows:
                payload["windows"] = _windows
            # The markers themselves, as received. Every per-window column above
            # is a lossy projection of this: a marker inside a window that was
            # dropped for being short appears in no column at all.
            _markers = list(getattr(self, "markers", []) or [])
            if _markers:
                payload["markers"] = {
                    "onset_lsl": [_ser(t) for t, _ in _markers],
                    "code": [int(c) for _, c in _markers],
                }

            snr = getattr(self, "snr_data", [])
            if snr:
                payload["data"]["snr_db"] = [_ser(v) for v in snr]
            reward = getattr(self, "reward_data", {})
            for m, vals in reward.items():
                if vals:
                    payload["data"][f"reward_{m}"] = [_ser(v) for v in vals]
            payload["columns"] = describe_nf_columns(
                nf_data=self.nf_data,
                windows=payload.get("windows"),
                reward=reward,
                snr=snr,
                meta=payload["meta"],
            )
            p = self.subject_dir / "beh" / f"{stem}_beh.json"
            with open(p, "w") as fh:
                json.dump(payload, fh, indent=2)
            saved["nf_data"] = p

            if bids_tsv:
                tsv_p = self.subject_dir / "beh" / f"{stem}_beh.tsv"
                # `sidecar=False`: BIDS would put the descriptions in
                # `<stem>_beh.json`, which is the session payload's own filename.
                # They go into that payload instead, under "columns", so the one
                # file serves as both record and sidecar.
                write_nf_beh_tsv(
                    tsv_p,
                    nf_data=self.nf_data,
                    windows=payload.get("windows"),
                    reward=reward,
                    snr=snr,
                    sidecar=False,
                    meta=payload["meta"],
                )
                saved["nf_tsv"] = tsv_p

        # ── Timing / delay statistics ────────────────────────────────────

        if getattr(self, "estimate_delays", False) and (
            acq_delay or artifact_delay or method_delay
        ):
            delays_payload: dict = {}
            if acq_delay and hasattr(self, "acq_delays"):
                delays_payload["acquisition"] = _summarize(self.acq_delays, delay_include_trace)
            if artifact_delay and hasattr(self, "artifact_delays"):
                delays_payload["artifact_correction"] = _summarize(
                    self.artifact_delays, delay_include_trace
                )
            if method_delay and hasattr(self, "method_delays"):
                delays_payload["methods"] = {
                    m: _summarize([float(v) for v in vals], delay_include_trace)
                    for m, vals in self.method_delays.items()
                }
            p = self.subject_dir / "delays" / f"{stem}_delays.json"
            with open(p, "w") as fh:
                json.dump(delays_payload, fh, indent=2)
            saved["delays"] = p

        # ── Raw M/EEG (pre-correction) ───────────────────────────────────

        if raw_data:
            chunks = getattr(self, "_raw_chunks", None)
            if chunks:
                raw_all = np.concatenate(chunks, axis=1)
                raw_nf = RawArray(raw_all, self.rec_info, verbose=False)
                p = self.subject_dir / "eeg" / f"{stem}_eeg.fif"
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

    # ── report helpers ───────────────────────────────────────────────────

    @staticmethod
    def _html_table(rows: list, *, header: Optional[tuple] = None) -> str:
        """Render ``[(label, value), ...]`` as a plain, readable HTML table."""
        style = "border-collapse:collapse;margin:0.5em 0;font-size:0.95em"
        cell = "border:1px solid #ccc;padding:4px 10px"
        out = [f"<table style='{style}'>"]
        if header is not None:
            out.append(
                "<tr>"
                + "".join(f"<th style='{cell};text-align:left'>{h}</th>" for h in header)
                + "</tr>"
            )
        for row in rows:
            out.append(
                "<tr>"
                + "".join(f"<td style='{cell}'>{'' if v is None else v}</td>" for v in row)
                + "</tr>"
            )
        out.append("</table>")
        return "".join(out)

    @staticmethod
    def _report_note(text: str, *, level: str = "info") -> str:
        """A coloured callout.  ``level`` is ``"info"``, ``"warn"`` or ``"bad"``."""
        colours = {
            "info": ("#eef4fb", "#2f6fad"),
            "warn": ("#fdf5e3", "#b8860b"),
            "bad": ("#fdecea", "#c0392b"),
        }
        bg, fg = colours.get(level, colours["info"])
        return (
            f"<div style='background:{bg};border-left:4px solid {fg};"
            f"padding:8px 12px;margin:0.6em 0'>{text}</div>"
        )

    def _report_times(self, n_rows: int) -> tuple:
        """Session-relative window onsets in seconds, and whether they are real.

        Returns ``(times, is_measured)``.  Real onsets come from
        :attr:`window_onsets`; the fallback is the nominal ``index * hop`` grid,
        which the acquisition loop is known to drift away from — so the caption
        has to say which one the reader is looking at.
        """
        onsets = list(getattr(self, "window_onsets", []) or [])[:n_rows]
        finite = [t for t in onsets if np.isfinite(t)]
        if finite:
            # Any finite onset is enough, and individual unknown ones stay NaN
            # rather than discarding the rest — a window whose onset the stream
            # had not timestamped yet is routine at session start, and `save()`
            # keeps the same origin and writes `None` for just that row.
            t0 = finite[0]
            return np.asarray([t - t0 for t in onsets], dtype=float), True
        hop = float(getattr(self, "winsize", 1.0)) / 2.0
        return np.arange(n_rows, dtype=float) * hop, False

    def _report_condition_spans(self, times: np.ndarray) -> dict:
        """Contiguous ``{condition: [(start, stop), ...]}`` runs over the session."""
        conditions = list(getattr(self, "window_conditions", []) or [])[: len(times)]
        if not conditions or len(times) == 0:
            return {}
        # From the configured hop, not from the first two onsets: either of those
        # may be NaN, which would make every span NaN-wide and silently vanish.
        hop = float(getattr(self, "winsize", 1.0)) / 2.0 or 1.0
        spans: dict = {}
        run_start, run_label = times[0], conditions[0]
        for i in range(1, len(conditions) + 1):
            label = conditions[i] if i < len(conditions) else object()
            if label != run_label:
                stop = times[i - 1] + hop
                # A run bounded by a window whose onset is unknown cannot be
                # drawn; skip it rather than emitting a NaN-wide span.
                if run_label is not None and np.isfinite(run_start) and np.isfinite(stop):
                    spans.setdefault(str(run_label), []).append((float(run_start), float(stop)))
                if i < len(conditions):
                    run_start, run_label = times[i], label
        return spans

    def _report_nf_figure(self, modalities: list):
        """Per-modality traces on a real time axis, with conditions and rewards."""
        n_rows = max((len(self.nf_data.get(m, [])) for m in modalities), default=0)
        times, measured = self._report_times(n_rows)
        spans = self._report_condition_spans(times)
        gate = getattr(self, "gate_conditions", None) or []
        rewards = getattr(self, "reward_data", {}) or {}
        protocols = getattr(self, "_protocols", {}) or {}
        combined = getattr(self, "_combined_name", None)

        # Shade the gated conditions when there is a gate; otherwise every named
        # condition, so an ungated but marked session still shows its structure.
        shade = [c for c in spans if not gate or c in gate]
        palette = ["#9ecae1", "#a1d99b", "#fdd0a2", "#dadaeb", "#c7e9c0"]

        fig, axes = plt.subplots(
            len(modalities), 1, figsize=(12, 2.4 * len(modalities)), sharex=True, squeeze=False
        )
        for i, mod in enumerate(modalities):
            ax = axes[i, 0]
            vals = np.asarray(self.nf_data.get(mod, []), dtype=float)
            t = times[: len(vals)]
            for j, condition in enumerate(shade):
                for k, (lo, hi) in enumerate(spans[condition]):
                    ax.axvspan(
                        lo,
                        hi,
                        color=palette[j % len(palette)],
                        alpha=0.35,
                        lw=0,
                        label=condition if (j == k == 0 or k == 0) else None,
                    )
            ax.plot(t, vals, lw=1.4, color="#31424e", zorder=3)

            # Rewarded windows, from the protocol's own record rather than a
            # threshold re-applied here — those can disagree for an adaptive one.
            reward = np.asarray(rewards.get(mod, []), dtype=float)[: len(vals)]
            if reward.size and np.any(reward > 0):
                hit = reward > 0
                ax.plot(
                    t[: len(hit)][hit],
                    vals[: len(hit)][hit],
                    ".",
                    ms=5,
                    color="#c0392b",
                    zorder=4,
                    label=f"rewarded ({int(hit.sum())}/{len(hit)})",
                )
            threshold = getattr(protocols.get(mod), "current_threshold", None)
            if threshold is not None and np.isfinite(threshold):
                # A snapshot, not a trace: an adaptive protocol moved during the
                # run and only its final value survives.
                ax.axhline(
                    float(threshold),
                    ls="--",
                    lw=1.0,
                    color="#c0392b",
                    alpha=0.7,
                    label="final threshold",
                )

            # _label_for already appends the instance label, so adding it again
            # renders "Power (alpha)\n(alpha)".
            ylabel = _label_for(mod)
            unit = _unit_for(mod)
            ax.set_ylabel(f"{ylabel}\n{unit}" if unit else ylabel, fontsize=8)
            if mod == combined:
                ax.set_facecolor("#fbfbfd")
            ax.grid(True, alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                seen: dict = {}
                for h, lbl in zip(handles, labels):
                    seen.setdefault(lbl, h)
                ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right", ncol=3)
        axes[-1, 0].set_xlabel("Time from first window (s)" if measured else "Nominal time (s)")
        fig.tight_layout()
        return fig, measured

    def _report_marker_figure(self):
        """Marker raster on the same session-relative axis as the traces."""
        markers = list(getattr(self, "markers", []) or [])
        if not markers:
            return None
        n_rows = max((len(v) for v in getattr(self, "nf_data", {}).values()), default=0)
        onsets = [t for t in (getattr(self, "window_onsets", []) or [])[:n_rows] if np.isfinite(t)]
        if not onsets:
            # The traces fell back to the nominal grid, which has no relation to
            # the marker clock. Two axes labelled the same way but anchored
            # differently is worse than no raster.
            return None
        t0 = onsets[0]
        inverse = {v: k for k, v in (getattr(self, "marker_id", {}) or {}).items()}
        codes = sorted({int(c) for _, c in markers})

        fig, ax = plt.subplots(figsize=(12, 1.0 + 0.35 * len(codes)))
        for row, code in enumerate(codes):
            times = [float(t) - t0 for t, c in markers if int(c) == code]
            ax.plot(times, [row] * len(times), "|", ms=14, mew=1.6, color="#2f6fad")
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels([f"{inverse.get(c, c)} ({c})" for c in codes], fontsize=8)
        ax.set_xlabel("Time from first window (s)")
        ax.set_ylim(-0.6, len(codes) - 0.4)
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        return fig

    def _report_quality_figure(self):
        """SNR over the session, when it was tracked."""
        snr = list(getattr(self, "snr_data", []) or [])
        if not snr:
            return None
        times, measured = self._report_times(len(snr))
        fig, ax = plt.subplots(figsize=(12, 2.4))
        ax.plot(times[: len(snr)], snr, lw=1.2, color="#2f6fad")
        ax.set_ylabel("SNR (dB)", fontsize=9)
        ax.set_xlabel("Time from first window (s)" if measured else "Nominal time (s)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig

    def _report_source_rows(self, modalities: list) -> list:
        """One row per source-space instance: ROIs, pairs, band, inverse method."""
        params = getattr(self, "mod_params_dict", {}) or {}
        rows = []
        for mod in modalities:
            base = split_modality(mod)[0]
            if not base.startswith("source"):
                continue
            entry = params.get(mod, {}) or {}
            rois = entry.get("rois")
            if isinstance(rois, (list, tuple)):
                names = []
                for roi in rois:
                    names.extend(roi.keys()) if isinstance(roi, dict) else names.append(str(roi))
                rois = ", ".join(names)
            pairs = entry.get("pairs") or ([entry["pair"]] if entry.get("pair") else None)
            frange = entry.get("frange")
            rows.append(
                (
                    mod,
                    entry.get("atlas") or getattr(self, "source_atlas", None),
                    entry.get("inverse_method") or entry.get("method"),
                    f"{frange[0]}–{frange[1]} Hz" if frange else None,
                    rois,
                    " ; ".join("↔".join(map(str, p)) for p in pairs) if pairs else None,
                )
            )
        return rows

    def create_report(
        self,
        overwrite: bool = True,
        include_psd: bool = True,
        include_nf_signal: bool = True,
        open_browser: bool = False,
        *,
        run: Optional[int] = None,
        include_markers: bool = True,
        include_quality: bool = True,
        include_source: bool = True,
        include_columns: bool = True,
    ) -> Path:
        """Generate a self-contained HTML report for the session.

        The report is organised into sections: a session summary, the
        neurofeedback traces, markers and gating, data quality, the source-space
        configuration, the baseline recording, and the column dictionary.
        Sections whose data the session does not have are omitted, so a plain
        sensor-space run produces a shorter report rather than empty panels.

        The summary is rendered from the same ``meta`` block :meth:`save` writes
        into the JSON, so the two cannot disagree.

        Parameters
        ----------
        overwrite : bool, default True
            Overwrite an existing report file with the same name.
        include_psd : bool, default True
            Add a baseline power-spectral-density plot (1–40 Hz).
        include_nf_signal : bool, default True
            Add the feature time-series, with task-condition shading and the
            windows that were rewarded.
        open_browser : bool, default False
            Open the saved report in the default web browser.
        run : int | None
            BIDS ``run-`` entity.  Defaults to the run :meth:`record_main`
            recorded, so the blocks of a :meth:`run_blocks` session each get
            their own report instead of overwriting one another.
        include_markers : bool, default True
            Add the marker raster and the gating summary, if a marker stream was
            attached.
        include_quality : bool, default True
            Add SNR, artifact rate, dropped windows, and the latency summary.
        include_source : bool, default True
            Add the ROIs, atlas and inverse method of each source-space
            modality.
        include_columns : bool, default True
            Add the data dictionary describing every saved column.

        Returns
        -------
        report_path : pathlib.Path
            Full path of the saved HTML report file.

        Raises
        ------
        RuntimeError
            If :meth:`record_baseline` has not been called yet.
        """
        if getattr(self, "raw_baseline", None) is None:
            raise RuntimeError(
                "create_report() needs a baseline recording. Call record_baseline() first."
            )
        self._ensure_dirs()

        # Prefer the parsed specs so the report can never disagree with what
        # the loop actually ran; fall back for a report built without one.
        modalities = [s.name for s in getattr(self, "_mod_specs", []) or []] or [
            m for m in (getattr(self, "nf_data", {}) or {})
        ]
        combined = getattr(self, "_combined_name", None)
        if (
            combined
            and combined in (getattr(self, "nf_data", {}) or {})
            and combined not in modalities
        ):
            # The trace the subject actually saw when a combiner is used; it is
            # not in `_mod_specs`, and leaving it out omits the feedback itself.
            modalities = modalities + [combined]
        meta = self._session_meta()
        title = f"Neurofeedback — sub-{self.subject_id} ses-{self.session}"
        report = Report(title=title, image_format="png")

        # ── Session summary ──────────────────────────────────────────────
        gate = meta["gate_conditions"]
        n_total = int(getattr(self, "n_total_windows", 0))
        # `n_gated_windows` counts the windows that were gated *out* -- the ones
        # where feedback was withheld. Reporting it as "in gate" inverts the
        # headline number, so the split is named once, here.
        n_withheld = int(meta["n_gated_windows"])
        n_delivered = max(0, n_total - n_withheld) if n_total else 0
        summary = [
            ("Subject", f"sub-{meta['subject_id']}"),
            ("Session", f"ses-{meta['session']}"),
            ("Data type", meta["data_type"]),
            ("Modalities", ", ".join(meta["modalities"]) or "—"),
            ("Sampling rate", f"{meta['sfreq_hz']:.1f} Hz" if meta["sfreq_hz"] else "—"),
            (
                "Window / hop",
                f"{meta['winsize_s']:.2f} s / {meta['hop_s']:.2f} s" if meta["winsize_s"] else "—",
            ),
            ("Duration", f"{meta['duration_s']:.1f} s" if meta["duration_s"] else "—"),
            ("Windows analysed", n_total or "—"),
            ("Windows dropped (short)", meta["n_short_windows"]),
            ("Artifact correction", meta["artifact_correction"]),
            (
                "Artifact rate",
                f"{meta['artifact_rate']:.1%}" if meta["artifact_rate"] is not None else "—",
            ),
            ("Z-score normalised", "yes" if meta["zscore_normalize"] else "no"),
            ("Gate conditions", ", ".join(gate) if gate else "none (ungated)"),
            ("Windows with feedback", f"{n_delivered} of {n_total}" if gate and n_total else "—"),
            ("Windows withheld", n_withheld if gate else "—"),
            ("Started", meta["start_time"] or "—"),
            ("Ended", meta["end_time"]),
        ]
        report.add_html(
            self._html_table(summary), title="Session summary", section="Summary", tags=("summary",)
        )

        # ── Neurofeedback traces ─────────────────────────────────────────
        if include_nf_signal and (getattr(self, "nf_data", None) or None) and modalities:
            fig_nf, measured = self._report_nf_figure(modalities)
            caption = (
                "Time axis from the recorded window onsets."
                if measured
                else "No window onsets were recorded, so the time axis is the nominal "
                "index × hop grid and drifts from real time."
            )
            if gate:
                caption += f" Shaded spans are the gated condition(s): {', '.join(gate)}."
            report.add_figure(
                fig=fig_nf,
                title="Feature time-series",
                caption=caption,
                section="Neurofeedback",
                tags=("neurofeedback",),
            )
            plt.close(fig_nf)

        # ── Markers and gating ───────────────────────────────────────────
        if include_markers and (getattr(self, "markers", None) or meta["gate_conditions"]):
            notes = []
            if gate and n_total and n_delivered == 0:
                notes.append(
                    self._report_note(
                        f"<b>No feedback was delivered.</b> Every one of the {n_total} "
                        f"windows fell outside <code>gate_conditions={gate}</code>. Check "
                        "that the paradigm publishes the expected codes, and that both "
                        "streams are clock-synchronised if they come from different hosts.",
                        level="bad",
                    )
                )
            elif gate and n_total and n_withheld == 0:
                notes.append(
                    self._report_note(
                        "Every window was inside the gate, so gating never suppressed "
                        "feedback. That is expected only if the task ran continuously.",
                        level="warn",
                    )
                )
            counts: dict = {}
            for condition in getattr(self, "window_conditions", []) or []:
                counts[str(condition)] = counts.get(str(condition), 0) + 1
            rows = [
                (
                    name,
                    meta["marker_id"].get(name, "—"),
                    n,
                    f"{n / max(1, sum(counts.values())):.1%}",
                    "yes" if (not gate or name in gate) else "no",
                )
                for name, n in sorted(counts.items(), key=lambda kv: -kv[1])
            ]
            html = "".join(notes)
            if rows:
                html += self._html_table(
                    rows, header=("Condition", "Code", "Windows", "Share", "In gate")
                )
            html += self._html_table(
                [
                    ("Markers received", len(getattr(self, "markers", []) or [])),
                    ("Marker codes", ", ".join(f"{k}={v}" for k, v in meta["marker_id"].items())),
                    ("Windows with feedback", f"{n_delivered} of {n_total or '?'}"),
                    ("Windows withheld by the gate", n_withheld),
                ]
            )
            report.add_html(html, title="Gating", section="Markers", tags=("markers",))
            fig_markers = self._report_marker_figure()
            if fig_markers is not None:
                report.add_figure(
                    fig=fig_markers,
                    title="Marker raster",
                    caption="Every marker as received, on the analysis time axis.",
                    section="Markers",
                    tags=("markers",),
                )
                plt.close(fig_markers)

        # ── Data quality ─────────────────────────────────────────────────
        if include_quality:
            rows = [
                ("Windows analysed", n_total or "—"),
                (
                    "Artifact-flagged windows",
                    f"{getattr(self, 'n_artifact_windows', 0)}"
                    + (f" ({meta['artifact_rate']:.1%})" if meta["artifact_rate"] else ""),
                ),
                ("Windows dropped (short)", meta["n_short_windows"]),
            ]
            notes = []
            if meta["n_short_windows"]:
                notes.append(
                    self._report_note(
                        f"{meta['n_short_windows']} window(s) were discarded for arriving "
                        "short. A non-integral <code>winsize × sfreq</code> drops every "
                        "window; a few scattered drops are normal.",
                        level="warn" if n_total else "bad",
                    )
                )
            hop_ms = meta["hop_s"] * 1e3
            delays = getattr(self, "method_delays", {}) or {}
            for name, vals in delays.items():
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=float) * 1e3
                rows.append(
                    (
                        f"Latency — {name}",
                        f"mean {arr.mean():.1f} ms, p95 {np.percentile(arr, 95):.1f} ms",
                    )
                )
            if delays and hop_ms:
                # p95 of the summed per-window cost. Summing each method's own
                # p95 assumes they all peak on the same window, which would
                # raise the alarm on runs that are comfortably inside the hop.
                traces = [np.asarray(v, dtype=float) for v in delays.values() if v]
                width = min(len(t) for t in traces)
                total_p95 = float(
                    np.percentile(np.sum([t[:width] for t in traces], axis=0) * 1e3, 95)
                )
                rows.append(("Latency — total p95", f"{total_p95:.1f} ms of a {hop_ms:.0f} ms hop"))
                if total_p95 > hop_ms:
                    notes.append(
                        self._report_note(
                            f"Feature computation (p95 {total_p95:.0f} ms) exceeds the "
                            f"{hop_ms:.0f} ms hop, so the loop cannot keep up and feedback "
                            "lags behind the subject.",
                            level="bad",
                        )
                    )
            report.add_html(
                "".join(notes) + self._html_table(rows),
                title="Data quality",
                section="Quality",
                tags=("quality",),
            )
            fig_snr = self._report_quality_figure()
            if fig_snr is not None:
                report.add_figure(fig=fig_snr, title="SNR", section="Quality", tags=("quality",))
                plt.close(fig_snr)

        # ── Source space ─────────────────────────────────────────────────
        # Tables, not renderings: a 3-D view needs pyvista/vtk, which is an
        # optional extra, and would make the whole report fail to build without it.
        if include_source:
            rows = self._report_source_rows(modalities)
            if rows:
                report.add_html(
                    self._html_table(
                        rows,
                        header=("Modality", "Atlas", "Inverse", "Band", "ROIs", "Pairs"),
                    ),
                    title="Source-space configuration",
                    section="Source space",
                    tags=("source",),
                )

        # ── Baseline ─────────────────────────────────────────────────────
        report.add_raw(
            self.raw_baseline,
            title="Baseline recording",
            psd=False,
            butterfly=False,
            tags=("baseline",),
        )
        if include_psd:
            spectrum = self.raw_baseline.compute_psd(fmax=40.0)
            # One axes per channel type, counted from the spectrum rather than
            # from the raw: compute_psd drops ECG, EMG, misc and bad channels,
            # so deriving the count from the raw over-counts and the plot then
            # refuses the axes list. MEG legitimately needs two (mag + grad).
            n_axes = max(1, len(set(spectrum.info.get_channel_types(unique=True))))
            # constrained_layout, not tight_layout: MNE's PSD plot adds an inset
            # axes that tight_layout cannot place, and warns about it every time.
            fig_psd, ax_psd = plt.subplots(
                n_axes, 1, figsize=(10, 4 * n_axes), layout="constrained"
            )
            axes = list(np.atleast_1d(ax_psd))
            spectrum.plot(axes=axes, show=False)
            # On the figure, not on the first axes: MNE titles each panel with
            # its channel type, and overwriting the first leaves an MEG report
            # with an unlabelled magnetometer panel beside a labelled gradiometer
            # one. The upper bound is the only one compute_psd was given.
            fig_psd.suptitle("Baseline PSD (up to 40 Hz)")
            report.add_figure(
                fig=fig_psd, title="Baseline PSD", section="Baseline", tags=("baseline",)
            )
            plt.close(fig_psd)

        # ── Column dictionary ────────────────────────────────────────────
        if include_columns and (getattr(self, "nf_data", None) or None):
            columns = describe_nf_columns(
                nf_data=self.nf_data,
                # Without `windows`, onset/duration/condition/n_markers/gated are
                # all absent -- the six columns a reader most needs explained,
                # and the ones save() does describe in the same sidecar.
                windows=self._session_windows(),
                reward=getattr(self, "reward_data", {}) or {},
                snr=getattr(self, "snr_data", []) or [],
                meta=meta,
            )
            # The sidecar also carries scalar session metadata (sfreq_hz and
            # friends) beside the per-column dicts; only the dicts are columns.
            rows = [
                (
                    name,
                    entry.get("LongName", ""),
                    entry.get("Units", ""),
                    entry.get("Description", ""),
                )
                for name, entry in columns.items()
                if isinstance(entry, dict)
            ]
            report.add_html(
                self._html_table(rows, header=("Column", "Long name", "Units", "Description")),
                title="Saved columns",
                section="Data dictionary",
                tags=("columns",),
            )

        # BIDS order puts `run-` after `task-`, and the stem record_main built
        # already has it — without one, block 2 overwrites block 1's report.
        if run is not None:
            # Zero-padded, matching the stem record_main builds, so a report
            # asked for by number sits beside the beh files of the same run.
            stem = _build_bids_stem(
                self.subject_id, self.session, "neurofeedback", f"{int(run):02d}"
            )
            # `run_blocks` rebinds nf_data, the onsets and the window counts on
            # every block, so this object only ever holds the *last* one. Naming
            # an earlier run would write that block's data under another block's
            # filename, which is worse than refusing.
            current = getattr(self, "_session_stem", None)
            if current is not None and current != stem:
                raise ValueError(
                    f"run={run} does not match the run this session last recorded "
                    f"({current!r}). The stream holds only the most recent run's data, so "
                    "a report for an earlier one would carry the wrong traces. Call "
                    "create_report() with no run= for the run just recorded."
                )
        else:
            stem = getattr(
                self,
                "_session_stem",
                _build_bids_stem(self.subject_id, self.session, "neurofeedback", None),
            )
        report_path = self.subject_dir / "reports" / f"{stem}_report.html"
        report.save(report_path, overwrite=overwrite, open_browser=open_browser)
        return report_path

    def __del__(self) -> None:
        """Release the thread pool, stop the mock player, disconnect the stream."""
        if getattr(self, "executor", None) is not None:
            try:
                # `wait=False`: a record_main that raised may have left work
                # queued, and blocking here would hang interpreter shutdown.
                self.executor.shutdown(wait=False)
            except Exception:
                pass
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
        if getattr(self, "marker_stream", None) is not None:
            try:
                self.marker_stream.disconnect()
            except Exception:
                pass
