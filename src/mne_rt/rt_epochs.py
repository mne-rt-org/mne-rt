"""Event-triggered real-time epoch accumulation.

Thin orchestration layer on top of :class:`mne_lsl.stream.EpochsStream`,
which handles all buffering, baseline correction, and rejection internally.

Typical workflow
----------------
::

    rt = RTEpochs(
        event_id={"target": 1, "standard": 2},
        event_channels="STI 014",
        tmin=-0.2, tmax=0.8,
    )
    rt.connect_to_lsl()
    rt.run(n_trials=80, show_erp=True)

Events may instead arrive on a separate LSL marker stream -- what a PsychoPy
paradigm publishes -- in which case ``event_channels`` names a channel of
*that* stream::

    rt = RTEpochs(event_id={"go": 1}, event_channels="markers")
    rt.connect_to_lsl(stream_name="ANT", event_stream_name="psychopy_markers")

Classes
-------
RTEpochs
    Event-triggered epoch accumulator backed by mne_lsl.EpochsStream.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Callable, Optional, Sequence, Union

import mne
import numpy as np

try:
    from mne_lsl.player import PlayerLSL
    from mne_lsl.stream import EpochsStream, StreamLSL

    _mne_lsl_available = True
except ImportError:
    _mne_lsl_available = False

from mne_rt._logging import logger, set_log_level


class RTEpochs:
    """Event-triggered epoch accumulator backed by :class:`mne_lsl.stream.EpochsStream`.

    Connects a :class:`~mne_lsl.stream.StreamLSL` to an
    :class:`~mne_lsl.stream.EpochsStream`, polls for new epochs, and
    optionally drives an :class:`~mne_rt.viz.TopoPlot` that redraws after
    every new trial.

    Parameters
    ----------
    event_id : dict[str, int]
        Condition label → marker integer, e.g. ``{"target": 1, "standard": 2}``.
    event_channels : str or list of str
        Channel(s) carrying the event codes.  Without an event stream these
        must be ``"stim"``-type channels of the M/EEG stream itself (e.g.
        ``"STI 014"``).  When :meth:`connect_to_lsl` is given an event stream
        they name channels of *that* stream instead -- see its Notes.
    tmin : float, default -0.2
        Epoch start in seconds relative to the event.
    tmax : float, default 0.8
        Epoch end in seconds relative to the event.
    baseline : tuple or None, default (None, 0)
        Baseline interval passed to :class:`~mne_lsl.stream.EpochsStream`.
        ``None`` disables correction.
    picks : str or list or None, default None
        Channel selection forwarded to :class:`~mne_lsl.stream.EpochsStream`.
    reject : dict or None, default None
        Peak-to-peak rejection thresholds, e.g. ``{"eeg": 150e-6}``.
    bufsize : int, default 200
        Number of epochs to keep in the :class:`~mne_lsl.stream.EpochsStream`
        internal ring buffer.
    on_trial : callable or None, default None
        Optional callback fired after every accepted epoch::

            def on_trial(n_accepted, data, event_code, condition):
                ...

        ``data`` is the single accepted epoch, ``(n_channels, n_times)``;
        ``event_code`` is its integer marker and ``condition`` the matching
        ``event_id`` label.
    verbose : bool or str or None, default None

    Attributes
    ----------
    epochs_stream_ : mne_lsl.stream.EpochsStream or None
        The underlying :class:`~mne_lsl.stream.EpochsStream` after
        :meth:`connect_to_lsl` has been called.
    event_stream_ : mne_lsl.stream.StreamLSL or None
        The separate marker stream, when :meth:`connect_to_lsl` was asked for
        one.  ``None`` when the events come from a stim channel of the M/EEG
        stream.
    n_accepted_ : int
        Running count of accepted epochs since :meth:`run` started.

    See Also
    --------
    mne_rt.viz.TopoPlot : Live scalp-layout ERP display driven by this class.
    mne_rt.viz.EpochPlot : Scrolling raw viewer with trigger/epoch overlays.
    mne_rt.RTStream : Continuous sliding-window stream processor.

    Examples
    --------
    >>> rt = RTEpochs(
    ...     event_id={"auditory": 1, "visual": 2},
    ...     event_channels="STI 014",
    ...     tmin=-0.2, tmax=0.5,
    ... )
    >>> rt.connect_to_lsl(mock_lsl=True, fname="sample_raw.fif")
    >>> rt.run(n_trials=20, show_erp=True)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        event_id: dict[str, int],
        event_channels: Union[str, list[str]],
        tmin: float = -0.2,
        tmax: float = 0.8,
        baseline: Optional[tuple] = (None, 0),
        picks: Optional[Union[str, list]] = None,
        reject: Optional[dict] = None,
        bufsize: int = 200,
        on_trial: Optional[Callable] = None,
        verbose: Union[bool, str, None] = None,
    ) -> None:
        set_log_level(verbose)

        if not _mne_lsl_available:
            raise ImportError("mne-lsl is required.  Install with: pip install mne-lsl")

        self.event_id = event_id
        self.event_channels = event_channels
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.picks = picks
        self.reject = reject
        self.bufsize = bufsize
        self.on_trial = on_trial

        self._stream: Optional[StreamLSL] = None
        self._player: Optional[PlayerLSL] = None
        self.epochs_stream_: Optional[EpochsStream] = None
        self.event_stream_: Optional[StreamLSL] = None
        self.n_accepted_: int = 0
        self._stop_event = threading.Event()
        self._connected = False

        # Populated by run() — persists for get_epochs/get_evoked/save
        self._buf_: Optional[np.ndarray] = None  # (n_trials, n_ch, n_t)
        self._cond_list_: list[str] = []
        self._code_list_: list[int] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect_to_lsl(
        self,
        stream_name: Optional[str] = None,
        mock_lsl: bool = False,
        fname: Optional[str] = None,
        timeout: float = 10.0,
        event_stream_name: Optional[str] = None,
        event_source_id: Optional[str] = None,
        event_stream_bufsize: int = 200,
        processing_flags: Optional[Union[str, Sequence[str]]] = None,
        verbose: Union[bool, str, None] = None,
    ) -> "RTEpochs":
        """Connect to an LSL stream and set up the EpochsStream.

        Parameters
        ----------
        stream_name : str or None
            LSL stream name.  ``None`` picks the first available stream.
        mock_lsl : bool
            Replay ``fname`` via :class:`~mne_lsl.player.PlayerLSL`.
        fname : str or None
            Path to a ``.fif`` file (required when ``mock_lsl=True``).
        timeout : float
            LSL connection timeout in seconds.
        event_stream_name : str or None, default None
            Name of a separate LSL stream carrying the event codes -- the
            marker outlet of a PsychoPy paradigm, for instance.  Leaving both
            this and ``event_source_id`` as ``None`` keeps the events on a
            stim channel of the M/EEG stream itself, which is the behaviour of
            earlier versions.
        event_source_id : str or None, default None
            Source ID of the event stream.  May be given instead of, or
            together with, ``event_stream_name``; the two must together
            identify exactly one stream.
        event_stream_bufsize : int, default 200
            Event-stream buffer size **in samples**.  A marker outlet is
            irregularly sampled, so a duration would carry no meaning.
        processing_flags : str or sequence of str or None, default None
            Forwarded to :meth:`mne_lsl.stream.StreamLSL.connect` for every
            stream opened here.  ``None`` selects ``"all"`` when an event
            stream is requested, and no flags otherwise -- see Notes.
        verbose : bool or str or None

        Returns
        -------
        self : RTEpochs

        Notes
        -----
        **Clock synchronisation.**  Two LSL streams share a time base only once
        their clocks are synchronised, so when an event stream is requested
        both streams are connected with ``processing_flags="all"``
        (``clocksync``, ``dejitter``, ``monotize``).  Without it, markers
        published from a second machine sit at an arbitrary offset from the
        M/EEG data and every epoch is cut in the wrong place, silently.  Pass
        ``processing_flags`` explicitly to override.

        **Publishing the markers.**  The outlet must satisfy three constraints,
        each of which otherwise fails confusingly or not at all:

        - it must be **numeric**.  mne-lsl refuses string streams outright, and
          ``channel_format="string"`` is what most PsychoPy marker examples
          use -- publish ``"int32"`` instead;
        - it should **label its channel**, since an outlet that does not is
          exposed as ``"0"``, ``"1"``, ... and ``event_channels`` has to match;
        - the codes must be positive integers no greater than 32767, matching
          the values of ``event_id``.

        A minimal publisher, which doubles as the reference for the paradigm
        side::

            from mne_lsl.lsl import StreamInfo, StreamOutlet

            sinfo = StreamInfo(
                "psychopy_markers", "Markers", 1, 0.0, "int32", "psychopy_uid"
            )
            sinfo.set_channel_names(["markers"])
            outlet = StreamOutlet(sinfo)
            outlet.push_sample([1])          # a "go" trial

        Examples
        --------
        Events from a stim channel of the M/EEG stream itself::

            rt = RTEpochs(event_id={"target": 1}, event_channels="STI 014")
            rt.connect_to_lsl(mock_lsl=True, fname="sample_raw.fif")

        Events from a PsychoPy marker outlet::

            rt = RTEpochs(event_id={"go": 1, "stop": 2}, event_channels="markers")
            rt.connect_to_lsl(
                stream_name="ANT", event_stream_name="psychopy_markers"
            )
        """
        if verbose is not None:
            set_log_level(verbose)

        # Clear handles from any previous connect: without this a second call
        # made *without* an event stream would still hand the disconnected one
        # from the first call to EpochsStream, which refuses it.
        self.epochs_stream_ = None
        self.event_stream_ = None
        self._connected = False

        if mock_lsl:
            if fname is None:
                raise ValueError("fname is required when mock_lsl=True.")
            logger.info("RTEpochs: starting mock PlayerLSL from %s", fname)
            self._player = PlayerLSL(fname, name="mne_rt_mock", chunk_size=16).start()
            time.sleep(1.5)
            stream_name = "mne_rt_mock"

        want_event_stream = event_stream_name is not None or event_source_id is not None
        if want_event_stream:
            if int(event_stream_bufsize) != event_stream_bufsize or event_stream_bufsize <= 0:
                raise ValueError(
                    "event_stream_bufsize is a number of samples and must be a "
                    f"positive integer; got {event_stream_bufsize!r}."
                )
            event_stream_bufsize = int(event_stream_bufsize)
            if processing_flags is None:
                # Two streams share a time base only once their clocks are synced.
                processing_flags = "all"

        logger.info("RTEpochs: connecting StreamLSL …")
        self._stream = StreamLSL(bufsize=4.0, name=stream_name)
        try:
            self._stream.connect(
                acquisition_delay=0.005,
                processing_flags=processing_flags,
                timeout=timeout,
            )
            logger.info(
                "RTEpochs: stream connected — %d ch @ %.0f Hz",
                self._stream.info["nchan"],
                self._stream.info["sfreq"],
            )

            if want_event_stream:
                self._connect_event_stream(
                    name=event_stream_name,
                    source_id=event_source_id,
                    bufsize=event_stream_bufsize,
                    processing_flags=processing_flags,
                    timeout=timeout,
                )

            logger.info("RTEpochs: setting up EpochsStream …")
            self.epochs_stream_ = EpochsStream(
                stream=self._stream,
                bufsize=self.bufsize,
                event_id=self.event_id,
                event_channels=self.event_channels,
                event_stream=self.event_stream_,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=self.baseline,
                picks=self.picks,
                reject=self.reject,
            ).connect(acquisition_delay=0.005)
        except BaseException:
            # Anything raised past this point leaves a running player and one
            # or two connected streams behind, with ``_connected`` still False
            # so the caller has no handle to clean them up with.
            self.disconnect()
            raise

        self._connected = True
        logger.info("RTEpochs: EpochsStream connected.")
        return self

    def _connect_event_stream(
        self,
        *,
        name: Optional[str],
        source_id: Optional[str],
        bufsize: int,
        processing_flags: Optional[Union[str, Sequence[str]]],
        timeout: float,
    ) -> None:
        """Connect the separate LSL stream carrying the event codes.

        Kept apart from :meth:`connect_to_lsl` only so its two failure modes
        can be reported against the *event* stream; mne-lsl raises for both,
        but from a context that does not say which of the two streams is at
        fault.
        """
        logger.info(
            "RTEpochs: connecting event StreamLSL (name=%r, source_id=%r) …",
            name,
            source_id,
        )
        # Held from construction onwards, so that a failure anywhere below is
        # still reachable by disconnect(). StreamLSL.connect() can raise with an
        # inlet already open and an acquisition thread already running.
        stream = StreamLSL(bufsize=bufsize, name=name, source_id=source_id)
        self.event_stream_ = stream
        try:
            stream.connect(
                acquisition_delay=0.005,
                processing_flags=processing_flags,
                timeout=timeout,
            )
        except RuntimeError as exc:
            if "string LSL streams" not in str(exc):
                raise
            raise RuntimeError(
                "The event stream publishes strings, which mne-lsl cannot read. "
                "Publish the marker codes from a numeric outlet instead, e.g. "
                "channel_format='int32' — note that PsychoPy's marker examples "
                "commonly default to channel_format='string'."
            ) from exc
        available = list(stream.info["ch_names"])
        wanted = (
            [self.event_channels]
            if isinstance(self.event_channels, str)
            else list(self.event_channels)
        )
        missing = [ch for ch in wanted if ch not in available]
        if missing:
            raise ValueError(
                f"Event channel(s) {missing} are not in the event stream, which "
                f"publishes {available}. An LSL outlet that does not label its "
                "channels is exposed by mne-lsl as '0', '1', … — either label "
                "the channel in the publisher, or pass the name it actually has."
            )
        logger.info(
            "RTEpochs: event stream connected — %d ch @ %.0f Hz, using %s",
            stream.info["nchan"],
            stream.info["sfreq"],
            wanted,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_trials: int = 100,
        show_erp: bool = False,
        erp_update_every: int = 1,
        poll_interval: float = 0.05,
        verbose: Union[bool, str, None] = None,
    ) -> "RTEpochs":
        """Run the epoch accumulation loop.

        Polls :attr:`~mne_lsl.stream.EpochsStream.n_new_epochs` and
        retrieves data in batches.  Blocks until ``n_trials`` accepted epochs
        have been collected or :meth:`stop` is called.

        Parameters
        ----------
        n_trials : int, default 100
            Stop after this many accepted epochs.
        show_erp : bool, default False
            Open an :class:`~mne_rt.viz.TopoPlot` that redraws every
            ``erp_update_every`` accepted epochs.
        erp_update_every : int, default 1
            ERP redraw cadence in number of accepted epochs.
        poll_interval : float, default 0.05
            Seconds to sleep between polling :attr:`n_new_epochs`.
        verbose : bool or str or None

        Returns
        -------
        self : RTEpochs
        """
        if verbose is not None:
            set_log_level(verbose)
        if not self._connected:
            raise RuntimeError("Call connect_to_lsl() before run().")

        es = self.epochs_stream_
        erp_plot = None
        if show_erp:
            from mne_rt.viz.topo_plot import TopoPlot

            erp_plot = TopoPlot(
                ch_names=list(es.info["ch_names"]),
                sfreq=es.info["sfreq"],
                tmin=self.tmin,
                tmax=self.tmax,
                event_id=self.event_id,
                info=es.info,  # pass real Info for accurate layout
                baseline=self.baseline,
            )
            erp_plot.show()

        inv_event = {v: k for k, v in self.event_id.items()}

        # Pre-allocate epoch buffer — avoids O(N²) np.stack per trial
        n_ch = es.info["nchan"]
        # es.times is authoritative: mne-lsl builds ceil((tmax - tmin) * sfreq)
        # samples with endpoint=False, one fewer than round(...) + 1. Sizing the
        # buffer ourselves left a trailing all-zero sample on every epoch.
        n_times = es.times.size
        self._buf_ = np.zeros((n_trials, n_ch, n_times), dtype=np.float32)
        self._cond_list_ = []
        self._code_list_ = []

        self._stop_event.clear()
        self.n_accepted_ = 0

        logger.info("RTEpochs: running — target %d trials …", n_trials)

        while self.n_accepted_ < n_trials and not self._stop_event.is_set():
            n_new = self.epochs_stream_.n_new_epochs
            if n_new == 0:
                time.sleep(poll_interval)
                continue

            # Retrieve all new epochs at once — shape (n_new, n_ch, n_times)
            data = self.epochs_stream_.get_data(n_epochs=n_new)
            events = self.epochs_stream_.events[-n_new:]

            for i in range(data.shape[0]):
                if self.n_accepted_ >= n_trials:
                    break
                code = int(events[i]) if events.ndim == 1 else int(events[i, 2])
                condition = inv_event.get(code, str(code))

                # Write into pre-allocated buffer (O(1) copy)
                ep = data[i]
                t = min(ep.shape[-1], n_times)
                self._buf_[self.n_accepted_, :, :t] = ep[:, :t]
                self._cond_list_.append(condition)
                self._code_list_.append(code)
                self.n_accepted_ += 1

                # on_trial now receives event_code + condition directly
                if self.on_trial is not None:
                    self.on_trial(
                        self.n_accepted_,
                        self._buf_[self.n_accepted_ - 1],  # view — no copy
                        code,
                        condition,
                    )

                if erp_plot is not None and self.n_accepted_ % erp_update_every == 0:
                    # Pass a view of the filled portion — no copy
                    erp_plot.update(self._buf_[: self.n_accepted_], list(self._cond_list_))

                logger.debug("RTEpochs: accepted %d (%s)", self.n_accepted_, condition)

        logger.info("RTEpochs: finished — %d epochs accepted.", self.n_accepted_)
        return self

    def stop(self) -> None:
        """Signal the run loop to stop after the current poll."""
        self._stop_event.set()

    def disconnect(self) -> None:
        """Disconnect EpochsStream, both StreamLSLs, and stop any mock player."""
        # The EpochsStream registers itself on the stream(s) it reads and
        # unregisters on disconnect, so it has to be torn down first.
        if self.epochs_stream_ is not None:
            try:
                self.epochs_stream_.disconnect()
            except Exception:
                pass
        for stream in (self.event_stream_, self._stream):
            if stream is not None:
                try:
                    stream.disconnect()
                except Exception:
                    pass
        if self._player is not None:
            try:
                self._player.stop()
            except Exception:
                pass
        self._connected = False
        logger.info("RTEpochs: disconnected.")

    # ------------------------------------------------------------------
    # Offline analysis helpers
    # ------------------------------------------------------------------

    def get_epochs(self) -> "mne.EpochsArray":
        """Return accumulated epochs as :class:`mne.EpochsArray`.

        Can be called mid-run or after :meth:`run` completes.  The returned
        object contains all epochs accepted so far and uses the real
        :class:`mne.Info` from the underlying stream (including channel
        positions and digitisation points).

        Returns
        -------
        epochs : mne.EpochsArray
            Shape ``(n_accepted, n_channels, n_times)``.

        Raises
        ------
        RuntimeError
            If called before :meth:`connect_to_lsl`.

        Examples
        --------
        >>> rt.run(n_trials=50, show_erp=True)
        >>> epochs = rt.get_epochs()
        >>> epochs.plot_image()
        """
        import mne

        if self.epochs_stream_ is None or self._buf_ is None:
            raise RuntimeError("No data yet — call connect_to_lsl() then run() first.")
        n = self.n_accepted_
        events = np.column_stack(
            [
                np.arange(n, dtype=int),
                np.zeros(n, dtype=int),
                np.array(self._code_list_[:n], dtype=int),
            ]
        )
        return mne.EpochsArray(
            self._buf_[:n].astype(np.float64),
            info=self.epochs_stream_.info,
            events=events,
            event_id=self.event_id,
            tmin=float(self.epochs_stream_.times[0]),
            verbose=False,
        )

    def get_evoked(self) -> "dict[str, mne.EvokedArray]":
        """Return per-condition grand-average as :class:`mne.EvokedArray` objects.

        Useful for immediate offline analysis, plotting with
        :func:`mne.viz.plot_evoked`, or source localisation via
        :meth:`get_source`.

        Returns
        -------
        evoked : dict[str, mne.EvokedArray]
            Mapping ``condition_label → EvokedArray``.  Conditions with
            zero accepted epochs are omitted.

        Examples
        --------
        >>> evoked = rt.get_evoked()
        >>> mne.viz.plot_evoked(evoked["auditory/left"])
        """
        epochs = self.get_epochs()
        result = {}
        for cond in self.event_id:
            try:
                result[cond] = epochs[cond].average()
            except KeyError:
                pass
        return result

    def save(self, path: str, overwrite: bool = False) -> None:
        """Save accumulated epochs to a ``-epo.fif`` file mid-run.

        The file can be reloaded offline with
        ``mne.read_epochs(path)`` and the full MNE analysis pipeline
        applied.

        Parameters
        ----------
        path : str
            Destination path.  Should end with ``-epo.fif`` or
            ``-epo.fif.gz`` to follow MNE naming conventions.
        overwrite : bool, default False
            Overwrite an existing file.

        Examples
        --------
        >>> rt.run(n_trials=30)
        >>> rt.save("session01-epo.fif", overwrite=True)
        """
        self.get_epochs().save(path, overwrite=overwrite, verbose=False)
        logger.info("RTEpochs: saved %d epochs to %s", self.n_accepted_, path)

    def get_source(
        self,
        inverse_operator,
        lambda2: float = 1.0 / 9.0,
        method: str = "dSPM",
    ) -> "dict[str, mne.SourceEstimate]":
        """Apply a pre-computed inverse operator to the current grand averages.

        Wraps :func:`mne.minimum_norm.apply_inverse` — load an existing
        inverse operator with
        ``mne.minimum_norm.read_inverse_operator(fname)``.

        Parameters
        ----------
        inverse_operator : mne.minimum_norm.InverseOperator
            Pre-computed inverse operator matching the stream's Info
            (same channels, same channel order).
        lambda2 : float, default 1/9
            Regularisation parameter (``1 / SNR²``).  Use ``1/9`` for
            SNR ≈ 3 (typical ERP), ``1.0`` for noisy single-trial data.
        method : str, default "dSPM"
            Inverse method: ``"MNE"``, ``"dSPM"``, ``"sLORETA"``, or
            ``"eLORETA"``.

        Returns
        -------
        stc_dict : dict[str, mne.SourceEstimate]
            Condition label → source estimate (vertex × time).

        Examples
        --------
        >>> inv_op = mne.minimum_norm.read_inverse_operator("fsaverage-inv.fif")
        >>> stc = rt.get_source(inv_op)
        >>> brain = mne_rt.viz.BrainPlot(subjects_fs_dir=sd)  # draws fsaverage
        >>> brain.update(stc["auditory/left"])
        """
        import mne.minimum_norm

        evoked = self.get_evoked()
        return {
            cond: mne.minimum_norm.apply_inverse(
                ev,
                inverse_operator,
                lambda2=lambda2,
                method=method,
                verbose=False,
            )
            for cond, ev in evoked.items()
        }
