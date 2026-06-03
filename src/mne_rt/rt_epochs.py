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

Classes
-------
RTEpochs
    Event-triggered epoch accumulator backed by mne_lsl.EpochsStream.
"""
from __future__ import annotations

import time
import threading
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np

try:
    from mne_lsl.stream import StreamLSL, EpochsStream
    from mne_lsl.player import PlayerLSL
    _mne_lsl_available = True
except ImportError:
    _mne_lsl_available = False

from mne_rt._logging import logger, set_log_level


class RTEpochs:
    """Event-triggered epoch accumulator backed by :class:`mne_lsl.stream.EpochsStream`.

    Connects a :class:`~mne_lsl.stream.StreamLSL` to an
    :class:`~mne_lsl.stream.EpochsStream`, polls for new epochs, and
    optionally drives an :class:`~mne_rt.viz.ERPPlot` that redraws after
    every new trial.

    Parameters
    ----------
    event_id : dict[str, int]
        Condition label → marker integer, e.g. ``{"target": 1, "standard": 2}``.
    event_channels : str or list of str
        Channel(s) in the LSL stream that carry the event codes (e.g.
        ``"STI 014"`` for a STIM channel, or ``"stim"``).
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

        ``new_data`` is ``(n_new, n_channels, n_times)``; ``all_events`` is
        the current :attr:`~mne_lsl.stream.EpochsStream.events` array.
    verbose : bool or str or None, default None

    Attributes
    ----------
    epochs_stream_ : mne_lsl.stream.EpochsStream or None
        The underlying :class:`~mne_lsl.stream.EpochsStream` after
        :meth:`connect_to_lsl` has been called.
    n_accepted_ : int
        Running count of accepted epochs since :meth:`run` started.

    See Also
    --------
    mne_rt.viz.ERPPlot : Live ERP display driven by this class.
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
        self.n_accepted_: int = 0
        self._stop_event = threading.Event()
        self._connected = False

        # Populated by run() — persists for get_epochs/get_evoked/save
        self._buf_: Optional[np.ndarray] = None        # (n_trials, n_ch, n_t)
        self._cond_list_: list[str]  = []
        self._code_list_: list[int]  = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect_to_lsl(
        self,
        stream_name: Optional[str] = None,
        mock_lsl: bool = False,
        fname: Optional[str] = None,
        timeout: float = 10.0,
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
        verbose : bool or str or None

        Returns
        -------
        self : RTEpochs
        """
        if verbose is not None:
            set_log_level(verbose)

        if mock_lsl:
            if fname is None:
                raise ValueError("fname is required when mock_lsl=True.")
            logger.info("RTEpochs: starting mock PlayerLSL from %s", fname)
            self._player = PlayerLSL(fname, name="mne_rt_mock", chunk_size=16).start()
            time.sleep(1.5)
            stream_name = "mne_rt_mock"

        logger.info("RTEpochs: connecting StreamLSL …")
        self._stream = StreamLSL(bufsize=4.0, name=stream_name)
        self._stream.connect(acquisition_delay=0.005, timeout=timeout)
        logger.info(
            "RTEpochs: stream connected — %d ch @ %.0f Hz",
            self._stream.info["nchan"], self._stream.info["sfreq"],
        )

        logger.info("RTEpochs: setting up EpochsStream …")
        self.epochs_stream_ = EpochsStream(
            stream=self._stream,
            bufsize=self.bufsize,
            event_id=self.event_id,
            event_channels=self.event_channels,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            picks=self.picks,
            reject=self.reject,
        ).connect(acquisition_delay=0.005)

        self._connected = True
        logger.info("RTEpochs: EpochsStream connected.")
        return self

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
            Open an :class:`~mne_rt.viz.ERPPlot` that redraws every
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
            from mne_rt.viz.erp_plot import ERPPlot
            erp_plot = ERPPlot(
                ch_names=list(es.info["ch_names"]),
                sfreq=es.info["sfreq"],
                tmin=self.tmin,
                tmax=self.tmax,
                event_id=self.event_id,
                info=es.info,          # pass real Info for accurate layout
                baseline=self.baseline,
            )
            erp_plot.show()

        inv_event = {v: k for k, v in self.event_id.items()}

        # Pre-allocate epoch buffer — avoids O(N²) np.stack per trial
        n_ch    = es.info["nchan"]
        n_times = int(round((self.tmax - self.tmin) * es.info["sfreq"])) + 1
        self._buf_       = np.zeros((n_trials, n_ch, n_times), dtype=np.float32)
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
            data   = self.epochs_stream_.get_data(n_epochs=n_new)
            events = self.epochs_stream_.events[-n_new:]

            for i in range(data.shape[0]):
                if self.n_accepted_ >= n_trials:
                    break
                code      = int(events[i]) if events.ndim == 1 else int(events[i, 2])
                condition = inv_event.get(code, str(code))

                # Write into pre-allocated buffer (O(1) copy)
                ep = data[i]
                t  = min(ep.shape[-1], n_times)
                self._buf_[self.n_accepted_, :, :t] = ep[:, :t]
                self._cond_list_.append(condition)
                self._code_list_.append(code)
                self.n_accepted_ += 1

                # on_trial now receives event_code + condition directly
                if self.on_trial is not None:
                    self.on_trial(
                        self.n_accepted_,
                        self._buf_[self.n_accepted_ - 1],   # view — no copy
                        code,
                        condition,
                    )

                if erp_plot is not None and self.n_accepted_ % erp_update_every == 0:
                    # Pass a view of the filled portion — no copy
                    erp_plot.update(self._buf_[:self.n_accepted_], list(self._cond_list_))

                logger.debug("RTEpochs: accepted %d (%s)", self.n_accepted_, condition)

        logger.info("RTEpochs: finished — %d epochs accepted.", self.n_accepted_)
        return self

    def stop(self) -> None:
        """Signal the run loop to stop after the current poll."""
        self._stop_event.set()

    def disconnect(self) -> None:
        """Disconnect EpochsStream, StreamLSL, and stop any mock player."""
        if self.epochs_stream_ is not None:
            try:
                self.epochs_stream_.disconnect()
            except Exception:
                pass
        if self._stream is not None:
            try:
                self._stream.disconnect()
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
            raise RuntimeError(
                "No data yet — call connect_to_lsl() then run() first."
            )
        n = self.n_accepted_
        events = np.column_stack([
            np.arange(n, dtype=int),
            np.zeros(n, dtype=int),
            np.array(self._code_list_[:n], dtype=int),
        ])
        return mne.EpochsArray(
            self._buf_[:n].astype(np.float64),
            info=self.epochs_stream_.info,
            events=events,
            event_id=self.event_id,
            tmin=self.tmin,
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
        >>> inv_op = mne.minimum_norm.read_inverse_operator("sample-inv.fif")
        >>> stc = rt.get_source(inv_op)
        >>> brain = mne_rt.BrainPlot(subject="sample", subjects_dir=sd)
        >>> brain.update(stc["auditory/left"].data.mean(-1))
        """
        import mne.minimum_norm
        evoked = self.get_evoked()
        return {
            cond: mne.minimum_norm.apply_inverse(
                ev, inverse_operator, lambda2=lambda2, method=method,
                verbose=False,
            )
            for cond, ev in evoked.items()
        }
