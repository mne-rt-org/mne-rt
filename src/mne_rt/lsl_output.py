"""LSL stream outlet for broadcasting real-time feature values.

:class:`LSLSender` creates a Lab Streaming Layer (LSL) outlet that pushes
computed feature values as a float-channel stream.  Any LSL-aware application
(Psychtoolbox, PsychoPy, OpenViBE, BCI2000, another MNE-RT instance, …) can
subscribe to this stream and use the values for stimulus control or
further analysis.

This is faster and more reliable than OSC for same-machine communication
because it uses shared memory / localhost TCP rather than UDP, and LSL
handles timestamping, buffering, and clock synchronisation automatically.

Use :class:`~mne_rt.osc.OSCSender` instead when the feedback application runs
on a *different machine* and supports OSC but not LSL.

Classes
-------
LSLSender
    Thread-safe LSL outlet that pushes NF values to downstream subscribers.

Examples
--------
Send alpha power into an LSL stream named ``"ANT_NF"``::

    sender = LSLSender(stream_name="ANT_NF", n_channels=1)
    sender.push(["sensor_power"], [0.42])
    sender.close()

Pass to :meth:`~mne_rt.RTStream.record_main` alongside (or instead of) OSC::

    nf.record_main(duration=300, modality="sensor_power",
                   lsl_sender=LSLSender())
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np

from mne_rt._logging import logger


class LSLSender:
    """Thread-safe LSL outlet that broadcasts NF feature values.

    Creates a single-source LSL stream with ``n_channels`` float32 channels
    (one per active NF modality).  Channel labels are published in the stream
    description, so subscribers can select a value by name rather than by
    position -- taken from ``channel_names`` if given, otherwise from the
    modality names on the first :meth:`push` call.

    Parameters
    ----------
    stream_name : str, default "ANT_NF"
        LSL stream name visible to subscribers.
    stream_type : str, default "NF"
        LSL content type (arbitrary string; "NF" is ANT convention).
    n_channels : int, default 8
        Maximum number of channels in the outlet.  If fewer modalities are
        active, unused channels are filled with ``0.0``.  You can leave
        this at the default and the outlet will resize automatically on
        first push if needed.
    srate : float, default 0.0
        Nominal sample rate in Hz.  ``0.0`` marks the stream as irregular
        (i.e. one sample per NF window, not a fixed rate).
    source_id : str, default "ant_nf_outlet"
        Unique source identifier embedded in the stream info.
    channel_names : sequence of str | None, default None
        Channel labels to publish in the stream description.  When omitted,
        they are taken from the modality names on the first :meth:`push`,
        which rebuilds the outlet once and so briefly drops any subscriber
        that had already resolved the stream.  Pass them here when they are
        known in advance to avoid that.

    Raises
    ------
    ImportError
        If neither ``mne_lsl`` nor ``pylsl`` is installed.

    Examples
    --------
    Basic usage::

        sender = LSLSender(stream_name="ANT_NF")
        sender.push(["sensor_power", "erd_ers"], [0.42, -1.2])
        sender.close()

    Context-manager usage::

        with LSLSender() as sender:
            for value in nf_stream:
                sender.push(["sensor_power"], [value])

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        stream_name: str = "ANT_NF",
        stream_type: str = "NF",
        n_channels: int = 8,
        srate: float = 0.0,
        source_id: str = "ant_nf_outlet",
        channel_names: Optional[Sequence[str]] = None,
    ) -> None:
        self._StreamInfo, self._StreamOutlet = self._import_lsl()

        self.stream_name = stream_name
        self.stream_type = stream_type
        self.srate = srate
        self.source_id = source_id

        self._n_channels = n_channels
        self._outlet = None
        self._channel_labels: list[str] = list(channel_names) if channel_names else []
        # Publishing names means rebuilding the outlet, which drops subscribers.
        # Naming the channels here avoids that entirely.
        self._names_published = bool(channel_names)
        self._lock = threading.Lock()

        self._outlet = self._make_outlet(n_channels, self._channel_labels)

    # Internal helpers

    @staticmethod
    def _import_lsl():
        """Return (StreamInfo, StreamOutlet) from whichever LSL binding is available."""
        try:
            from mne_lsl.lsl import StreamInfo, StreamOutlet

            return StreamInfo, StreamOutlet
        except ImportError:
            pass
        try:
            from pylsl import StreamInfo, StreamOutlet  # type: ignore[no-redef]

            return StreamInfo, StreamOutlet
        except ImportError as exc:
            raise ImportError(
                "LSLSender requires mne_lsl or pylsl.\n"
                "Install ANT with its standard dependencies:  pip install ANT\n"
                "mne_lsl is a core dependency and should already be present."
            ) from exc

    def _make_outlet(self, n_channels: int, labels: Optional[Sequence[str]] = None):
        info = self._StreamInfo(
            name=self.stream_name,
            stype=self.stream_type,
            n_channels=n_channels,
            sfreq=self.srate,
            dtype="float32",
            source_id=self.source_id,
        )
        if labels:
            # Labels, so subscribers can find a value by name. Padded, as the outlet
            # may be wider than the active modalities.
            names = list(labels)[:n_channels]
            names += [f"ch{i}" for i in range(len(names), n_channels)]
            try:
                # mne_lsl's StreamInfo; the pylsl fallback has no such setter,
                # and labels are a convenience, never worth failing a session for.
                info.set_channel_names(names)
            except Exception:
                logger.debug("Could not set LSL channel names.", exc_info=True)
        return self._StreamOutlet(info)

    def _ensure_channels(self, n: int, labels: Optional[Sequence[str]] = None) -> None:
        """Recreate the outlet if the channel count needs to grow, or to name it.

        Rebuilding drops existing subscribers, so it happens at most twice in a
        session: if the outlet has to widen, and once to publish channel names
        the first time a caller supplies them.  Later name changes are recorded
        but do not rebuild — otherwise alternating :meth:`push_value` calls
        would tear the outlet down on every sample.
        """
        grow = n > self._n_channels
        name_it = bool(labels) and not self._names_published
        if labels:
            self._channel_labels = list(labels)
        if not (grow or name_it):
            return
        self._outlet.close() if hasattr(self._outlet, "close") else None
        self._n_channels = max(n, self._n_channels)
        self._names_published = self._names_published or name_it
        self._outlet = self._make_outlet(self._n_channels, self._channel_labels)

    # Public interface

    def push(
        self,
        modalities: Sequence[str],
        values: Sequence[float],
    ) -> None:
        """Push one NF sample into the LSL outlet.

        Parameters
        ----------
        modalities : sequence of str
            Active modality names (used to set channel labels on first call).
        values : sequence of float
            Corresponding NF feature values, same length as *modalities*.

        Raises
        ------
        ValueError
            If ``modalities`` and ``values`` have different lengths, or if a
            value is not a number.
        """
        if len(modalities) != len(values):
            raise ValueError(
                f"modalities and values must have the same length; "
                f"got {len(modalities)} and {len(values)}."
            )

        n = len(values)
        with self._lock:
            # Labels first: they are published in the StreamInfo, so the outlet
            # has to carry them before the sample goes out, not after.
            self._ensure_channels(n, modalities)
            # mne-lsl needs a NumPy array (a list raises), in the outlet's float32.
            # Entries past `n` stay zero: the padding of a wider outlet.
            sample = np.zeros(self._n_channels, dtype=np.float32)
            sample[:n] = values
            self._outlet.push_sample(sample)

    def push_value(self, modality: str, value: float) -> None:
        """Push a single-channel NF value.

        Parameters
        ----------
        modality : str
            Modality name.
        value : float
            NF feature value.
        """
        self.push([modality], [value])

    def close(self) -> None:
        """Destroy the LSL outlet and release resources.

        After calling this the sender should not be used again.
        """
        with self._lock:
            if self._outlet is not None:
                try:
                    if hasattr(self._outlet, "close"):
                        self._outlet.close()
                    del self._outlet
                except Exception:
                    pass
                self._outlet = None

    # Properties

    @property
    def n_channels(self) -> int:
        """Current number of channels in the LSL outlet."""
        return self._n_channels

    @property
    def channel_labels(self) -> list[str]:
        """Modality names from the most recent :meth:`push` call."""
        return list(self._channel_labels)

    def __repr__(self) -> str:
        return (
            f"LSLSender(stream_name={self.stream_name!r}, "
            f"n_channels={self._n_channels}, "
            f"active={self._outlet is not None})"
        )

    def __enter__(self) -> "LSLSender":
        return self

    def __exit__(self, *_) -> None:
        self.close()
