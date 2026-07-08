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
from typing import Sequence


class LSLSender:
    """Thread-safe LSL outlet that broadcasts NF feature values.

    Creates a single-source LSL stream with ``n_channels`` float32 channels
    (one per active NF modality).  Channel labels are set from the modality
    names on the first :meth:`push` call.

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
    ) -> None:
        self._StreamInfo, self._StreamOutlet = self._import_lsl()

        self.stream_name = stream_name
        self.stream_type = stream_type
        self.srate = srate
        self.source_id = source_id

        self._n_channels = n_channels
        self._outlet = None
        self._channel_labels: list[str] = []
        self._lock = threading.Lock()

        self._outlet = self._make_outlet(n_channels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _make_outlet(self, n_channels: int):
        info = self._StreamInfo(
            name=self.stream_name,
            stype=self.stream_type,
            n_channels=n_channels,
            sfreq=self.srate,
            dtype="float32",
            source_id=self.source_id,
        )
        return self._StreamOutlet(info)

    def _ensure_channels(self, n: int) -> None:
        """Recreate the outlet if the channel count needs to grow."""
        if n <= self._n_channels:
            return
        self._outlet.close() if hasattr(self._outlet, "close") else None
        self._n_channels = n
        self._outlet = self._make_outlet(n)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
            If ``modalities`` and ``values`` have different lengths.
        """
        if len(modalities) != len(values):
            raise ValueError(
                f"modalities and values must have the same length; "
                f"got {len(modalities)} and {len(values)}."
            )

        n = len(values)
        with self._lock:
            self._ensure_channels(n)
            # Pad with zeros if outlet has more channels than active modalities
            sample = list(values) + [0.0] * (self._n_channels - n)
            self._outlet.push_sample(sample)
            self._channel_labels = list(modalities)

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
