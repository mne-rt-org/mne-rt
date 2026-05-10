"""Open Sound Control (OSC) output for the Advanced Neurofeedback Toolbox.

Allows NF feature values to be streamed in real-time to any OSC-capable
application — Max/MSP, SuperCollider, Pure Data, TouchDesigner, Unity, etc.

Requirements
------------
``python-osc`` is bundled with ANT (no extra install needed).  If for
any reason it is missing, :class:`OSCSender` raises :exc:`ImportError`
at construction time.

OSC address scheme
------------------
Each modality is sent to a unique address::

    /ant/<modality>   float32   <value>

All active modalities can also be bundled into a single UDP datagram::

    /ant/bundle   str str ...   "mod1 mod2 ..."   float32 float32 ...

Examples
--------
Send alpha power to SuperCollider on localhost::

    sender = OSCSender(host="127.0.0.1", port=57120)
    sender.send("sensor_power", 0.42)
    sender.close()

Or pass it to :meth:`~ant.NFRealtime.record_main`::

    nf.record_main(duration=300, modality="sensor_power",
                   osc_sender=OSCSender(port=9000))

Classes
-------
OSCSender
    Thread-safe OSC client for real-time NF data streaming.
"""
from __future__ import annotations

import threading
from typing import Sequence, Union


class OSCSender:
    """Thread-safe OSC client that broadcasts NF feature values.

    Sends one UDP packet per NF value (or one bundle per update cycle)
    to an OSC server at the given host/port.  Safe to call from any
    thread.

    Parameters
    ----------
    host : str, default "127.0.0.1"
        Destination IP address or hostname.
    port : int, default 9000
        Destination UDP port.
    prefix : str, default "/ant"
        OSC address prefix.  Each modality is sent to
        ``<prefix>/<modality>``.
    bundle : bool, default False
        If ``True``, pack all modality values into a single OSC bundle
        per update cycle (one UDP packet) instead of one packet per
        modality.

    Raises
    ------
    ImportError
        If ``python-osc`` is not installed (should not occur with a
        standard ANT install).

    Examples
    --------
    Basic usage::

        sender = OSCSender(host="127.0.0.1", port=9000)
        sender.send("sensor_power", 0.35)
        sender.send_all(["sensor_power", "erd_ers"], [0.35, -12.4])
        sender.close()

    Custom prefix (maps to ``/nf/sensor_power``)::

        sender = OSCSender(prefix="/nf")

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        prefix: str = "/ant",
        bundle: bool = False,
    ) -> None:
        try:
            from pythonosc.udp_client import SimpleUDPClient  # type: ignore
            from pythonosc.osc_bundle_builder import OscBundleBuilder  # type: ignore
            from pythonosc.osc_message_builder import OscMessageBuilder  # type: ignore
            import pythonosc.osc_bundle as _osc_bundle
            self._SimpleUDPClient = SimpleUDPClient
            self._OscBundleBuilder = OscBundleBuilder
            self._OscMessageBuilder = OscMessageBuilder
            self._osc_bundle = _osc_bundle
        except ImportError as exc:
            raise ImportError(
                "python-osc is required for OSC output.  "
                "Re-install ANT or run:  pip install python-osc"
            ) from exc

        self.host = host
        self.port = port
        self.prefix = prefix.rstrip("/")
        self.bundle = bundle

        self._client = SimpleUDPClient(host, port)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def send(self, modality: str, value: float) -> None:
        """Send a single NF value immediately.

        Parameters
        ----------
        modality : str
            Modality name (e.g. ``"sensor_power"``).  Appended to the
            OSC prefix to form the address.
        value : float
            Current NF feature value.

        Notes
        -----
        The OSC message sent is::

            <prefix>/<modality>  float32  <value>
        """
        address = f"{self.prefix}/{modality}"
        with self._lock:
            self._client.send_message(address, float(value))

    def send_all(
        self,
        modalities: Sequence[str],
        values: Sequence[float],
    ) -> None:
        """Send all NF values for one update cycle.

        When ``bundle=True`` (set in :meth:`__init__`), packs everything
        into a single OSC bundle datagram.  Otherwise sends one message
        per modality.

        Parameters
        ----------
        modalities : sequence of str
            Active modality names, in order.
        values : sequence of float
            Corresponding feature values.

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

        with self._lock:
            if self.bundle:
                self._send_bundle(modalities, values)
            else:
                for mod, val in zip(modalities, values):
                    self._client.send_message(
                        f"{self.prefix}/{mod}", float(val)
                    )

    def send_raw(self, address: str, *args) -> None:
        """Send an arbitrary OSC message to a custom address.

        Parameters
        ----------
        address : str
            Full OSC address (e.g. ``"/my/custom/path"``).
        *args
            OSC arguments (int, float, str, bytes).
        """
        with self._lock:
            self._client.send_message(address, list(args) if len(args) > 1 else (args[0] if args else 0))

    def close(self) -> None:
        """Close the underlying UDP socket.

        After calling this method the sender should not be used again.
        """
        try:
            self._client._sock.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def target(self) -> str:
        """Human-readable target as ``"host:port"``."""
        return f"{self.host}:{self.port}"

    def __repr__(self) -> str:
        return (
            f"OSCSender(host={self.host!r}, port={self.port}, "
            f"prefix={self.prefix!r}, bundle={self.bundle})"
        )

    def __enter__(self) -> "OSCSender":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_bundle(
        self,
        modalities: Sequence[str],
        values: Sequence[float],
    ) -> None:
        """Pack all messages into one OSC bundle and dispatch."""
        import pythonosc.osc_bundle_builder as bb
        import pythonosc.osc_message_builder as mb
        import time

        builder = bb.OscBundleBuilder(bb.IMMEDIATELY)
        for mod, val in zip(modalities, values):
            msg = mb.OscMessageBuilder(address=f"{self.prefix}/{mod}")
            msg.add_arg(float(val))
            builder.add_content(msg.build())

        bundle = builder.build()
        self._client._sock.sendto(bundle.dgram, (self.host, self.port))
