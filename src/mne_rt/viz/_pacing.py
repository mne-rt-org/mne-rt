"""Slow the live viewers down while the Qt event loop runs late."""

from __future__ import annotations

import time


class LoadShedder:
    """Stretch the viewers' timer intervals while the event loop is overloaded.

    A probe timer measures how late it fires. On a machine that cannot draw
    every viewer at its full rate, the loop otherwise spends all its time
    redrawing and clicks and window events wait behind the frames, so the
    session looks hung. Here the viewers' intervals grow together while the
    probe runs late and shrink back once it is on time: fewer frames, a
    responsive window.

    Parameters
    ----------
    timers : list of (QTimer, int)
        Each viewer timer with its full-rate interval in ms.
    max_factor : float
        The most the intervals are stretched, as a multiple of full rate.
    """

    PROBE_MS = 100
    LATE_MS = 15.0  # a probe this late, on average, means the loop is overloaded
    ON_TIME_MS = 4.0

    def __init__(self, timers, max_factor: float = 4.0) -> None:
        from qtpy.QtCore import QTimer

        self._timers = list(timers)
        self._max = float(max_factor)
        self.factor = 1.0
        self._lag = 0.0
        self._last = None
        self._probe = QTimer()
        self._probe.setInterval(self.PROBE_MS)
        self._probe.timeout.connect(lambda: self.observe(time.perf_counter()))

    def start(self) -> None:
        self._last = None
        self._probe.start()

    def stop(self) -> None:
        self._probe.stop()

    def observe(self, now: float) -> None:
        """Take one probe firing at ``now`` (seconds) and adjust the timers."""
        if self._last is not None:
            late = max(0.0, (now - self._last) * 1e3 - self.PROBE_MS)
            self._lag = 0.7 * self._lag + 0.3 * late
            if self._lag > self.LATE_MS:
                self._set(min(self.factor * 1.25, self._max))
            elif self._lag < self.ON_TIME_MS:
                self._set(max(self.factor / 1.1, 1.0))
        self._last = now

    def _set(self, factor: float) -> None:
        if factor == self.factor:
            return
        self.factor = factor
        for timer, base_ms in self._timers:
            timer.setInterval(int(round(base_ms * factor)))
