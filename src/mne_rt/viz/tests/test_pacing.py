"""LoadShedder: the viewers slow down together while the event loop runs late."""

from __future__ import annotations

from mne_rt.viz._pacing import LoadShedder


def _timers():
    from qtpy.QtCore import QTimer

    fast, slow = QTimer(), QTimer()
    fast.setInterval(33)
    slow.setInterval(200)
    return [(fast, 33), (slow, 200)]


def _probe(shedder, late_ms, n):
    t = 0.0
    for _ in range(n):
        t += (shedder.PROBE_MS + late_ms) / 1e3
        shedder.observe(t)


def test_an_on_time_loop_keeps_the_full_rate(qt_app):
    timers = _timers()
    shedder = LoadShedder(timers)
    _probe(shedder, late_ms=1.0, n=50)
    assert shedder.factor == 1.0
    assert [t.interval() for t, _ in timers] == [33, 200]


def test_a_late_loop_stretches_every_timer_together_up_to_the_cap(qt_app):
    timers = _timers()
    shedder = LoadShedder(timers, max_factor=4.0)
    _probe(shedder, late_ms=60.0, n=50)
    assert shedder.factor == 4.0
    assert [t.interval() for t, _ in timers] == [132, 800]


def test_the_full_rate_comes_back_once_the_loop_is_on_time(qt_app):
    timers = _timers()
    shedder = LoadShedder(timers)
    _probe(shedder, late_ms=60.0, n=30)
    assert shedder.factor > 1.0
    _probe(shedder, late_ms=0.0, n=200)
    assert shedder.factor == 1.0
    assert [t.interval() for t, _ in timers] == [33, 200]
