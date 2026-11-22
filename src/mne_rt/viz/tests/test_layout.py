"""Viewer windows fit the screen and tile over it without overlapping."""

from __future__ import annotations

import pytest

from mne_rt.viz import _layout
from mne_rt.viz._layout import _row_sizes, fit_to_screen, tile_windows

pytestmark = pytest.mark.usefixtures("_destroy_widgets")


def _windows(sizes):
    from qtpy.QtWidgets import QWidget

    out = []
    for w, h in sizes:
        win = QWidget()
        win.resize(w, h)
        out.append(win)
    return out


@pytest.mark.parametrize(
    "n, rows", [(1, [1]), (2, [2]), (3, [1, 2]), (4, [1, 3]), (5, [1, 2, 2]), (6, [1, 2, 3])]
)
def test_from_three_windows_the_first_has_the_top_row(n, rows):
    """In a session the first is the participant's NF display."""
    assert _row_sizes(n) == rows


def test_a_window_wider_than_the_screen_is_capped_to_it(qt_app, monkeypatch):
    monkeypatch.setattr(_layout, "_headless", lambda: False)
    (win,) = _windows([(10, 10)])
    geo = _layout.available_geometry(win)
    fit_to_screen(win, 10 * geo.width(), 10 * geo.height())
    assert win.width() <= geo.width() and win.height() <= geo.height()
    fit_to_screen(win, 200, 150)  # one that fits keeps its size
    assert (win.width(), win.height()) == (200, 150)


def test_headless_windows_keep_their_size(qt_app):
    """Offscreen (tests, CI, docs) there is no screen: exports keep full size."""
    assert _layout._headless()  # this suite runs offscreen
    (win,) = _windows([(10, 10)])
    fit_to_screen(win, 5000, 3000)
    assert (win.width(), win.height()) == (5000, 3000)


@pytest.mark.parametrize("platform, ok", [("xcb", True), ("cocoa", True), ("wayland", False)])
def test_windows_are_not_arranged_where_they_cannot_be_placed(monkeypatch, platform, ok):
    """Wayland ignores move(): tiling would only shrink the windows."""
    monkeypatch.setattr(_layout, "_platform", lambda: platform)
    assert _layout.can_arrange_windows() is ok


def test_tiled_windows_fill_the_screen_without_overlapping(qt_app):
    from qtpy.QtCore import QRect

    geo = QRect(0, 0, 1920, 1080)
    wins = _windows([(1440, 800), (1200, 540), (1600, 1000), (300, 200)])
    tile_windows([wins[0], None, *wins[1:]], geometry=geo)  # None: a viewer not shown
    rects = [w.geometry() for w in wins]
    for i, a in enumerate(rects):
        assert geo.contains(a)
        for b in rects[i + 1 :]:
            assert not a.intersects(b)
    assert (wins[3].width(), wins[3].height()) == (300, 200)  # never enlarged
    assert rects[0].topLeft() == geo.topLeft()
    assert rects[0].width() == 1440  # alone in the top row: its full width


def test_on_a_real_screen_the_windows_are_tiled_and_tiled_again(qt_app, monkeypatch):
    """There, frames may come after show() (X11): tiled once now, once later."""
    from qtpy.QtCore import QRect, QTimer

    geo = QRect(0, 0, 1600, 900)
    later = []
    monkeypatch.setattr(_layout, "can_arrange_windows", lambda: True)
    monkeypatch.setattr(_layout, "available_geometry", lambda widget=None: geo)
    monkeypatch.setattr(QTimer, "singleShot", lambda ms, fn: later.append((ms, fn)))
    wins = _windows([(1440, 800), (1200, 540), (1500, 720)])
    for w in wins:
        w.show()
    tile_windows(wins)
    assert wins[0].frameGeometry().topLeft() == geo.topLeft()  # the frame, not the client area
    assert [ms for ms, _ in later] == [_layout._RETILE_MS]
    wins[1].move(900, 800)  # say, where a window manager put it
    later[0][1]()
    assert geo.contains(wins[1].frameGeometry())


def test_the_second_pass_skips_windows_closed_in_between(qt_app, monkeypatch):
    from qtpy.QtCore import QRect

    monkeypatch.setattr(_layout, "available_geometry", lambda widget=None: QRect(0, 0, 800, 600))
    wins = _windows([(300, 200), (300, 200)])
    wins[0].show()
    _layout._retile(wins, gap=6)  # the hidden one is left out, not moved
    assert wins[0].frameGeometry().topLeft().isNull()
    wins[0].hide()
    _layout._retile(wins, gap=6)  # nothing left to tile


def test_tiling_is_skipped_where_windows_cannot_be_placed(qt_app):
    wins = _windows([(300, 200), (300, 200)])
    for w in wins:
        w.move(50, 60)
    tile_windows(wins)  # headless here: no screen to tile over
    assert [w.pos().x() for w in wins] == [50, 50]


def test_the_screen_is_the_window_s_own_once_shown(qt_app):
    (win,) = _windows([(200, 100)])
    unshown = _layout.available_geometry(win)
    win.show()
    assert _layout.available_geometry(win) == win.screen().availableGeometry()
    assert unshown.width() > 0
