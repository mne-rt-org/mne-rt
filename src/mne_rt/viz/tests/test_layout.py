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
