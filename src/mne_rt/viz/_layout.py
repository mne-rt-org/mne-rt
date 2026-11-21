"""Window sizes and placement that fit the screen the viewers open on."""

from __future__ import annotations

import math

# A window opens at most this share of the screen's free area, so its title
# bar and edges stay reachable.
_MAX_SHARE = 0.92
# Frames are added by some window managers (X11) only after a window is shown,
# so the tiling is done again once they have had time to appear.
_RETILE_MS = 300


def _platform() -> str:
    from qtpy.QtGui import QGuiApplication

    return (QGuiApplication.platformName() or "").lower()


def _headless() -> bool:
    """No screen to fit: offscreen rendering (tests, CI, docs) and the like."""
    return _platform() in ("offscreen", "minimal", "vnc")


def available_geometry(widget=None):
    """The free area (no taskbar or menu bar) of the screen a window is on.

    For a window not shown yet, the screen under the mouse pointer, where new
    windows open on macOS and most window managers, rather than the primary
    one, which on a two-monitor setup may be the other screen.
    """
    from qtpy.QtGui import QCursor, QGuiApplication

    screen = None
    if widget is not None and widget.isVisible():
        screen = widget.screen()
    if screen is None:
        screen = QGuiApplication.screenAt(QCursor.pos())
    return (screen or QGuiApplication.primaryScreen()).availableGeometry()


def fit_size(width: int, height: int, widget=None) -> tuple[int, int]:
    """``width`` x ``height``, capped to the free area of the screen."""
    if _headless():
        return int(width), int(height)
    geo = available_geometry(widget)
    return (
        min(int(width), int(geo.width() * _MAX_SHARE)),
        min(int(height), int(geo.height() * _MAX_SHARE)),
    )


def fit_to_screen(window, width: int, height: int) -> None:
    """Resize ``window`` to ``width`` x ``height``, but no larger than its screen allows.

    The viewers have fixed default sizes (RawPlot's is 1500 x 720), wider than
    a 1366 x 768 laptop screen, where they opened partly off it. Headless, the
    size is kept as given.
    """
    window.resize(*fit_size(width, height, window))


def _row_sizes(n: int) -> list[int]:
    """Windows per row, top first. From three windows up, the first has the top
    row to itself: in a session that is the participant's NF display."""
    if n <= 2:
        return [n]
    rest = n - 1
    rows = 1 if rest <= 3 else 2
    base, extra = divmod(rest, rows)
    return [1] + [base + (1 if r >= rows - extra else 0) for r in range(rows)]


def can_arrange_windows() -> bool:
    """Whether top-level windows can be placed: not headless, and not Wayland,
    where the compositor ignores ``move()`` and tiling would only shrink them."""
    return not _headless() and not _platform().startswith("wayland")


def tile_windows(windows, geometry=None, gap: int = 6) -> None:
    """Arrange ``windows`` over the screen in rows, in order.

    Each window is shrunk to its cell, frame included, but never made larger
    than it is. On a screen too small for the windows' minimum sizes they can
    still overlap: Qt does not shrink a window below its minimum.

    Parameters
    ----------
    windows : list of QWidget | None
        Top-level windows, most important first; ``None`` entries are skipped.
    geometry : QRect | None
        The area to tile. By default the free area of the first window's
        screen, and only where windows can be placed (see
        :func:`can_arrange_windows`); there, tiling is repeated once the window
        manager has added the frames.
    gap : int
        Pixels between windows.
    """
    windows = [w for w in windows if w is not None]
    if not windows:
        return
    if geometry is None:
        if not can_arrange_windows():
            return
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()  # so frames already added are measured
        _tile(windows, available_geometry(windows[0]), gap)
        QTimer.singleShot(_RETILE_MS, lambda: _retile(windows, gap))
        return
    _tile(windows, geometry, gap)


def _retile(windows, gap: int) -> None:
    try:
        alive = [w for w in windows if w.isVisible()]
    except RuntimeError:  # a window deleted in the meantime
        return
    if alive:
        _tile(alive, available_geometry(alive[0]), gap)


def _tile(windows, geo, gap: int) -> None:
    sizes = _row_sizes(len(windows))
    cell_h = (geo.height() - gap * (len(sizes) - 1)) // len(sizes)
    it = iter(windows)
    for r, n_in_row in enumerate(sizes):
        cell_w = (geo.width() - gap * (n_in_row - 1)) // n_in_row
        y = geo.y() + r * (cell_h + gap)
        for c in range(n_in_row):
            w = next(it)
            # The frame (title bar, borders) around the client area resize() sets.
            frame_w = max(0, w.frameGeometry().width() - w.width())
            frame_h = max(0, w.frameGeometry().height() - w.height())
            w.resize(
                max(1, min(w.width(), cell_w - frame_w)), max(1, min(w.height(), cell_h - frame_h))
            )
            w.move(geo.x() + c * (cell_w + gap), y)
