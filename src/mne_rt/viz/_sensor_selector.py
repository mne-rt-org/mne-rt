"""Clickable mini scalp map for picking channels, shared by the live viewers.

Functions
---------
sensor_positions
    Normalised 2-D sensor positions for a list of channels.

Classes
-------
SensorSelector
    Head outline with one clickable dot per channel.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont

from mne_rt._logging import logger
from mne_rt._montage import make_standard_montage


def _from_layout(ch_names: list[str], layout) -> Optional[list[tuple[float, float]]]:
    if layout is None:
        return None
    name_xy: dict[str, tuple[float, float]] = {}
    for name, pos in zip(layout.names, layout.pos):
        xc = float(pos[0] + pos[2] / 2.0)
        yc = float(pos[1] + pos[3] / 2.0)
        name_xy[name] = (xc, 1.0 - yc)  # yn=0 = frontal
    n_matched = sum(1 for c in ch_names if c in name_xy)
    if n_matched < len(ch_names) // 2:
        return None
    positions: list[tuple[float, float]] = []
    fb = 0
    for ch in ch_names:
        if ch in name_xy:
            positions.append(name_xy[ch])
        else:
            # Columns of 20 at the left edge: in a single one, the 21st and
            # later fell below the view, where they could not be clicked.
            positions.append((0.02 + 0.05 * (fb // 20), (fb % 20) * 0.05))
            fb += 1
    return positions


def sensor_positions(
    ch_names: list[str], info=None, montage: Optional[str] = "standard_1020"
) -> list[tuple[float, float]]:
    """Return normalised ``(xn, yn)`` per channel, ``yn=0`` frontal.

    Positions come from *info* when it has a usable layout, else from the
    standard *montage*, else the channels are placed on a circle. Channels
    without a position in a layout sit in a column at the left edge.

    Parameters
    ----------
    ch_names : list of str
        Channels in display order.
    info : mne.Info | None
        Measurement info, used first.
    montage : str | None
        Name of a standard montage to fall back on.

    Returns
    -------
    positions : list of (float, float)
        One position per channel, both coordinates in ``[0, 1]``.
    """
    ch_names = list(ch_names)
    try:
        import mne
    except ImportError:
        mne = None
    if mne is not None:
        if info is not None:
            try:
                # find_layout leaves bad channels out; they still need a place
                info = info.copy()
                info["bads"] = []
                # and refuses channels without a position, which all sit at 0;
                # _from_layout puts them in a column at the edge instead
                placed = [
                    i
                    for i, ch in enumerate(info["chs"])
                    if np.all(np.isfinite(ch["loc"][:3])) and np.any(ch["loc"][:3])
                ]
                if 0 < len(placed) < len(info["chs"]):
                    info = mne.pick_info(info, placed)
                pos = _from_layout(ch_names, mne.channels.find_layout(info))
                if pos is not None:
                    return pos
            except Exception as exc:
                logger.debug("Sensor layout from info failed: %s", exc)
        if montage is not None:
            try:
                tmp = mne.create_info(ch_names, sfreq=1.0, ch_types="eeg", verbose=False)
                mont = make_standard_montage(montage)
                tmp.set_montage(mont, on_missing="ignore", verbose=False)
                # Through the info branch, which sets aside channels the montage
                # does not place; find_layout refuses them, and a direct call
                # here fell back to the circle as soon as there was one.
                return sensor_positions(ch_names, tmp, montage=None)
            except Exception as exc:
                logger.debug("Sensor layout from montage %r failed: %s", montage, exc)
    logger.warning("No sensor positions found: falling back to a circular layout.")
    n = len(ch_names)
    return [
        (
            0.5 + 0.42 * math.cos(2 * math.pi * i / n - math.pi / 2),
            0.5 + 0.42 * math.sin(2 * math.pi * i / n - math.pi / 2),
        )
        for i in range(n)
    ]


class _ClickableScatter(pg.ScatterPlotItem):
    """A scatter that emits :attr:`sigClicked` for right clicks as well.

    pyqtgraph's own emits it for the left button only, so a right click on a
    dot never reached :meth:`SensorSelector._on_click`.
    """

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.RightButton:
            points = self.pointsAt(ev.pos())
            if len(points) > 0:
                ev.accept()
                self.sigClicked.emit(self, points, ev)
                return
        super().mouseClickEvent(ev)


class SensorSelector(pg.PlotWidget):
    """Head outline with one clickable dot per channel.

    A left click adds the channel to the selection or removes it; the dot
    grows and takes the accent colour while selected. A right click emits
    :attr:`channelRightClicked`, e.g. to mark the channel bad. Bad channels
    are drawn as hollow rings.

    Parameters
    ----------
    ch_names : list of str
        Channels, in the order of *positions*.
    positions : list of (float, float)
        Normalised positions, ``yn=0`` frontal (see :func:`sensor_positions`).
    selected : iterable of str
        Channels selected at start.
    max_selected : int | None
        Clicks that would select more channels are ignored.
    min_selected : int
        Clicks that would select fewer channels are ignored.
    size : int
        Width and height of the widget in pixels.
    background, outline, accent, bad, hint_color : str
        Colours of the background, head outline and idle dots, selected
        dots, bad-channel rings, and hint text.
    hint : str
        Text under the head.
    parent : QWidget | None
        Parent widget.
    """

    #: Emitted with the selected channel names after every change by a click.
    selectionChanged = Signal(list)
    #: Emitted with a channel name when its dot is right-clicked.
    channelRightClicked = Signal(str)

    def __init__(
        self,
        ch_names: list[str],
        positions: list[tuple[float, float]],
        selected: Iterable[str] = (),
        *,
        max_selected: Optional[int] = None,
        min_selected: int = 1,
        size: int = 184,
        background: str = "#161b22",
        outline: str = "#30363d",
        accent: str = "#3b82f6",
        bad: str = "#ef4444",
        hint: str = "click to select",
        hint_color: str = "#8b949e",
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self._ch_names = list(ch_names)
        self._positions = list(positions)
        self._selected = [ch for ch in selected if ch in self._ch_names]
        self._bads: set[str] = set()
        self._max_selected = max_selected
        self._min_selected = min_selected
        self._outline = outline
        self._accent = accent
        self._bad = bad

        self.setFixedSize(size, size)
        self.setBackground(background)
        self.hideAxis("bottom")
        self.hideAxis("left")
        vb = self.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setMenuEnabled(False)
        vb.setAspectLocked(True)
        # A small margin so the nose isn't clipped
        vb.setRange(xRange=(-0.06, 1.06), yRange=(-0.06, 1.14), padding=0)

        pen = pg.mkPen(outline, width=1.5)
        theta = np.linspace(0, 2 * np.pi, 160)
        self.plot(0.5 + 0.48 * np.cos(theta), 0.5 + 0.48 * np.sin(theta), pen=pen)
        self.plot([0.47, 0.5, 0.53, 0.47], [0.97, 1.06, 0.97, 0.97], pen=pen)
        for side in (-1, 1):
            self.plot(
                0.5 + np.linspace(0.48 * side, 0.56 * side, 8),
                0.5 + 0.06 * np.sin(np.linspace(0, np.pi, 8)),
                pen=pen,
            )

        self._scatter = _ClickableScatter(
            hoverable=True, tip=lambda x, y, data: str(data) if data else ""
        )
        self._scatter.sigClicked.connect(self._on_click)
        self.addItem(self._scatter)

        text = pg.TextItem(hint, color=hint_color, anchor=(0.5, 1))
        text.setFont(QFont("Helvetica", 7))
        text.setPos(0.5, -0.02)
        self.addItem(text)
        self._refresh()

    @property
    def selected(self) -> list[str]:
        """Selected channels, in the order they were selected."""
        return list(self._selected)

    def set_selected(self, channels: Iterable[str]) -> None:
        """Replace the selection without emitting :attr:`selectionChanged`."""
        self._selected = [ch for ch in channels if ch in self._ch_names]
        self._refresh()

    def set_bads(self, channels: Iterable[str]) -> None:
        """Draw *channels* as bad."""
        self._bads = {ch for ch in channels if ch in self._ch_names}
        self._refresh()

    def toggle(self, ch: str) -> bool:
        """Select or deselect *ch* as a click would; return whether it changed."""
        if ch not in self._ch_names:
            return False
        if ch in self._selected:
            if len(self._selected) <= self._min_selected:
                return False
            self._selected.remove(ch)
        else:
            if self._max_selected is not None and len(self._selected) >= self._max_selected:
                return False
            self._selected.append(ch)
        self._refresh()
        self.selectionChanged.emit(self.selected)
        return True

    def _refresh(self) -> None:
        spots = []
        for ch, (xn, yn) in zip(self._ch_names, self._positions):
            # Into the circle, then flip: pyqtgraph's y grows upwards, yn=0 is frontal
            tx = 0.5 + (xn - 0.5) * 0.9
            ty = 1.0 - (0.5 + (yn - 0.5) * 0.9)
            selected = ch in self._selected
            color = self._accent if selected else self._outline
            if ch in self._bads:
                brush, pen = pg.mkBrush(None), pg.mkPen(self._bad, width=1.5)
            else:
                brush, pen = pg.mkBrush(color), pg.mkPen(None)
            spots.append(
                {
                    "pos": (tx, ty),
                    "data": ch,
                    "brush": brush,
                    "pen": pen,
                    "size": 10 if selected else 6,
                }
            )
        self._scatter.setData(spots=spots)

    def _on_click(self, *args) -> None:
        # pyqtgraph passes (scatter, points, event)
        points = args[1] if len(args) >= 2 else args[0]
        event = args[2] if len(args) >= 3 else None
        right = event is not None and event.button() == Qt.MouseButton.RightButton
        for pt in points[:1]:  # overlapping dots: the topmost only
            ch = pt.data()
            if ch is None:
                continue
            if right:
                self.channelRightClicked.emit(ch)
            else:
                self.toggle(ch)
