"""The brain display's vertices, without pyvista or Qt.

The acquisition loop maps source values onto them for the display, and must
import without the display's dependencies.
"""

from __future__ import annotations

import numpy as np

#: Vertices per hemisphere of the ico-5 cortex the display draws.
N_ICO5 = 10242


def to_display(values: np.ndarray, vertices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """One value per source, as ``(lh, rh)`` arrays over the display's vertices.

    ``vertices`` are the source estimate's vertex numbers per hemisphere; a
    display vertex without a source is NaN, drawn without activity.
    """
    n_lh = len(vertices[0])
    out = []
    for vertno, part in zip(vertices, (values[:n_lh], values[n_lh:])):
        full = np.full(N_ICO5, np.nan)
        full[vertno] = part
        out.append(full)
    return out[0], out[1]
