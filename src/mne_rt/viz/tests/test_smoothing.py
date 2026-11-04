"""Display smoothing shared by the viewers."""

from __future__ import annotations

import numpy as np


def test_brain_display_smoothing_survives_a_nan():
    from mne_rt.viz._smoothing import ema_step

    ema = np.array([1.0, np.nan, 1.0])
    out = ema_step(ema, np.array([np.nan, 2.0, 2.0]), 0.5)
    np.testing.assert_allclose(out, [1.0, 2.0, 1.5])
