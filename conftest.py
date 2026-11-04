"""Test-session defaults, set before pytest imports anything from mne_rt.

At the repository root, not in the package: pytest loads it before
src/mne_rt/conftest.py, whose import runs mne_rt/__init__.py, which already
imports matplotlib and Qt. Set from there, these defaults came too late.
"""

import os

# The suite runs without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
