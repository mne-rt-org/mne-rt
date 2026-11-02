"""Built-in montage names that work across the MNE versions we support.

MNE 1.13 renamed the FieldTrip-derived 10-05 montages (``"standard_1020"`` is
now ``"colin27_1020"``). There the old names warn and stop being listed by
:func:`mne.channels.get_builtin_montages`, and MNE 1.14 is to refuse them; the
new names do not exist before 1.13. A name is therefore translated to the one
this MNE has, whichever of the two a caller used.
"""

from __future__ import annotations

from typing import Optional

import mne

_RENAMED_IN_MNE_1_13 = {
    "standard_1005": "colin27_1005",
    "standard_1020": "colin27_1020",
    "standard_alphabetic": "colin27_alphabetic",
    "standard_postfixed": "colin27_postfixed",
    "standard_prefixed": "colin27_prefixed",
    "standard_primed": "colin27_primed",
}
_OLD_NAMES = {new: old for old, new in _RENAMED_IN_MNE_1_13.items()}


def builtin_montage(name: str) -> Optional[str]:
    """``name`` as this MNE calls it, or ``None`` if it is not a built-in montage."""
    available = set(mne.channels.get_builtin_montages())
    for candidate in (name, _RENAMED_IN_MNE_1_13.get(name), _OLD_NAMES.get(name)):
        if candidate in available:
            return candidate
    return None


def make_standard_montage(name: str):
    """:func:`mne.channels.make_standard_montage`, accepting either name."""
    return mne.channels.make_standard_montage(builtin_montage(name) or name)
