"""Logging and verbosity utilities for MNE-RT.

Integrates with MNE's logging infrastructure so that setting ``verbose``
on any MNE-RT function also silences or expands MNE's own output consistently.

Usage
-----
Apply ``@verbose`` to any public function or method that accepts a
``verbose`` keyword argument::

    from mne_rt._logging import verbose, logger

    @verbose
    def my_function(a, b, verbose=None):
        logger.info("Computing …")
        ...

The ``verbose`` parameter accepts the same values as ``mne.set_log_level``:

* ``None``    — leave current level unchanged
* ``True``    — ``"INFO"``
* ``False``   — ``"WARNING"``
* ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``
* An integer MNE/Python logging level
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Union

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("ant")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[ANT] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)

logger.propagate = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LEVEL_MAP: dict = {
    None: None,
    True: logging.INFO,
    False: logging.WARNING,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def set_log_level(verbose: Union[bool, str, int, None]) -> None:
    """Set the ANT (and MNE) logging level.

    Parameters
    ----------
    verbose : bool | str | int | None
        Desired verbosity level.

        * ``None``    — no change
        * ``True``    — ``"INFO"``
        * ``False``   — ``"WARNING"``
        * ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``
        * An integer :mod:`logging` level

    See Also
    --------
    mne.set_log_level : Set MNE's own logging level.
    """
    if verbose is None:
        return

    if isinstance(verbose, str):
        verbose = verbose.upper()

    level = _LEVEL_MAP.get(verbose, verbose)
    if not isinstance(level, int):
        raise ValueError(
            f"verbose must be None, bool, an int, or one of "
            f"'DEBUG','INFO','WARNING','ERROR','CRITICAL'; got {verbose!r}."
        )
    logger.setLevel(level)

    # Mirror to MNE so MNE-internal calls respect the same level
    try:
        import mne

        mne_level = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
        }.get(level, "WARNING")
        mne.set_log_level(mne_level, add_frames=0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def verbose(func):
    """Decorator that applies the ``verbose`` kwarg to ANT and MNE logging.

    Wrap any function that has a ``verbose=None`` parameter::

        @verbose
        def compute(data, verbose=None):
            ...

    The decorator sets the ANT log level for the duration of the call,
    then restores the previous level.

    Parameters
    ----------
    func : callable
        Function to wrap. Must accept ``verbose`` as a keyword (or positional)
        argument.

    Returns
    -------
    wrapper : callable
    """
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    verbose_idx = param_names.index("verbose") if "verbose" in param_names else None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Resolve verbose value from kwargs or positional args
        v = kwargs.get("verbose", None)
        if v is None and verbose_idx is not None and verbose_idx < len(args):
            v = args[verbose_idx]

        if v is None:
            return func(*args, **kwargs)

        old_level = logger.level
        try:
            set_log_level(v)
            return func(*args, **kwargs)
        finally:
            logger.setLevel(old_level)

    return wrapper
