"""Tests for mne_rt._logging (verbosity utilities)."""

from __future__ import annotations

import logging

import pytest

from mne_rt._logging import logger, set_log_level, verbose

# ------------------------------------------------------------------
# set_log_level
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_log_level():
    """Prevent tests from leaking their log-level change to other tests."""
    old_level = logger.level
    yield
    logger.setLevel(old_level)


def test_set_log_level_none_is_noop():
    logger.setLevel(logging.WARNING)
    set_log_level(None)
    assert logger.level == logging.WARNING


def test_set_log_level_true_is_info():
    set_log_level(True)
    assert logger.level == logging.INFO


def test_set_log_level_false_is_warning():
    set_log_level(False)
    assert logger.level == logging.WARNING


@pytest.mark.parametrize(
    "name,expected",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_set_log_level_string_levels(name, expected):
    set_log_level(name)
    assert logger.level == expected


def test_set_log_level_lowercase_string():
    set_log_level("debug")
    assert logger.level == logging.DEBUG


def test_set_log_level_int():
    set_log_level(logging.ERROR)
    assert logger.level == logging.ERROR


def test_set_log_level_invalid_raises():
    with pytest.raises(ValueError, match="verbose must be"):
        set_log_level("not_a_level")


# ------------------------------------------------------------------
# verbose decorator
# ------------------------------------------------------------------


def test_verbose_sets_level_for_call_duration():
    logger.setLevel(logging.WARNING)
    levels_seen = []

    @verbose
    def func(verbose=None):
        levels_seen.append(logger.level)

    func(verbose="DEBUG")
    assert levels_seen == [logging.DEBUG]
    # Restored after the call
    assert logger.level == logging.WARNING


def test_verbose_none_leaves_level_unchanged():
    logger.setLevel(logging.ERROR)

    @verbose
    def func(verbose=None):
        return logger.level

    level_inside = func(verbose=None)
    assert level_inside == logging.ERROR
    assert logger.level == logging.ERROR


def test_verbose_positional_argument():
    levels_seen = []

    @verbose
    def func(a, verbose=None):
        levels_seen.append(logger.level)
        return a

    result = func(42, "INFO")
    assert result == 42
    assert levels_seen == [logging.INFO]


def test_verbose_restores_level_after_exception():
    logger.setLevel(logging.WARNING)

    @verbose
    def func(verbose=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        func(verbose="DEBUG")
    assert logger.level == logging.WARNING


def test_verbose_preserves_function_return_value():
    @verbose
    def func(a, b, verbose=None):
        return a + b

    assert func(2, 3, verbose="INFO") == 5


def test_verbose_preserves_wrapped_metadata():
    @verbose
    def my_func(verbose=None):
        """Docstring."""

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "Docstring."
