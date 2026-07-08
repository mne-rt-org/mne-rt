"""Verify that all public API symbols can be imported."""

import pytest


def test_top_level_imports():
    import mne_rt

    assert hasattr(mne_rt, "RTStream")
    assert hasattr(mne_rt, "RTEpochs")
    assert hasattr(mne_rt, "BrainPlot")
    assert hasattr(mne_rt, "RawPlot")
    assert hasattr(mne_rt, "ORICA")
    assert hasattr(mne_rt, "GEDAIDenoiser")
    assert hasattr(mne_rt, "OSCSender")
    assert hasattr(mne_rt, "set_log_level")
    assert hasattr(mne_rt, "__version__")


def test_version_string():
    import mne_rt

    assert isinstance(mne_rt.__version__, str)
    assert len(mne_rt.__version__) > 0


def test_set_log_level():
    from mne_rt import set_log_level

    set_log_level("WARNING")
    set_log_level(False)
    set_log_level(True)


def test_submodule_imports():
    from mne_rt._logging import logger, set_log_level
    from mne_rt.osc import OSCSender
    from mne_rt.tools import ORICA, GEDAIDenoiser


def test_cli_importable():
    from mne_rt.cli import _build_parser, main

    parser = _build_parser()
    assert parser is not None


def test_rtstream_importable():
    from mne_rt import RTStream

    assert RTStream is not None


def test_rtepochs_importable():
    from mne_rt import RTEpochs

    assert RTEpochs is not None
