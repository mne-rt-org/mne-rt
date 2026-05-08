"""Verify that all public API symbols can be imported."""

import pytest


def test_top_level_imports():
    import ant
    assert hasattr(ant, "NFRealtime")
    assert hasattr(ant, "BrainPlot")
    assert hasattr(ant, "NFSignalPlot")
    assert hasattr(ant, "ORICA")
    assert hasattr(ant, "GEDAIDenoiser")
    assert hasattr(ant, "OSCSender")
    assert hasattr(ant, "set_log_level")
    assert hasattr(ant, "__version__")


def test_version_string():
    import ant
    assert isinstance(ant.__version__, str)
    assert len(ant.__version__) > 0


def test_set_log_level():
    from ant import set_log_level
    set_log_level("WARNING")
    set_log_level(False)
    set_log_level(True)


def test_submodule_imports():
    from ant.tools import ORICA, GEDAIDenoiser
    from ant.osc import OSCSender
    from ant._logging import logger, set_log_level


def test_cli_importable():
    from ant.cli import main, _build_parser
    parser = _build_parser()
    assert parser is not None


def test_realtime_nf_importable():
    from ant.realtime_nf import NFRealtime
    assert NFRealtime is not None
