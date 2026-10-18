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
    assert hasattr(mne_rt, "AdaptiveMultibandGEDAIDenoiser")
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
    from mne_rt.tools import ORICA, AdaptiveMultibandGEDAIDenoiser, GEDAIDenoiser


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


def test_gedai_is_optional(monkeypatch):
    """`gedai` is an opt-in extra under a noncommercial licence, so mne-rt must
    import without it and only fail, clearly, when the feature is used.

    Simulated by setting the module's recorded import error, which is what the
    module does when the import fails; this runs whether or not gedai is
    installed.
    """
    import mne_rt
    from mne_rt.tools import adaptive_gedai

    assert hasattr(mne_rt, "AdaptiveMultibandGEDAIDenoiser")
    monkeypatch.setattr(adaptive_gedai, "_GEDAI_IMPORT_ERROR", ImportError("no gedai"))
    with pytest.raises(ImportError, match=r"mne-rt\[gedai\]"):
        adaptive_gedai.AdaptiveMultibandGEDAIDenoiser(n_channels=4, sfreq=250.0)


def test_gedai_not_a_core_or_full_dependency():
    """Installing mne-rt, or mne-rt[full], must not pull in gedai."""
    import pathlib
    import tomllib

    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    assert not any("gedai" in dep for dep in project["dependencies"])
    assert not any("gedai" in dep for dep in project["optional-dependencies"]["full"])
    assert project["optional-dependencies"]["gedai"]
