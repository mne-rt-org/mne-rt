"""Tests for NFRealtime instantiation and validation (no LSL required)."""

import tempfile
import pytest


@pytest.fixture()
def subjects_dir(tmp_path):
    return str(tmp_path)


def test_valid_instantiation(subjects_dir):
    from ant import NFRealtime
    nf = NFRealtime(
        subject_id="sub01",
        visit=1,
        session="main",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
        data_type="eeg",
    )
    assert nf.subject_id == "sub01"
    assert nf.visit == 1
    assert nf.session == "main"


def test_invalid_subject_id(subjects_dir):
    from ant import NFRealtime
    with pytest.raises(ValueError, match="subject_id"):
        NFRealtime(
            subject_id="",
            visit=1, session="main",
            subjects_dir=subjects_dir, montage="easycap-M1",
        )


def test_invalid_visit(subjects_dir):
    from ant import NFRealtime
    with pytest.raises(ValueError, match="visit"):
        NFRealtime(
            subject_id="sub01",
            visit=0, session="main",
            subjects_dir=subjects_dir, montage="easycap-M1",
        )


def test_invalid_session(subjects_dir):
    from ant import NFRealtime
    with pytest.raises(ValueError, match="session"):
        NFRealtime(
            subject_id="sub01",
            visit=1, session="unknown",
            subjects_dir=subjects_dir, montage="easycap-M1",
        )


def test_invalid_data_type(subjects_dir):
    from ant import NFRealtime
    with pytest.raises(ValueError, match="data_type"):
        NFRealtime(
            subject_id="sub01",
            visit=1, session="main",
            subjects_dir=subjects_dir, montage="easycap-M1",
            data_type="fnirs",
        )


def test_invalid_artifact_correction(subjects_dir):
    from ant import NFRealtime
    with pytest.raises(ValueError, match="artifact_correction"):
        NFRealtime(
            subject_id="sub01",
            visit=1, session="main",
            subjects_dir=subjects_dir, montage="easycap-M1",
            artifact_correction="invalid_method",
        )


def test_subjects_dir_stored(tmp_path):
    from ant import NFRealtime
    nf = NFRealtime(
        subject_id="newsub",
        visit=1, session="baseline",
        subjects_dir=str(tmp_path), montage="easycap-M1",
    )
    from pathlib import Path
    assert Path(nf.subjects_dir) == tmp_path


def test_modality_params_setter(subjects_dir):
    from ant import NFRealtime
    nf = NFRealtime(
        subject_id="sub01", visit=1, session="main",
        subjects_dir=subjects_dir, montage="easycap-M1",
    )
    nf.modality_params = {"sensor_power": {"frange": [8, 12]}}
    assert nf.modality_params["sensor_power"]["frange"] == [8, 12]


def test_modality_params_invalid(subjects_dir):
    from ant import NFRealtime
    nf = NFRealtime(
        subject_id="sub01", visit=1, session="main",
        subjects_dir=subjects_dir, montage="easycap-M1",
    )
    with pytest.raises(ValueError):
        nf.modality_params = "not_a_dict"
