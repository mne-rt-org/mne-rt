"""Tests for save_as_bids (minimal BIDS backend)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import mne

from mne_rt.tools.bids_io import save_as_bids


# ------------------------------------------------------------------
# Fixture: minimal raw + nf_data
# ------------------------------------------------------------------

@pytest.fixture
def minimal_raw():
    sfreq = 256.0
    n_ch = 4
    n_s = int(sfreq * 2)
    data = np.zeros((n_ch, n_s))
    ch_names = ["Fz", "Cz", "Pz", "Oz"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture
def nf_data():
    return {
        "alpha_power": [0.1, 0.2, 0.3],
        "theta_power": [0.05, 0.06, 0.07],
    }


# ------------------------------------------------------------------
# Files created
# ------------------------------------------------------------------

def test_creates_expected_files(tmp_path, minimal_raw, nf_data):
    save_as_bids(
        minimal_raw, nf_data, tmp_path, subject="01",
        session="nf", task="neurofeedback"
    )
    assert (tmp_path / "dataset_description.json").exists()
    assert (tmp_path / "participants.tsv").exists()
    # Minimal backend: eeg .fif file
    fif = list(tmp_path.rglob("*_eeg.fif"))
    assert len(fif) == 1
    # Behavioural TSV
    tsv = list(tmp_path.rglob("*_beh.tsv"))
    assert len(tsv) == 1


def test_no_session_level(tmp_path, minimal_raw, nf_data):
    """When session=None, no ses- directory should appear."""
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="02", session=None)
    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir()]
    assert not any(d.startswith("ses-") for d in dirs)


def test_with_session_level(tmp_path, minimal_raw, nf_data):
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="03", session="01")
    ses_dirs = list(tmp_path.rglob("ses-01"))
    assert len(ses_dirs) == 1


# ------------------------------------------------------------------
# TSV content
# ------------------------------------------------------------------

def test_beh_tsv_columns(tmp_path, minimal_raw, nf_data):
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="04", session="nf")
    tsv_path = next(tmp_path.rglob("*_beh.tsv"))
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
    assert set(header) == {"alpha_power", "theta_power"}


def test_beh_tsv_row_count(tmp_path, minimal_raw, nf_data):
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="05", session="nf")
    tsv_path = next(tmp_path.rglob("*_beh.tsv"))
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    # 1 header + 3 data rows
    assert len(rows) == 4


def test_beh_tsv_na_for_missing_values(tmp_path, minimal_raw):
    """Shorter column should get n/a for missing rows."""
    nf = {"alpha_power": [0.1, 0.2, 0.3], "beta_power": [0.9]}  # different lengths
    save_as_bids(minimal_raw, nf, tmp_path, subject="06", session="nf")
    tsv_path = next(tmp_path.rglob("*_beh.tsv"))
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
    assert rows[1]["beta_power"] == "n/a"
    assert rows[2]["beta_power"] == "n/a"


# ------------------------------------------------------------------
# dataset_description.json
# ------------------------------------------------------------------

def test_dataset_description_content(tmp_path, minimal_raw, nf_data):
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="07")
    desc = json.loads((tmp_path / "dataset_description.json").read_text())
    assert desc["BIDSVersion"] == "1.9.0"
    assert any(g["Name"] == "MNE-RT" for g in desc["GeneratedBy"])


# ------------------------------------------------------------------
# participants.tsv
# ------------------------------------------------------------------

def test_participants_tsv_updated(tmp_path, minimal_raw, nf_data):
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="08")
    tsv_path = tmp_path / "participants.tsv"
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        ids = [r["participant_id"] for r in reader]
    assert "sub-08" in ids


def test_participants_tsv_no_duplicate(tmp_path, minimal_raw, nf_data):
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="09", overwrite=True)
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="09", overwrite=True)
    tsv_path = tmp_path / "participants.tsv"
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        ids = [r["participant_id"] for r in reader]
    assert ids.count("sub-09") == 1


# ------------------------------------------------------------------
# overwrite=False raises
# ------------------------------------------------------------------

def test_overwrite_false_raises(tmp_path, minimal_raw, nf_data):
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="10", session="nf")
    with pytest.raises(FileExistsError):
        save_as_bids(
            minimal_raw, nf_data, tmp_path, subject="10",
            session="nf", overwrite=False
        )


def test_overwrite_true_succeeds(tmp_path, minimal_raw, nf_data):
    save_as_bids(minimal_raw, nf_data, tmp_path, subject="11", session="nf")
    # Should not raise
    save_as_bids(
        minimal_raw, nf_data, tmp_path, subject="11",
        session="nf", overwrite=True
    )


# ------------------------------------------------------------------
# Return value
# ------------------------------------------------------------------

def test_returns_output_dir_path(tmp_path, minimal_raw, nf_data):
    result = save_as_bids(minimal_raw, nf_data, tmp_path, subject="12")
    assert isinstance(result, Path)
    assert result == tmp_path


# ------------------------------------------------------------------
# empty nf_data
# ------------------------------------------------------------------

def test_empty_nf_data_skips_tsv(tmp_path, minimal_raw):
    save_as_bids(minimal_raw, {}, tmp_path, subject="13", session="nf")
    tsv_files = list(tmp_path.rglob("*_beh.tsv"))
    assert len(tsv_files) == 0
