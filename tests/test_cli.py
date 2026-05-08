"""Tests for the ANT command-line interface (parser only, no I/O)."""

import pytest
from ant.cli import _build_parser


@pytest.fixture()
def parser():
    return _build_parser()


def test_version_flag(parser):
    args = parser.parse_args(["--version"])
    assert args.version is True


def test_info_subcommand(parser):
    args = parser.parse_args(["info"])
    assert args.command == "info"


def test_demo_defaults(parser):
    args = parser.parse_args(["demo"])
    assert args.command == "demo"
    assert args.duration == 60.0
    assert args.modality == ["sensor_power"]
    assert args.winsize == 1.0
    assert args.no_signal is False
    assert args.no_raw is False
    assert args.brain is False


def test_demo_custom(parser):
    args = parser.parse_args([
        "demo",
        "--duration", "30",
        "--modality", "sensor_power", "band_ratio",
        "--winsize", "2.0",
        "--no-signal",
    ])
    assert args.duration == 30.0
    assert args.modality == ["sensor_power", "band_ratio"]
    assert args.winsize == 2.0
    assert args.no_signal is True


def test_baseline_required_args(parser):
    args = parser.parse_args([
        "baseline",
        "--subject", "sub01",
        "--subjects-dir", "/tmp",
    ])
    assert args.command == "baseline"
    assert args.subject == "sub01"
    assert args.subjects_dir == "/tmp"
    assert args.duration == 120.0
    assert args.mock is False


def test_baseline_missing_subject_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["baseline", "--subjects-dir", "/tmp"])


def test_run_required_args(parser):
    args = parser.parse_args([
        "run",
        "--subject", "sub01",
        "--subjects-dir", "/tmp",
        "--duration", "300",
    ])
    assert args.command == "run"
    assert args.duration == 300.0


def test_run_missing_duration_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args([
            "run",
            "--subject", "sub01",
            "--subjects-dir", "/tmp",
        ])


def test_run_osc_args(parser):
    args = parser.parse_args([
        "run",
        "--subject", "sub01",
        "--subjects-dir", "/tmp",
        "--duration", "60",
        "--osc-host", "192.168.1.10",
        "--osc-port", "9001",
        "--osc-prefix", "/nf",
    ])
    assert args.osc_host == "192.168.1.10"
    assert args.osc_port == 9001
    assert args.osc_prefix == "/nf"


def test_run_artifact_correction(parser):
    for method in ("lms", "orica", "gedai"):
        args = parser.parse_args([
            "run",
            "--subject", "sub01",
            "--subjects-dir", "/tmp",
            "--duration", "60",
            "--artifact-correction", method,
        ])
        assert args.artifact_correction == method


def test_surf_choices(parser):
    for surf in ("inflated", "pial", "white", "sphere"):
        args = parser.parse_args(["demo", "--surf", surf])
        assert args.surf == surf


def test_invalid_surf_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--surf", "banana"])
