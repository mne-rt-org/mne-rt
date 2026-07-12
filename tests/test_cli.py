"""Tests for the mne-rt command-line interface (parser only, no I/O)."""

import pytest

from mne_rt.cli import _build_parser, _build_protocol_from_args
from mne_rt.protocols import ThresholdProtocol, ZScoreProtocol


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
    assert args.duration == 120.0
    assert "sensor_power" in args.modality
    assert args.winsize == 1.0
    assert args.no_nf is False
    assert args.no_raw is False
    assert args.no_brain is False


def test_demo_custom(parser):
    args = parser.parse_args(
        [
            "demo",
            "--duration",
            "30",
            "--modality",
            "sensor_power",
            "band_ratio",
            "--winsize",
            "2.0",
            "--no-nf",
        ]
    )
    assert args.duration == 30.0
    assert args.modality == ["sensor_power", "band_ratio"]
    assert args.winsize == 2.0
    assert args.no_nf is True


def test_demo_protocol_defaults(parser):
    args = parser.parse_args(["demo"])
    assert args.protocol is None
    assert not hasattr(args, "threshold")
    assert args.threshold_direction == "up"
    assert not hasattr(args, "zscore_threshold")
    assert not hasattr(args, "zscore_warmup")
    assert not hasattr(args, "zscore_min_std")


def test_demo_protocol_threshold_custom(parser):
    args = parser.parse_args(
        [
            "demo",
            "--protocol",
            "threshold",
            "--threshold",
            "0.5",
            "--threshold-direction",
            "down",
        ]
    )
    assert args.protocol == "threshold"
    assert args.threshold == 0.5
    assert args.threshold_direction == "down"


def test_demo_protocol_zscore_custom(parser):
    args = parser.parse_args(
        [
            "demo",
            "--protocol",
            "zscore",
            "--zscore-threshold",
            "1.5",
            "--zscore-warmup",
            "10",
            "--zscore-min-std",
            "1e-15",
            "--threshold-direction",
            "down",
        ]
    )
    assert args.protocol == "zscore"
    assert args.zscore_threshold == 1.5
    assert args.zscore_warmup == 10
    assert args.zscore_min_std == 1e-15
    assert args.threshold_direction == "down"


def test_demo_invalid_protocol_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--protocol", "staircase"])


def test_demo_invalid_threshold_direction_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--threshold-direction", "sideways"])


def test_baseline_required_args(parser):
    args = parser.parse_args(
        [
            "baseline",
            "--subject",
            "sub01",
            "--subjects-dir",
            "/tmp",
        ]
    )
    assert args.command == "baseline"
    assert args.subject == "sub01"
    assert args.subjects_dir == "/tmp"
    assert args.duration == 120.0
    assert args.mock is False


def test_baseline_missing_subject_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["baseline", "--subjects-dir", "/tmp"])


def test_run_required_args(parser):
    args = parser.parse_args(
        [
            "run",
            "--subject",
            "sub01",
            "--subjects-dir",
            "/tmp",
            "--duration",
            "300",
        ]
    )
    assert args.command == "run"
    assert args.duration == 300.0


def test_run_missing_duration_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--subject",
                "sub01",
                "--subjects-dir",
                "/tmp",
            ]
        )


def test_run_osc_args(parser):
    args = parser.parse_args(
        [
            "run",
            "--subject",
            "sub01",
            "--subjects-dir",
            "/tmp",
            "--duration",
            "60",
            "--osc-host",
            "192.168.1.10",
            "--osc-port",
            "9001",
            "--osc-prefix",
            "/nf",
        ]
    )
    assert args.osc_host == "192.168.1.10"
    assert args.osc_port == 9001
    assert args.osc_prefix == "/nf"


def test_run_artifact_correction(parser):
    for method in ("lms", "orica", "gedai"):
        args = parser.parse_args(
            [
                "run",
                "--subject",
                "sub01",
                "--subjects-dir",
                "/tmp",
                "--duration",
                "60",
                "--artifact-correction",
                method,
            ]
        )
        assert args.artifact_correction == method


def test_run_protocol_defaults(parser):
    args = parser.parse_args(
        [
            "run",
            "--subject",
            "sub01",
            "--subjects-dir",
            "/tmp",
            "--duration",
            "60",
        ]
    )
    assert args.protocol is None
    assert not hasattr(args, "threshold")
    assert args.threshold_direction == "up"
    assert not hasattr(args, "zscore_threshold")
    assert not hasattr(args, "zscore_warmup")
    assert not hasattr(args, "zscore_min_std")


def test_run_protocol_threshold_custom(parser):
    args = parser.parse_args(
        [
            "run",
            "--subject",
            "sub01",
            "--subjects-dir",
            "/tmp",
            "--duration",
            "60",
            "--protocol",
            "threshold",
            "--threshold",
            "0.5",
            "--threshold-direction",
            "down",
        ]
    )
    assert args.protocol == "threshold"
    assert args.threshold == 0.5
    assert args.threshold_direction == "down"


def test_run_protocol_zscore_custom(parser):
    args = parser.parse_args(
        [
            "run",
            "--subject",
            "sub01",
            "--subjects-dir",
            "/tmp",
            "--duration",
            "60",
            "--protocol",
            "zscore",
            "--zscore-threshold",
            "1.5",
            "--zscore-warmup",
            "10",
            "--zscore-min-std",
            "1e-15",
        ]
    )
    assert args.protocol == "zscore"
    assert args.zscore_threshold == 1.5
    assert args.zscore_warmup == 10
    assert args.zscore_min_std == 1e-15


def test_run_invalid_protocol_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--subject",
                "sub01",
                "--subjects-dir",
                "/tmp",
                "--duration",
                "60",
                "--protocol",
                "staircase",
            ]
        )


def test_run_invalid_threshold_direction_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--subject",
                "sub01",
                "--subjects-dir",
                "/tmp",
                "--duration",
                "60",
                "--threshold-direction",
                "sideways",
            ]
        )


def test_build_protocol_from_args_none(parser):
    args = parser.parse_args(["demo"])
    assert _build_protocol_from_args(args) is None


def test_build_protocol_from_args_threshold(parser, capsys):
    args = parser.parse_args(
        ["demo", "--protocol", "threshold", "--threshold", "0.3", "--threshold-direction", "down"]
    )
    protocol = _build_protocol_from_args(args)
    assert isinstance(protocol, ThresholdProtocol)
    assert protocol.threshold == 0.3
    assert protocol.direction == "down"
    assert "ThresholdProtocol" in capsys.readouterr().out


def test_build_protocol_from_args_zscore(parser, capsys):
    args = parser.parse_args(
        [
            "demo",
            "--protocol",
            "zscore",
            "--zscore-threshold",
            "1.2",
            "--zscore-warmup",
            "15",
            "--zscore-min-std",
            "1e-15",
            "--threshold-direction",
            "down",
        ]
    )
    protocol = _build_protocol_from_args(args)
    assert isinstance(protocol, ZScoreProtocol)
    assert protocol.zscore_threshold == 1.2
    assert protocol.warmup_windows == 15
    assert protocol.min_std == 1e-15
    assert protocol.direction == "down"
    assert "ZScoreProtocol" in capsys.readouterr().out


def test_build_protocol_from_args_zscore_default_min_std(parser):
    args = parser.parse_args(["demo", "--zscore-threshold", "1.2"])
    protocol = _build_protocol_from_args(args)
    assert protocol.min_std == 1e-6


def test_build_protocol_from_args_threshold_inferred(parser, capsys):
    args = parser.parse_args(["demo", "--threshold", "2e-13", "--threshold-direction", "up"])
    assert args.protocol is None
    protocol = _build_protocol_from_args(args)
    assert isinstance(protocol, ThresholdProtocol)
    assert protocol.threshold == 2e-13
    assert protocol.direction == "up"
    assert "ThresholdProtocol" in capsys.readouterr().out


def test_build_protocol_from_args_zscore_inferred(parser, capsys):
    args = parser.parse_args(["demo", "--zscore-threshold", "1.2"])
    assert args.protocol is None
    protocol = _build_protocol_from_args(args)
    assert isinstance(protocol, ZScoreProtocol)
    assert protocol.zscore_threshold == 1.2
    assert "ZScoreProtocol" in capsys.readouterr().out


def test_build_protocol_from_args_ambiguous_raises(parser):
    args = parser.parse_args(["demo", "--threshold", "0.3", "--zscore-threshold", "1.0"])
    with pytest.raises(SystemExit):
        _build_protocol_from_args(args)


def test_build_protocol_from_args_zscore_min_std_alone_infers_protocol(parser, capsys):
    args = parser.parse_args(["demo", "--zscore-min-std", "1e-15"])
    assert args.protocol is None
    protocol = _build_protocol_from_args(args)
    assert isinstance(protocol, ZScoreProtocol)
    assert protocol.min_std == 1e-15
    assert "ZScoreProtocol" in capsys.readouterr().out


def test_surf_choices(parser):
    for surf in ("inflated", "pial", "white", "sphere"):
        args = parser.parse_args(["demo", "--surf", surf])
        assert args.surf == surf


def test_invalid_surf_raises(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--surf", "banana"])
