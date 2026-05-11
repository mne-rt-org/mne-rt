"""Command-line interface for the Advanced Neurofeedback Toolbox (ANT).

Usage
-----
::

    ant --help
    ant --version
    ANT info
    ANT demo [options]
    ANT baseline [options]
    ANT run [options]

Install note
------------
After ``pip install -e .`` add the entry-point to ``pyproject.toml``::

    [project.scripts]
    ant = "ant.cli:main"

Then ``ant`` becomes a shell command.
"""
from __future__ import annotations

import argparse
import sys
import textwrap


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ant",
        description=textwrap.dedent("""\
            Advanced Neurofeedback Toolbox (ANT)
            ─────────────────────────────────────
            Real-time M/EEG neurofeedback for research and clinical use.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use 'ant <command> --help' for per-command options.",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print the installed ANT version and exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging verbosity level (default: WARNING).",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    _add_info_parser(subparsers)
    _add_demo_parser(subparsers)
    _add_baseline_parser(subparsers)
    _add_run_parser(subparsers)

    return parser


# ---------------------------------------------------------------------------
# Sub-command parsers
# ---------------------------------------------------------------------------

def _add_info_parser(sub):
    p = sub.add_parser(
        "info",
        help="Display system and dependency information.",
        description="Print ANT version, Python version, and key dependency versions.",
    )
    return p


def _add_demo_parser(sub):
    p = sub.add_parser(
        "demo",
        help="Launch a demo NF session from simulated EEG data.",
        description=textwrap.dedent("""\
            Run a full demo neurofeedback session using simulated EEG.
            No amplifier or file is required.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--duration", type=float, default=120.0,
        help="Session duration in seconds (default: 120).",
    )
    p.add_argument(
        "--modality", nargs="+",
        default=["sensor_power", "band_ratio", "entropy", "hjorth"],
        metavar="MODALITY",
        help=(
            "NF modality(ies) to demonstrate.  "
            "Available: sensor_power, band_ratio, entropy, hjorth, "
            "sensor_connectivity, erd_ers, laterality, spectral_centroid, cfc_sensor.  "
            "(default: sensor_power band_ratio entropy hjorth)"
        ),
    )
    p.add_argument(
        "--winsize", type=float, default=1.0,
        help="Analysis window length in seconds (default: 1.0).",
    )
    p.add_argument(
        "--no-signal", action="store_true",
        help="Disable the NF signal plot window.",
    )
    p.add_argument(
        "--no-raw", action="store_true",
        help="Disable the raw stream viewer.",
    )
    p.add_argument(
        "--subjects-fs-dir", metavar="DIR",
        help=(
            "FreeSurfer subjects directory (must contain fsaverage5).  "
            "Auto-detected from FREESURFER_HOME/subjects if not given."
        ),
    )
    p.add_argument(
        "--no-brain", action="store_true",
        help="Disable the 3D brain activation display even if FreeSurfer is found.",
    )
    p.add_argument(
        "--surf",
        choices=["inflated", "pial", "white", "sphere"],
        default="inflated",
        help="Cortical surface geometry for brain display (default: inflated).",
    )
    return p


def _add_baseline_parser(sub):
    p = sub.add_parser(
        "baseline",
        help="Record a resting-state baseline session.",
        description=textwrap.dedent("""\
            Connect to a live LSL stream (or simulate one) and record a
            baseline segment.  The inverse operator is computed and saved.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_session_args(p)
    p.add_argument(
        "--duration", type=float, default=120.0,
        help="Baseline duration in seconds (default: 120).",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use simulated data instead of a live LSL stream.",
    )
    p.add_argument(
        "--fname", metavar="FILE",
        help="Any MNE-readable file to simulate (.fif, .vhdr, .edf, .bdf, .set, …) — requires --mock.",
    )
    return p


def _add_run_parser(sub):
    p = sub.add_parser(
        "run",
        help="Run a closed-loop neurofeedback main session.",
        description=textwrap.dedent("""\
            Connects to an LSL stream, extracts real-time NF features,
            and drives all configured visualisation windows.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_session_args(p)
    p.add_argument(
        "--duration", type=float, required=True,
        help="Session duration in seconds.",
    )
    p.add_argument(
        "--modality", nargs="+",
        default=["sensor_power"],
        metavar="MODALITY",
        help="NF modality(ies) to extract (default: sensor_power).",
    )
    p.add_argument(
        "--winsize", type=float, default=1.0,
        help="Analysis window length in seconds (default: 1.0).",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use simulated data instead of a live LSL stream.",
    )
    p.add_argument(
        "--fname", metavar="FILE",
        help="Any MNE-readable file to simulate (.fif, .vhdr, .edf, .bdf, .set, …) — requires --mock.",
    )
    p.add_argument(
        "--artifact-correction",
        choices=["lms", "orica", "gedai", "asr", "maxwell"],
        default=None,
        help=(
            "Real-time artifact correction method (default: none). "
            "lms/orica/gedai/asr: EEG/MEG. maxwell: MEG only (SSS/tSSS)."
        ),
    )
    p.add_argument(
        "--no-signal", action="store_true",
        help="Disable the NF signal plot window.",
    )
    p.add_argument(
        "--no-raw", action="store_true",
        help="Disable the raw stream viewer.",
    )
    p.add_argument(
        "--brain", action="store_true",
        help="Show the 3D brain activation display.",
    )
    p.add_argument(
        "--surf",
        choices=["inflated", "pial", "white", "sphere"],
        default="inflated",
        help="Brain surface geometry (default: inflated).",
    )
    p.add_argument(
        "--ring-buffer", action="store_true",
        help="Use sliding ring-buffer acquisition (50%% overlap).",
    )
    p.add_argument(
        "--osc-host", metavar="HOST", default=None,
        help="Enable OSC output and send to this host (e.g. 127.0.0.1).",
    )
    p.add_argument(
        "--osc-port", type=int, default=9000, metavar="PORT",
        help="OSC destination port (default: 9000).",
    )
    p.add_argument(
        "--osc-prefix", metavar="PREFIX", default="/ant",
        help="OSC address prefix (default: /ant).",
    )
    return p


def _add_common_session_args(p: argparse.ArgumentParser) -> None:
    """Add subject/visit/session args shared by baseline and run sub-commands."""
    p.add_argument(
        "--subject", required=True, metavar="ID",
        help="Subject identifier string.",
    )
    p.add_argument(
        "--visit", type=int, default=1, metavar="N",
        help="Visit number (default: 1).",
    )
    p.add_argument(
        "--subjects-dir", required=True, metavar="DIR",
        help="Root directory containing one folder per subject.",
    )
    p.add_argument(
        "--montage", default="easycap-M1", metavar="NAME",
        help="EEG montage name or .bvct file path (default: easycap-M1).",
    )
    p.add_argument(
        "--data-type", choices=["eeg", "meg"], default="eeg",
        help="Recording modality (default: eeg).",
    )
    p.add_argument(
        "--subjects-fs-dir", metavar="DIR",
        help="FreeSurfer subjects directory (required for source modalities).",
    )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_info(args) -> None:
    """Print version and dependency information."""
    import platform

    lines = [
        "Advanced Neurofeedback Toolbox (ANT)",
        "─" * 40,
    ]
    try:
        from ant import __version__
        lines.append(f"  ANT version  : {__version__}")
    except Exception:
        lines.append("  ANT version  : unknown")

    lines.append(f"  Python       : {platform.python_version()}")

    for pkg in ["mne", "numpy", "scipy", "pyvista", "PyQt6", "pyqtgraph",
                "mne_lsl", "mne_connectivity", "mne_features"]:
        try:
            from importlib.metadata import version
            lines.append(f"  {pkg:<20}: {version(pkg)}")
        except Exception:
            lines.append(f"  {pkg:<20}: not installed")

    print("\n".join(lines))


def _cmd_demo(args) -> None:
    """Run a demo NF session from simulated EEG."""
    from ant import NFRealtime, set_log_level

    set_log_level(args.verbose)

    print("ANT Demo — simulating EEG …")
    import os
    import tempfile
    from pathlib import Path

    tmp = tempfile.mkdtemp(prefix="ant_demo_")
    subjects_dir = tmp

    # Resolve FreeSurfer subjects directory: explicit arg → env vars → known paths
    subjects_fs_dir = getattr(args, "subjects_fs_dir", None)
    if subjects_fs_dir is None:
        _candidates = []
        if os.environ.get("FREESURFER_HOME"):
            _candidates.append(Path(os.environ["FREESURFER_HOME"]) / "subjects")
        if os.environ.get("SUBJECTS_DIR"):
            _candidates.append(Path(os.environ["SUBJECTS_DIR"]))
        _candidates += [
            Path("/Applications/freesurfer/dev/subjects"),
            Path("/usr/local/freesurfer/subjects"),
        ]
        for _d in _candidates:
            if _d.is_dir() and (_d / "fsaverage5").is_dir():
                subjects_fs_dir = str(_d)
                break

    show_brain = (subjects_fs_dir is not None) and not getattr(args, "no_brain", False)
    if show_brain:
        print(f"Brain activation: using {subjects_fs_dir}")

    nf = NFRealtime(
        subject_id="demo",
        visit=1,
        session="main",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
        data_type="eeg",
        subjects_fs_dir=subjects_fs_dir if show_brain else None,
        verbose=args.verbose,
    )
    nf.connect_to_lsl(mock_lsl=True, verbose=args.verbose)

    needs_baseline = any(m in args.modality for m in ["erd_ers"]) or show_brain
    if needs_baseline:
        print("Recording brief baseline …")
        nf.record_baseline(baseline_duration=10, verbose=args.verbose)

    nf.record_main(
        duration=args.duration,
        modality=args.modality,
        winsize=args.winsize,
        show_nf_signal=not args.no_signal,
        show_raw_signal=not args.no_raw,
        show_brain_activation=show_brain,
        brain_surf=getattr(args, "surf", "pial"),
        verbose=args.verbose,
    )


def _cmd_baseline(args) -> None:
    """Record a baseline session."""
    from ant import NFRealtime, set_log_level
    set_log_level(args.verbose)

    nf = NFRealtime(
        subject_id=args.subject,
        visit=args.visit,
        session="baseline",
        subjects_dir=args.subjects_dir,
        montage=args.montage,
        data_type=args.data_type,
        subjects_fs_dir=args.subjects_fs_dir,
        verbose=args.verbose,
    )
    nf.connect_to_lsl(
        mock_lsl=args.mock,
        fname=getattr(args, "fname", None),
        verbose=args.verbose,
    )
    nf.record_baseline(baseline_duration=args.duration, verbose=args.verbose)
    print(f"Baseline complete.  Data saved to: {nf.subject_dir}")


def _cmd_run(args) -> None:
    """Run a main NF session."""
    from ant import NFRealtime, set_log_level
    set_log_level(args.verbose)

    artifact_correction = args.artifact_correction or False

    nf = NFRealtime(
        subject_id=args.subject,
        visit=args.visit,
        session="main",
        subjects_dir=args.subjects_dir,
        montage=args.montage,
        data_type=args.data_type,
        subjects_fs_dir=getattr(args, "subjects_fs_dir", None),
        artifact_correction=artifact_correction,
        verbose=args.verbose,
    )
    nf.connect_to_lsl(
        mock_lsl=args.mock,
        fname=getattr(args, "fname", None),
        verbose=args.verbose,
    )

    if artifact_correction == "gedai":
        print("Fitting GEDAI denoiser from baseline …")
        nf.record_baseline(baseline_duration=60, verbose=args.verbose)
        nf.fit_gedai()
    elif artifact_correction == "asr":
        print("Fitting ASR denoiser from baseline …")
        nf.record_baseline(baseline_duration=60, verbose=args.verbose)
        nf.fit_asr()
    elif artifact_correction == "maxwell":
        print("Computing SSS/tSSS Maxwell filter from sensor geometry …")
        nf.fit_maxwell()

    osc_sender = None
    if getattr(args, "osc_host", None):
        from ant.osc import OSCSender
        osc_sender = OSCSender(
            host=args.osc_host,
            port=args.osc_port,
            prefix=args.osc_prefix,
        )
        print(f"OSC output → {osc_sender.target}  prefix={osc_sender.prefix}")

    try:
        nf.record_main(
            duration=args.duration,
            modality=args.modality,
            winsize=args.winsize,
            show_nf_signal=not args.no_signal,
            show_raw_signal=not args.no_raw,
            show_brain_activation=args.brain,
            use_ring_buffer=args.ring_buffer,
            osc_sender=osc_sender,
            verbose=args.verbose,
        )
    finally:
        if osc_sender is not None:
            osc_sender.close()

    print(f"Session complete.  Data saved to: {nf.subject_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str | None
        Command-line arguments.  ``None`` uses ``sys.argv[1:]``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        try:
            from ant import __version__
            print(f"ANT {__version__}")
        except Exception:
            print("ANT (version unknown)")
        return

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "info":     _cmd_info,
        "demo":     _cmd_demo,
        "baseline": _cmd_baseline,
        "run":      _cmd_run,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
