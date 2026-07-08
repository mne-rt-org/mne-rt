"""Command-line interface for MNE-RT.

Usage
-----
::

    mne-rt --help
    mne-rt --version
    mne-rt info
    mne-rt demo     [options]
    mne-rt baseline [options]
    mne-rt run      [options]

Install note
------------
After ``pip install -e .`` the entry-point in ``pyproject.toml`` exposes
the ``mne-rt`` shell command::

    [project.scripts]
    mne-rt = "mne_rt.cli:main"
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mne-rt",
        description=textwrap.dedent("""\
            MNE-RT
            -------------------------------------
            Real-time M/EEG signal processing and analysis.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use 'mne-rt <command> --help' for per-command options.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed MNE-RT version and exit.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging verbosity level (default: WARNING).",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    _add_info_parser(subparsers)
    _add_demo_parser(subparsers)
    _add_demo_erp_parser(subparsers)
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
        description="Print MNE-RT version, Python version, and key dependency versions.",
    )
    return p


def _add_demo_parser(sub):
    p = sub.add_parser(
        "demo",
        help="Launch a demo real-time session from simulated EEG data.",
        description=textwrap.dedent("""\
            Run a full demo real-time session using simulated EEG.
            No amplifier or recording file is required.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Session duration in seconds (default: 120).",
    )
    p.add_argument(
        "--modality",
        nargs="+",
        default=["sensor_power", "band_ratio", "entropy", "hjorth"],
        metavar="MODALITY",
        help=(
            "Feature modality(ies) to demonstrate.  "
            "Available: sensor_power, band_ratio, entropy, hjorth, "
            "sensor_connectivity, erd_ers, laterality, spectral_centroid, "
            "cfc_sensor, scp, peak_alpha_freq, connectivity_ratio.  "
            "(default: sensor_power band_ratio entropy hjorth)"
        ),
    )
    p.add_argument(
        "--winsize",
        type=float,
        default=1.0,
        help="Analysis window length in seconds (default: 1.0).",
    )
    p.add_argument(
        "--no-nf",
        action="store_true",
        help="Disable the scrolling real-time NF signal plot (NFPlot).",
    )
    p.add_argument(
        "--no-raw",
        action="store_true",
        help="Disable the scrolling raw M/EEG viewer (RawPlot).",
    )
    p.add_argument(
        "--no-topomap",
        action="store_true",
        help="Disable the real-time scalp topomap display (TopomapPlot).",
    )
    p.add_argument(
        "--no-brain",
        action="store_true",
        help="Disable the 3-D brain activation display even if FreeSurfer is found.",
    )
    p.add_argument(
        "--subjects-fs-dir",
        metavar="DIR",
        help=(
            "FreeSurfer subjects directory (must contain fsaverage5).  "
            "Auto-detected from FREESURFER_HOME/subjects if not given."
        ),
    )
    p.add_argument(
        "--surf",
        choices=["inflated", "pial", "white", "sphere"],
        default="inflated",
        help="Cortical surface geometry for brain display (default: inflated).",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=0.25,
        metavar="ALPHA",
        help=(
            "EMA smoothing factor for feature values (default: 0.25). "
            "1.0 = no smoothing; 0.1 = heavy smoothing."
        ),
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving session data and report at the end of the demo.",
    )
    return p


def _add_demo_erp_parser(sub):
    p = sub.add_parser(
        "demo-erp",
        help="Launch an ERP demo: RTEpochs + epoch visualisation plots.",
        description=textwrap.dedent("""\
            Run a demo that streams MNE sample-dataset EEG through a mock LSL
            player, collects auditory epochs via RTEpochs, and drives five
            epoch visualisation plots (EpochPlot, TopoPlot, ButterflyPlot,
            CompareEvoked, TFRPlot).  Downloads MNE sample data on first run
            (~1.5 GB).
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=70,
        metavar="N",
        help="Number of EEG trials to collect (default: 70).",
    )
    p.add_argument(
        "--no-epoch-plot",
        action="store_true",
        help="Disable the EpochPlot scrolling raw viewer with trigger overlays.",
    )
    p.add_argument(
        "--no-topo",
        action="store_true",
        help="Disable the scalp-layout TopoPlot (ERP display).",
    )
    p.add_argument(
        "--no-butterfly",
        action="store_true",
        help="Disable the ButterflyPlot (all-channel overlay).",
    )
    p.add_argument(
        "--no-compare",
        action="store_true",
        help="Disable CompareEvoked (per-channel comparison with peak markers).",
    )
    p.add_argument(
        "--no-tfr",
        action="store_true",
        help="Disable the TFRPlot (Morlet wavelet time-frequency heatmaps).",
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
        "--duration",
        type=float,
        default=120.0,
        help="Baseline duration in seconds (default: 120).",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use simulated data instead of a live LSL stream.",
    )
    p.add_argument(
        "--fname",
        metavar="FILE",
        help="Any MNE-readable file to simulate (.fif, .vhdr, .edf, .bdf, .set, ...) - requires --mock.",
    )
    return p


def _add_run_parser(sub):
    p = sub.add_parser(
        "run",
        help="Run a real-time M/EEG main session.",
        description=textwrap.dedent("""\
            Connect to an LSL stream, extract real-time features,
            and drive all configured visualisation windows.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_session_args(p)
    p.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Session duration in seconds.",
    )
    p.add_argument(
        "--modality",
        nargs="+",
        default=["sensor_power"],
        metavar="MODALITY",
        help="Feature modality(ies) to extract (default: sensor_power).",
    )
    p.add_argument(
        "--winsize",
        type=float,
        default=1.0,
        help="Analysis window length in seconds (default: 1.0).",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use simulated data instead of a live LSL stream.",
    )
    p.add_argument(
        "--fname",
        metavar="FILE",
        help="Any MNE-readable file to simulate (.fif, .vhdr, .edf, .bdf, .set, ...) - requires --mock.",
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
        "--no-nf",
        action="store_true",
        help="Disable the scrolling real-time NF signal plot (NFPlot).",
    )
    p.add_argument(
        "--no-raw",
        action="store_true",
        help="Disable the scrolling raw M/EEG viewer (RawPlot).",
    )
    p.add_argument(
        "--topomap",
        action="store_true",
        help="Show the real-time scalp topomap display (TopomapPlot).",
    )
    p.add_argument(
        "--brain",
        action="store_true",
        help="Show the 3-D brain activation display.",
    )
    # ERP / epoch plot flags
    p.add_argument(
        "--topo",
        action="store_true",
        help="Enable the scalp-layout TopoPlot ERP display (requires --stim-ch).",
    )
    p.add_argument(
        "--epoch-plot",
        action="store_true",
        help="Enable the EpochPlot scrolling raw viewer with trigger overlays (requires --stim-ch).",
    )
    p.add_argument(
        "--butterfly",
        action="store_true",
        help="Enable the ButterflyPlot overlay (requires --stim-ch).",
    )
    p.add_argument(
        "--compare-evoked",
        action="store_true",
        help="Enable CompareEvoked per-channel plot (requires --stim-ch).",
    )
    p.add_argument(
        "--tfr",
        action="store_true",
        help="Enable the TFRPlot Morlet wavelet heatmap (requires --stim-ch).",
    )
    p.add_argument(
        "--stim-ch",
        metavar="CHANNEL",
        help="Stimulus/trigger channel name for epoch extraction (e.g. STI 014).",
    )
    p.add_argument(
        "--tmin",
        type=float,
        default=-0.1,
        metavar="SEC",
        help="Epoch start relative to stimulus in seconds (default: -0.1).",
    )
    p.add_argument(
        "--tmax",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Epoch end relative to stimulus in seconds (default: 0.5).",
    )
    p.add_argument(
        "--event-id",
        nargs="+",
        metavar="NAME=CODE",
        help=(
            "Event definitions as NAME=CODE pairs, e.g. "
            "--event-id left=1 right=2 (default: stimulus=1)."
        ),
    )
    p.add_argument(
        "--surf",
        choices=["inflated", "pial", "white", "sphere"],
        default="inflated",
        help="Brain surface geometry (default: inflated).",
    )
    p.add_argument(
        "--osc-host",
        metavar="HOST",
        default=None,
        help="Enable OSC output and send to this host (e.g. 127.0.0.1).",
    )
    p.add_argument(
        "--osc-port",
        type=int,
        default=9000,
        metavar="PORT",
        help="OSC destination port (default: 9000).",
    )
    p.add_argument(
        "--osc-prefix",
        metavar="PREFIX",
        default="/mne_rt",
        help="OSC address prefix (default: /mne_rt).",
    )
    p.add_argument(
        "--lsl-output",
        action="store_true",
        help=(
            "Broadcast feature values as an LSL stream outlet named 'MNE_RT'.  "
            "Any LSL-aware application (PsychoPy, Psychtoolbox, OpenViBE, ...) "
            "can subscribe to this stream.  Faster and more reliable than OSC "
            "for same-machine integration."
        ),
    )
    p.add_argument(
        "--lsl-stream-name",
        metavar="NAME",
        default="MNE_RT",
        help="LSL outlet stream name (default: MNE_RT).  Only used with --lsl-output.",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=0.25,
        metavar="ALPHA",
        help=(
            "EMA smoothing factor for feature values (default: 0.25). "
            "1.0 = no smoothing; 0.1 = heavy smoothing."
        ),
    )
    return p


def _add_common_session_args(p: argparse.ArgumentParser) -> None:
    """Add subject/session args shared by baseline and run sub-commands."""
    p.add_argument(
        "--subject",
        required=True,
        metavar="ID",
        help="Subject identifier (BIDS subject label, e.g. 'sub01').",
    )
    p.add_argument(
        "--session",
        default="01",
        metavar="LABEL",
        help="BIDS session label (e.g. '01', 'pre', 'week1'; default: '01').",
    )
    p.add_argument(
        "--subjects-dir",
        required=True,
        metavar="DIR",
        help="Root directory containing one folder per subject.",
    )
    p.add_argument(
        "--montage",
        default="easycap-M1",
        metavar="NAME",
        help="EEG montage name or .bvct file path (default: easycap-M1).",
    )
    p.add_argument(
        "--data-type",
        choices=["eeg", "meg"],
        default="eeg",
        help="Recording modality (default: eeg).",
    )
    p.add_argument(
        "--subjects-fs-dir",
        metavar="DIR",
        help="FreeSurfer subjects directory (required for source modalities).",
    )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_info(args) -> None:
    """Print version and dependency information."""
    import platform

    lines = [
        "MNE-RT - Real-Time M/EEG Analysis",
        "-" * 40,
    ]
    try:
        from mne_rt import __version__

        lines.append(f"  mne-rt version : {__version__}")
    except Exception:
        lines.append("  mne-rt version : unknown")

    lines.append(f"  Python         : {platform.python_version()}")

    for pkg in [
        "mne",
        "numpy",
        "scipy",
        "pyvista",
        "qtpy",
        "pyqtgraph",
        "mne_lsl",
        "mne_connectivity",
        "mne_features",
    ]:
        try:
            from importlib.metadata import version

            lines.append(f"  {pkg:<22}: {version(pkg)}")
        except Exception:
            lines.append(f"  {pkg:<22}: not installed")

    print("\n".join(lines))


def _cmd_demo(args) -> None:
    """Run a demo real-time session from simulated EEG."""
    from mne_rt import RTStream, set_log_level

    set_log_level(args.verbose)

    print("MNE-RT Demo - simulating EEG ...")
    import os
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="mne_rt_demo_"))
    subjects_dir = str(tmp)

    # Use the bundled pericalcarine simulation (loops automatically via n_repeat=inf)
    _pkg_root = Path(__file__).parent.parent.parent  # repo root
    fname_sim = _pkg_root / "data" / "simulated" / "pericalcarine-lh_10Hz_1-raw.fif"
    if not fname_sim.is_file():
        raise FileNotFoundError(
            f"Demo simulation file not found: {fname_sim}\n"
            "Re-run from the MNE-RT repository root or reinstall the package."
        )

    # Resolve FreeSurfer subjects directory: explicit arg -> env vars -> known paths
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

    show_brain = (subjects_fs_dir is not None) and not getattr(args, "no_brain", True)
    show_topo = not getattr(args, "no_topomap", False)
    do_save = not getattr(args, "no_save", True)
    if show_brain:
        print(f"Brain activation: using {subjects_fs_dir}")

    nf = RTStream(
        subject_id="demo",
        session="01",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
        data_type="eeg",
        subjects_fs_dir=subjects_fs_dir if show_brain else None,
        verbose=args.verbose,
    )
    nf.connect_to_lsl(mock_lsl=True, fname=str(fname_sim), verbose=args.verbose)

    # Always record a brief baseline so the report and ERD/ERS work
    print("Recording brief baseline (10 s) ...")
    nf.record_baseline(baseline_duration=10, verbose=args.verbose)

    nf.record_main(
        duration=args.duration,
        modality=args.modality,
        winsize=args.winsize,
        signal_smoothing=args.smoothing,
        show_nf_signal=not args.no_nf,
        show_raw_signal=not args.no_raw,
        show_topo=show_topo,
        show_brain_activation=show_brain,
        brain_surf=getattr(args, "surf", "pial"),
        save_raw=do_save,
        verbose=args.verbose,
    )

    if do_save:
        saved = nf.save()
        for kind, path in saved.items():
            print(f"  [{kind}] -> {path}")
        try:
            report_path = nf.create_report()
            print(f"  [report] -> {report_path}")
        except Exception as exc:
            print(f"  [report] skipped ({exc})")


def _cmd_demo_erp(args) -> None:
    """Run an ERP demo using MNE sample data and the five epoch plot windows."""
    import os
    import sys
    import threading
    import time

    os.environ.setdefault("MPLBACKEND", "Agg")

    import mne
    import numpy as np

    print("MNE-RT ERP Demo - downloading/loading MNE sample data ...")
    data_path = mne.datasets.sample.data_path()
    raw_full = mne.io.read_raw_fif(
        str(data_path) + "/MEG/sample/sample_audvis_raw.fif",
        preload=True,
        verbose=False,
    )
    raw_full.filter(1.0, 40.0, verbose=False)
    raw_demo = raw_full.copy().crop(tmax=277.0)

    mock_path = "/tmp/mne_rt_demo_erp_raw.fif"
    raw_demo.save(mock_path, overwrite=True, verbose=False)
    print(f"  Saved {raw_demo.times[-1]:.0f} s demo file -> {mock_path}")

    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)

    stim_ch = "STI 014"
    event_id = {"auditory/left": 1, "auditory/right": 2}
    tmin, tmax = -0.1, 0.4

    show_epoch_plot = not getattr(args, "no_epoch_plot", False)
    show_topo = not args.no_topo
    show_butterfly = not args.no_butterfly
    show_compare = not args.no_compare
    show_tfr = not args.no_tfr

    if not any([show_epoch_plot, show_topo, show_butterfly, show_compare, show_tfr]):
        print("All plots disabled - nothing to show.  Remove --no-* flags to enable plots.")
        return

    print("\n-- Connecting RTEpochs ----------------------------------------------")
    from mne_rt import RTEpochs

    rt = RTEpochs(
        event_id=event_id,
        event_channels=stim_ch,
        tmin=tmin,
        tmax=tmax,
        baseline=(None, 0),
        picks="eeg",
        reject={"eeg": 150e-6},
    )
    rt.connect_to_lsl(mock_lsl=True, fname=mock_path, timeout=15.0)

    from mne_rt.viz.butterfly_plot import ButterflyPlot
    from mne_rt.viz.compare_evoked import CompareEvoked
    from mne_rt.viz.epoch_plot import EpochPlot
    from mne_rt.viz.tfr_plot import TFRPlot
    from mne_rt.viz.topo_plot import TopoPlot as ERPTopoPlot

    ch_names = list(rt.epochs_stream_.info["ch_names"])
    sfreq = rt.epochs_stream_.info["sfreq"]

    common_kw = dict(
        ch_names=ch_names,
        sfreq=sfreq,
        tmin=tmin,
        tmax=tmax,
        event_id=event_id,
        info=rt.epochs_stream_.info,
        baseline=(None, 0),
    )

    # EpochPlot: scrolling raw viewer - epoch data is fed chunk-by-chunk
    # so the trigger marker lands at the correct t=0 position.
    # scale_uv=10 because ERP amplitudes are ~1-10 uV (default 100 is too coarse)
    epoch_w = (
        EpochPlot(
            ch_names=ch_names,
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            event_id=event_id,
            time_window=15.0,
            scale_uv=10.0,
        )
        if show_epoch_plot
        else None
    )

    topo_w = ERPTopoPlot(**common_kw) if show_topo else None
    butt_w = ButterflyPlot(**common_kw) if show_butterfly else None
    comp_w = CompareEvoked(**common_kw) if show_compare else None
    tfr_w = TFRPlot(**common_kw) if show_tfr else None

    _interactive = os.environ.get("QT_QPA_PLATFORM") != "offscreen"
    # EpochPlot gets a wider window at the top; the other four tile below it.
    if epoch_w is not None:
        if _interactive:
            epoch_w.move(0, 0)
            epoch_w.resize(1460, 440)
        epoch_w.show()

    positions = [(0, 460), (730, 460), (0, 980), (730, 980)]
    idx = 0
    for w in (topo_w, butt_w, comp_w, tfr_w):
        if w is not None:
            if _interactive:
                x, y = positions[idx]
                w.move(x, y)
                w.resize(720, 500)
            w.show()
            idx += 1
    app.processEvents()

    # Number of samples before the trigger within each epoch
    n_pre = int(round(abs(tmin) * sfreq))

    update_times: list[float] = []

    def on_trial(n_accepted: int, data, event_code: int, condition: str) -> None:
        t0 = time.perf_counter()
        batch = rt._buf_[:n_accepted]
        conds = list(rt._cond_list_)

        # Feed the epoch into EpochPlot split at t=0 so the trigger marker
        # lands at the correct position.  `data` is the epoch itself:
        # shape (n_ch, n_times) - NOT a batch, so no [-1] indexing.
        if epoch_w is not None:
            epoch_w.push(data[:, :n_pre])  # pre-trigger samples
            epoch_w.push_trigger(code=event_code)
            epoch_w.push(data[:, n_pre:])  # post-trigger samples

        if topo_w is not None:
            topo_w.update(batch, conds)
        if butt_w is not None:
            butt_w.update(batch, conds)
        if comp_w is not None:
            comp_w.update(batch, conds)
        if tfr_w is not None:
            tfr_w.update(batch, conds)
        dt = (time.perf_counter() - t0) * 1000
        update_times.append(dt)
        print(f"  trial {n_accepted:3d} | {condition:<22s} | update {dt:.1f} ms")
        app.processEvents()

    rt.on_trial = on_trial

    print(f"\n-- Collecting {args.n_trials} trials ---------------------------------------")
    done = threading.Event()

    def _collect():
        rt.run(n_trials=args.n_trials, show_erp=False)
        done.set()

    threading.Thread(target=_collect, daemon=True).start()

    all_windows = [w for w in (epoch_w, topo_w, butt_w, comp_w, tfr_w) if w is not None]
    sentinel = all_windows[0] if all_windows else None
    while not done.is_set():
        app.processEvents()
        time.sleep(0.02)
        if sentinel is not None and not sentinel.isVisible():
            print("\n  Window closed - stopping early.")
            rt.stop()
            break

    rt.disconnect()
    print(f"\n  Accepted: {rt.n_accepted_} trials")
    if update_times:
        print(f"  Latency : mean={np.mean(update_times):.1f} ms  max={np.max(update_times):.1f} ms")

    if any(w.isVisible() for w in all_windows):
        print("\nAll trials collected - close all windows to exit.")
        app.exec()
    else:
        print("\nDone.")


def _cmd_baseline(args) -> None:
    """Record a baseline session."""
    from mne_rt import RTStream, set_log_level

    set_log_level(args.verbose)

    nf = RTStream(
        subject_id=args.subject,
        session=args.session,
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
    """Run a real-time M/EEG session."""
    import sys
    import threading
    import time

    from mne_rt import RTStream, set_log_level

    set_log_level(args.verbose)

    want_epoch_plots = any(
        [
            getattr(args, "topo", False),
            getattr(args, "epoch_plot", False),
            getattr(args, "butterfly", False),
            getattr(args, "compare_evoked", False),
            getattr(args, "tfr", False),
        ]
    )

    if want_epoch_plots and not getattr(args, "stim_ch", None):
        import sys as _sys

        print(
            "ERROR: --topo / --epoch-plot / --butterfly / --compare-evoked / --tfr require "
            "--stim-ch <CHANNEL>  (e.g. --stim-ch 'STI 014')",
            file=_sys.stderr,
        )
        _sys.exit(1)

    artifact_correction = args.artifact_correction or False

    nf = RTStream(
        subject_id=args.subject,
        session=args.session,
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
        print("Fitting GEDAI denoiser from baseline ...")
        nf.record_baseline(baseline_duration=60, verbose=args.verbose)
        nf.fit_gedai()
    elif artifact_correction == "asr":
        print("Fitting ASR denoiser from baseline ...")
        nf.record_baseline(baseline_duration=60, verbose=args.verbose)
        nf.fit_asr()
    elif artifact_correction == "maxwell":
        print("Computing SSS/tSSS Maxwell filter from sensor geometry ...")
        nf.fit_maxwell()

    osc_sender = None
    if getattr(args, "osc_host", None):
        from mne_rt.osc import OSCSender

        osc_sender = OSCSender(
            host=args.osc_host,
            port=args.osc_port,
            prefix=args.osc_prefix,
        )
        print(f"OSC output -> {osc_sender.target}  prefix={osc_sender.prefix}")

    lsl_sender = None
    if getattr(args, "lsl_output", False):
        from mne_rt.lsl_output import LSLSender

        lsl_sender = LSLSender(stream_name=getattr(args, "lsl_stream_name", "MNE_RT"))
        print(f"LSL output -> stream '{lsl_sender.stream_name}'")

    # -- Epoch plot windows (RTEpochs side-channel) ------------------------
    rt_epochs = None
    epoch_w = topo_w = butt_w = comp_w = tfr_w = None

    if want_epoch_plots:
        # Parse --event-id name=code pairs; default to stimulus=1
        raw_event_id = getattr(args, "event_id", None) or ["stimulus=1"]
        event_id: dict[str, int] = {}
        for pair in raw_event_id:
            name, _, code = pair.partition("=")
            event_id[name.strip()] = int(code.strip())

        tmin = getattr(args, "tmin", -0.1)
        tmax = getattr(args, "tmax", 0.5)

        from mne_rt import RTEpochs

        rt_epochs = RTEpochs(
            event_id=event_id,
            event_channels=args.stim_ch,
            tmin=tmin,
            tmax=tmax,
            baseline=(None, 0),
        )
        rt_epochs.connect_to_lsl(
            mock_lsl=args.mock,
            fname=getattr(args, "fname", None),
            timeout=30.0,
        )

        from mne_rt.viz.butterfly_plot import ButterflyPlot
        from mne_rt.viz.compare_evoked import CompareEvoked
        from mne_rt.viz.epoch_plot import EpochPlot
        from mne_rt.viz.tfr_plot import TFRPlot
        from mne_rt.viz.topo_plot import TopoPlot as ERPTopoPlot

        _run_ch_names = list(rt_epochs.epochs_stream_.info["ch_names"])
        _run_sfreq = rt_epochs.epochs_stream_.info["sfreq"]

        common_kw = dict(
            ch_names=_run_ch_names,
            sfreq=_run_sfreq,
            tmin=tmin,
            tmax=tmax,
            event_id=event_id,
            info=rt_epochs.epochs_stream_.info,
            baseline=(None, 0),
        )

        epoch_w = (
            EpochPlot(
                ch_names=_run_ch_names,
                sfreq=_run_sfreq,
                tmin=tmin,
                tmax=tmax,
                event_id=event_id,
                time_window=15.0,
                scale_uv=10.0,
            )
            if getattr(args, "epoch_plot", False)
            else None
        )
        topo_w = ERPTopoPlot(**common_kw) if args.topo else None
        butt_w = ButterflyPlot(**common_kw) if args.butterfly else None
        comp_w = CompareEvoked(**common_kw) if args.compare_evoked else None
        tfr_w = TFRPlot(**common_kw) if args.tfr else None

        _interactive = os.environ.get("QT_QPA_PLATFORM") != "offscreen"
        if epoch_w is not None:
            if _interactive:
                epoch_w.move(0, 0)
                epoch_w.resize(1460, 440)
            epoch_w.show()
        positions = [(0, 460), (730, 460), (0, 980), (730, 980)]
        idx = 0
        for w in (topo_w, butt_w, comp_w, tfr_w):
            if w is not None:
                if _interactive:
                    x, y = positions[idx]
                    w.move(x, y)
                    w.resize(720, 500)
                w.show()
                idx += 1

        _n_pre = int(round(abs(tmin) * _run_sfreq))

        def _epoch_on_trial(n_accepted, data, event_code, condition):
            batch = rt_epochs._buf_[:n_accepted]
            conds = list(rt_epochs._cond_list_)
            if epoch_w is not None:
                # data is (n_ch, n_times) - the single epoch, not a batch
                epoch_w.push(data[:, :_n_pre])
                epoch_w.push_trigger(code=event_code)
                epoch_w.push(data[:, _n_pre:])
            if topo_w is not None:
                topo_w.update(batch, conds)
            if butt_w is not None:
                butt_w.update(batch, conds)
            if comp_w is not None:
                comp_w.update(batch, conds)
            if tfr_w is not None:
                tfr_w.update(batch, conds)

        rt_epochs.on_trial = _epoch_on_trial

        _epoch_done = threading.Event()

        def _collect_epochs():
            rt_epochs.run(n_trials=999_999, show_erp=False)
            _epoch_done.set()

        threading.Thread(target=_collect_epochs, daemon=True).start()
        print(f"Epoch plots active - stimulus channel: {args.stim_ch}")

    try:
        nf.record_main(
            duration=args.duration,
            modality=args.modality,
            winsize=args.winsize,
            signal_smoothing=args.smoothing,
            show_nf_signal=not args.no_nf,
            show_raw_signal=not args.no_raw,
            show_topo=args.topomap,
            show_brain_activation=args.brain,
            osc_sender=osc_sender,
            lsl_sender=lsl_sender,
            verbose=args.verbose,
        )
    finally:
        if rt_epochs is not None:
            rt_epochs.stop()
            rt_epochs.disconnect()
        if osc_sender is not None:
            osc_sender.close()
        if lsl_sender is not None:
            lsl_sender.close()

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
            from mne_rt import __version__

            print(f"mne-rt {__version__}")
        except Exception:
            print("mne-rt (version unknown)")
        return

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "info": _cmd_info,
        "demo": _cmd_demo,
        "demo-erp": _cmd_demo_erp,
        "baseline": _cmd_baseline,
        "run": _cmd_run,
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
