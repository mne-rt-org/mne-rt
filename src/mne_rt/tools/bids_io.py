"""BIDS-format I/O utilities for MNE-RT session data.

Functions
---------
save_as_bids
    Save a raw EEG recording and NF feature time-series in BIDS layout.

Notes
-----
``mne_bids`` is used when installed (``pip install mne-bids``).  When it is
absent, a minimal BIDS-compliant layout is written manually: the raw recording
as ``.fif``, NF data as ``_beh.tsv``, and stub
``dataset_description.json`` / ``participants.tsv`` files.

References
----------
Appelhoff, S., et al. (2019). MNE-BIDS: Organizing electrophysiological data
into the BIDS format and facilitating their analysis.
Journal of Open Source Software, 4(44), 1896.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional, Union

import mne
import numpy as np

logger = logging.getLogger(__name__)


def save_as_bids(
    raw: mne.io.BaseRaw,
    nf_data: dict,
    output_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None,
    task: str = "neurofeedback",
    run: Optional[str] = None,
    overwrite: bool = False,
    verbose: Union[bool, str, None] = None,
) -> Path:
    """Save a neurofeedback session in BIDS format.

    Writes the raw EEG recording as a BIDS EEG dataset and exports the
    per-window NF feature time-series as a ``*_beh.tsv`` behavioural
    side-car file.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG (or MEG) recording to export.
    nf_data : dict
        Dictionary mapping modality name → list of per-window values
        (as returned by :meth:`~mne_rt.RTStream.record_main`).
    output_dir : str | Path
        Root BIDS directory (created if it does not exist).
    subject : str
        BIDS subject label (without the ``sub-`` prefix).
    session : str | None, default None
        BIDS session label (without the ``ses-`` prefix).  ``None`` omits
        the session level.
    task : str, default "neurofeedback"
        BIDS task label.
    run : str | None, default None
        BIDS run label (without the ``run-`` prefix).
    overwrite : bool, default False
        Overwrite existing files.
    verbose : bool | str | None, default None
        Verbosity passed to mne-bids.

    Returns
    -------
    bids_path : Path
        Root BIDS output directory.

    Raises
    ------
    FileExistsError
        If ``overwrite=False`` and output files already exist.

    Notes
    -----
    If ``mne_bids`` is installed (``pip install mne-bids``) it is used for
    writing the EEG data with full BIDS compliance.  Otherwise a minimal
    BIDS-like layout is written manually (EEG as ``.fif``, NF data as
    ``_beh.tsv``, and stub ``dataset_description.json`` /
    ``participants.tsv`` files).

    .. versionadded:: 1.0.0
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only the import picks the backend: an ImportError from inside mne-bids (a
    # missing writer, say) is the user's to see, not a cue to fall back.
    try:
        import mne_bids  # noqa: F401
    except ImportError:
        has_mne_bids = False
    else:
        has_mne_bids = True

    if has_mne_bids:
        _save_with_mne_bids(
            raw=raw,
            nf_data=nf_data,
            output_dir=output_dir,
            subject=subject,
            session=session,
            task=task,
            run=run,
            overwrite=overwrite,
            verbose=verbose,
        )
    else:
        logger.info(
            "mne-bids is not installed; falling back to minimal BIDS layout. "
            "Install with: pip install mne-bids"
        )
        # mne-bids writes these itself, and refuses a subject already listed.
        _write_dataset_description(output_dir, overwrite=overwrite)
        _update_participants_tsv(output_dir, subject=subject)
        _save_minimal_bids(
            raw=raw,
            nf_data=nf_data,
            output_dir=output_dir,
            subject=subject,
            session=session,
            task=task,
            run=run,
            overwrite=overwrite,
        )

    return output_dir


# mne-bids backend


def _save_with_mne_bids(
    raw: mne.io.BaseRaw,
    nf_data: dict,
    output_dir: Path,
    subject: str,
    session: Optional[str],
    task: str,
    run: Optional[str],
    overwrite: bool,
    verbose: Union[bool, str, None],
) -> None:
    """Write raw data via mne_bids.write_raw_bids and NF data as _beh.tsv."""
    import mne_bids

    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        root=output_dir,
        datatype="eeg",
    )
    # A recording made live is in memory, not in a file mne-bids could copy, so
    # it is converted; BIDS-EEG does not take FIF, and BrainVision is its usual
    # format.
    mne_bids.write_raw_bids(
        raw,
        bids_path=bids_path,
        allow_preload=True,
        format="BrainVision",
        overwrite=overwrite,
        verbose=verbose,
    )

    _write_nf_beh_tsv(
        nf_data=nf_data,
        output_dir=output_dir,
        subject=subject,
        session=session,
        task=task,
        run=run,
        overwrite=overwrite,
    )
    logger.info("Session saved via mne-bids to %s", output_dir)


# Minimal (manual) BIDS backend


def _save_minimal_bids(
    raw: mne.io.BaseRaw,
    nf_data: dict,
    output_dir: Path,
    subject: str,
    session: Optional[str],
    task: str,
    run: Optional[str],
    overwrite: bool,
) -> None:
    """Write raw as .fif and NF data as _beh.tsv in a BIDS folder tree."""
    entity_dir = _build_entity_dir(output_dir=output_dir, subject=subject, session=session)
    eeg_dir = entity_dir / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    stem = _build_bids_stem(subject=subject, session=session, task=task, run=run)
    fif_path = eeg_dir / f"{stem}_eeg.fif"
    if fif_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {fif_path}. Set overwrite=True to overwrite."
        )
    raw.save(fif_path, overwrite=overwrite, verbose=False)
    logger.info("Raw saved to %s", fif_path)

    _write_nf_beh_tsv(
        nf_data=nf_data,
        output_dir=output_dir,
        subject=subject,
        session=session,
        task=task,
        run=run,
        overwrite=overwrite,
    )


# Shared helpers


def _build_entity_dir(output_dir: Path, subject: str, session: Optional[str]) -> Path:
    parts = [f"sub-{subject}"]
    if session is not None:
        parts.append(f"ses-{session}")
    return output_dir.joinpath(*parts)


def _build_bids_stem(
    subject: str,
    session: Optional[str],
    task: str,
    run: Optional[str],
) -> str:
    parts = [f"sub-{subject}"]
    if session is not None:
        parts.append(f"ses-{session}")
    parts.append(f"task-{task}")
    if run is not None:
        parts.append(f"run-{run}")
    return "_".join(parts)


def _write_nf_beh_tsv(
    nf_data: dict,
    output_dir: Path,
    subject: str,
    session: Optional[str],
    task: str,
    run: Optional[str],
    overwrite: bool,
    windows: Optional[dict] = None,
    reward: Optional[dict] = None,
    snr: Optional[list] = None,
) -> None:
    """Write per-window NF feature values to a _beh.tsv side-car file."""
    entity_dir = _build_entity_dir(output_dir=output_dir, subject=subject, session=session)
    beh_dir = entity_dir / "beh"
    beh_dir.mkdir(parents=True, exist_ok=True)

    stem = _build_bids_stem(subject=subject, session=session, task=task, run=run)
    tsv_path = beh_dir / f"{stem}_beh.tsv"
    if tsv_path.exists() and not overwrite:
        raise FileExistsError(
            f"Behavioural TSV already exists: {tsv_path}. Set overwrite=True to overwrite."
        )

    if not nf_data:
        logger.info("nf_data is empty; skipping _beh.tsv")
        return

    write_nf_beh_tsv(
        tsv_path, nf_data=nf_data, windows=windows, reward=reward, snr=snr, sidecar=True
    )
    logger.info("NF behavioural data saved to %s", tsv_path)


#: Columns that carry a time in seconds. These need fixed-point formatting: at a
#: session-relative onset of ~1234 s, ``%.6g`` rounds to 10 ms, which is far too
#: coarse to align a window against a stimulus log.
_TIME_COLUMNS = ("onset", "duration")

#: Trial-level columns, written next to the timing they describe and ahead of
#: the features. Each appears only when the session recorded it, so a run with
#: no marker stream produces the same table as before.
_MARKER_COLUMNS = ("condition", "n_markers", "gated")


def _nf_columns(*, nf_data, windows=None, reward=None, snr=None):
    """Assemble the ordered column names and their per-window series.

    BIDS requires ``onset`` first and ``duration`` second, so the timing columns
    lead; features, rewards and SNR follow.
    """
    columns: list[str] = []
    combined: dict[str, list] = {}
    for name in _TIME_COLUMNS + _MARKER_COLUMNS:
        if windows and windows.get(name):
            columns.append(name)
            combined[name] = list(windows[name])
    for name, vals in nf_data.items():
        columns.append(name)
        combined[name] = list(vals)
    for name, vals in (reward or {}).items():
        if vals:
            columns.append(f"reward_{name}")
            combined[f"reward_{name}"] = list(vals)
    if snr:
        columns.append("snr_db")
        combined["snr_db"] = list(snr)
    return columns, combined


def write_nf_beh_tsv(
    tsv_path: Union[str, Path],
    *,
    nf_data: dict,
    windows: Optional[dict] = None,
    reward: Optional[dict] = None,
    snr: Optional[list] = None,
    sidecar: bool = False,
    meta: Optional[dict] = None,
) -> Path:
    """Write per-window neurofeedback values as a BIDS ``_beh.tsv``.

    The single writer for both :meth:`~mne_rt.RTStream.save` and
    :func:`save_as_bids`, which previously carried separate implementations that
    had already drifted apart on which columns they emit and how they format
    floats.

    Parameters
    ----------
    tsv_path : str | path-like
        Destination ``.tsv`` file.
    nf_data : dict
        ``{modality: [value per window]}``. Ragged lists are padded with ``n/a``.
    windows : dict | None
        Optional per-window timing, ``{"onset": [...], "duration": [...]}``.
        BIDS requires ``onset`` first and ``duration`` second, so these are
        emitted ahead of every feature column.
    reward : dict | None
        ``{modality: [magnitude per window]}``, written as ``reward_<modality>``.
        Empty series are skipped.
    snr : list | None
        Per-window SNR in dB, written as ``snr_db``.
    sidecar : bool, default False
        Also write the ``_beh.json`` sidecar describing the columns. BIDS wants
        one whenever column names are not part of the standard vocabulary,
        which for a neurofeedback file is all of them.
    meta : dict | None
        Session metadata for the sidecar (``sfreq_hz``, ``winsize_s`` …).

    Returns
    -------
    path : Path
        The file written.

    Notes
    -----
    Feature values are written to six significant figures, matching what the
    BIDS writer has always emitted. That is a readable table, not the archival
    record: the session JSON keeps full float precision, so prefer it when
    re-analysing. Time columns are fixed-point instead, because six significant
    figures on an onset of ~1e3 s would quantise it to 10 ms.
    """
    tsv_path = Path(tsv_path)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    columns, combined = _nf_columns(nf_data=nf_data, windows=windows, reward=reward, snr=snr)

    n_rows = max((len(v) for v in combined.values()), default=0)

    with open(tsv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(columns)
        for i in range(n_rows):
            row = []
            for col in columns:
                vals = combined[col]
                if i >= len(vals):
                    row.append("n/a")
                    continue
                val = vals[i]
                if val is None:
                    # BIDS spells a missing value "n/a"; str(None) would write
                    # the literal word "None" into the table.
                    row.append("n/a")
                    continue
                if isinstance(val, (int, float, np.floating, np.integer)):
                    fmt = "%.6f" if col in _TIME_COLUMNS else "%.6g"
                    row.append(fmt % float(val))
                else:
                    row.append(str(val))
            writer.writerow(row)

    if sidecar:
        _write_beh_sidecar(tsv_path.with_suffix(".json"), columns=columns, meta=meta or {})
    return tsv_path


def describe_nf_columns(
    *,
    nf_data: dict,
    windows: Optional[dict] = None,
    reward: Optional[dict] = None,
    snr: Optional[list] = None,
    meta: Optional[dict] = None,
) -> dict:
    """Describe the ``_beh.tsv`` columns, in BIDS sidecar form.

    Exposed separately because :meth:`~mne_rt.RTStream.save` cannot write a
    standalone sidecar: BIDS would name it ``<stem>_beh.json``, which is already
    the session payload's filename. That payload embeds this dict instead.
    """
    columns = _nf_columns(nf_data=nf_data, windows=windows, reward=reward, snr=snr)[0]
    return _column_descriptions(columns, meta or {})


def _write_beh_sidecar(path: Path, *, columns: list, meta: dict) -> None:
    """Write the ``_beh.json`` sidecar describing each TSV column."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_column_descriptions(columns, meta), fh, indent=2)


def _column_descriptions(columns: list, meta: dict) -> dict:
    """Describe each ``_beh.tsv`` column, as BIDS asks for non-standard names."""
    descriptions: dict[str, dict] = {}
    for col in columns:
        if col == "onset":
            descriptions[col] = {
                "LongName": "Analysis window onset",
                "Description": (
                    "Onset of the analysis window, relative to the first window of "
                    "the run. Absolute onsets on the LSL clock are in the session "
                    "JSON, for alignment against a stimulus log."
                ),
                "Units": "s",
            }
        elif col == "duration":
            descriptions[col] = {
                "LongName": "Analysis window duration",
                "Description": "Length of the analysis window.",
                "Units": "s",
            }
        elif col == "condition":
            descriptions[col] = {
                "LongName": "Experimental condition",
                "Description": (
                    "Label of the most recent marker at or before the end of this "
                    "window, from the marker stream's marker_id map. Latched, so it "
                    "is defined for every window after the first marker."
                ),
            }
            _levels = {k: k for k in sorted(meta.get("marker_id", {}) or {})}
            if _levels:
                # Omitted rather than written as null: a reader doing
                # ``Levels.items()`` would get an AttributeError, and codes with
                # no label are recorded as the code itself and never appear here.
                descriptions[col]["Levels"] = _levels
        elif col == "n_markers":
            descriptions[col] = {
                "LongName": "Markers received for this window",
                "Description": (
                    "Number of markers that arrived since the previous window ended. "
                    "Windows overlap by half, so counting markers *inside* each window "
                    "would count most of them twice; this partitions the run, and the "
                    "column sums to the number of markers received."
                ),
            }
        elif col == "gated":
            descriptions[col] = {
                "LongName": "Feedback withheld",
                "Description": (
                    "1 when the window fell outside gate_conditions, so no protocol "
                    "was evaluated and nothing was sent to the stimulus program. The "
                    "feature value is still recorded."
                ),
                "Levels": {"0": "feedback delivered", "1": "feedback withheld"},
            }
        elif col == "snr_db":
            descriptions[col] = {
                "LongName": "Signal-to-noise ratio",
                "Description": "In-band power over out-of-band power for the window.",
                "Units": "dB",
            }
        elif col.startswith("reward_"):
            descriptions[col] = {
                "LongName": f"Reward magnitude for {col[len('reward_') :]}",
                "Description": ("Protocol magnitude when the reward criterion was met, else 0.0."),
            }
        else:
            descriptions[col] = {
                "LongName": f"Neurofeedback value for {col}",
                "Description": (
                    "Feature value for this analysis window, after any z-scoring and smoothing."
                ),
                "Units": "z-score" if meta.get("zscore_normalize") else "arbitrary",
            }
    for key in ("sfreq_hz", "winsize_s", "hop_s"):
        if meta.get(key) is not None:
            descriptions[key] = meta[key]
    return descriptions


def _write_dataset_description(output_dir: Path, overwrite: bool) -> None:
    """Write a minimal dataset_description.json if it does not exist."""
    desc_path = output_dir / "dataset_description.json"
    if desc_path.exists() and not overwrite:
        return
    payload = {
        "Name": "MNE-RT Dataset",
        "BIDSVersion": "1.9.0",
        "GeneratedBy": [{"Name": "MNE-RT", "Version": "1.0.0"}],
        "DatasetType": "raw",
    }
    with open(desc_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("dataset_description.json written to %s", desc_path)


def _update_participants_tsv(output_dir: Path, subject: str) -> None:
    """Create or append to participants.tsv."""
    tsv_path = output_dir / "participants.tsv"
    sub_label = f"sub-{subject}"

    if tsv_path.exists():
        with open(tsv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)
            fieldnames = reader.fieldnames or ["participant_id"]
        existing_ids = {r.get("participant_id") for r in rows}
        if sub_label in existing_ids:
            return
        rows.append({"participant_id": sub_label})
    else:
        fieldnames = ["participant_id"]
        rows = [{"participant_id": sub_label}]

    with open(tsv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("participants.tsv updated at %s", tsv_path)
