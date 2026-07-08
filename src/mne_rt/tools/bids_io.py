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

    _write_dataset_description(output_dir, overwrite=overwrite)
    _update_participants_tsv(output_dir, subject=subject)

    try:
        import mne_bids  # noqa: F401

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
    except ImportError:
        logger.info(
            "mne-bids is not installed; falling back to minimal BIDS layout. "
            "Install with: pip install mne-bids"
        )
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


# ---------------------------------------------------------------------------
# mne-bids backend
# ---------------------------------------------------------------------------


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
    mne_bids.write_raw_bids(
        raw,
        bids_path=bids_path,
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


# ---------------------------------------------------------------------------
# Minimal (manual) BIDS backend
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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

    columns = list(nf_data.keys())
    n_rows = max(len(v) for v in nf_data.values())

    with open(tsv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(columns)
        for i in range(n_rows):
            row = []
            for col in columns:
                vals = nf_data[col]
                val = vals[i] if i < len(vals) else "n/a"
                row.append(f"{val:.6g}" if isinstance(val, (int, float, np.floating)) else str(val))
            writer.writerow(row)

    logger.info("NF behavioural data saved to %s", tsv_path)


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
