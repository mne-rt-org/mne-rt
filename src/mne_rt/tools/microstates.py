"""Real-time EEG microstate backfitting and per-state metrics.

Each sample of a window is assigned to the template map with the highest
*absolute* spatial correlation — polarity is ignored, as is standard in
microstate analysis — and per-state metrics are then computed from the labels.

The backfitting rule and the metric definitions follow pycrostates
(``ModKMeans.predict`` with ``factor=0``, and
``Segmentation.compute_parameters``), so a value computed here means the same
as one computed offline with pycrostates. pycrostates itself is not required;
a fitted ``ModKMeans`` can nonetheless be passed as the templates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Per-state metrics, named as in pycrostates.
METRICS = ("timecov", "meandurs", "occurrences", "gev", "mean_corr")


def resolve_templates(
    templates,
    ch_names: Sequence[str] | None = None,
    state_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return ``(centers, ch_names, state_names)`` from an array or a fitted model.

    Parameters
    ----------
    templates : array-like | pycrostates ModKMeans
        Template maps, shape ``(n_states, n_channels)`` — or
        ``(n_channels, n_states)``, which is transposed — or a fitted pycrostates
        ``ModKMeans``, whose ``cluster_centers_``, channel names and
        ``cluster_names`` are used.
    ch_names : sequence of str | None
        Channel name of each template column. Required for an array; ignored for
        a fitted model, which carries its own.
    state_names : sequence of str | None
        Name of each state. Defaults to ``"0"``, ``"1"``, ... for an array; taken
        from ``cluster_names`` for a fitted model.

    Returns
    -------
    centers : ndarray, shape (n_states, n_channels)
    ch_names : list of str
    state_names : list of str
    """
    if hasattr(templates, "cluster_centers_") and hasattr(templates, "info"):
        centers = np.asarray(templates.cluster_centers_, dtype=float)
        ch_names = list(templates.info["ch_names"])
        names = getattr(templates, "cluster_names", None)
        state_names = list(names) if names is not None else None
    else:
        if templates is None:
            raise ValueError(
                "microstates needs template maps: pass `templates` (an "
                "(n_states, n_channels) array or a fitted pycrostates ModKMeans) "
                "through `modality_params`."
            )
        centers = np.asarray(templates, dtype=float)
        if centers.ndim != 2:
            raise ValueError(f"templates must be 2-D, got shape {centers.shape}.")
        if ch_names is None:
            raise ValueError(
                "microstates needs `template_ch_names` — the channel of each "
                "template column — to align the maps with the stream. It is only "
                "optional when `templates` is a fitted pycrostates ModKMeans."
            )
        ch_names = list(ch_names)
        if centers.shape[1] != len(ch_names):
            if centers.shape[0] == len(ch_names):
                centers = centers.T
            else:
                raise ValueError(
                    f"templates have shape {centers.shape}, which does not match "
                    f"{len(ch_names)} template_ch_names in either orientation."
                )
    if len(set(ch_names)) != len(ch_names):
        raise ValueError("template_ch_names contains duplicates.")
    if state_names is None:
        state_names = [str(i) for i in range(centers.shape[0])]
    state_names = [str(s) for s in state_names]
    if len(state_names) != centers.shape[0]:
        raise ValueError(
            f"{len(state_names)} state_names given for {centers.shape[0]} template maps."
        )
    return centers, ch_names, state_names


def zscore_templates(centers: np.ndarray) -> np.ndarray:
    """Z-score each template map across channels, as backfitting expects."""
    centered = centers - centers.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True)
    if np.any(std == 0):
        raise ValueError("A template map is constant across channels and cannot be fitted.")
    return centered / std


def backfit(data: np.ndarray, templates_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label each sample with the template of highest absolute spatial correlation.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        EEG window, already filtered. It is average-referenced here, per sample.
    templates_z : ndarray, shape (n_states, n_channels)
        Template maps z-scored across channels (see :func:`zscore_templates`),
        in the same channel order as ``data``.

    Returns
    -------
    labels : ndarray of int, shape (n_samples,)
    corr : ndarray, shape (n_states, n_samples)
        Pearson spatial correlation of every template with every sample.
    gfp : ndarray, shape (n_samples,)
        Global field power: the standard deviation across channels.
    """
    x = data - data.mean(axis=0, keepdims=True)
    gfp = x.std(axis=0)
    # A sample with no spatial variance (a null map) correlates with nothing;
    # dividing by 1 leaves it at zero instead of producing NaN.
    corr = templates_z @ (x / np.where(gfp == 0, 1.0, gfp)) / x.shape[0]
    labels = np.argmax(np.abs(corr), axis=0)
    return labels, corr, gfp


def reject_edge_segments(labels: np.ndarray) -> np.ndarray:
    """Mark the first and the last segment as unlabelled (``-1``).

    This is pycrostates' ``reject_edges`` rule. Edge segments are truncated, so
    their durations are not real. Note that a window consisting of a single
    segment is rejected entirely, again as in pycrostates; that is rare over a
    whole recording but not over a short window.
    """
    labels = np.array(labels, copy=True)
    first = (labels != labels[0]).argmax()
    labels[:first] = -1
    last = np.flip(labels != labels[-1]).argmax()
    labels[-last:] = -1  # `last == 0` means one segment: `labels[0:]`, all of it
    return labels


def microstate_parameters(
    labels: np.ndarray,
    corr: np.ndarray,
    gfp: np.ndarray,
    sfreq: float,
    state: int,
) -> dict[str, float]:
    """Metrics of one state, with pycrostates' definitions.

    Unlabelled samples (``-1``) are left out of the time-based denominators but
    kept in the GEV denominator, as in pycrostates.

    Returns
    -------
    dict
        ``timecov`` — fraction of labelled time in the state;
        ``meandurs`` — mean segment duration (s);
        ``occurrences`` — segments per second of labelled time;
        ``gev`` — global explained variance contributed by the state;
        ``mean_corr`` — mean absolute correlation of its samples with its map.
        All are ``0.0`` when the state does not occur.
    """
    mask = labels == state
    n_labelled = int(np.count_nonzero(labels != -1))
    if not mask.any() or n_labelled == 0:
        return dict.fromkeys(METRICS, 0.0)
    r = corr[state, mask]
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    lengths = edges[1::2] - edges[::2]  # contiguous runs of the state
    return {
        "timecov": float(lengths.sum() / n_labelled),
        "meandurs": float(lengths.mean() / sfreq),
        "occurrences": float(lengths.size / n_labelled * sfreq),
        "gev": float(np.sum((gfp[mask] * r) ** 2) / np.sum(gfp**2)),
        "mean_corr": float(np.mean(np.abs(r))),
    }
