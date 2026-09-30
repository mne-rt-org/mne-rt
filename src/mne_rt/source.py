"""Source-space modelling: ROIs, head models, and cached linear operators.

This module turns a sensor-space data window into **ROI time courses** in one
matrix multiplication.

The naive route — :func:`~mne.beamformer.apply_lcmv_raw` followed by
:func:`~mne.extract_label_time_course` — is far too slow for a closed loop: on a
5 mm whole-brain volume source space (14 629 points) the label extraction alone
costs 650–800 ms per window, against a typical 500 ms hop. But with
``pick_ori="max-power"`` the beamformer is a fixed linear map, and
``mode="mean"`` / ``"mean_flip"`` label extraction is a fixed sparse average, so
the whole chain collapses to a single ``(n_roi, n_channels)`` matrix that can be
built once and applied with a ``~1 ms`` matmul::

    R = A @ W @ whitener        # built once, in _prep
    roi_tc = R @ data           # per window

:meth:`SourceModel.roi_kernel` builds ``R``; the result is numerically identical
to the MNE path (verified to better than 1e-10 relative error in the test
suite).

Regions of interest
-------------------
ROIs are named groups of atlas labels.  A single volumetric atlas —
FreeSurfer's ``aparc+aseg`` — carries both cortical parcels
(``ctx-lh-parsopercularis``) and subcortical structures (``Left-Hippocampus``),
so cortical and subcortical ROIs can live in one volume source space without
needing a mixed (surface + volume) model.

:data:`ROI_ALIASES` gives friendly names for a few common language ROIs; any
atlas label name is also accepted directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
from mne import read_labels_from_annot
from mne.label import label_sign_flip

from mne_rt._logging import logger

__all__ = [
    "ROI",
    "ROI_ALIASES",
    "SourceModel",
    "list_rois",
    "resolve_rois",
]


# ---------------------------------------------------------------------------
# ROI aliases
# ---------------------------------------------------------------------------

#: Friendly ROI names → the atlas labels they are composed of.
#:
#: These are *conventions*, not settled anatomy — the boundaries of "Broca's
#: area" in particular are contested.  Pass explicit atlas label names instead
#: if your protocol defines them differently.
ROI_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "aparc+aseg": {
        # Broca ≈ BA44 + BA45 (pars opercularis + pars triangularis)
        "Broca": ("ctx-lh-parsopercularis", "ctx-lh-parstriangularis"),
        "Broca-rh": ("ctx-rh-parsopercularis", "ctx-rh-parstriangularis"),
        # Wernicke ≈ posterior superior temporal / supramarginal
        "Wernicke": ("ctx-lh-superiortemporal", "ctx-lh-supramarginal"),
        "Wernicke-rh": ("ctx-rh-superiortemporal", "ctx-rh-supramarginal"),
        "Hippocampus-lh": ("Left-Hippocampus",),
        "Hippocampus-rh": ("Right-Hippocampus",),
    },
    "aparc": {
        "Broca": ("parsopercularis-lh", "parstriangularis-lh"),
        "Broca-rh": ("parsopercularis-rh", "parstriangularis-rh"),
        "Wernicke": ("superiortemporal-lh", "supramarginal-lh"),
        "Wernicke-rh": ("superiortemporal-rh", "supramarginal-rh"),
    },
}


@dataclass(frozen=True)
class ROI:
    """A named region of interest, possibly spanning several atlas labels.

    Attributes
    ----------
    name : str
        User-facing name, e.g. ``"Broca"``.
    members : tuple of str
        Atlas label names merged into this ROI.
    kind : {"volume", "surface"}
        Which kind of source space the labels are defined on.
    """

    name: str
    members: tuple[str, ...]
    kind: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<ROI {self.name!r} ({self.kind}, {len(self.members)} label(s))>"


# ---------------------------------------------------------------------------
# Atlas helpers
# ---------------------------------------------------------------------------


def _row_supports(rows: dict) -> dict:
    """Number of source points each label row actually draws on.

    Counting *source points* rather than atlas voxels matters for multi-label
    ROIs: a label whose voxels all fall outside the source space contributes an
    all-zero row, and weighting it by its voxel count would both suppress the
    "no source points" warning and scale down the labels that do contribute.
    """
    return {name: int(np.count_nonzero(row)) for name, row in rows.items()}


def _with_average_reference(info):
    """Return ``info`` with an average-EEG-reference projection attached.

    :meth:`~mne_rt.RTStream._prepare_raw_array` applies this projection to every
    analysis window, and :func:`~mne.minimum_norm.apply_inverse_raw` requires it
    outright. Building the inverse or the beamformer from an ``info`` without it
    makes MNE warn that the covariance is adversely affected, and leaves the
    whitener inconsistent with the data it is later applied to.

    Returns the input unchanged when there are no EEG channels, or when the
    projection is already present.
    """
    from mne.io import RawArray

    if not any(ch["kind"] == 2 for ch in info["chs"]):  # 2 = FIFFV_EEG_CH
        return info
    if any(proj["desc"].startswith("Average EEG") for proj in info["projs"]):
        return info
    raw = RawArray(np.zeros((info["nchan"], 1)), info.copy(), verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    return raw.info


def _is_volume_atlas(atlas: str) -> bool:
    """Volumetric atlases are ``.mgz`` volumes; surface ones are annot names."""
    return str(atlas).endswith((".mgz", ".mgh")) or "aseg" in str(atlas)


def _volume_atlas_path(atlas: str, subject: str, subjects_dir) -> Path:
    """Resolve a volumetric atlas name to a file under ``<subject>/mri/``."""
    atlas = str(atlas)
    if atlas.endswith((".mgz", ".mgh")) and Path(atlas).is_file():
        return Path(atlas)
    stem = atlas[:-4] if atlas.endswith((".mgz", ".mgh")) else atlas
    if subjects_dir is None:
        raise ValueError(
            f"`subjects_dir` is required to locate the volumetric atlas {atlas!r}. "
            "Pass subjects_fs_dir=... (or an absolute path to the .mgz)."
        )
    path = Path(subjects_dir) / subject / "mri" / f"{stem}.mgz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Volumetric atlas {stem!r} not found for subject {subject!r}: {path} "
            "does not exist. FreeSurfer ships 'aseg' and 'aparc+aseg'."
        )
    return path


def list_rois(*, atlas: str, subject: str = "fsaverage", subjects_dir=None) -> list[str]:
    """List every label name available in an atlas, plus any aliases.

    Parameters
    ----------
    atlas : str
        Volumetric atlas stem (e.g. ``"aparc+aseg"``) or surface annotation
        name (e.g. ``"aparc"``).
    subject : str
        FreeSurfer subject identifier.
    subjects_dir : path-like | None
        FreeSurfer subjects directory.

    Returns
    -------
    names : list of str
        Sorted label names.  Alias names from :data:`ROI_ALIASES` are listed
        first so they are easy to spot.
    """
    aliases = sorted(ROI_ALIASES.get(str(atlas), {}))
    return aliases + sorted(
        _atlas_label_names(atlas=atlas, subject=subject, subjects_dir=subjects_dir)
    )


def _atlas_label_names(*, atlas: str, subject: str, subjects_dir) -> list[str]:
    """The atlas's own label names, without the convenience aliases."""
    if _is_volume_atlas(atlas):
        import mne

        path = _volume_atlas_path(atlas, subject, subjects_dir)
        return list(mne.get_volume_labels_from_aseg(str(path)))
    return [
        label.name
        for label in read_labels_from_annot(
            subject=subject, parc=atlas, subjects_dir=subjects_dir, verbose=False
        )
    ]


def resolve_rois(
    names: Union[str, Sequence],
    *,
    atlas: str,
    subject: str = "fsaverage",
    subjects_dir=None,
) -> list[ROI]:
    """Resolve ROI names (or aliases, or explicit label groups) into :class:`ROI` objects.

    Parameters
    ----------
    names : str | sequence
        Each entry may be:

        * an alias from :data:`ROI_ALIASES` (e.g. ``"Broca"``),
        * a bare atlas label name (e.g. ``"Left-Hippocampus"``),
        * a ``{name: [label, ...]}`` mapping, or a ``(name, [label, ...])``
          pair, to define an ROI explicitly.
    atlas : str
        Volumetric atlas stem or surface annotation name.
    subject : str
        FreeSurfer subject identifier.
    subjects_dir : path-like | None
        FreeSurfer subjects directory.

    Returns
    -------
    rois : list of ROI

    Raises
    ------
    ValueError
        If a label name is not present in the atlas.  The message lists close
        matches to make typos easy to spot.
    """
    if isinstance(names, (str, dict)):
        names = [names]

    kind = "volume" if _is_volume_atlas(atlas) else "surface"
    # Validate against the atlas's own labels, not list_rois() — that also
    # includes the alias names, which are not labels and would let an alias
    # used inside an explicit mapping pass here and fail later in the label
    # operator with a bare KeyError.
    available = set(_atlas_label_names(atlas=atlas, subject=subject, subjects_dir=subjects_dir))
    alias_table = ROI_ALIASES.get(str(atlas), {})

    def _expand(member):
        """Aliases are usable wherever a label name is, including in mappings."""
        return alias_table.get(str(member), (member,))

    rois: list[ROI] = []
    for entry in names:
        if isinstance(entry, dict):
            items = list(entry.items())
        elif isinstance(entry, (tuple, list)) and len(entry) == 2 and not isinstance(entry[1], str):
            items = [(entry[0], entry[1])]
        else:
            name = str(entry)
            items = [(name, alias_table.get(name, (name,)))]

        for roi_name, members in items:
            members = (members,) if isinstance(members, str) else tuple(members)
            members = tuple(label for member in members for label in _expand(member))
            missing = [m for m in members if m not in available]
            if missing:
                hints = [
                    a for a in sorted(available) if any(m.lower() in a.lower() for m in missing)
                ]
                raise ValueError(
                    f"ROI {roi_name!r}: label(s) {missing} not found in atlas {atlas!r} "
                    f"for subject {subject!r}."
                    + (f" Did you mean any of {hints[:5]}?" if hints else "")
                    + " Use mne_rt.source.list_rois() to see what is available."
                )
            rois.append(ROI(name=str(roi_name), members=members, kind=kind))

    seen = [r.name for r in rois]
    duplicates = {n for n in seen if seen.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate ROI name(s): {sorted(duplicates)}.")
    return rois


# ---------------------------------------------------------------------------
# SourceModel
# ---------------------------------------------------------------------------


class SourceModel:
    """Sensor data → ROI time courses, via one cached linear operator.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space the inverse/beamformer was built on.  Needed to map
        atlas labels onto source-estimate rows.
    atlas : str
        Volumetric atlas stem (``"aparc+aseg"``) or surface annotation name
        (``"aparc"``).
    subject : str
        FreeSurfer subject identifier.
    subjects_dir : path-like | None
        FreeSurfer subjects directory.
    filters : instance of Beamformer | None
        Fitted LCMV spatial filter.  Required for the fast kernel path.
    inverse : instance of InverseOperator | None
        Minimum-norm inverse operator.
    inverse_method : str
        Minimum-norm method: ``"MNE"``, ``"dSPM"``, ``"sLORETA"`` or
        ``"eLORETA"``.
    info : instance of Info | None
        Measurement info, required to build a cached kernel for the
        minimum-norm path (see :meth:`source_operator`).
    lambda2 : float
        Regularisation for the minimum-norm inverse.
    pick_ori : str | None
        Source orientation. ``"normal"`` on a surface source space keeps the
        solution linear; ``None`` on a volume source space yields a magnitude,
        which is *not* linear (see Notes).

    Notes
    -----
    Both supported inverses are **linear** in the sensor data as long as the
    orientation is fixed, which is what makes the cached kernel exact:

    * LCMV with ``pick_ori="max-power"`` reduces to ``weights @ whitener``.
    * A minimum-norm operator with ``pick_ori="normal"`` is recovered by
      pushing an identity "recording" through
      :func:`~mne.minimum_norm.apply_inverse_raw` — the result *is* the
      operator matrix, using only public API.

    Free-orientation solutions (LCMV with ``pick_ori=None``/``"vector"``, or a
    minimum-norm estimate on a volume source space) combine three orientations
    by norm. That is non-linear and discards phase, which would make imaginary
    coherence meaningless, so those configurations report
    ``supports_kernel = False`` and refuse the kernel rather than silently
    returning a plausible-looking number.
    """

    def __init__(
        self,
        *,
        src,
        atlas: str,
        subject: str = "fsaverage",
        subjects_dir=None,
        filters=None,
        inverse=None,
        inverse_method: str = "dSPM",
        info=None,
        lambda2: float = 1.0 / 9.0,
        pick_ori: Optional[str] = "normal",
    ) -> None:
        self.src = src
        self.atlas = atlas
        self.subject = subject
        self.subjects_dir = subjects_dir
        self.filters = filters
        self.inverse = inverse
        self.inverse_method = inverse_method
        self.info = info
        self.lambda2 = lambda2
        self.pick_ori = pick_ori

        if filters is None and inverse is None:
            raise ValueError("SourceModel needs either `filters` (LCMV) or `inverse` (min-norm).")

    # -- introspection --------------------------------------------------

    @property
    def kind(self) -> str:
        """``"volume"`` or ``"surface"``, taken from the source space."""
        return "volume" if self.src.kind == "volume" else "surface"

    @property
    def n_sources(self) -> int:
        return int(sum(s["nuse"] for s in self.src))

    @property
    def channel_names(self) -> list:
        """Channels the operator expects, in the order its columns are in.

        This is **not** necessarily the recording's channel list: building a
        forward model drops EEG channels with no digitised position, and
        ``make_lcmv``/``make_inverse_operator`` additionally drop
        ``info["bads"]``.  :func:`~mne.beamformer.apply_lcmv_raw` handles this
        internally by selecting channels from the data; the cached-kernel path
        must do the same, or a single bad channel makes every ``kernel @ data``
        fail on a shape mismatch — and a same-count reordering would silently
        produce wrong ROI time courses.
        """
        if self.filters is not None:
            return list(self.filters["ch_names"])
        return list(self.inverse["info"]["ch_names"])

    def channel_picks(self, ch_names: Sequence[str]) -> np.ndarray:
        """Row indices selecting/reordering ``ch_names`` into operator order."""
        lookup = {name: i for i, name in enumerate(ch_names)}
        missing = [name for name in self.channel_names if name not in lookup]
        if missing:
            raise ValueError(
                f"The source operator expects channel(s) {missing}, which the data does "
                "not provide. The recording must contain every channel the forward model "
                "was built from."
            )
        return np.array([lookup[name] for name in self.channel_names], dtype=int)

    @property
    def supports_kernel(self) -> bool:
        """Whether the fast cached-kernel path is available.

        True when the source estimate is a *linear* function of the sensor data:
        an LCMV beamformer with a fixed orientation, or a minimum-norm operator
        with ``pick_ori="normal"``. Free-orientation solutions combine three
        orientations by norm and are excluded.
        """
        if self.filters is not None:
            return not self.filters["is_free_ori"]
        return self.inverse is not None and self.pick_ori == "normal" and self.info is not None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        how = "LCMV" if self.filters is not None else self.inverse_method
        fast = "kernel" if self.supports_kernel else "fallback"
        return f"<SourceModel | {self.kind}, {self.n_sources} sources, {how}, {fast}>"

    # -- operators ------------------------------------------------------

    def source_operator(self) -> np.ndarray:
        """Linear map from sensor data to source time courses, shape ``(n_src, n_ch)``.

        For LCMV this mirrors what :func:`~mne.beamformer.apply_lcmv_raw` does
        internally: whiten (or project) the data, then apply the beamformer
        weights.

        For a minimum-norm operator it recovers the same matrix by pushing an
        identity "recording" through :func:`~mne.minimum_norm.apply_inverse_raw`
        — the columns of the result are the response to a unit signal on each
        channel, i.e. the operator itself. This uses only public API and is
        exact for ``"MNE"`` and to floating-point for the noise-normalised
        methods.
        """
        if not self.supports_kernel:
            raise RuntimeError(
                "A cached source operator requires a source estimate that is linear "
                "in the sensor data: an LCMV beamformer with pick_ori='max-power', "
                "or a minimum-norm operator with pick_ori='normal' (and `info` set). "
                f"This model is {self!r}. Free-orientation solutions combine three "
                "orientations by norm, which discards phase."
            )

        if self.filters is not None:
            weights = self.filters["weights"]
            whitener = self.filters.get("whitener")
            if whitener is not None:
                return weights @ whitener
            return weights @ self.filters["proj"]

        import mne
        from mne.io import RawArray
        from mne.minimum_norm import apply_inverse_raw

        # Restrict to the channels the operator was built on, in its own order.
        ch_names = list(self.inverse["info"]["ch_names"])
        info = mne.pick_info(
            self.info, [self.info["ch_names"].index(ch) for ch in ch_names], verbose=False
        )
        eye = RawArray(np.eye(len(ch_names)), info, verbose=False)
        if not any(p["desc"].startswith("Average EEG") for p in info["projs"]):
            # apply_inverse_raw requires an average reference for EEG; the same
            # projection is applied to every real window by _prepare_raw_array.
            eye.set_eeg_reference("average", projection=True, verbose=False)
        return apply_inverse_raw(
            eye,
            self.inverse,
            lambda2=self.lambda2,
            method=self.inverse_method,
            pick_ori=self.pick_ori,
            verbose=False,
        ).data

    def label_operator(self, rois: Sequence[ROI], *, mri_resolution: bool = True) -> np.ndarray:
        """Sparse-average matrix mapping source points to ROIs, shape ``(n_roi, n_src)``.

        Equivalent to :func:`~mne.extract_label_time_course` with
        ``mode="mean"`` (volume) or ``"mean_flip"`` (surface), but built once
        rather than recomputed every window.
        """
        if self.kind == "volume":
            return self._volume_label_operator(rois, mri_resolution=mri_resolution)
        return self._surface_label_operator(rois)

    def roi_kernel(self, rois: Sequence[ROI], *, mri_resolution: bool = True) -> np.ndarray:
        """The full sensor → ROI operator, shape ``(n_roi, n_channels)``.

        Apply it to a data window with :meth:`apply`, or simply ``kernel @ data``.
        """
        return self.label_operator(rois, mri_resolution=mri_resolution) @ self.source_operator()

    # -- application ----------------------------------------------------

    def apply(
        self,
        data: np.ndarray,
        *,
        kernel=None,
        ch_picks=None,
        label_operator=None,
        info=None,
        pick_ori: Optional[str] = None,
    ) -> np.ndarray:
        """Return ROI time courses ``(n_roi, n_times)`` for one data window.

        Uses ``kernel`` when given — one matmul, and the whole point of this
        class.  Otherwise applies the source estimate the slow way and reduces
        it with ``label_operator``; correct, but ~two orders of magnitude
        slower and unsuitable for a closed loop.

        Note the label reduction is the *same* matrix either way, so the two
        paths agree exactly; only the source step differs.
        """
        if kernel is not None:
            if ch_picks is None:
                raise ValueError(
                    "`ch_picks` is required with `kernel`, to select the channels the "
                    "operator was built from (see SourceModel.channel_picks)."
                )
            return kernel @ data[ch_picks]
        if label_operator is None:
            raise ValueError("Provide either `kernel` (fast path) or `label_operator`.")
        return label_operator @ self.source_estimate(data, info=info, pick_ori=pick_ori)

    def source_estimate(self, data: np.ndarray, *, info=None, pick_ori=None) -> np.ndarray:
        """Source time courses ``(n_src, n_times)`` via the full MNE route."""
        from mne.beamformer import apply_lcmv_raw
        from mne.io import RawArray
        from mne.minimum_norm import apply_inverse_raw

        info = info if info is not None else self.info
        if info is None:
            raise ValueError("The fallback path needs `info` to wrap the window as a Raw.")
        raw = RawArray(data, info, verbose=False)
        # apply_inverse_raw requires an average EEG reference; this mirrors what
        # RTStream._prepare_raw_array applies to every analysis window.
        if any(ch["kind"] == 2 for ch in info["chs"]) and not any(
            proj["desc"].startswith("Average EEG") for proj in info["projs"]
        ):
            raw.set_eeg_reference("average", projection=True, verbose=False)
        if self.filters is not None:
            stc = apply_lcmv_raw(raw, self.filters, verbose=False)
        else:
            stc = apply_inverse_raw(
                raw,
                self.inverse,
                lambda2=self.lambda2,
                method=self.inverse_method,
                pick_ori=pick_ori if pick_ori is not None else self.pick_ori,
                verbose=False,
            )
        return stc.data

    # -- internals ------------------------------------------------------

    def _volume_label_operator(self, rois, *, mri_resolution: bool) -> np.ndarray:
        try:
            from mne.source_estimate import _volume_labels
        except ImportError as exc:  # pragma: no cover - depends on MNE internals
            raise RuntimeError(
                "This MNE version does not expose mne.source_estimate._volume_labels, "
                "which the cached-kernel path relies on. Use the fallback path."
            ) from exc

        members = [m for roi in rois for m in roi.members]
        path = _volume_atlas_path(self.atlas, self.subject, self.subjects_dir)
        labels = _volume_labels(self.src, (str(path), members), mri_resolution=mri_resolution)

        vertno = np.concatenate([s["vertno"] for s in self.src])
        position = {int(v): i for i, v in enumerate(vertno)}
        n_src = self.n_sources

        # One row per atlas label, then merge the labels making up each ROI.
        rows: dict[str, np.ndarray] = {}
        for label in labels:
            if isinstance(label, dict):
                # mri_resolution=True: csr maps source points -> the label's MRI
                # voxels, so the label mean is the column mean of that matrix.
                rows[label["name"]] = np.asarray(label["csr"].mean(axis=0)).ravel()
            else:
                idx = [position[int(v)] for v in np.asarray(label.vertices) if int(v) in position]
                row = np.zeros(n_src)
                if idx:
                    row[idx] = 1.0 / len(idx)
                rows[label.name] = row

        return self._merge_rows(rois, rows, _row_supports(rows), n_src)

    def _surface_label_operator(self, rois) -> np.ndarray:
        labels = read_labels_from_annot(
            subject=self.subject, parc=self.atlas, subjects_dir=self.subjects_dir, verbose=False
        )
        by_name = {label.name: label for label in labels}

        # Row offset of each hemisphere within the stacked source estimate.
        offsets, running = [], 0
        for s in self.src:
            offsets.append(running)
            running += s["nuse"]
        n_src = running

        rows: dict[str, np.ndarray] = {}
        for name in {m for roi in rois for m in roi.members}:
            label = by_name[name]
            hemi = 0 if label.hemi == "lh" else 1
            vertno = self.src[hemi]["vertno"]
            _, idx, _ = np.intersect1d(vertno, label.vertices, return_indices=True)
            row = np.zeros(n_src)
            if len(idx):
                # mean_flip: sign-align vertices before averaging, so that
                # opposing dipole orientations do not cancel.
                flip = np.asarray(label_sign_flip(label, self.src)).ravel()
                if len(flip) != len(idx):
                    # Silently falling back to an unsigned mean here would
                    # reintroduce exactly the cancellation mean_flip prevents.
                    raise RuntimeError(
                        f"label_sign_flip returned {len(flip)} signs for label {name!r} but "
                        f"{len(idx)} of its vertices are in the source space. Refusing to "
                        "guess an alignment; use mode='mean' if sign flipping is not wanted."
                    )
                row[offsets[hemi] + idx] = flip / len(idx)
            rows[name] = row

        return self._merge_rows(rois, rows, _row_supports(rows), n_src)

    @staticmethod
    def _merge_rows(rois, rows, counts, n_src: int) -> np.ndarray:
        """Combine per-label rows into one row per ROI, weighted by label size."""
        operator = np.zeros((len(rois), n_src))
        for i, roi in enumerate(rois):
            total = sum(counts.get(m, 0) for m in roi.members)
            if total == 0:
                warn(
                    f"ROI {roi.name!r} contains no source points; its time course "
                    "will be all zeros. The ROI may be too small for the source-space "
                    "spacing, or outside the head model.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            # Weight each member by how many source points it contributes, so a
            # multi-label ROI equals the mean over all its source points.
            for member in roi.members:
                if counts.get(member, 0):
                    operator[i] += rows[member] * (counts[member] / total)
        return operator

    # -- construction ---------------------------------------------------

    @classmethod
    def from_stream(
        cls,
        rt,
        *,
        atlas: Optional[str] = None,
        method: str = "LCMV",
        reg: float = 0.05,
        pick_ori: str = "max-power",
        weight_norm: str = "unit-noise-gain",
    ) -> "SourceModel":
        """Build a :class:`SourceModel` from a session that has recorded a baseline.

        Parameters
        ----------
        rt : instance of RTStream
            Session with ``src``/``fwd``/``data_cov`` set — i.e. one on which
            :meth:`~mne_rt.RTStream.compute_inv_operator` has run, directly or
            through the first source modality that needed it.
        atlas : str | None
            Overrides the session's ``source_atlas``.
        method : str
            ``"LCMV"`` for the beamformer, or a minimum-norm method name.
        reg, pick_ori, weight_norm
            Passed to :func:`~mne.beamformer.make_lcmv`.
        """
        from mne.beamformer import make_lcmv

        src = getattr(rt, "src", None)
        if src is None:
            raise RuntimeError(
                "No source space on the session. Run record_baseline(), then let a "
                "source modality build the head model, or call "
                "compute_inv_operator() yourself, before building a SourceModel."
            )
        atlas = atlas if atlas is not None else getattr(rt, "source_atlas", "aparc")
        info = _with_average_reference(rt.rec_info)

        if method == "LCMV":
            data_cov = getattr(rt, "data_cov", None)
            if data_cov is None:
                raise RuntimeError(
                    "An LCMV beamformer needs a data covariance. Run record_baseline() "
                    "first, or pass data_cov=... to compute_inv_operator()."
                )
            filters = make_lcmv(
                # `info` carries the average-reference projection that
                # RTStream._prepare_raw_array applies to every analysis window;
                # without it MNE warns that the covariance is adversely affected,
                # and the whitener would not match the data it is applied to.
                info,
                rt.fwd,
                data_cov,  # data covariance, NOT the noise covariance
                reg=reg,
                noise_cov=getattr(rt, "noise_cov", None),
                pick_ori=pick_ori,
                weight_norm=weight_norm,
                rank=None,
                verbose=False,
            )
            logger.info("SourceModel: built LCMV filters (pick_ori=%s).", pick_ori)
            return cls(
                src=src,
                atlas=atlas,
                subject=rt.subject_fs_id,
                subjects_dir=rt.subjects_fs_dir,
                filters=filters,
                info=info,
            )

        inverse = getattr(rt, "inv", None)
        if inverse is None:
            raise RuntimeError(
                f"Source method {method!r} needs a minimum-norm inverse operator, but the "
                "session has none. Call compute_inv_operator(make_inverse=True), or use "
                "method='LCMV'."
            )
        # A volume source space has no cortical normal, so the estimate is a
        # magnitude rather than a signed time course.
        mn_pick_ori = "normal" if src.kind != "volume" else None
        return cls(
            src=src,
            atlas=atlas,
            subject=rt.subject_fs_id,
            subjects_dir=rt.subjects_fs_dir,
            inverse=inverse,
            inverse_method=method,
            info=info,
            pick_ori=mn_pick_ori,
        )
