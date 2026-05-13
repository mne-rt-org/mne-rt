"""Real-time bad channel detection for streaming M/EEG.

Classes
-------
BadChannelDetector
    Multi-criterion online bad channel monitor with rolling-window voting.

Notes
-----
The approach follows the methods used in EEGLAB's ``clean_rawdata`` plugin
:footcite:p:`Mullen2015` and MNE's automated quality checks, adapted for
chunk-by-chunk streaming data with no look-ahead.

References
----------
Mullen, T. R., et al. (2015). Real-time neuroimaging and cognitive monitoring
using wearable dry EEG. *IEEE Trans. Biomed. Eng.*, 62(11), 2553–2567.
https://doi.org/10.1109/TBME.2015.2481482
"""
from __future__ import annotations

import collections
import warnings
from typing import Optional, Union

import numpy as np


class BadChannelDetector:
    """Multi-criterion real-time bad channel detector for streaming M/EEG.

    Evaluates each incoming data window against up to four independent
    criteria.  A channel is flagged as bad only when it exceeds its criterion
    in at least ``min_bad_frac`` of the rolling window history — this voting
    mechanism avoids false positives from transient artifacts.

    **Criteria**

    ``"flat"``
        Channels whose RMS amplitude falls below ``flat_threshold`` (dead
        electrode / disconnected lead).

    ``"variance"``
        Channels whose RMS amplitude is a statistical outlier across all
        channels, measured by a robust z-score
        (:math:`z = (\\text{rms} - \\text{median}) / (\\text{MAD} \\times 1.4826)`).
        Catches both excessively noisy channels and channels that have gone
        unusually quiet.

    ``"correlation"``
        Channels whose mean Pearson correlation with their *K* nearest
        spatial neighbours drops below ``corr_threshold``.  A channel that
        has lost contact or broken its reference will de-correlate from
        its neighbours while the neighbours remain correlated with each other.
        Requires channel positions to be set in ``info`` (i.e. montage applied).

    ``"hf_noise"``
        Channels with an abnormally high ratio of high-frequency power
        (> ``hf_cutoff`` Hz) to broadband power.  Catches channels
        contaminated by EMG, electrode cable noise, or loose connections.
        Uses the same MAD-based robust z-score as ``"variance"``.

    Parameters
    ----------
    info : mne.Info
        MNE channel information.  Must contain ``ch_names`` and ``sfreq``.
        For ``"correlation"``, channel 3-D positions must be set
        (``set_montage`` applied).
    method : {"all", "flat", "variance", "correlation", "hf_noise"} or list
        Criterion or list of criteria to evaluate.  ``"all"`` enables
        all four criteria.  Default is ``"all"``.
    flat_threshold : float, default 1e-7
        Minimum RMS amplitude (in raw data units — V for EEG, T for MEG) for
        a channel not to be considered flat.  Channels below this value are
        dead.  Default 100 nV = ``1e-7`` V.
    variance_threshold : float, default 5.0
        Robust z-score cutoff for the variance criterion.  A channel whose
        RMS deviates by more than this many MAD-units from the channel median
        is flagged.  Default ``5.0`` (conservative; reduce to 3–4 to be more
        aggressive).
    corr_threshold : float, default 0.4
        Minimum mean Pearson correlation with spatial neighbours.  Channels
        below this value are poorly coupled to their surroundings.
        Default ``0.4``.
    hf_threshold : float, default 5.0
        Robust z-score cutoff for the HF noise criterion.  Default ``5.0``.
    hf_cutoff : float, default 40.0
        Frequency in Hz above which power is classified as *high-frequency*
        for the noise criterion.  Default ``40.0`` Hz.
    n_neighbors : int, default 4
        Number of nearest spatial neighbours used in the correlation criterion.
        Default ``4``.
    history_windows : int, default 30
        Number of per-window bad-flags to retain in the rolling history.
        Combined with ``min_bad_frac`` this sets the effective time-scale for
        declaring a channel persistently bad.
    min_bad_frac : float, default 0.5
        Fraction of rolling-history windows in which a channel must be flagged
        before it is declared bad.  ``0.5`` = majority vote.
        ``1.0`` = must be bad in every recent window (very lenient).
        ``0.1`` = bad in any 10 % of windows (very strict).

    Attributes
    ----------
    bad_channels_ : list of str
        Channel names currently declared bad.  Updated on every
        :meth:`update` call.
    scores_ : dict of str → float
        Per-channel composite badness score in ``[0, 1]``: fraction of recent
        windows in which the channel was flagged by *any* active criterion.
    n_windows_ : int
        Total number of windows processed since initialisation or last
        :meth:`reset`.

    Examples
    --------
    Basic usage — update once per NF window and pass bad channels to MNE::

        detector = BadChannelDetector(raw.info, method="all")
        while streaming:
            window = stream.get_data(1.0)   # 1 s chunk
            bad = detector.update(window)
            print("Bad channels:", bad)

    Use only the variance + flat criteria (no montage required)::

        detector = BadChannelDetector(
            info, method=["flat", "variance"], variance_threshold=4.0
        )

    .. versionadded:: 1.0.0
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        info,
        method: Union[str, list[str]] = "all",
        flat_threshold: float = 1e-7,
        variance_threshold: float = 5.0,
        corr_threshold: float = 0.4,
        hf_threshold: float = 5.0,
        hf_cutoff: float = 40.0,
        n_neighbors: int = 4,
        history_windows: int = 30,
        min_bad_frac: float = 0.5,
    ) -> None:
        self._info = info
        self._ch_names: list[str] = list(info["ch_names"])
        self._sfreq: float = float(info["sfreq"])
        self._n_ch: int = len(self._ch_names)

        _valid = {"all", "flat", "variance", "correlation", "hf_noise"}
        if isinstance(method, str):
            method = list(_valid - {"all"}) if method == "all" else [method]
        for m in method:
            if m not in _valid:
                raise ValueError(f"Unknown method {m!r}. Choose from {_valid}.")
        self._methods: set[str] = set(method)

        if flat_threshold <= 0:
            raise ValueError("flat_threshold must be > 0")
        if not (0 < min_bad_frac <= 1):
            raise ValueError("min_bad_frac must be in (0, 1]")

        self.flat_threshold = flat_threshold
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.hf_threshold = hf_threshold
        self.hf_cutoff = hf_cutoff
        self.n_neighbors = n_neighbors
        self.history_windows = history_windows
        self.min_bad_frac = min_bad_frac

        # Rolling history: per-channel deque of bool (True = bad in that window)
        self._history: dict[str, collections.deque] = {
            ch: collections.deque(maxlen=history_windows)
            for ch in self._ch_names
        }

        self.bad_channels_: list[str] = []
        self.scores_: dict[str, float] = {ch: 0.0 for ch in self._ch_names}
        self.n_windows_: int = 0

        # Pre-compute neighbour indices (may be empty if no positions)
        self._neighbor_idx: Optional[dict[int, list[int]]] = None
        if "correlation" in self._methods:
            self._neighbor_idx = self._build_neighbor_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, data: np.ndarray) -> list[str]:
        """Process one data window and return current bad-channel list.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            One analysis window of raw M/EEG data.  The channel order must
            match ``info["ch_names"]``.

        Returns
        -------
        bad_channels : list of str
            Channel names currently declared bad (majority vote over recent
            history).

        Raises
        ------
        ValueError
            If ``data.shape[0] != n_channels``.
        """
        if data.shape[0] != self._n_ch:
            raise ValueError(
                f"Expected {self._n_ch} channels, got {data.shape[0]}."
            )

        flagged = np.zeros(self._n_ch, dtype=bool)

        if "flat" in self._methods:
            flagged |= self._criterion_flat(data)

        if "variance" in self._methods:
            flagged |= self._criterion_variance(data)

        if "correlation" in self._methods and self._neighbor_idx is not None:
            flagged |= self._criterion_correlation(data)

        if "hf_noise" in self._methods:
            flagged |= self._criterion_hf_noise(data)

        # Update rolling history
        for i, ch in enumerate(self._ch_names):
            self._history[ch].append(bool(flagged[i]))

        self.n_windows_ += 1
        self._update_bad_list()
        return list(self.bad_channels_)

    def get_bad_channels(self) -> list[str]:
        """Return the current list of declared bad channels.

        Returns
        -------
        bad_channels : list of str
            Channel names that have been bad in ≥ ``min_bad_frac`` of
            the most recent ``history_windows`` windows.
        """
        return list(self.bad_channels_)

    def get_scores(self) -> dict[str, float]:
        """Return per-channel badness scores in the range [0, 1].

        A score of ``1.0`` means the channel was flagged in every recent
        window; ``0.0`` means it was never flagged.

        Returns
        -------
        scores : dict of str → float
        """
        return dict(self.scores_)

    def reset(self) -> None:
        """Clear rolling history and reset all counters.

        Constructor parameters and neighbour indices are preserved.
        """
        for dq in self._history.values():
            dq.clear()
        self.bad_channels_ = []
        self.scores_ = {ch: 0.0 for ch in self._ch_names}
        self.n_windows_ = 0

    # ------------------------------------------------------------------
    # Criteria
    # ------------------------------------------------------------------

    def _criterion_flat(self, data: np.ndarray) -> np.ndarray:
        """Flag channels whose RMS is below flat_threshold."""
        rms = np.sqrt(np.mean(data ** 2, axis=1))
        return rms < self.flat_threshold

    def _criterion_variance(self, data: np.ndarray) -> np.ndarray:
        """Flag channels whose RMS is a robust outlier across channels."""
        rms = np.sqrt(np.mean(data ** 2, axis=1))
        median = np.median(rms)
        mad = np.median(np.abs(rms - median)) * 1.4826 + 1e-30
        z = np.abs((rms - median) / mad)
        return z > self.variance_threshold

    def _criterion_correlation(self, data: np.ndarray) -> np.ndarray:
        """Flag channels whose mean correlation with neighbours is low."""
        flagged = np.zeros(self._n_ch, dtype=bool)
        if self._neighbor_idx is None:
            return flagged

        # Demean channels (required for valid Pearson correlation)
        dm = data - data.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(dm, axis=1, keepdims=True) + 1e-30
        dm_norm = dm / norms

        for i, nbrs in self._neighbor_idx.items():
            if not nbrs:
                continue
            corrs = dm_norm[i] @ dm_norm[nbrs].T / data.shape[1]
            mean_corr = float(np.mean(np.abs(corrs)))
            if mean_corr < self.corr_threshold:
                flagged[i] = True

        return flagged

    def _criterion_hf_noise(self, data: np.ndarray) -> np.ndarray:
        """Flag channels with abnormally high HF power fraction."""
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / self._sfreq)
        fft_amp = np.abs(np.fft.rfft(data, axis=1)) ** 2

        total_power = fft_amp.sum(axis=1) + 1e-30
        hf_mask = freqs > self.hf_cutoff
        hf_power = fft_amp[:, hf_mask].sum(axis=1)
        hf_ratio = hf_power / total_power

        median = np.median(hf_ratio)
        mad = np.median(np.abs(hf_ratio - median)) * 1.4826 + 1e-30
        z = (hf_ratio - median) / mad
        return z > self.hf_threshold

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_bad_list(self) -> None:
        """Recompute bad_channels_ and scores_ from current rolling history."""
        bad = []
        for ch in self._ch_names:
            history = self._history[ch]
            if not history:
                score = 0.0
            else:
                score = sum(history) / len(history)
            self.scores_[ch] = score
            if score >= self.min_bad_frac:
                bad.append(ch)
        self.bad_channels_ = bad

    def _build_neighbor_index(self) -> Optional[dict[int, list[int]]]:
        """Build a mapping from channel index to K nearest neighbours.

        Uses the 3-D electrode positions stored in ``info['chs'][k]['loc'][:3]``.
        Returns ``None`` when positions are not available.
        """
        positions = []
        for ch in self._info["chs"]:
            loc = np.asarray(ch["loc"][:3])
            positions.append(loc)

        positions = np.array(positions)
        # Check for channels with no valid position
        valid = ~np.any(np.isnan(positions) | (positions == 0).all(axis=1),
                        axis=1)

        if valid.sum() < 2:
            warnings.warn(
                "BadChannelDetector: fewer than 2 channels have valid positions; "
                "skipping the correlation criterion.",
                RuntimeWarning,
                stacklevel=4,
            )
            return None

        neighbor_idx: dict[int, list[int]] = {}
        for i in range(self._n_ch):
            if not valid[i]:
                neighbor_idx[i] = []
                continue
            # Euclidean distances to all other valid channels
            diffs = positions - positions[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf  # exclude self
            dists[~valid] = np.inf  # exclude invalid channels
            k = min(self.n_neighbors, int(valid.sum()) - 1)
            nbrs = np.argsort(dists)[:k].tolist()
            neighbor_idx[i] = nbrs

        return neighbor_idx

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BadChannelDetector("
            f"n_channels={self._n_ch}, "
            f"methods={sorted(self._methods)}, "
            f"bad={self.bad_channels_}, "
            f"n_windows={self.n_windows_})"
        )
