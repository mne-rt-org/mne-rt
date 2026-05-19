"""Real-time Maxwell filtering (SSS / tSSS) for streaming MEG data.

Classes
-------
RTMaxwellFilter
    Pre-computed SSS operator for zero-latency real-time MEG cleaning.

References
----------
Taulu, S., Kajola, M., & Simola, J. (2004). Suppression of interference and
artifacts by the Signal Space Separation Method. *Brain Topogr.*, 16(4),
269–275. https://doi.org/10.1023/B:BRAT.0000032864.93890.f9

Taulu, S., & Simola, J. (2006). Spatiotemporal signal space separation method
for rejecting nearby interference in MEG measurements. *Phys. Med. Biol.*,
51(7), 1759–1768. https://doi.org/10.1088/0031-9155/51/7/008
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ant._logging import logger


class RTMaxwellFilter:
    """Real-time Maxwell filtering (SSS / tSSS) for streaming MEG data.

    Pre-computes the Signal Space Separation (SSS) projection matrix
    :footcite:p:`taulu2004suppression` **once** from sensor geometry, then applies it as
    a single matrix multiply per incoming chunk — zero added latency, numerically
    equivalent to offline MNE.  Temporal SSS (tSSS) :footcite:p:`taulu2006spatiotemporal` can
    optionally run on a rolling buffer to remove interference that leaks into
    the internal subspace.

    See :ref:`denoising-maxwell` for the full mathematical background.

    Parameters
    ----------
    int_order : int, default 8
        Internal spherical-harmonic expansion order.
        :math:`L_{\\mathrm{in}} = 8` → :math:`(L+1)^2-1 = 80` moments
        (MNE default, adequate for all standard MEG systems).
    ext_order : int, default 3
        External expansion order (``3`` → 16 external moments).
    origin : array-like of shape (3,) | "auto", default "auto"
        SSS expansion origin in metres (head frame).  ``"auto"`` fits a
        sphere to the head digitisation; falls back to ``(0, 0, 0.04) m``
        when digitisation is absent.
    st_duration : float | None, default None
        tSSS temporal buffer in seconds.  ``None`` → spatial SSS only.
        Typical values: 10 s for persistent shielding leakage; 1–4 s for
        moving subjects.
    st_correlation : float, default 0.98
        Minimum inside–outside correlation for tSSS suppression.  Lower
        values are more aggressive.
    st_update_interval : int, default 1
        Apply tSSS every *N* incoming chunks.  Increase to reduce CPU load
        when ``winsize`` is small; SSS is used between updates.
    calibration : str | None, default None
        Fine-calibration ``.dat`` file (Elekta/MEGIN).  Corrects sensor
        position and orientation errors; typically improves noise floor
        by 10–20 %.
    cross_talk : str | None, default None
        Cross-talk compensation ``.fif`` file.  Compensates flux leakage
        between adjacent sensors.
    coord_frame : {"head", "meg"}, default "head"
        Coordinate frame for the spherical-harmonic expansion.
    regularize : {"in", None}, default "in"
        Internal-moment Tikhonov regularisation passed to MNE.
    mag_scale : float, default 100.0
        Magnetometer/gradiometer balance factor.

    Raises
    ------
    ImportError
        If MNE-Python is not installed.
    ValueError
        If no MEG channels are found in the supplied info.
    RuntimeError
        If :meth:`transform` is called before :meth:`fit`.

    See Also
    --------
    mne.preprocessing.compute_maxwell_basis : Underlying SSS basis function.
    mne.preprocessing.maxwell_filter : Full offline Maxwell filtering.
    ant.NFRealtime.fit_maxwell : Fit from a connected MEG session.

    Notes
    -----
    **No baseline recording is required.**  The SSS operator depends only on
    sensor geometry, not on brain signal statistics.  Simply call
    :meth:`~ant.NFRealtime.connect_to_lsl` then :meth:`~ant.NFRealtime.fit_maxwell`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    SSS-only (fastest, no latency):

    >>> rt_mf = RTMaxwellFilter()
    >>> rt_mf.fit(raw.info)
    >>> clean = rt_mf.transform(meg_chunk)

    tSSS with fine calibration and empty room:

    >>> rt_mf = RTMaxwellFilter(st_duration=10.0, calibration="sss_cal.dat")
    >>> rt_mf.fit(raw.info, empty_room_raw=er_raw)
    >>> clean = rt_mf.transform(meg_chunk)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        int_order: int = 8,
        ext_order: int = 3,
        origin: Union[str, tuple] = "auto",
        st_duration: Optional[float] = None,
        st_correlation: float = 0.98,
        st_update_interval: int = 1,
        calibration: Optional[str] = None,
        cross_talk: Optional[str] = None,
        coord_frame: str = "head",
        regularize: Optional[str] = "in",
        mag_scale: float = 100.0,
    ) -> None:
        if int_order < 1:
            raise ValueError("`int_order` must be a positive integer (≥ 1).")
        if ext_order < 0:
            raise ValueError("`ext_order` must be a non-negative integer (≥ 0).")
        if not 0.0 < st_correlation <= 1.0:
            raise ValueError("`st_correlation` must be in (0, 1].")

        self.int_order = int_order
        self.ext_order = ext_order
        self.origin = origin
        self.st_duration = st_duration
        self.st_correlation = st_correlation
        self.st_update_interval = max(1, int(st_update_interval))
        self.calibration = calibration
        self.cross_talk = cross_talk
        self.coord_frame = coord_frame
        self.regularize = regularize
        self.mag_scale = mag_scale

        self._fitted: bool = False
        self._meg_good_picks: Optional[np.ndarray] = None
        self._meg_all_picks: Optional[np.ndarray] = None
        self._P_sss: Optional[np.ndarray] = None
        self._resolved_origin: Union[str, tuple] = "auto"
        self.n_use_in: int = 0

        # tSSS rolling-buffer state
        self._buffer: Optional[np.ndarray] = None
        self._buf_samp: int = 0
        self._sfreq: float = 0.0
        self._chunk_count: int = 0
        self._info = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        info,
        empty_room_raw=None,
    ) -> "RTMaxwellFilter":
        """Pre-compute the SSS projection operator from MEG sensor geometry.

        No data is required — the operator depends only on sensor positions.
        An optional empty-room recording adds noise-informed regularisation.

        Parameters
        ----------
        info : mne.Info
            MEG measurement info with sensor positions and the
            device-to-head transform (``dev_head_t``).
        empty_room_raw : mne.io.Raw | None, default None
            Empty-room recording in the same coordinate frame.  When given,
            the operator is extracted via system identification so that
            noise-informed regularisation is built into the cached matrix
            (see class docstring).  ``None`` uses geometric regularisation
            only via :func:`~mne.preprocessing.compute_maxwell_basis`.

        Returns
        -------
        self : RTMaxwellFilter
        """
        import mne
        from mne.preprocessing import compute_maxwell_basis

        self._info = info.copy()
        self._sfreq = float(info["sfreq"])

        # Channel index arrays
        self._meg_good_picks = mne.pick_types(info, meg=True, exclude="bads")
        self._meg_all_picks  = mne.pick_types(info, meg=True, exclude=[])
        n_meg_good = len(self._meg_good_picks)
        n_ch = len(info["ch_names"])

        if n_meg_good == 0:
            raise ValueError(
                "No MEG channels found in info. "
                "RTMaxwellFilter operates on MEG data only."
            )

        # Resolve expansion origin — fall back gracefully when digitisation is absent
        origin = self.origin
        if origin == "auto":
            try:
                mne.bem._check_origin("auto", info, self.coord_frame, disp=False)
            except RuntimeError:
                origin = (0.0, 0.0, 0.04)  # 4 cm: centre of a typical head
                logger.warning(
                    "RTMaxwellFilter: no head digitisation found; "
                    "using origin=(0, 0, 0.04) m."
                )
        self._resolved_origin = origin

        common_kw = dict(
            int_order=self.int_order,
            ext_order=self.ext_order,
            origin=origin,
            calibration=self.calibration,
            coord_frame=self.coord_frame,
            regularize=self.regularize,
            ignore_ref=True,
            bad_condition="warning",
            mag_scale=self.mag_scale,
        )

        if empty_room_raw is None:
            logger.info(
                "RTMaxwellFilter: computing SSS basis (int=%d, ext=%d) …",
                self.int_order, self.ext_order,
            )
            S, pS, _reg_moments, n_use_in = compute_maxwell_basis(
                info, **common_kw, verbose=False
            )
            # P_sss = S_in @ S_in†  —  eq. 38, Taulu & Kajola 2005
            self._P_sss = S[:, :n_use_in] @ pS[:n_use_in]
            self.n_use_in = n_use_in
            logger.info(
                "RTMaxwellFilter: SSS operator ready — %d/%d internal moments retained.",
                n_use_in, (self.int_order + 1) ** 2 - 1,
            )
        else:
            logger.info(
                "RTMaxwellFilter: extracting noise-informed SSS operator "
                "from empty-room recording …"
            )
            self._P_sss, self.n_use_in = self._extract_operator_sysid(
                info, n_ch, empty_room_raw, common_kw
            )
            logger.info(
                "RTMaxwellFilter: noise-informed operator ready (n_internal=%d).",
                self.n_use_in,
            )

        # Initialise tSSS buffer
        if self.st_duration is not None:
            self._buf_samp = max(int(self.st_duration * self._sfreq), 1)
            self._buffer = np.zeros((n_ch, 0), dtype=np.float64)
            self._chunk_count = 0

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply Maxwell filtering to a streaming MEG window.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Raw MEG window.  Non-MEG channels are passed through unchanged.

        Returns
        -------
        clean : ndarray, shape (n_channels, n_samples)
            SSS- or tSSS-cleaned data.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._check_fitted()
        if self.st_duration is None:
            return self._apply_sss(data)
        return self._apply_tsss(data)

    # ------------------------------------------------------------------
    # Internal: SSS
    # ------------------------------------------------------------------

    def _apply_sss(self, data: np.ndarray) -> np.ndarray:
        """Apply the cached projector: P_sss maps good → all MEG channels.

        Bad channels are reconstructed from surrounding sensors as a side
        effect of the spherical-harmonic reconstruction.
        """
        out = data.copy()
        out[self._meg_all_picks] = self._P_sss @ data[self._meg_good_picks]
        return out

    # ------------------------------------------------------------------
    # Internal: rolling-buffer tSSS
    # ------------------------------------------------------------------

    def _apply_tsss(self, data: np.ndarray) -> np.ndarray:
        """Two-stage tSSS: instant SSS then periodic MNE temporal correction."""
        import mne
        from mne.preprocessing import maxwell_filter

        n_chunk = data.shape[1]

        # Stage 1: instant spatial SSS on the incoming chunk
        sss_data = self._apply_sss(data)

        # Accumulate buffer; trim to st_duration
        self._buffer = np.concatenate([self._buffer, sss_data], axis=1)
        if self._buffer.shape[1] > self._buf_samp:
            self._buffer = self._buffer[:, -self._buf_samp:]

        self._chunk_count += 1

        # SSS fallback while buffer is filling
        if self._buffer.shape[1] < self._buf_samp:
            logger.debug(
                "RTMaxwellFilter: buffer %.1f/%.1f s — SSS fallback.",
                self._buffer.shape[1] / self._sfreq, self.st_duration,
            )
            return sss_data

        # Stage 2: temporal tSSS on the full buffer (periodic)
        if self._chunk_count % self.st_update_interval != 0:
            return sss_data

        try:
            buf_raw = mne.io.RawArray(
                self._buffer.copy(), self._info.copy(), verbose=False
            )
            tsss_kw = dict(
                origin=self._resolved_origin,
                int_order=self.int_order,
                ext_order=self.ext_order,
                calibration=self.calibration,
                cross_talk=self.cross_talk,
                coord_frame=self.coord_frame,
                regularize=self.regularize,
                mag_scale=self.mag_scale,
                st_duration=self.st_duration,
                st_correlation=self.st_correlation,
                st_only=True,     # skip spatial SSS (already applied above)
                ignore_ref=True,
                bad_condition="warning",
                verbose=False,
            )
            clean_buf = maxwell_filter(buf_raw, **tsss_kw).get_data()
            self._buffer = clean_buf          # update buffer to cleaned version
            return clean_buf[:, -n_chunk:]    # return latest chunk only
        except Exception as exc:
            logger.warning(
                "RTMaxwellFilter: tSSS step failed (%s) — SSS fallback.", exc
            )
            return sss_data

    # ------------------------------------------------------------------
    # Internal: system-identification operator extraction
    # ------------------------------------------------------------------

    def _extract_operator_sysid(
        self,
        info,
        n_ch: int,
        empty_room_raw,
        common_kw: dict,
    ) -> tuple[np.ndarray, int]:
        """Recover P_sss via least squares from a Gaussian test signal.

        System-identification approach:

        1. Generate random test signal :math:`\\mathbf{X}` on MEG channels.
        2. Apply MNE maxwell_filter (with ``noise_cov`` from empty room) to get
           output :math:`\\mathbf{Y}`.
        3. Solve :math:`\\mathbf{P}_{\\mathrm{sss}} = \\mathbf{Y}\\,\\mathbf{X}^+`
           via SVD-based pseudoinverse.

        This embeds fine calibration, cross-talk compensation, and noise-
        informed regularisation into the single cached matrix.
        """
        import mne
        from mne.preprocessing import maxwell_filter, maxwell_filter_prepare_emptyroom

        n_meg_good = len(self._meg_good_picks)
        n_samp = max(n_meg_good * 4, 500)
        rng = np.random.default_rng(0)
        X_all = np.zeros((n_ch, n_samp), dtype=np.float64)
        X_all[self._meg_good_picks] = rng.standard_normal((n_meg_good, n_samp)) * 1e-12

        test_raw = mne.io.RawArray(X_all, info.copy(), verbose=False)
        er_prep = maxwell_filter_prepare_emptyroom(
            empty_room_raw, raw=test_raw, verbose=False
        )
        filtered = maxwell_filter(
            test_raw,
            **common_kw,
            cross_talk=self.cross_talk,
            noise_cov=er_prep,
            st_duration=None,
            ignore_ref=True,
            bad_condition="warning",
            verbose=False,
        ).get_data()

        Y = filtered[self._meg_all_picks]         # (n_meg_all, n_samp)
        X = X_all[self._meg_good_picks]           # (n_meg_good, n_samp)
        P_sss = Y @ np.linalg.pinv(X)             # (n_meg_all, n_meg_good)

        sv = np.linalg.svd(P_sss, compute_uv=False)
        n_use_in = int(np.sum(sv > sv[0] * 1e-6))
        return P_sss, n_use_in

    # ------------------------------------------------------------------
    # Properties & helpers
    # ------------------------------------------------------------------

    @property
    def sss_projector(self) -> np.ndarray:
        """Cached SSS matrix, shape (n_meg_all, n_meg_good).

        Maps good-channel input to the SSS-reconstructed output for all
        MEG channels, interpolating bad sensors from surrounding ones.
        """
        self._check_fitted()
        return self._P_sss

    @property
    def mode(self) -> str:
        """Operating mode: ``"sss"`` or ``"tsss"``."""
        return "tsss" if self.st_duration is not None else "sss"

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before using this method.")

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "not fitted"
        return (
            f"RTMaxwellFilter("
            f"int_order={self.int_order}, "
            f"ext_order={self.ext_order}, "
            f"mode={self.mode!r}, "
            f"state={state!r})"
        )
