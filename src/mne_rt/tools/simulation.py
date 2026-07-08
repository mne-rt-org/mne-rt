"""Simulated M/EEG data generation for offline testing and demos.

This module provides two simulation functions:

* :func:`simulate_raw` — physics-based EEG/MEG via the MNE fsaverage source
  space and forward model (slow; requires the MNE sample dataset).
* :func:`simulate_nf_session` — fast parametric EEG with realistic 1/f
  background, alpha oscillations, eye blinks, muscle bursts, and slow drift
  (no MRI data required; useful for unit tests and NF algorithm development).

Functions
---------
simulate_raw
    Generate a synthetic EEG or MEG Raw object with a dipolar source.
simulate_nf_session
    Generate a realistic multi-artifact EEG simulation for NF testing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import mne
import numpy as np
from mne.datasets import fetch_fsaverage
from mne.label import select_sources
from mne.simulation import (
    SourceSimulator,
    add_eog,
    add_noise,
)
from mne.simulation import (
    simulate_raw as _mne_simulate_raw,
)

_MEG_AMPLITUDE_SCALE = 1e-12  # 1 pT — typical MEG dipole projection amplitude
_EEG_AMPLITUDE_SCALE = 10e-9  # 10 nV scaling factor (matches original code)


def simulate_raw(
    brain_label: str,
    frequency: float,
    amplitude: float,
    duration: float,
    gap_duration: float,
    n_repetition: int,
    start: float,
    data_type: str = "eeg",
    sfreq: float = 256.0,
    n_eeg_channels: int = 64,
    iir_filter: list = [0.2, -0.2, 0.04],
    add_eog_artifacts: bool = True,
    fname_save: Optional[Union[str, Path]] = None,
    verbose: Union[bool, str, None] = None,
) -> mne.io.RawArray:
    """Generate a synthetic EEG or MEG recording with a sinusoidal source.

    A forward solution is created for the ``fsaverage`` template brain and a
    sinusoidal dipole is injected into the specified cortical label.  The
    signal is projected to sensor space, optionally repeated with inter-epoch
    gaps, sensor noise is added, and the result is returned (and optionally
    saved) as an :class:`mne.io.RawArray`.

    Parameters
    ----------
    brain_label : str
        Regexp matching a cortical label in the fsaverage parcellation (e.g.
        ``"bankssts-lh"`` for alpha, ``"precentral-lh"`` for motor).
    frequency : float
        Frequency of the simulated sine wave (Hz).
    amplitude : float
        Amplitude scaling factor.  Multiplied by ``10e-9`` for EEG or
        ``1e-12`` for MEG; pass ``1.0`` for a standard physiological signal.
    duration : float
        Duration of each signal epoch (seconds).
    gap_duration : float
        Silence gap between consecutive epochs (seconds).
    n_repetition : int
        Number of epochs to simulate.
    start : float
        Start time of the first epoch (seconds from recording start).
    data_type : {"eeg", "meg"}, default "eeg"
        Sensor modality to simulate.  ``"meg"`` creates a magnetometer
        (gradiometer-free) sensor layout using the ``Vectorview-all`` template.
    sfreq : float, default 256.0
        Sampling frequency (Hz).  Used only when ``data_type="eeg"`` to build
        the synthetic sensor layout.
    n_eeg_channels : int, default 64
        Number of EEG channels.  Must be one of the standard MNE montage
        channel counts (32, 64, 128, or 256 channels of ``biosemi*``/``easycap*``
        montages).  Ignored when ``data_type="meg"``.
    iir_filter : array_like, default [0.2, -0.2, 0.04]
        IIR denominator coefficients passed to :func:`mne.simulation.add_noise`.
    add_eog_artifacts : bool, default True
        If ``True``, add simulated EOG blink artefacts.
    fname_save : str | Path | None, default None
        Path to write the output ``.fif`` file.  If ``None``, the file is
        saved to ``data/simulated/<label>_<freq>Hz_<data_type>-raw.fif``
        relative to the repository root (or current working directory).
    verbose : bool | str | int | None, default None
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        The simulated raw recording.

    Raises
    ------
    ValueError
        If ``data_type`` is not ``"eeg"`` or ``"meg"``.

    Notes
    -----
    The function requires the MNE ``fsaverage`` dataset which is downloaded
    automatically on first call via :func:`mne.datasets.fetch_fsaverage`.

    For MEG, a Vectorview-all info template is used (magnetometers + planar
    gradiometers).  For EEG, a ``biosemi64`` (or ``biosemi32``/``biosemi128``
    for other channel counts) standard layout is created programmatically.

    Examples
    --------
    Simulate alpha-band EEG in the left parieto-occipital region::

        from mne_rt.tools.simulation import simulate_raw
        raw = simulate_raw(
            brain_label="bankssts-lh",
            frequency=10.0,
            amplitude=1.0,
            duration=2.0,
            gap_duration=1.0,
            n_repetition=5,
            start=0.0,
            data_type="eeg",
        )

    Simulate beta-band MEG over left motor cortex::

        raw = simulate_raw(
            brain_label="precentral-lh",
            frequency=20.0,
            amplitude=1.0,
            duration=2.0,
            gap_duration=1.0,
            n_repetition=5,
            start=0.0,
            data_type="meg",
        )
    """
    if data_type not in ("eeg", "meg"):
        raise ValueError(f"data_type must be 'eeg' or 'meg', got {data_type!r}")

    mne.set_log_level(verbose=verbose)

    # ------------------------------------------------------------------
    # Build sensor info
    # ------------------------------------------------------------------
    raw_info = _make_sensor_info(data_type, sfreq, n_eeg_channels)

    # ------------------------------------------------------------------
    # fsaverage source space + forward solution
    # ------------------------------------------------------------------
    fs_dir = fetch_fsaverage(verbose=verbose)
    subjects_dir = os.path.dirname(fs_dir)
    subject = "fsaverage"
    trans = "fsaverage"
    src_fif = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem_fif = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    fwd = mne.make_forward_solution(
        raw_info,
        trans=trans,
        src=src_fif,
        bem=bem_fif,
        meg=(data_type == "meg"),
        eeg=(data_type == "eeg"),
        verbose=verbose,
    )
    src = fwd["src"]

    # ------------------------------------------------------------------
    # Source time series
    # ------------------------------------------------------------------
    tstep = 1.0 / raw_info["sfreq"]
    n_samples = int(duration * raw_info["sfreq"])
    t = np.arange(n_samples) * tstep

    amp_scale = _MEG_AMPLITUDE_SCALE if data_type == "meg" else _EEG_AMPLITUDE_SCALE
    source_time_series = np.sin(2.0 * np.pi * frequency * t) * amp_scale * amplitude

    # Pick a single central vertex in the target label
    selected_label = mne.read_labels_from_annot(
        subject, regexp=brain_label, subjects_dir=subjects_dir, verbose=verbose
    )[0]
    label = select_sources(
        subject,
        selected_label,
        location="center",
        extent=1,
        grow_outside=True,
        subjects_dir=subjects_dir,
    )

    # Build events: one onset per epoch, separated by gap_duration samples
    gap_samples = int(gap_duration * raw_info["sfreq"])
    start_samples = int(start * raw_info["sfreq"])
    events = np.zeros((n_repetition, 3), dtype=int)
    events[:, 0] = start_samples + gap_samples * np.arange(n_repetition)
    events[:, 2] = 1

    source_sim = SourceSimulator(src, tstep=tstep)
    source_sim.add_data(label, source_time_series, events)

    # ------------------------------------------------------------------
    # Project to sensors and add noise
    # ------------------------------------------------------------------
    raw = _mne_simulate_raw(raw_info, source_sim, forward=fwd, verbose=verbose)
    cov = mne.make_ad_hoc_cov(raw.info, verbose=verbose)
    add_noise(raw, cov, iir_filter=iir_filter, verbose=verbose)
    if add_eog_artifacts:
        add_eog(raw, verbose=verbose)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if fname_save is None:
        sim_dir = Path.cwd() / "data" / "simulated"
        sim_dir.mkdir(parents=True, exist_ok=True)
        fname_save = sim_dir / f"{brain_label}_{frequency}Hz_{data_type}-raw.fif"
    raw.save(fname=Path(fname_save), overwrite=True)

    return raw


def simulate_nf_session(
    duration: float = 120.0,
    sfreq: float = 256.0,
    n_channels: int = 64,
    alpha_frange: tuple = (8.0, 13.0),
    alpha_amplitude: float = 15e-6,
    background_amplitude: float = 5e-6,
    n_blinks: int = 15,
    blink_amplitude: float = 150e-6,
    muscle_rate: float = 0.05,
    muscle_amplitude: float = 30e-6,
    drift_amplitude: float = 20e-6,
    alpha_reactivity: bool = True,
    nf_epoch_fraction: float = 0.5,
    rng_seed: Optional[Union[int, None]] = None,
    verbose: Union[bool, str, None] = None,
) -> tuple:
    """Generate a realistic multi-artifact EEG simulation for NF testing.

    Produces synthetic EEG with:

    * **1/f background noise** (pink noise) on all channels
    * **Alpha-band oscillations** with realistic spatial topography
      (occipital channels have strongest alpha)
    * **Alpha reactivity**: alpha power is reduced during simulated NF
      "active" epochs (mimics ERD during cognitive effort)
    * **Eye-blink artefacts** at realistic rate (≈15/min) with frontal
      topography
    * **Muscle noise bursts** as short high-frequency transients
    * **Slow electrode drift** (low-frequency random walk)

    Parameters
    ----------
    duration : float, default 120.0
        Total recording duration in seconds.
    sfreq : float, default 256.0
        Sampling rate in Hz.
    n_channels : int, default 64
        Number of EEG channels (must be 32, 64, 128, or 256 for standard
        biosemi/easycap montages).
    alpha_frange : tuple, default (8.0, 13.0)
        Alpha-band frequency range (Hz).
    alpha_amplitude : float, default 15e-6
        Peak alpha oscillation amplitude at occipital electrodes (V).
    background_amplitude : float, default 5e-6
        Broadband 1/f noise amplitude (V).
    n_blinks : int, default 15
        Number of blink artefacts to inject.
    blink_amplitude : float, default 150e-6
        Peak blink amplitude (V) at Fp1/Fp2.
    muscle_rate : float, default 0.05
        Fraction of windows contaminated with muscle bursts (0–1).
    muscle_amplitude : float, default 30e-6
        Peak muscle burst amplitude (V).
    drift_amplitude : float, default 20e-6
        Peak slow-drift amplitude (V).
    alpha_reactivity : bool, default True
        If True, alpha power is reduced by ~50% during NF epochs.
    nf_epoch_fraction : float, default 0.5
        Fraction of total duration where NF-state (reduced alpha) applies.
    rng_seed : int | None, default None
        Random seed for reproducibility.
    verbose : bool | str | None, default None
        MNE verbosity.

    Returns
    -------
    raw : mne.io.RawArray
        Simulated raw EEG.
    nf_state : ndarray, shape (n_samples,)
        Boolean mask: True where simulated NF-state (reduced alpha) is active.

    Notes
    -----
    The function uses a standard biosemi64 montage channel layout. Channels
    are sorted into occipital (O1, Oz, O2, POz, PO3, PO4, PO7, PO8),
    frontal (Fp1, Fp2, AF7, AF8), and remaining groups to assign
    spatially realistic source weights.

    References
    ----------
    Niedermeyer, E., & da Silva, F. L. (2005). Electroencephalography:
    Basic Principles, Clinical Applications, and Related Fields. LWW.

    Pfurtscheller, G., & Aranibar, A. (1977). Event-related cortical
    desynchronisation detected by power measurements of scalp EEG.
    Electroencephalography and clinical Neurophysiology, 42(6), 817–826.
    """
    from scipy.signal import butter, sosfiltfilt

    mne.set_log_level(verbose=verbose)
    rng = np.random.default_rng(rng_seed)

    n_samples = int(duration * sfreq)
    info = _make_sensor_info("eeg", sfreq, n_channels)
    ch_names = info["ch_names"]

    # ── Spatial weight vectors ─────────────────────────────────────────────
    occipital_names = {"O1", "Oz", "O2", "POz", "PO3", "PO4", "PO7", "PO8"}
    frontal_names = {"Fp1", "Fp2", "AF7", "AF8"}

    alpha_weights = np.zeros(n_channels)
    blink_weights = np.zeros(n_channels)
    for i, ch in enumerate(ch_names):
        if ch in occipital_names:
            alpha_weights[i] = 1.0
        elif ch in frontal_names:
            alpha_weights[i] = 0.1
            blink_weights[i] = 1.0
        else:
            alpha_weights[i] = 0.3
            blink_weights[i] = 0.05

    # Normalise weights so they sum to 1 (max-normalise for spatial realism)
    if alpha_weights.max() > 0:
        alpha_weights /= alpha_weights.max()
    if blink_weights.max() > 0:
        blink_weights /= blink_weights.max()

    # ── NF state mask (consecutive block in the middle of the recording) ──
    nf_state = np.zeros(n_samples, dtype=bool)
    nf_n = int(nf_epoch_fraction * n_samples)
    nf_start = (n_samples - nf_n) // 2
    nf_state[nf_start : nf_start + nf_n] = True

    # ── 1. Pink noise (all channels) ──────────────────────────────────────
    def _pink_noise(n: int) -> np.ndarray:
        white = rng.standard_normal(n)
        fft_vals = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        # Avoid division by zero at DC; set DC to 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(freqs == 0, 0.0, 1.0 / np.sqrt(freqs))
        fft_vals *= scale
        pink = np.fft.irfft(fft_vals, n=n)
        # Normalise to unit RMS
        rms = np.sqrt(np.mean(pink**2))
        return pink / (rms + 1e-30)

    data = np.zeros((n_channels, n_samples))
    for c in range(n_channels):
        data[c] = _pink_noise(n_samples) * background_amplitude

    # ── 2. Alpha oscillations ──────────────────────────────────────────────
    iaf = (alpha_frange[0] + alpha_frange[1]) / 2.0  # individual alpha frequency
    t = np.arange(n_samples) / sfreq
    # Random phase offset per channel to avoid perfectly synchronous oscillations
    phase_offsets = rng.uniform(0, 2 * np.pi, n_channels)

    for c in range(n_channels):
        alpha_signal = np.sin(2.0 * np.pi * iaf * t + phase_offsets[c])
        if alpha_reactivity:
            # Reduce alpha amplitude by 50% during NF state
            amplitude_mod = np.where(nf_state, 0.5, 1.0)
            alpha_signal = alpha_signal * amplitude_mod
        data[c] += alpha_weights[c] * alpha_amplitude * alpha_signal

    # ── 3. Eye blink artefacts ────────────────────────────────────────────
    if n_blinks > 0:
        blink_duration_s = 0.2  # 200 ms Gaussian envelope
        blink_sigma_samples = int(blink_duration_s / 2 * sfreq)
        blink_half_width = 3 * blink_sigma_samples
        blink_t = np.arange(-blink_half_width, blink_half_width + 1)
        blink_template = np.exp(-(blink_t**2) / (2 * blink_sigma_samples**2))

        blink_times = rng.integers(blink_half_width, n_samples - blink_half_width, size=n_blinks)
        for bt in blink_times:
            start = bt - blink_half_width
            end = bt + blink_half_width + 1
            actual_len = min(end, n_samples) - max(start, 0)
            tmpl_start = max(0, -start)
            tmpl_end = tmpl_start + actual_len
            data_start = max(0, start)
            for c in range(n_channels):
                data[c, data_start : data_start + actual_len] += (
                    blink_weights[c] * blink_amplitude * blink_template[tmpl_start:tmpl_end]
                )

    # ── 4. Muscle bursts ──────────────────────────────────────────────────
    window_size = int(sfreq)  # 1-second windows
    n_windows = n_samples // window_size
    n_muscle_windows = int(muscle_rate * n_windows)
    muscle_window_indices = rng.choice(n_windows, size=n_muscle_windows, replace=False)

    for wi in muscle_window_indices:
        ws = wi * window_size
        we = ws + window_size
        # Band-limited high-frequency noise (30–150 Hz)
        burst = rng.standard_normal((n_channels, window_size))
        sos = butter(
            4,
            [30.0 / (sfreq / 2), min(150.0, sfreq / 2 - 1) / (sfreq / 2)],
            btype="bandpass",
            output="sos",
        )
        for c in range(n_channels):
            burst[c] = sosfiltfilt(sos, burst[c])
        # Normalise burst to unit RMS then scale
        rms_burst = np.sqrt(np.mean(burst**2, axis=1, keepdims=True))
        burst /= rms_burst + 1e-30
        data[:, ws:we] += muscle_amplitude * burst

    # ── 5. Slow electrode drift ───────────────────────────────────────────
    drift_raw = rng.standard_normal((n_channels, n_samples))
    sos_drift = butter(
        2,
        0.5 / (sfreq / 2),
        btype="low",
        output="sos",
    )
    for c in range(n_channels):
        drift_filtered = sosfiltfilt(sos_drift, drift_raw[c])
        max_drift = np.abs(drift_filtered).max()
        if max_drift > 0:
            drift_filtered /= max_drift
        data[c] += drift_amplitude * drift_filtered

    raw = mne.io.RawArray(data, info, verbose=verbose)
    return raw, nf_state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_sensor_info(data_type: str, sfreq: float, n_eeg_channels: int) -> mne.Info:
    """Build an :class:`mne.Info` object for the requested modality.

    For EEG a standard montage is used; for MEG the MNE sample dataset
    Vectorview info is used as template (magnetometers + gradiometers).
    """
    if data_type == "meg":
        try:
            sample_dir = mne.datasets.sample.data_path(verbose=False)
        except Exception as exc:
            raise RuntimeError(
                "MEG simulation requires the MNE sample dataset. "
                "Install it with: mne.datasets.sample.data_path()"
            ) from exc
        raw_fname = os.path.join(sample_dir, "MEG", "sample", "sample_audvis_raw.fif")
        info = mne.io.read_info(raw_fname, verbose=False)
        # Keep only MEG channels; resample info sfreq if different
        meg_picks = mne.pick_types(info, meg=True, eeg=False, stim=False, exclude="bads")
        info = mne.pick_info(info, sel=meg_picks)
        info["sfreq"] = float(sfreq)
        return info

    # EEG — use a standard biosemi/easycap montage
    montage_map = {
        32: "easycap-M10",
        64: "biosemi64",
        128: "biosemi128",
        256: "biosemi256",
    }
    montage_name = montage_map.get(n_eeg_channels, "biosemi64")
    montage = mne.channels.make_standard_montage(montage_name)
    info = mne.create_info(
        ch_names=montage.ch_names,
        sfreq=float(sfreq),
        ch_types="eeg",
    )
    info.set_montage(montage)
    return info
