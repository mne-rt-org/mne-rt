import os.path as op
import time
from copy import deepcopy
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import sparse
from scipy.integrate import simpson
from scipy.signal import butter, get_window, periodogram, welch
from scipy.spatial.distance import pdist, squareform

try:
    from specparam import SpectralModel as FOOOF
except ImportError:
    from fooof import FOOOF  # legacy name
try:
    from pyunlocbox import functions
    from pyunlocbox import solvers as _pux_solvers

    _pyunlocbox_available = True
except Exception:
    _pyunlocbox_available = False
try:
    import pyvista as pv

    _pyvista_available = True
except ImportError:
    pv = None  # type: ignore[assignment]
    _pyvista_available = False
import mne
from mne import (
    compute_raw_covariance,
    make_bem_model,
    make_bem_solution,
    make_forward_solution,
    read_labels_from_annot,
    setup_source_space,
    setup_volume_source_space,
)
from mne.coreg import Coregistration
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator
from mne.preprocessing import ICA
from mne.surface import read_surface
from mne.time_frequency import psd_array_multitaper
from mne.viz import create_3d_figure, get_brain_class, set_3d_backend
from mne_icalabel import label_components
from nibabel.freesurfer import read_morph_data

from mne_rt._logging import logger
from mne_rt._naming import split_modality


def timed(func):
    """Decorator to measure execution time of a function.

    Parameters
    ----------
    func : callable
            Function to be timed.

    Returns
    -------
    wrapper : callable
            Wrapped function that returns both the function's output
            and its execution time in seconds.

    Notes
    -----
    The decorated function returns a tuple ``(value, elapsed_time)``,
    where ``value`` is the original return value of the function,
    and ``elapsed_time`` is the runtime measured with
    :func:`time.perf_counter`.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        return value, toc - tic

    return wrapper


def get_canonical_freqs(frange_name):
    """
    Map a canonical frequency band name to its frequency range.

    Parameters
    ----------
    frange_name : str
            Name of the frequency band. Supported values include:
            'delta', 'theta', 'alpha', 'lower_alpha', 'upper_alpha',
            'smr', 'beta', 'lower_beta', 'upper_beta',
            'gamma', 'lower_gamma', 'upper_gamma'.

    Returns
    -------
    List[float]
            Two-element list containing [low_freq, high_freq] in Hz.

    Raises
    ------
    ValueError
            If the provided frequency band name is not defined.
    """

    freq_bands = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "alpha": [8, 13],
        "lower_alpha": [8, 10],
        "upper_alpha": [10, 13],
        "smr": [12, 15],
        "beta": [15, 30],
        "lower_beta": [15, 20],
        "upper_beta": [20, 30],
        "gamma": [30, 80],
        "lower_gamma": [30, 50],
        "upper_gamma": [50, 80],
    }
    try:
        return freq_bands[frange_name]
    except KeyError:
        raise ValueError(
            f"Frequency range '{frange_name}' is not defined. "
            f"Available options: {list(freq_bands.keys())}"
        )


def get_params(config_file, modality, modality_params, instance=None):
    """Load and update parameters for a given neurofeedback (NF) modality.

    Parameters
    ----------
    config_file : str | path-like
            Path to the YAML configuration file containing modality parameters.
    modality : str
            Name of the neurofeedback modality. Must be present in the
            ``NF_modality`` section of the config file. An instanced name
            (``"source_connectivity@theta"``, see :mod:`mne_rt._naming`) is
            accepted and split, so its base selects the config entry.
    modality_params : dict | None
            Optional parameter overrides, in either of two forms:

            * **nested** — ``{modality_name: {param: value}}``, the form
              :meth:`~mne_rt.RTStream.record_main` passes when several
              modalities run together.  Only the entries matching this
              modality are applied; entries for other modalities are ignored.
              Keys may be base modalities or instanced names; a base entry
              acts as a default for every instance of it, and an instance
              entry takes precedence.
            * **flat** — ``{param: value}``, applied directly to ``modality``.

            The two are told apart by checking whether any top-level key names
            a modality defined in the config file, or an instance of one.
    instance : str | None
            Full instanced name whose overrides should be applied on top of
            the base entry's.  Defaults to ``modality`` itself.  Passing it
            explicitly lets a caller that has already parsed the name avoid a
            second split.

    Returns
    -------
    params : dict
            Dictionary containing updated parameters for the specified modality.

    Raises
    ------
    ValueError
            If the modality is not found in the config file.
    ValueError
            If ``instance`` and ``modality`` name different base modalities.
    ValueError
            If an override names a parameter this modality does not have.

    Notes
    -----
    This function first loads default parameters for the requested modality
    from the YAML file. If ``modality_params`` is provided, the defaults
    are updated with the user-specified overrides.

    Examples
    --------
    Three bands of one measure, sharing everything but their frequency range::

        modality_params = {
            "source_connectivity": {"method": "imcoh"},   # applies to all three
            "source_connectivity@theta": {"frange": [4, 8]},
            "source_connectivity@alpha": {"frange": [8, 13]},
            "source_connectivity@beta": {"frange": [13, 30]},
        }
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    known = set(config["NF_modality"])
    base, _ = split_modality(modality)
    if instance is None:
        instance = modality
    elif split_modality(instance)[0] != base:
        raise ValueError(
            f"`instance` {instance!r} is not an instance of modality {modality!r}; "
            "they must share a base modality."
        )

    if base not in known:
        named = f" (from {instance!r})" if instance != base else ""
        raise ValueError(
            f"Unknown modality {base!r}{named}, must be one of {list(config['NF_modality'].keys())}"
        )

    def _merge(params, overrides, named_as):
        unknown = set(overrides) - set(params)
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {sorted(unknown)} for modality {named_as!r}. "
                f"Available: {sorted(params)}"
            )
        for key, value in overrides.items():
            # Only dict-valued parameters merge; scalars and lists are replaced.
            if isinstance(params[key], dict) and isinstance(value, dict):
                params[key].update(value)
            else:
                params[key] = value

    # get params for this modality
    params = deepcopy(config["NF_modality"][base])
    if modality_params:
        # Nested {modality: {...}} or flat {param: ...}?  A key naming any known
        # modality — or an instance of one — means the caller used the nested form.
        nested = any(
            key in known or split_modality(key)[0] in known for key in map(str, modality_params)
        )
        if nested:
            # The base entry is the default for every instance of it; the
            # instance's own entry is layered on top and wins.
            _merge(params, modality_params.get(base) or {}, base)
            if instance != base:
                _merge(params, modality_params.get(instance) or {}, instance)
        else:
            _merge(params, modality_params, instance)
    return params


# ``spectral_connectivity_time`` implements a smaller set of metrics than
# ``spectral_connectivity_epochs``.  Map the ones we advertise onto what it
# actually supports; the bool says whether to take the imaginary part after.
_TIME_CON_ALIASES = {"imcoh": ("cohy", True)}


def resolve_connectivity_method(method):
    """Map an NF connectivity metric onto a ``spectral_connectivity_time`` method.

    :func:`mne_connectivity.spectral_connectivity_time` has no ``"imcoh"`` in its
    bivariate dispatch table and raises ``KeyError`` if asked for one.  Since the
    imaginary part of coherency *is* the imaginary coherence by definition, we
    request ``"cohy"`` and take the imaginary part ourselves — an exact identity,
    not an approximation.

    Parameters
    ----------
    method : str
            Connectivity metric as named in the config file.

    Returns
    -------
    method : str
            Metric to pass to :func:`~mne_connectivity.spectral_connectivity_time`.
    take_imag : bool
            Whether the returned (complex) values must be reduced to their
            imaginary part.
    """
    return _TIME_CON_ALIASES.get(method, (method, False))


def resolve_n_cycles(freqs, n_cycles, winsize):
    """Resolve the Morlet ``n_cycles`` for an analysis window of ``winsize`` seconds.

    A Morlet wavelet of ``n`` cycles at ``f`` Hz spans ``n / f`` seconds and must
    fit inside the analysis window.  With the fixed ``n_cycles=5`` used previously,
    a theta band (4 Hz) needs 1.25 s and therefore cannot be estimated at the
    default 1 s window at all.

    Parameters
    ----------
    freqs : array-like
            Frequencies the wavelets are centred on, in Hz.
    n_cycles : float | array-like | "auto"
            Number of cycles per wavelet.  ``"auto"`` scales with the window as
            ``max(2, freqs * winsize / 3)``, which keeps roughly three wavelets
            inside the window while never dropping below two cycles.
    winsize : float
            Analysis window length in seconds.

    Returns
    -------
    n_cycles : ndarray
            One value per entry in ``freqs``.

    Raises
    ------
    ValueError
            If any wavelet would be longer than the analysis window.
    """
    freqs = np.asarray(freqs, dtype=float)
    if isinstance(n_cycles, str):
        if n_cycles != "auto":
            raise ValueError(f"n_cycles must be a number, an array, or 'auto'; got {n_cycles!r}.")
        n_cycles = np.maximum(2.0, freqs * float(winsize) / 3.0)
    n_cycles = np.broadcast_to(np.asarray(n_cycles, dtype=float), freqs.shape).copy()

    # mne-connectivity rejects freqs < n_cycles / winsize; raise our own message
    # first so the user is told which knob to turn.
    lengths = n_cycles / freqs
    too_long = lengths > float(winsize)
    if np.any(too_long):
        i = int(np.argmax(too_long))
        raise ValueError(
            f"n_cycles={n_cycles[i]:g} at {freqs[i]:g} Hz needs {lengths[i]:.3g} s of data "
            f"but winsize is {winsize:g} s. Increase winsize (>= {lengths[i]:.3g} s), raise "
            "the lower edge of 'frange', or lower n_cycles. Note n_cycles='auto' scales "
            "with winsize."
        )
    return n_cycles


def compute_bandpower(
    data, sfreq, band, method="fft", relative=True, window="hann", n_fft=None, **kwargs
):
    """
    Compute the bandpower of EEG/MEG channels using various PSD estimation methods.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
            Input time series data.
    sfreq : float
            Sampling frequency in Hz.
    band : tuple of float
            Frequency band of interest (low, high) in Hz.
    method : {'fft', 'periodogram', 'welch', 'multitaper'}, default='fft'
            PSD estimation method.
    relative : bool, default=True
            Normalize by total power if True.
    window : str, default='hann'
            Window type (used only for 'fft').
    n_fft : int | None
            Number of FFT points (used only for 'fft').
    **kwargs : additional keyword arguments
            Passed to the respective PSD functions.

    Returns
    -------
    bandpower : ndarray, shape (n_channels,)
            Bandpower of each channel in the requested frequency band.
    """
    assert data.ndim == 2, "Input must be 2D: (n_channels, n_samples)"
    assert len(band) == 2 and band[0] <= band[1], "Band must be (low, high)"

    n_channels, n_samples = data.shape
    n_fft = n_fft or int(2 ** np.ceil(np.log2(n_samples)))

    if method == "fft":
        win = get_window(window, n_samples, fftbins=True)
        data_win = data * win
        freqs = np.fft.rfftfreq(n_fft, d=1 / sfreq)
        psd = (np.abs(np.fft.rfft(data_win, n=n_fft)) ** 2) / (sfreq * np.sum(win**2))

    elif method == "periodogram":
        freqs, psd = periodogram(data, sfreq, axis=1, **kwargs)

    elif method == "welch":
        freqs, psd = welch(data, sfreq, axis=1, **kwargs)

    elif method == "multitaper":
        psd, freqs = psd_array_multitaper(data, sfreq, verbose="ERROR", **kwargs)

    else:
        raise ValueError(f"Unsupported method '{method}'.")

    # Frequency band selection
    mask = (freqs >= band[0]) & (freqs <= band[1])
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    bp = simpson(psd[:, mask], dx=freq_res, axis=1)
    if relative:
        bp /= simpson(psd, dx=freq_res, axis=1)

    return bp


def compute_instantaneous_phase(data, sfreq, frange, channel_indices=None):
    """Estimate instantaneous phase via the analytic signal (Hilbert transform).

    Bandpass-filters the data in ``frange``, then computes the analytic signal
    using :func:`scipy.signal.hilbert`.  Returns the instantaneous phase (rad)
    at the *last sample* of the window, which is the current-time phase
    estimate.  Using ``sosfiltfilt`` (zero-phase filtering) removes edge
    distortion at the cost of requiring the full window to be available.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
            Input time series.
    sfreq : float
            Sampling frequency in Hz.
    frange : tuple of float
            (low, high) frequency band in Hz for the bandpass filter.
    channel_indices : array_like of int | None, default None
            Channels to average before phase estimation. ``None`` → all channels.

    Returns
    -------
    phase : float
            Instantaneous phase in radians at the last sample (in ``[-π, π]``).
    amplitude : float
            Instantaneous amplitude (envelope) at the last sample.

    Notes
    -----
    For NF protocols, use the returned ``phase`` to trigger phase-specific
    stimulation or to reward phase alignment across channels.  The ``amplitude``
    can serve as a gating signal (only respond to phase when amplitude is high).

    References
    ----------
    Pikovsky, A., Rosenblum, M., & Kurths, J. (2001). Synchronization: a
    universal concept in nonlinear sciences. Cambridge University Press.

    Canolty, R. T., & Knight, R. T. (2010). The functional role of
    cross-frequency coupling. Trends in Cognitive Sciences, 14(11), 506–515.
    """
    from scipy.signal import butter, hilbert, sosfiltfilt

    lo, hi = frange
    sos = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="bandpass", output="sos")

    if channel_indices is not None:
        x = data[np.array(channel_indices)].mean(axis=0)
    else:
        x = data.mean(axis=0)

    filtered = sosfiltfilt(sos, x)
    analytic = hilbert(filtered)
    phase = float(np.angle(analytic[-1]))
    amplitude = float(np.abs(analytic[-1]))
    return phase, amplitude


def compute_fft(sfreq, winsize, freq_range, freq_res=1):
    """Compute FFT-related quantities for spectral analysis.

    Parameters
    ----------
    sfreq : float
            Sampling frequency in Hz.
    winsize : float
            Window size in seconds.
    freq_range : tuple of float
            Frequency range of interest (low, high) in Hz.
    freq_res : int, optional
            Frequency resolution factor. Default is 1.

    Returns
    -------
    fft_window : ndarray, shape (n_times,)
            Hanning window of length ``winsize * sfreq`` in samples.
    freq_band : ndarray, shape (n_freqs,)
            Array of frequencies within the specified range.
    freq_band_idxs : ndarray, shape (n_freqs,)
            Indices of ``frequencies`` corresponding to ``freq_band``.
    frequencies : ndarray, shape (n_freqs_total,)
            Array of frequencies from the FFT across the full spectrum.
    """
    winsize_in_samples = sfreq * winsize
    frequencies = np.fft.rfftfreq(n=int(winsize_in_samples) * freq_res, d=1 / sfreq)
    freq_band_idxs = np.where(
        np.logical_and(freq_range[0] <= frequencies, frequencies <= freq_range[1])
    )[0]
    freq_band = frequencies[freq_band_idxs]
    fft_window = np.hanning(winsize_in_samples)

    return fft_window, freq_band, freq_band_idxs, frequencies


def butter_bandpass(l_freq, h_freq, sfreq, order):
    """Design a Butterworth band-pass filter.

    Parameters
    ----------
    l_freq : float
            Low cut-off frequency in Hz.
    h_freq : float
            High cut-off frequency in Hz.
    sfreq : float
            Sampling frequency in Hz.
    order : int
            Order of the Butterworth filter.

    Returns
    -------
    sos : ndarray
            Second-order sections representation of the band-pass filter,
            suitable for use with :func:`scipy.signal.sosfiltfilt`.
    """
    nyq = 0.5 * sfreq
    (low, high) = (l_freq / nyq, h_freq / nyq)
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def estimate_aperiodic_component(raw_baseline, picks, method, freq_range=(1, 20), verbose=None):
    """
    Estimate the aperiodic (1/f) component of a PSD using FOOOF.

    Parameters
    ----------
    raw_baseline : mne.io.BaseRaw
            MNE Raw object containing baseline data.
    picks : str | list
            Channels to include in PSD computation.
    psd_params : dict
            Parameters passed to `raw_baseline.compute_psd()`.

    freq_range : tuple of float, optional (default=(1, 20))
            Frequency range in Hz for fitting the model.
    verbose : bool, optional
            Whether to print FOOOF fitting progress.

    Returns
    -------
    aperiodic_params : ndarray
            Parameters of the aperiodic component (offset, slope, knee if used).
    peak_params : ndarray
            Parameters of oscillatory peaks identified in the spectrum.
    """
    spectrum = raw_baseline.compute_psd(picks=picks, method=method, fmax=80)
    fm = FOOOF(verbose=verbose)

    fm.fit(spectrum.freqs, spectrum.get_data().mean(axis=0), freq_range=freq_range)
    return fm.aperiodic_params_, fm.peak_params_


def _compute_inv_operator(
    raw_baseline,
    subject_fs_id="fsaverage",
    subjects_fs_dir=None,
    data_type="eeg",
    loose=0.2,
    depth=0.8,
    noise_cov_method="ad_hoc",
    reg=0.1,
    *,
    src_type="surface",
    vol_pos=5.0,
    vol_atlas="aparc+aseg",
    volume_labels=None,
    make_inverse=True,
):
    """
    Compute the inverse operator for EEG/MEG source localization.

    This function sets up the forward and inverse models required
    to project sensor-level data into source space. Depending
    on whether the subject is ``fsaverage`` or an individual, the
    function handles source space, BEM, and coregistration steps
    differently.

    Parameters
    ----------
    raw_baseline : mne.io.Raw
            The baseline raw recording used to estimate the noise
            covariance and forward model.
    subject : str, default='fsaverage'
            The subject identifier. If 'fsaverage', a template anatomy
            provided by MNE is used. Otherwise, a subject-specific model
            is built using individual MRI and digitization data.
    data_type : {'eeg', 'meg'}, default='eeg'
            Modality flag. Controls whether the forward solution is
            computed for EEG or MEG channels.
    src_type : {'surface', 'volume'}, default='surface'
            Source-space geometry. ``'surface'`` keeps the historical
            cortical-surface behaviour unchanged. ``'volume'`` builds a
            volumetric grid, which is what subcortical ROIs (hippocampus,
            amygdala, thalamus) require — they do not exist on the cortical
            surface at all.
    vol_pos : float, default=5.0
            Grid spacing in mm for ``src_type='volume'``.
    vol_atlas : str, default='aparc+aseg'
            Volumetric atlas stem under ``<subject>/mri/`` used when
            ``volume_labels`` is given. ``aparc+aseg`` carries cortical
            parcels *and* subcortical structures, so a single volume source
            space covers both.
    volume_labels : list of str | None, default=None
            Restrict the volume source space to these atlas labels. Strongly
            recommended: a whole-brain 5 mm grid is ~14 600 source points,
            whereas the labels for a typical ROI set total a few hundred —
            which is the difference between a real-time-capable pipeline and
            one that is not.
    make_inverse : bool, default=True
            Build a minimum-norm inverse operator. Set ``False`` for a
            beamformer-only session, where only ``fwd`` and the covariances
            are needed.

    Returns
    -------
    inverse_operator : dict | None
            The inverse operator object created with MNE-Python. This can
            be passed to functions such as ``apply_inverse`` or
            ``apply_inverse_epochs`` to estimate source time courses.
            ``None`` when ``make_inverse=False``.
    fwd : instance of Forward
            The forward solution.
    noise_cov : instance of Covariance
            The noise covariance.
    src : instance of SourceSpaces
            The source space the forward model was built on. Returned because
            it is needed to map atlas labels onto source estimates and cannot
            otherwise be recovered from a beamformer.

    Notes
    -----
    - If ``subject='fsaverage'``:
            * Fetches the template FreeSurfer subject using
            :func:`mne.datasets.fetch_fsaverage`.
            * Uses precomputed fsaverage source space and BEM solution.
            * Uses the default ``'fsaverage'`` trans file.

    - If an individual subject is specified:
            * Requires subject-specific source space and BEM setup.
            * Runs an automatic coregistration using fiducials and ICP
            alignment (with multiple iterations and nasion weighting).
            * Builds a BEM model and solution with :func:`mne.make_bem_model`
            and :func:`mne.make_bem_solution`.
            * Estimates the head-MRI transform from the coregistration.
    """

    if src_type not in ("surface", "volume"):
        raise ValueError(f"`src_type` must be 'surface' or 'volume', got {src_type!r}.")

    if subject_fs_id == "fsaverage":
        fs_dir = fetch_fsaverage()
        bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
        trans = "fsaverage"
        if src_type == "surface":
            src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
        elif volume_labels is None and float(vol_pos) == 5.0:
            # fetch_fsaverage ships a prebuilt 5 mm whole-brain volume source
            # space, so skip the (slow) interpolator construction entirely.
            src = op.join(fs_dir, "bem", "fsaverage-vol-5-src.fif")
        else:
            src = setup_volume_source_space(
                subject=subject_fs_id,
                pos=vol_pos,
                mri=f"{vol_atlas}.mgz",
                bem=bem,
                volume_label=volume_labels,
                # one SourceSpaces, not one per label: downstream label
                # extraction reads a single interpolator/vertno
                single_volume=True,
                subjects_dir=op.dirname(fs_dir),
                add_interpolator=True,
                verbose=False,
            )

    else:
        if src_type == "surface":
            src = setup_source_space(subject=subject_fs_id, subjects_dir=subjects_fs_dir)
        bem_model = make_bem_model(subject=subject_fs_id, subjects_dir=subjects_fs_dir)
        bem = make_bem_solution(bem_model)
        if src_type == "volume":
            src = setup_volume_source_space(
                subject=subject_fs_id,
                pos=vol_pos,
                mri=f"{vol_atlas}.mgz",
                bem=bem,
                volume_label=volume_labels,
                single_volume=True,
                subjects_dir=subjects_fs_dir,
                add_interpolator=True,
                verbose=False,
            )
        coreg = Coregistration(
            raw_baseline.info, subject=subject_fs_id, subjects_dir=subjects_fs_dir, fiducials="auto"
        )
        coreg.fit_fiducials()
        coreg.fit_icp(n_iterations=40, nasion_weight=2.0)
        coreg.omit_head_shape_points(distance=5.0 / 1000)
        coreg.fit_icp(n_iterations=40, nasion_weight=10)
        trans = coreg.trans

    # Drop EEG channels that have no digitized location; make_forward_solution
    # raises RuntimeError on any EEG channel whose loc vector is all-zero.
    raw_fwd = raw_baseline.copy()
    missing_loc = [
        ch["ch_name"]
        for ch in raw_fwd.info["chs"]
        if ch["kind"] == 2
        and (  # 2 = FIFFV_EEG_CH
            np.any(np.isnan(ch["loc"][:3])) or np.allclose(ch["loc"][:3], 0.0)
        )
    ]
    if missing_loc:
        raw_fwd.drop_channels(missing_loc)

    # EEG needs the average reference: apply_inverse_raw requires it, and the
    # covariances must match the windows, which all get it.
    if data_type == "eeg" and not any(
        proj["desc"].startswith("Average EEG") for proj in raw_fwd.info["projs"]
    ):
        raw_fwd.set_eeg_reference("average", projection=True, verbose=False)

    fwd = make_forward_solution(
        raw_fwd.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=(data_type == "meg"),
        eeg=(data_type == "eeg"),
    )
    if noise_cov_method == "ad_hoc":
        noise_cov = mne.make_ad_hoc_cov(raw_fwd.info)
    else:
        noise_cov = compute_raw_covariance(raw_fwd, method=noise_cov_method)
    if reg > 0.0 and noise_cov_method != "ad_hoc":
        noise_cov = mne.cov.regularize(
            noise_cov,
            raw_fwd.info,
            eeg=reg,
            mag=reg,
            grad=reg,
            verbose=False,
        )
    inverse_operator = None
    if make_inverse:
        if src_type == "volume":
            # A volume source space has no cortical normal, so orientation
            # cannot be constrained: MNE requires loose=1.0 and depth=None.
            if loose != 1.0 or depth is not None:
                logger.info(
                    "Volume source space: forcing loose=1.0, depth=None "
                    "(orientation cannot be constrained without a cortical surface)."
                )
            loose, depth = 1.0, None
        inverse_operator = make_inverse_operator(
            raw_fwd.info, fwd, noise_cov, loose=loose, depth=depth
        )

    # fwd["src"], as `src` may have been a path. `raw_fwd` is the recording the
    # forward and covariances were built from, to estimate a data covariance on.
    return inverse_operator, fwd, noise_cov, fwd["src"], raw_fwd


def weight_to_degree_map(n_nodes):
    """
    Construct linear mappings between edge weights and node degrees.

    Given a graph with `n_nodes`, this function builds a sparse matrix
    that maps edge weights to node degrees and its transpose. The
    resulting linear operators are useful in graph optimization problems.

    Parameters
    ----------
    n_nodes : int
            Number of nodes in the graph.

    Returns
    -------
    k : callable
            Function mapping edge weights (1D array of length n_edges) to
            node degrees (1D array of length n_nodes).
    kt : callable
            Function mapping node degrees back to edge weights.

    Notes
    -----
    The number of edges is ``n_edges = n_nodes * (n_nodes - 1) / 2``.
    Internally, a sparse COO matrix of shape ``(n_nodes, n_edges)`` is
    constructed.
    """

    n_edges = n_nodes * (n_nodes - 1) // 2

    row_idx1 = np.repeat(np.arange(n_nodes - 1), np.arange(n_nodes - 1, 0, -1))
    row_idx2 = np.concatenate([np.arange(i + 1, n_nodes) for i in range(n_nodes - 1)])

    row_idx = np.concatenate((row_idx1, row_idx2))
    col_idx = np.concatenate((np.arange(n_edges), np.arange(n_edges)))
    vals = np.ones(len(row_idx))

    coo = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(n_nodes, n_edges))

    return lambda w: coo.dot(w), lambda d: coo.T.dot(d)


_log_degree_barrier_cache: dict = {}


def log_degree_barrier(
    signals,
    dist_type,
    alpha,
    beta,
    step=0.5,
    max_iter=10000,
    rtol=1.0e-5,
    w0=None,
    normalize=True,
    cache_key=None,
):
    """
    Graph learning with a log-barrier degree constraint.

    Builds a weighted graph from smooth signals by solving::

        min_w  2 <w, z> - α Σ log(k(w)) + β ||w||²

    Parameters
    ----------
    signals : array of shape (n_nodes, n_samples)
            Input data (node-wise signals).
    dist_type : str
            Distance metric (passed to ``scipy.spatial.distance.pdist``).
    alpha : float
            Weight of the log-barrier degree penalty.
    beta : float
            Weight of the L2 regularization term.
    step : float
            Initial step size for the optimization.
    w0 : array or None
            Initial edge weights. ``None`` uses zeros (or the cached warm-start).
    max_iter : int
            Maximum iterations for the solver.
    rtol : float
            Relative tolerance for convergence.
    normalize : bool
            If True, normalize the returned adjacency matrix to [0, 1].
    cache_key : hashable or None
            When not None, the previous solution for this key is used as
            warm-start and updated in-place after each call.

    Returns
    -------
    graph_matrix : ndarray of shape (n_nodes, n_nodes)
            Learned (normalized) adjacency matrix.
    """
    n_nodes = signals.shape[0]

    # Fast path for 2-node case: analytic solution of the scalar problem
    # min_w 2wz - alpha*log(w) + beta*w^2, w >= 0
    # derivative: 2z - alpha/w + 2*beta*w = 0  →  2*beta*w^2 + 2z*w - alpha = 0
    if n_nodes == 2:
        z = pdist(signals, dist_type)
        z_norm = z[0] / (np.max(z) + 1e-12)
        if beta > 0:
            disc = 4 * z_norm**2 + 8 * beta * alpha
            w_opt = (-2 * z_norm + np.sqrt(max(disc, 0))) / (4 * beta)
        else:
            w_opt = alpha / (2 * z_norm + 1e-12)
        w_opt = max(w_opt, 0.0)
        mat = np.array([[0.0, w_opt], [w_opt, 0.0]])
        if normalize and w_opt > 0:
            mat /= w_opt
        return mat

    # General case
    z = pdist(signals, dist_type)
    z_max = np.max(z)
    if z_max > 0:
        z = z / z_max

    # Warm-starting from cache
    if w0 is None:
        w0 = _log_degree_barrier_cache.get(cache_key, np.zeros_like(z))

    k, kt = weight_to_degree_map(n_nodes)
    norm_k = np.sqrt(2 * (n_nodes - 1))

    if not _pyunlocbox_available:
        raise ImportError(
            "pyunlocbox is required for sensor_graph / source_graph modalities. "
            "Install it with: pip install pyunlocbox"
        )

    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - 2 * gamma * z)

    f2 = functions.func()
    f2._eval = lambda w: -alpha * np.sum(np.log(np.maximum(np.finfo(float).eps, k(w))))
    f2._prox = lambda d, gamma: np.maximum(0, 0.5 * (d + np.sqrt(d**2 + 4 * alpha * gamma)))

    f3 = functions.func()
    f3._eval = lambda w: beta * np.sum(w**2)
    f3._grad = lambda w: 2 * beta * w
    lipg = 2 * beta

    stepsize = step / (1 + lipg + norm_k)
    solver = _pux_solvers.mlfbf(L=k, Lt=kt, step=stepsize)
    problem = _pux_solvers.solve(
        [f1, f2, f3],
        x0=w0.copy(),
        solver=solver,
        maxit=max_iter,
        rtol=rtol,
        verbosity="NONE",
    )
    sol = problem["sol"]

    # Store warm-start for next call
    if cache_key is not None:
        _log_degree_barrier_cache[cache_key] = sol.copy()

    mat = squareform(sol)
    if normalize:
        m = mat.max()
        if m > 0:
            mat = mat / m

    return mat


def plot_glass_brain(bl1, bl2=None):
    """Plot one or two brain labels on an fsaverage glass brain.

    Parameters
    ----------
    bl1 : str
            Name of the first brain label (must exist in the 'aparc' parcellation).
    bl2 : str | None, optional
            Name of the second brain label. If None (default), only ``bl1`` is
            displayed.

    Returns
    -------
    fig_brain : matplotlib.figure.Figure
            Figure containing the glass brain plots with highlighted labels.

    Notes
    -----
    This function uses the ``fsaverage`` subject and the ``aparc`` parcellation
    from FreeSurfer. The labels are displayed on a cortical surface rendering
    using four different views (frontal, dorsal, frontal, lateral).
    """
    brain_kwargs = dict(alpha=0.15, background="white", cortex="low_contrast", size=(800, 600))
    brain_labels = read_labels_from_annot(subject="fsaverage", parc="aparc")
    bl_names = [bl.name for bl in brain_labels]
    views = ["frontal", "dorsal", "frontal", "frontal"]
    azimuths = [180, 0, 0, -90]
    fig_brain, axs = plt.subplots(1, 4, figsize=(12, 3))

    for view, azimuth, ax in zip(views, azimuths, axs):
        create_3d_figure(size=(100, 100), bgcolor=(0, 0, 0))
        Brain = get_brain_class()
        brain = Brain("fsaverage", hemi="both", surf="pial", **brain_kwargs)

        if bl2 is None:
            idx = bl_names.index(bl1)
            brain.add_label(
                brain_labels[idx], hemi="both", color="#d62728", borders=False, alpha=0.8
            )

        if bl2 is not None:
            for bl, color in zip([bl1, bl2], ["#1f77b4", "#d62728"]):
                idx = bl_names.index(bl)
                brain.add_label(
                    brain_labels[idx], hemi="both", color=color, borders=False, alpha=0.8
                )

        brain.show_view(view=view, azimuth=azimuth)
        img = brain.screenshot()
        ax.imshow(img)
        ax.axis("off")
    fig_brain.tight_layout()

    return fig_brain


def remove_blinks_lms(data, ref_ch_idx=0, n_taps=5, mu=0.01):
    """Remove blink artifacts from EEG/MEG using an adaptive LMS filter.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input M/EEG data.
    ref_ch_idx : int, default 0
        Index of the reference (artifact) channel (e.g. Fp1, EOG).
    n_taps : int, default 5
        Number of tapped-delay filter coefficients.
    mu : float, default 0.01
        LMS step size (learning rate).

    Returns
    -------
    cleaned : ndarray, shape (n_channels, n_times)
        Artifact-attenuated data.
    """
    n_channels, n_times = data.shape
    ref = data[ref_ch_idx]

    # Build tapped-delay matrix: column k = ref shifted k samples
    X = np.zeros((n_times, n_taps))
    for k in range(n_taps):
        X[k:, k] = ref[: n_times - k]

    cleaned = data.copy()
    W = np.zeros((n_channels, n_taps))

    for t in range(n_times):
        x = X[t]  # (n_taps,)
        y = W @ x  # (n_ch,) estimated artifact
        e = data[:, t] - y  # (n_ch,) residual
        cleaned[:, t] = e
        W += mu * np.outer(e, x)  # batch weight update for all channels

    # Reference channel is not filtered
    cleaned[ref_ch_idx] = data[ref_ch_idx]
    return cleaned


def create_blink_template(
    raw,
    max_iter=800,
    method="infomax",
    n_components=0.95,
    fit_params=None,
    random_state=0,
    iclabel_threshold=0.5,
):
    """Create a template for the eye-blink ICA component from raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (already filtered and with a montage set).
    max_iter : int, default 800
        Maximum ICA fitting iterations.
    method : str, default "infomax"
        ICA algorithm (``"infomax"``, ``"fastica"``, ``"picard"``).
    n_components : int | float | None, default 0.95
        PCA dimensionality before ICA.  Float in (0, 1) → variance fraction.
        Falls back to ``5`` if the initial fit raises an exception.
    fit_params : dict | None, default None
        Extra solver arguments.  ``None`` uses ``{"extended": True}`` for
        infomax (recommended for biological signals).
    random_state : int | None, default 0
        Random seed for reproducibility.
    iclabel_threshold : float, default 0.5
        Minimum ICLabel ``"eye blink"`` probability required to accept a
        component as a blink source.

    Returns
    -------
    template_blink_comp : np.ndarray or None
        Spatial topography of the blink component (shape: n_channels,), or
        ``None`` if no component exceeds *iclabel_threshold*.
    """
    _fit_params = fit_params if fit_params is not None else {"extended": True}

    ica = ICA(
        n_components=n_components,
        max_iter=max_iter,
        method=method,
        fit_params=_fit_params,
        random_state=random_state,
    )
    try:
        ica.fit(raw)
    except Exception:
        ica = ICA(
            n_components=5,
            max_iter=max_iter,
            method=method,
            fit_params=_fit_params,
            random_state=random_state,
        )
        ica.fit(raw)

    ic_dict = label_components(raw, ica, method="iclabel")
    ic_labels = ic_dict["labels"]
    ic_probs = ic_dict["y_pred_proba"]

    eye_indices = [
        i
        for i, lbl in enumerate(ic_labels)
        if lbl == "eye blink" and ic_probs[i] >= iclabel_threshold
    ]

    template_blink_comp = None
    if eye_indices:
        blink_idx = max(eye_indices, key=lambda i: ic_probs[i])
        template_blink_comp = ica.get_components()[:, blink_idx]

    return template_blink_comp


def setup_surface(subjects_dir=None, hemi_distance=20.0, surf="inflated", grade=None):
    """Prepare a PyVista mesh of the fsaverage brain for both hemispheres.

    Uses the MNE-bundled ``fsaverage`` subject (auto-downloaded on first
    call via :func:`mne.datasets.fetch_fsaverage`).  The ``subjects_dir``
    parameter is accepted for API compatibility but is no longer required.

    Parameters
    ----------
    subjects_dir : str | Path | None
            Ignored — kept for backwards compatibility.
    hemi_distance : float, default 100.0
            Gap in mm between the medial walls of the two hemispheres.
            Each hemisphere is centred so that its medial face sits at
            ±hemi_distance/2, giving a consistent gap across all surfaces.
    surf : str, default "inflated"
            Surface geometry to load.  One of ``"inflated"``, ``"pial"``,
            ``"white"``, ``"sphere"``.
    grade : int | None, default None
            Build the mesh of the ico-``grade`` subdivision instead of the full
            surface. fsaverage numbers its vertices hierarchically, so those
            are its first vertices and MNE's icosahedron triangles join them.
            ``5`` gives exactly the 10 242 vertices per hemisphere of an ico-5
            source space, so ``nn_map`` is then the identity.

            .. versionadded:: 1.3.0

    Returns
    -------
    hemi_offsets : dict
            Mapping ``{"lh": 0, "rh": n_lh_vertices}``.
    scalars_full : ndarray, shape (n_total_vertices,)
            Zero-initialised activity array covering both hemispheres.
    mesh : pyvista.PolyData
            Combined bilateral mesh with ``"base"`` (sulcal depth) and
            ``"activity"`` scalar arrays.
    verts_stc : dict
            Mapping ``{"lh": ndarray, "rh": ndarray}`` of ico-5 source
            vertex indices (0..10241 for each hemisphere).
    nn_map : dict
            Mapping ``{"lh": ndarray, "rh": ndarray}``.  For each full-
            surface vertex, the index of the nearest ico-5 source vertex.
            Used to spread source-space values to the full mesh so that
            the activity overlay looks smooth rather than point-like.
    """
    if not _pyvista_available:
        raise ImportError(
            "pyvista is required for brain surface visualisation. "
            "Install it with:  pip install 'ANT[viz]'"
        )
    from scipy.spatial import cKDTree

    fs_dir = Path(fetch_fsaverage(verbose=False))
    set_3d_backend("pyvistaqt")

    # Load sulcal depth
    lh_sulc = read_morph_data(fs_dir / "surf" / "lh.sulc")
    rh_sulc = read_morph_data(fs_dir / "surf" / "rh.sulc")
    sulc_all = np.hstack([lh_sulc, rh_sulc])

    n_src_verts = 10242  # ico-5 source space vertices per hemisphere
    verts_all = []
    faces_all = []
    offset = 0
    hemi_offsets = {}
    nn_map = {}

    # Load surfaces, build nn_map, then centre each hemisphere symmetrically
    for hemi in ["lh", "rh"]:
        surf_path = fs_dir / "surf" / f"{hemi}.{surf}"
        verts_surf, faces_surf = read_surface(surf_path)
        if grade is not None:
            from mne.surface import _get_ico_surface

            faces_surf = _get_ico_surface(grade)["tris"]
            verts_surf = verts_surf[: faces_surf.max() + 1]

        # Nearest-neighbour map: index of closest source vertex for every
        # full-surface vertex (computed before translation, same coord frame)
        _, nn = cKDTree(verts_surf[:n_src_verts]).query(verts_surf)
        nn_map[hemi] = nn  # shape (n_hemi_verts,), values in [0, n_src_verts)

        # Centre each hemisphere so medial wall sits at ±hemi_distance/2.
        # This keeps the gap consistent across surfaces (inflated/pial/white).
        if hemi == "lh":
            verts_surf[:, 0] -= verts_surf[:, 0].max() + hemi_distance / 2
        else:
            verts_surf[:, 0] -= verts_surf[:, 0].min() - hemi_distance / 2

        hemi_offsets[hemi] = offset
        verts_all.append(verts_surf)

        # Convert faces to PyVista format
        faces_pv = (
            np.hstack([np.full((faces_surf.shape[0], 1), 3), faces_surf + offset])
            .astype(np.int64)
            .ravel()
        )
        faces_all.append(faces_pv)

        offset += verts_surf.shape[0]

    # Combine hemispheres
    verts_all = np.vstack(verts_all)
    faces_all = np.hstack(faces_all)

    # Create PyVista mesh
    mesh = pv.PolyData(verts_all, faces_all)
    if grade is not None:
        n = hemi_offsets["rh"]
        sulc_all = np.hstack([lh_sulc[:n], rh_sulc[:n]])
    mesh["base"] = sulc_all

    verts_stc = {
        "lh": np.arange(n_src_verts),
        "rh": np.arange(n_src_verts),
    }
    scalars_full = np.zeros(mesh.n_points)
    mesh["activity"] = scalars_full

    return hemi_offsets, scalars_full, mesh, verts_stc, nn_map


def setup_plotter(mesh, clim=[0, 0.6], camera_position="yz", azimuth=45):
    """Initialize PyVista plotter, add mesh, and set camera."""
    plotter = pv.Plotter(window_size=(1800, 1200), lighting="three lights")
    plotter.set_background("black")

    # Add base mesh (sulcal depth)
    plotter.add_mesh(mesh, scalars="base", cmap="Greys", smooth_shading=True, show_scalar_bar=False)

    # Add activity overlay as semi-transparent layer
    plotter.add_mesh(
        mesh,
        scalars="activity",
        cmap="hot",
        opacity=0.6,
        clim=clim,
        smooth_shading=True,
        show_scalar_bar=False,
        interpolate_before_map=True,
    )
    plotter.add_scalar_bar(
        title="Activity",
        italic=True,
        vertical=True,
        position_x=0.85,
        position_y=0.1,
        height=0.8,
        color="white",
        title_font_size=16,
        label_font_size=14,
    )
    plotter.enable_eye_dome_lighting()

    # Set camera
    plotter.camera_position = camera_position
    plotter.camera.azimuth = azimuth
    plotter.show(interactive_update=True, auto_close=False)

    return plotter
