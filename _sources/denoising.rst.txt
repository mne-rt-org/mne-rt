.. _denoising:

Artifact Correction
===================

MNE-RT provides five real-time artifact correction methods that can be selected
via :class:`~mne_rt.RTStream` (``artifact_correction=`` parameter) or the
:doc:`CLI <cli>` (``--artifact-correction``).  All methods expose a common
``fit`` / ``transform`` interface and operate sample-by-sample (or
chunk-by-chunk) during the closed-loop session.

.. contents:: Methods
   :local:
   :depth: 1

----

.. _denoising-lms:

Adaptive LMS filter
-------------------

**Class:** :class:`mne_rt.tools.AdaptiveLMSFilter`

The Least Mean Squares (LMS) algorithm
:footcite:p:`widrow1988adaptive` is a stochastic gradient-descent adaptive filter
designed to cancel a reference signal (e.g., an EOG or ECG channel) from
the M/EEG data channels.

**Model.** For channel :math:`i` at time :math:`t`, the cleaned signal is:

.. math::

   y_i(t) = x_i(t) - \mathbf{w}_i(t)^\top \mathbf{r}(t)

where :math:`x_i(t)` is the raw sample, :math:`\mathbf{r}(t) \in \mathbb{R}^q`
is the reference signal vector (possibly delayed copies of the artefact channel),
and :math:`\mathbf{w}_i(t)` is the adaptive weight vector.

**Weight update** (Widrow–Hoff rule):

.. math::

   \mathbf{w}_i(t+1) = \mathbf{w}_i(t) + \mu\, y_i(t)\, \mathbf{r}(t)

where :math:`\mu > 0` is the step size (learning rate).  The algorithm
converges when the cross-correlation between :math:`\mathbf{r}(t)` and the
residual :math:`y_i(t)` reaches zero — i.e., when the filter has learned to
predict the artifact contribution in channel :math:`i` from the reference.

**Stability condition.** The step size must satisfy
:math:`0 < \mu < 2/(\lambda_{\max}\, q)`, where :math:`\lambda_{\max}` is the
largest eigenvalue of the reference autocorrelation matrix.  MNE-RT uses a
normalised variant (NLMS) that adapts :math:`\mu` automatically:

.. math::

   \mu_{\mathrm{norm}}(t) = \frac{\mu_0}{\epsilon + \|\mathbf{r}(t)\|^2}

**When to use.** Best suited for ocular (EOG) or cardiac (ECG) artifacts
when a clean reference channel is available.  Requires no calibration
baseline — the filter adapts online from sample 1.

----

.. _denoising-orica:

ORICA — Online Recursive ICA
-----------------------------

**Class:** :class:`mne_rt.tools.ORICA`

ORICA :footcite:p:`choi2005blind` is an online (sample-recursive) variant of
Independent Component Analysis (ICA) :footcite:p:`bell1995information` that
continuously updates the unmixing matrix :math:`\mathbf{W}` as new data arrive.

**ICA model.** EEG is assumed to be a linear mixture of independent sources:

.. math::

   \mathbf{x}(t) = \mathbf{A}\, \mathbf{s}(t),
   \quad \mathbf{s}(t) = \mathbf{W}\, \mathbf{x}(t)

where :math:`\mathbf{A} \in \mathbb{R}^{p \times p}` is the mixing matrix,
:math:`\mathbf{W} = \mathbf{A}^{-1}` is the unmixing matrix, and
:math:`\mathbf{s}(t)` are the independent components (ICs).

**Recursive update.** ORICA updates :math:`\mathbf{W}` with each new sample
using an exponentially weighted natural-gradient rule:

.. math::

   \mathbf{W}(t) \leftarrow \mathbf{W}(t)
   + \lambda(t)\,
   \bigl[\mathbf{I} - \mathbf{f}\!\left(\mathbf{s}(t)\right)\mathbf{s}(t)^\top\bigr]\,
   \mathbf{W}(t)

where :math:`\mathbf{f}(\cdot)` is a nonlinear score function (e.g.,
:math:`f(s) = \tanh(s)` for super-Gaussian sources) and :math:`\lambda(t)` is
a decreasing forgetting factor that controls how fast old data are discarded.

**Artifact removal.** ICs whose kurtosis, focal power, or spatial map
match known artifact signatures (blinks, muscle, ECG) are identified and
zeroed before back-projection:

.. math::

   \hat{\mathbf{x}}(t) = \mathbf{A}_{\text{clean}}\,
   \mathbf{W}_{\text{clean}}\, \mathbf{x}(t)

where the subscript "clean" denotes the subsets of columns / rows with
artifact ICs removed.

**When to use.** Suitable for EEG with diverse artifact types.  No reference
channel is required.  Because :math:`\mathbf{W}` adapts continuously, ORICA
tracks slow non-stationarities (electrode drift, changing impedances).

----

.. _denoising-gedai:

GEDAI — GED-based Artifact Isolation
--------------------------------------

**Class:** :class:`mne_rt.tools.GEDAIDenoiser`

GEDAI uses Generalized Eigendecomposition (GED) :footcite:p:`cohen2022tutorial` to find
spatial filters that maximally separate brain signal from broadband activity,
optionally anchored to a leadfield-derived forward model :footcite:p:`ros2025return`.

**GED formulation.** MNE-RT uses a *band-vs-broadband* formulation.
Let :math:`\mathbf{R}_\text{band}` be the covariance of the band-limited signal
(e.g. 8–30 Hz for alpha/beta) and :math:`\mathbf{R}_\text{broad}` be the
broadband covariance.  The GED solves:

.. math::

   \mathbf{R}_\text{band}\,\mathbf{W} =
   \mathbf{R}_\text{broad}\,\mathbf{W}\,\boldsymbol{\Lambda}

The columns of :math:`\mathbf{W}` are spatial filters and the eigenvalues
:math:`\lambda_k = \mathbf{w}_k^\top \mathbf{R}_\text{band}\, \mathbf{w}_k \;/\;
\mathbf{w}_k^\top \mathbf{R}_\text{broad}\, \mathbf{w}_k`
express the *band-to-broadband power ratio* along each filter direction.

**Eigenvalue interpretation.**  Filters are sorted by :math:`\lambda_k`
in **descending** order:

* **Large** :math:`\lambda_k` — high band/broadband ratio → **brain-like**
  (targeted oscillation dominant).
* **Small** :math:`\lambda_k` — low band/broadband ratio → **non-brain**
  (broadband, artifact-like).

**Artifact isolation.** :meth:`~mne_rt.tools.GEDAIDenoiser.find_noise_components`
returns the :math:`n_\text{noise}` components with the **smallest** eigenvalues
(least band-specific), which are then zeroed before back-projection:

.. math::

   \hat{\mathbf{X}} = \mathbf{W}_{\text{brain}}\,
   \mathbf{W}_{\text{brain}}^\dagger\, \mathbf{X}

where :math:`\mathbf{W}_{\text{brain}}` is the subset of :math:`\mathbf{W}`
with the largest eigenvalues (brain-like directions retained).

**Leadfield incorporation.** When a forward model is available
(:meth:`~mne_rt.tools.GEDAIDenoiser.fit_from_leadfield`), GEDAI replaces
:math:`\mathbf{R}_\text{band}` with the leadfield outer product
:math:`\mathbf{L}\mathbf{L}^\top`, directly constraining filters to be
consistent with neural generators.

**When to use.** Most powerful when a resting-state baseline is available
and the artifact has a stable spatial signature (cardiac, powerline).
Requires a calibration recording (``fit_from_raw()`` uses the first N seconds).

----

.. _denoising-asr:

ASR — Artifact Subspace Reconstruction
---------------------------------------

**Class:** :class:`mne_rt.tools.ASRDenoiser`

ASR :footcite:p:`mullen2015real` :footcite:p:`de2018robust` learns the
covariance statistics of a clean baseline segment, then projects out
components that deviate beyond a threshold in subsequent data windows.

**Calibration.** The baseline recording is divided into overlapping windows of
length :math:`T_{\mathrm{win}}`.  The sample covariance per window is:

.. math::

   \mathbf{C}_j = \frac{1}{n_j} \mathbf{X}_j \mathbf{X}_j^\top

Windows with the highest total power (top ``max_dropout_fraction`` fraction
by :math:`\operatorname{tr}(\mathbf{C}_j)`) are discarded as likely artifactual.
The mean of the remaining *clean* covariances is eigendecomposed:

.. math::

   \mathbf{C}_0 = \mathbf{U}\,\mathrm{diag}(d_1,\dots,d_p)\,\mathbf{U}^\top

Per-component RMS thresholds are set as:

.. math::

   T_k = \sigma \cdot \sqrt{d_k}, \qquad k = 1,\dots,p

where :math:`\sigma` = ``cutoff`` (default 5).

**Cleaning (online).** For each incoming window
:math:`\mathbf{X} \in \mathbb{R}^{p \times n}`:

1. Project to calibration-component space:
   :math:`\mathbf{Z} = \mathbf{U}^\top(\mathbf{X} - \bar{\mathbf{x}})`.
2. Compute per-component RMS:
   :math:`r_k = \sqrt{\operatorname{mean}(Z_k^2)}`.
3. Zero artifact-dominated components: :math:`Z_k \leftarrow 0` if
   :math:`r_k > T_k`.
4. Back-project and restore the mean:
   :math:`\hat{\mathbf{X}} = \mathbf{U}\mathbf{Z} + \bar{\mathbf{x}}`.

Because steps 1–4 reduce to two matrix multiplies, the per-chunk runtime is
:math:`\mathcal{O}(p^2 n)`.

**When to use.** General-purpose EEG artifact suppressor.  Robust to
transient high-amplitude artifacts (muscle bursts, electrode pops).
Requires a brief clean baseline (``fit_asr()`` records 60 s by default).

----

.. _denoising-maxwell:

RTMaxwellFilter — Real-Time SSS / tSSS
----------------------------------------

**Class:** :class:`mne_rt.tools.RTMaxwellFilter`

The Signal Space Separation (SSS) method :footcite:p:`taulu2004suppression` and its
temporal extension (tSSS) :footcite:p:`taulu2006spatiotemporal` exploit the physics of
quasi-static magnetic fields to separate brain-origin signals from external
interference in MEG recordings.

**Physical basis.** Inside the sensor array the magnetic field satisfies the
Laplace equation :math:`\nabla^2 \phi = 0`.  Solutions are decomposed into a
basis of *spherical harmonic multipoles*:

.. math::

   \phi(\mathbf{r}) =
   \sum_{\ell=1}^{L_{\mathrm{in}}} \sum_{m=-\ell}^{\ell}
   \alpha_{\ell m}\, \phi_{\ell m}^{\mathrm{in}}(\mathbf{r})
   \;+\;
   \sum_{\ell=1}^{L_{\mathrm{ex}}} \sum_{m=-\ell}^{\ell}
   \beta_{\ell m}\, \phi_{\ell m}^{\mathrm{ex}}(\mathbf{r})

where :math:`\phi^{\mathrm{in}}_{\ell m}` are the *internal* basis functions
(sources inside the array — brain signals) and
:math:`\phi^{\mathrm{ex}}_{\ell m}` are the *external* basis functions
(sources outside — environmental noise).

**SSS projector.** Let :math:`\mathbf{S}_{\mathrm{in}}` and
:math:`\mathbf{S}_{\mathrm{ex}}` be the basis matrices evaluated at all
sensor positions.  The SSS projector onto the internal subspace is:

.. math::

   \mathbf{P}_{\mathrm{SSS}} = \mathbf{S}_{\mathrm{in}}
   \mathbf{S}_{\mathrm{in}}^\dagger

where :math:`(\cdot)^\dagger` denotes the Moore–Penrose pseudoinverse.
Applied to raw MEG data :math:`\mathbf{X}`:

.. math::

   \hat{\mathbf{X}}_{\mathrm{SSS}} = \mathbf{P}_{\mathrm{SSS}}\, \mathbf{X}

This single matrix multiply (cached at calibration time) suppresses all signal
components that cannot originate inside the sensor array.

**tSSS — temporal extension.** Internal components that correlate with
external ones over a sliding time window indicate that interference has
"leaked" into the internal space (e.g., due to a nearby ferromagnetic object).
tSSS :footcite:p:`taulu2006spatiotemporal` removes these correlated components by projecting
the internal coefficients onto the subspace *orthogonal* to the external ones:

.. math::

   \hat{\boldsymbol{\alpha}} = \bigl(\mathbf{I}
   - \Pi_{\mathrm{corr}}\bigr)\, \boldsymbol{\alpha}

where :math:`\Pi_{\mathrm{corr}}` is the projector onto the directions of
:math:`\boldsymbol{\alpha}` that correlate above threshold with the external
coefficient matrix over the buffer.

**Real-time implementation in MNE-RT.**
:class:`~mne_rt.tools.RTMaxwellFilter` caches :math:`\mathbf{P}_{\mathrm{SSS}}`
once during ``fit()`` using :func:`mne.preprocessing.compute_maxwell_basis`,
then applies it as a matrix multiply per incoming chunk.  This gives:

- Zero additional latency beyond a matrix multiply per chunk.
- Numerical equivalence with offline MNE
  :func:`~mne.preprocessing.maxwell_filter` (verified to machine-epsilon
  level).

tSSS runs periodically on a rolling buffer with
:func:`~mne.preprocessing.maxwell_filter` (``st_only=True``, skipping the
already-applied spatial step).

**Empty-room noise covariance.** When an empty-room recording is provided,
:class:`~mne_rt.tools.RTMaxwellFilter` uses a system-identification approach:
MNE's full Maxwell filter (with noise-informed regularisation) is applied to a
Gaussian test signal, and the effective operator is recovered by least squares:

.. math::

   \mathbf{P}_{\mathrm{SSS}}^{\mathrm{ER}} =
   \mathbf{Y}\, \mathbf{X}^\dagger

where :math:`\mathbf{X}` is the test input (good-channel MEG) and
:math:`\mathbf{Y}` is the filtered output.  This transparently incorporates
fine calibration, cross-talk compensation, and noise-informed regularisation
into the single cached matrix — no re-implementation of SSS is required.

**Baseline requirement.** Unlike ASR or GEDAI, :class:`~mne_rt.tools.RTMaxwellFilter`
requires *no* baseline recording — :math:`\mathbf{P}_{\mathrm{SSS}}` depends
only on sensor geometry, which is fixed at scanner installation.

**When to use.** MEG data only.  Mandatory first denoising step in any MEG
pipeline.  Use SSS mode for environmental noise suppression; add tSSS when
nearby ferromagnetic objects or implants are present.

----

.. _denoising-bad-channel:

Bad Channel Detection
---------------------

**Class:** :class:`mne_rt.tools.BadChannelDetector`

Artifact correction assumes all channels are functional.  A disconnected or
noisy electrode contaminating adjacent channels can degrade every downstream
method.  :class:`~mne_rt.tools.BadChannelDetector` evaluates each incoming data
window against up to four independent criteria and uses a rolling-window
majority vote to flag persistently bad channels.

**Criteria**

``"flat"``
    Channels whose RMS amplitude falls below ``flat_threshold``.  Catches
    disconnected leads and dead electrodes.

``"variance"``
    Channels whose RMS is a *robust* outlier across all channels, measured
    by a MAD-based z-score:
    :math:`z = (\text{RMS} - \text{median})\;/\;(\text{MAD} \times 1.4826)`.
    Flags both excessively noisy and suspiciously quiet channels.

``"correlation"``
    Channels whose mean Pearson correlation with their :math:`K` nearest
    spatial neighbours falls below ``corr_threshold``.  Requires channel
    positions to be set in ``info`` (montage applied).

``"hf_noise"``
    Channels with an abnormally high ratio of HF power (> ``hf_cutoff`` Hz)
    to broadband power, using the same MAD z-score.  Catches EMG, loose
    cable noise, and intermittent contacts.

**Rolling-window vote.**  A channel is declared bad only when it is flagged
by at least one criterion in ≥ ``min_bad_frac`` of the last
``history_windows`` windows.  With the default ``min_bad_frac=0.5`` this
is a majority vote over the past 30 windows, suppressing false positives
from transient artifacts.

**Usage example**::

    detector = BadChannelDetector(raw.info, method="all")
    while streaming:
        window = stream.get_data(1.0)        # 1-second chunk
        bad_channels = detector.update(window)
        raw.info["bads"] = bad_channels

----

.. _denoising-riemannian-potato:

Riemannian Potato — Online Artifact Detection
---------------------------------------------

**Class:** :class:`mne_rt.tools.RiemannianPotatoDetector`

The *Riemannian Potato* :footcite:p:`barachant2014plug,barthelemy2019riemannian` is a
covariance-based method for **detecting** — rather than correcting —
artifactual data windows in real time.  It operates on the manifold of
symmetric positive-definite (SPD) matrices by computing the Riemannian
distance from each incoming covariance matrix to the geometric mean of a
clean calibration set.  When this distance exceeds a z-score threshold,
the window is flagged as an artifact and can be discarded or buffered for
later ICA clean-up.

**Why Riemannian geometry?**  Standard Euclidean distances on covariance
matrices are sensitive to the arbitrary scaling of individual channels.
The Riemannian (affine-invariant) metric is scale-invariant: multiplying
all channels by any positive constant leaves the distance unchanged.  This
makes the potato robust to amplitude drift and electrode-impedance changes
across sessions.

**Algorithm.**  Let :math:`\Sigma_t` be the sample covariance matrix of
window :math:`t` and :math:`\bar{\Sigma}` be the geometric mean of
calibration covariances.  The z-score is:

.. math::

    z_t = \frac{\delta(\Sigma_t, \bar{\Sigma}) - \mu_\delta}{\sigma_\delta}

where :math:`\delta` is the Riemannian distance and :math:`\mu_\delta`,
:math:`\sigma_\delta` are the mean and standard deviation of distances
computed over the calibration set.  Window :math:`t` is declared an
artifact when :math:`z_t > \theta` (default :math:`\theta = 3`).

.. note::

   The Riemannian Potato is a **detection** method, not a correction method.
   It identifies and rejects artifactual windows; it does not reconstruct
   clean signal from contaminated data.  Pair it with
   :class:`~mne_rt.tools.ORICA` or :class:`~mne_rt.tools.ASRDenoiser` when online
   *correction* is needed.

**Usage example**::

    from mne_rt.tools import RiemannianPotatoDetector

    # Calibrate on 60 clean 1-second windows
    detector = RiemannianPotatoDetector(threshold=3.0)
    detector.fit(clean_windows)       # shape (n_windows, n_ch, n_samples)

    while streaming:
        window = stream.get_data(1.0)
        is_clean, z_score = detector.detect(window)
        if is_clean:
            compute_nf_feature(window)
        else:
            logger.debug("Artifact detected (z=%.2f) — window skipped", z_score)

**When to use.**  Any EEG or MEG setup where transient artifacts (blinks,
muscle bursts, electrode pops) need to be excluded from the feature
computation.  Particularly effective when reliable reference channels for
LMS are unavailable.  Requires a short clean calibration phase (≥ 30 windows
recommended) before streaming begins.

----

.. _denoising-comparison:

Method comparison
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 12 14 14 14 14

   * - Method
     - Modality
     - Baseline needed
     - Reference ch.
     - Adapts online
     - Primary target
   * - LMS
     - EEG / MEG
     - No
     - Yes (EOG/ECG)
     - Yes
     - Ocular, cardiac
   * - ORICA
     - EEG / MEG
     - No
     - No
     - Yes
     - Mixed artifacts
   * - GEDAI
     - EEG / MEG
     - Yes (N s)
     - No
     - No
     - Cardiac, powerline
   * - ASR
     - EEG / MEG
     - Yes (N s)
     - No
     - No
     - Transient high-amplitude
   * - RTMaxwellFilter
     - MEG only
     - No
     - No
     - No (periodic tSSS)
     - Environmental noise
   * - BadChannelDetector
     - EEG / MEG
     - No
     - No
     - Yes (rolling)
     - Disconnected / noisy channels
   * - RiemannianPotatoDetector
     - EEG / MEG
     - Yes (N windows)
     - No
     - No (detection only)
     - Transient artifacts (blinks, muscle, pops)

----

See also the :ref:`artifact comparison example <sphx_glr_auto_examples_ex_artifact_comparison.py>`
for a quantitative benchmark of LMS, ASR, and GEDAI on simulated EEG, and the
:ref:`RTMaxwell example <sphx_glr_auto_examples_ex_maxwell_realtime.py>`
for a demonstration of real-time SSS on the MNE sample MEG dataset.

----

References
----------

.. footbibliography::
