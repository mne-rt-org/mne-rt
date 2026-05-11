.. _denoising:

Artifact Correction
===================

ANT provides five real-time artifact correction methods that can be selected
via :class:`~ant.NFRealtime` (``artifact_correction=`` parameter) or the
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

**Class:** no public API — used internally via ``artifact_correction="lms"``

The Least Mean Squares (LMS) algorithm
:footcite:p:`Widrow1960` is a stochastic gradient-descent adaptive filter
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
largest eigenvalue of the reference autocorrelation matrix.  ANT uses a
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

**Class:** :class:`ant.tools.ORICA`

ORICA :footcite:p:`Choi2002` is an online (sample-recursive) variant of
Independent Component Analysis (ICA) :footcite:p:`BellSejnowski1995` that
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

**Class:** :class:`ant.tools.GEDAIDenoiser`

GEDAI uses Generalized Eigendecomposition (GED) :footcite:p:`Cohen2022` to find
spatial filters that maximally separate brain signal from artifact
subspaces estimated from a baseline recording and (optionally) a
leadfield-derived forward model.

**GED formulation.** Given a *signal* covariance :math:`\mathbf{S}` (broadband
or band-limited EEG) and a *noise* covariance :math:`\mathbf{N}` (artifact
reference or baseline residual), GED finds the matrix :math:`\mathbf{W}` that
simultaneously diagonalises both:

.. math::

   \mathbf{S}\,\mathbf{W} = \mathbf{N}\,\mathbf{W}\,\boldsymbol{\Lambda}

The columns of :math:`\mathbf{W}` are the *generalized eigenvectors*
(spatial filters), and the diagonal entries of :math:`\boldsymbol{\Lambda}`
are the *generalized eigenvalues* (signal-to-noise ratios per filter).

**Artifact isolation.** Filters with large :math:`\lambda_k`
capture artifact-dominated directions.  These are projected out:

.. math::

   \hat{\mathbf{X}} = \mathbf{W}_{\text{brain}}\,
   \mathbf{W}_{\text{brain}}^\top\, \mathbf{X}

where :math:`\mathbf{W}_{\text{brain}}` retains only filters with eigenvalues
below a threshold :math:`\lambda_{\text{thresh}}`.

**Leadfield incorporation.** When a forward model is available, GEDAI
constrains the signal subspace to be consistent with neural generators,
improving separation from far-field artifacts (e.g., ECG).

**When to use.** Most powerful when a resting-state baseline is available
and the artifact has a stable spatial signature (cardiac, powerline).
Requires a calibration recording (``fit_gedai()`` runs a 60 s baseline).

----

.. _denoising-asr:

ASR — Artifact Subspace Reconstruction
---------------------------------------

**Class:** :class:`ant.tools.ASRDenoiser`

ASR :footcite:p:`Mullen2015` :footcite:p:`deCheveigneArzounian2018` learns the
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

**Class:** :class:`ant.tools.RTMaxwellFilter`

The Signal Space Separation (SSS) method :footcite:p:`Taulu2004` and its
temporal extension (tSSS) :footcite:p:`Taulu2006` exploit the physics of
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
tSSS :footcite:p:`Taulu2006` removes these correlated components by projecting
the internal coefficients onto the subspace *orthogonal* to the external ones:

.. math::

   \hat{\boldsymbol{\alpha}} = \bigl(\mathbf{I}
   - \Pi_{\mathrm{corr}}\bigr)\, \boldsymbol{\alpha}

where :math:`\Pi_{\mathrm{corr}}` is the projector onto the directions of
:math:`\boldsymbol{\alpha}` that correlate above threshold with the external
coefficient matrix over the buffer.

**Real-time implementation in ANT.**
:class:`~ant.tools.RTMaxwellFilter` caches :math:`\mathbf{P}_{\mathrm{SSS}}`
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
:class:`~ant.tools.RTMaxwellFilter` uses a system-identification approach:
MNE's full Maxwell filter (with noise-informed regularisation) is applied to a
Gaussian test signal, and the effective operator is recovered by least squares:

.. math::

   \mathbf{P}_{\mathrm{SSS}}^{\mathrm{ER}} =
   \mathbf{Y}\, \mathbf{X}^\dagger

where :math:`\mathbf{X}` is the test input (good-channel MEG) and
:math:`\mathbf{Y}` is the filtered output.  This transparently incorporates
fine calibration, cross-talk compensation, and noise-informed regularisation
into the single cached matrix — no re-implementation of SSS is required.

**Baseline requirement.** Unlike ASR or GEDAI, :class:`~ant.tools.RTMaxwellFilter`
requires *no* baseline recording — :math:`\mathbf{P}_{\mathrm{SSS}}` depends
only on sensor geometry, which is fixed at scanner installation.

**When to use.** MEG data only.  Mandatory first denoising step in any MEG
pipeline.  Use SSS mode for environmental noise suppression; add tSSS when
nearby ferromagnetic objects or implants are present.

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
     - Yes (60 s)
     - No
     - No
     - Cardiac, powerline
   * - ASR
     - EEG / MEG
     - Yes (60 s)
     - No
     - No
     - Transient high-amplitude
   * - RTMaxwellFilter
     - MEG only
     - No
     - No
     - No (periodic tSSS)
     - Environmental noise

----

References
----------

.. footbibliography::
