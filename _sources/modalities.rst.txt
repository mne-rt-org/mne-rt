.. _modalities:

Real-time Feature Modalities
============================

MNE-RT implements 20 real-time feature modalities spanning sensor-space and
source-space features, from simple band-power estimates to graph-theoretic
functional connectivity measures.  Each modality can be used as a feedback
signal, a monitoring metric, or an input to an offline analysis pipeline.
Select a modality via the ``--modality`` flag of :doc:`cli` or programmatically
via :class:`~mne_rt.RTStream`; each is identified by the config key listed below.

.. contents::
   :local:
   :depth: 1

----

.. _modality-sensor_power:

Sensor Power
~~~~~~~~~~~~

**Config key:** ``sensor_power``

Mean spectral power across selected channels within a target frequency band.
For each channel :math:`i`, the one-sided power spectral density
:math:`S_i(f)` is estimated via Welch's method.
The per-channel band power is averaged over the target band
:math:`[f_1, f_2]` and then averaged across all :math:`N_\mathrm{ch}` channels
in the electrode set:

.. math::

   P = \frac{1}{N_\mathrm{ch}} \sum_{i=1}^{N_\mathrm{ch}}
       \frac{1}{|\mathcal{F}|} \sum_{f \,\in\, [f_1,\, f_2]} S_i(f)

where :math:`\mathcal{F}` is the set of discrete frequency bins in the band.

----

.. _modality-band_ratio:

Band Ratio
~~~~~~~~~~

**Config key:** ``band_ratio``

Ratio of mean band power in two frequency bands.
Let :math:`P_1` and :math:`P_2` denote the mean band powers (computed as in
:ref:`modality-sensor_power`) for bands :math:`[f_{1a}, f_{1b}]` and
:math:`[f_{2a}, f_{2b}]`, respectively:

.. math::

   R = \frac{P_1}{P_2}

A common example is the theta/beta ratio used as an index of attentional
engagement.

----

.. _modality-erd_ers:

ERD / ERS
~~~~~~~~~

**Config key:** ``erd_ers``

Event-related desynchronisation (ERD) and event-related synchronisation (ERS)
quantify power changes relative to a resting-state baseline
:footcite:p:`pfurtscheller1999event`.
Let :math:`B` be the mean band power computed from the baseline recording and
:math:`P` the mean band power in the current analysis window:

.. math::

   \mathrm{ERD/ERS} = \frac{P - B}{B} \times 100\;\%

Negative values indicate desynchronisation (power decrease); positive values
indicate synchronisation (power increase).

----

.. _modality-laterality:

Laterality
~~~~~~~~~~

**Config key:** ``laterality``

Inter-hemispheric log power asymmetry between right and left electrode sets.
Let :math:`P_R` and :math:`P_L` be the mean band powers of the right- and
left-hemisphere channels, respectively:

.. math::

   \mathrm{LAT} = \log\!\left(\frac{P_R}{P_L}\right)

Positive values reflect greater right-hemisphere power; negative values
reflect greater left-hemisphere power.
The logarithmic transformation renders the index symmetric around zero.

----

.. _modality-laterality_erd_ers:

Laterality ERD/ERS
~~~~~~~~~~~~~~~~~~

**Config key:** ``laterality_erd_ers``

ERD/ERS-based laterality index — the difference of the ERD/ERS values
computed separately for the right and left hemisphere electrode sets:

.. math::

   \mathrm{LAT}_{\mathrm{ERD/ERS}} =
       \mathrm{ERD/ERS}_\mathrm{right} - \mathrm{ERD/ERS}_\mathrm{left}

This measure combines baseline normalisation (from ERD/ERS) with hemispheric
asymmetry (from laterality) and is particularly useful for motor-imagery
neurofeedback and BCI applications.

----

.. _modality-instantaneous_phase:

Instantaneous Phase
~~~~~~~~~~~~~~~~~~~

**Config key:** ``instantaneous_phase``

The instantaneous phase and amplitude of the analytic signal in a target
frequency band, estimated via the Hilbert transform
:footcite:p:`le2001comparison`.

The channel time series :math:`x(t)` is first zero-phase bandpass-filtered
(4th-order Butterworth, via ``sosfiltfilt``) to isolate the target band
:math:`[f_1, f_2]`, then extended to the analytic signal
:math:`z(t) = x(t) + i\,\hat{x}(t)`, where :math:`\hat{x}(t)` is the
Hilbert transform of the filtered signal:

.. math::

   \phi(t) = \angle\, z(t), \qquad A(t) = |z(t)|

The feature value is the instantaneous phase :math:`\phi \in (-\pi, \pi]`
at the *last sample* of the current analysis window, averaged across the
selected channels.  The amplitude envelope :math:`A` is returned as a
secondary output and can be used as a gating signal (e.g. stimulate only
when amplitude is high).

Instantaneous phase is particularly useful for:

* **Phase-triggered stimulation** — close the feedback loop at a target
  phase angle (e.g., trough of the alpha cycle)
* **Cross-frequency coupling analysis** — use as the phase-providing
  input to the CFC modality
* **Phase synchrony feedback** — reward or monitor convergence of phase
  between two electrode pairs

----

.. _modality-scp:

Slow Cortical Potentials
~~~~~~~~~~~~~~~~~~~~~~~~

**Config key:** ``scp``

Slow cortical potentials (SCP) are ultra-low-frequency shifts in cortical
excitability with time constants of 0.5 – 10 seconds
:footcite:p:`birbaumer1990slow`.  The signal is extracted via low-pass
filtering and channel averaging:

1. Apply an optional high-pass filter at ``highpass`` Hz (set to 0 for
   DC-coupled acquisition).
2. Apply a low-pass filter at ``lowpass`` Hz (typically 1 Hz) using a
   4th-order Butterworth filter via ``sosfilt``.
3. Collapse channels using the selected ``reference`` (``"mean"`` or
   ``"median"``).
4. Return the mean amplitude of the filtered signal across the window.

.. math::

   \mathrm{SCP} = \operatorname{mean}\!\bigl(\mathrm{LP}_{f_\mathrm{low}}\{x(t)\}\bigr)

Positive SCP values reflect cortical deactivation (slow positive shift);
negative SCP values reflect activation (slow negative shift).

----

.. _modality-peak_alpha_freq:

Peak Alpha Frequency
~~~~~~~~~~~~~~~~~~~~

**Config key:** ``peak_alpha_freq``

Real-time tracker of the individual peak alpha frequency (PAF), smoothed
with an exponential moving average (EMA) across consecutive analysis windows.

For each window, the Welch PSD is computed and the frequency with maximum
power within the search band :math:`[f_1, f_2]` is identified:

.. math::

   f^*_t = \operatorname*{argmax}_{f \in [f_1, f_2]}\;
           \frac{1}{N_\mathrm{ch}} \sum_{i=1}^{N_\mathrm{ch}} S_i(f)

The instantaneous estimate is smoothed by an EMA with coefficient
:math:`\alpha \in [0, 1)`:

.. math::

   \widehat{f}^\mathrm{PAF}_t =
       \alpha\,\widehat{f}^\mathrm{PAF}_{t-1}
       + (1 - \alpha)\,f^*_t

At :math:`\alpha = 0` the output is the raw instantaneous peak; at
:math:`\alpha = 0.99` the estimate changes very slowly, acting as a
long-term personalised frequency anchor.

The EMA state is carried across windows via a shared mutable reference, so
the tracker persists correctly over the full session.

----

.. _modality-hjorth:

Hjorth Parameters
~~~~~~~~~~~~~~~~~

**Config key:** ``hjorth``

Hjorth parameters are pure time-domain descriptors computed from the variance
of a signal :math:`x(t)` and its successive derivatives. :footcite:p:`hjorth1970eeg`

Let :math:`x_i` denote channel :math:`i` after bandpass filtering.
Define sample variances:

.. math::

   \sigma^2_x = \operatorname{Var}(x), \quad
   \sigma^2_{x'} = \operatorname{Var}\!\left(\tfrac{dx}{dt}\right), \quad
   \sigma^2_{x''} = \operatorname{Var}\!\left(\tfrac{d^2x}{dt^2}\right).

The three Hjorth parameters are:

.. math::

   \text{Activity}  &= \sigma^2_x \\[4pt]
   \text{Mobility}  &= \sqrt{\frac{\sigma^2_{x'}}{\sigma^2_x}} \\[4pt]
   \text{Complexity}&= \frac{\sqrt{\sigma^2_{x''}/\sigma^2_{x'}}}
                             {\sqrt{\sigma^2_{x'}/\sigma^2_x}}

**Mobility** approximates the dominant frequency of the signal (in units of
rad/sample); **Complexity** quantifies how closely the signal resembles a pure
sine wave — a pure oscillation has Complexity ≈ 1.

MNE-RT reports the mean of Mobility and Complexity averaged across all channels
in the selected electrode set.

----

.. _modality-spectral_centroid:

Spectral Centroid
~~~~~~~~~~~~~~~~~

**Config key:** ``spectral_centroid``

The spectral centroid estimates the *centre of mass* of the power spectrum
within a frequency band, giving a single-number proxy for the dominant
frequency of the signal at any given moment.

Given channel data :math:`x_i(t)` and its one-sided power spectral density
:math:`S_i(f)` (estimated via Welch's method), the spectral centroid for
channel :math:`i` is:

.. math::

   f_{\mathrm{centroid},i} =
   \frac{\displaystyle\sum_{f \,\in\, [f_1,\, f_2]} f\; S_i(f)}
        {\displaystyle\sum_{f \,\in\, [f_1,\, f_2]} S_i(f)}

MNE-RT reports the **mean centroid across channels** in the selected electrode
set as the feature value:

.. math::

   SC = \frac{1}{N_{\mathrm{ch}}} \sum_{i=1}^{N_{\mathrm{ch}}}
        f_{\mathrm{centroid},i}

:math:`SC` shifts upward when neural activity moves toward the upper edge of
the band (e.g., during cognitive load in the alpha band) and downward when
activity consolidates near the lower edge.

----

.. _modality-entropy:

Entropy
~~~~~~~

**Config key:** ``entropy``

MNE-RT supports three entropy variants that capture different aspects of signal
complexity.

**Spectral entropy** — normalised Shannon entropy of the power spectral density:

.. math::

   H_\mathrm{spec} = -\sum_{f} \tilde{S}(f)\,\log\!\bigl(\tilde{S}(f)\bigr)

where :math:`\tilde{S}(f) = S(f) / \sum_{f'} S(f')` is the normalised PSD
(treated as a probability distribution over frequencies).

**Approximate entropy** (ApEn) — regularity statistic for a time series of
length :math:`N` with embedding dimension :math:`m` and tolerance :math:`r` :footcite:p:`pincus1991approximate`:

.. math::

   \operatorname{ApEn}(m, r) =
       \phi^{(m)}(r) - \phi^{(m+1)}(r)

where :math:`\phi^{(m)}(r) = \frac{1}{N - m + 1} \sum_{i} \log C_i^{(m)}(r)`
and :math:`C_i^{(m)}(r)` is the fraction of :math:`m`-length template vectors
within Chebyshev distance :math:`r` of the :math:`i`-th template.

**Sample entropy** (SampEn) — a bias-corrected alternative that excludes
self-matches :footcite:p:`richman2000physiological`:

.. math::

   \operatorname{SampEn}(m, r) =
       -\log\!\left(\frac{B^{(m+1)}}{A^{(m)}}\right)

where :math:`A^{(m)}` counts matching :math:`m`-length template pairs and
:math:`B^{(m+1)}` counts matching :math:`(m+1)`-length template pairs
(self-matches excluded from both).

----

.. _modality-argmax_freq:

Argmax Frequency
~~~~~~~~~~~~~~~~

**Config key:** ``argmax_freq``

The dominant frequency is the frequency bin with maximum power within the
target band.
Given the Welch PSD :math:`S(f)` averaged across channels, the feature value is:

.. math::

   f^* = \operatorname*{argmax}_{f \,\in\, [f_1,\, f_2]}\; S(f)

This feature tracks slow shifts in the dominant oscillatory peak, for example
the individual alpha frequency (IAF) during rest or task conditions.

----

.. _modality-individual_peak_power:

Individual Peak Power
~~~~~~~~~~~~~~~~~~~~~

**Config key:** ``individual_peak_power``

Power at the subject's individual spectral peak, estimated using the
`FOOOF / specparam <https://fooof-tools.github.io/fooof>`_ parametric spectral model :footcite:p:`donoghue2020parameterizing`.
The model decomposes the PSD into an aperiodic (1/f) component and a
superimposed set of Gaussian peaks:

.. math::

   \log S(f) = L(f) + \sum_{k} G_k(f)

where :math:`L(f) = b - \log(k + f^{\chi})` is the aperiodic component
(offset :math:`b`, knee :math:`k`, exponent :math:`\chi`) and
:math:`G_k(f) = a_k \exp\!\left(-\tfrac{(f - \mu_k)^2}{2\sigma_k^2}\right)`
are Gaussian peaks with centre :math:`\mu_k`, height :math:`a_k`, and width
:math:`\sigma_k`.

The individual alpha frequency (IAF) is identified as the peak centre
:math:`\mu_k` closest to the canonical alpha band, and the feature value is the
aperiodic-corrected power :math:`a_k` at that peak.

----

.. _modality-cfc_sensor:

Cross-Frequency Coupling (Sensor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Config key:** ``cfc_sensor``

Cross-frequency coupling (CFC) quantifies interactions between the phase of a
slow oscillation and the amplitude of a faster oscillation.
MNE-RT uses Phase-Amplitude Coupling (PAC) via the **Modulation Index** (MI),
which measures the Kullback–Leibler divergence between the observed amplitude
distribution :math:`p(\theta)` (mean amplitude as a function of phase bin
:math:`\theta`) and a uniform distribution :footcite:p:`tort2010measuring`:

.. math::

   \mathrm{MI} =
       \frac{D_\mathrm{KL}\bigl(p(\theta) \,\|\, \mathcal{U}\bigr)}{\log N}

where :math:`N` is the number of phase bins and :math:`\mathcal{U}` is the
uniform distribution.
A value of :math:`\mathrm{MI} = 0` indicates no coupling; larger values
indicate stronger modulation of high-frequency amplitude by low-frequency
phase.

MNE-RT can also produce a **comodulogram** — a two-dimensional map of MI values
over a grid of phase-frequency / amplitude-frequency pairs — to identify
which frequency combinations exhibit significant coupling.

----

.. _modality-sensor_connectivity:

Sensor Connectivity
~~~~~~~~~~~~~~~~~~~

**Config key:** ``sensor_connectivity``

Pairwise functional connectivity between sensor channels.
MNE-RT directly calls the measures implemented in `MNE-Connectivity <https://mne.tools/mne-connectivity/stable/index.html>`_.

See `MNE-Connectivity <https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_time.html#mne_connectivity.spectral_connectivity_time>`_ 
for the detailed list of supported methods. The feature value is the connectivity between a specified pair of channels
or the mean connectivity across a set of channel pairs.

----

.. _modality-connectivity_ratio:

Connectivity Ratio
~~~~~~~~~~~~~~~~~~

**Config key:** ``connectivity_ratio``

Ratio of functional connectivity between two channel pairs:

.. math::

   R = \frac{\mathrm{conn}(A_1, B_1)}{\mathrm{conn}(A_2, B_2)}

where each connectivity value is computed using the same spectral measure
as :ref:`modality-sensor_connectivity` (e.g. coherence, PLV, imaginary
coherence).  A common use is laterality of interhemispheric connectivity
(e.g. :math:`\mathrm{coh}(C3, C4) / \mathrm{coh}(F3, F4)`).

----

.. _modality-sensor_graph:

Sensor Graph
~~~~~~~~~~~~

**Config key:** ``sensor_graph``

Graph-Laplacian learning from sensor-space signals estimates a sparse
functional connectivity graph :math:`\mathcal{G} = (\mathcal{V}, \mathbf{W})`
:footcite:p:`kalofolias2016learn` :footcite:p:`shabestari2026shared`.

Given a signal matrix :math:`\mathbf{X} \in \mathbb{R}^{p \times n}`
(:math:`p` sensor channels, :math:`n` samples), MNE-RT solves the
**log-degree barrier** optimisation problem:

.. math::

   \min_{\mathbf{W} \geq 0,\,\mathbf{W} = \mathbf{W}^\top}
   \;\alpha \operatorname{tr}(\mathbf{X}^\top \mathbf{L} \mathbf{X})
   \;-\; \beta \mathbf{1}^\top \log(\mathbf{W}\mathbf{1})
   \;+\; \tfrac{1}{2}\|\mathbf{W}\|_F^2

where :math:`\mathbf{L} = \operatorname{Diag}(\mathbf{W}\mathbf{1}) - \mathbf{W}`
is the combinatorial graph Laplacian and the log-degree term prevents
degenerate (all-zero) solutions.

* :math:`\alpha` — data-fidelity weight (larger → smoother, more connected graph)
* :math:`\beta` — log-degree regularisation (larger → more uniform node degrees)

The feature value is the edge weight :math:`W_{ij}` between two specified sensor
nodes (e.g. a left–right electrode pair), centred by subtracting a small
offset so that values oscillate around zero.
The optimisation is solved via the proximal splitting scheme of 
`pyunlocbox <https://pyunlocbox.readthedocs.io/en/stable/>`_.

----

.. _modality-source_power:

Source Power
~~~~~~~~~~~~

**Config key:** ``source_power``

Band power in source space, requiring a pre-computed inverse operator
(MNE minimum-norm or beamformer).
The inverse operator :math:`\mathbf{M}` projects sensor data
:math:`\mathbf{X} \in \mathbb{R}^{p \times n}` to source estimates
:math:`\hat{\mathbf{J}} = \mathbf{M}\mathbf{X} \in \mathbb{R}^{q \times n}`:

.. math::

   P_\mathrm{source} = \frac{1}{N_\mathrm{src}}
       \sum_{s=1}^{N_\mathrm{src}}
       \frac{1}{|\mathcal{F}|} \sum_{f \,\in\, [f_1,\, f_2]} S_s(f)

where :math:`S_s(f)` is the Welch PSD of source :math:`s` and
:math:`N_\mathrm{src}` is the number of source vertices in the region of
interest.
Baseline recording (``mne-rt baseline``) is required to compute the inverse
operator before running this modality.

----

.. _modality-source_connectivity:

Source Connectivity
~~~~~~~~~~~~~~~~~~~

**Config key:** ``source_connectivity``

Functional connectivity computed on source-space time series rather than
sensor signals, reducing the influence of field spread and volume conduction.
After projecting sensor data to source estimates with the inverse operator,
the same measures as :ref:`modality-sensor_connectivity` are applied.

Source connectivity is more anatomically interpretable than sensor
connectivity and is recommended when an MRI and FreeSurfer reconstruction are
available.

----

.. _modality-source_graph:

Source Graph
~~~~~~~~~~~~

**Config key:** ``source_graph``

Graph-Laplacian learning applied to source-space signals.
After projecting sensor data to source estimates with the inverse operator,
the same optimisation as :ref:`modality-sensor_graph` is solved on the
source time-series matrix :math:`\hat{\mathbf{J}} \in \mathbb{R}^{q \times n}`
:footcite:p:`kalofolias2016learn` :footcite:p:`shabestari2026shared`:

.. math::

   \min_{\mathbf{W} \geq 0,\,\mathbf{W} = \mathbf{W}^\top}
   \;\alpha \operatorname{tr}(\hat{\mathbf{J}}^\top \mathbf{L} \hat{\mathbf{J}})
   \;-\; \beta \mathbf{1}^\top \log(\mathbf{W}\mathbf{1})
   \;+\; \tfrac{1}{2}\|\mathbf{W}\|_F^2

The feature value is the learned edge weight :math:`W_{ij}` between two specified
brain atlas parcels or source vertices.
Anatomical accuracy is improved relative to sensor-space graph learning
because volume conduction is partially mitigated by the inverse solution.

----

See :doc:`cli` for the full list of command-line options and flags used to
configure these modalities.

.. footbibliography::
