.. _cli:

Command-Line Interface
======================

ANT provides a ``ant`` shell command with four sub-commands.

.. code-block:: text

    ant --help
    ant --version
    ANT info
    ANT demo [options]
    ANT baseline --subject ID --subjects-dir DIR [options]
    ANT run     --subject ID --subjects-dir DIR --duration N [options]

``ANT info``
------------

Print the installed ANT version and all key dependency versions.

.. code-block:: console

    $ ANT info
    Advanced Neurofeedback Toolbox (ANT)
    ────────────────────────────────────
      ANT version  : 1.0.0
      Python       : 3.11.9
      mne          : 1.8.0
      ...

``ANT demo``
------------

Launch a full demo NF session from simulated EEG (no amplifier needed).

.. code-block:: console

    $ ANT demo
    $ ANT demo --duration 120 --modality sensor_power erd_ers
    $ ANT demo --brain --subjects-fs-dir /path/to/fsaverage

Options:

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Flag
     - Default
     - Description
   * - ``--duration``
     - 60
     - Session duration in seconds
   * - ``--modality``
     - sensor_power
     - NF modality(ies) to demonstrate
   * - ``--winsize``
     - 1.0
     - Analysis window length (s)
   * - ``--no-signal``
     - —
     - Disable NF signal plot
   * - ``--no-raw``
     - —
     - Disable raw stream viewer
   * - ``--brain``
     - —
     - Enable 3D brain display
   * - ``--subjects-fs-dir``
     - —
     - FreeSurfer subjects dir (for ``--brain``)
   * - ``--surf``
     - inflated
     - Brain surface geometry

``ANT baseline``
----------------

Record a resting-state baseline and compute the inverse operator.

.. code-block:: console

    $ ANT baseline --subject sub01 --subjects-dir /data/subjects
    $ ANT baseline --subject sub01 --subjects-dir /data --duration 180 --mock

Required:

- ``--subject ID`` — subject identifier string
- ``--subjects-dir DIR`` — root directory with one folder per subject

Options:

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Flag
     - Default
     - Description
   * - ``--duration``
     - 120
     - Baseline duration in seconds
   * - ``--mock``
     - —
     - Use simulated data instead of live LSL
   * - ``--fname``
     - —
     - BrainVision ``.vhdr`` file (with ``--mock``)
   * - ``--montage``
     - easycap-M1
     - EEG montage name or ``.bvct`` path
   * - ``--data-type``
     - eeg
     - ``eeg`` or ``meg``

``ANT run``
-----------

Run a closed-loop NF main session.

.. code-block:: console

    $ ANT run --subject sub01 --subjects-dir /data --duration 600 \
              --modality sensor_power erd_ers --show-nf-signal

    # With OSC output to Max/MSP
    $ ANT run --subject sub01 --subjects-dir /data --duration 600 \
              --modality sensor_power --osc-host 127.0.0.1 --osc-port 9000

Options (in addition to baseline flags):

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Flag
     - Default
     - Description
   * - ``--duration``
     - *required*
     - Session duration in seconds
   * - ``--modality``
     - sensor_power
     - NF modality(ies) to extract
   * - ``--winsize``
     - 1.0
     - Analysis window length (s)
   * - ``--artifact-correction``
     - —
     - ``lms``, ``orica``, or ``gedai``
   * - ``--ring-buffer``
     - —
     - Use sliding ring-buffer (50 % overlap)
   * - ``--no-signal``
     - —
     - Disable NF signal plot
   * - ``--no-raw``
     - —
     - Disable raw stream viewer
   * - ``--brain``
     - —
     - Enable 3D brain display
   * - ``--osc-host``
     - —
     - Enable OSC output; target hostname
   * - ``--osc-port``
     - 9000
     - OSC destination port
   * - ``--osc-prefix``
     - /ant
     - OSC address prefix

Available NF modalities
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Modality key
     - Description
   * - ``sensor_power``
     - Mean band power across channels
   * - ``band_ratio``
     - Power ratio between two frequency bands (e.g. θ/β)
   * - ``erd_ers``
     - Event-related de/synchronisation (baseline normalised)
   * - ``laterality``
     - Log power asymmetry between right and left hemispheres
   * - ``hjorth``
     - Hjorth activity, mobility, complexity (time-domain)
   * - ``spectral_centroid``
     - Frequency-weighted spectral centroid
   * - ``entropy``
     - Spectral, approximate, or sample entropy
   * - ``argmax_freq``
     - Dominant frequency peak
   * - ``individual_peak_power``
     - Power at the individualised spectral peak
   * - ``cfc_sensor``
     - Cross-frequency coupling (sensor space)
   * - ``sensor_connectivity``
     - Functional connectivity (PLI, correlation)
   * - ``sensor_graph``
     - Graph-Laplacian learning from sensor connectivity
   * - ``source_power``
     - Source-space band power (requires inverse operator)
   * - ``source_connectivity``
     - Source-space functional connectivity
   * - ``source_graph``
     - Graph-Laplacian learning from source connectivity

Mathematical background
-----------------------

.. _hjorth-equations:

Hjorth parameters
~~~~~~~~~~~~~~~~~

Hjorth parameters are pure time-domain descriptors computed from the variance
of a signal :math:`x(t)` and its successive derivatives.

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
sine wave — a pure oscillation has complexity ≈ 1.

ANT reports the mean of Mobility and Complexity averaged across all channels
in the selected electrode set.

.. _graph-equations:

Graph-Laplacian learning
~~~~~~~~~~~~~~~~~~~~~~~~

The ``sensor_graph`` and ``source_graph`` modalities estimate a sparse
graph :math:`\mathcal{G} = (\mathcal{V}, \mathbf{W})` whose edge weights
encode functional coupling between nodes (channels or brain regions).

Given a signal matrix :math:`\mathbf{X} \in \mathbb{R}^{p \times n}` (
:math:`p` nodes, :math:`n` samples), ANT uses the
**log-degree barrier** graph learning problem (Kalofolias, 2016):

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

The NF value is the edge weight :math:`W_{ij}` between the two specified
nodes (e.g. a left–right electrode pair or two brain-atlas parcels), centred
by subtracting a small offset so that values oscillate around zero.

The optimisation is solved via the proximal splitting scheme of
``pyunlocbox``.

.. _spectral-centroid-equations:

Spectral centroid
~~~~~~~~~~~~~~~~~

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

where :math:`[f_1, f_2]` is the target frequency band.

ANT reports the **mean centroid across channels** in the selected electrode
set as the NF feature:

.. math::

   SC = \frac{1}{N_{\mathrm{ch}}} \sum_{i=1}^{N_{\mathrm{ch}}}
        f_{\mathrm{centroid},i}

**Interpretation** — :math:`SC` shifts upward when neural activity moves
toward the upper edge of the band (e.g., during cognitive load in the
alpha band) and downward when activity consolidates near the lower edge.
Tracking :math:`SC` within the alpha band (8–12 Hz) is sensitive to
individual alpha-peak frequency (IAF) dynamics without requiring explicit
peak detection.
