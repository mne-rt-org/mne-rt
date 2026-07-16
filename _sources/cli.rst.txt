.. _cli:

CLI
===

MNE-RT provides an ``mne-rt`` shell command with five sub-commands.

.. code-block:: text

    mne-rt --help
    mne-rt --version
    mne-rt info
    mne-rt demo         [options]
    mne-rt demo-erp     [options]
    mne-rt baseline     --subject ID --subjects-dir DIR [options]
    mne-rt run          --subject ID --subjects-dir DIR --duration N [options]

Global options:

.. list-table::
   :header-rows: 1
   :widths: 20 15 50

   * - Flag
     - Default
     - Description
   * - ``--verbose``, ``-v``
     - WARNING
     - Logging verbosity level: ``DEBUG``, ``INFO``, ``WARNING``, or ``ERROR``

``mne-rt info``
---------------

Print the installed MNE-RT version and all key dependency versions.

.. code-block:: console

    $ mne-rt info
    MNE-RT — Real-Time M/EEG Analysis
    ──────────────────────────────────
      mne-rt version : 1.0.0
      Python         : 3.11.9
      mne            : 1.8.0
      ...

``mne-rt demo``
---------------

Launch a full demo real-time session from simulated EEG (no amplifier needed).
The real-time scalp topomap (δ/θ/α/β/γ) is shown by default.  The 3-D brain
display is enabled automatically when a FreeSurfer ``fsaverage5`` directory
is found (via ``FREESURFER_HOME`` or ``--subjects-fs-dir``).

.. code-block:: console

    $ mne-rt demo
    $ mne-rt demo --duration 120 --modality sensor_power erd_ers
    $ mne-rt demo --no-topomap
    $ mne-rt demo --no-brain
    $ mne-rt demo --threshold 2e-13 --threshold-direction up
    $ mne-rt demo --zscore-threshold 0.5 --zscore-warmup 20 --zscore-min-std 1e-15

Options:

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Flag
     - Default
     - Description
   * - ``--duration``
     - 120
     - Session duration in seconds
   * - ``--modality``
     - sensor_power band_ratio entropy hjorth
     - Feature modality(ies) to demonstrate
   * - ``--winsize``
     - 1.0
     - Analysis window length (s)
   * - ``--no-nf``
     - —
     - Disable the scrolling real-time NF signal plot (:class:`~mne_rt.viz.NFPlot`)
   * - ``--no-raw``
     - —
     - Disable the scrolling raw M/EEG viewer (:class:`~mne_rt.viz.RawPlot`)
   * - ``--no-topomap``
     - —
     - Disable the real-time scalp topomap (enabled by default)
   * - ``--no-brain``
     - —
     - Disable 3-D brain display (auto-enabled when FreeSurfer is found)
   * - ``--subjects-fs-dir``
     - —
     - FreeSurfer subjects directory (auto-detected from ``FREESURFER_HOME``
       or ``SUBJECTS_DIR`` if not given)
   * - ``--surf``
     - inflated
     - Brain surface geometry (``inflated``, ``pial``, or ``white``)
   * - ``--smoothing``
     - 0.25
     - EMA smoothing factor applied to each feature value (``1.0`` = no smoothing,
       ``0.1`` = heavy smoothing). Lower values produce smoother output at the
       cost of slower response to real changes in brain activity.
   * - ``--no-save``
     - —
     - Skip saving session data and report
   * - ``--protocol``
     - —
     - Reward protocol applied to the first modality, drawn as a dashed
       horizontal line on :class:`~mne_rt.viz.NFPlot`.  ``threshold`` ->
       :class:`~mne_rt.protocols.ThresholdProtocol` (fixed level, see
       ``--threshold``).  ``zscore`` -> :class:`~mne_rt.protocols.ZScoreProtocol`
       (adaptive boundary tracking a rolling mean/std, see
       ``--zscore-threshold``/``--zscore-warmup``).  Usually inferred
       automatically from whichever of ``--threshold`` or ``--zscore-*`` you
       pass; only needed to disambiguate if both are given together.  Omit
       all three to run without a protocol (no line shown)
   * - ``--threshold``
     - 0.0
     - Fixed reward level; passing this alone enables
       :class:`~mne_rt.protocols.ThresholdProtocol`
   * - ``--threshold-direction``
     - up
     - Reward direction, shared by both protocol types: ``up`` rewards
       values above the boundary, ``down`` rewards values below it
   * - ``--zscore-threshold``
     - 0.5
     - Minimum |z-score| required to reward; passing this or
       ``--zscore-warmup`` alone enables
       :class:`~mne_rt.protocols.ZScoreProtocol`
   * - ``--zscore-warmup``
     - 20
     - Windows used only to seed the rolling mean/std baseline before any
       reward can be issued; passing this or ``--zscore-threshold`` alone
       enables :class:`~mne_rt.protocols.ZScoreProtocol`
   * - ``--zscore-min-std``
     - 1e-6
     - Floor on the running standard deviation, in the modality's raw
       units.  **Must be well below your signal's real magnitude** or it
       dominates and the threshold line ends up wildly off-scale -- e.g.
       for ``sensor_power`` (~1e-13 range), try ``1e-15``

``mne-rt demo-erp``
--------------------

Launch an ERP demo that streams MNE sample-dataset EEG through a mock LSL
player, collects auditory epochs trial-by-trial via :class:`~mne_rt.RTEpochs`,
and drives the four epoch visualisation windows.
Downloads the MNE sample dataset on first run (~1.5 GB).

.. code-block:: console

    $ mne-rt demo-erp                          # all four plot windows
    $ mne-rt demo-erp --n-trials 50            # stop after 50 trials
    $ mne-rt demo-erp --no-tfr                 # skip the TFR heatmap
    $ mne-rt demo-erp --no-compare --no-tfr    # TopoPlot + ButterflyPlot only

Options:

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Flag
     - Default
     - Description
   * - ``--n-trials``
     - 70
     - Number of EEG trials to collect before stopping
   * - ``--no-topo``
     - —
     - Disable the scalp-layout :class:`~mne_rt.viz.TopoPlot` (ERP display)
   * - ``--no-butterfly``
     - —
     - Disable the :class:`~mne_rt.viz.ButterflyPlot` (all-channel overlay)
   * - ``--no-compare``
     - —
     - Disable :class:`~mne_rt.viz.CompareEvoked` (per-channel comparison)
   * - ``--no-tfr``
     - —
     - Disable the :class:`~mne_rt.viz.TFRPlot` (Morlet wavelet heatmaps)

``mne-rt baseline``
--------------------

Record a resting-state baseline and compute the noise covariance and inverse
operator needed for source-space modalities.

.. code-block:: console

    $ mne-rt baseline --subject sub01 --subjects-dir /data/subjects
    $ mne-rt baseline --subject sub01 --subjects-dir /data --duration 180 --mock
    $ mne-rt baseline --subject sub01 --subjects-dir /data --mock \
                      --fname /path/to/recording.fif

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
   * - ``--session``
     - 01
     - BIDS session label (e.g. ``01``, ``pre``, ``week1``)
   * - ``--mock``
     - —
     - Use simulated data instead of live LSL
   * - ``--fname``
     - —
     - Any MNE-readable file (``.fif``, ``.vhdr``, ``.edf``, ``.bdf``,
       ``.set``, …) for ``--mock`` playback
   * - ``--montage``
     - easycap-M1
     - EEG montage name or ``.bvct`` path
   * - ``--data-type``
     - eeg
     - ``eeg`` or ``meg``
   * - ``--subjects-fs-dir``
     - —
     - FreeSurfer subjects directory (required for source-space modalities)

``mne-rt run``
--------------

Run a real-time M/EEG session with feature extraction, feedback protocols,
and live visualisation.

.. code-block:: console

    $ mne-rt run --subject sub01 --subjects-dir /data --duration 600 \
                 --modality sensor_power erd_ers

    # With epoch plots (requires a stimulus channel in the recording)
    $ mne-rt run --subject sub01 --subjects-dir /data --duration 600 \
                 --modality sensor_power \
                 --topo --tfr --stim-ch "STI 014" \
                 --event-id left=1 right=2

    # With topomap display
    $ mne-rt run --subject sub01 --subjects-dir /data --duration 600 \
                 --modality sensor_power --topomap

    # With OSC output to Max/MSP
    $ mne-rt run --subject sub01 --subjects-dir /data --duration 600 \
                 --modality sensor_power --osc-host 127.0.0.1 --osc-port 9000

    # With LSL output (faster, same-machine integration)
    $ mne-rt run --subject sub01 --subjects-dir /data --duration 600 \
                 --modality sensor_power --lsl-output

    # From a mock file with artifact correction
    $ mne-rt run --subject sub01 --subjects-dir /data --duration 300 \
                 --mock --fname /data/sub01/session.fif \
                 --artifact-correction asr --topo

    # With a fixed reward threshold shown live on NFPlot
    $ mne-rt run --subject sub01 --subjects-dir /data --duration 600 \
                 --modality sensor_power \
                 --threshold 2e-13 --threshold-direction up

    # With an adaptive z-score boundary shown live on NFPlot
    # (--zscore-min-std must be well below the modality's raw scale --
    # sensor_power is ~1e-13, so the default 1e-6 floor would dominate)
    $ mne-rt run --subject sub01 --subjects-dir /data --duration 600 \
                 --modality sensor_power \
                 --zscore-threshold 0.5 --zscore-warmup 20 --zscore-min-std 1e-15

Options (inherits all ``baseline`` flags above, plus):

.. list-table::
   :header-rows: 1
   :widths: 28 15 42

   * - Flag
     - Default
     - Description
   * - ``--duration``
     - *required*
     - Session duration in seconds
   * - ``--modality``
     - sensor_power
     - One or more feature modalities (space-separated).  See table below.
   * - ``--winsize``
     - 1.0
     - Analysis window length (s)
   * - ``--artifact-correction``
     - —
     - ``lms``, ``orica``, ``gedai``, ``asr``, or ``maxwell`` (MEG only)
   * - ``--no-nf``
     - —
     - Disable the scrolling real-time NF signal plot (:class:`~mne_rt.viz.NFPlot`)
   * - ``--protocol``
     - —
     - Reward protocol applied to the first modality, drawn as a dashed
       horizontal line on :class:`~mne_rt.viz.NFPlot`.  ``threshold`` ->
       :class:`~mne_rt.protocols.ThresholdProtocol` (fixed level, see
       ``--threshold``).  ``zscore`` -> :class:`~mne_rt.protocols.ZScoreProtocol`
       (adaptive boundary tracking a rolling mean/std, see
       ``--zscore-threshold``/``--zscore-warmup``).  Usually inferred
       automatically from whichever of ``--threshold`` or ``--zscore-*`` you
       pass; only needed to disambiguate if both are given together.  Omit
       all three to run without a protocol (no line shown)
   * - ``--threshold``
     - 0.0
     - Fixed reward level; passing this alone enables
       :class:`~mne_rt.protocols.ThresholdProtocol`
   * - ``--threshold-direction``
     - up
     - Reward direction, shared by both protocol types: ``up`` rewards
       values above the boundary, ``down`` rewards values below it
   * - ``--zscore-threshold``
     - 0.5
     - Minimum |z-score| required to reward; passing this or
       ``--zscore-warmup`` alone enables
       :class:`~mne_rt.protocols.ZScoreProtocol`
   * - ``--zscore-warmup``
     - 20
     - Windows used only to seed the rolling mean/std baseline before any
       reward can be issued; passing this or ``--zscore-threshold`` alone
       enables :class:`~mne_rt.protocols.ZScoreProtocol`
   * - ``--zscore-min-std``
     - 1e-6
     - Floor on the running standard deviation, in the modality's raw
       units.  **Must be well below your signal's real magnitude** or it
       dominates and the threshold line ends up wildly off-scale -- e.g.
       for ``sensor_power`` (~1e-13 range), try ``1e-15``
   * - ``--no-raw``
     - —
     - Disable the scrolling raw M/EEG viewer (:class:`~mne_rt.viz.RawPlot`)
   * - ``--topomap``
     - —
     - Enable real-time scalp topomap (δ/θ/α/β/γ bands)
   * - ``--brain``
     - —
     - Enable 3-D brain display (requires inverse operator)
   * - ``--topo``
     - —
     - Enable :class:`~mne_rt.viz.TopoPlot` ERP display (requires ``--stim-ch``)
   * - ``--butterfly``
     - —
     - Enable :class:`~mne_rt.viz.ButterflyPlot` (requires ``--stim-ch``)
   * - ``--compare-evoked``
     - —
     - Enable :class:`~mne_rt.viz.CompareEvoked` (requires ``--stim-ch``)
   * - ``--tfr``
     - —
     - Enable :class:`~mne_rt.viz.TFRPlot` Morlet wavelet heatmap
       (requires ``--stim-ch``)
   * - ``--stim-ch``
     - —
     - Stimulus/trigger channel name, e.g. ``STI 014`` — required when
       any epoch plot flag is used
   * - ``--tmin``
     - -0.1
     - Epoch start relative to stimulus (s)
   * - ``--tmax``
     - 0.5
     - Epoch end relative to stimulus (s)
   * - ``--event-id``
     - stimulus=1
     - Condition definitions as ``NAME=CODE`` pairs, e.g.
       ``--event-id left=1 right=2``
   * - ``--surf``
     - inflated
     - Brain surface geometry
   * - ``--osc-host``
     - —
     - Enable OSC output; target hostname (e.g. ``127.0.0.1``)
   * - ``--osc-port``
     - 9000
     - OSC destination port
   * - ``--osc-prefix``
     - /mne_rt
     - OSC address prefix
   * - ``--lsl-output``
     - —
     - Push feature values into an LSL stream outlet (faster than OSC for
       same-machine integration; readable by PsychoPy, Psychtoolbox, …)
   * - ``--lsl-stream-name``
     - MNE_RT
     - LSL stream name (only with ``--lsl-output``)
   * - ``--smoothing``
     - 0.25
     - EMA smoothing factor applied to each feature value (``1.0`` = no smoothing,
       ``0.1`` = heavy smoothing)

Available feature modalities
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 52

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
   * - ``laterality_erd_ers``
     - ERD/ERS computed separately per hemisphere
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
   * - ``instantaneous_phase``
     - Instantaneous phase and amplitude envelope via Hilbert analytic signal
   * - ``scp``
     - Slow cortical potential — low-pass mean amplitude shift (DC-coupled)
   * - ``peak_alpha_freq``
     - Real-time peak alpha frequency tracker with EMA smoothing
   * - ``sensor_connectivity``
     - Functional connectivity (coh, plv, pli, wpli, corr, …); method set in config
   * - ``connectivity_ratio``
     - Ratio of connectivity between two channel pairs (e.g. C3–C4 / F3–F4)
   * - ``sensor_graph``
     - Graph-Laplacian learning from sensor connectivity
   * - ``source_power``
     - Source-space band power (requires inverse operator)
   * - ``source_connectivity``
     - Source-space functional connectivity
   * - ``source_graph``
     - Graph-Laplacian learning from source connectivity

For the mathematical background of every modality, see :doc:`modalities`.

Real-time visualisation
-----------------------

All eight plot windows are available as CLI flags and can be combined freely:

.. list-table::
   :header-rows: 1
   :widths: 22 58

   * - Flag
     - Plot class
   * - ``--nf`` *(default on)*
     - :class:`~mne_rt.viz.NFPlot` — scrolling multi-channel NF signal monitor
   * - ``--raw`` *(default on)*
     - :class:`~mne_rt.viz.RawPlot` — scrolling raw M/EEG channel viewer (bad-channel / bad-segment marking)
   * - ``--epoch-plot``
     - :class:`~mne_rt.viz.EpochPlot` — scrolling raw viewer with trigger/epoch overlays (requires ``--stim-ch``)
   * - ``--topomap``
     - :class:`~mne_rt.viz.TopomapPlot` — live per-band scalp topography
   * - ``--brain``
     - :class:`~mne_rt.viz.BrainPlot` — interactive 3-D cortical surface
   * - ``--topo``
     - :class:`~mne_rt.viz.TopoPlot` — scalp-layout ERP with ±SEM shading
   * - ``--butterfly``
     - :class:`~mne_rt.viz.ButterflyPlot` — all channels overlaid, region-coloured
   * - ``--compare-evoked``
     - :class:`~mne_rt.viz.CompareEvoked` — per-channel comparison with SEM ribbons
   * - ``--tfr``
     - :class:`~mne_rt.viz.TFRPlot` — Morlet wavelet TFR heatmaps

The first four (``--nf``, ``--raw``, ``--topomap``, ``--brain``) work with
continuous feature extraction (``mne-rt run``).  The epoch-based windows
(``--epoch-plot``, ``--erp``, ``--butterfly``, ``--compare-evoked``, ``--tfr``)
require ``--stim-ch``.  Use ``mne-rt demo-erp`` to try them without a live recording.

See :doc:`visualization` for screenshots and feature descriptions of each plot.
