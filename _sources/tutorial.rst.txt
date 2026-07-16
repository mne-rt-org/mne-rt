.. _tutorial:

User Guide
==========

This page introduces the core concepts behind real-time M/EEG processing
and explains how MNE-RT maps those concepts onto a concrete session workflow.
No prior BCI or neurofeedback experience is required.

.. contents::
   :local:
   :depth: 2

----

Real-time M/EEG processing
--------------------------

Real-time M/EEG processing refers to any pipeline that acquires brain signals,
extracts features *as fast as the data arrives*, and delivers results — feedback,
monitoring outputs, or triggers — within a latency budget of milliseconds to
seconds.  MNE-RT covers the full range: continuous neurofeedback, event-related
ERP/TFR analysis, BCI decoding, and passive brain monitoring.

.. raw:: html

   <div style="overflow-x:auto; margin:20px 0;">
   <table style="border-collapse:separate; border-spacing:0; width:100%;
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          font-size:13px; background:#f8fafc; border-radius:12px;
          box-shadow:0 1px 4px rgba(0,0,0,.08);">
   <tr>
     <td style="padding:20px 24px; border-right:1px solid #e2e8f0; width:33%;">
       <div style="font-weight:700; color:#1e40af; margin-bottom:6px;">① Brain signal</div>
       <div style="color:#475569;">EEG or MEG electrodes pick up cortical oscillations
       from the scalp surface in real time.</div>
     </td>
     <td style="padding:20px 24px; border-right:1px solid #e2e8f0; width:33%;">
       <div style="font-weight:700; color:#059669; margin-bottom:6px;">② Feature extraction</div>
       <div style="color:#475569;">A signal processing pipeline distils the raw signal into
       one number per window — e.g., alpha power, laterality, or source connectivity.</div>
     </td>
     <td style="padding:20px 24px; width:33%;">
       <div style="font-weight:700; color:#d97706; margin-bottom:6px;">③ Feedback</div>
       <div style="color:#475569;">The feature value drives a visual, auditory, or
       haptic output signal — or is streamed for offline analysis or BCI control.</div>
     </td>
   </tr>
   </table>
   </div>

   <div style="text-align:center; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        font-size:13px; color:#64748b; margin:8px 0 24px 0;">
   ↑ The loop closes when the participant's self-regulation attempt changes the brain
   signal, which in turn changes the feedback — enabling operant conditioning of cortical activity.
   </div>

**Clinical applications** include ADHD (theta/beta training), chronic tinnitus
(auditory cortex and thalamo-cortical desynchronisation), post-stroke motor
rehabilitation (sensorimotor rhythm training), and depression (frontal alpha
asymmetry training).

----

MNE-RT Session Workflow
------------------------

Every session follows the same three-phase structure:

.. raw:: html

   <div style="margin:24px 0;">
   <div style="display:flex; flex-direction:column; gap:0;
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px; box-sizing: border-box;">

   <div style="display:flex; align-items:stretch; gap:0; box-sizing: border-box;">
     <div style="background:#1e40af; color:white; font-weight:700; padding:14px 18px;
          border-radius:10px 0 0 0; display:flex; align-items:center; width:150px; flex-shrink: 0;
          white-space:nowrap; box-sizing: border-box;">① Acquisition</div>
     <div style="background:#eff6ff; border:1px solid #bfdbfe; border-left:none;
          padding:14px 18px; border-radius:0 10px 0 0; flex:1; color:#1e3a8a; box-sizing: border-box;">
       Connect to an EEG/MEG amplifier (or mock replay) via LSL.
       <code style="background:#dbeafe; padding:1px 5px; border-radius:4px;">RTStream.connect_to_lsl()</code>
     </div>
   </div>

   <div style="display:flex; align-items:stretch; gap:0; box-sizing: border-box;">
     <div style="background:#065f46; color:white; font-weight:700; padding:14px 18px;
          display:flex; align-items:center; width:150px; flex-shrink: 0; white-space:nowrap; box-sizing: border-box;">② Baseline</div>
     <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-left:none;
          border-top:none; padding:14px 18px; flex:1; color:#14532d; box-sizing: border-box;">
       Record 2–5 min of resting-state data, automatically detects bad
       channels, computes noise covariance, and builds the inverse operator.
       <code style="background:#dcfce7; padding:1px 5px; border-radius:4px;">RTStream.record_baseline()</code>
     </div>
   </div>

   <div style="display:flex; align-items:stretch; gap:0; box-sizing: border-box;">
     <div style="background:#92400e; color:white; font-weight:700; padding:14px 18px;
          border-radius:0 0 0 10px; display:flex; align-items:center; width:150px; flex-shrink: 0;
          white-space:nowrap; box-sizing: border-box;">③ Real-time</div>
     <div style="background:#fffbeb; border:1px solid #fde68a; border-left:none;
          border-top:none; padding:14px 18px; border-radius:0 0 10px 0; flex:1; color:#78350f; box-sizing: border-box;">
       Run the closed-loop feedback loop for the prescribed duration.
       Feature values, reward signals, and quality metrics are streamed live and saved to BIDS.
       <code style="background:#fef3c7; padding:1px 5px; border-radius:4px;">RTStream.record_main()</code>
     </div>
   </div>

   </div>
   </div>

Phase 2 (baseline) is **optional for sensor-space modalities** but required
for source-space modalities (it provides the noise covariance used to build
the inverse operator) and for ERD/ERS-based modalities (it provides the
reference power).

----

Closed-Loop Architecture
------------------------

The diagram below shows how data flows inside MNE-RT during a processing window:

.. image:: _static/code_design.svg
   :alt: MNE-RT closed-loop architecture diagram
   :align: center
   :width: 75%

|
All modality computations run in a
:class:`concurrent.futures.ThreadPoolExecutor`, so slow operations (e.g.
source-space inverse) never block the feedback display.

----

Key Concepts
------------

Feature Modality
~~~~~~~~~~~

A **modality** is the brain feature that is being fed back to the participant.
MNE-RT implements 20 modalities across sensor space, connectivity, and source
space.  You select one or more per session:

.. raw:: html

   <div style="overflow-x:auto; margin:12px 0 20px 0;">
   <table style="border-collapse:collapse; width:100%;
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <thead>
     <tr>
       <th style="background:#334155;color:white;padding:7px 14px;border-radius:8px 0 0 0;">Category</th>
       <th style="background:#334155;color:white;padding:7px 14px;">Modalities</th>
       <th style="background:#334155;color:white;padding:7px 14px;border-radius:0 8px 0 0;">Typical use case</th>
     </tr>
   </thead>
   <tbody>
     <tr style="background:#f8fafc;">
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-weight:600;">Sensor power</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-family:monospace;font-size:12px;">sensor_power · band_ratio · erd_ers · laterality · laterality_erd_ers</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;">Alpha, SMR, or theta training; motor imagery</td>
     </tr>
     <tr style="background:#f1f5f9;">
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-weight:600;">Time-domain</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-family:monospace;font-size:12px;">hjorth · entropy · scp · instantaneous_phase</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;">SCP biofeedback; phase-triggered stimulation</td>
     </tr>
     <tr style="background:#f8fafc;">
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-weight:600;">Spectral peak</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-family:monospace;font-size:12px;">spectral_centroid · argmax_freq · peak_alpha_freq</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;">Individual alpha frequency tracking</td>
     </tr>
     <tr style="background:#f1f5f9;">
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-weight:600;">Connectivity</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-family:monospace;font-size:12px;">sensor_connectivity · connectivity_ratio · cfc_sensor · sensor_graph</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;">Inter-hemispheric or thalamocortical coupling</td>
     </tr>
     <tr style="background:#f8fafc;">
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-weight:600;">Source space</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;font-family:monospace;font-size:12px;">source_power · source_connectivity · source_graph</td>
       <td style="padding:6px 14px;border:1px solid #e2e8f0;">Region-specific analysis; tinnitus, depression</td>
     </tr>
   </tbody>
   </table>
   </div>

See :doc:`modalities` for the mathematical definition of each.

Feedback Protocol
~~~~~~~~~~~

A **protocol** decides *when* to deliver a reward and *how much*.  It sits
between the raw modality value and the feedback display.  MNE-RT ships ten
protocols:

.. raw:: html

   <pre style="background:#1e293b; color:#e2e8f0; padding:16px 20px;
        border-radius:10px; font-size:12px; line-height:1.8; overflow-x:auto;
        font-family:'JetBrains Mono','Fira Code','Courier New',monospace;">
   feature value  ──▶  Protocol  ──▶  (crossed: bool, magnitude: float)
                    │
                    ├── ThresholdProtocol   fixed or slowly-adapting threshold
                    ├── ZScoreProtocol      running z-score; self-calibrating
                    ├── PercentileProtocol  tracks own rolling distribution
                    ├── LinearTrendProtocol rewards sustained directional change
                    ├── RLProtocol          ε-greedy; finds threshold automatically
                    ├── TransferProtocol    seeded from a prior session file
                    ├── OperantProtocol     FR / VR / FI / VI reward schedules
                    ├── ShamProtocol        double-blind sham wrapper
                    ├── UpDownStaircaseProtocol  converges to target success rate
                    └── MultiBandProtocol   two-band AND / OR logic
   </pre>

See :doc:`protocols` for formulas and selection guidance.

Feature Combiner
~~~~~~~~~~~~~~~~

A **feature combiner** reduces N parallel feature values to a single mixed
score.  This is useful when several modalities capture complementary aspects of
the brain state and you want one unified number for the protocol and display.
MNE-RT provides four combiners in :mod:`mne_rt.combiners`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Strategy
   * - :class:`~mne_rt.combiners.WeightedSumCombiner`
     - Normalised weighted sum ``Σ(wᵢ·xᵢ)/Σ(wᵢ)``.  Intuitive, interpretable.
       Negative weights allowed (e.g. subtract theta from alpha).
   * - :class:`~mne_rt.combiners.GeometricMeanCombiner`
     - Weighted geometric mean ``exp(Σ wᵢ·log(xᵢ)/Σwᵢ)``.  Best for positive
       ratio or power features whose effects are multiplicative.
   * - :class:`~mne_rt.combiners.ZScoredNormCombiner`
     - Z-score each feature against a warmup baseline, then take the Euclidean
       norm divided by ``√n``.  Returns a unit-free "deviation from baseline"
       score regardless of feature units or dynamic range.
   * - :class:`~mne_rt.combiners.LearnedCombiner`
     - Pass the feature vector to any fitted ``sklearn``-compatible estimator
       (e.g. ``Ridge``, ``PLSRegression``).  Requires an offline calibration
       step but can capture non-linear feature interactions.

Example — blend alpha power and interhemispheric laterality::

    from mne_rt.combiners import WeightedSumCombiner

    combiner = WeightedSumCombiner(
        weights={"sensor_power": 0.6, "laterality": 0.4}
    )
    # Call once per window inside your analysis loop:
    mixed = combiner.combine({"sensor_power": 1.5, "laterality": 0.3})

Analysis Window
~~~~~~~~~~~~~~~

The **analysis window** (``winsize``) is the length of the EEG/MEG segment
used to compute each feature value.  Shorter windows give more frequent updates
(lower latency) but noisier estimates; longer windows give more stable estimates
at the cost of update frequency.

.. raw:: html

   <div style="overflow-x:auto; margin:12px 0 20px 0;">
   <table style="border-collapse:collapse; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <thead>
     <tr>
       <th style="background:#334155;color:white;padding:6px 14px;">Window size</th>
       <th style="background:#334155;color:white;padding:6px 14px;">Update rate</th>
       <th style="background:#334155;color:white;padding:6px 14px;">Best for</th>
     </tr>
   </thead>
   <tbody>
     <tr style="background:#f8fafc;"><td style="padding:5px 14px;border:1px solid #e2e8f0;">0.5 s</td><td style="padding:5px 14px;border:1px solid #e2e8f0;">2 Hz</td><td style="padding:5px 14px;border:1px solid #e2e8f0;">Fast motor-imagery, SCP, instantaneous phase</td></tr>
     <tr style="background:#f1f5f9;"><td style="padding:5px 14px;border:1px solid #e2e8f0;">1.0 s</td><td style="padding:5px 14px;border:1px solid #e2e8f0;">1 Hz</td><td style="padding:5px 14px;border:1px solid #e2e8f0;">Standard alpha / SMR training <em>(recommended default)</em></td></tr>
     <tr style="background:#f8fafc;"><td style="padding:5px 14px;border:1px solid #e2e8f0;">2.0 s</td><td style="padding:5px 14px;border:1px solid #e2e8f0;">0.5 Hz</td><td style="padding:5px 14px;border:1px solid #e2e8f0;">Source space, connectivity (needs longer segment for stable PSD)</td></tr>
     <tr style="background:#f1f5f9;"><td style="padding:5px 14px;border:1px solid #e2e8f0;">4.0 s</td><td style="padding:5px 14px;border:1px solid #e2e8f0;">0.25 Hz</td><td style="padding:5px 14px;border:1px solid #e2e8f0;">Low-frequency bands (delta, theta); graph learning</td></tr>
   </tbody>
   </table>
   </div>

----

Step-by-Step: First Session
----------------------------

**Step 1 — Install MNE-RT**

.. code-block:: bash

    pip install "mne-rt[full]"

**Step 2 — Run the demo (no hardware needed)**

The demo streams simulated EEG internally and runs a full real-time session with
live topo display:

.. code-block:: bash

    mne-rt demo --duration 60 --modality sensor_power erd_ers

**Step 3 — Record a baseline**

Connect your amplifier, then record a 2-minute resting-state baseline.
MNE-RT detects bad channels, computes ICA, and saves a noise covariance file:

.. code-block:: bash

    mne-rt baseline --subject sub01 --subjects-dir /data/subjects

Or in Python:

.. code-block:: python

    from mne_rt import RTStream

    nf = RTStream(
        subject_id="sub01",
        session="01",
        subjects_dir="/data/subjects",
        montage="easycap-M1",
    )
    nf.connect_to_lsl()          # connects to live amplifier
    nf.record_baseline(duration=120)

**Step 4 — Run the real-time session**

.. code-block:: python

    from mne_rt.protocols import ZScoreProtocol

    proto = ZScoreProtocol(direction="up", zscore_threshold=0.5, warmup_windows=20)

    nf.record_main(
        duration=600,                       # 10 minutes
        modality=["sensor_power"],          # alpha power
        protocol=proto,
        show_nf_signal=True,
        show_topo=True,
        track_artifact_rate=True,           # flag windows with artefacts
        track_snr=True,                     # per-window SNR log
    )
    nf.save()                               # writes BIDS-compatible output

With ``show_nf_signal=True``, the protocol's current threshold — fixed for
:class:`~mne_rt.protocols.ThresholdProtocol`, or converted from the z-score
boundary here since ``proto`` is a :class:`~mne_rt.protocols.ZScoreProtocol`
— is drawn live as a dashed line on the NFPlot trace, so the participant
(or operator) can see exactly where the reward boundary sits at every moment.

**Step 5 — Inspect results**

.. code-block:: python

    import json
    data = json.load(open("sub-sub01_ses-01_task-nf_beh.json"))
    nf_values  = data["data"]["sensor_power"]
    print(f"Mean alpha power : {sum(nf_values)/len(nf_values):.4f}")
    print(f"Artifact rate    : {nf.artifact_rate:.1%}")

----

Multi-Session Protocols
-----------------------

For longitudinal training studies, MNE-RT provides two patterns:

**Multi-block session** — run several recording blocks with rest periods in between,
all within one Python call:

.. code-block:: python

    nf.run_blocks(
        blocks=[
            {"duration": 300, "modality": ["sensor_power"]},
            {"duration": 300, "modality": ["sensor_power"]},
        ],
        rest_duration=60,           # 60 s rest between blocks
    )
    # Results per block: nf.block_nf_data, nf.block_artifact_rates

**Cross-session transfer** — seed the protocol's running statistics from the
previous session's file so rewards start immediately (no warmup phase):

.. code-block:: python

    from mne_rt.protocols import TransferProtocol

    proto = TransferProtocol(
        fname="sub-sub01_ses-01_task-nf_beh.json",
        modality="sensor_power",
        direction="up",
        zscore_threshold=0.5,
    )
    nf.record_main(duration=600, modality=["sensor_power"], protocol=proto)

**Offline replay** — re-run the exact same signal-processing pipeline on a
previously recorded file, useful for parameter tuning without a participant:

.. code-block:: python

    nf.replay(
        fname="sub-sub01_ses-01_task-nf_raw.fif",
        duration=600,
        modality=["erd_ers"],
    )

----

Choosing a Modality and Protocol
---------------------------------

.. raw:: html

   <div style="overflow-x:auto; margin:12px 0 20px 0;">
   <table style="border-collapse:collapse; width:100%;
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <thead>
     <tr>
       <th style="background:#1e40af;color:white;padding:7px 14px;border-radius:8px 0 0 0;">Clinical goal</th>
       <th style="background:#1e40af;color:white;padding:7px 14px;">Modality</th>
       <th style="background:#1e40af;color:white;padding:7px 14px;">Protocol</th>
       <th style="background:#1e40af;color:white;padding:7px 14px;border-radius:0 8px 0 0;">Notes</th>
     </tr>
   </thead>
   <tbody>
     <tr style="background:#eff6ff;">
       <td style="padding:6px 14px;border:1px solid #dbeafe;font-weight:600;">Alpha up-training</td>
       <td style="padding:6px 14px;border:1px solid #dbeafe;font-family:monospace;font-size:12px;">sensor_power</td>
       <td style="padding:6px 14px;border:1px solid #dbeafe;font-family:monospace;font-size:12px;">ZScoreProtocol</td>
       <td style="padding:6px 14px;border:1px solid #dbeafe;">Classic relaxation / pain / tinnitus protocol</td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:6px 14px;border:1px solid #dcfce7;font-weight:600;">Motor imagery (BCI)</td>
       <td style="padding:6px 14px;border:1px solid #dcfce7;font-family:monospace;font-size:12px;">laterality / erd_ers</td>
       <td style="padding:6px 14px;border:1px solid #dcfce7;font-family:monospace;font-size:12px;">ThresholdProtocol</td>
       <td style="padding:6px 14px;border:1px solid #dcfce7;">C3 vs C4 ERD for left/right hand imagery</td>
     </tr>
     <tr style="background:#eff6ff;">
       <td style="padding:6px 14px;border:1px solid #dbeafe;font-weight:600;">ADHD (theta/beta)</td>
       <td style="padding:6px 14px;border:1px solid #dbeafe;font-family:monospace;font-size:12px;">band_ratio</td>
       <td style="padding:6px 14px;border:1px solid #dbeafe;font-family:monospace;font-size:12px;">PercentileProtocol</td>
       <td style="padding:6px 14px;border:1px solid #dbeafe;">Reward when theta/beta ratio drops below 75th percentile</td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:6px 14px;border:1px solid #dcfce7;font-weight:600;">Tinnitus (source)</td>
       <td style="padding:6px 14px;border:1px solid #dcfce7;font-family:monospace;font-size:12px;">source_power / source_connectivity</td>
       <td style="padding:6px 14px;border:1px solid #dcfce7;font-family:monospace;font-size:12px;">ZScoreProtocol</td>
       <td style="padding:6px 14px;border:1px solid #dcfce7;">Requires MRI and baseline inverse operator</td>
     </tr>
     <tr style="background:#eff6ff;">
       <td style="padding:6px 14px;border:1px solid #dbeafe;font-weight:600;">SCP biofeedback</td>
       <td style="padding:6px 14px;border:1px solid #dbeafe;font-family:monospace;font-size:12px;">scp</td>
       <td style="padding:6px 14px;border:1px solid #dbeafe;font-family:monospace;font-size:12px;">ThresholdProtocol</td>
       <td style="padding:6px 14px;border:1px solid #dbeafe;">Requires DC-coupled amplifier; epilepsy, attention</td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:6px 14px;border:1px solid #dcfce7;font-weight:600;">No calibration data</td>
       <td style="padding:6px 14px;border:1px solid #dcfce7;font-family:monospace;font-size:12px;"><em>any</em></td>
       <td style="padding:6px 14px;border:1px solid #dcfce7;font-family:monospace;font-size:12px;">RLProtocol</td>
       <td style="padding:6px 14px;border:1px solid #dcfce7;">Self-calibrating — no baseline or warmup needed</td>
     </tr>
   </tbody>
   </table>
   </div>

----

Quality Monitoring
------------------

MNE-RT tracks two session-level quality metrics automatically when requested:

.. code-block:: python

    nf.record_main(
        ...,
        track_artifact_rate=True,    # default threshold: 100 µV
        track_snr=True,
    )

    # After session:
    print(f"Artifact rate : {nf.artifact_rate:.1%}")   # fraction of windows with any channel > 100 µV
    print(f"Mean SNR      : {sum(nf.snr_data)/len(nf.snr_data):.1f} dB")

A session with artifact rate > 20 % should be reviewed for electrode impedance
issues or movement artefacts before proceeding.

For detailed real-time quality control, see the
:class:`~mne_rt.tools.BadChannelDetector` and
:class:`~mne_rt.tools.RiemannianPotatoDetector` classes documented in
:doc:`denoising`.

----

Further Reading
---------------

- :doc:`modalities` — mathematical definitions of all 20 feature modalities
- :doc:`protocols` — reward protocol formulas, selection table, and examples
- :doc:`denoising` — artifact correction algorithms and benchmarks
- :doc:`api` — complete class and function reference
- :doc:`cli` — full CLI reference (``mne-rt run``, ``mne-rt baseline``, ``mne-rt demo``)
