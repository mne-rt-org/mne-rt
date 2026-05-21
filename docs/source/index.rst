.. raw:: html

    <div style="text-align:center; margin-bottom:20px;">
        <img src="_static/ANT_Logo_Horizontal.svg" alt="ANT Logo" width="500"/>
    </div>

**Advanced Neurofeedback Toolbox (ANT)** is an open-source Python library for
**real-time M/EEG neurofeedback**, built on `MNE-Python <https://mne.tools>`_
and the `Lab Streaming Layer <https://labstreaminglayer.org>`_ (LSL).
It covers the full closed-loop pipeline, from amplifier to 3-D brain display,
in a single, researcher-friendly API designed for both clinical and basic-science
applications.

.. raw:: html

    <div style="height:16px;"></div>

Key capabilities
----------------

.. list-table::
   :widths: 1 45
   :header-rows: 0

   * - ✔
     - **Multiple real-time NF modalities** — sensor power, ERD/ERS, Hjorth parameters,
       spectral centroid and peak, band ratio, cross-frequency coupling (CFC),
       connectivity, instantaneous phase (Hilbert analytic
       signal), graph-theory metrics, and more.
       See :doc:`modalities` for the full list.

   * - ✔
     - **Sensor and source-space** processing using
       `MNE <https://mne.tools>`_ inverse operators (eLORETA, MNE, dSPM).
       Compute and stream cortical-source activity.

   * - ✔
     - **Live artifact correction** —
       :class:`~ant.tools.ORICA` (online ICA),
       :class:`~ant.tools.AdaptiveLMSFilter` (adaptive least-mean-squares),
       :class:`~ant.tools.GEDAIDenoiser` (generalised eigendecomposition),
       :class:`~ant.tools.ASRDenoiser` (artifact subspace reconstruction), and
       :class:`~ant.tools.RTMaxwellFilter` (real-time Maxwell/SSS filtering for MEG).
       See :doc:`denoising` for algorithm details and benchmarks.

   * - ✔
     - **Real-time quality control** — :class:`~ant.tools.BadChannelDetector`
       flags flat, noisy, or de-correlated channels every window using a
       robust rolling-vote mechanism.  :class:`~ant.tools.RiemannianPotatoDetector`
       detects artifactual epochs by measuring Riemannian distance from a
       clean-data geometric mean on the SPD manifold.

   * - ✔
     - **Adaptive NF protocols** — :class:`~ant.protocols.ThresholdProtocol`,
       :class:`~ant.protocols.ZScoreProtocol`,
       :class:`~ant.protocols.PercentileProtocol`,
       :class:`~ant.protocols.LinearTrendProtocol`
       (OLS-based trend reward — encourages sustained directional change rather
       than single-window threshold crossings),
       :class:`~ant.protocols.ShamProtocol`
       (double-blind sham wrapper for within-session RCTs),
       :class:`~ant.protocols.UpDownStaircaseProtocol`
       (adaptive psychophysics staircase converging to a target success rate),
       :class:`~ant.protocols.MultiBandProtocol`
       (simultaneous two-band reward — e.g., alpha↑ + theta↓),
       :class:`~ant.protocols.RLProtocol`
       (ε-greedy reinforcement-learning threshold search — fully self-calibrating),
       :class:`~ant.protocols.OperantProtocol`
       (partial reinforcement schedules: FR, VR, FI, VI — wraps any inner protocol), and
       :class:`~ant.protocols.TransferProtocol`
       (cross-session z-score seeded from a prior session file — zero warmup)
       give fine-grained control over when to issue a reward.
       See :doc:`protocols` for formulas, selection guide, and examples.

   * - ✔
     - **Three parallel visualisation windows** — a scrolling
       :class:`~ant.viz.NFSignalPlot`, a live
       `MNE <https://mne.tools/stable/visualization.html>`_-style :class:`~ant.viz.TopoPlot`,
       and an interactive :class:`~ant.viz.BrainPlot` (3-D cortical surface with
       colour-mapped activity, hemisphere toggles, and surface switching).

   * - ✔
     - **Dual feedback output** — broadcast values via OSC (Max/MSP,
       SuperCollider, TouchDesigner) with :class:`~ant.osc.OSCSender`, or over
       `LSL <https://labstreaminglayer.org>`_ with :class:`~ant.lsl_output.LSLSender`
       for low-latency same-machine integration with PsychoPy, OpenViBE, BCI2000,
       and other LSL-aware apps.

   * - ✔
     - **CLI** — launch full NF sessions with a single ``ANT run`` command,
       driven by a YAML config file. See :doc:`cli`.

.. raw:: html

    <div style="height:16px;"></div>

.. tabs::

   .. tab:: NF Signal

      .. raw:: html

         <div style="text-align:center; margin: 20px 0;">
             <video width="100%" style="max-width: 850px; border-radius: 15px; display: block; margin: 0 auto;" autoplay muted loop>
                 <source src="_static/NFSignal.mp4" type="video/mp4">
             </video>
         </div>

   .. tab:: Topo Plot

      .. raw:: html

         <div style="text-align:center; margin: 20px 0;">
             <video width="100%" style="max-width: 850px; border-radius: 15px; display: block; margin: 0 auto;" autoplay muted loop>
                 <source src="_static/TopoPlot.mp4" type="video/mp4">
             </video>
         </div>

   .. tab:: Brain Plot

      .. raw:: html

         <div style="text-align:center; margin: 20px 0;">
             <video width="100%" style="max-width: 850px; border-radius: 15px; display: block; margin: 0 auto;" autoplay muted loop>
                 <source src="_static/BrainPlot.mp4" type="video/mp4">
             </video>
         </div>

.. toctree::
   :hidden:
   :caption: Getting started

   install
   tutorial

.. toctree::
   :hidden:
   :caption: Reference

   api
   cli
   denoising
   modalities
   protocols

.. toctree::
   :hidden:
   :caption: Examples

   auto_examples/index

.. raw:: html

    <div style="height:24px;"></div>

Quick install
-------------

.. tabs::

   .. tab:: pip

      .. code-block:: bash

          pip install ant-nf                 # core  (MNE, LSL, OSC included)
          pip install "ant-nf[full]"         # + 3-D viz, dev tools, docs

   .. tab:: uv

      .. code-block:: bash

          # Install uv once
          curl -LsSf https://astral.sh/uv/install.sh | sh

          uv pip install ant-nf
          uv pip install "ant-nf[full]"      # + 3-D viz, dev tools, docs

          # Editable install from source
          git clone https://github.com/payamsash/ANT.git
          cd ANT && uv pip install -e ".[dev]"

   .. tab:: conda / mamba

      .. code-block:: bash

          mamba create -n ant python=3.11
          mamba activate ant
          pip install "ant-nf[full]"

See :doc:`install` for full instructions.

Quick start
-----------

.. code-block:: python

    from ant import NFRealtime

    nf = NFRealtime(
        "sub01",
        session="01",
        subjects_dir="/data/subjects",
        montage="easycap-M1",
    )
    nf.connect_to_lsl(mock_lsl=True)          # or connect to a real amplifier
    nf.record_main(
        duration=300,
        modality=["sensor_power", "erd_ers"],
        show_nf_signal=True,
    )

Pipeline overview
-----------------

.. raw:: html

   <div style="overflow-x:auto; margin:16px 0;">
   <table style="border-collapse:separate; border-spacing:0; width:100%; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <thead>
     <tr>
       <th style="background:#1e40af;color:white;padding:8px 14px;border-radius:8px 0 0 0;white-space:nowrap;">Stage</th>
       <th style="background:#1e40af;color:white;padding:8px 14px;">Class / Function</th>
       <th style="background:#1e40af;color:white;padding:8px 14px;border-radius:0 8px 0 0;">Notes</th>
     </tr>
   </thead>
   <tbody>
     <tr style="background:#eff6ff;">
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;font-weight:600;white-space:nowrap;">① Acquisition</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;"><code>NFRealtime.connect_to_lsl()</code></td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Hardware amplifier or mock replay via mne-lsl <code>StreamInlet</code></td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;font-weight:600;white-space:nowrap;">② Baseline</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;"><code>NFRealtime.record_baseline()</code></td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">Bad-channel detection · ICA · noise covariance · inverse operator</td>
     </tr>
     <tr style="background:#eff6ff;">
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;font-weight:600;white-space:nowrap;">③ Quality control</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;"><code>BadChannelDetector</code> · <code>RiemannianPotatoDetector</code></td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Flags flat / noisy channels and artifactual covariance windows every epoch</td>
     </tr>
     <tr style="background:#fffbeb;">
       <td style="padding:7px 14px;border-bottom:1px solid #fef3c7;font-weight:600;white-space:nowrap;">④ Artifact correction</td>
       <td style="padding:7px 14px;border-bottom:1px solid #fef3c7;"><code>ORICA</code> · <code>AdaptiveLMS</code> · <code>GEDAIDenoiser</code> · <code>ASRDenoiser</code> · <code>RTMaxwellFilter</code></td>
       <td style="padding:7px 14px;border-bottom:1px solid #fef3c7;">One or more methods selected per session; all operate sample-by-sample</td>
     </tr>
     <tr style="background:#f5f3ff;">
       <td style="padding:7px 14px;border-bottom:1px solid #ede9fe;font-weight:600;white-space:nowrap;">⑤ Feature extraction</td>
       <td style="padding:7px 14px;border-bottom:1px solid #ede9fe;"><code>record_main(modality=[…])</code></td>
       <td style="padding:7px 14px;border-bottom:1px solid #ede9fe;">20 NF modalities in sensor or source space; parallel thread-pool per window</td>
     </tr>
     <tr style="background:#ecfdf5;">
       <td style="padding:7px 14px;border-bottom:1px solid #d1fae5;font-weight:600;white-space:nowrap;">⑥ NF protocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #d1fae5;"><code>ThresholdProtocol</code> · <code>ZScoreProtocol</code> · <code>PercentileProtocol</code> · <code>LinearTrendProtocol</code> · <code>ShamProtocol</code> · <code>UpDownStaircaseProtocol</code> · <code>MultiBandProtocol</code> · <code>RLProtocol</code> · <code>OperantProtocol</code> · <code>TransferProtocol</code></td>
       <td style="padding:7px 14px;border-bottom:1px solid #d1fae5;">Maps raw feature value → reward signal; all are stateful and adaptive</td>
     </tr>
     <tr style="background:#eff6ff;">
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;font-weight:600;white-space:nowrap;">⑦ Visualisation</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;"><code>NFSignalPlot</code> · <code>TopoPlot</code> · <code>BrainPlot</code></td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Scrolling signal · per-band scalp topo · 3-D cortical surface with Qt controls</td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:7px 14px;font-weight:600;white-space:nowrap;">⑧ Feedback output</td>
       <td style="padding:7px 14px;"><code>OSCSender</code> · <code>LSLSender</code></td>
       <td style="padding:7px 14px;">UDP/OSC to Max·MSP, SuperCollider, PD · LSL for PsychoPy, OpenViBE, BCI2000</td>
     </tr>
   </tbody>
   </table>
   </div>

Cite
----

If you use ANT, please cite :footcite:`shabestari2025advances`.

.. footbibliography::

.. tabs::

    .. tab:: APA

        .. code-block:: none

            Shabestari, P. S., Ribes, D., Défayes, L., Cai, D., Groves, E.,
            Behjat, H. H., … & Neff, P. (2025). Advances on Real Time M/EEG
            Neural Feature Extraction. IEEE CBMS 2025.

    .. tab:: BibTeX

        .. code-block:: bibtex

            @inproceedings{shabestari2025advances,
                title   = {Advances on Real Time M/EEG Neural Feature Extraction},
                author  = {Shabestari, Payam S and others},
                booktitle = {2025 IEEE 38th CBMS},
                pages   = {337--338},
                year    = {2025},
                organization = {IEEE}
            }

.. raw:: html

    <div style="height:20px;"></div>

.. image:: _static/SNF.png
    :align: right
    :alt: SNSF
    :width: 320

Development was supported by the
`Swiss National Science Foundation <https://www.snf.ch/en>`_ (grant number — 208164).
