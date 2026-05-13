.. raw:: html

    <div style="text-align:center; margin-bottom:20px;">
        <img src="_static/ANT_Logo_Horizontal.svg" alt="ANT Logo" width="500"/>
    </div>

.. raw:: html

    <div style="height:20px;"></div>

**Advanced Neurofeedback Toolbox (ANT)** is an open-source Python package for
**real-time M/EEG neurofeedback**, built on MNE-Python and the Lab Streaming
Layer (LSL).  It covers the full closed-loop pipeline — from amplifier to
3D brain display — in a single, researcher-friendly API.

.. raw:: html

    <div style="height:10px;"></div>

Key capabilities
----------------

- **Real-time NF feature extraction** — 17 modalities (alpha power,
  ERD/ERS, Hjorth, spectral centroid, CFC, wPLI, graph metrics, …)
- **Sensor-space & source-space** processing with MNE inverse operators
- **Live artifact correction** — ORICA, adaptive LMS, GEDAI, ASR, Maxwell/tSSS
- **Real-time quality control** — :class:`~ant.tools.BadChannelDetector`
  flags flat, noisy, or de-correlated channels on every incoming window
- **NF protocols** — threshold, z-score, percentile, and linear-trend reward
  criteria with rolling statistics
- **Three parallel visualisation windows** — StreamViewer, NF signal plot,
  3D brain surface
- **OSC and LSL output** — broadcast feedback values via OSC (Max/MSP,
  SuperCollider) or LSL (:class:`~ant.lsl_output.LSLSender`) for
  lower-latency same-machine feedback
- **CLI** — run full sessions with a single ``ANT run`` command

.. raw:: html

    <div style="height:20px;"></div>

.. raw:: html

    <!-- Demo video -->
    <div style="text-align:center; margin-bottom: 20px;">
        <video width="680" height="392" autoplay muted loop
               style="border-radius: 12px; border: 2px solid rgba(200,200,200,0.3);
                      box-shadow: 0 8px 20px rgba(0,0,0,0.4);">
            <source src="_static/nf_demo.mov" type="video/quicktime">
        </video>
    </div>

    <div style="display:flex; justify-content:center; gap:16px; margin-top:10px;">
        <video width="332" height="192" autoplay muted loop
               style="border-radius: 12px; border: 2px solid rgba(200,200,200,0.3);
                      box-shadow: 0 6px 15px rgba(0,0,0,0.3);">
            <source src="_static/brain.mov" type="video/quicktime">
        </video>
        <video width="332" height="192" autoplay muted loop
               style="border-radius: 12px; border: 2px solid rgba(200,200,200,0.3);
                      box-shadow: 0 6px 15px rgba(0,0,0,0.3);">
            <source src="_static/VisualTree.mp4" type="video/mp4">
        </video>
    </div>

.. toctree::
   :hidden:
   :caption: Getting started

   install

.. toctree::
   :hidden:
   :caption: Reference

   api
   cli
   denoising
   modalities

.. toctree::
   :hidden:
   :caption: Examples

   auto_examples/index

.. raw:: html

    <div style="height:30px;"></div>

Quick install
-------------

.. code-block:: bash

    pip install ANT                 # core (OSC output included)
    pip install "ANT[full]"         # all extras (viz, dev, docs)

See :doc:`install` for conda / uv and editable-install instructions.

Quick start
-----------

.. code-block:: python

    from ant import NFRealtime

    nf = NFRealtime("sub01", visit=1, session="main",
                    subjects_dir="/data/subjects",
                    montage="easycap-M1")
    nf.connect_to_lsl(mock_lsl=True)              # or real amplifier
    nf.record_main(duration=300,
                   modality=["sensor_power", "erd_ers"],
                   show_nf_signal=True)

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
`Swiss National Science Foundation <https://www.snf.ch/en>`_ (grant number - 208164).
