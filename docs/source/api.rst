.. _api:

API Reference
=============

This page provides the complete API reference for **MNE-RT**.

Core
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.RTStream
   mne_rt.RTEpochs
   mne_rt.ArrayStream

Visualisation
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.viz.NFPlot
   mne_rt.viz.RawPlot
   mne_rt.viz.EpochPlot
   mne_rt.viz.BrainPlot
   mne_rt.viz.TopomapPlot
   mne_rt.viz.TopoPlot
   mne_rt.viz.ButterflyPlot
   mne_rt.viz.TFRPlot
   mne_rt.viz.CompareEvoked

Artifact correction
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.tools.AdaptiveLMSFilter
   mne_rt.tools.ORICA
   mne_rt.tools.GEDAIDenoiser
   mne_rt.tools.ASRDenoiser
   mne_rt.tools.RTMaxwellFilter

Quality control
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.tools.BadChannelDetector
   mne_rt.tools.RiemannianPotatoDetector

NF Protocols
------------

See :doc:`protocols` for the full protocol guide with formulas and examples.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.protocols.ThresholdProtocol
   mne_rt.protocols.ZScoreProtocol
   mne_rt.protocols.PercentileProtocol
   mne_rt.protocols.LinearTrendProtocol
   mne_rt.protocols.ShamProtocol
   mne_rt.protocols.UpDownStaircaseProtocol
   mne_rt.protocols.MultiBandProtocol
   mne_rt.protocols.RLProtocol
   mne_rt.protocols.OperantProtocol
   mne_rt.protocols.TransferProtocol

Feature combiners
-----------------

Reduce multiple parallel NF feature values to a single mixed feedback score.
See :class:`~mne_rt.combiners.FeatureCombiner` for the base-class interface.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.combiners.FeatureCombiner
   mne_rt.combiners.WeightedSumCombiner
   mne_rt.combiners.GeometricMeanCombiner
   mne_rt.combiners.ZScoredNormCombiner
   mne_rt.combiners.LearnedCombiner

Feedback output
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.OSCSender
   mne_rt.LSLSender

Tools & utilities
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.tools.simulate_raw
   mne_rt.tools.simulate_nf_session
   mne_rt.modalities.ModalityMixin

BIDS I/O
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.tools.save_as_bids

Logging
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_rt.set_log_level
