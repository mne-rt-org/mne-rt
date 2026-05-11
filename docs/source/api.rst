.. _api:

API Reference
=============

This page provides the complete API reference for the
**Advanced Neurofeedback Toolbox (ANT)**.

Core
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.NFRealtime

Visualisation
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.viz.NFSignalPlot
   ant.viz.BrainPlot

Artifact correction
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.tools.ORICA
   ant.tools.GEDAIDenoiser
   ant.tools.ASRDenoiser
   ant.tools.RTMaxwellFilter

NF Protocols
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.protocols.ThresholdProtocol

OSC output
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.osc.OSCSender

Tools & utilities
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.tools.simulation.simulate_raw
   ant.tools.simulation.simulate_eeg_raw

Logging
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.set_log_level
