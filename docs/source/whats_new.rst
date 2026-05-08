.. _whats_new:

What's new
==========

.. _changes_1_0_0:

Version 1.0.0
-------------

*Released: May 2026*

New features
^^^^^^^^^^^^

- :class:`~ant.tools.GEDAIDenoiser` — GED-based spatial filter for real-time
  artifact removal via generalised eigendecomposition.
- :class:`~ant.osc.OSCSender` — send NF feature values over OSC to Max/MSP,
  SuperCollider, Pure Data, or any OSC-compatible endpoint.  OSC support is
  now bundled in the core package (no separate ``[osc]`` extra needed).
- :func:`~ant.set_log_level` and ``@verbose`` decorator following MNE-Python
  verbosity conventions.
- Four new NF modalities: ``erd_ers``, ``laterality``, ``hjorth``,
  ``spectral_centroid``.
- :class:`~ant.viz.BrainPlot` — interactive 3D cortical surface with
  threshold, opacity, and colour-map sliders; hemisphere toggles; surface
  switching (``inflated``, ``pial``, ``white``, ``sphere``) via keyboard
  shortcuts; dark navy background.
- :class:`~ant.viz.NFSignalPlot` — scrolling multi-modality NF signal monitor
  with fine graph-paper grid, per-modality auto-range, 30-fps linear
  interpolation between window estimates, and a file-save screenshot dialog.
- Complete CLI: ``ANT info``, ``ANT demo``, ``ANT baseline``, ``ANT run``.
- ``pip install ANT``, ``uv pip install ANT``, and conda ``environment.yml``
  installation paths.
- Bundled ``config_methods.yml`` so installed wheels work without the
  source tree.
- :meth:`~ant.NFRealtime.get_blink_template` now exposes full MNE ICA
  constructor parameters (``n_components``, ``fit_params``, ``random_state``)
  and an ICAlabel confidence ``threshold``.

Improvements
^^^^^^^^^^^^

- :class:`~ant.tools.ORICA` whitening matrix inverse fixed; mixing matrix
  cached; added ``denoise()`` and ``update_and_denoise()`` convenience methods.
- ``log_degree_barrier`` warm-starting cache, 2-node analytic fast path,
  and configurable convergence tolerance.
- All public classes and methods now have MNE-style NumPy docstrings with
  ``Parameters``, ``Returns``, ``Examples``, and ``See Also`` sections.
- Brain and NF signal plots now run in separate Qt timer loops (33 ms signal,
  200 ms brain) so VTK renders never block the signal display.
