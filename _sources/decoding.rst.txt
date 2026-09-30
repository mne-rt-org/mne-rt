.. _decoding:

Real-time Decoding
===================

:class:`~mne_rt.RTDecode` wraps an :mod:`mne.decoding` (:class:`~mne.decoding.CSP`,
or :class:`~mne.decoding.Scaler` + :class:`~mne.decoding.Vectorizer`) and
scikit-learn classifier ``sklearn.pipeline.Pipeline`` for real-time
single-trial classification — e.g. motor-imagery BCI control, or any other
paradigm where each acquisition window should be classified live.

Unlike the config-driven NF modalities in :doc:`modalities`, a decoder is a
Python object you fit **offline** on labelled calibration epochs, then attach
to a live session with :meth:`~mne_rt.RTStream.set_decoder`:

.. code-block:: text

   calibration epochs  ->  RTDecode.fit(X, y)  ->  RTStream.set_decoder()
   live window  ->  RTStream.record_main()  ->  RTDecode.predict_proba()  ->  nf_data["decode"]

.. contents::
   :local:
   :depth: 1

----

Fitting a decoder offline
--------------------------

``X`` is a 3D array of labelled calibration epochs
``(n_epochs, n_channels, n_times)`` — from a dedicated calibration block, a
prior session, or a subset of trials held out from the current one. ``y`` is
one label per epoch (numeric or string; string labels such as
``"left"``/``"right"`` are supported and give more readable
``decoder.classes_``).

.. code-block:: python

    from mne_rt import RTDecode

    X_cal = ...  # shape (n_epochs, n_channels, n_times)
    y_cal = ...  # shape (n_epochs,), e.g. ["left", "right", "left", ...]

    decoder = RTDecode(info=epochs_info, spatial_filter="csp", n_components=4)
    decoder.fit(X_cal, y_cal)

Two ``spatial_filter`` choices are available:

* ``"csp"`` (default) — :class:`~mne.decoding.CSP`, log-variance of the most
  discriminative spatial filters. Well suited to oscillatory paradigms such
  as motor imagery; requires at least two classes.
* ``"scaler"`` — :class:`~mne.decoding.Scaler` (channel-type-aware
  standardisation) + :class:`~mne.decoding.Vectorizer`. A simpler,
  filter-agnostic alternative when CSP's oscillatory-power assumption
  doesn't fit the decoding target (e.g. ERP decoding).

Any scikit-learn classifier can be passed via ``estimator=`` (default
``LogisticRegression()``); use one supporting ``predict_proba`` if you plan
to attach the decoder to a live session (see below).

----

Attaching and running it live
--------------------------------

A fitted decoder is attached with :meth:`~mne_rt.RTStream.set_decoder`, then
requested like any other modality via ``modality=["decode"]``:

.. code-block:: python

    from mne_rt import RTStream

    nf = RTStream(subject_id="sub01", montage="easycap-M1", data_type="eeg")
    nf.connect_to_lsl(...)  # or connect_to_array(...) for LSL-free testing/demos
    nf.set_decoder(decoder)
    nf.record_main(modality=["decode"], winsize=2.0)

    proba_right = nf.nf_data["decode"]  # one probability per window

``record_main`` queries :meth:`~mne_rt.RTDecode.predict_proba` once per
acquisition window and reports the probability of
``decoder.classes_[class_index]`` (``class_index`` defaults to ``1``,
configurable via ``config_methods.yml``'s ``decode`` section or
``modality_params``). The value flows through the same pipeline as every
other modality — EMA smoothing, z-scoring, reward protocols, plotting, and
BIDS export — which is why the ``"decode"`` modality always reports a
continuous probability rather than a discrete predicted label: a discrete
label doesn't survive smoothing/z-scoring meaningfully. Use
:meth:`~mne_rt.RTDecode.predict` directly (outside ``record_main``) if you
need the discrete label.

.. important::

   ``winsize`` in :meth:`~mne_rt.RTStream.record_main` must match the epoch
   duration ``X_cal`` was fit on, and the channel **count and order** must
   match too — a channel-count mismatch raises a clear error at
   ``record_main()`` start, but a same-count reordering is not detected and
   will silently degrade predictions.

----

Validating a decoder before deploying it
-------------------------------------------

``RTDecode.pipeline`` is a plain scikit-learn ``sklearn.pipeline.Pipeline``,
so any scikit-learn model-selection tool works directly on it — cross-validate
before attaching a decoder to a live session:

.. code-block:: python

    from sklearn.model_selection import ShuffleSplit, cross_val_score

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    scores = cross_val_score(decoder.pipeline, X_cal, y_cal, cv=cv)
    print(f"Accuracy: {scores.mean():.2f} +/- {scores.std():.2f}")

For CSP decoders, the learned spatial patterns can be inspected the same way
as in MNE's own decoding tutorials:

.. code-block:: python

    csp = decoder.pipeline.named_steps["csp"]
    csp.plot_patterns(epochs_info, components=range(csp.n_components))

See :ref:`sphx_glr_auto_examples_ex_motor_imagery_decode.py` for a complete
worked example (PhysioNet motor imagery, CSP, cross-validation, and a live
:meth:`~mne_rt.RTStream.connect_to_array` streaming demo).

For deeper offline validation before deploying a decoder — beyond a single
cross-validated accuracy number — MNE's own decoding examples gallery covers
techniques not wrapped by :class:`~mne_rt.RTDecode` but useful at the
calibration stage:

* `Decoding sensor space data with generalization across time and conditions
  <https://mne.tools/stable/auto_examples/decoding/decoding_time_generalization_conditions.html>`_ —
  checks whether a classifier trained at one time point generalises to
  others, useful for choosing *when* in the trial to fit the RTDecode window.
* `Representational Similarity Analysis
  <https://mne.tools/stable/auto_examples/decoding/decoding_rsa_sgskip.html>`_ —
  a representational-geometry analysis (not a classifier), useful for
  understanding *what* is separable in your data before committing to a
  single decision boundary.

----

See also
----------

* :doc:`modalities` — the full set of config-driven NF feature modalities,
  including :ref:`modality-decode`.
* :doc:`cli` — ``mne-rt demo-decode`` runs this whole workflow end-to-end
  from the command line.
* :class:`~mne_rt.combiners.LearnedCombiner` — for blending an already
  fitted regressor over *reduced scalar* NF features (rather than raw
  windows) into a combined feedback score.
