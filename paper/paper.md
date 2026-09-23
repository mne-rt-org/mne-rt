---
title: 'MNE-RT: A Python package for real-time M/EEG neurofeedback and brain-computer interfaces'
tags:
  - Python
  - neuroscience
  - electroencephalography
  - magnetoencephalography
  - neurofeedback
  - brain-computer interface
  - real-time
  - source localisation
authors:
  - name: Payam S. Shabestari
    orcid: 0000-0002-2647-4891
    corresponding: true
    affiliation: "1, 2, 3"
affiliations:
  - name: Department of Otorhinolaryngology, Head and Neck Surgery, University Hospital Zurich, University of Zurich, Zurich, Switzerland
    index: 1
    ror: 01462r250
  - name: Neuroscience Center Zurich, ETH Zurich and University of Zurich, Zurich, Switzerland
    index: 2
  - name: Fondation Campus Biotech Geneva (FCBG), Geneva, Switzerland
    index: 3
date: 22 September 2026
bibliography: paper.bib
---

# Summary

Closed-loop neurophysiology has to answer a question that offline analysis never
faces: can the pipeline get from amplifier to feedback before the brain state it
measured has passed? In neurofeedback and brain--computer interfacing (BCI), the
whole chain --- acquisition, artifact handling, feature extraction, decision and
display --- must complete inside a single analysis window, typically a few hundred
milliseconds.

`MNE-RT` is a Python package that implements that chain on top of `MNE-Python`
[@gramfort2013meg] and `MNE-LSL` [@mnelsl]. An experiment is an object. `RTStream`
connects to a Lab Streaming Layer (LSL) source [@kothe2025lsl], records a
resting-state baseline, and then runs a closed loop in which every window is
cleaned, reduced to one or more neural features, standardised against that
baseline, evaluated by an adaptive reward protocol, and pushed simultaneously to
live displays, an LSL outlet and OSC receivers. Sessions are written to disk in a
BIDS-compatible layout [@pernet2019bids] alongside per-window latency,
artifact-rate and signal-to-noise records, so that a session can be audited or
replayed offline through the identical code path.

The package provides 20 feature modalities spanning sensor and source space, 10
adaptive protocols, 5 online artifact-correction methods, 9 live visualisation
windows, and an `mne-rt` command-line interface. Three hardware-free paths ---
replaying any MNE-readable recording through a mock LSL stream, streaming an
in-memory array, and the bundled `mne-rt demo` --- allow the complete pipeline to
be run, taught and tested without an amplifier.

![The `MNE-RT` closed loop. Data arrive over LSL, are corrected in place, reduced
to features by a thread pool, standardised and combined, evaluated by an adaptive
protocol, and returned to the participant through displays, an LSL outlet or OSC.
\label{fig:arch}](figure1_architecture.png)

# Statement of need

Neurofeedback and BCI laboratories still assemble their real-time stack by hand
[@sitaram2017cln]. Streaming, artifact rejection, feature extraction, adaptive
thresholding, stimulus communication and logging are typically glued together per
study, and the glue --- not the science --- is where reproducibility is lost: the
threshold rule lives in an undocumented script, the achieved latency is never
measured, and the session cannot be replayed.

`MNE-RT` targets researchers who already analyse M/EEG with `MNE-Python` and want
their online experiment written in the same vocabulary, with the protocol, the
artifact method and the achieved timing recorded as data rather than as lab lore.

The gap it fills is specific. Sensor-space band power is well served by existing
real-time tools. *Source-space* neurofeedback is not, because the obvious
implementation misses the deadline. Training a region-to-region interaction ---
for example imaginary coherency [@nolte2004imcoh] between an LCMV-beamformed
[@vanveen1997lcmv] pair of regions of interest --- requires projecting every
window into source space and collapsing thousands of grid points into a handful of
ROI time courses. Measured on a 5 mm whole-brain volume source space (14,629
points), the straightforward `MNE-Python` path costs 650--800 ms per window
against a 500 ms hop, so the loop never closes. Deep targets make this worse and
more interesting: a volume source space additionally exposes subcortical ROIs such
as hippocampus, amygdala and thalamus, which no sensor-space feature can
isolate.
`MNE-RT` makes these targets reachable online, and in doing so makes a class of
closed-loop experiment practical that previously was not.

# State of the field

Within the MNE ecosystem, `mne-realtime` was the original real-time extension; it
is no longer maintained. `MNE-LSL` [@mnelsl] is actively maintained and provides
modern LSL bindings, stream inlets and outlets, file replay and online epoching.
`MNE-RT` does not duplicate it --- it *depends* on it as its acquisition backbone.
Contributing upstream was therefore not the alternative to building `MNE-RT`:
`MNE-LSL` is deliberately a streaming library, and neurofeedback protocols,
source-space feature kernels, reward logic and session provenance are
application-level concerns outside its scope.

Outside the MNE ecosystem, `Timeflux` [@clisson2019timeflux] and NeuXus model
real-time processing as a configurable graph of nodes; `MEDUSA` [@medusa] offers a
broad BCI ecosystem with its own paradigms and signal-processing stack; `OpenViBE`
[@renard2010openvibe] and `BCI2000` [@schalk2004bci2000] are mature C++ platforms
driven through graphical designers; NeuroPype is proprietary. These are capable
systems, but each asks the user to adopt a separate data model and toolchain, and
none offers an anatomically constrained source-space feature path. `MNE-RT`
contributes the combination that was missing: an MNE-native framework in which the
online feature is expressed with the same objects, montages, forward models and
inverse operators as the offline analysis that motivated it, and in which
source-space features run inside the real-time budget.

# Software design

`MNE-RT` is deliberately script-shaped rather than graph-shaped. A session reads
as `connect_to_lsl` → `record_baseline` → `record_main` → `save`, mirroring how
`MNE-Python` users already write offline analyses, instead of requiring an
experiment to be expressed as a dataflow configuration. The cost is a large
session class; the benefit is that an experiment is a readable, version-controlled
script.

The central design decision is how source-space features meet the deadline.
Rather than accelerating the generic pipeline, `MNE-RT` exploits the fact that it
is linear: LCMV with `pick_ori="max-power"` and minimum-norm variants with
`pick_ori="normal"` are linear operators, and `mean`/`mean_flip` label extraction
is a fixed sparse average. `SourceModel.roi_kernel()` therefore collapses
whitening, inverse operator and label extraction into a single
`(n_roi, n_channels)` matrix applied per window as one matrix product, reducing
650--800 ms to roughly 1 ms while agreeing with the `MNE-Python` result to
better than 1e-10 relative error, which the test suite asserts directly.
Free-orientation configurations are *not* linear --- taking the norm across
three orientations discards phase --- so the kernel is refused rather than returning a
plausible-looking wrong number. The same precompute-early principle governs the
rest of the loop: bad-channel detection, ICA, noise covariance, forward solution
and the Maxwell/SSS basis [@taulu2006spatiotemporal] are all built during the
baseline phase, leaving the online step a bounded per-window cost. The head model
is built lazily, so sensor-space sessions never download anatomical data.

Everything that varies between studies is a strategy object behind a small
interface. Protocols expose `evaluate(value) -> (crossed, magnitude)`, which keeps
z-score, percentile, staircase [@levitt1971transformed], reinforcement-learning,
operant-schedule, multi-band, cross-session-transfer and double-blind sham
protocols interchangeable; artifact correctors expose `fit`/`transform`, covering
adaptive LMS, ORICA, GEDAI, ASR [@mullen2015real] and real-time Maxwell filtering;
feature combiners include one that accepts any scikit-learn estimator. Features
themselves span band power and its derivatives (ERD/ERS
[@pfurtscheller1999event], laterality, individual peak frequency via spectral
parameterisation [@donoghue2020parameterizing]), connectivity, cross-frequency
coupling [@tort2010measuring], graph learning [@kalofolias2016learn] and online
decoding. A naming scheme (`base@label`) lets one measure run concurrently in
several bands with independent state and independent output channels.

Concurrency is explicit rather than framework-mediated: a daemon thread runs a
50%-overlap sliding window and fans modalities out across a thread pool, while the
Qt main thread refreshes displays at 30 fps, so visualisation cannot block
acquisition. Window onsets are taken from the LSL clock and recorded as `NaN`
rather than substituted when unavailable, keeping traces honestly alignable to a
stimulus log. Interoperability is through standard protocols --- an LSL outlet and
OSC --- rather than a bundled stimulus engine, so `MNE-RT` drives PsychoPy,
Max/MSP or Unity without owning them.

# Research impact statement

`MNE-RT` has been developed in the open since May 2024 --- over two years, more
than 500 commits and over 30 merged pull requests --- and is released on both
PyPI and conda-forge with versioned documentation, a tutorial series and an
executable example gallery. Correctness is enforced by over 950 automated tests
across 38 modules, run in continuous integration on Python 3.11--3.13, including
a job that resolves every dependency to its declared lower bound.

Its feature-extraction layer was presented at IEEE CBMS 2025
[@shabestari2025advances]. Development has been supported by Swiss National
Science Foundation grant 208164, *Advancing Neurofeedback in Tinnitus*, and the
package is the online implementation vehicle for methods developed in that
programme, including the holistic graph-learning network analysis reported by
@shabestari2026shared. The source-space capability described above was built to
meet the requirements of an ongoing clinical collaboration on neurofeedback for
aphasia, in which region-to-region connectivity between language areas is trained
in real time from a 64-channel EEG stream; that study is the first external
deployment and motivated the volume-source-space and connectivity work released in
version 1.2.

# AI usage disclosure

Generative AI tools (Anthropic's Claude, used through Claude Code) assisted with
parts of the implementation, the test suite, the documentation and the drafting of
this manuscript. All AI-assisted output was reviewed, corrected and validated by
the author, who takes full responsibility for the correctness, originality and
licensing of the software and of this paper. Numerical claims in this paper were
verified against the package's own test suite and measurements.

# Acknowledgements

This work was supported by the Swiss National Science Foundation (grant 208164,
*Advancing Neurofeedback in Tinnitus*). I thank the `MNE-Python` and `MNE-LSL`
developer communities, on whose work this package depends.

# References
