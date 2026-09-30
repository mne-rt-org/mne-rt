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
  - name: Eric Larson
    orcid: 0000-0003-4782-5360
    affiliation: 4
  - name: Delphine Ribes
    orcid: 0000-0001-5527-1076
    affiliation: 5
  - name: Patrick Neff
    orcid: 0000-0003-3174-4910
    affiliation: 1
affiliations:
  - name: Department of Otorhinolaryngology, Head and Neck Surgery, University Hospital Zurich, University of Zurich, Zurich, Switzerland
    index: 1
    ror: 01462r250
  - name: Neuroscience Center Zurich, ETH Zurich and University of Zurich, Zurich, Switzerland
    index: 2
  - name: Fondation Campus Biotech Geneva (FCBG), Geneva, Switzerland
    index: 3
  - name: University of Washington, Seattle, WA, United States
    index: 4
    ror: 00cvxb145
  - name: EPFL+ECAL Lab, EPFL, Lausanne, Switzerland
    index: 5
    ror: 02s376052
date: 23 September 2026
bibliography: paper.bib
---

# Summary

In neurofeedback and brain-computer interfacing (BCI), acquisition, artifact
handling, feature extraction, decision and display must all complete within a
single analysis window, typically a few hundred milliseconds.

`MNE-RT` is a Python package that implements this processing chain on top of
`MNE-Python` [@gramfort2013meg] and `MNE-LSL` [@mnelsl]. A session is expressed
as a single object. `RTStream` connects to a Lab Streaming Layer (LSL) source
[@kothe2025lsl], records a resting-state baseline, and then runs a loop in which
each window is cleaned, reduced to one or more neural features, standardised
against that baseline, evaluated by an adaptive reward protocol, and published to
live displays, an LSL outlet and OSC receivers. Sessions are written to disk in a
BIDS-compatible layout [@pernet2019bids] together with per-window latency,
artifact-rate and signal-to-noise records, so that a session can be audited or
replayed offline through the same code path.

The package provides 20 feature modalities spanning sensor and source space, 10
adaptive protocols, 5 online artifact-correction methods, 9 live visualisation
windows, and an `mne-rt` command-line interface. Three hardware-free acquisition
paths (replay of any MNE-readable recording through a mock LSL stream, streaming
from an in-memory array, and the bundled `mne-rt demo`) allow the pipeline to be
run, taught and tested without an amplifier.

![The `MNE-RT` closed loop. Data arrive over LSL, are corrected in place, reduced
to features by a thread pool, standardised and combined, evaluated by an adaptive
protocol, and returned to the participant through displays, an LSL outlet or OSC.
\label{fig:arch}](figure1_architecture.png)

# Statement of need

Real-time M/EEG stacks are commonly assembled per study [@sitaram2017cln],
combining streaming, artifact rejection, feature extraction, adaptive
thresholding, stimulus communication and logging from separate components. The
resulting configuration is frequently not recorded with the data: the
thresholding rule, the achieved latency and the artifact method may exist only in
a local script.

`MNE-RT` is intended for researchers who already analyse M/EEG with `MNE-Python`
and who want the online experiment expressed with the same objects, with the
protocol, the artifact method and the measured timing stored alongside the
recorded session.

Sensor-space band power is supported by several existing real-time tools.
Source-space features are less well covered, largely for computational reasons.
Training a region-to-region interaction, for example imaginary coherency
[@nolte2004imcoh] between a pair of regions of interest reconstructed with an
LCMV beamformer [@vanveen1997lcmv], requires projecting each window into source
space and reducing thousands of grid points to a small number of ROI time
courses. Measured on a 5 mm whole-brain volume source space (14,629 points), the
direct `MNE-Python` implementation costs 650 to 800 ms per window against a
500 ms hop, so the loop does not close in real time. A volume source space also
permits subcortical labels to be specified, although the spatial resolution
attainable from a scalp montage limits the extent to which deep sources can be
separated from overlying cortex.

# State of the field

Within the MNE ecosystem, `mne-realtime` was the original real-time extension and
is no longer maintained. `MNE-LSL` [@mnelsl] is actively maintained and provides
modern LSL bindings, stream inlets and outlets, file replay and online epoching.
`MNE-RT` depends on it as its acquisition backbone rather than duplicating it.
`MNE-LSL` is scoped as a streaming library, whereas neurofeedback protocols,
source-space feature kernels, reward logic and session provenance are
application-level concerns outside that scope.

Outside the MNE ecosystem, `Timeflux` [@clisson2019timeflux] and NeuXus model
real-time processing as a configurable graph of nodes; `MEDUSA` [@medusa]
provides a broad BCI ecosystem with its own paradigms and signal-processing
stack; `OpenViBE` [@renard2010openvibe] and `BCI2000` [@schalk2004bci2000] are
mature C++ platforms driven through graphical designers; NeuroPype is
proprietary. Each of these requires the user to adopt a separate data model and
toolchain, and none provides an anatomically constrained source-space feature
path. `MNE-RT` occupies the remaining position: an MNE-native framework in which
the online feature is expressed with the same objects, montages, forward models
and inverse operators as the offline analysis that motivated it, and in which
source-space features are computed within the real-time budget.

# Software design

`MNE-RT` is script-shaped rather than graph-shaped. A session reads as
`connect_to_lsl` → `record_baseline` → `record_main` → `save`, following the
structure of an offline `MNE-Python` analysis rather than requiring the
experiment to be expressed as a dataflow configuration. The cost of this choice
is a large session class; the consequence is that an experiment is a readable,
version-controlled script.

Source-space features meet the deadline by exploiting the linearity of the
operators involved, rather than by accelerating the generic pipeline. LCMV with
`pick_ori="max-power"` and minimum-norm variants with `pick_ori="normal"` are
linear operators, and `mean`/`mean_flip` label extraction is a fixed sparse
average. `SourceModel.roi_kernel()` therefore collapses whitening, inverse
operator and label extraction into a single `(n_roi, n_channels)` matrix, applied
per window as one matrix product. This reduces the figure quoted above to
approximately 1 ms per window, and agrees with the `MNE-Python` result to better
than 1e-10 relative error, which the test suite asserts directly.
Free-orientation configurations are not linear, since taking the norm across
three orientations discards phase, and the kernel is refused in that case rather
than returning an incorrect value. The same precompute strategy applies to the
rest of the loop: bad-channel detection, ICA, noise covariance, forward solution
and the Maxwell/SSS basis [@taulu2006spatiotemporal] are all constructed during
the baseline phase, leaving a bounded per-window cost online. The head model is
built lazily, so sensor-space sessions do not download anatomical data.

Everything that varies between studies is a strategy object behind a small
interface. Protocols expose `evaluate(value) -> (crossed, magnitude)`, which
keeps z-score, percentile, staircase [@levitt1971transformed],
reinforcement-learning, operant-schedule, multi-band, cross-session-transfer and
double-blind sham protocols interchangeable; artifact correctors expose
`fit`/`transform`, covering adaptive LMS, ORICA, GEDAI, ASR [@mullen2015real] and
real-time Maxwell filtering; feature combiners include one that accepts any
scikit-learn estimator. Features themselves span band power and its derivatives
(ERD/ERS [@pfurtscheller1999event], laterality, individual peak frequency via
spectral parameterisation [@donoghue2020parameterizing]), connectivity,
cross-frequency coupling [@tort2010measuring], graph learning
[@kalofolias2016learn] and online decoding. A naming scheme (`base@label`) allows
one measure to run concurrently in several bands with independent state and
independent output channels.

Concurrency is explicit rather than framework-mediated. A daemon thread runs a
50%-overlap sliding window and fans modalities out across a thread pool, while
the Qt main thread refreshes displays at 30 fps, so that visualisation cannot
block acquisition. Window onsets are taken from the LSL clock and recorded as
`NaN` rather than substituted when unavailable, which keeps traces alignable to a
stimulus log. Interoperability is through standard protocols, an LSL outlet and
OSC, rather than through a bundled stimulus engine, so `MNE-RT` can drive
PsychoPy, Max/MSP or Unity without owning them.

# Research impact statement

`MNE-RT` has been developed in the open since May 2024, comprising more than 500
commits and over 30 merged pull requests, and is released on both PyPI and
conda-forge with versioned documentation, a tutorial series and an executable
example gallery. Over 950 automated tests across 38 modules run in continuous
integration on Python 3.11 to 3.13, including a job that resolves every
dependency to its declared lower bound.

Its feature-extraction layer was presented at IEEE CBMS 2025
[@shabestari2025advances]. Development has been supported by Swiss National
Science Foundation grant 208164, *Advancing Neurofeedback in Tinnitus*, and the
package is the online implementation vehicle for methods developed in that
programme, including the holistic graph-learning network analysis reported by
@shabestari2026shared. The source-space capability described above was developed
for an ongoing clinical collaboration on neurofeedback for aphasia, in which
region-to-region connectivity between language areas is trained in real time from
a 64-channel EEG stream; that study is the first external deployment, and
motivated the volume-source-space and connectivity work released in version 1.2.

# AI usage disclosure

Generative AI tools (Anthropic's Claude, used through Claude Code) assisted with
parts of the implementation, the test suite, the documentation and the drafting
of this manuscript. All AI-assisted output was reviewed, corrected and validated
by the authors, who take full responsibility for the correctness, originality and
licensing of the software and of this paper. Numerical claims in this paper were
verified against the package's own test suite and measurements.

# Acknowledgements

This work was supported by the Swiss National Science Foundation (grant 208164,
*Advancing Neurofeedback in Tinnitus*). We thank the `MNE-Python` and `MNE-LSL`
developer communities, on whose work this package depends.

# References
