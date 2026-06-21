<p align="center">
  <img src="https://raw.githubusercontent.com/payamsash/mne-rt/main/docs/source/_static/mne_rt_logo.svg" alt="MNE-RT Logo" width="480"/>
</p>

<p align="center">
  <strong>MNE-RT — Real-time M/EEG Signal Processing</strong><br>
  From amplifier to 3-D brain display in a single, researcher-friendly API
</p>

<p align="center">
  <a href="https://github.com/payamsash/mne-rt/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/payamsash/mne-rt?color=green"></a>
  <a href="https://pypi.org/project/mne-rt/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mne-rt?color=blue"></a>
  <a href="https://pypi.org/project/mne-rt/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mne-rt"></a>
  <a href="https://payamsash.github.io/mne-rt/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-brightgreen?logo=readthedocs&logoColor=white"></a>
</p>

---

**MNE-RT** is an open-source Python library for **real-time M/EEG signal
processing**, built on [MNE-Python](https://mne.tools) and
[MNE-LSL](https://mne.tools/mne-lsl). It covers the full closed-loop pipeline —
from amplifier to 3-D brain display — in a single API designed for neurofeedback,
BCI, and clinical or basic-science monitoring.

## Ecosystem

Several MNE-affiliated packages address real-time M/EEG at different levels of
abstraction. Here is how they relate:

**[mne-realtime](https://github.com/mne-tools/mne-realtime)** was the original
real-time extension for [MNE-Python](https://mne.tools). It offered TCP/IP client
classes for a handful of acquisition systems (FieldTrip buffer, RtClient,
StimServer) and a basic `RtEpochs` object for online epoching and classification.
The package is **no longer maintained** and should not be used for new projects.

**[mne-lsl](https://mne.tools/mne-lsl)** is an actively maintained, low-to-mid-level
streaming library. It provides modern Python bindings for the
[Lab Streaming Layer (LSL)](https://github.com/sccn/labstreaminglayer) C++ library 
(replacement for the older [pylsl](https://github.com/labstreaminglayer/pylsl))
— together with high-level objects: `StreamInlet` and `StreamOutlet` for reading
and writing LSL streams, `StreamPlayer` for replaying recorded files without
hardware, `StreamRecorder` for saving streams to disk, and `EpochStream` for
real-time epoching with online filtering. [MNE-RT](https://payamsash.github.io/mne-rt/) 
uses [MNE-LSL](https://mne.tools/mne-lsl) as its data-acquisition backbone.

**[MNE-RT](https://payamsash.github.io/mne-rt/)** (this package) is an actively
maintained, high-level neurofeedback and BCI application framework built on top of
[MNE-Python](https://mne.tools) and [MNE-LSL](https://mne.tools/mne-lsl). It adds
the full closed-loop pipeline that neither of the above provides: **neural
feature extraction modalities** spanning sensor and source space (band power,
ERD/ERS, laterality index, Hjorth parameters, spectral centroid, cross-frequency
coupling, functional connectivity, graph Laplacian); **adaptive feedback
protocols** (z-score, threshold, percentile, staircase, operant conditioning,
reinforcement learning, sham, multi-band, and cross-session transfer); **online
artifact correction methods** (ASR, adaptive LMS, GEDAI, ORICA, real-time
Maxwell/SSS for MEG); and **live visualisation windows** (scrolling raw signal,
neurofeedback signal, epoch overlays, scalp topography, interactive 3-D brain
surface, butterfly plot, ERP comparison, and time-frequency heatmaps). It also
handles feature combining, external feedback output via OSC and LSL outlets,
BIDS-compatible session saving, and a full CLI.

## Highlights

| Feature | Details |
|---|---|
| **20 NF modalities** | Band power, ERD/ERS, laterality, Hjorth, spectral centroid, CFC, connectivity, graph Laplacian — sensor and source space |
| **10 adaptive protocols** | Z-score, threshold, percentile, staircase, operant, RL, sham, multi-band, and cross-session transfer |
| **5 artifact correction methods** | ASR, adaptive LMS, GEDAI, ORICA, real-time Maxwell/SSS (MEG) |
| **9 live visualisation windows** | Scrolling raw · NF signal · epoch overlays · scalp topo · 3-D brain · butterfly · ERP comparison · TFR heatmaps |
| **Feature combiners** | Weighted sum, geometric mean, z-scored norm, or any sklearn estimator |
| **External feedback output** | OSC (Max/MSP, SuperCollider) and LSL outlet (PsychoPy, OpenViBE, BCI2000) |
| **BIDS-compatible saving** | Session JSON + TSV with full metadata, artifact rate, and SNR |
| **CLI** | `mne-rt info` · `mne-rt demo` · `mne-rt baseline` · `mne-rt run` |
| **Mock mode** | Full pipeline without hardware via built-in LSL replay |

## Installation

```bash
pip install mne-rt                 # core package
pip install "mne-rt[full]"         # + 3-D viz, dev tools, docs
```

<details>
<summary>Other installation methods</summary>

**uv (fast Rust-based installer):**
```bash
uv pip install mne-rt
uv pip install "mne-rt[full]"
```

**Development install from source:**
```bash
git clone https://github.com/payamsash/mne-rt.git
cd mne-rt
pip install -e ".[dev]"
```

</details>

Verify:
```bash
mne-rt info     # prints MNE-RT and dependency versions
mne-rt demo     # runs a 60-second mock neurofeedback session
```

## Quick start

```python
from mne_rt import RTStream
from mne_rt.protocols import ZScoreProtocol

# 1 — Create a session object
nf = RTStream(
    subject_id="sub01",
    session="01",
    subjects_dir="/data/subjects",
    montage="easycap-M1",
)

# 2 — Connect to a live LSL stream (or replay a file without hardware)
nf.connect_to_lsl(mock_lsl=True)

# 3 — Record a resting-state baseline (bad channels, ICA, noise cov)
nf.record_baseline(duration=120)

# 4 — Run a closed-loop NF session
nf.record_main(
    duration=300,
    modality=["sensor_power", "erd_ers"],
    protocol=ZScoreProtocol(direction="up", zscore_threshold=0.5),
    show_nf_signal=True,
    show_topo=True,
)

# 5 — Save results (BIDS-compatible JSON + TSV)
nf.save()
```

## CLI

```bash
# Print version and all dependency versions
mne-rt info

# Quick demo — no amplifier needed
mne-rt demo --duration 60 --modality sensor_power erd_ers

# Record a resting-state baseline
mne-rt baseline --subject sub01 --subjects-dir /data --session 01

# Run a full session with artifact correction and live displays
mne-rt run --subject sub01 --subjects-dir /data --duration 600 \
           --modality sensor_power erd_ers \
           --artifact-correction asr \
           --topo --brain
```

## Documentation

Full documentation (API reference, tutorials, visualization gallery) is available at
**[payamsash.github.io/mne-rt](https://payamsash.github.io/mne-rt/)**.

## Cite

If you use MNE-RT in your research, please cite:

```bibtex
@inproceedings{shabestari2025advances,
  title        = {Advances on Real Time {M/EEG} Neural Feature Extraction},
  author       = {Shabestari, Payam S and Ribes, Delphine and D{\'e}fayes, Lara
                  and Cai, Danpeng and Groves, Emily and Behjat, Harry H
                  and Van de Ville, Dimitri and Kleinjung, Tobias
                  and Naas, Adrian and Henchoz, Nicolas and others},
  booktitle    = {2025 IEEE 38th International Symposium on Computer-Based
                  Medical Systems (CBMS)},
  pages        = {337--338},
  year         = {2025},
  organization = {IEEE}
}
```

## Acknowledgements

Development was supported by the
[Swiss National Science Foundation](https://www.snf.ch/en) (grant 208164 —
*Advancing Neurofeedback in Tinnitus*).

## License

[MIT License](LICENSE) — © 2025 Payam S. Shabestari
