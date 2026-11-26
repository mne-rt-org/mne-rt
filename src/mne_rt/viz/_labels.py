"""Display names and units of the modalities, for the NF plot and the report.

Here rather than in :mod:`~mne_rt.viz.nf_plot`, which needs Qt: the session
report uses them too, and must work without a Qt binding.
"""

from __future__ import annotations

from mne_rt._naming import split_modality

_LABELS = {
    "sensor_power": "Sensor Power",
    "band_ratio": "Band Ratio",
    "source_power": "Source Power",
    "sensor_connectivity": "Sensor Connectivity",
    "source_connectivity": "Source Connectivity",
    "sensor_graph": "Sensor Graph",
    "source_graph": "Source Graph",
    "entropy": "Entropy",
    "argmax_freq": "Peak Frequency",
    "individual_peak_power": "Peak Power",
    "cfc_sensor": "Sensor CFC",
    "erd_ers": "ERD/ERS",
    "laterality": "Laterality",
    "hjorth": "Hjorth",
    "spectral_centroid": "Spectral Centroid",
    "decode": "Decoder Probability",
}

_UNITS = {
    "sensor_power": "V²/Hz",
    "band_ratio": "",
    "source_power": "a.u.",
    "sensor_connectivity": "",
    "source_connectivity": "",
    "sensor_graph": "",
    "source_graph": "",
    "entropy": "",
    "argmax_freq": "Hz",
    "individual_peak_power": "V²/Hz",
    "cfc_sensor": "",
    "erd_ers": "%",
    "laterality": "",
    "hjorth": "",
    "spectral_centroid": "Hz",
    "decode": "",
}


def _label_for(mod: str) -> str:
    """Axis label for a modality, with its instance label in parentheses."""
    base, label = split_modality(mod)
    text = _LABELS.get(base, base)
    return f"{text} ({label})" if label else text


def _unit_for(mod: str) -> str:
    """Unit string for a modality; an instance inherits its base's unit."""
    return _UNITS.get(split_modality(mod)[0], "")
