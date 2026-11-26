"""Real-time visualisation components for MNE-RT.

Classes
-------
NFPlot
    Scrolling dark-themed multi-channel NF signal monitor (Qt + pyqtgraph).
RawPlot
    Scrolling dark-themed raw M/EEG channel viewer (Qt + pyqtgraph).
EpochPlot
    Scrolling raw viewer with epoch/trigger overlays (tmin/tmax window, event
    markers, shaded epoch regions).
BrainPlot
    Interactive 3D cortical surface with activity overlay (PyVista).
TopomapPlot
    Real-time scalp topomap showing per-band power distribution (matplotlib).
TopoPlot
    Live-updating scalp-layout ERP display — one mini-plot per electrode.
TFRPlot
    Real-time Morlet wavelet TFR heatmaps per channel and condition.
ButterflyPlot
    Real-time butterfly overlay: all channels per condition coloured by
    scalp region.
CompareEvoked
    Real-time per-channel condition comparison with SEM shading, peak
    markers, and a clickable scalp-topomap for interactive channel selection.
ConnectivityPlot
    The pairs a connectivity modality computes, live: edges on the head or
    between ROIs on a circle, and each pair's value over time.
"""

import importlib
import importlib.util

# Each viewer is imported when first used, as MNE does its submodules: most
# need Qt, and `import mne_rt` must work without a Qt binding installed.
_VIEWERS = {
    "BrainPlot": "brain_plot",
    "ButterflyPlot": "butterfly_plot",
    "CompareEvoked": "compare_evoked",
    "ConnectivityPlot": "connectivity_plot",
    "EpochPlot": "epoch_plot",
    "NFPlot": "nf_plot",
    "RawPlot": "raw_plot",
    "TFRPlot": "tfr_plot",
    "TopoPlot": "topo_plot",
    "TopomapPlot": "topomap_plot",
}

_QT_BINDINGS = ("PyQt6", "PySide6", "PyQt5", "PySide2")


def _has_qt() -> bool:
    """Whether a Qt binding is installed."""
    return any(importlib.util.find_spec(b) for b in _QT_BINDINGS)


def require_qt(what: str) -> None:
    """Raise an ImportError saying how to install a Qt binding, when none is.

    For each place that opens windows, before it imports Qt: pyqtgraph's and
    qtpy's own errors do not say how to install one for mne-rt.
    """
    if not _has_qt():
        raise ImportError(
            f"{what} needs a Qt binding, and none is installed. Install one with: "
            "pip install 'mne-rt[pyqt6]' (or 'mne-rt[pyside6]')."
        )


def __getattr__(name: str):
    if name not in _VIEWERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f".{_VIEWERS[name]}", __name__)
    except ImportError as exc:
        if not _has_qt():
            try:
                require_qt(f"mne_rt.viz.{name}")
            except ImportError as hint:
                raise hint from exc
        raise
    value = getattr(module, name)
    globals()[name] = value  # once imported, a plain attribute
    return value


def __dir__():
    return sorted(set(globals()) | set(_VIEWERS))


__all__ = [
    "NFPlot",
    "RawPlot",
    "EpochPlot",
    "BrainPlot",
    "TopomapPlot",
    "TopoPlot",
    "TFRPlot",
    "ButterflyPlot",
    "CompareEvoked",
    "ConnectivityPlot",
]
