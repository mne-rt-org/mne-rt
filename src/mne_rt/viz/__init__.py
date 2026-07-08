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
"""

from .brain_plot import BrainPlot
from .butterfly_plot import ButterflyPlot
from .compare_evoked import CompareEvoked
from .epoch_plot import EpochPlot
from .nf_plot import NFPlot
from .raw_plot import RawPlot
from .tfr_plot import TFRPlot
from .topo_plot import TopoPlot
from .topomap_plot import TopomapPlot

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
]
