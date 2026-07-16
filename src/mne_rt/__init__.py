"""MNE-RT — real-time M/EEG signal processing.

MNE-RT is an open-source Python package for **real-time processing and
visualisation of M/EEG data**.

Main entry points
-----------------
:class:`~mne_rt.RTStream`
    Real-time M/EEG session controller — LSL streaming, artifact correction,
    feature extraction, and visualisation.
:class:`~mne_rt.RTEpochs`
    Event-triggered epoch accumulator with per-trial feature statistics.
:class:`~mne_rt.viz.NFPlot`
    Scrolling multi-channel real-time NF signal monitor (Qt + pyqtgraph).
:class:`~mne_rt.viz.RawPlot`
    Scrolling raw M/EEG channel viewer (Qt + pyqtgraph).
:class:`~mne_rt.viz.TopoPlot`
    Live-updating scalp-layout ERP display — one mini-plot per electrode,
    ±SEM shading, re-referencing, and unit-aware amplitude display.
:class:`~mne_rt.viz.ButterflyPlot`
    All channels overlaid per condition, coloured by scalp region.
:class:`~mne_rt.viz.TFRPlot`
    Real-time Morlet wavelet time-frequency representation per channel.
:class:`~mne_rt.viz.CompareEvoked`
    Large per-channel ERP comparison with ±SEM ribbons, peak markers, and
    a clickable scalp-topomap in the sidebar for interactive channel selection.
:class:`~mne_rt.viz.BrainPlot`
    Interactive 3D cortical surface with activity overlay (PyVista).
:class:`~mne_rt.tools.ORICA`
    Online Recursive ICA for real-time artifact removal.
:class:`~mne_rt.tools.GEDAIDenoiser`
    GED-based spatial filter for artifact identification and removal.

Verbosity
---------
All public methods accept a ``verbose`` keyword argument following
MNE's convention.  You can also set the package-wide level::

    import mne_rt
    mne_rt.set_log_level("INFO")   # or True / False / "DEBUG" / "WARNING"
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mne-rt")
except PackageNotFoundError:
    __version__ = "0.0.0"

from mne_rt._logging import logger, set_log_level  # noqa: F401 — public API
from mne_rt.combiners import (  # noqa: F401 — public API
    FeatureCombiner,
    GeometricMeanCombiner,
    LearnedCombiner,
    WeightedSumCombiner,
    ZScoredNormCombiner,
)
from mne_rt.lsl_output import LSLSender
from mne_rt.osc import OSCSender
from mne_rt.protocols import (
    LinearTrendProtocol,
    PercentileProtocol,
    ThresholdProtocol,
    ZScoreProtocol,
)
from mne_rt.rt_epochs import RTEpochs
from mne_rt.rt_stream import ArrayStream, RTStream
from mne_rt.tools import ORICA, GEDAIDenoiser
from mne_rt.tools.asr import ASRDenoiser
from mne_rt.tools.bad_channel_detector import BadChannelDetector
from mne_rt.tools.lms import AdaptiveLMSFilter
from mne_rt.tools.maxwell import RTMaxwellFilter
from mne_rt.viz import (
    BrainPlot,
    ButterflyPlot,
    CompareEvoked,
    EpochPlot,
    NFPlot,
    RawPlot,
    TFRPlot,
    TopomapPlot,
    TopoPlot,
)

__all__ = [
    "RTStream",
    "ArrayStream",
    "RTEpochs",
    "BrainPlot",
    "NFPlot",
    "RawPlot",
    "EpochPlot",
    "TopomapPlot",
    "TopoPlot",
    "ButterflyPlot",
    "TFRPlot",
    "CompareEvoked",
    "ORICA",
    "GEDAIDenoiser",
    "AdaptiveLMSFilter",
    "ASRDenoiser",
    "RTMaxwellFilter",
    "ThresholdProtocol",
    "ZScoreProtocol",
    "PercentileProtocol",
    "LinearTrendProtocol",
    "BadChannelDetector",
    "OSCSender",
    "LSLSender",
    "FeatureCombiner",
    "WeightedSumCombiner",
    "GeometricMeanCombiner",
    "ZScoredNormCombiner",
    "LearnedCombiner",
    "set_log_level",
    "logger",
    "__version__",
]
