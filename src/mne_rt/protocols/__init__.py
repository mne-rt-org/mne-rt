from .linear_trend import LinearTrendProtocol
from .multiband import MultiBandProtocol
from .operant import OperantProtocol
from .percentile import PercentileProtocol
from .rl_protocol import RLProtocol
from .sham import ShamProtocol
from .staircase import UpDownStaircaseProtocol
from .threshold import ThresholdProtocol
from .transfer import TransferProtocol
from .zscores import ZScoreProtocol

__all__ = [
    "ThresholdProtocol",
    "ZScoreProtocol",
    "PercentileProtocol",
    "LinearTrendProtocol",
    "ShamProtocol",
    "UpDownStaircaseProtocol",
    "MultiBandProtocol",
    "RLProtocol",
    "OperantProtocol",
    "TransferProtocol",
]

# So Sphinx registers mne_rt.protocols.X, not mne_rt.protocols.zscores.X,
# and cross-references to them resolve.
for _cls in [
    ThresholdProtocol,
    ZScoreProtocol,
    PercentileProtocol,
    LinearTrendProtocol,
    ShamProtocol,
    UpDownStaircaseProtocol,
    MultiBandProtocol,
    RLProtocol,
    OperantProtocol,
    TransferProtocol,
]:
    _cls.__module__ = "mne_rt.protocols"
del _cls
