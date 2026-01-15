"""
Deep Learning - CNN & Hybrid Architectures (6)
31. DLinear
32. N-BEATS
33. N-HiTS
34. TCN
35. TiDE
36. TimesNet
"""

from .dlinear import DLinearModel
from .nbeats import NBEATSModel
from .nhits import NHiTSModel
from .tcn import TCNModel
from .tide import TiDEModel
from .timesnet import TimesNetModel

__all__ = [
    "DLinearModel",
    "NBEATSModel",
    "NHiTSModel",
    "TCNModel",
    "TiDEModel",
    "TimesNetModel",
]
