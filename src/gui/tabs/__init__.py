"""
GUI Tab modulok - MBO Trading Strategy Analyzer
Minden tab külön Mixin osztályként van implementálva.
"""

from .data_loading import DataLoadingMixin
from .analysis import AnalysisMixin
from .results import ResultsMixin

__all__ = [
    "DataLoadingMixin",
    "AnalysisMixin",
    "ResultsMixin",
]
