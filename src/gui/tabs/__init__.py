"""
GUI Tab modulok - MBO Trading Strategy Analyzer
Minden tab külön Mixin osztályként van implementálva.
"""

from gui.tabs.data_loading import DataLoadingMixin
from gui.tabs.analysis import AnalysisMixin
from gui.tabs.results import ResultsMixin

__all__ = [
    "DataLoadingMixin",
    "AnalysisMixin",
    "ResultsMixin",
]
