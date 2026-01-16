"""
Analysis module for MBO Trading Strategy Analyzer.
"""
from .inspection import InspectionEngine
from .metrics import FinancialMetrics
from .engine import (
    AnalysisEngine,
    AnalysisContext,
    AnalysisResult,
    AnalysisProgress,
    analyze_strategy,
    ResourceManager,
    get_resource_manager,
    init_resource_manager,
)
from .dual_executor import (
    run_dual_model_mode,
    detect_data_mode,
    get_dual_mode_models,
    DUAL_MODE_SUPPORTED_MODELS,  # Legacy alias
)
from .dual_task import (
    train_dual_model_task,
    apply_recursive_forecasting,
    init_worker_environment,
)

__all__ = [
    # Inspection
    "InspectionEngine",
    # Metrics
    "FinancialMetrics",
    # Analysis Engine
    "AnalysisEngine",
    "AnalysisContext",
    "AnalysisResult",
    "AnalysisProgress",
    "analyze_strategy",
    # Resource Manager
    "ResourceManager",
    "get_resource_manager",
    "init_resource_manager",
    # Dual Model Mode
    "run_dual_model_mode",
    "detect_data_mode",
    "get_dual_mode_models",
    "DUAL_MODE_SUPPORTED_MODELS",  # Legacy
    # Dual Task (worker infrastructure)
    "train_dual_model_task",
    "apply_recursive_forecasting",
    "init_worker_environment",
]
