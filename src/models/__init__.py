"""
MBO Trading Strategy Analyzer - Model Registry
Dinamikus model felfedezés és regisztráció.

Minden modell a saját fájljában tárolja:
- MODEL_INFO: név, kategória, GPU/batch támogatás
- PARAM_DEFAULTS: alapértelmezett paraméterek
- PARAM_OPTIONS: dropdown opciók

A GUI innen kérdezi le az összes információt dinamikusan.
"""

import importlib
import inspect
import os
from typing import Dict, List, Optional, Type

from models.base import BaseModel, ModelInfo


# Fix kategória sorrend (nem ABC rendezés!)
CATEGORY_ORDER = [
    "Statistical Models",
    "Smoothing & Decomposition",
    "Classical Machine Learning",
    "Deep Learning - RNN-based",
    "Deep Learning - CNN & Hybrid Architectures",
    "Deep Learning - Transformer-based",
    "Deep Learning - Graph & Specialized Neural Networks",
    "Meta-Learning & AutoML",
    "Bayesian & Probabilistic Methods",
    "Frequency Domain & Signal Processing",
    "Distance & Similarity-based",
    "State Space & Other",
    "Symbolic Regression",
]

# Fix modell sorrend kategóriánként (nem ABC rendezés!)
MODEL_ORDER = {
    "Statistical Models": [
        "ADIDA", "ARIMA", "ARIMAX", "Auto-ARIMA", "CES",
        "Change Point Detection", "GAM", "GARCH Family", "OGARCH",
        "Quantile Regression", "SARIMA", "VAR", "VECM"
    ],
    "Smoothing & Decomposition": [
        "ETS", "Exponential Smoothing", "MSTL", "STL", "Theta"
    ],
    "Classical Machine Learning": [
        "Gradient Boosting", "KNN Regressor", "LightGBM",
        "Random Forest", "SVR", "XGBoost"
    ],
    "Deep Learning - RNN-based": [
        "DeepAR", "ES-RNN", "GRU", "LSTM", "MQRNN", "Seq2Seq"
    ],
    "Deep Learning - CNN & Hybrid Architectures": [
        "DLinear", "N-BEATS", "N-HiTS", "TCN", "TiDE", "TimesNet"
    ],
    "Deep Learning - Transformer-based": [
        "Autoformer", "FEDFormer", "FiTS", "Informer",
        "PatchTST", "TFT", "Transformer", "iTransformer"
    ],
    "Deep Learning - Graph & Specialized Neural Networks": [
        "Diffusion", "KAN", "MTGNN", "Neural ARIMA", "Neural Basis Functions",
        "Neural GAM", "Neural ODE", "Neural Quantile Regression",
        "Neural VAR", "Neural Volatility", "Spiking Neural Networks", "StemGNN"
    ],
    "Meta-Learning & AutoML": [
        "DARTS", "FFORMA", "GFM", "Meta-learning", "MoE",
        "Multi-task Learning", "NAS"
    ],
    "Bayesian & Probabilistic Methods": [
        "BSTS", "Conformal Prediction", "Gaussian Process",
        "Monte Carlo", "Prophet"
    ],
    "Frequency Domain & Signal Processing": [
        "DFT", "FFT", "Periodogram", "Spectral Analysis",
        "SSA", "Wavelet Analysis", "Welchs Method"
    ],
    "Distance & Similarity-based": [
        "DTW", "k-NN", "k-Shape", "Matrix Profile"
    ],
    "State Space & Other": [
        "Kalman Filter", "State Space Model", "TDA", "Time Series Ensemble"
    ],
    "Symbolic Regression": [
        "GPLearn", "PySR", "PySindy"
    ],
}


class ModelRegistry:
    """
    Dinamikus model registry - felderíti és regisztrálja az összes modellt.

    Használat:
        registry = ModelRegistry()

        # Kategóriák lekérdezése
        categories = registry.get_categories()

        # Modellek egy kategóriában
        models = registry.get_models_in_category("Statistical Models")

        # Model osztály lekérése
        model_class = registry.get_model_class("ARIMA")

        # Paraméterek
        defaults = registry.get_param_defaults("ARIMA")
        options = registry.get_param_options("ARIMA")
    """

    _instance: Optional['ModelRegistry'] = None
    _initialized: bool = False

    def __new__(cls) -> 'ModelRegistry':
        """Singleton pattern - csak egy instance létezhet."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Inicializálás - csak egyszer fut le."""
        if ModelRegistry._initialized:
            return

        self._models: Dict[str, Type[BaseModel]] = {}
        self._categories: Dict[str, List[str]] = {}
        self._discover_models()
        ModelRegistry._initialized = True

    def _discover_models(self):
        """Felderíti az összes modellt a models/ mappából."""
        models_dir = os.path.dirname(__file__)

        # Végigmegyünk az almappákon
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)

            # Csak mappákat nézünk, __pycache__-t kihagyjuk
            if not os.path.isdir(subdir_path):
                continue
            if subdir.startswith('_') or subdir == '__pycache__':
                continue

            # Végigmegyünk a .py fájlokon
            for filename in os.listdir(subdir_path):
                if not filename.endswith('.py'):
                    continue
                if filename.startswith('_'):
                    continue

                module_name = filename[:-3]  # .py levágása
                full_module = f"models.{subdir}.{module_name}"

                try:
                    self._import_and_register(full_module)
                except Exception as e:
                    # Hiba esetén folytatjuk a többi modellel
                    print(f"Warning: Could not load {full_module}: {e}")

    def _import_and_register(self, module_name: str):
        """Importál egy modult és regisztrálja a BaseModel leszármazottakat."""
        module = importlib.import_module(module_name)

        # Megkeressük a BaseModel leszármazottakat
        for _, obj in inspect.getmembers(module, inspect.isclass):
            # Csak BaseModel leszármazottak, de nem maga a BaseModel
            if not issubclass(obj, BaseModel):
                continue
            if obj is BaseModel:
                continue

            # Ellenőrizzük, hogy van-e MODEL_INFO
            if not hasattr(obj, 'MODEL_INFO') or obj.MODEL_INFO is None:
                continue

            model_info = obj.MODEL_INFO
            model_name = model_info.name
            category = model_info.category

            # Regisztráljuk a modellt
            self._models[model_name] = obj

            # Kategóriához hozzáadjuk
            if category not in self._categories:
                self._categories[category] = []
            if model_name not in self._categories[category]:
                self._categories[category].append(model_name)

    def get_categories(self) -> List[str]:
        """Visszaadja az összes kategória nevét fix sorrendben."""
        # Fix sorrend használata - csak a ténylegesen létező kategóriák
        result = []
        for cat in CATEGORY_ORDER:
            if cat in self._categories:
                result.append(cat)
        # Ha van olyan kategória ami nincs a fix listában, azt a végére tesszük
        for cat in self._categories:
            if cat not in result:
                result.append(cat)
        return result

    def get_models_in_category(self, category: str) -> List[str]:
        """Visszaadja a kategóriában lévő model neveket fix sorrendben."""
        models = self._categories.get(category, [])
        if not models:
            return []

        # Ha van fix sorrend ehhez a kategóriához, azt használjuk
        if category in MODEL_ORDER:
            ordered = []
            for model_name in MODEL_ORDER[category]:
                if model_name in models:
                    ordered.append(model_name)
            # Ha van olyan modell ami nincs a fix listában, azt a végére tesszük
            for model_name in models:
                if model_name not in ordered:
                    ordered.append(model_name)
            return ordered

        return models

    def get_all_models(self) -> List[str]:
        """Visszaadja az összes model nevét (rendezve)."""
        return sorted(self._models.keys())

    def get_model_class(self, model_name: str) -> Optional[Type[BaseModel]]:
        """Visszaadja a model osztályt név alapján."""
        return self._models.get(model_name)

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Visszaadja a model info-t."""
        model_class = self.get_model_class(model_name)
        if model_class:
            return model_class.MODEL_INFO
        return None

    def get_param_defaults(self, model_name: str) -> Dict[str, str]:
        """Visszaadja a model alapértelmezett paramétereit."""
        model_class = self.get_model_class(model_name)
        if model_class:
            return model_class.PARAM_DEFAULTS.copy()
        return {}

    def get_param_options(self, model_name: str) -> Dict[str, List[str]]:
        """Visszaadja a model paraméter opcióit (dropdown-okhoz)."""
        model_class = self.get_model_class(model_name)
        if model_class:
            return model_class.PARAM_OPTIONS.copy()
        return {}

    def supports_gpu(self, model_name: str) -> bool:
        """Visszaadja, hogy a model támogatja-e a GPU-t."""
        info = self.get_model_info(model_name)
        return info.supports_gpu if info else False

    def supports_batch(self, model_name: str) -> bool:
        """Visszaadja, hogy a model támogatja-e a batch módot."""
        info = self.get_model_info(model_name)
        return info.supports_batch if info else False

    def get_category_for_model(self, model_name: str) -> Optional[str]:
        """Visszaadja, hogy melyik kategóriában van a model."""
        info = self.get_model_info(model_name)
        return info.category if info else None

    def get_model_count(self) -> int:
        """Visszaadja a regisztrált modellek számát."""
        return len(self._models)

    def get_category_count(self) -> int:
        """Visszaadja a kategóriák számát."""
        return len(self._categories)

    def reload(self):
        """Újratölti az összes modellt (fejlesztéshez hasznos)."""
        self._models.clear()
        self._categories.clear()
        self._discover_models()


# Singleton instance - könnyű hozzáférés
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Visszaadja a global model registry instance-t."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


# Convenience functions - rövidebb hívásokhoz
def get_categories() -> List[str]:
    """Visszaadja az összes kategóriát."""
    return get_registry().get_categories()


def get_models_in_category(category: str) -> List[str]:
    """Visszaadja a kategóriában lévő modelleket."""
    return get_registry().get_models_in_category(category)


def get_all_models() -> List[str]:
    """Visszaadja az összes modellt."""
    return get_registry().get_all_models()


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Visszaadja a model info-t."""
    return get_registry().get_model_info(model_name)


def get_param_defaults(model_name: str) -> Dict[str, str]:
    """Visszaadja az alapértelmezett paramétereket."""
    return get_registry().get_param_defaults(model_name)


def get_param_options(model_name: str) -> Dict[str, List[str]]:
    """Visszaadja a paraméter opciókat."""
    return get_registry().get_param_options(model_name)


def supports_gpu(model_name: str) -> bool:
    """Támogatja-e a GPU-t."""
    return get_registry().supports_gpu(model_name)


def supports_batch(model_name: str) -> bool:
    """Támogatja-e a batch módot."""
    return get_registry().supports_batch(model_name)
