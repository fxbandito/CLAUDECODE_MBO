"""
Deep Learning - Graph & Specialized Neural Networks (12)
45. Diffusion
46. KAN
47. MTGNN
48. Neural ARIMA
49. Neural Basis Functions
50. Neural GAM
51. Neural ODE
52. Neural Quantile Regression
53. Neural VAR
54. Neural Volatility
55. Spiking Neural Networks
56. StemGNN
"""

from .diffusion import DiffusionModel
from .kan import KANModel
from .mtgnn import MTGNNModel
from .neural_arima import NeuralARIMAModel
from .neural_basis_functions import NeuralBasisFunctionsModel
from .neural_gam import NeuralGAMModel
from .neural_ode import NeuralODEModel
from .neural_quantile_regression import NeuralQuantileRegressionModel
from .neural_var import NeuralVARModel
from .neural_volatility import NeuralVolatilityModel
from .spiking_neural_networks import SpikingNeuralNetworksModel
from .stemgnn import StemGNNModel

__all__ = [
    "DiffusionModel",
    "KANModel",
    "MTGNNModel",
    "NeuralARIMAModel",
    "NeuralBasisFunctionsModel",
    "NeuralGAMModel",
    "NeuralODEModel",
    "NeuralQuantileRegressionModel",
    "NeuralVARModel",
    "NeuralVolatilityModel",
    "SpikingNeuralNetworksModel",
    "StemGNNModel",
]
