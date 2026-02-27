"""
NeuralNetworkFromScratch
A minimal deep learning framework built using NumPy.
"""

__version__ = "0.1.0"

# Layers
from .layers import (
    Layer,
    Dense,
    Activation,
    ReLu,
    Sigmoid,
    Softmax,
    Linear,
    Tanh,
    BatchNorm,
    Dropout,
)

# Model
from .model import NeuralNetwork

# Utils
from .utils import (
    History,
    Scaler,
    split_train_test,
    split_train_validation,
)

__all__ = [
    # Core
    "NeuralNetwork",

    # Base
    "Layer",

    # Layers
    "Dense",
    "Activation",
    "ReLu",
    "Sigmoid",
    "Softmax",
    "Linear",
    "Tanh",
    "BatchNorm",
    "Dropout",

    # Utilities
    "History",
    "Scaler",
    "split_train_test",
    "split_train_validation",
]