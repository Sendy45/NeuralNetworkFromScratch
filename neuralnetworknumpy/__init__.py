"""
neuralnetworknumpy
A minimal deep learning framework built using NumPy.
"""

__version__ = "0.2.0"

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
    Conv2D,
    Flatten,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    DepthwiseConv2D,
    DepthwiseSeparableConv2D,
    SpatiallySeparableConv2D,
    ResidualBlock,
    BatchNorm2D,
    GroupConv2D,
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

    # 1D Layers
    "Dense",
    "Activation",
    "ReLu",
    "Sigmoid",
    "Softmax",
    "Linear",
    "Tanh",
    "BatchNorm",
    "Dropout",

    # 2D Layers
    "Conv2D",
    "Flatten",
    "MaxPooling2D",
    "AveragePooling2D",
    "GlobalAveragePooling2D",
    "DepthwiseConv2D",
    "DepthwiseSeparableConv2D",
    "SpatiallySeparableConv2D",
    "ResidualBlock",
    "BatchNorm2D",
    "GroupConv2D",

    # Utilities
    "History",
    "Scaler",
    "split_train_test",
    "split_train_validation",
]