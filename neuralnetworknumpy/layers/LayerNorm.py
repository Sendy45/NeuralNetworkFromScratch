from neuralnetworknumpy.backend import np
from .Layer import Layer

class LayerNorm(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X, training=None):
        pass

    def backward(self, dA):
        pass

    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
        pass