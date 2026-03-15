from .Layer import Layer
import numpy as np


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, A_prev, training=None):
        self.input_shape = A_prev.shape
        return A_prev.reshape(A_prev.shape[0], -1).T

    def _backward(self, dA, skip_activation=False):
        return dA.T.reshape(self.input_shape)

    def _update(self, *args, **kwargs):
        pass
