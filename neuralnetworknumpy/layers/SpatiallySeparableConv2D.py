from .Layer import Layer
from .Conv2D import Conv2D

class SpatiallySeparableConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="valid"):
        super().__init__()

        if isinstance(kernel_size, tuple):
            k = kernel_size[0]
        else:
            k = kernel_size

        self.filter1 = Conv2D(filters, (k, 1), strides=strides, padding=padding)
        self.filter2 = Conv2D(filters, (1, k), strides=(1, 1), padding=padding)

    def _forward(self, A_prev, training=None):
        x = self.filter1._forward(A_prev, training=training)
        return self.filter2._forward(x, training=training)

    def _backward(self, dZ, skip_activation=False):
        dA = self.filter2._backward(dZ)
        return self.filter1._backward(dA)

    def _update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
        self.filter1._update(lambda_, lr, beta1, beta2, _eps, optimizer, t)
        self.filter2._update(lambda_, lr, beta1, beta2, _eps, optimizer, t)