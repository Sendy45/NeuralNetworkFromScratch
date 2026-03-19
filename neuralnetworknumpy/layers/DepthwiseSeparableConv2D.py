from .DepthwiseConv2D import DepthwiseConv2D
from .Layer import Layer
from .Conv2D import Conv2D

class DepthwiseSeparableConv2D(Layer):
    """
        Factorized convolution: depthwise + pointwise.

        Step 1: DepthwiseConv2D (spatial filtering per channel)
        Step 2: 1×1 Conv2D (channel mixing)

        Reduces computation compared to standard Conv2D.
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="valid"):
        super().__init__()
        self.depthwise = DepthwiseConv2D(kernel_size, strides=strides, padding=padding)
        self.pointwise = Conv2D(filters, 1)


    def _forward(self, A_prev, training=None):
        self.A_prev = A_prev
        x = self.depthwise._forward(A_prev, training=training)
        return self.pointwise._forward(x, training=training)

    def _backward(self, dZ, skip_activation=False):

        dA = self.pointwise._backward(dZ)
        return self.depthwise._backward(dA)


    def _update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
        self.depthwise._update(lambda_, lr, beta1, beta2, _eps, optimizer, t)
        self.pointwise._update(lambda_, lr, beta1, beta2, _eps, optimizer, t)