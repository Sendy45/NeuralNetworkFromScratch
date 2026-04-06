from .Layer import Layer
from .Conv2D import Conv2D

class SpatiallySeparableConv2D(Layer):
    """
        Factorized convolution: spatial separation.

        Replaces K×K convolution with:
        - K×1 convolution
        - 1×K convolution

        Reduces computation while approximating full spatial convolution.

        Not recommended to use spatially separable convolution.
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="valid"):
        super().__init__()

        # Handle non tuple kernel size
        if isinstance(kernel_size, tuple):
            k = kernel_size[0]
        else:
            k = kernel_size

        self.filter1 = Conv2D(filters, (k, 1), strides=strides, padding=padding)
        self.filter2 = Conv2D(filters, (1, k), strides=(1, 1), padding=padding)

    def forward(self, A_prev, training=None):
        x = self.filter1.forward(A_prev, training=training)
        return self.filter2.forward(x, training=training)

    def backward(self, dZ, skip_activation=False):
        dA = self.filter2.backward(dZ)
        return self.filter1.backward(dA)

    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
        self.filter1.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)
        self.filter2.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)

    def get_params(self):
        return self.filter1.get_params() + self.filter2.get_params()

    def _child_attrs(self):
        return ["filter1", "filter2"]