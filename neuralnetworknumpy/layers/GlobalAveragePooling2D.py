from .Layer import Layer
from neuralnetworknumpy.backend import np

class GlobalAveragePooling2D(Layer):
    """
        Reduces each feature map to a single value.

        Converts (m, H, W, C) -> (m, C) by averaging spatial dimensions.

        Often used before classification layers to reduce parameters.

        Good large number of channels, not recommended for low number of channels
    """
    def __init__(self):
        super().__init__()

    def forward(self, A_prev, training=None):

        self.A_prev = A_prev

        # Operates the mean on the height and width dimensionalities for all the channels
        self.Z = np.mean(A_prev, axis=(1,2))

        self.A = self.Z  # needed by NeuralNetwork._compute_loss reg term
        return self.Z

    def backward(self, dA, skip_activation=False):

        # Output shape
        m, H, W, C = self.A_prev.shape

        # Broadcast dA to A shape and fill with the partial derivative of 1/(H*W)
        # ∂Z/∂A_prev = 1/(H*W)
        dA_prev = np.broadcast_to(dA[:, None, None, :], (m, H, W, C)).copy()
        dA_prev /= (H * W)

        return dA_prev


    def update(self, *args, **kwargs):
        pass

    def _cache_attrs(self): return ["A_prev", "A", "Z"]
