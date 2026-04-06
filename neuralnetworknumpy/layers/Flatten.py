from .Layer import Layer


class Flatten(Layer):
    """
        Flattens multidimensional input into 2D.

        Converts (m, H, W, C) -> (m, H*W*C).

        Used to transition from convolutional layers to fully connected layers.
    """
    def __init__(self):
        super().__init__()

    # Reshape 4D tensors into 2D
    # (m, H, W, C) -> (m, H*W*C)
    def forward(self, A_prev, training=None):
        self.input_shape = A_prev.shape
        return A_prev.reshape(A_prev.shape[0], -1)

    # Reshape 2D tensors into 4D
    # (m, H*W*C) -> (m, H, W, C)
    def backward(self, dA, skip_activation=False):
        return dA.reshape(self.input_shape)

    def update(self, *args, **kwargs):
        pass

    def _cache_attrs(self): return ["input_shape", "A_prev"]