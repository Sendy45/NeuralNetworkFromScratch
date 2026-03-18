from .Layer import Layer

class ResidualBlock(Layer):
    def __init__(self, layers, projection=None):
        super().__init__()
        self.layers = layers         # main path
        self.projection = projection # optional shortcut (Conv1x1

    def _forward(self, A_prev, training=None):
        self.A_prev = A_prev

        out = A_prev
        for layer in self.layers:
            out = layer._forward(out, training=training)

        shortcut_out = A_prev
        if self.projection is not None:
            shortcut_out = self.projection._forward(shortcut_out, training=training)

        # output = F(x) + x
        self.Z = out + shortcut_out
        self.A = self.Z
        return self.Z

    def _backward(self, dZ):
        dZ_main = dZ

        # Main path
        for layer in reversed(self.layers):
            dZ_main = layer._backward(dZ_main)

        # Shortcut path
        if self.projection is not None:
            dZ_short = self.projection._backward(dZ)
        else:
            dZ_short = dZ

        return dZ_main + dZ_short

    def _update(self, *args):
        for layer in self.layers:
            layer._update(*args)
        if self.projection is not None:
            self.projection._update(*args)

