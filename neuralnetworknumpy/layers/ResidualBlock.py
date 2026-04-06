from .Layer import Layer

class ResidualBlock(Layer):
    """
        Residual connection block (skip connection).

        Output = F(x) + x (or projection(x) if dimensions differ).

        Helps training deep networks by preserving gradients.
    """
    def __init__(self, layers, projection=None):
        super().__init__()
        self.layers = layers         # main path
        self.projection = projection # optional shortcut (Conv1x1)

    def forward(self, A_prev, training=None):
        self.A_prev = A_prev

        out = A_prev

        # Find the F(x), main output
        for layer in self.layers:
            out = layer.forward(out, training=training)

        shortcut_out = A_prev
        # Handle projection if needed
        # output and shortcut_out shapes not aligning due to strides
        if self.projection is not None:
            shortcut_out = self.projection.forward(shortcut_out, training=training)

        # output = F(x) + x
        self.Z = out + shortcut_out
        self.A = self.Z
        return self.Z

    def backward(self, dZ):
        dZ_main = dZ

        # Main path
        for layer in reversed(self.layers):
            dZ_main = layer.backward(dZ_main)

        # Shortcut path
        if self.projection is not None:
            dZ_short = self.projection.backward(dZ)
        else:
            dZ_short = dZ

        return dZ_main + dZ_short

    def update(self, *args):
        # Update each layer inside of block
        for layer in self.layers:
            layer.update(*args)

        # Update projection conv2d (1x1 conv)
        if self.projection is not None:
            self.projection.update(*args)


    def get_params(self):
        total = sum(l.get_params() for l in self.layers)
        if self.projection is not None:
            total += self.projection.get_params()
        return total

    def _cache_attrs(self):
        return ["A_prev", "Z", "A"]

    def _child_attrs(self):
        return ["layers", "projection"]

    def children(self):
        kids = list(self.layers)
        if self.projection is not None:
            kids.append(self.projection)
        return kids