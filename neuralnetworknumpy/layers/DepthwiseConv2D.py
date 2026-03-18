from .GroupConv2D import GroupConv2D

class DepthwiseConv2D(GroupConv2D):
    """
        Depthwise convolution: one K_h×K_w filter per input channel, no mixing.
        Equivalent to GroupConv2D(filters=C_in, groups=C_in).

        Weight shape: (C_in, 1, K_h, K_w, 1)
          [one group per channel, one output filter, kernel, one input channel]
    """
    def __init__(self, kernel_size, strides:tuple=(1, 1), padding:str="valid", kernel_initializer: str=None):
        super().__init__(
            filters=None, # resolved to C_in in build()
            kernel_size=kernel_size,
            groups=None,  # resolved to C_in in build()
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer
        )

    def build(self, input_size):
        self.filters = input_size
        self.groups = input_size
        super().build(input_size)


