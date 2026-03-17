from .Conv2D import Conv2D
import numpy as np


class DepthwiseConv2D(Conv2D):
    def __init__(self, kernel_size, strides:tuple=(1, 1), padding:str="valid", kernel_initializer: str=None):
        super().__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer
        )

    def build(self, input_size):
        self.filters = input_size
        super().build(input_size)


    def _initialize_weights(self):

        K_h, K_w = self.kernel_size
        C_in = self.in_size

        fan_in = K_h * K_w
        fan_out = K_h * K_w

        if self.kernel_initializer == "he":
            # W ~ N(0, √(2/in))
            std = np.sqrt(2.0 / fan_in)
            self.W = np.random.randn(K_h, K_w, C_in).astype(np.float32) * std

        elif self.kernel_initializer == "xavier":
            # U(-√(6/(in+out)),√(6/(in+out)))
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W = np.random.uniform(-limit, limit, (K_h, K_w, C_in)).astype(np.float32)

        else:
            self.W = np.random.randn(K_h, K_w, C_in).astype(np.float32) * 0.01


    def _forward(self, A_prev, training=None):
        # A_prev shape: (m, H, W, C_in) - row-major, as Conv2D expects
        # Build layer if it's its first run
        if self.W is None:
            self.build(A_prev.shape[-1])

        # P_h, P_w - padding height and width
        # S_h, S_w - strides height and width
        # K_h, K_w - kernel height and width
        P_h, P_w = self._padding_val
        S_h, S_w = self.strides
        K_h, K_w = self.kernel_size

        # m - data length
        # H_in, W_in - matrix size
        # C_in - number of channels
        self.m, H_in, W_in, C_in = A_prev.shape

        # Add padding to matrix
        self.A_pad = np.pad(
            A_prev,
            ((0, 0), (P_h, P_h), (P_w, P_w), (0, 0)),
            mode="constant"
        )

        # Calculate output matrix size
        # Used to create output matrix and fill it values
        H_out = (H_in + 2 * P_h - K_h) // S_h + 1
        W_out = (W_in + 2 * P_w - K_w) // S_w + 1

        # Extract all patches at once using as_strided
        # shape: (m, H_out, W_out, K_h, K_w, C_in)
        s = self.A_pad.strides
        patches = np.lib.stride_tricks.as_strided(
            self.A_pad,
            shape=(self.m, H_out, W_out, K_h, K_w, C_in),
            strides=(s[0], s[1] * S_h, s[2] * S_w, s[1], s[2], s[3])
        )

        # W shape - (K_h, K_w, C_in)
        # patches shape - (m, H_out, W_out, K_h, K_w, C_in)
        # Multiply patch by filter weights
        # then sum over K_h and K_w - channels stay independent
        # sum over axes 3,4 (K_h, K_w)
        self.patches = patches

        # Matmul to perform convolution
        # Z = A * W + b
        Z = (patches * self.W).sum(axis=(3, 4)) + self.b.T  # (m, H_out, W_out, C_in)

        self.Z = Z.astype(np.float32)
        self.A_prev = A_prev
        self.A = self.Z  # needed by NeuralNetwork._compute_loss reg term
        return self.Z

    def _backward(self, dZ, skip_activation=False):

        # P_h, P_w - padding height and width
        # S_h, S_w - strides height and width
        # K_h, K_w - kernel height and width
        P_h, P_w = self._padding_val
        S_h, S_w = self.strides
        K_h, K_w = self.kernel_size

        _, H_out, W_out, _ = dZ.shape

        # db = sum(dZ)
        db = np.sum(dZ, axis=(0, 1, 2)).reshape(self.filters, 1)

        # dW: (filters, K_h*K_w*C_in)
        # dW = dZ * A_pad (cols and dZ_col - reshaped into matrices for matmul)
        dZ_expanded = dZ[:, :, :, np.newaxis, np.newaxis, :] #(m, H_out, W_out, K_h, K_w, C_in)
        dW = (dZ_expanded * self.patches).sum(axis=(0,1,2))

        # dA: scatter gradients back via col2im
        # dA = dZ * W
        dcols = dZ_expanded * self.W #(m*H_out*W_out, K_h*K_w*C_in)



        # Scatter dcols back into dA_pad
        # Each position (i,j) contributed a patch A_pad[:, i*S:i*S+K, j*S:j*S+K, :]
        # to the forward pass
        # In Reverse: accumulate gradients back into the same spatial
        # positions using stride tricks (np.lib.stride_tricks.as_strided)
        #
        # as_strided Creates a VIEW of dA_pad shaped like the patches extracted,
        # without copying any data
        # Writing into this view writes directly into dA_pad
        #
        # dA_pad shape:      (m, H_pad, W_pad, C_in)
        # patch view shape:  (m, H_out, W_out, K_h, K_w, C_in)
        #
        # strides explanation (each number = bytes to step for that dimension):
        #   m dim     - same as dA_pad's m stride
        #   H_out dim - move S_h rows in dA_pad   = S_h * (W_pad * C_in) * itemsize
        #   W_out dim - move S_w cols in dA_pad   = S_w * C_in * itemsize
        #   K_h dim   - move 1 row in dA_pad      = (W_pad * C_in) * itemsize
        #   K_w dim   - move 1 col in dA_pad      = C_in * itemsize
        #   C_in dim  - move 1 channel in dA_pad  = itemsize

        dA_pad = np.zeros_like(self.A_pad)
        s = dA_pad.strides  # (s_m, s_h, s_w, s_c)

        patch_view = np.lib.stride_tricks.as_strided(
            dA_pad,
            shape=(self.m, H_out, W_out, K_h, K_w, dA_pad.shape[-1]),
            strides=(s[0], s[1] * S_h, s[2] * S_w, s[1], s[2], s[3])
        )

        # np.add.at is needed instead of += because multiple patches can overlap
        # (when stride < kernel size). += would only apply the last write;
        # np.add.at correctly accumulates all contributions.
        np.add.at(patch_view, np.index_exp[:], dcols)

        # Create slices to Remove padding
        h_sl = slice(P_h, -P_h) if P_h > 0 else slice(None)
        w_sl = slice(P_w, -P_w) if P_w > 0 else slice(None)

        self.dW = dW
        self.db = db
        return dA_pad[:, h_sl, w_sl, :]
