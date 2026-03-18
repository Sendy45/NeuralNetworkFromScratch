from .Conv2D import Conv2D
import numpy as np

class GroupConv2D(Conv2D):
    """
        Grouped 2D Convolution.

        Divides the C_in input channels into `groups` equal groups and applies
        an independent (C_out/groups) - filter convolution to each.
        The per-group outputs are concatenated to give (m, H_out, W_out, C_out).

        Special cases:
          groups=1  ->  standard Conv2D
          groups=C_in, filters=C_in  ->  DepthwiseConv2D (one filter per channel)

        Weight shape: (G, C_out/G, K_h, K_w, C_in/G)
          [groups, filters_per_group, kernel_h, kernel_w, in_channels_per_group]

        All optimiser states and _update() are inherited from Conv2D unchanged -
        they work on any W shape because they use element wise operations.
    """
    def __init__(self, filters, kernel_size, groups=1, strides:tuple=(1, 1), padding:str="valid", kernel_initializer: str=None):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer
        )

        self.groups = groups # number of groups


    def build(self, input_size):
        if input_size % self.groups != 0:
            raise ValueError(f"Input channels {input_size} not divisible by groups {self.groups}")
        if self.filters % self.groups != 0:
            raise ValueError(f"Filters {self.filters} not divisible by groups {self.groups}")
        super().build(input_size)


    def _initialize_weights(self):

        G = self.groups
        K_h, K_w = self.kernel_size

        Cg_in = self.in_size // G
        Cg_out = self.filters // G
        fan_in = Cg_in * K_h * K_w
        fan_out = Cg_out * K_h * K_w

        if self.kernel_initializer == "he":
            # W ~ N(0, √(2/in))
            std = np.sqrt(2.0 / fan_in)
            self.W = np.random.randn(G, Cg_out, K_h, K_w, Cg_in).astype(np.float32) * std

        elif self.kernel_initializer == "xavier":
            # U(-√(6/(in+out)),√(6/(in+out)))
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W = np.random.uniform(-limit, limit, (G, Cg_out, K_h, K_w, Cg_in)).astype(np.float32)

        else:
            self.W = np.random.randn(G, Cg_out, K_h, K_w, Cg_in).astype(np.float32) * 0.01


    def _forward(self, A_prev, training=None):
        # A_prev shape: (m, H, W, C_in) - row-major, as Conv2D expects
        # Build layer if it's its first run
        if self.W is None:
            self.build(A_prev.shape[-1])

        G = self.groups
        # P_h, P_w - padding height and width
        # S_h, S_w - strides height and width
        # K_h, K_w - kernel height and width
        P_h, P_w = self._padding_val
        S_h, S_w = self.strides
        K_h, K_w = self.kernel_size

        Ks = K_h * K_w

        # m - data length
        # H_in, W_in - matrix size
        # C_in - number of channels
        self.m, H_in, W_in, C_in = A_prev.shape

        Cg_in = C_in // G
        Cg_out = self.filters // G


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

        N = self.m * H_out * W_out
        self._N = N

        # Extract all patches at once using as_strided
        # im2col: (m, H_out, W_out, K_h, K_w, C_in) -> (N, K², C_in)

        s = self.A_pad.strides
        patches = np.lib.stride_tricks.as_strided(
            self.A_pad,
            shape=(self.m, H_out, W_out, K_h, K_w, C_in),
            strides=(s[0], s[1] * S_h, s[2] * S_w, s[1], s[2], s[3])
        )

        # Multiply patch by filter weights
        # then sum over K_h and K_w - channels stay independent
        # Store contiguous copy of patches for use in _backward
        self.patches = np.ascontiguousarray(patches).reshape(N, Ks, C_in)

        # Split channels into groups and reshape for batched matmul:
        # (N, K², C_in) -> (N, K², G, Cg_in) → (G, N, K²·Cg_in)
        # The C_in axis is a contiguous block of [g0_ch0..g0_chN, g1_ch0..g1_chN, ...]
        # so group g occupies columns g*Cg_in .. (g+1)*Cg_in - a simple reshape.
        cols_g = np.ascontiguousarray(
            self.patches.reshape(N, Ks, G, Cg_in).transpose(2, 0, 1, 3)
        ).reshape(G, N, Ks * Cg_in)  # (G, N, K²·Cg_in)

        # W: (G, Cg_out, K_h, K_w, Cg_in) -> (G, Cg_out, K²·Cg_in)
        W_mat = self.W.reshape(G, Cg_out, Ks * Cg_in)

        # Single batched matmul: (G, N, K²·Cg_in) @ (G, K²·Cg_in, Cg_out) -> (G, N, Cg_out)
        Z = np.matmul(cols_g, W_mat.transpose(0, 2, 1))  # (G, N, Cg_out)

        # Reassemble: interleave groups back along channel axis
        # (G, N, Cg_out) -> (N, G, Cg_out) -> (N, C_out) -> (m, H_out, W_out, C_out)
        Z = Z.transpose(1, 0, 2).reshape(self.m, H_out, W_out, self.filters)
        Z = Z + self.b.T


        self.Z = Z.astype(np.float32)
        self.A_prev = A_prev
        self.A = self.Z  # needed by NeuralNetwork._compute_loss reg term
        return self.Z

    def _backward(self, dZ, skip_activation=False):

        G = self.groups

        # P_h, P_w - padding height and width
        # S_h, S_w - strides height and width
        # K_h, K_w - kernel height and width
        P_h, P_w = self._padding_val
        S_h, S_w = self.strides
        K_h, K_w = self.kernel_size

        Ks = K_h * K_w

        _, H_out, W_out, _ = dZ.shape

        N = self._N
        C_in = self.in_size
        Cg_in = C_in // G
        Cg_out = self.filters // G

        # db = sum(dZ)
        db = np.sum(dZ, axis=(0, 1, 2)).reshape(self.filters, 1)

        # Split dZ into groups
        # C_out groups are contiguous: (N, G, Cg_out) → (G, N, Cg_out)
        dZ_g = np.ascontiguousarray(
            dZ.reshape(N, G, Cg_out).transpose(1, 0, 2)
        )  # (G, N, Cg_out)

        # Recompute cols_g from cached patches (cheaper than storing it)
        cols_g = np.ascontiguousarray(
            self.patches.reshape(N, Ks, G, Cg_in).transpose(2, 0, 1, 3)
        ).reshape(G, N, Ks * Cg_in)  # (G, N, K²·Cg_in)

        # dW: (G, Cg_out, N) @ (G, N, K²·Cg_in) -> (G, Cg_out, K²·Cg_in)
        dW = np.matmul(dZ_g.transpose(0, 2, 1), cols_g)  # (G, Cg_out, K²·Cg_in)
        dW = dW.reshape(self.W.shape)

        # dcols: (G, N, Cg_out) @ (G, Cg_out, K²·Cg_in) -> (G, N, K²·Cg_in)
        # Then merge groups back to get per-input-channel gradients
        W_mat = self.W.reshape(G, Cg_out, Ks * Cg_in)
        dcols = np.matmul(dZ_g, W_mat)  # (G, N, K²·Cg_in)

        # (G, N, K²·Cg_in) -> (N, K², G, Cg_in) -> (m, H_out, W_out, K_h, K_w, C_in)
        dcols = (
            dcols.reshape(G, N, Ks, Cg_in)
            .transpose(1, 2, 0, 3)  # (N, K², G, Cg_in)
            .reshape(self.m, H_out, W_out, K_h, K_w, C_in)
        )

        # Scatter dcols back into dA_pad
        # Each position (i,j) contributed a patch A_pad[:, i*S:i*S+K, j*S:j*S+K, :]
        # to the forward pass
        # In Reverse: accumulate gradients back into the same spatial positions

        # dA_pad shape:      (m, H_pad, W_pad, C_in)
        # patch view shape:  (m, H_out, W_out, K_h, K_w, C_in)

        dA_pad = np.zeros_like(self.A_pad)

        for i in range(K_h):
            for j in range(K_w):
                dA_pad[:,
                i:i + H_out * S_h:S_h,
                j:j + W_out * S_w:S_w,
                :] += dcols[:, :, :, i, j, :]

        # Create slices to Remove padding
        h_sl = slice(P_h, -P_h) if P_h > 0 else slice(None)
        w_sl = slice(P_w, -P_w) if P_w > 0 else slice(None)

        self.dW = dW
        self.db = db
        return dA_pad[:, h_sl, w_sl, :]