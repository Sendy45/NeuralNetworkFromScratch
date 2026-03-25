from .Layer import Layer
import numpy as np


class AveragePooling2D(Layer):
    """
        Downsamples input by averaging values in each window.

        Produces smoother feature maps compared to max pooling.

        No learnable parameters.
    """
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same"):
        super().__init__()

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self._padding_val = None

    def build(self, input_size):
        PS_h, PS_w = self.pool_size

        if self.padding == "same":
            self._padding_val = ((PS_h - 1) // 2, (PS_w - 1) // 2)
        else:  # "valid"
            self._padding_val = (0, 0)

    def forward(self, A_prev, training=None):
        # A_prev shape: (m, H, W, C_in) - row-major, as MaxPooling2D expects
        # Build layer if it's its first run
        if self._padding_val is None:
            self.build(A_prev.shape[-1])

        # P_h, P_w - padding height and width
        # S_h, S_w - strides height and width
        # PS_h, PS_w - Pool height and width
        P_h, P_w = self._padding_val
        S_h, S_w = self.strides
        PS_h, PS_w = self.pool_size

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
        H_out = (H_in + 2 * P_h - PS_h) // S_h + 1
        W_out = (W_in + 2 * P_w - PS_w) // S_w + 1

        # Extract all patches at once using as_strided
        # Same trick as Conv2D - no data copied, just a view into A_pad
        # shape: (m, H_out, W_out, PS_h, PS_w, C_in)
        s = self.A_pad.strides
        patches = np.lib.stride_tricks.as_strided(
            self.A_pad,
            shape=(self.m, H_out, W_out, PS_h, PS_w, C_in),
            strides=(s[0], s[1] * S_h, s[2] * S_w, s[1], s[2], s[3])
        )

        # Flatten each patch -> (m, H_out, W_out, PS_h*PS_w, C_in)
        # Enables average over the patch dimension (axis 3)
        patches_flat = patches.reshape(self.m, H_out, W_out, PS_h * PS_w, C_in)

        # average over the flattened patch dimension
        # shape: (m, H_out, W_out, C_in)
        self.Z = np.average(patches_flat, axis=3).astype(np.float32)

        # Cache whether this pool is non-overlapping
        # Used in backward to choose between direct assign and add.at.
        self._non_overlapping = (S_h >= PS_h and S_w >= PS_w)

        self.A_prev = A_prev
        self.A = self.Z  # needed by NeuralNetwork._compute_loss reg term
        return self.Z

    def backward(self, dA, skip_activation=False):

        # P_h, P_w - padding height and width
        # S_h, S_w - strides height and width
        # PS_h, PS_w - Pool height and width
        P_h, P_w = self._padding_val
        S_h, S_w = self.strides
        PS_h, PS_w = self.pool_size

        _, H_out, W_out, C_in = dA.shape

        dA_pad = np.zeros_like(self.A_pad)

        # dA = dA / pool_size (reverse average)
        # Each output gradient gets split equally across all patch positions
        # shape: (m, H_out, W_out, C_in)
        dA_distributed = dA / (PS_h * PS_w)

        # Expand matrix and handle overlap
        # Build patch position offsets within a pool window
        # ph_idx: all row offsets inside patch - (PS_h, 1)
        # pw_idx: all col offsets inside patch - (1, PS_w)
        ph_idx = np.arange(PS_h).reshape(PS_h, 1)
        pw_idx = np.arange(PS_w).reshape(1, PS_w)

        # Build the absolute coordinates in dA_pad for every patch start position
        # i, j are all H_out/W_out positions broadcast across the batch
        # shape after arange+reshape: (1, H_out, 1, 1) and (1, 1, W_out, 1)
        i_idx = np.arange(H_out).reshape(1, H_out, 1, 1, 1, 1)
        j_idx = np.arange(W_out).reshape(1, 1, W_out, 1, 1, 1)

        # ph/pw offsets: (1, 1, 1, PS_h, PS_w, 1)
        patch_h = ph_idx.reshape(1, 1, 1, PS_h, 1, 1)
        patch_w = pw_idx.reshape(1, 1, 1, 1, PS_w, 1)

        # Absolute position in dA_pad for every (i, j, ph, pw) combination
        # shape: (1, H_out, W_out, PS_h, PS_w, 1) - broadcasts over m and C
        abs_row = i_idx * S_h + patch_h
        abs_col = j_idx * S_w + patch_w

        # Data and channel indices - broadcast to same shape
        m_idx = np.arange(self.m).reshape(self.m, 1, 1, 1, 1, 1)
        c_idx = np.arange(C_in).reshape(1, 1, 1, 1, 1, C_in)

        # Expand dA_distributed to match the patch dims
        # (m, H_out, W_out, C_in) -> (m, H_out, W_out, 1, 1, C_in)
        dA_expanded = dA_distributed.reshape(self.m, H_out, W_out, 1, 1, C_in)

        # Scatter equally to every position in every patch
        # dA_pad[m, abs_row, abs_col, c] += dA[n,i,j,c]
        if self._non_overlapping:
            # When stride >= pool_size the pool
            # windows never overlap, so each position in dA_pad receives a gradient
            # contribution from exactly one output cell. There are no repeated indices,
            # making += equivalent to =.
            dA_pad[m_idx, abs_row, abs_col, c_idx] = dA_expanded
        else:
            # Overlapping pools (stride < pool_size): indices can repeat, must accumulate
            np.add.at(dA_pad, (m_idx, abs_row, abs_col, c_idx), dA_expanded)

        # Create slices to Remove padding
        h_sl = slice(P_h, -P_h) if P_h > 0 else slice(None)
        w_sl = slice(P_w, -P_w) if P_w > 0 else slice(None)

        return dA_pad[:, h_sl, w_sl, :]


    def update(self, *args, **kwargs):
        pass
