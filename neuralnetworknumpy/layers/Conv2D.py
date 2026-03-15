from .Layer import Layer
import numpy as np


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="valid"):
        super().__init__()
        self.in_size = None
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides
        self.padding = padding
        self.out_size = filters

        self.initializer = None

        self._padding_val = None  # resolved (P_h, P_w) after build

    def build(self, input_size):
        """input_size = C_in (number of input channels)"""
        self.in_size = input_size
        K_h, K_w = self.kernel_size

        # He initialization - matches Dense default
        std = np.sqrt(2.0 / (K_h * K_w * input_size))
        self.W = np.random.randn(self.filters, K_h, K_w, input_size).astype(np.float32) * std
        self.b = np.zeros((self.filters, 1), dtype=np.float32)

        # Optimizer states - same pattern as Dense
        self.mW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        if self.padding == "same":
            self._padding_val = ((K_h - 1) // 2, (K_w - 1) // 2)
        else:  # "valid"
            self._padding_val = (0, 0)

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

        # Pure math based approach, O(n^4), not optimized at all
        """
        self.Z = np.zeros((self.m, H_out, W_out, self.filters), dtype=np.float32)

        for n in range(self.m):
            for f in range(self.filters):
                for i in range(H_out):
                    for j in range(W_out):
                        vs, ve = i * S_h, i * S_h + K_h
                        hs, he = j * S_w, j * S_w + K_w
                        self.Z[n, i, j, f] = np.sum(self.A_pad[n, vs:ve, hs:he, :] * self.W[f]) + self.b[f, 0]
        """

        # im2col: extract every patch into a matrix
        # Reshape data and weights into matrices for matmul
        # cols shape: (m * H_out * W_out,  K_h * K_w * C_in)
        cols = np.array([
            self.A_pad[:, i * S_h:i * S_h + K_h, j * S_w:j * S_w + K_w, :]
            .reshape(self.m, -1)
            for i in range(H_out)
            for j in range(W_out)
        ])  # (H_out*W_out, m, K_h*K_w*C_in)

        self.cols = cols.transpose(1, 0, 2).reshape(self.m * H_out * W_out, -1)

        # W_col shape: (filters, K_h*K_w*C_in)
        W_col = self.W.reshape(self.filters, -1)

        # Matmul to perform convolution
        # Z = A * W + b
        Z = (self.cols @ W_col.T + self.b.T)  # (m*H_out*W_out, filters)
        self.Z = Z.reshape(self.m, H_out, W_out, self.filters)


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
        dZ_col = dZ.reshape(self.m * H_out * W_out, self.filters)
        dW = (dZ_col.T @ self.cols).reshape(self.W.shape)

        # dA: scatter gradients back via col2im
        # dA = dZ * W
        W_col = self.W.reshape(self.filters, -1)
        # dZ_col shape: (m*H_out*W_out, filters)
        # (m*H_out*W_out, filters) @ (filters, K_h*K_w*C_in) -> (m*H_out*W_out, K_h*K_w*C_in)
        dcols = (dZ_col @ W_col).reshape(self.m, H_out, W_out, K_h, K_w, -1)

        # Pure math based approach, O(n^2), not optimized at all
        """dA_pad = np.zeros_like(self.A_pad)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                vs, ve = i * S_h, i * S_h + K_h
                hs, he = j * S_w, j * S_w + K_w
                dA_pad[:, vs:ve, hs:he, :] += dcols[:, idx, :].reshape(self.m, K_h, K_w, -1)
                idx += 1"""

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


    def _update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
        m = self.A_prev.shape[0]  # batch size (row-major: axis 0)

        if optimizer == "adamW":
            dw = self.dW
        else:
            dw = self.dW + (lambda_ / m) * self.W  # L2

        if optimizer == "momentum":
            self.vW = beta1 * self.vW + dw
            self.vb = beta1 * self.vb + self.db
            update_w, update_b = self.vW, self.vb

        elif optimizer in ("adam", "adamW"):
            self.mW = beta1 * self.mW + (1 - beta1) * dw
            self.mb = beta1 * self.mb + (1 - beta1) * self.db
            self.vW = beta2 * self.vW + (1 - beta2) * (dw ** 2)
            self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

            mW_hat = self.mW / (1 - beta1 ** t)
            mb_hat = self.mb / (1 - beta1 ** t)
            vW_hat = self.vW / (1 - beta2 ** t)
            vb_hat = self.vb / (1 - beta2 ** t)

            update_w = mW_hat / (np.sqrt(vW_hat) + _eps)
            update_b = mb_hat / (np.sqrt(vb_hat) + _eps)

        elif optimizer == "rmsprop":
            self.vW = beta2 * self.vW + (1 - beta2) * (dw ** 2)
            self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)
            update_w = dw / (np.sqrt(self.vW) + _eps)
            update_b = self.db / (np.sqrt(self.vb) + _eps)

        else:  # SGD
            update_w, update_b = dw, self.db

        self.W -= lr * update_w
        self.b -= lr * update_b

        if optimizer == "adamW":
            self.W *= (1 - lr * lambda_)