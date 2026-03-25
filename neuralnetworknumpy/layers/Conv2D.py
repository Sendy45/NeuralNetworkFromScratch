from .Layer import Layer
import numpy as np


class Conv2D(Layer):
    """
        Standard 2D convolution layer.

        Applies multiple learnable filters over spatial input.
        Each filter spans all input channels and produces one output channel.

        Weight shape: (filters, K_h, K_w, C_in)
          [num_filters, kernel height, kernel width, input channels]
    """
    def __init__(self, filters, kernel_size, strides:tuple=(1, 1), padding:str="valid", kernel_initializer: str=None):
        super().__init__()
        self.in_size = None
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides
        self.padding = padding
        self.out_size = filters

        self.kernel_initializer = kernel_initializer

        if not kernel_initializer:
            self.kernel_initializer = "he"

        self._padding_val = None  # resolved (P_h, P_w) after build

    def build(self, input_size):
        """input_size = C_in (number of input channels)"""
        self.in_size = input_size
        K_h, K_w = self.kernel_size

        self._initialize_weights()
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

    def _initialize_weights(self):

        K_h, K_w = self.kernel_size
        C_in = self.in_size
        C_out = self.filters

        fan_in = C_in * K_h * K_w
        fan_out = C_out * K_h * K_w

        if self.kernel_initializer == "he":
            # W ~ N(0, √(2/in))
            std = np.sqrt(2.0 / fan_in)
            self.W = np.random.randn(C_out, K_h, K_w, C_in).astype(np.float32) * std

        elif self.kernel_initializer == "xavier":
            # U(-√(6/(in+out)),√(6/(in+out)))
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W = np.random.uniform(-limit, limit, (C_out, K_h, K_w, C_in)).astype(np.float32)

        else:
            self.W = np.random.randn(C_out, K_h, K_w, C_in).astype(np.float32) * 0.01

    def forward(self, A_prev, training=None):
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

        """# im2col: extract every patch into a matrix
        # Reshape data and weights into matrices for matmul
        # cols shape: (m * H_out * W_out,  K_h * K_w * C_in)
        cols = np.array([
            self.A_pad[:, i * S_h:i * S_h + K_h, j * S_w:j * S_w + K_w, :]
            .reshape(self.m, -1)
            for i in range(H_out)
            for j in range(W_out)
        ])  # (H_out*W_out, m, K_h*K_w*C_in)

        self.cols = cols.transpose(1, 0, 2).reshape(self.m * H_out * W_out, -1)"""

        # Create sliding window view
        s = self.A_pad.strides

        patches = np.lib.stride_tricks.as_strided(
            self.A_pad,
            shape=(self.m, H_out, W_out, K_h, K_w, C_in),
            strides=(s[0], s[1] * S_h, s[2] * S_w, s[1], s[2], s[3])
        )

        # Flatten patches
        # np.ascontiguousarray to make the copy explicit and ensure the result is cache-friendly for matmul.
        self.cols = np.ascontiguousarray(patches).reshape(self.m * H_out * W_out, -1)

        # W_col shape: (filters, K_h*K_w*C_in)
        self.W_col = self.W.reshape(self.filters, -1)

        # Matmul to perform convolution
        # Z = A * W + b
        Z = (self.cols @ self.W_col.T + self.b.T)  # (m*H_out*W_out, filters)
        self.Z = Z.reshape(self.m, H_out, W_out, self.filters)


        self.A_prev = A_prev
        self.A = self.Z  # needed by NeuralNetwork._compute_loss reg term
        return self.Z

    def backward(self, dZ, skip_activation=False):

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
        # dZ_col shape: (m*H_out*W_out, filters)
        # (m*H_out*W_out, filters) @ (filters, K_h*K_w*C_in) -> (m*H_out*W_out, K_h*K_w*C_in)
        dcols = (dZ_col @ self.W_col).reshape(self.m, H_out, W_out, K_h, K_w, -1)

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
        # In Reverse: accumulate gradients back into the same spatial positions

        # dA_pad shape:      (m, H_pad, W_pad, C_in)
        # patch view shape:  (m, H_out, W_out, K_h, K_w, C_in)
        dA_pad = np.zeros_like(self.A_pad)

        for i in range(K_h):
            for j in range(K_w):
                # Each iteration is a single vectorised C call — no Python overhead
                # on the inner (m, H_out, W_out, C_in) dimensions.
                dA_pad[:, i:i + H_out * S_h:S_h, j:j + W_out * S_w:S_w, :] += dcols[:, :, :, i, j, :]


        # Create slices to Remove padding
        h_sl = slice(P_h, -P_h) if P_h > 0 else slice(None)
        w_sl = slice(P_w, -P_w) if P_w > 0 else slice(None)

        self.dW = dW
        self.db = db
        return dA_pad[:, h_sl, w_sl, :]


    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        if optimizer == "adamW":
            dw = self.dW
        else:
            m = self.A_prev.shape[0]  # batch size (row-major: axis 0)
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