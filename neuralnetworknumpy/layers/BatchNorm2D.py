from .Layer import Layer
import numpy as np


class BatchNorm2D(Layer):
    """
        Batch normalization for convolutional inputs.

        Normalizes each channel across batch and spatial dimensions.

        Learns scale (gamma) and shift (beta) parameters.

        Stabilizes and accelerates training.
    """
    def __init__(self, momentum=0.9):
        super().__init__()
        # Momentum for running mean/variance (used during inference)
        # Higher = smoother updates, slower adaptation
        self.momentum = momentum

        # Small constant to avoid division by zero
        self._eps = 1e-08

        # Learnable parameters
        self.gamma = None  # scale parameter (γ)
        self.beta = None  # shift parameter (β)

        # Running statistics (used during inference)
        self.running_mean = None
        self.running_var = None

        self.std_inv = None

    def build(self, input_size):
        # input_size = C_in
        # Initialize γ to 1 - keeps normalized values unchanged initially (scaling by 1 - unchanged)
        self.gamma = np.ones((1, 1, 1, input_size), dtype=np.float32)

        # Initialize β to 0 - no shift initially (shifting by 0 - unchanged)
        self.beta = np.zeros((1, 1, 1, input_size), dtype=np.float32)

        # Running mean initialized to 0
        self.running_mean = np.zeros((1, 1, 1, input_size), dtype=np.float32)
        # Running variance initialized to 1
        self.running_var = np.ones((1, 1, 1, input_size), dtype=np.float32)

    def forward(self, A_prev, training=True):
        # A_prev shape: (batch_size, H, W, channels)

        # Lazy initialization (build on first forward pass)
        if self.gamma is None:
            self.build(A_prev.shape[-1])

        self.A_prev = A_prev  # store input for backward pass

        if training:
            # Compute batch statistics

            # Mean across batch (per feature)
            self.mean = np.mean(A_prev, axis=(0, 1, 2), keepdims=True)

            # Variance across batch (per feature)
            self.x_mu = A_prev - self.mean  # cached for backward
            self.var = (self.x_mu * self.x_mu).mean(axis=(0, 1, 2), keepdims=True)

            # Compute inverse std
            var_inv = 1. / np.sqrt(self.var + self._eps)

            # Reuse in backward pass
            self.std_inv = var_inv

            # Normalize
            # X̂ = (X - μB) / √(σB^2 + ε)
            self.X_hat = (A_prev - self.mean) * var_inv

            # Scale and shift
            # A = γ * X̂ + β
            self.A = self.gamma * self.X_hat + self.beta

            # Update running stats
            self.running_mean = (
                    self.momentum * self.running_mean
                    + (1 - self.momentum) * self.mean
            )

            self.running_var = (
                    self.momentum * self.running_var
                    + (1 - self.momentum) * self.var
            )

        else:
            # Inference mode
            # Use running (global) statistics instead of batch stats
            # Normalize
            self.X_hat = (A_prev - self.running_mean) / np.sqrt(self.running_var + self._eps)
            # Scale and shift
            self.A = self.gamma * self.X_hat + self.beta

        return self.A

    def backward(self, dA, skip_activation=False):
        # dA shape: (batch_size, H, W, channels)
        # Gradient of loss w.r.t. output of BatchNorm

        N = np.prod(dA.shape[:3])  # m * H * W

        # Scale and Shift derivatives
        # dγ = Σ (dA * X̂)
        dgamma = np.sum(dA * self.X_hat, axis=(0, 1, 2), keepdims=True)
        # dβ = Σ (dA)
        dbeta = np.sum(dA, axis=(0, 1, 2), keepdims=True)

        # Scaling
        # dX̂ = dA * γ
        dX_hat = dA * self.gamma

        # Total derivative
        # Combine all paths
        # dX = dX̂ / sqrt(var+eps) + dvar * 2 * (X - μ) / N + dmean / N
        si2 = self.std_inv * self.std_inv  # std_inv² — avoids ** operator
        sum1 = np.sum(dX_hat, axis=(0, 1, 2), keepdims=True)
        sum2 = np.sum(dX_hat * self.x_mu, axis=(0, 1, 2), keepdims=True)

        dX = (1.0 / N) * self.std_inv * (
                N * dX_hat - sum1 - self.x_mu * si2 * sum2
        )

        """# Variance derivative
        # Derivative of inverse
        # dvar = Σ (dX̂ * (X - μ) * -0.5 * (σ² + ε)^(-3/2))
        dvar = np.sum(dX_hat * (self.A_prev - self.mean) * -0.5 * self.var_inv ** 3,
                      axis=(0, 1, 2), keepdims=True)

        # Mean derivative
        # dmean = Σ (dX̂ * - (σ² + ε)^(1/2)) + dvar * Σ (-2 * (X - μ)) / N
        dmean = (
                np.sum(dX_hat * -self.var_inv, axis=(0, 1, 2), keepdims=True)
                + dvar * np.sum(-2. * (self.A_prev - self.mean), axis=(0, 1, 2), keepdims=True) / N
        )

        # Total derivative
        # Combine all paths
        # dX = dX̂ / sqrt(var+eps) + dvar * 2 * (X - μ) / N + dmean / N
        dX = (
                dX_hat * self.var_inv
                + dvar * 2 * (self.A_prev - self.mean) / N
                + dmean / N
        )"""

        # Normalize gradients by batch size
        self.dgamma = dgamma / N
        self.dbeta = dbeta / N

        return dX

    def update(self, lambda_, lr, beta1, beta2, eps, optimizer, t):
        # Simple SGD update (no regularization typically applied to γ, β)
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

