from .Layer import Layer
import numpy as np


class BatchNorm(Layer):
    """
        Batch normalization for dense layers.

        Normalizes features across the batch dimension.

        Learns scale (gamma) and shift (beta).

        Improves stability and convergence.
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

    def build(self, input_size):
        # Initialize γ to 1 - keeps normalized values unchanged initially (scaling by 1 - unchanged)
        self.gamma = np.ones((1, input_size), dtype=np.float32)

        # Initialize β to 0 - no shift initially (shifting by 0 - unchanged)
        self.beta = np.zeros((1, input_size), dtype=np.float32)

        # Running mean initialized to 0
        self.running_mean = np.zeros((1, input_size), dtype=np.float32)
        # Running variance initialized to 1
        self.running_var = np.ones((1, input_size), dtype=np.float32)

    def forward(self, A_prev, training=True):
        # A_prev shape: (batch_size, features)

        # Lazy initialization (build on first forward pass)
        if self.gamma is None:
            self.build(A_prev.shape[-1])

        self.A_prev = A_prev  # store input for backward pass

        if training:
            # Compute batch statistics

            # Mean across batch (per feature)
            self.mean = np.mean(A_prev, axis=0, keepdims=True)

            # Variance across batch (per feature)
            self.var = np.var(A_prev, axis=0, keepdims=True)

            # Normalize
            # X̂ = (X - μB) / √(σB^2 + ε)
            self.X_hat = (A_prev - self.mean) / np.sqrt(self.var + self._eps)

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
        # dA shape: (batch_size, features)
        # Gradient of loss w.r.t. output of BatchNorm

        m = dA.shape[0]  # batch size

        # Scale and Shift derivatives
        # dγ = Σ (dA * X̂)
        dgamma = np.sum(dA * self.X_hat, axis=0, keepdims=True)
        # dβ = Σ (dA)
        dbeta = np.sum(dA, axis=0, keepdims=True)

        # Scaling
        # dX̂ = dA * γ
        dX_hat = dA * self.gamma

        # Precompute inverse std deviation
        var_inv = 1. / np.sqrt(self.var + self._eps)

        # Variance derivative
        # Derivative of inverse
        # dvar = Σ (dX̂ * (X - μ) * -0.5 * (σ² + ε)^(-3/2))
        dvar = np.sum(dX_hat * (self.A_prev - self.mean) * -0.5 * var_inv ** 3,
                      axis=0, keepdims=True)

        # Mean derivative
        # dmean = Σ (dX̂ * - (σ² + ε)^(1/2)) + dvar * Σ (-2 * (X - μ)) / m
        dmean = (
                np.sum(dX_hat * -var_inv, axis=(0, ), keepdims=True)
                + dvar * np.sum(-2. * (self.A_prev - self.mean), axis=(0, ), keepdims=True) / m
        )

        # Total derivative
        # Combine all paths
        # dX = dX̂ / sqrt(var+eps) + dvar * 2 * (X - μ) / m + dmean / m
        dX = (
                dX_hat * var_inv
                + dvar * 2 * (self.A_prev - self.mean) / m
                + dmean / m
        )

        # Normalize gradients by batch size
        self.dgamma = dgamma / m
        self.dbeta = dbeta / m

        return dX

    def update(self, lambda_, lr, beta1, beta2, eps, optimizer, t):
        # Simple SGD update (no regularization typically applied to γ, β)
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

