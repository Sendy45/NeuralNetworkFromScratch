from neuralnetworknumpy.backend import np
from .Layer import Layer

class LayerNorm(Layer):
    """
            Layer Normalization over the feature dimension.

            Normalizes each token independently across its embedding features,
            stabilizing activations and improving training convergence.

            Unlike BatchNorm, statistics are computed per sample (B, T) and
            do not depend on other samples in the batch.

            Input shape: (B, T, D)

            Learns scale (gamma) and shift (beta) parameters.

            Output shape: (B, T, D)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Small constant to avoid division by zero
        self._eps = 1e-08

        # Learnable parameters
        # (1, 1, D) to broadcast over (B, T, D)
        self.gamma = np.ones((1, 1, embed_dim), dtype=np.float32)
        self.beta = np.zeros((1, 1, embed_dim), dtype=np.float32)

        # Optimizer moments
        self.dgamma = None
        self.dbeta = None
        self.mgamma = np.zeros_like(self.gamma)
        self.vgamma = np.zeros_like(self.gamma)
        self.mbeta = np.zeros_like(self.beta)
        self.vbeta = np.zeros_like(self.beta)

    def forward(self, X, training=None):
        self.X = X # (B, T, D)

        # Mean across features
        self.mean = np.mean(X, axis=-1, keepdims=True)
        # Variance across features
        self.var = np.var(X, axis=-1, keepdims=True)

        # Normalize
        # X̂ = (X - μF) / √(σF^2 + ε)
        self.X_hat = (X - self.mean) / np.sqrt(self.var + self._eps)

        # Scale and shift
        # A = γ * X̂ + β
        return self.gamma * self.X_hat + self.beta

    def backward(self, dA):
        B, T, D = dA.shape
        m = D # Across features

        # Scale and Shift derivatives (over B and T - axis(0,1))
        # dγ = Σ (dA * X̂)
        dgamma = np.sum(dA * self.X_hat, axis=(0, 1), keepdims=True) # (1, 1, D)
        # dβ = Σ (dA)
        dbeta = np.sum(dA, axis=(0, 1), keepdims=True) # (1, 1, D)

        # Scaling
        # dX̂ = dA * γ
        dX_hat = dA * self.gamma # (B, T, D)

        # Precompute inverse std deviation
        var_inv = 1. / np.sqrt(self.var + self._eps)

        # Variance derivative
        # Derivative of inverse
        # dvar = Σ (dX̂ * (X - μ) * -0.5 * (σ² + ε)^(-3/2))
        dvar = np.sum(dX_hat * (self.X - self.mean) * -0.5 * var_inv ** 3,
                      axis=-1, keepdims=True) # (B, T, 1)

        # Mean derivative
        # dmean = Σ (dX̂ * - (σ² + ε)^(1/2)) + dvar * Σ (-2 * (X - μ)) / m
        dmean = (
                np.sum(dX_hat * -var_inv, axis=-1, keepdims=True)
                + dvar * np.sum(-2. * (self.X - self.mean), axis=-1, keepdims=True) / m
        )

        # Total derivative
        # Combine all paths
        # dX = dX̂ / sqrt(var+eps) + dvar * 2 * (X - μ) / m + dmean / m
        dX = (
                dX_hat * var_inv
                + dvar * 2 * (self.X - self.mean) / m
                + dmean / m
        )

        # Normalize gradients by embed dim
        self.dgamma = dgamma / (B * T)
        self.dbeta = dgamma / (B * T)

        return dX # (B, T, D)


    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
        if optimizer in ("adam", "adamW"):
            self.mgamma = beta1 * self.mgamma + (1 - beta1) * self.dgamma
            self.vgamma = beta2 * self.vgamma + (1 - beta2) * self.dgamma ** 2
            self.mbeta = beta1 * self.mbeta + (1 - beta1) * self.dbeta
            self.vbeta = beta2 * self.vbeta + (1 - beta2) * self.dbeta ** 2

            mg_hat = self.mgamma / (1 - beta1 ** t)
            vg_hat = self.vgamma / (1 - beta2 ** t)
            mb_hat = self.mbeta / (1 - beta1 ** t)
            vb_hat = self.vbeta / (1 - beta2 ** t)

            self.gamma -= lr * mg_hat / (np.sqrt(vg_hat) + _eps)
            self.beta -= lr * mb_hat / (np.sqrt(vb_hat) + _eps)
        else:
            # SGD fallback — same as your BatchNorm
            self.gamma -= lr * self.dgamma
            self.beta -= lr * self.dbeta

    def get_params(self):
        return self.gamma.size + self.beta.size if self.gamma is not None else 0

    def describe(self):
        return f"LayerNorm        dim={self.embed_dim}"

    def _cache_attrs(self):
        return ["X", "mean", "var", "X_hat", "dgamma", "dbeta"]