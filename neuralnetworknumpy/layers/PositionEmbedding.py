from .Layer import Layer
import numpy as np

class PositionEmbedding(Layer):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.embed_dim = None

        self.P = None

        # Gradients
        self.dP = None

        # Optimizer states
        self.vP = None
        self.mP = None

    def build(self, D):
        self.embed_dim = D

        # Small init — embeddings scale badly with large values
        self.P = np.random.random((self.seq_len, self.embed_dim)) * 0.01

        # Initialize optimizer states
        self.vP = np.zeros_like(self.P)
        self.mP = np.zeros_like(self.P)



    def _forward(self, X, training=None):
        # X: (B, T, D)
        B, T, D = X.shape

        if self.P is None:
            self.build(D)

        self.T = T
        self.X = X

        # A_{b,t,d} = X_{b,t,d} + P_{t,d}
        # (T, D) → (B, T, D)
        self.A = X + self.P[:T]  # (batch, seq_len, embed_dim)
        return self.A

    def _backward(self, dA):
        # Gradient flows unchanged to embedding
        # dA = ∂J / ∂A = 1
        dX = dA

        # Accumulate gradients for positional embeddings
        # sum over batch
        # ∂J / ∂P_{t, d} = Σ (∂J / ∂A_{b, t, d})
        self.dP = np.zeros_like(self.P)
        self.dP[:self.T] = np.sum(dA, axis=0)  # (T, D)

        return dX

    def _update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        if optimizer == "adamW":
            dP = self.dP  # pure gradient
        else:
            # Get batch size from the last dZ calculation to scale regularization
            m = self.A.shape[0]
            dP = self.dP + (lambda_ / m) * self.P  # L2 regularization

        if optimizer == "momentum":
            self.vP = beta1 * self.vP + dP

            update_p = self.vP

        elif optimizer == "adam" or optimizer == "adamW":

            self.mP = beta1 * self.mP + (1 - beta1) * dP

            self.vP = beta2 * self.vP + (1 - beta2) * (dP ** 2)

            m_p_hat = self.mP / (1 - beta1 ** t)

            v_p_hat = self.vP / (1 - beta2 ** t)

            update_p = m_p_hat / (np.sqrt(v_p_hat) + _eps)

        elif optimizer == "rmsprop":
            self.vP = beta2 * self.vP + (1 - beta2) * (dP ** 2)

            update_p = dP / (np.sqrt(self.vP) + _eps)

        else:
            # Classic SGD
            update_p = dP

        # P = P - lr * (dP + λ * P)
        self.P -= lr * update_p

        # Decouple (adamW) Weight Decay
        if optimizer == "adamW":
            self.P *= (1 - lr * lambda_)  # Decoupled weight decay


