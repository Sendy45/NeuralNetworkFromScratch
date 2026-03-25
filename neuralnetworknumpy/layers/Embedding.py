from .Layer import Layer
import numpy as np

class Embedding(Layer):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    # Small init — embeddings scale badly with large values
    def build(self):
        self.W = np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32) * 0.01
        self.dW = np.zeros_like(self.W)
        # Adam states
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)

    def forward(self, X, training=None):
        if self.W is None:
            self.build()
        # cache for backward
        self.input_ids = X  # (batch, seq_len) integers
        self.A = self.W[X]  # (batch, seq_len, embed_dim)
        return self.A

    def backward(self, dout):
        # Put gradient per weight back in its position
        # shape - (vocab_size, embed_size)
        # using add.at in case of token appearing more then once (like +=)
        self.dW[:] = 0  # reset before accumulating
        np.add.at(self.dW, self.input_ids, dout)

        return None # no gradient for integer inputs

    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        if optimizer == "adamW":
            dw = self.dW  # pure gradient
        else:
            m = self.input_ids.size          # number of token positions seen
            dw = self.dW + (lambda_ / m) * self.W  # L2 regularization

        if optimizer == "momentum":
            self.vW = beta1 * self.vW + dw
            update_w = self.vW

        elif optimizer == "adam" or optimizer == "adamW":

            self.mW = beta1 * self.mW + (1 - beta1) * dw

            self.vW = beta2 * self.vW + (1 - beta2) * (dw ** 2)

            m_w_hat = self.mW / (1 - beta1 ** t)

            v_w_hat = self.vW / (1 - beta2 ** t)

            update_w = m_w_hat / (np.sqrt(v_w_hat) + _eps)

        elif optimizer == "rmsprop":
            self.vW = beta2 * self.vW + (1 - beta2) * (dw ** 2)

            update_w = dw / (np.sqrt(self.vW) + _eps)

        else:
            # Classic SGD
            update_w = dw

        # W = W - lr * (dW + λ * W)
        self.W -= lr * update_w

        # Decouple (adamW) Weight Decay
        if optimizer == "adamW":
            self.W *= (1 - lr * lambda_)  # Decoupled weight decay

