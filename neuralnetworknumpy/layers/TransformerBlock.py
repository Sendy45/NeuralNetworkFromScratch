from .Layer import Layer
from .FeedForwardNetwork import FeedForwardNetwork
from .LayerNorm import LayerNorm
from .MultiHeadAttention import MultiHeadAttention
from neuralnetworknumpy.masks import causal_mask


class TransformerBlock(Layer):
    """
        Transformer block with Multi-Head Attention and FeedForward layers.

        Attention attends to input sequence with causal masking.
        Residual connections and LayerNorm applied before and after each sub-layer.

        Input shape: (B, T, D)
        Output shape: (B, T, D)
    """
    def __init__(self, model_dim, n_heads, ffn_dim):
        super().__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.norm1 = LayerNorm(model_dim)
        self.norm2 = LayerNorm(model_dim)
        self.attn = MultiHeadAttention(model_dim, n_heads)
        self.ffn = FeedForwardNetwork(model_dim, ffn_dim)

    def forward(self, X, training=None):
        T = X.shape[1]
        mask = causal_mask(T)  # (T, T) - broadcasts over batch

        # Multi-Head Attention
        mha_X = self.attn.forward(X, mask=mask, training=training)

        self.residual1 = mha_X + X  # cache for backward

        # Add & Norm 1
        norm1_X = self.norm1.forward(self.residual1, training=training)

        # Feed Forward
        ffn_X = self.ffn.forward(norm1_X, training=training)

        self.residual2 = ffn_X + self.residual1  # cache for backward

        # Add & Norm 2
        norm2_X = self.norm2.forward(self.residual2, training=training)

        return norm2_X

    def backward(self, dX):
        # Norm2 backward - gradient splits into ffn and norm1 skip
        d_residual2 = self.norm2.backward(dX)

        # FFN backward
        d_ffn = self.ffn.backward(d_residual2)

        # norm1_X received gradient from both ffn backward & skip from res2
        d_norm1 = d_ffn + d_residual2

        # Norm1 backward - gradient splits into attention and input skip
        d_residual1 = self.norm1.backward(d_norm1)

        # Attention backward
        d_mha = self.attn.backward(d_residual1)

        # X received gradient from both attention backward & skip from residual1
        return d_mha + d_residual1

    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        self.attn.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)
        self.ffn.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)
        self.norm1.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)
        self.norm2.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)

    def get_params(self):
        return sum(sub.get_params() for sub in [
            self.attn, self.ffn, self.norm1, self.norm2
        ])

    def describe(self): return f"TransformerBlock d_model={self.model_dim} heads={self.n_heads} ffn={self.ffn_dim}"

    def _cache_attrs(self): return ["residual1", "residual2"]

    def _child_attrs(self): return ["attn", "ffn", "norm1", "norm2"]
