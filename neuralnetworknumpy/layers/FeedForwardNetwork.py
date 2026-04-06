from .Layer import Layer
from .Dense import Dense
from .Activation import ReLu

class FeedForwardNetwork(Layer):
    """
        Position-wise Feed-Forward Network (FFN) for transformers.

        Input: (B, T, D) - sequence embeddings
        Output: (B, T, D) - transformed embeddings

        Computation:
            FFN(x) = Dense2(ReLu(Dense1(x)))
            - Dense1: projects D -> ffn_dim
            - ReLu: elementwise activation
            - Dense2: projects ffn_dim -> D
    """
    def __init__(self, model_dim, ffn_dim):
        super().__init__()
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim

        self.dense1 = Dense(ffn_dim, model_dim)
        self.relu = ReLu()
        self.dense2 = Dense(model_dim, ffn_dim)

    def forward(self, X, training=None):

        # FFN(x)=ReLu(xW1+b1)W2+b2
        X = self.dense1.forward(X)
        X = self.relu.forward(X)
        X = self.dense2.forward(X)
        return X

    def backward(self, dA):

        dX = self.dense2.backward(dA)
        dX = self.relu.backward(dX)
        dX = self.dense1.backward(dX)

        return dX

    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        self.dense1.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)
        self.dense2.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)

    def describe(self): return f"FFN              {self.model_dim} → {self.ffn_dim} → {self.model_dim}"

    def _child_attrs(self): return ["dense1", "relu", "dense2"]
