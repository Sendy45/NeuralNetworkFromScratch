from neuralnetworknumpy.backend import np
from .Layer import Layer
from .Activation import Softmax

class SingleHeadAttention(Layer):
    def __init__(self, embed_dim, key_dim, output_projection=True):
        super().__init__()

        self.key_dim = key_dim
        self.softmax = Softmax()
        self.output_projection = output_projection

        # Scale for weights
        scale = np.sqrt(2.0 / (embed_dim + key_dim))

        self.W_q = np.random.randn(embed_dim, key_dim) * scale # (D, Dk)
        self.W_k = np.random.randn(embed_dim, key_dim) * scale
        self.W_v = np.random.randn(embed_dim, key_dim) * scale

        # Output projection - only when used standalone
        if output_projection:
            self.W_o = np.random.randn(key_dim, embed_dim) * scale  # (Dk, D)
            self.mW_o = np.zeros_like(self.W_o)
            self.vW_o = np.zeros_like(self.W_o)
        else:
            self.W_o = None

        # Optimizer moments
        self.mW_q = np.zeros_like(self.W_q)
        self.vW_q = np.zeros_like(self.W_q)
        self.mW_k = np.zeros_like(self.W_k)
        self.vW_k = np.zeros_like(self.W_k)
        self.mW_v = np.zeros_like(self.W_v)
        self.vW_v = np.zeros_like(self.W_v)


    def forward(self, emb, mask=None,training=None):
        B, T, D = emb.shape # (Batch_size, Seq_len, Embedding_dims)

        Q = emb @ self.W_q # (B, T, Dk)
        K = emb @ self.W_k # (B, T, Dk)
        V = emb @ self.W_v # (B, T, Dk)

        # Attention = softmax( (K.T @ Q) / √dk ) @ V

        # K.T @ Q
        scores = Q @ K.transpose(0, 2, 1)   # (B, T, Dk) @ (B, Dk, T) = (B, T, T)

        # / √dk
        scores /= np.sqrt(self.key_dim) # Keep numerical stability

        # Change later to optional
        # Masking - prevent future tokens from affecting past ones
        mask = np.triu(np.full((T, T), -np.inf), k=1)
        scores += mask

        # Softmax pass
        scores = self.softmax.forward(scores)
        # Get change to embeddings
        # Weighted sum for Values
        output = scores @ V

        self.scores = scores # cache for backprop
        self.emb = emb
        self.Q = Q
        self.K = K
        self.V = V

        # Projection (value up) for standalone layer
        if self.output_projection:
            self.pre_proj = output # cache for backprop
            output = output @ self.W_o

        return output

    def backward(self, dlogits, training=None):
        dOutputs = dlogits

        if self.output_projection:
            # dO = dY * Wo.T
            dOutputs = dlogits @ self.W_o.T

            B, T, D = self.pre_proj.shape
            # dWo = output.T * dY
            self.dW_o = self.pre_proj.reshape(B * T, D).T @ dlogits.reshape(B * T, -1)

        # d(QK.T) = dO @ V
        dScores = dOutputs @ self.V.transpose(0, 2, 1)
        # dV = d(QK.T) * dO
        dV = self.scores.transpose(0, 2, 1) @ dOutputs

        # Softmax
        # dS = softmax'(dS)
        self.softmax.A = self.scores
        dSoftmax = self.softmax.backward(dScores)

        # Scale
        # dZ = dS / √dk
        dZ = dSoftmax / np.sqrt(self.key_dim)

        # QK^T
        # dQ = dZ * K
        dQ = dZ @ self.K
        # dK = dZ * Q
        dK = dZ.transpose(0, 2, 1) @ self.Q

        # Flatten for weight grads
        B, T, D = self.emb.shape

        emb_flat = self.emb.reshape(B * T, D)
        dQ_flat = dQ.reshape(B * T, -1)
        dK_flat = dK.reshape(B * T, -1)
        dV_flat = dV.reshape(B * T, -1)

        # Gradients

        # dWq = X.T * dQ
        self.dW_q = emb_flat.T @ dQ_flat
        # dWk = X.T * dK
        self.dW_k = emb_flat.T @ dK_flat
        # dWv = X.T * dV
        self.dW_v = emb_flat.T @ dV_flat

        # Input gradients
        # dXq = dQ * Wq.T
        dX_q = dQ @ self.W_q.T
        # dXk = dK * Wk.T
        dX_k = dK @ self.W_k.T
        # dXv = dV * Wv.T
        dX_v = dV @ self.W_v.T

        # dX = dXq + dXk + dXv
        dX = dX_q + dX_k + dX_v

        return dX

    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        params = [
            # (W, dW, m, v, b, db, mb, vb)
            ("Q", self.W_q, self.dW_q, self.mW_q, self.vW_q),

            ("K", self.W_k, self.dW_k, self.mW_k, self.vW_k),

            ("V", self.W_v, self.dW_v, self.mW_v, self.vW_v),

        ]

        # Add output projection only if it exists
        if self.output_projection and self.W_o is not None:
            params.append(("O", self.W_o, self.dW_o, self.mW_o, self.vW_o))

        for name, W, dW, mW, vW in params:

            B = self.emb.shape[0]  # batch size

            # ---- L2 regularization ----
            if optimizer != "adamW":
                dW = dW + (lambda_ / B) * W

            # ---- OPTIMIZERS ----

            if optimizer == "momentum":
                vW[:] = beta1 * vW + dW
                upd_W = vW


            elif optimizer in ("adam", "adamW"):
                # ---- weights ----
                mW[:] = beta1 * mW + (1 - beta1) * dW
                vW[:] = beta2 * vW + (1 - beta2) * (dW ** 2)

                mW_hat = mW / (1 - beta1 ** t)
                vW_hat = vW / (1 - beta2 ** t)

                upd_W = mW_hat / (np.sqrt(vW_hat) + _eps)


            elif optimizer == "rmsprop":
                vW[:] = beta2 * vW + (1 - beta2) * (dW ** 2)
                upd_W = dW / (np.sqrt(vW) + _eps)


            else:
                # SGD
                upd_W = dW

            # ---- Apply updates ----
            W -= lr * upd_W


            # ---- AdamW decoupled weight decay ----
            if optimizer == "adamW":
                W *= (1 - lr * lambda_)