from .Layer import Layer
from neuralnetworknumpy.backend import np

class RNN(Layer):
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size # H
        self.embed_dim = embed_dim # D

        # Weights for input and hidden state
        self.W_xh = np.random.randn(embed_dim, hidden_size) * np.sqrt(2.0 / embed_dim) # (D, H)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # (H, H) # keep small for recurrent

        # Biases for hidden state
        self.b_h = np.zeros((1, hidden_size))

        # Adam/momentum moments for input and hidden weights
        self.mW_xh = np.zeros((embed_dim, hidden_size))
        self.vW_xh = np.zeros((embed_dim, hidden_size))

        self.mW_hh = np.zeros((hidden_size, hidden_size))
        self.vW_hh = np.zeros((hidden_size, hidden_size))

        self.mb_h = np.zeros((1, hidden_size))
        self.vb_h = np.zeros((1, hidden_size))


    # RNN preserves past weights and calcs via the hidden state
    # h_t - current hidden state, is the result of former h_t-1, and current input x_t
    # this way, RNN remembers past inputs - but is suffering from vanishing gradient
    def forward(self, emb, h_init=None, c_init=None, training=None):
        B, T, D = emb.shape # (batch_size, seq_len, emb_dim)
        H = self.hidden_size

        # h_t - current hidden state
        # Allow passing in an initial hidden state (for decoder)
        h = np.zeros((B, H)) if h_init is None else h_init

        outputs = []

        # For backprop
        self.last_h = []
        self.last_x = []
        self.last_h_prev = []
        self.h_init = h
        self.last_batch_size = B

        for t in range(T): # Goes over each token (seq_len)
            # Current token batch and embedding dims
            x_t = emb[:, t, :] # (B, D)

            self.last_h_prev.append(h) # Store previous h

            # Hidden state - result of activation on dZ
            # h_t = tanh(W_hh * h_t-1 + W_xh * X_t)
            h = np.tanh(h @ self.W_hh + x_t @ self.W_xh + self.b_h)

            # Build last h and x for backprop
            self.last_h.append(h)
            self.last_x.append(x_t)

            outputs.append(h)

        # Built hidden state again (connect all tokens outputs)
        return np.stack(outputs, axis=1) # (B, T, H)




    def backward(self, dlogits):
        B, T, _ = dlogits.shape # (batch_size, seq_len, vocab_size OR hidden_size)
        H = self.hidden_size # hidden_state
        D = self.embed_dim # embedding_dims

        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)

        demb = np.zeros((B, T, D))  # <- gradient to return

        dh_next = np.zeros((B, H)) # hidden state gradient

        dz_all = np.zeros((B, T, H)) # dz - for biases


        # Backprop through time
        # Goes through all hidden states - last to first
        for t in reversed(range(T)):
            # last hidden and input from forward pass
            h_t = self.last_h[t]  # (B, H)
            x_t = self.last_x[t]  # (B, D)
            h_prev = self.last_h_prev[t] # h_t-1

            # passed directly from Dense.backward
            dh = dlogits[:, t, :]  # (B, H)

            dh += dh_next  # add future gradient

            # tanh derivative - (Activation derivative)
            dz = dh * (1 - h_t ** 2)  # (B, H)
            dz_all[:, t, :] = dz # Save for bias

            # Weight gradients
            # dW_xh = ∑ (X_t * dZ)
            dW_xh += x_t.T @ dz  # (D, H)
            # dW_hh = ∑ (h_t-1 * dZ)
            dW_hh += h_prev.T @ dz  # (H, H)

            # Input gradient
            # demb = dZ * W_xh
            demb[:, t, :] = dz @ self.W_xh.T

            # Pass gradient backward in time
            dh_next = dz @ self.W_hh.T  # (B, H)

        # Save gradients
        self.dW_xh = dW_xh
        self.dW_hh = dW_hh

        # Biases gradients
        # db_h = ∑ (dZ)
        self.db_h = np.sum(dz_all, axis=(0, 1), keepdims=False).reshape(1, -1)  # (1, H)

        # Expose gradient w.r.t. the initial hidden state
        # so Seq2Seq can pass it back to the encoder
        self.dh_init = dh_next  # (B, H)  - gradient w.r.t. h_init

        return demb


    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        # --- L2 regularization (skipped for adamW, handled via weight decay instead) ---
        if optimizer == "adamW":
            dW_xh = self.dW_xh
            dW_hh = self.dW_hh
        else:
            dW_xh = self.dW_xh + (lambda_ / self.last_batch_size) * self.W_xh
            dW_hh = self.dW_hh + (lambda_ / self.last_batch_size) * self.W_hh


        if optimizer == "momentum":
            self.vW_xh = beta1 * self.vW_xh + dW_xh
            self.vW_hh = beta1 * self.vW_hh + dW_hh
            self.vb_h = beta1 * self.vb_h + self.db_h

            upd_W_xh, upd_W_hh = self.vW_xh, self.vW_hh
            upd_b_h = self.vb_h

        elif optimizer in ("adam", "adamW"):
            # First moment (mean)
            self.mW_xh = beta1 * self.mW_xh + (1 - beta1) * dW_xh
            self.mW_hh = beta1 * self.mW_hh + (1 - beta1) * dW_hh
            self.mb_h = beta1 * self.mb_h + (1 - beta1) * self.db_h

            # Second moment (variance)
            self.vW_xh = beta2 * self.vW_xh + (1 - beta2) * dW_xh ** 2
            self.vW_hh = beta2 * self.vW_hh + (1 - beta2) * dW_hh ** 2
            self.vb_h = beta2 * self.vb_h + (1 - beta2) * self.db_h ** 2

            # Bias correction
            mW_xh_hat = self.mW_xh / (1 - beta1 ** t)
            mW_hh_hat = self.mW_hh / (1 - beta1 ** t)
            mb_h_hat = self.mb_h / (1 - beta1 ** t)

            vW_xh_hat = self.vW_xh / (1 - beta2 ** t)
            vW_hh_hat = self.vW_hh / (1 - beta2 ** t)
            vb_h_hat = self.vb_h / (1 - beta2 ** t)

            upd_W_xh = mW_xh_hat / (np.sqrt(vW_xh_hat) + _eps)
            upd_W_hh = mW_hh_hat / (np.sqrt(vW_hh_hat) + _eps)
            upd_b_h = mb_h_hat / (np.sqrt(vb_h_hat) + _eps)


        elif optimizer == "rmsprop":
            self.vW_xh = beta2 * self.vW_xh + (1 - beta2) * dW_xh ** 2
            self.vW_hh = beta2 * self.vW_hh + (1 - beta2) * dW_hh ** 2
            self.vb_h = beta2 * self.vb_h + (1 - beta2) * self.db_h ** 2

            upd_W_xh = dW_xh / (np.sqrt(self.vW_xh) + _eps)
            upd_W_hh = dW_hh / (np.sqrt(self.vW_hh) + _eps)
            upd_b_h = self.db_h / (np.sqrt(self.vb_h) + _eps)


        else:
            # SGD
            upd_W_xh, upd_W_hh = dW_xh, dW_hh
            upd_b_h = self.db_h


        # --- Apply updates ---
        self.W_xh -= lr * upd_W_xh
        self.W_hh -= lr * upd_W_hh
        self.b_h -= lr * upd_b_h


        # --- AdamW decoupled weight decay ---
        if optimizer == "adamW":
            self.W_xh *= (1 - lr * lambda_)
            self.W_hh *= (1 - lr * lambda_)