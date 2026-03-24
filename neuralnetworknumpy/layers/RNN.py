from .Layer import Layer
import numpy as np

class RNN(Layer):
    def __init__(self, embed_dim, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size # H
        self.embed_dim = embed_dim # D
        self.vocab_size = vocab_size # V

        # Weights for input, hidden state and output
        self.W_xh = np.random.randn(embed_dim, hidden_size) * np.sqrt(2.0 / embed_dim) # (D, H)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # (H, H) # keep small for recurrent
        self.W_hy = np.random.randn(hidden_size, vocab_size) * np.sqrt(2.0 / hidden_size) # (H, V)

        # Biases for hidden state and output
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, vocab_size))

        # Adam/momentum moments for all three weight matrices
        self.mW_xh = np.zeros((embed_dim, hidden_size))
        self.vW_xh = np.zeros((embed_dim, hidden_size))

        self.mW_hh = np.zeros((hidden_size, hidden_size))
        self.vW_hh = np.zeros((hidden_size, hidden_size))

        self.mW_hy = np.zeros((hidden_size, vocab_size))
        self.vW_hy = np.zeros((hidden_size, vocab_size))

        self.mb_h = np.zeros((1, hidden_size))
        self.vb_h = np.zeros((1, hidden_size))

        self.mb_y = np.zeros((1, vocab_size))
        self.vb_y = np.zeros((1, vocab_size))

    # RNN preserves past weights and calcs via the hidden state
    # h_t - current hidden state, is the result of former h_t-1, and current input x_t
    # this way, RNN remembers past inputs - but is suffering from vanishing gradient
    def _forward(self, emb, training=None):
        B, T, D = emb.shape # (batch_size, seq_len, emb_dim)
        H = self.hidden_size

        # h_t - current hidden state
        h = np.zeros((B, H))
        outputs = []

        # For backprop
        self.last_h = []
        self.last_x = []

        for t in range(T): # Goes over each token (seq_len)
            # Current token batch and embedding dims
            x_t = emb[:, t, :] # (B, D)

            # Hidden state - result of activation on dZ
            # h_t = tanh(W_hh * h_t-1 + W_xh * X_t)
            h = np.tanh(h @ self.W_hh + x_t @ self.W_xh + self.b_h)

            # Build last h and x for backprop
            self.last_h.append(h)
            self.last_x.append(x_t)

            outputs.append(h)

        # Built hidden state again (connect all tokens outputs)
        h_stack = np.stack(outputs, axis=1) # (B, T, H)

        # logits = h * W_hy + b_y
        logits = h_stack @ self.W_hy + self.b_y # (B, T, V)

        return logits

    def _backward(self, dlogits):
        B, T, V = dlogits.shape # (batch_size, seq_len, vocab_size)
        H = self.hidden_size # hidden_state
        D = self.embed_dim # embedding_dims

        self.last_batch_size = B # save for update

        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)

        demb = np.zeros((B, T, D))  # <- gradient to return

        dh_next = np.zeros((B, H)) # hidden state gradient

        dz_all = np.zeros((B, T, H)) # dz - for biases

        # Backprop through time
        # Goes through all hidden states - last to first
        for t in reversed(range(T)):
            # last hidden and input from forward pass
            h_t = self.last_h[t]  # (B, H)
            x_t = self.last_x[t]  # (B, D)

            # h_t-1
            h_prev = self.last_h[t - 1] if t > 0 else np.zeros_like(h_t)

            # Get gradient for current t
            dlogits_t = dlogits[:, t, :]  # (B, V)

            # Output layer
            # dW_hy = ∑ (h_t * dlogits_t)
            dW_hy += h_t.T @ dlogits_t  # (H, V)

            # Hidden layer
            # dh = dlogits_t * W_hy + ∑ (dz * W_hh)
            dh = dlogits_t @ self.W_hy.T  # (B, H)
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
        self.dW_hy = dW_hy

        # Biases gradients
        # db_h = ∑ (dZ)
        self.db_h = np.sum(dz_all, axis=(0, 1), keepdims=False).reshape(1, -1)  # (1, H)
        # db_y = ∑ (dlogits)
        self.db_y = np.sum(dlogits, axis=(0, 1), keepdims=False).reshape(1, -1)  # (1, V)

        return demb


    def _update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        # --- L2 regularization (skipped for adamW, handled via weight decay instead) ---
        if optimizer == "adamW":
            dW_xh = self.dW_xh
            dW_hh = self.dW_hh
            dW_hy = self.dW_hy
        else:
            dW_xh = self.dW_xh + (lambda_ / self.last_batch_size) * self.W_xh
            dW_hh = self.dW_hh + (lambda_ / self.last_batch_size) * self.W_hh
            dW_hy = self.dW_hy + (lambda_ / self.last_batch_size) * self.W_hy

        if optimizer == "momentum":
            self.vW_xh = beta1 * self.vW_xh + dW_xh
            self.vW_hh = beta1 * self.vW_hh + dW_hh
            self.vW_hy = beta1 * self.vW_hy + dW_hy
            self.vb_h = beta1 * self.vb_h + self.db_h
            self.vb_y = beta1 * self.vb_y + self.db_y

            upd_W_xh, upd_W_hh, upd_W_hy = self.vW_xh, self.vW_hh, self.vW_hy
            upd_b_h, upd_b_y = self.vb_h, self.vb_y

        elif optimizer in ("adam", "adamW"):
            # First moment (mean)
            self.mW_xh = beta1 * self.mW_xh + (1 - beta1) * dW_xh
            self.mW_hh = beta1 * self.mW_hh + (1 - beta1) * dW_hh
            self.mW_hy = beta1 * self.mW_hy + (1 - beta1) * dW_hy
            self.mb_h = beta1 * self.mb_h + (1 - beta1) * self.db_h
            self.mb_y = beta1 * self.mb_y + (1 - beta1) * self.db_y

            # Second moment (variance)
            self.vW_xh = beta2 * self.vW_xh + (1 - beta2) * dW_xh ** 2
            self.vW_hh = beta2 * self.vW_hh + (1 - beta2) * dW_hh ** 2
            self.vW_hy = beta2 * self.vW_hy + (1 - beta2) * dW_hy ** 2
            self.vb_h = beta2 * self.vb_h + (1 - beta2) * self.db_h ** 2
            self.vb_y = beta2 * self.vb_y + (1 - beta2) * self.db_y ** 2

            # Bias correction
            mW_xh_hat = self.mW_xh / (1 - beta1 ** t)
            mW_hh_hat = self.mW_hh / (1 - beta1 ** t)
            mW_hy_hat = self.mW_hy / (1 - beta1 ** t)
            mb_h_hat = self.mb_h / (1 - beta1 ** t)
            mb_y_hat = self.mb_y / (1 - beta1 ** t)

            vW_xh_hat = self.vW_xh / (1 - beta2 ** t)
            vW_hh_hat = self.vW_hh / (1 - beta2 ** t)
            vW_hy_hat = self.vW_hy / (1 - beta2 ** t)
            vb_h_hat = self.vb_h / (1 - beta2 ** t)
            vb_y_hat = self.vb_y / (1 - beta2 ** t)

            upd_W_xh = mW_xh_hat / (np.sqrt(vW_xh_hat) + _eps)
            upd_W_hh = mW_hh_hat / (np.sqrt(vW_hh_hat) + _eps)
            upd_W_hy = mW_hy_hat / (np.sqrt(vW_hy_hat) + _eps)
            upd_b_h = mb_h_hat / (np.sqrt(vb_h_hat) + _eps)
            upd_b_y = mb_y_hat / (np.sqrt(vb_y_hat) + _eps)

        elif optimizer == "rmsprop":
            self.vW_xh = beta2 * self.vW_xh + (1 - beta2) * dW_xh ** 2
            self.vW_hh = beta2 * self.vW_hh + (1 - beta2) * dW_hh ** 2
            self.vW_hy = beta2 * self.vW_hy + (1 - beta2) * dW_hy ** 2
            self.vb_h = beta2 * self.vb_h + (1 - beta2) * self.db_h ** 2
            self.vb_y = beta2 * self.vb_y + (1 - beta2) * self.db_y ** 2

            upd_W_xh = dW_xh / (np.sqrt(self.vW_xh) + _eps)
            upd_W_hh = dW_hh / (np.sqrt(self.vW_hh) + _eps)
            upd_W_hy = dW_hy / (np.sqrt(self.vW_hy) + _eps)
            upd_b_h = self.db_h / (np.sqrt(self.vb_h) + _eps)
            upd_b_y = self.db_y / (np.sqrt(self.vb_y) + _eps)

        else:
            # SGD
            upd_W_xh, upd_W_hh, upd_W_hy = dW_xh, dW_hh, dW_hy
            upd_b_h, upd_b_y = self.db_h, self.db_y

        # --- Apply updates ---
        self.W_xh -= lr * upd_W_xh
        self.W_hh -= lr * upd_W_hh
        self.W_hy -= lr * upd_W_hy
        self.b_h -= lr * upd_b_h
        self.b_y -= lr * upd_b_y

        # --- AdamW decoupled weight decay ---
        if optimizer == "adamW":
            self.W_xh *= (1 - lr * lambda_)
            self.W_hh *= (1 - lr * lambda_)
            self.W_hy *= (1 - lr * lambda_)