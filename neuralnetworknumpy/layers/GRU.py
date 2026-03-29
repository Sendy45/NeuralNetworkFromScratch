from .Layer import Layer
import numpy as np

from .Activation import Sigmoid, Tanh, ReLu, Linear

CANDIDATE_ACTIVATION_LAYERS = {
        "tanh": Tanh,
        "relu": ReLu,
        None: Linear,
    }

class GRU(Layer):
    def __init__(self, embed_dim, hidden_size, candidate_activation="tanh"):
        super().__init__()
        self.hidden_size = hidden_size # H
        self.embed_dim = embed_dim # D

        # Update Gate
        self.W_zx = np.random.randn(embed_dim, hidden_size) * 0.01 # (D, H)
        self.W_zh = np.random.randn(hidden_size, hidden_size) * 0.01 # (H, H)
        self.b_z = np.zeros((1, hidden_size)) # (1, H)
        self.z_activation = Sigmoid()

        # Reset Gate
        self.W_rx = np.random.randn(embed_dim, hidden_size) * 0.01  # (D, H)
        self.W_rh = np.random.randn(hidden_size, hidden_size) * 0.01  # (H, H)
        self.b_r = np.zeros((1, hidden_size)) # (1, H)
        self.r_activation = Sigmoid()


        # Weights for input and hidden state
        self.W_xh = np.random.randn(embed_dim, hidden_size) * np.sqrt(2.0 / embed_dim) # (D, H)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # (H, H) # keep small for recurrent
        # Biases for hidden state
        self.b_h = np.zeros((1, hidden_size))

        if candidate_activation not in CANDIDATE_ACTIVATION_LAYERS:
            raise ValueError(f"Invalid layer_type '{candidate_activation}'. Choose from {list(CANDIDATE_ACTIVATION_LAYERS)}")

        self.candidate_activation = CANDIDATE_ACTIVATION_LAYERS[candidate_activation]()


        # Adam/momentum moments for input and hidden weights
        self.mW_xh = np.zeros((embed_dim, hidden_size))
        self.vW_xh = np.zeros((embed_dim, hidden_size))

        self.mW_hh = np.zeros((hidden_size, hidden_size))
        self.vW_hh = np.zeros((hidden_size, hidden_size))

        self.mb_h = np.zeros((1, hidden_size))
        self.vb_h = np.zeros((1, hidden_size))

        self.mW_zx = np.zeros((embed_dim, hidden_size))
        self.vW_zx = np.zeros((embed_dim, hidden_size))

        self.mW_zh = np.zeros((hidden_size, hidden_size))
        self.vW_zh = np.zeros((hidden_size, hidden_size))

        self.mb_z = np.zeros((1, hidden_size))
        self.vb_z = np.zeros((1, hidden_size))

        self.mW_rx = np.zeros((embed_dim, hidden_size))
        self.vW_rx = np.zeros((embed_dim, hidden_size))

        self.mW_rh = np.zeros((hidden_size, hidden_size))
        self.vW_rh = np.zeros((hidden_size, hidden_size))

        self.mb_r = np.zeros((1, hidden_size))
        self.vb_r = np.zeros((1, hidden_size))


    # RNN preserves past weights and calcs via the hidden state
    # h_t - current hidden state, is the result of former h_t-1, and current input x_t
    # this way, RNN remembers past inputs - but is suffering from vanishing gradient
    def forward(self, emb, h_init=None, c_init=None, training=None):
        B, T, D = emb.shape # (batch_size, seq_len, emb_dim)
        H = self.hidden_size

        self.last_batch_size = B  # Store for L2 in update

        # h_t - current hidden state
        # Allow passing in an initial hidden state (for decoder)
        h = np.zeros((B, H)) if h_init is None else h_init


        outputs = []

        # Cache for backprop
        self.last_h = []
        self.last_x = []
        self.last_z = []
        self.last_r = []
        self.last_h_tilde = []

        self.h_init_cache = np.zeros((B, H)) if h_init is None else h_init.copy()

        for t in range(T): # Goes over each token (seq_len)
            # Current token batch and embedding dims
            x_t = emb[:, t, :] # (B, D)

            # ---------------
            # Update gate
            # ---------------

            # z_t = σ( x_t * self.W_zx + h_t-1 * self.W_zh + self.b_z )
            z_t = self.z_activation.forward(x_t @ self.W_zx + h @ self.W_zh + self.b_z)

            # ---------------
            # Reset gate
            # ---------------

            # r_t = σ( x_t * self.W_rx + h_t-1 * self.W_rh + self.b_r )
            r_t = self.r_activation.forward(x_t @ self.W_rx + h @ self.W_rh + self.b_r)

            # ---------------
            # Candidate hidden - result of activation
            # ---------------

            # h_t' = ϕ( W_xh * X_t + W_hh * ( r_t ⊙ h_t-1 ) + b_h )
            #h_tilde  = np.tanh((r_t * h) @ self.W_hh + x_t @ self.W_xh + self.b_h)
            h_tilde = self.candidate_activation.forward((r_t * h) @ self.W_hh + x_t @ self.W_xh + self.b_h)

            # ---------------
            # Final hidden state
            # ---------------

            # h_t = ( 1 - z_t ) ⊙ h_t-1 + z_t ⊙ h_t'
            h = (1-z_t) * h + z_t * h_tilde

            # Build last h and x for backprop
            self.last_h.append(h)
            self.last_x.append(x_t)
            self.last_z.append(z_t)
            self.last_r.append(r_t)
            self.last_h_tilde.append(h_tilde)

            outputs.append(h)

        # Built hidden state again (connect all tokens outputs)
        self.h_T = h # Store for seq2seq like models
        return np.stack(outputs, axis=1)
        # (B, T, H)



    def backward(self, dlogits):
        B, T, _ = dlogits.shape # (batch_size, seq_len, vocab_size OR hidden_size)
        H = self.hidden_size # hidden_state
        D = self.embed_dim # embedding_dims

        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)

        dW_rx = np.zeros_like(self.W_rx)
        dW_rh = np.zeros_like(self.W_rh)

        dW_zx = np.zeros_like(self.W_zx)
        dW_zh = np.zeros_like(self.W_zh)

        db_h = np.zeros_like(self.b_h)
        db_r = np.zeros_like(self.b_r)
        db_z = np.zeros_like(self.b_z)

        demb = np.zeros((B, T, D))  # <- gradient to return

        dh_next = np.zeros((B, H)) # hidden state gradient


        # Backprop through time
        # Goes through all hidden states - last to first
        for t in reversed(range(T)):
            # last hidden and input from forward pass
            x_t = self.last_x[t]  # (B, D)
            z_t = self.last_z[t]
            r_t = self.last_r[t]
            h_tilde = self.last_h_tilde[t]

            # h_t-1
            h_prev = self.last_h[t - 1] if t > 0 else self.h_init_cache

            # Seq2Seq mode - dlogits is already dh (gradient w.r.t hidden state)
            # passed directly from Dense.backward in Seq2Seq
            dh = dlogits[:, t, :]  # (B, H)

            dh += dh_next  # add future gradient

            # ---------------
            # Hidden State
            # ---------------

            dh_tilde = dh * z_t

            # h' Activation backward pass
            self.candidate_activation.A = h_tilde
            da_t = self.candidate_activation.backward(dh_tilde)

            # Gradients for hidden layer
            # dW_xh = ∑ (X_t * da_t)
            dW_xh += x_t.T @ da_t
            # dW_hh = ∑ ((r_t ⊙ h_t-1).T * da_t)
            dW_hh += ( r_t * h_prev ).T @ da_t
            # db_h = ∑ (da_t)
            db_h += np.sum(da_t, axis=0, keepdims=True)

            # Repeated calc
            d_temp = ( da_t @ self.W_hh.T )

            # ---------------
            # Reset Gate
            # ---------------

            dr_t = d_temp * h_prev

            # Reset gate Activation backward pass
            self.r_activation.A = r_t
            da_r = self.r_activation.backward(dr_t)

            # Gradients for reset gate
            # dW_rx = ∑ (X_t * da_r)
            dW_rx += x_t.T @ da_r
            # dW_rh = ∑ (h_t-1 * da_r)
            dW_rh += h_prev.T @ da_r
            # db_r = ∑ (da_r)
            db_r += np.sum(da_r, axis=0, keepdims=True)

            # ---------------
            # Update Gate
            # ---------------

            dz_t = dh * ( h_tilde - h_prev )
            # Update gate Activation backward pass
            self.z_activation.A = z_t
            da_z = self.z_activation.backward(dz_t)

            # Gradients for update gate
            # dW_zx = ∑ (X_t * da_z)
            dW_zx += x_t.T @ da_z
            # dW_zh = ∑ (h_t-1 * da_z)
            dW_zh += h_prev.T @ da_z
            # db_z = ∑ (da_z)
            db_z += np.sum(da_z, axis=0, keepdims=True)

            # ---------------
            # Final Gradients
            # ---------------

            # dx_t = sum of dx_t per h, r and z
            # dx_t_h = da_t * W_xh.T
            dx_t_h = da_t @ self.W_xh.T
            # dx_t_r = da_r * W_rx.T
            dx_t_r = da_r @ self.W_rx.T
            # dx_t_z = da_z * W_zx.T
            dx_t_z = da_z @ self.W_zx.T

            dx_t = dx_t_h + dx_t_r + dx_t_z

            # dh_prev = sum of dh_prev direct, candidate, r and z
            # dh_t-1_dir = dh ⊙ ( 1 - z_t )
            dh_prev_direct = dh * (1 - z_t)
            # dh_t-1_cand = ( da_t * W_hh.T ) ⊙ r_t
            dh_prev_candidate = d_temp * r_t
            # dh_t-1_r = da_r * W_rh.T
            dh_prev_r = da_r @ self.W_rh.T
            # dh_t-1_z = da_z * W_zh.T
            dh_prev_z = da_z @ self.W_zh.T

            dh_prev = dh_prev_direct + dh_prev_candidate + dh_prev_r + dh_prev_z

            # Save for next iteration
            dh_next = dh_prev

            # Fill demb per time step
            demb[:, t, :] = dx_t

        # Save gradients
        self.dW_xh = dW_xh
        self.dW_hh = dW_hh
        self.dW_rx = dW_rx
        self.dW_rh = dW_rh
        self.dW_zx = dW_zx
        self.dW_zh = dW_zh
        self.db_h = db_h
        self.db_r = db_r
        self.db_z = db_z

        """ seq2seq """
        # Expose gradient w.r.t. the initial hidden state
        # so Seq2Seq can pass it back to the encoder
        self.dh_init = dh_next  # (B, H)  - gradient w.r.t. h_init

        return demb

    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        params = [
            # (W, dW, m, v, b, db, mb, vb)
            ("xh", self.W_xh, self.dW_xh, self.mW_xh, self.vW_xh,
             self.b_h, self.db_h, self.mb_h, self.vb_h),

            ("hh", self.W_hh, self.dW_hh, self.mW_hh, self.vW_hh,
             None, None, None, None),  # no bias here

            ("rx", self.W_rx, self.dW_rx, self.mW_rx, self.vW_rx,
             self.b_r, self.db_r, self.mb_r, self.vb_r),

            ("rh", self.W_rh, self.dW_rh, self.mW_rh, self.vW_rh,
             None, None, None, None),

            ("zx", self.W_zx, self.dW_zx, self.mW_zx, self.vW_zx,
             self.b_z, self.db_z, self.mb_z, self.vb_z),

            ("zh", self.W_zh, self.dW_zh, self.mW_zh, self.vW_zh,
             None, None, None, None),
        ]

        for name, W, dW, mW, vW, b, db, mb, vb in params:

            # ---- L2 regularization ----
            if optimizer != "adamW":
                dW = dW + (lambda_ / self.last_batch_size) * W

            # ---- OPTIMIZERS ----

            if optimizer == "momentum":
                vW[:] = beta1 * vW + dW
                upd_W = vW

                if b is not None:
                    vb[:] = beta1 * vb + db
                    upd_b = vb

            elif optimizer in ("adam", "adamW"):
                # ---- weights ----
                mW[:] = beta1 * mW + (1 - beta1) * dW
                vW[:] = beta2 * vW + (1 - beta2) * (dW ** 2)

                mW_hat = mW / (1 - beta1 ** t)
                vW_hat = vW / (1 - beta2 ** t)

                upd_W = mW_hat / (np.sqrt(vW_hat) + _eps)

                # ---- bias ----
                if b is not None:
                    mb[:] = beta1 * mb + (1 - beta1) * db
                    vb[:] = beta2 * vb + (1 - beta2) * (db ** 2)

                    mb_hat = mb / (1 - beta1 ** t)
                    vb_hat = vb / (1 - beta2 ** t)

                    upd_b = mb_hat / (np.sqrt(vb_hat) + _eps)

            elif optimizer == "rmsprop":
                vW[:] = beta2 * vW + (1 - beta2) * (dW ** 2)
                upd_W = dW / (np.sqrt(vW) + _eps)

                if b is not None:
                    vb[:] = beta2 * vb + (1 - beta2) * (db ** 2)
                    upd_b = db / (np.sqrt(vb) + _eps)

            else:
                # SGD
                upd_W = dW
                if b is not None:
                    upd_b = db

            # ---- Apply updates ----
            W -= lr * upd_W

            if b is not None:
                b -= lr * upd_b

            # ---- AdamW decoupled weight decay ----
            if optimizer == "adamW":
                W *= (1 - lr * lambda_)