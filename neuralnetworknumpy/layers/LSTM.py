from .Layer import Layer
from neuralnetworknumpy.backend import np
from .Activation import Sigmoid, Tanh, ReLu, Linear


class LSTM(Layer):
    """
        LSTM recurrent layer.

        Uses gated mechanisms to control information flow:

        - Forget gate (f):
            Decides what information from the previous cell state to discard.

        - Input gate (i):
            Controls how much new information is written to the cell state.

        - Candidate (c'):
            New information that could be added to the cell state.

        - Output gate (o):
            Determines what part of the cell state is exposed as the hidden state.

        Input shape: (B, T, D)

        Output shape: (B, T, H)
    """
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size # H
        self.embed_dim = embed_dim # D

        # Scale for weights
        scale = np.sqrt(2.0 / (embed_dim + hidden_size))

        # Forget Gate
        self.W_fx = np.random.randn(embed_dim, hidden_size) * scale # (D, H)
        self.W_fh = np.random.randn(hidden_size, hidden_size) * scale # (H, H)
        self.b_f = np.zeros((1, hidden_size)) # (1, H)
        self.f_activation = Sigmoid()

        # Input Gate
        self.W_ix = np.random.randn(embed_dim, hidden_size) * scale  # (D, H)
        self.W_ih = np.random.randn(hidden_size, hidden_size) * scale  # (H, H)
        self.b_i = np.zeros((1, hidden_size)) # (1, H)
        self.i_activation = Sigmoid()

        # Output Gate
        self.W_ox = np.random.randn(embed_dim, hidden_size) * scale  # (D, H)
        self.W_oh = np.random.randn(hidden_size, hidden_size) * scale  # (H, H)
        self.b_o = np.zeros((1, hidden_size))  # (1, H)
        self.o_activation = Sigmoid()

        # C
        self.W_cx = np.random.randn(embed_dim, hidden_size) * scale
        self.W_ch = np.random.randn(hidden_size, hidden_size) * scale
        self.b_c = np.zeros((1, hidden_size))
        self.c_activation = Tanh()



        # Adam/momentum moments for input and hidden weights
        self.mW_fx = np.zeros((embed_dim, hidden_size))
        self.vW_fx = np.zeros((embed_dim, hidden_size))

        self.mW_fh = np.zeros((hidden_size, hidden_size))
        self.vW_fh = np.zeros((hidden_size, hidden_size))

        self.mb_f = np.zeros((1, hidden_size))
        self.vb_f = np.zeros((1, hidden_size))

        self.mW_ix = np.zeros((embed_dim, hidden_size))
        self.vW_ix = np.zeros((embed_dim, hidden_size))

        self.mW_ih = np.zeros((hidden_size, hidden_size))
        self.vW_ih = np.zeros((hidden_size, hidden_size))

        self.mb_i = np.zeros((1, hidden_size))
        self.vb_i = np.zeros((1, hidden_size))

        self.mW_ox = np.zeros((embed_dim, hidden_size))
        self.vW_ox = np.zeros((embed_dim, hidden_size))

        self.mW_oh = np.zeros((hidden_size, hidden_size))
        self.vW_oh = np.zeros((hidden_size, hidden_size))

        self.mb_o = np.zeros((1, hidden_size))
        self.vb_o = np.zeros((1, hidden_size))

        self.mW_cx = np.zeros((embed_dim, hidden_size))
        self.vW_cx = np.zeros((embed_dim, hidden_size))

        self.mW_ch = np.zeros((hidden_size, hidden_size))
        self.vW_ch = np.zeros((hidden_size, hidden_size))

        self.mb_c = np.zeros((1, hidden_size))
        self.vb_c = np.zeros((1, hidden_size))


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

        # c_t - current c
        c = np.zeros((B, H)) if c_init is None else c_init

        outputs = []

        # Cache for backprop
        self.last_h = []
        self.last_x = []
        self.last_f = []
        self.last_i = []
        self.last_o = []

        self.last_c = []
        self.last_c_tilde = []

        self.h_init_cache = np.zeros((B, H)) if h_init is None else h_init.copy()
        self.c_init_cache = np.zeros((B, H)) if c_init is None else c_init.copy()

        for t in range(T): # Goes over each token (seq_len)
            # Current token batch and embedding dims
            x_t = emb[:, t, :] # (B, D)

            # ---------------
            # Forget gate
            # ---------------

            # f_t = σ( x_t * self.W_fx + h_t-1 * self.W_fh + self.b_f )
            f_t = self.f_activation.forward(x_t @ self.W_fx + h @ self.W_fh + self.b_f)

            # ---------------
            # Input gate
            # ---------------

            # i_t = σ( x_t * self.W_ix + h_t-1 * self.W_ih + self.b_i )
            i_t = self.i_activation.forward(x_t @ self.W_ix + h @ self.W_ih + self.b_i)

            # c_t' = σ( x_t * self.W_cx + h_t-1 * self.W_ch + self.b_c )
            c_tilde = self.c_activation.forward(x_t @ self.W_cx + h @ self.W_ch + self.b_c)

            # c_t = f_t ⊙ c_t-1 + i_t ⊙ c_t'
            c = f_t * c + i_t * c_tilde

            # ---------------
            # Output gate
            # ---------------

            # o_t = σ( x_t * self.W_ox + h_t-1 * self.W_oh + self.b_o )
            o_t = self.o_activation.forward(x_t @ self.W_ox + h @ self.W_oh + self.b_o)

            # ---------------
            # Final hidden state
            # ---------------

            # h_t = o_t ⊙ tanh(c_t)
            h = o_t * np.tanh(c)

            # Build last h and x for backprop
            self.last_h.append(h)
            self.last_x.append(x_t)
            self.last_f.append(f_t)
            self.last_i.append(i_t)
            self.last_o.append(o_t)
            self.last_c.append(c)
            self.last_c_tilde.append(c_tilde)

            outputs.append(h)

        # Built hidden state again (connect all tokens outputs)
        self.h_T = h # Store for seq2seq like models
        self.c_T = c # Store for seq2seq like models
        return np.stack(outputs, axis=1) # (B, T, H)



    def backward(self, dlogits, dc_external=None):
        B, T, _ = dlogits.shape # (batch_size, seq_len, vocab_size OR hidden_size)
        H = self.hidden_size # hidden_state
        D = self.embed_dim # embedding_dims

        # Initialize gradients
        dW_fx = np.zeros_like(self.W_fx)
        dW_fh = np.zeros_like(self.W_fh)

        dW_ix = np.zeros_like(self.W_ix)
        dW_ih = np.zeros_like(self.W_ih)

        dW_ox = np.zeros_like(self.W_ox)
        dW_oh = np.zeros_like(self.W_oh)

        dW_cx = np.zeros_like(self.W_cx)
        dW_ch = np.zeros_like(self.W_ch)

        db_f = np.zeros_like(self.b_f)
        db_i = np.zeros_like(self.b_i)
        db_o = np.zeros_like(self.b_o)
        db_c = np.zeros_like(self.b_c)

        demb = np.zeros((B, T, D))  # <- gradient to return

        dh_next = np.zeros((B, H)) # hidden state gradient

        # If an external dc gradient is passed in (from Seq2Seq encoder backward),
        # seed dc_next with it instead of zeros
        dc_next = np.zeros((B, H)) if dc_external is None else dc_external[:, -1, :]


        # Backprop through time
        # Goes through all hidden states - last to first
        for t in reversed(range(T)):
            # last params from forward pass
            x_t = self.last_x[t]  # (B, D)
            f_t = self.last_f[t]
            i_t = self.last_i[t]
            o_t = self.last_o[t]
            c = self.last_c[t]
            c_tilde = self.last_c_tilde[t]

            # h_t-1
            h_prev = self.last_h[t - 1] if t > 0 else self.h_init_cache
            # c_t-1
            c_prev = self.last_c[t - 1] if t > 0 else self.c_init_cache

            # Seq2Seq mode - dlogits is already dh (gradient w.r.t hidden state)
            # passed directly from Dense.backward in Seq2Seq
            dh = dlogits[:, t, :]  # (B, H)

            dh += dh_next  # add future gradient

            # ---------------
            # Output Gate
            # ---------------

            # do_t = dh_t ⊙ tanh(c_t)
            c_t_tanh = np.tanh(c)
            do_t = dh * c_t_tanh

            self.o_activation.A = o_t
            da_o = self.o_activation.backward(do_t)

            # Gradients for output gate
            # dW_ox = ∑ (X_t * da_o)
            dW_ox += x_t.T @ da_o
            # dW_oh = ∑ (h_t-1 * da_o)
            dW_oh += h_prev.T @ da_o
            # db_o = ∑ (da_o)
            db_o += np.sum(da_o, axis=0, keepdims=True)

            # ---------------
            # Cell Gradient
            # ---------------

            # dc_t(from h) = dh_t ⊙ o_t ⊙ ( 1 - tanh^2(c_t) )
            dc_t_h = dh * o_t * ( 1 - c_t_tanh * c_t_tanh )

            # dc_t = dc_t(from h) + dc_t(next)
            dc = dc_t_h + dc_next

            # dc_t' = dc_t ⊙ i_t
            dc_tilde = dc * i_t

            self.c_activation.A = c_tilde
            da_c_tilde = self.c_activation.backward(dc_tilde)

            # Gradients for cell
            # dW_cx = ∑ (X_t * da_f)
            dW_cx += x_t.T @ da_c_tilde
            # dW_ch = ∑ (h_t-1 * da_f)
            dW_ch += h_prev.T @ da_c_tilde
            # db_c = ∑ (da_f)
            db_c += np.sum(da_c_tilde, axis=0, keepdims=True)

            # ---------------
            # Forget Gate
            # ---------------

            df_t = dc * c_prev

            self.f_activation.A = f_t
            da_f = self.f_activation.backward(df_t)

            # Gradients for forget gate
            # dW_fx = ∑ (X_t * da_f)
            dW_fx += x_t.T @ da_f
            # dW_fh = ∑ (h_t-1 * da_f)
            dW_fh += h_prev.T @ da_f
            # db_f = ∑ (da_f)
            db_f += np.sum(da_f, axis=0, keepdims=True)

            # ---------------
            # Input Gate
            # ---------------

            # di_t = dc_t ⊙ c_t'
            di_t = dc * c_tilde

            self.i_activation.A = i_t
            da_i = self.i_activation.backward(di_t)

            # Gradients for input gate
            # dW_ix = ∑ (X_t * da_i)
            dW_ix += x_t.T @ da_i
            # dW_ih = ∑ (h_t-1 * da_i)
            dW_ih += h_prev.T @ da_i
            # db_i = ∑ (da_i)
            db_i += np.sum(da_i, axis=0, keepdims=True)

            # ---------------
            # Final Gradients
            # ---------------

            # dx_t = sum of dx_t per f, i, o and c'
            # dx_t_f = da_f * W_fx.T
            dx_t_f = da_f @ self.W_fx.T
            # dx_t_i = da_i * W_ix.T
            dx_t_i = da_i @ self.W_ix.T
            # dx_t_o = da_o * W_ox.T
            dx_t_o = da_o @ self.W_ox.T
            # dx_t_c' = da_c' * W_cx.T
            dx_t_c_tilde = da_c_tilde @ self.W_cx.T

            dx_t = dx_t_f + dx_t_i + dx_t_o + dx_t_c_tilde

            # dh_t-1 = sum of dh_t-1 per f, i, o and c
            # dh_t-1_f = da_f * W_fh.T
            dh_prev_f = da_f @ self.W_fh.T
            # dh_t-1_i = da_i * W_ih.T
            dh_prev_i = da_i @ self.W_ih.T
            # ddh_t-1_o = da_o * W_oh.T
            dh_prev_o = da_o @ self.W_oh.T
            # dh_t-1_c' = da_c' * W_ch.T
            dh_prev_c_tilde = da_c_tilde @ self.W_ch.T

            dh_prev = dh_prev_f + dh_prev_i + dh_prev_o + dh_prev_c_tilde

            # Save for next iteration
            dh_next = dh_prev
            # dc_t+1 = f_t * dc_t
            dc_next = f_t * dc

            # Fill demb per time step
            demb[:, t, :] = dx_t

        # Save gradients
        self.dW_fx = dW_fx
        self.dW_fh = dW_fh
        self.dW_ix = dW_ix
        self.dW_ih = dW_ih
        self.dW_ox = dW_ox
        self.dW_oh = dW_oh
        self.dW_cx = dW_cx
        self.dW_ch = dW_ch
        self.db_f = db_f
        self.db_i = db_i
        self.db_o = db_o
        self.db_c = db_c

        """ seq2seq """
        # Expose gradient w.r.t. the initial hidden state
        # so Seq2Seq can pass it back to the encoder
        self.dh_init = dh_next  # (B, H)  - gradient w.r.t. h_init
        self.dc_init = dc_next

        return demb

    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        params = [
            # (W, dW, m, v, b, db, mb, vb)
            ("fx", self.W_fx, self.dW_fx, self.mW_fx, self.vW_fx,
             self.b_f, self.db_f, self.mb_f, self.vb_f),

            ("fh", self.W_fh, self.dW_fh, self.mW_fh, self.vW_fh,
             None, None, None, None),

            ("ix", self.W_ix, self.dW_ix, self.mW_ix, self.vW_ix,
             self.b_i, self.db_i, self.mb_i, self.vb_i),

            ("ih", self.W_ih, self.dW_ih, self.mW_ih, self.vW_ih,
             None, None, None, None),

            ("ox", self.W_ox, self.dW_ox, self.mW_ox, self.vW_ox,
             self.b_o, self.db_o, self.mb_o, self.vb_o),

            ("oh", self.W_oh, self.dW_oh, self.mW_oh, self.vW_oh,
             None, None, None, None),

            ("cx", self.W_cx, self.dW_cx, self.mW_cx, self.vW_cx,
             self.b_c, self.db_c, self.mb_c, self.vb_c),

            ("ch", self.W_ch, self.dW_ch, self.mW_ch, self.vW_ch,
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