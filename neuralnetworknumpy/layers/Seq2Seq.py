from logging import raiseExceptions

from .Layer import Layer
from .GRU import GRU
from .RNN import RNN
from .LSTM import LSTM
from .Dense import Dense
import numpy as np
from .Embedding import Embedding
from .Activation import Softmax

LAYER_REGISTRY = {
        "RNN": RNN,
        "GRU": GRU,
        "LSTM": LSTM,
    }

class Seq2Seq(Layer):
    """
    Encoder-Decoder sequence to sequence block.

    Forward pass (training, teacher_forcing=True):
        1. Encoder RNN reads X:        (B, T_in, E)  -> hidden states, takes final h
        2. Decoder RNN steps T_out times, starting from encoder's final h
           - at each step fed the REAL previous token (teacher forcing)
        3. Output Dense projects each decoder hidden state to vocab logits

    Forward pass (inference, teacher_forcing=False):
        - Decoder feeds its own previous prediction back in at each step

    Shapes:
        X        : (B, T_in)           integer token ids  (embedding done inside)
        y        : (B, T_out)          integer token ids  (embedding done inside, for teacher forcing)
        output   : (B, T_out, V)       logits
    """



    def __init__(self, vocab_size, embed_dim, hidden_size, layer_type="RNN"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        if layer_type not in LAYER_REGISTRY:
            raise ValueError(f"Invalid layer_type '{layer_type}'. Choose from {list(LAYER_REGISTRY)}")

        cell = LAYER_REGISTRY[layer_type]

        # --------------
        # Encoder
        # --------------
        # Reads source sequence
        self.encoder_embedding = Embedding(vocab_size, embed_dim)
        self.encoder = cell(embed_dim, hidden_size)

        # --------------
        # Decoder
        # --------------
        # Generates target sequence
        self.decoder_embedding = Embedding(vocab_size, embed_dim)
        self.decoder = cell(embed_dim, hidden_size)

        # Project decoder hidden state -> vocab logits
        self.out_proj = Dense(vocab_size, hidden_size)
        self.out_softmax = Softmax()


    def forward(self, X, training=True):
        # When called from NeuralNetwork, y is packed with X as a tuple
        if isinstance(X, tuple):
            X, y = X
            teacher_forcing = True
        else:
            y = None
            teacher_forcing = False

        # X : (B, T_in)   source token ids
        # y : (B, T_out)  target token ids - required during training
        B, T_in = X.shape

        # --------------
        # Encoder
        # --------------
        enc_emb = self.encoder_embedding.forward(X, training)   # (B, T_in, E)
        enc_h   = self.encoder.forward(enc_emb)             # (B, T_in, H)
        h       = enc_h[:, -1, :]                            # (B, H) - context vector

        # Storage for backward pass
        self.enc_emb = enc_emb

        # --------------
        # Decoder
        # --------------
        T_out = y.shape[1] if y is not None else 50

        # Next decoder input
        # Build input ids for decoder
        # Starts empty - add token every iteration
        # <PAD> token id - index 0 in vocab

        # teacher forcing - feed real next token
        if teacher_forcing and y is not None:
            # Shift right: prepend start token, drop last token
            start = np.full((B, 1), 0, dtype=np.int32)  # (B, 1)
            dec_input_ids = np.concatenate([start, y[:, :-1]], axis=1)  # (B, T_out)
        else:
            dec_input_ids = np.zeros((B, T_out), dtype=np.int32)

        # Embed full decoder input sequence
        dec_emb = self.decoder_embedding.forward(dec_input_ids, training)  # (B, T_out, E)

        # Run decoder RNN on full sequence
        dec_h = self.decoder.forward(dec_emb, h_init=h)  # (B, T_out, H)

        # Project all timesteps at once
        # Flatten dec_h for Dense layer
        dec_h_flat = dec_h.reshape(B * T_out, self.hidden_size)  # (B*T, H)
        logits_flat = self.out_proj.forward(dec_h_flat)  # (B*T, V)
        probs_flat = self.out_softmax.forward(logits_flat)  # (B*T, V)
        self.probs_flat = probs_flat.copy()

        self.A = probs_flat.reshape(B, T_out, -1)  # (B, T_out, V)
        return self.A


    def backward(self, dA):
        """
        dA : (B, T_out, V)  gradient from loss w.r.t. logits
        Returns gradient w.r.t. encoder input embeddings (passed to enc_embedding)
        """
        B, T_out, V = dA.shape

        # Softmax backward
        # flatten for Dense and Softmax layers
        dlogits_flat = dA.reshape(B * T_out, V)  # (B*T, V)
        #self.out_softmax.A = self.probs_flat
        dlogits_flat = self.out_softmax.backward(dlogits_flat)  # (B*T, V)

        # Output projection backward
        dh_flat = self.out_proj.backward(dlogits_flat)  # (B*T, H)
        # Restore shape for RNN
        dh = dh_flat.reshape(B, T_out, self.hidden_size)  # (B, T_out, H)

        # Decoder RNN backward (full sequence)
        ddec_emb = self.decoder.backward(dh)  # (B, T_out, E)

        # Decoder embedding backward
        self.decoder_embedding.backward(ddec_emb)

        # Encoder backward
        dh_enc = np.zeros((B, self.enc_emb.shape[1], self.hidden_size)) # (B, T_in, H)
        dh_enc[:, -1, :] = self.decoder.dh_init  # grad from decoder init
        denc_emb = self.encoder.backward(dh_enc)
        self.encoder_embedding.backward(denc_emb)


        return np.zeros_like(self.enc_emb)     # no upstream layer before embeddings

    # ------------------------------------------------------------------

    def update(self, *args, **kwargs):
        self.encoder_embedding.update(*args, **kwargs)
        self.decoder_embedding.update(*args, **kwargs)
        self.encoder.update(*args, **kwargs)
        self.decoder.update(*args, **kwargs)
        self.out_proj.update(*args, **kwargs)