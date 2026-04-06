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
        Seq2Seq layer with optional RNN, GRU, or LSTM cell.

        Encoder reads source token sequence and produces hidden states.
        Decoder generates target sequence, optionally using teacher forcing.
        Final decoder hidden states are projected to vocab logits.

        Input shapes:
            X (source ids): (B, T_in)
            y (target ids, optional): (B, T_out)

        Output shape:
            (B, T_out, V)
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
        enc_h = self.encoder.forward(enc_emb)             # (B, T_in, H)
        h = self.encoder.h_T
        c = self.encoder.c_T

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
            start = np.full((B, 1), 1, dtype=np.int32)  # (B, 1)
            dec_input_ids = np.concatenate([start, y[:, :-1]], axis=1)  # (B, T_out)
        else:
            dec_input_ids = np.zeros((B, T_out), dtype=np.int32)

        # Embed full decoder input sequence
        dec_emb = self.decoder_embedding.forward(dec_input_ids, training)  # (B, T_out, E)

        # Run decoder RNN on full sequence
        dec_h = self.decoder.forward(dec_emb, h_init=h, c_init=c)  # (B, T_out, H)

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

        # Also pass dc back - create a matching c gradient buffer
        dc_enc = np.zeros_like(dh_enc)
        dc_enc[:, -1, :] = self.decoder.dc_init

        denc_emb = self.encoder.backward(dh_enc, dc_enc)
        self.encoder_embedding.backward(denc_emb)


        return np.zeros_like(self.enc_emb)     # no upstream layer before embeddings

    # ------------------------------------------------------------------

    def update(self, *args, **kwargs):
        self.encoder_embedding.update(*args, **kwargs)
        self.decoder_embedding.update(*args, **kwargs)
        self.encoder.update(*args, **kwargs)
        self.decoder.update(*args, **kwargs)
        self.out_proj.update(*args, **kwargs)

    def get_params(self):
        return sum(sub.get_params() for sub in self.children())

    def describe(self):
        return f"Seq2Seq          hidden={self.hidden_size}"

    def _cache_attrs(self):
        return ["enc_emb", "A", "probs_flat"]

    def _child_attrs(self):
        return ["encoder_embedding", "encoder",
                "decoder_embedding", "decoder", "out_proj"]

    def children(self):
        return [self.encoder_embedding, self.encoder,
                self.decoder_embedding, self.decoder, self.out_proj]