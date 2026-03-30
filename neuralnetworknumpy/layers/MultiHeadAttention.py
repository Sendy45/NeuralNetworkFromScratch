from neuralnetworknumpy.backend import np
from .Layer import Layer
from .SingleHeadAttention import SingleHeadAttention
from .Dense import Dense


class MultiHeadAttention(Layer):
    def __init__(self, embed_dim, heads_num):
        super().__init__()

        self.key_dim = embed_dim // heads_num

        self.heads = [SingleHeadAttention(embed_dim, self.key_dim, output_projection=False) for _ in range(heads_num)]

        self.output_projection = Dense(embed_dim, inputs=heads_num * self.key_dim)


    def forward(self, emb, mask=None,training=None):
        outputs = [h.forward(emb) for h in self.heads]
        outputs = np.concatenate(outputs, axis=-1)
        outputs = self.output_projection.forward(outputs)
        return outputs

    def backward(self, dlogits, training=None):
        dlogits = self.output_projection.backward(dlogits)

        # Split dlogits along last dim - give each head his own gradient slice
        dlogits_split = np.split(dlogits, len(self.heads), axis=-1)

        demb_list = [h.backward(dl) for h, dl in zip(self.heads, dlogits_split)]

        # Sum head gradients to get input gradient
        demb = sum(demb_list)

        return demb

    def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

        self.output_projection.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)

        for h in self.heads:
            h.update(lambda_, lr, beta1, beta2, _eps, optimizer, t)