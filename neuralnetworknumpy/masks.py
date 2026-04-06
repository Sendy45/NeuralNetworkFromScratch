# masks.py
from .backend import np
def causal_mask(T):
    """Blocks future positions. Shape: (T, T) - broadcasts over batch."""
    return np.triu(np.ones((T, T)), k=1).astype(bool)

def padding_mask(token_ids, pad_id=0):
    """Blocks PAD tokens. Shape: (B, 1, S) - broadcasts over query positions."""
    return (token_ids == pad_id)[:, np.newaxis, :]

def combined_mask(token_ids, pad_id=0):
    """Causal + padding for decoder self-attention. Shape: (B, T, T)."""
    T = token_ids.shape[1]
    causal  = causal_mask(T)                        # (T, T)
    padding = padding_mask(token_ids, pad_id)        # (B, 1, T)
    return causal | padding                          # (B, T, T)