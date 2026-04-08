# neuralnetworknumpy

A deep learning framework built from scratch using NumPy. Implements forward propagation, backpropagation, convolutional layers, recurrent layers, transformer blocks, residual connections, and common optimizers — no PyTorch or TensorFlow required.

---

## Installation

```bash
pip install neuralnetworknumpy
```

Or from source:

```bash
git clone https://github.com/Sendy45/NeuralNetworkFromScratch.git
cd neuralnetworknumpy
pip install .
```

**Dependencies:** `numpy`, `tqdm`  
**Optional (for examples):** `keras` (datasets only)

---

## Quick Start

### Dense network (MNIST)

```python
import numpy as np
from keras.datasets import mnist
from neuralnetworknumpy import NeuralNetwork
from neuralnetworknumpy.layers import Dense, ReLu, BatchNorm, Dropout, Softmax

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test  = X_test.reshape(-1, 784).astype(np.float32)  / 255.0

model = NeuralNetwork([
    Dense(256), ReLu(), BatchNorm(), Dropout(0.2),
    Dense(128), ReLu(), BatchNorm(),
    Dense(10),  Softmax()
])

model.compile(optimizer="adam", loss_type="cross_entropy", lr=0.001)
history = model.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=10, batch_size=64)
print(f"Val accuracy: {model.evaluate(X_test, y_test):.4f}")
```

### Convolutional network with residual blocks (Fashion-MNIST)

```python
import numpy as np
from keras.datasets import fashion_mnist
from neuralnetworknumpy import NeuralNetwork
from neuralnetworknumpy.layers import (
    Conv2D, DepthwiseSeparableConv2D, MaxPooling2D, AveragePooling2D,
    Flatten, Dense, ReLu, Softmax, BatchNorm2D, ResidualBlock
)

(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
train_X = train_X.astype(np.float32) / 255.0
test_X  = test_X.astype(np.float32)  / 255.0
train_X = train_X[..., np.newaxis]   # (N, 28, 28, 1)
test_X  = test_X[..., np.newaxis]

def res_block(in_ch, out_ch, stride=(1, 1)):
    projection = None
    if in_ch != out_ch or stride != (1, 1):
        projection = Conv2D(out_ch, 1, strides=stride, padding="same")
    return ResidualBlock([
        Conv2D(out_ch, (3, 3), strides=stride, padding="same"),
        BatchNorm2D(), ReLu(),
        DepthwiseSeparableConv2D(out_ch, (3, 3), padding="same"),
        BatchNorm2D(), ReLu(),
    ], projection=projection)

model = NeuralNetwork([
    Conv2D(16, (3, 3), padding="same"), BatchNorm2D(), ReLu(),
    MaxPooling2D((2, 2)),               # → 14×14×16
    res_block(16, 32),
    MaxPooling2D((2, 2)),               # → 7×7×32
    res_block(32, 64),
    AveragePooling2D((7, 7)),           # global avg pool → 1×1×64
    Flatten(),
    Dense(128), ReLu(),
    Dense(10),  Softmax()
])

model.compile(optimizer="adam", loss_type="cross_entropy", lr=0.001, lambda_=0.0001)
history = model.fit(train_X, train_y, X_val=test_X, y_val=test_y, epochs=20, batch_size=64)
model.save("fashion_model")
```

### Transformer language model (WikiText)

```python
import numpy as np
from neuralnetworknumpy import NeuralNetwork
from neuralnetworknumpy.layers import (
    Embedding, PositionEmbedding, TransformerBlock, Dense, Softmax
)
from neuralnetworknumpy.tokenizer import Tokenizer
from neuralnetworknumpy.learning_rate import LinearWarmup, CosineDecay, SequentialLR

tokenizer = Tokenizer()
tokenizer.load("tokenizer.json")
vocab_size = len(tokenizer.vocab)

SEQ_LEN   = 64
EMBED_DIM = 256
N_HEADS   = 8
N_BLOCKS  = 4
D_FFN     = EMBED_DIM * 4

model = NeuralNetwork([
    Embedding(vocab_size, EMBED_DIM),
    PositionEmbedding(SEQ_LEN),
    *[TransformerBlock(EMBED_DIM, N_HEADS, D_FFN) for _ in range(N_BLOCKS)],
    Dense(vocab_size, EMBED_DIM),
    Softmax()
])

total_steps  = 40 * (500_000 // 64)
warmup_steps = 4000

schedule = SequentialLR(
    schedules=[
        LinearWarmup(warmup_steps=warmup_steps, max_lr=0.001),
        CosineDecay(max_steps=total_steps - warmup_steps, base_lr=0.001),
    ],
    boundaries=[warmup_steps]
)

model.compile(optimizer="adam", loss_type="cross_entropy", lr=schedule, task="language_model")
model.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=40, batch_size=64)

# Generate text
print(model.generate("the researchers discovered", tokenizer, max_new_tokens=50, seq_len=SEQ_LEN))
```

### Seq2Seq with LSTM

```python
from neuralnetworknumpy import NeuralNetwork
from neuralnetworknumpy.layers import Seq2Seq
from neuralnetworknumpy.tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.load("tokenizer.json")

model = NeuralNetwork([
    Seq2Seq(vocab_size=len(tokenizer.vocab), embed_dim=128, hidden_size=256, layer_type="LSTM")
])

model.compile(optimizer="adam", loss_type="cross_entropy", lr=0.001, task="language_model")
model.fit((X_src, X_trg), y, epochs=20, batch_size=64)

print(model.generate("how are you", tokenizer, max_new_tokens=30, mode="seq2seq"))
```

---

## API Reference

### `NeuralNetwork`

```python
model = NeuralNetwork(layers)
```

| Method | Description |
|---|---|
| `compile(loss_type, optimizer, lr, lambda_, task)` | Set training hyperparameters |
| `fit(X, y, X_val, y_val, epochs, batch_size)` | Train the model, returns `History` |
| `predict(X)` | Returns class label predictions |
| `predict_proba(X)` | Returns raw output activations |
| `evaluate(X, y)` | Returns accuracy |
| `save(path)` | Serialise model to `.pkl` |
| `NeuralNetwork.load(path)` | Load a saved model |
| `summary()` | Print layer descriptions and parameter counts |
| `generate(prompt_ids, tokenizer, ...)` | Autoregressive text generation |
| `check_gradient(X, y)` | Numerical gradient check for debugging |

**Optimizers:** `"adam"`, `"adamW"`, `"momentum"`, `"rmsprop"`, `"sgd"`  
**Loss functions:** `"cross_entropy"`, `"mse"`  
**Tasks:** `"classification"`, `"language_model"`

---

### Layers

#### Dense

| Layer | Constructor | Notes |
|---|---|---|
| `Dense` | `Dense(units)` | Fully connected |
| `BatchNorm` | `BatchNorm(momentum=0.9)` | For 1D feature vectors |
| `Dropout` | `Dropout(rate)` | Dropped during training only |

#### Activations

`ReLu()` · `Sigmoid()` · `Softmax()` · `Tanh()` · `Linear()`

#### Convolutional

All conv layers expect input shape `(batch, H, W, channels)`.

| Layer | Constructor | Notes |
|---|---|---|
| `Conv2D` | `Conv2D(filters, kernel_size, strides, padding)` | Standard 2D convolution |
| `GroupConv2D` | `GroupConv2D(filters, kernel_size, groups, strides, padding)` | `groups=1` → Conv2D, `groups=C_in` → depthwise |
| `DepthwiseConv2D` | `DepthwiseConv2D(kernel_size, strides, padding)` | One filter per input channel |
| `DepthwiseSeparableConv2D` | `DepthwiseSeparableConv2D(filters, kernel_size, strides, padding)` | Depthwise + pointwise |
| `SpatiallySeparableConv2D` | `SpatiallySeparableConv2D(filters, kernel_size)` | Row × column factored convolution |
| `BatchNorm2D` | `BatchNorm2D(momentum=0.9)` | Normalises over spatial + batch dims |

#### Pooling

| Layer | Constructor | Notes |
|---|---|---|
| `MaxPooling2D` | `MaxPooling2D(pool_size, strides, padding)` | Takes max in each window |
| `AveragePooling2D` | `AveragePooling2D(pool_size, strides, padding)` | Takes average in each window |
| `GlobalAveragePooling2D` | `GlobalAveragePooling2D()` | Collapses H×W → 1 per channel |

#### Structural

| Layer | Constructor | Notes |
|---|---|---|
| `Flatten` | `Flatten()` | `(m, H, W, C)` → `(m, H*W*C)` |
| `ResidualBlock` | `ResidualBlock(layers, projection=None)` | Skip connection |

#### Recurrent

| Layer | Constructor | Notes |
|---|---|---|
| `RNN` | `RNN(embed_dim, hidden_size)` | Simple recurrent, suffers from vanishing gradient |
| `GRU` | `GRU(embed_dim, hidden_size)` | Gated — update + reset gates |
| `LSTM` | `LSTM(embed_dim, hidden_size)` | Gated — forget, input, output, cell |
| `Seq2Seq` | `Seq2Seq(vocab_size, embed_dim, hidden_size, layer_type)` | Encoder-decoder with RNN/GRU/LSTM |

#### Transformer

| Layer | Constructor | Notes |
|---|---|---|
| `Embedding` | `Embedding(vocab_size, embed_dim)` | Token id → dense vector |
| `PositionEmbedding` | `PositionEmbedding(seq_len)` | Learnable positional encoding |
| `TransformerBlock` | `TransformerBlock(model_dim, n_heads, ffn_dim)` | Causal self-attention + FFN + LayerNorm |
| `MultiHeadAttention` | `MultiHeadAttention(embed_dim, heads_num)` | Multi-head scaled dot-product attention |
| `LayerNorm` | `LayerNorm(embed_dim)` | Normalises over feature dimension |

---

### Learning Rate Schedules

```python
from neuralnetworknumpy.learning_rate import (
    LinearWarmup, CosineDecay, StepDecay, ExponentialDecay, SequentialLR
)
```

| Schedule | Constructor | Notes |
|---|---|---|
| `LinearWarmup` | `LinearWarmup(warmup_steps, max_lr)` | Ramps from 0 to `max_lr` |
| `CosineDecay` | `CosineDecay(max_steps, base_lr)` | Cosine annealing to 0 |
| `StepDecay` | `StepDecay(drop_rate, step_size, base_lr)` | Drops by factor every N steps |
| `ExponentialDecay` | `ExponentialDecay(drop_rate, base_lr)` | Continuous exponential decay |
| `SequentialLR` | `SequentialLR(schedules, boundaries)` | Chains schedules together |

**Warmup + cosine (recommended for transformers):**

```python
schedule = SequentialLR(
    schedules=[
        LinearWarmup(warmup_steps=4000, max_lr=0.001),
        CosineDecay(max_steps=total_steps - 4000, base_lr=0.001),
    ],
    boundaries=[4000]
)
model.compile(optimizer="adam", lr=schedule, ...)
```

---

### Masking

```python
from neuralnetworknumpy.masks import causal_mask, padding_mask, combined_mask
```

| Function | Description |
|---|---|
| `causal_mask(T)` | Upper-triangle mask — blocks future positions |
| `padding_mask(token_ids, pad_id)` | Blocks PAD tokens |
| `combined_mask(token_ids, pad_id)` | Causal + padding combined |

---

### Tokenizer

```python
from neuralnetworknumpy.tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit(text, vocab_size=8000)   # train BPE tokenizer
tokenizer.save("tokenizer.json")
tokenizer.load("tokenizer.json")

ids  = tokenizer.encode("hello world")
text = tokenizer.decode(ids)
```

**Download pretrained tokenizer (trained on WikiText-103):**

```python
from neuralnetworknumpy.pretrained import download_tokenizer

path = download_tokenizer()   # downloads tokenizer.json if not present
tokenizer.load(path)
```

---

### Text Generation

```python
# Transformer LM
output = model.generate(
    prompt_ids   = tokenizer.encode("the scientists discovered"),
    tokenizer    = tokenizer,
    max_new_tokens = 50,
    temperature  = 0.8,
    seq_len      = 64,
    mode         = "transformer",
    top_k        = 10
)

# Seq2Seq
output = model.generate(
    prompt_ids   = tokenizer.encode("how are you"),
    tokenizer    = tokenizer,
    max_new_tokens = 30,
    mode         = "seq2seq"
)
```

---

### Utilities

```python
from neuralnetworknumpy.utils import History, Scaler, split_train_test, split_train_validation
```

| Utility | Description |
|---|---|
| `Scaler(mode)` | `"standard"` or `"minmax"` normalisation |
| `split_train_test(X, y, test_ratio)` | Random train/test split |
| `split_train_validation(X, y, val_ratio)` | Random train/val split |
| `History` | Returned by `model.fit()` — tracks loss and metrics per epoch |

---

## Save and Load

```python
model.save("my_model")                    # writes my_model.pkl
model2 = NeuralNetwork.load("my_model")   # loads it back
print(model2.evaluate(X_test, y_test))
```

---

## Project Structure