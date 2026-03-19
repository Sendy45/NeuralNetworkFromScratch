# neuralnetworknumpy

A deep learning framework built from scratch using NumPy. Implements forward propagation, backpropagation, convolutional layers, residual connections, and common optimizers — no PyTorch or TensorFlow required.

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
from neuralnetworknumpy import NeuralNetwork, Dense, ReLu, BatchNorm, Dropout, Softmax

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
from neuralnetworknumpy import (
    NeuralNetwork, Conv2D, DepthwiseSeparableConv2D,
    MaxPooling2D, AveragePooling2D, Flatten, Dense,
    ReLu, Softmax, BatchNorm2D, ResidualBlock
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

---

## API Reference

### `NeuralNetwork`

```python
model = NeuralNetwork(layers)
```

| Method | Description |
|---|---|
| `compile(loss_type, optimizer, lr, lambda_, beta1, beta2)` | Set training hyperparameters |
| `fit(X, y, X_val, y_val, epochs, batch_size)` | Train the model, returns `History` |
| `predict(X)` | Returns class label predictions |
| `predict_proba(X)` | Returns raw output activations |
| `evaluate(X, y)` | Returns accuracy |
| `save(path)` | Serialise weights to `.npz` |
| `NeuralNetwork.load(path)` | Load a saved model |
| `summary()` | Print layer shapes and parameter counts |

**Optimizers:** `"adam"`, `"adamW"`, `"momentum"`, `"rmsprop"`, `"sgd"`  
**Loss functions:** `"cross_entropy"`, `"mse"`

---

### Layers

#### Dense layers

| Layer | Constructor | Notes |
|---|---|---|
| `Dense` | `Dense(units)` | Fully connected |
| `BatchNorm` | `BatchNorm(momentum=0.9)` | For 1D feature vectors |
| `Dropout` | `Dropout(rate)` | Dropped during training only |

#### Activations

`ReLu()` · `Sigmoid()` · `Softmax()` · `Tanh()` · `Linear()`

#### 2D convolutional layers

All conv layers expect input shape `(batch, H, W, channels)`.

| Layer | Constructor | Notes |
|---|---|---|
| `Conv2D` | `Conv2D(filters, kernel_size, strides, padding)` | Standard 2D convolution |
| `GroupConv2D` | `GroupConv2D(filters, kernel_size, groups, strides, padding)` | Grouped convolution; `groups=1` → Conv2D, `groups=C_in` → depthwise |
| `DepthwiseConv2D` | `DepthwiseConv2D(kernel_size, strides, padding)` | One filter per input channel |
| `DepthwiseSeparableConv2D` | `DepthwiseSeparableConv2D(filters, kernel_size, strides, padding)` | Depthwise + pointwise |
| `SpatiallySeparableConv2D` | `SpatiallySeparableConv2D(filters, kernel_size, ...)` | Row × column factored convolution |
| `BatchNorm2D` | `BatchNorm2D(momentum=0.9)` | Normalises over spatial+batch dims |

#### Pooling

| Layer | Constructor | Default stride |
|---|---|---|
| `MaxPooling2D` | `MaxPooling2D(pool_size, strides, padding)` | Equal to `pool_size` |
| `AveragePooling2D` | `AveragePooling2D(pool_size, strides, padding)` | Equal to `pool_size` |
| `GlobalAveragePooling2D` | `GlobalAveragePooling2D()` | Collapses H×W → 1×1 |

#### Structural

| Layer | Constructor | Notes |
|---|---|---|
| `Flatten` | `Flatten()` | `(m, H, W, C)` → `(m, H*W*C)` |
| `ResidualBlock` | `ResidualBlock(layers, projection=None)` | Skip connection; pass a `Conv2D(1×1)` as `projection` when channels change |

**`ResidualBlock` example:**

```python
# Channels stay the same — no projection needed
ResidualBlock([Conv2D(32, 3, padding="same"), BatchNorm2D(), ReLu()])

# Channels change — projection required
ResidualBlock(
    [Conv2D(64, 3, strides=(2,2), padding="same"), BatchNorm2D(), ReLu()],
    projection=Conv2D(64, 1, strides=(2,2), padding="same")
)
```

**`GroupConv2D` example:**

```python
# 2-group convolution: splits 32 input channels into 2 independent groups
GroupConv2D(filters=64, kernel_size=3, groups=2, padding="same")

# Equivalent to DepthwiseConv2D when groups == C_in
GroupConv2D(filters=32, kernel_size=3, groups=32, padding="same")
```

---

### Utilities

```python
from neuralnetworknumpy import History, Scaler, split_train_test, split_train_validation
```

**`Scaler`**

```python
scaler = Scaler(mode="standard")   # or "minmax"
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

**`split_train_test` / `split_train_validation`**

```python
X_train, y_train, X_test, y_test  = split_train_test(X, y, test_ratio=0.2)
X_train, y_train, X_val,  y_val   = split_train_validation(X, y, val_ratio=0.2)
```

**`History`** — returned by `model.fit()`:

```python
history.history["loss"]         # list of per-epoch loss values
history.history["val_accuracy"] # list of per-epoch validation accuracy
```

---

## Save and Load

```python
model.save("my_model")           # writes my_model.npz

model2 = NeuralNetwork.load("my_model.npz")
print(model2.evaluate(X_test, y_test))
```

Saved layers: `Dense`, `Conv2D`, `GroupConv2D`, `DepthwiseConv2D`, `DepthwiseSeparableConv2D`, `BatchNorm`, `BatchNorm2D`, `MaxPooling2D`, `AveragePooling2D`, `GlobalAveragePooling2D`, `ResidualBlock`, `Dropout`, all activations.

---

## Project Structure

```
neuralnetworknumpy/
├── __init__.py
├── layers/
│   ├── Layer.py
│   ├── Dense.py
│   ├── Conv2D.py
│   ├── GroupConv2D.py          # GroupConv2D + DepthwiseConv2D
│   ├── DepthwiseSeparableConv2D.py
│   ├── SpatiallySeparableConv2D.py
│   ├── BatchNorm2D.py
│   ├── MaxPooling2D.py
│   ├── AveragePooling2D.py
│   ├── GlobalAveragePooling2D.py
│   ├── ResidualBlock.py
│   ├── Flatten.py
│   └── activations.py
├── model/
│   └── NeuralNetwork.py
└── utils/
    └── utils.py
```

---

## License

MIT License — see `LICENSE` for details.

## Author

Created by **Itamar Senderovitz**.