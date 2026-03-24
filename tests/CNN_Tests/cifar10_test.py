"""
CIFAR-10 CNN test for neuralnetworknumpy
----------------------------------------
Dataset : 50 000 train / 10 000 test, 32x32 RGB, 10 classes
          airplane · automobile · bird · cat · deer
          dog · frog · horse · ship · truck

Expected accuracy after 10 epochs: ~60-65 %
(pure-NumPy CNN on 32x32 colour images — reasonable baseline)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

from neuralnetworknumpy import (
    NeuralNetwork,
    Conv2D, DepthwiseSeparableConv2D,
    MaxPooling2D, AveragePooling2D,
    BatchNorm2D, Flatten, Dense,
    ReLu, Softmax, Dropout, ResidualBlock,
)

# ── labels ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── data ──────────────────────────────────────────────────────────────────
(train_X, train_y), (test_X, test_y) = cifar10.load_data()

# Labels come as (N, 1) — flatten to (N,)
train_y = train_y.flatten()
test_y  = test_y.flatten()

# Normalize to [0, 1]
train_X = train_X.astype(np.float32) / 255.0
test_X  = test_X.astype(np.float32)  / 255.0


# shapes: (50000, 32, 32, 3) and (10000, 32, 32, 3)
print(f"Train : {train_X.shape}  labels: {train_y.shape}")
print(f"Test  : {test_X.shape}  labels: {test_y.shape}")

# ── model ─────────────────────────────────────────────────────────────────
# Lightweight CNN sized for a pure-NumPy framework:
#   Stage 1  32x32 → 16x16  (32 filters)
#   Stage 2  16x16 →  8x8   (64 filters)
#   Stage 3   8x8  →  1x1   global average pool
#   Head      Dense 128 → 10
#
# Uses DepthwiseSeparableConv2D to keep parameter count low and
# forward/backward passes tractable without a GPU.

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
    # Stem: learn basic edges on full 32×32 input
    Conv2D(32, (3, 3), padding="same"), BatchNorm2D(), ReLu(),
    MaxPooling2D((2, 2)),                               # → 16×16×32

    # Stage 2: richer features
    res_block(32, 64),
    MaxPooling2D((2, 2)),                               # → 8×8×64

    # Stage 3: deep features, shrink spatial dims
    res_block(64, 128),
    AveragePooling2D((8, 8)),                           # → 1×1×128 (global avg pool)

    # Classifier head
    Flatten(),
    Dense(128), ReLu(), Dropout(0.3),
    Dense(10),  Softmax(),
])

model.compile(
    loss_type  = "cross_entropy",
    optimizer  = "adam",
    lr         = 0.001,
    lambda_    = 0.0001,
    beta1      = 0.9,
    beta2      = 0.999,
)

# ── train ─────────────────────────────────────────────────────────────────
print("\nTraining...\n")
t0 = time.time()

history = model.fit(
    X         = train_X,
    y         = train_y,
    X_val     = test_X,
    y_val     = test_y,
    epochs    = 15,
    batch_size= 64,
)

elapsed = time.time() - t0
print(f"\nTotal training time: {elapsed/60:.1f} min")

# ── evaluate ──────────────────────────────────────────────────────────────
test_acc = model.evaluate(test_X, test_y)
print(f"Test accuracy: {test_acc*100:.2f}%")

model.save("cifar10_model")
print("Model saved to cifar10_model.npz")

# ── learning curves ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["loss"],         label="train loss")
ax1.plot(history.history["val_loss"],     label="val loss")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

ax2.plot(history.history["accuracy"],     label="train acc")
ax2.plot(history.history["val_accuracy"], label="val acc")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

plt.tight_layout()
plt.savefig("cifar10_curves.png", dpi=120)
plt.show()

# ── sample predictions ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
indices = np.random.choice(len(test_X), 10, replace=False)

for ax, idx in zip(axes.flat, indices):
    x     = test_X[idx:idx+1]          # (1, 32, 32, 3)
    pred  = model.predict(x)[0]
    true  = test_y[idx]
    color = "green" if pred == true else "red"
    ax.imshow(test_X[idx])
    ax.set_title(f"P: {CLASS_NAMES[pred]}\nT: {CLASS_NAMES[true]}", color=color, fontsize=8)
    ax.axis("off")

plt.suptitle("Green = correct  |  Red = wrong", fontsize=10)
plt.tight_layout()
plt.savefig("cifar10_predictions.png", dpi=120)
plt.show()