import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from neuralnetworknumpy import NeuralNetwork

import matplotlib.pyplot as plt
from keras.datasets import cifar10
import numpy as np

model = NeuralNetwork.load("cifar10_model.pkl")


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
plt.show()