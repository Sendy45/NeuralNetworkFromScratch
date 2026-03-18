"""
Quick MNIST test for Conv2D → Flatten → Dense pipeline.
Run from your project root where the layers are importable.
"""

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from neuralnetworknumpy import Conv2D, Flatten, Dense, ReLu, Softmax, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, DepthwiseSeparableConv2D, ResidualBlock, BatchNorm2D
from neuralnetworknumpy import NeuralNetwork

# Load & prep data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = (x_train[:10000].astype(np.float32) / 255.0)[..., np.newaxis]  # (5000, 28, 28, 1)
y_train = y_train[:10000]  # you forgot this line

x_test  = (x_test[:1000].astype(np.float32)  / 255.0)[..., np.newaxis]
y_test = y_test[:1000]

print('x_train shape:', x_train.shape)

block = ResidualBlock(
    [
        Conv2D(8, (3, 3),strides=(2,2), padding="same"),
        BatchNorm2D(),
        ReLu(),
        DepthwiseSeparableConv2D(4, (3, 3), padding="same"),
        ReLu(),
    ],
    projection=Conv2D(4, 1, strides=(2,2), padding="same")
)
model = NeuralNetwork([
    block,
    AveragePooling2D((3, 3)),
    Flatten(),
    Dense(10),
    Softmax(),
])

model.compile(optimizer="adam", lr=0.01)
model.fit(x_train, y_train, epochs=2, batch_size=256)

acc = model.evaluate(x_test, y_test[:1000])
print(f"Test accuracy: {acc:.2%}")
assert acc > 0.70, f"Expected >70%, got {acc:.2%}"

model.summary()

model.save("conv2d_model.h5")

model_loaded = NeuralNetwork.load("conv2d_model.h5.npz")

acc = model_loaded.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.2%}")