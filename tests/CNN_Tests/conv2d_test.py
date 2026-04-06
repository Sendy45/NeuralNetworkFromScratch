import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from keras.datasets import fashion_mnist
from neuralnetworknumpy import (
    NeuralNetwork, Conv2D, DepthwiseSeparableConv2D,
    MaxPooling2D, AveragePooling2D, Flatten, Dense,
    ReLu, Softmax, BatchNorm2D, ResidualBlock
)

(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
train_X = train_X[:10000].astype(np.float32) / 255.0
train_y = train_y[:10000]
test_X  = test_X.astype(np.float32)  / 255.0
train_X = train_X[..., np.newaxis]
test_X  = test_X[..., np.newaxis]

def res_block(in_ch, out_ch, stride=(1, 1)):
    projection = None
    if in_ch != out_ch or stride != (1, 1):
        projection = Conv2D(out_ch, 1, strides=stride, padding="same")
    return ResidualBlock([
        Conv2D(out_ch, (3, 3), strides=stride, padding="same"),
        BatchNorm2D(), ReLu(),
        DepthwiseSeparableConv2D(out_ch, (3, 3), padding="same"),
        ReLu(),
    ], projection=projection)

model = NeuralNetwork([
    # Stage 1 — 28×28×16
    Conv2D(16, (3, 3), padding="same"), BatchNorm2D(), ReLu(),
    MaxPooling2D((2, 2)),                               # → 14×14×16

    # Stage 2 — 14×14×32
    res_block(16, 32, stride=(1, 1)),
    MaxPooling2D((2, 2)),                               # → 7×7×32

    Flatten(),
    Dense(128), ReLu(),
    Dense(10), Softmax()
])

model.compile(
    loss_type="cross_entropy",
    optimizer="adam",
    lr=0.001,
    lambda_=0.0001,
    beta1=0.9,
    beta2=0.999
)

history = model.fit(
    X=train_X, y=train_y,
    X_val=test_X, y_val=test_y,
    epochs=3,
    batch_size=64
)

model.summary()


model.save("conv2d_model.h5")