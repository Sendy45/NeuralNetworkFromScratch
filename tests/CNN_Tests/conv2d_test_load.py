import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from neuralnetworknumpy import NeuralNetwork

(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
train_X = train_X[:10000].astype(np.float32) / 255.0
train_y = train_y[:10000]
test_X  = test_X.astype(np.float32)  / 255.0
train_X = train_X[..., np.newaxis]
test_X  = test_X[..., np.newaxis]

model = NeuralNetwork.load("conv2d_model.h5")

# Assume X_test[0:1] is a single image
feature_maps = model.visualize_feature_maps(test_X[0:1], layer_index=0)  # first conv layer

num_maps = feature_maps.shape[-1]
plt.figure(figsize=(15, 5))
for i in range(num_maps):
    plt.subplot(1, num_maps, i+1)
    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.show()