import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from neuralnetworknumpy import NeuralNetwork, Scaler

# ----------------------------

# Load saved model

# ----------------------------

model = NeuralNetwork.load("model.h5.npz")

print("Model loaded successfully")
model.summary()

# ----------------------------

# Load MNIST test data

# ----------------------------

(_, _), (test_X, test_y) = mnist.load_data()

test_X = test_X.reshape(10000, 784).astype("float32")

scaler = Scaler("minmax")
test_X = scaler.fit_transform(test_X)

# transpose to match network format

test_X = test_X.T

print("Test shape:", test_X.shape)

# ----------------------------

# Test prediction on random samples

# ----------------------------

num_tests = 10
correct = 0

for i in range(num_tests):

    idx = np.random.randint(test_X.shape[1])

    x = test_X[:, idx:idx+1]
    y_true = test_y[idx]

    y_pred = model.predict(x)

    print(f"Sample {i+1}: Pred={y_pred}, True={y_true}")

    if y_pred == y_true:
        correct += 1

    if i == 0:
        plt.imshow(x.reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {y_pred}, True: {y_true}")
        plt.show()


print("\nTest Accuracy (sampled):", correct / num_tests)
