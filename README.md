# NeuralNetworkFromScratch

A lightweight Python library implementing a fully functional neural network **from scratch using NumPy**, without relying on machine learning frameworks such as TensorFlow or PyTorch.

The goal of this project is to provide a clear and educational implementation of neural networks, including forward propagation, backpropagation, normalization, and regularization techniques.

---

## Features

* Fully connected neural network implementation
* Modular layer system
* Forward and backward propagation
* Batch normalization
* Dropout regularization
* ReLU and Softmax activation functions
* Dataset scaling utilities
* Train / validation split helpers

---

## Installation

Install from PyPI:

```bash
pip install neuralnetwork-from-scratch
```

Or install from source:

```bash
git clone https://github.com/Sendy45/NeuralNetworkFromScratch.git
cd NeuralNetworkFromScratch
pip install .
```

---

## Example Usage

```python
import numpy as np
from keras.datasets import mnist

from NeuralNetworkFromScratch import (
    NeuralNetwork,
    Dense,
    ReLu,
    BatchNorm,
    Dropout,
    Softmax
)

# load dataset
(X_train, y_train), _ = mnist.load_data()

# flatten images
X_train = X_train.reshape(-1, 784) / 255.0

model = NeuralNetwork([
    Dense(64, inputs=784),
    ReLu(),
    BatchNorm(),
    Dropout(0.1),
    Dense(10),
    Softmax()
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy"
)

model.fit(X_train, y_train, epochs=10, batch_size=32)
```

---

## Project Structure

```
NeuralNetworkFromScratch
│
├── neuralnet
│   ├── __init__.py
│   ├── network.py
│   ├── layers.py
│   ├── activations.py
│   └── utils.py
│
├── tests
├── README.md
└── pyproject.toml
```

---

## Goals of the Project

This project was designed to:

* Demonstrate **how neural networks work internally**
* Provide a **clean NumPy-based implementation**
* Serve as an **educational resource for learning deep learning fundamentals**

Unlike production ML frameworks, this project prioritizes **clarity and learning over performance**.

---

## Dependencies

* numpy
* tqdm

Optional dependencies used in examples:

* matplotlib
* keras (for datasets such as MNIST)

---

## License

This project is licensed under the MIT License.

---

## Author

Created by **Itamar Senderovitz**.
