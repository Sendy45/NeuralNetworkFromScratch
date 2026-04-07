import time

from keras.datasets import mnist
import numpy as np
import plt

from neuralnetworknumpy import NeuralNetwork, Dense, ReLu, BatchNorm, Dropout, Softmax, Scaler, \
    split_train_validation
from neuralnetworknumpy.learning_rate import LearningRate, LinearWarmup

model = NeuralNetwork([
    Dense(64, inputs=784),
    ReLu(),
    BatchNorm(),
    Dropout(0.1),
    Dense(52),
    ReLu(),
    Dropout(0.2),
    Dense(32),
    ReLu(),
    Dense(10),
    Softmax()
])

# Load MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Flatten images and normalize to [0,1]
train_X = train_X.reshape(60000, 784).astype('float32')
test_X  = test_X.reshape(10000, 784).astype('float32')

scaler = Scaler("minmax")
train_X = scaler.fit_transform(train_X)
test_X = scaler.fit_transform(test_X)

# Split into training and validation
train_X, train_y, val_X, val_y = split_train_validation(train_X, train_y, 0.1)


print("Train X:", train_X.shape, "Train y:", train_y.shape)
print("Validation X:", val_X.shape, "Validation y:", val_y.shape)
print("Test X:", test_X.shape, "Test y:", test_y.shape)


start_time = time.time()
lr = LinearWarmup(10, 0.001)
model.compile(loss_type="cross_entropy", optimizer="adamW", lr=lr, lambda_=0.01, beta1=0.9, beta2=0.999)
history = model.fit(train_X, train_y, val_X, val_y, epochs=10, batch_size=64)
model.summary()
model.save("model.h5")

print(f"Training time: {time.time() - start_time}")


idx = np.random.randint(test_X.shape[0])   # random sample index

x = test_X[idx:idx+1]    # shape (1, 784)
y_true = test_y[idx]

y_pred = model.predict(x)

plt.imshow(x.reshape(28,28), cmap="gray")
plt.title(f"Pred: {y_pred}, True: {y_true}")
plt.show()