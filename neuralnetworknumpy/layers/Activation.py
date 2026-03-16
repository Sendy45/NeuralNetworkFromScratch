from .Layer import Layer
import numpy as np

class Activation(Layer):
  def __init__(self):
    super().__init__()

  def _update(self, *args, **kwargs):
    pass  # no parameters to update

class ReLu(Activation):

  def _forward(self, Z, training=None):
    self.Z = Z
    self.A = np.maximum(0, Z)
    return self.A

  def _backward(self, dA):
    return dA * (self.Z > 0)

class Sigmoid(Activation):

  def _forward(self, Z, training=None):
      self.Z = Z
      self.A = 1 / (1 + np.exp(-Z))
      return self.A

  def _backward(self, dA):
      return dA * self.A * (1 - self.A)

class Softmax(Activation):

  def _forward(self, Z, training=None):
      self.Z = Z
      shifted = Z - np.max(Z, axis=1, keepdims=True)
      exp_vals = np.exp(shifted)
      self.A = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
      return self.A

  def _backward(self, dA):
      s = np.sum(dA * self.A, axis=1, keepdims=True)
      return self.A * (dA - s)

class Linear(Activation):

  def _forward(self, Z, training=None):
      self.Z = Z
      self.A = Z
      return self.A

  def _backward(self, dA):
      return dA  # derivative is 1

class Tanh(Activation):

  def _forward(self, Z, training=None):
      self.Z = Z
      self.A = np.tanh(Z)
      return self.A

  def _backward(self, dA):
      return dA * (1 - self.A ** 2)



