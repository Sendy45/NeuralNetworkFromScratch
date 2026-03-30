from .Layer import Layer
from neuralnetworknumpy.backend import np


# Abstract activation class
class Activation(Layer):
  """
    Base class for activation functions.

    Applies elementwise non linearity.

    No learnable parameters.
  """
  def __init__(self):
    super().__init__()

  def update(self, *args, **kwargs):
    pass  # no parameters to update


class ReLu(Activation):
  """
    Rectified Linear Unit activation.

    f(x) = max(0, x)

    Introduces non-linearity and helps avoid vanishing gradients.
  """
  # A = max(0, Z)
  def forward(self, Z, training=None):
    self.Z = Z
    self.A = np.maximum(0, Z)
    return self.A

  # dA = dZ * (Z > 0)
  def backward(self, dZ):
    return dZ * (self.Z > 0)

class Sigmoid(Activation):
  """
    Sigmoid activation.

    f(x) = 1 / (1 + e^-x)

    Outputs values in range (0, 1), often used for binary classification.
  """
  # A = 1 / (1 + e^-Z)
  def forward(self, Z, training=None):
      self.Z = Z
      self.A = 1 / (1 + np.exp(-Z))
      return self.A

  # dA = dZ * A * (1 - A)
  def backward(self, dZ):
      return dZ * self.A * (1 - self.A)

class Softmax(Activation):
  """
    Softmax activation.

    Converts logits into probability distribution over classes.

    Output sums to 1 across classes.

    Commonly used in multi-class classification.
  """
  # A = e^(Z - max(Z)) / ∑ (e^(Z - max(Z)))
  def forward(self, Z, training=None):
      self.Z = Z
      shifted = Z - np.max(Z, axis=-1, keepdims=True)
      exp_vals = np.exp(shifted)
      self.A = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
      return self.A

  # dA = A * (dZ - ∑(dZ * A))
  def backward(self, dZ):
      s = np.sum(dZ * self.A, axis=-1, keepdims=True)
      return self.A * (dZ - s)

class Linear(Activation):
  """
    Identity activation function.

    f(x) = x

    Used when no non linearity is desired.
  """
  # A = Z
  def forward(self, Z, training=None):
      self.Z = Z
      self.A = Z
      return self.A

  # dA = dZ
  def backward(self, dZ):
      return dZ  # derivative is 1

class Tanh(Activation):
  """
    Hyperbolic tangent activation.

    f(x) = tanh(x)

    Outputs values in range (-1, 1), zero-centered.
  """
  # A = tanh(Z)
  def forward(self, Z, training=None):
      self.Z = Z
      self.A = np.tanh(Z)
      return self.A

  # dA = dZ * (1 - A^2)
  def backward(self, dZ):
      return dZ * (1 - self.A ** 2)



