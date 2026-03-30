from .Layer import Layer
from neuralnetworknumpy.backend import np


class Dropout(Layer):
  """
    Regularization layer that randomly zeros activations during training.

    Prevents overfitting by reducing co-adaptation of neurons.

    Disabled during inference.
  """
  def __init__(self, rate):
    super().__init__()
    self.rate = rate  # probability of dropping a unit
    self.mask = None

  def forward(self, A_prev, training=True):
    if not training:
      # No dropout during inference
      self.mask = np.ones_like(A_prev)
      self.A = A_prev
      return self.A

    # Create dropout mask
    self.mask = np.random.rand(A_prev.shape[0], A_prev.shape[1]) > self.rate
    # Apply mask AND scale (inverted dropout)
    self.A = (A_prev * self.mask) / (1 - self.rate)

    return self.A

  def backward(self, dA, skip_activation=False):
    # Backprop only through active neurons
    dA_prev = (dA * self.mask) / (1 - self.rate)
    return dA_prev

  def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
    # Dropout layer has no trainable parameters
    pass