class Layer:
  def __init__(self):
    # Trainable parameters (some layers won't use them)
    self.W = None
    self.b = None

    # Gradients
    self.dW = None
    self.db = None

    # Optimizer states
    self.vW = None
    self.vb = None
    self.mW = None
    self.mb = None

    # Forward pass values
    self.A = None
    self.A_prev = None  # Input to this layer

    # Backprop pass values
    self.Z = None

  # Forward and backward are abstract methods — override in subclasses
  def _forward(self, A_prev, training=None):
    raise NotImplementedError

  def _backward(self, dA):
    raise NotImplementedError

  def _update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
    raise NotImplementedError


