class Layer:
  """
    Base class for all layers in the network.

    Defines the standard interface:
    - _forward: compute output activations
    - _backward: compute gradients
    - _update: update parameters

    Enables modular stacking of layers inside NeuralNetwork.
  """

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
  def forward(self, A_prev, training=None):
    raise NotImplementedError

  def backward(self, dA):
    raise NotImplementedError

  # Update model parameters to enable learning
  # handles optimizers
  def update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
    raise NotImplementedError


