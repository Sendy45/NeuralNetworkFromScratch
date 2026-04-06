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

  # Return total number of trainable parameters
  def get_params(self) -> int:
      return 0  # default: no params (activations, pooling, etc.)

  # Return a short human-readable description of this layer
  def describe(self) -> str:
      return type(self).__name__

  # Remove forward-pass cache to save memory before serialisation
  def strip_cache(self):
      for attr in self._cache_attrs():
          self.__dict__.pop(attr, None)
      for child in (getattr(self, c, None) for c in self._child_attrs()):
          if child is not None:
              if isinstance(child, (list, tuple)):
                  for c in child:
                      c.strip_cache()
              else:
                  child.strip_cache()

  # Override to list forward-pass cache attribute names
  def _cache_attrs(self):
      return []

  # Override to list attribute names that hold sub-layers
  def _child_attrs(self):
      return []

