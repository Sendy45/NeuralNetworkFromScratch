from .Layer import Layer
import numpy as np

class BatchNorm(Layer):
  def __init__(self, momentum=0.9):
    super().__init__()
    self.momentum = momentum
    self._eps = 1e-08

    self.gamma = None
    self.beta = None

    self.running_mean = None
    self.running_var = None

  def build(self, input_size):
    self.gamma = np.ones((input_size, 1), dtype=np.float32)
    self.beta = np.zeros((input_size, 1), dtype=np.float32)

    self.running_mean = np.zeros((input_size, 1), dtype=np.float32)
    self.running_var = np.ones((input_size, 1), dtype=np.float32)

  def _forward(self, A_prev, training=True):
    if self.gamma is None:
      self.build(A_prev.shape[0])

    self.A_prev = A_prev

    if training:
      self.mean = np.mean(A_prev, axis=1, keepdims=True)
      self.var = np.var(A_prev, axis=1, keepdims=True)

      # X̂ = (X - μB) / √(σB^2 + ε)
      self.X_hat = (A_prev - self.mean) / np.sqrt(self.var + self._eps)
      # A = γ * X̂ + β
      self.A = self.gamma * self.X_hat + self.beta

      # Update running stats
      self.running_mean = (
          self.momentum * self.running_mean
          + (1 - self.momentum) * self.mean
      )

      self.running_var = (
          self.momentum * self.running_var
          + (1 - self.momentum) * self.var
      )

    else:
      self.X_hat = (A_prev - self.running_mean) / np.sqrt(self.running_var + self._eps)
      self.A = self.gamma * self.X_hat + self.beta

    return self.A

  def _backward(self, dA, skip_activation=False):

    m = dA.shape[1]

    dgamma = np.sum(dA * self.X_hat, axis=1, keepdims=True)
    dbeta  = np.sum(dA, axis=1, keepdims=True)

    dX_hat = dA * self.gamma

    var_inv = 1. / np.sqrt(self.var + self._eps)

    dvar = np.sum(dX_hat * (self.A_prev - self.mean) * -0.5 * var_inv**3,
                  axis=1, keepdims=True)

    dmean = (
        np.sum(dX_hat * -var_inv, axis=1, keepdims=True)
        + dvar * np.mean(-2. * (self.A_prev - self.mean), axis=1, keepdims=True)
    )

    dX = (
        dX_hat * var_inv
        + dvar * 2 * (self.A_prev - self.mean) / m
        + dmean / m
    )

    self.dgamma = dgamma / m
    self.dbeta  = dbeta / m

    return dX



  def _update(self, lambda_, lr, beta1, beta2, eps, optimizer, t):
    self.gamma -= lr * self.dgamma
    self.beta -= lr * self.dbeta

