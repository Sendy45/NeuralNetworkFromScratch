import numpy as np

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


class Dense(Layer):
  def __init__(self, units: int, inputs:int=0, kernel_initializer: str=None):
    super().__init__()

    self.units = units
    self.in_size = inputs
    self.out_size = units
    self.kernel_initializer = kernel_initializer

    if not kernel_initializer:
      self._set_default_initializers()


  def _set_default_initializers(self):
      self.initializer = "he"
      """if self.activation == "relu":
          self.initializer = "he"
      elif self.activation in ["sigmoid", "tanh", "softmax", "linear"]:
          self.initializer = "xavier"
      else:
          raise Exception("Invalid activation function")"""

  def build(self, input_size):
      self.in_size = input_size
      self._initialize_weights()

      self.b = np.zeros((self.out_size, 1), dtype=np.float32)

      # Optimizer state
      self.vW = np.zeros_like(self.W).astype(np.float32)
      self.vb = np.zeros_like(self.b).astype(np.float32)
      self.mW = np.zeros_like(self.W).astype(np.float32)
      self.mb = np.zeros_like(self.b).astype(np.float32)

  def _initialize_weights(self):

      if self.initializer == "he":
          # W ~ N(0, √(2/in))
          std = np.sqrt(2.0 / self.in_size)
          self.W = np.random.randn(self.out_size, self.in_size).astype(np.float32) * std

      elif self.initializer == "xavier":
          # U(-√(6/(in+out)),√(6/(in+out)))
          limit = np.sqrt(6.0 / (self.in_size + self.out_size))
          self.W = np.random.uniform(-limit, limit, (self.out_size, self.in_size)).astype(np.float32)

      else:
          self.W = np.random.randn(self.out_size, self.in_size).astype(np.float32) * 0.01

  def _forward(self, A_prev, training=None):
      if self.W is None:
          self.build(A_prev.shape[0])

      self.A_prev = A_prev
      self.Z = np.dot(self.W, A_prev) + self.b

      return self.Z

  def _backward(self, dA, skip_activation=False):

      # dW_i = dZ_i · A_{i}^T
      # Gradient of the loss w.r.t. weights of layer i
      self.dW = np.dot(dA, self.A_prev.T)

      # dB_i = sum(dZ_i) over the batch
      # Gradient of the loss w.r.t. biases of layer i
      self.db = np.sum(dA, axis=1, keepdims=True)

      # Gradient to pass backward
      # dA_prev_i = W_i · dA_i
      return np.dot(self.W.T, dA)


  def _update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):

      if optimizer == "adamW":
          dw = self.dW # pure gradient
      else:
          # Get batch size from the last dZ calculation to scale regularization
          m = self.A.shape[1]
          dw = self.dW + (lambda_ / m) * self.W # L2 regularization

      if optimizer == "momentum":
        self.vW = beta1 * self.vW + dw
        self.vb = beta1 * self.vb + self.db

        update_w = self.vW
        update_b = self.vb

      elif optimizer == "adam" or optimizer == "adamW":

        self.mW = beta1 * self.mW + (1 - beta1) * dw
        self.mb = beta1 * self.mb + (1 - beta1) * self.db

        self.vW = beta2 * self.vW + (1 - beta2) * (dw ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        m_w_hat = self.mW / (1 - beta1 ** t)
        m_b_hat = self.mb / (1 - beta1 ** t)

        v_w_hat = self.vW / (1 - beta2 ** t)
        v_b_hat = self.vb / (1 - beta2 ** t)


        update_w = m_w_hat / (np.sqrt(v_w_hat) + _eps)
        update_b = m_b_hat / (np.sqrt(v_b_hat) + _eps)

      elif optimizer == "rmsprop":
        self.vW = beta2 * self.vW + (1 - beta2) * (dw ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        update_w = dw / (np.sqrt(self.vW) + _eps)
        update_b = self.db / (np.sqrt(self.vb) + _eps)

      else:
        # Classic SGD
        update_w = dw
        update_b = self.db


      # W = W - lr * (dW + λ * W)
      self.W -= lr * update_w
      self.b -= lr * update_b

      # Decouple (adamW) Weight Decay
      if optimizer == "adamW":
        self.W *= (1 - lr * lambda_)  # Decoupled weight decay


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
      shifted = Z - np.max(Z, axis=0, keepdims=True)
      exp_vals = np.exp(shifted)
      self.A = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
      return self.A

  def _backward(self, dA):
      s = np.sum(dA * self.A, axis=0, keepdims=True)
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

class Dropout(Layer):
  def __init__(self, rate):
    super().__init__()
    self.rate = rate  # probability of dropping a unit
    self.mask = None

  def _forward(self, A_prev, training=True):
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

  def _backward(self, dA, skip_activation=False):
    # Backprop only through active neurons
    dA_prev = (dA * self.mask) / (1 - self.rate)
    return dA_prev

  def _update(self, lambda_, lr, beta1, beta2, _eps, optimizer, t):
    # Dropout layer has no trainable parameters
    pass